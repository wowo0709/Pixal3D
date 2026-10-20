"""Render archived predictions with a common native one-million-face budget.

The original generated mesh and its surface metrics remain untouched. This
new rendering protocol applies the same native Mesh.simplify rule to every
prediction, independently of model, sample or metric. Original-material GT
scoring remains a separate, shared-reference phase.
"""
from __future__ import annotations
import argparse,json,os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if gpu not in {'0','1','2','3'} or gpu not in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','):
        raise ValueError('one assigned GPU required')
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.8,0)
    root=Path(plan['generation_root']);output=Path(plan['output']);output.mkdir(exist_ok=False)
    generated=json.loads((root/'generation.json').read_text())
    if generated['GT_geometry_or_coords_used_in_generation']:raise ValueError('actual generated geometry required')
    from pixal3d.representations import Mesh,MeshWithVoxel
    from pixal3d.renderers import MeshRenderer,PbrMeshRenderer,EnvMap
    from pixal3d.utils.loss_utils import psnr,ssim,lpips
    from .topology_labels import mask_agreement
    import utils3d
    scale=plan['row']['mesh_scale']
    if plan.get('prediction_render_face_budget')!=1000000:
        raise ValueError('the same explicit native million-face rendering budget is required for every arm')
    with np.load(root/'prediction.npz') as z:
        mesh=MeshWithVoxel(torch.from_numpy(z['vertices']).cuda(),
            torch.from_numpy(z['faces']).cuda(),(z['origin']/scale).tolist(),float(z['voxel_size'])/scale,
            torch.from_numpy(z['coords']).cuda(),torch.from_numpy(z['attrs']).cuda(),
            torch.Size(z['voxel_shape'].tolist()),
            dict(base_color=slice(0,3),metallic=slice(3,4),roughness=slice(4,5),alpha=slice(5,6)))
    raw_vertices,raw_faces=len(mesh.vertices),len(mesh.faces)
    if raw_faces>1000000:
        mesh.simplify(target=1000000)
    if (not torch.isfinite(mesh.vertices).all() or mesh.faces.min()<0
            or mesh.faces.max()>=len(mesh.vertices)):
        raise ValueError('finite in-range postprocessed rendering geometry required')
    geometry=dict(status='PASS_NATIVE_COMMON_RENDER_GEOMETRY',
        policy='native Mesh.simplify to1000000 faces only when raw_faces exceeds1000000',
        raw_vertices=raw_vertices,raw_faces=raw_faces,render_vertices=len(mesh.vertices),render_faces=len(mesh.faces),
        simplified=raw_faces>1000000,normalized_decoder_frame_preserved=True,
        original_prediction_unchanged=True,raw_surface_metrics_unchanged=True,per_output_alignment=False,
        same_budget_for_all_predictions=True,GT_geometry_used_for_simplification=False,
        GLB_export_uses_same_face_budget_but_full_export_remeshing_not_applied=True)
    np.savez_compressed(output/'render-geometry.npz',vertices=mesh.vertices.detach().cpu().numpy(),faces=mesh.faces.detach().cpu().numpy())
    (output/'render-geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
    mesh.vertices=mesh.vertices/scale
    with np.load(root/'geometry-evaluation.npz') as z:
        target=Mesh(torch.from_numpy(z['target_world']).cuda(),torch.from_numpy(z['target_faces']).cuda())
        rotation=torch.eye(4,device='cuda');rotation[:3,:3]=torch.from_numpy(z['anchor_rotation']).cuda()
    renderer=PbrMeshRenderer();renderer.rendering_options.resolution=512
    renderer.rendering_options.near=.01;renderer.rendering_options.far=100
    renderer.rendering_options.ssaa=1;renderer.rendering_options.peel_layers=4
    gt_renderer=MeshRenderer(dict(resolution=512,near=.01,far=100,ssaa=1,antialias=False))
    hdr=cv2.imread(plan['environment_map'],cv2.IMREAD_UNCHANGED)
    if hdr is None:raise ValueError('explicit existing environment map required')
    envmap=EnvMap(torch.tensor(cv2.cvtColor(hdr,cv2.COLOR_BGR2RGB),dtype=torch.float32,device='cuda'))
    flip=torch.diag(torch.tensor([1.,-1.,-1.,1.],device='cuda'))
    records=[]
    with torch.inference_mode():
        for view in plan['views']:
            i=view['input_camera_index']
            c2w=torch.tensor(plan['row']['transform_matrix'][i],dtype=torch.float32,device='cuda')
            world_ext=flip@torch.linalg.inv(c2w)
            intr=utils3d.torch.intrinsics_from_fov(fov_x=torch.tensor(plan['row']['camera_angle_x'][i],device='cuda'),
                fov_y=torch.tensor(plan['row']['camera_angle_x'][i],device='cuda'))
            image=Image.open(view['GT_RGBA']).convert('RGBA')
            if image.size!=(512,512):raise ValueError('GT camera calibration requires original512px RGBA')
            rgba=torch.tensor(np.asarray(image).copy(),device='cuda').float()/255.
            rgb=(rgba[...,:3]*rgba[...,3:4]).permute(2,0,1)[None]
            gt_mask=rgba[...,3]>.5
            raster=gt_renderer.render(target,world_ext,intr,return_types=['mask'])['mask']>.5
            iou,containment=mask_agreement(raster,gt_mask,alpha_cutouts=True)
            with torch.random.fork_rng(devices=[0]):
                torch.cuda.manual_seed(plan['AO_seed'])
                render=renderer.render(mesh,world_ext@torch.linalg.inv(rotation),intr,envmap=envmap)
            pred=render['shaded'][None].float();pred_mask=render['alpha'].reshape(512,512)>.5
            union=pred_mask|gt_mask;intersection=pred_mask&gt_mask
            if not union.any():raise ValueError('empty shared rendering foreground')
            row=dict(camera_id=view['camera_id'],input_camera_index=i,
                M_IoU=float(intersection.sum().double()/union.sum().double()),
                PSNR=float(psnr(pred,rgb)),SSIM=float(ssim(pred,rgb)),
                LPIPS=float(lpips(pred,rgb)),
                foreground_PSNR=float(-10*((pred-rgb).square()[union[None,None].expand_as(pred)]).mean().clamp_min(1e-12).log10()),
                GT_geometry_RGBA_mask_iou=iou,GT_geometry_RGBA_containment=containment,
                original_target_camera=True,paired_textured_M1_reference=True,
                shared_lighting_with_original_Cycles_GT_not_proven=True)
            arrays={k:render[k].detach().float().cpu().numpy() for k in ['shaded','base_color','alpha','mask']}
            arrays.update(GT_RGB=rgb[0].cpu().numpy(),GT_alpha=rgba[...,3].cpu().numpy(),
                extrinsics=world_ext.cpu().numpy(),intrinsics=intr.cpu().numpy())
            np.savez_compressed(output/f'camera{view["camera_id"]:02d}.npz',**arrays)
            records.append(row);print(json.dumps(row),flush=True)
    result=dict(status='PASS_ACTUAL_CASCADE_MULTIVIEW_RGB_ALPHA_AND_CAMERA_AUDIT',
        sample_id=generated['sample_id'],per_view=records,
        view_average={k:float(np.mean([r[k] for r in records])) for k in ['M_IoU','PSNR','SSIM','LPIPS','foreground_PSNR']},
        raw_arrays_saved=True,prediction_is_generated_full_cascade=True,
        GT_loaded_only_for_scoring=True,GT_shape_not_used_for_prediction=True,
        novel_camera_claim=False,shared_rendering_lighting_not_yet_proven=True,
        diagnostic_only=True,paper_evaluation_complete=False,
        prediction_geometry_postprocess=geometry,
        DreamSim_and_CLIP_not_measured=True,FID_KID_not_measured=True)
    (output/'render-evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
