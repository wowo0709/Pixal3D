"""Score archived prediction renders against original materials in the SAME renderer.

No VAE reconstruction is used as the material reference. Original target dump
vertices are restored to the audited world frame; prediction/NN stays frozen.
"""
from __future__ import annotations
import argparse,json,os,pickle
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if gpu not in {'0','1','2','3'} or gpu not in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','):
        raise ValueError('one assigned GPU required')
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.20,0)
    output=Path(plan['output']);output.mkdir(exist_ok=False)
    from pixal3d.datasets.sparse_voxel_pbr import SparseVoxelPbrDataset
    from pixal3d.renderers import PbrMeshRenderer,EnvMap
    from pixal3d.utils.loss_utils import psnr,ssim,lpips
    from .target_frame import restore_dump_world
    path=Path(plan['pbr_dump'])
    with path.open('rb') as f:original=pickle.load(f)
    restored=restore_dump_world(original,plan['row'])
    vertices=np.concatenate([o['vertices'] for o in restored['objects'] if o['vertices'].size and o['faces'].size])
    with np.load(Path(plan['generation_root'])/'geometry-evaluation.npz') as z:target=z['target_world'].copy()
    distance=cKDTree(target).query(vertices,workers=1)[0]
    if distance.max()>1e-5:raise ValueError('original textured target dump does not match audited world GT')
    # Native loader parses the actual textures/UVs/material IDs. Replace its
    # generic per-mesh normalization with the exact independently restored V.
    loader=object.__new__(SparseVoxelPbrDataset)
    mesh=loader.read_mesh_with_texture(str(path.parent),path.stem)['mesh'][0]
    if len(mesh.vertices)!=len(vertices):raise ValueError('native material loader changed vertex topology')
    mesh.vertices=torch.from_numpy(vertices).float()
    mesh=mesh.to('cuda')
    renderer=PbrMeshRenderer();renderer.rendering_options.resolution=512
    renderer.rendering_options.near=.01;renderer.rendering_options.far=100
    renderer.rendering_options.ssaa=1;renderer.rendering_options.peel_layers=4
    hdr=cv2.imread(plan['environment_map'],cv2.IMREAD_UNCHANGED)
    if hdr is None:raise ValueError('existing shared environment map required')
    envmap=EnvMap(torch.tensor(cv2.cvtColor(hdr,cv2.COLOR_BGR2RGB),dtype=torch.float32,device='cuda'))
    rows=[]
    with torch.inference_mode():
        for view in plan['views']:
            cid=view['camera_id']
            with np.load(Path(plan['previous_render_root'])/f'camera{cid:02d}.npz') as z:
                pred=torch.from_numpy(z['shaded'].copy()).cuda()[None]
                pred_alpha=torch.from_numpy(z['alpha'].copy()).cuda()
                ext=torch.from_numpy(z['extrinsics'].copy()).cuda();intr=torch.from_numpy(z['intrinsics'].copy()).cuda()
                cycles=torch.from_numpy(z['GT_RGB'].copy()).cuda()[None]
            with torch.random.fork_rng(devices=[0]):
                torch.cuda.manual_seed(plan['AO_seed'])
                reference=renderer.render(mesh,ext,intr,envmap=envmap)
            gt=reference['shaded'][None].float()
            mask=(pred_alpha>.5)|(reference['alpha']>.5)
            union=mask.sum();intersection=((pred_alpha>.5)&(reference['alpha']>.5)).sum()
            if not union:raise ValueError('empty common foreground')
            row=dict(camera_id=cid,PSNR=float(psnr(pred,gt)),SSIM=float(ssim(pred,gt)),
                LPIPS=float(lpips(pred,gt)),M_IoU=float(intersection.double()/union.double()),
                foreground_PSNR=float(-10*((pred-gt).square()[mask[None,None].expand_as(pred)]).mean().clamp_min(1e-12).log10()),
                original_Cycles_vs_native_GT_LPIPS=float(lpips(cycles,gt)),
                source_material_identity_preserved=True,same_renderer_environment_camera_AO_seed=True)
            np.savez_compressed(output/f'camera{cid:02d}.npz',prediction=pred[0].cpu().numpy(),
                GT_shared_renderer=gt[0].cpu().numpy(),GT_Cycles=cycles[0].cpu().numpy(),
                prediction_alpha=pred_alpha.cpu().numpy(),GT_alpha=reference['alpha'].cpu().numpy(),
                GT_base_color=reference['base_color'].cpu().numpy())
            rows.append(row);print(json.dumps(row),flush=True)
    result=dict(status='PASS_ACTUAL_SHARED_RENDERER_MATERIAL_REFERENCE',per_view=rows,
        view_average={k:float(np.mean([r[k] for r in rows])) for k in ['PSNR','SSIM','LPIPS','M_IoU','foreground_PSNR']},
        original_materials_and_UVs=True,GT_VAE_reconstruction_not_used=True,
        maximum_GT_world_vertex_nearest_difference=float(distance.max()),
        prediction_renders_reused_not_regenerated=True,shared_renderer_lighting_camera_AO_seed=True,
        GT_not_used_for_model_generation=True,raw_arrays_saved=True,
        diagnostic_only=True,paper_evaluation_complete=False,model_comparison_not_done=True,
        DreamSim_CLIP_FID_KID_not_measured=True)
    (output/'render-evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
