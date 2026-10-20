"""Original four-stage MV zero-shot cascade from an exact-replayed SS output.

Ground-truth geometry is loaded only after generation, for coordinate audits
and metrics. This entry does not train or modify any shared source module.
"""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .cascade_artifact_inputs import native_shape_support,prepared_views
from .surface_metrics import compare_surfaces,sample_surface
from .cascade_from_ss_artifact import load_stage,sha
from .zero_shot_cascade_inputs import validate_zero_shot_cascade


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if gpu not in {'0','1','2','3'} or gpu not in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','):
        raise ValueError('one explicitly assigned physical GPU0-3 required')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(plan['allocator_fraction'],0)
    output=Path(plan['output']);output.mkdir(exist_ok=False)
    row=plan['row'];sample_id=row['sample_id']
    previous=json.loads(Path(plan['SS_evaluation']).read_text())
    protocol=validate_zero_shot_cascade(plan,previous)
    entry=next(r for r in previous['native_SS_artifacts']['artifacts'] if r['sample_id']==sample_id)
    artifact=Path(plan['SS_evaluation']).parent/entry['artifact']
    if sha(artifact)!=entry['artifact_SHA256']:
        raise ValueError('actual SS archive bytes changed')
    with np.load(artifact) as z:
        if str(z['sample_id'])!=sample_id:raise ValueError('SS sample identity differs')
        occupancy=np.unpackbits(z['prediction_occupancy_packed'],count=64**3).reshape(64,64,64).astype(bool)
    coords=native_shape_support(occupancy).cuda()
    images=[]
    for path in protocol['RGB_paths']:
        with Image.open(path) as source:images.append(source.copy())
    if len(images)!=4:raise ValueError('exactly four calibrated RGB inputs required')
    from pixal3d import models
    from pixal3d.pipelines import pixal3d_image_to_3d as module
    from pixal3d.pipelines import samplers
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    from .frozen_projection_streaming import install_frozen_projection_streaming
    install_frozen_projection_streaming(max_assets=1)
    specs=plan['stages'];cfgs={name:json.loads(Path(s['config']).read_text()) for name,s in specs.items()}
    stage_keys={'shape512':'shape_slat_flow_model_512','shape1024':'shape_slat_flow_model_1024','pbr1024':'tex_slat_flow_model_1024'}
    loaded={stage_keys[name]:load_stage(cfgs[name],spec['checkpoint']) for name,spec in specs.items()}
    for key,path in plan['decoders'].items():
        if not Path(path+'.json').is_file() or not Path(path+'.safetensors').is_file():
            raise ValueError('explicit local decoder pair required: '+path)
        loaded[key]=models.from_pretrained(path).eval().requires_grad_(False)
    pipeline_cfg=json.loads(Path(plan['pipeline_config']).read_text())['args']
    constructors={};parameters={}
    for stage in ['sparse_structure','shape_slat','tex_slat']:
        s=pipeline_cfg[stage+'_sampler'];constructors[stage+'_sampler']=getattr(samplers,s['name'])(**s['args'])
        parameters[stage+'_sampler_params']=s['params']
    pipeline=module.Pixal3DImageTo3DPipeline(loaded,**constructors,**parameters,
        shape_slat_normalization=pipeline_cfg['shape_slat_normalization'],
        tex_slat_normalization=pipeline_cfg['tex_slat_normalization'],low_vram=True)
    for stage,attr in [('shape512','image_cond_model_shape_512'),('shape1024','image_cond_model_shape_1024'),('pbr1024','image_cond_model_tex_1024')]:
        ec=dict(cfgs[stage]['trainer']['args']['image_cond_model']['args']);ec['model_name']=plan['DINO_local_path']
        if ec['image_size']!=cfgs[stage]['dataset']['args']['image_size']:
            raise ValueError('native dataset/extractor input sizes must agree')
        extractor=DinoV3ProjFeatureExtractor(**ec).eval().requires_grad_(False)
        if extractor.use_naf_upsample:extractor._load_naf()
        extractor.forward=torch.autocast('cuda',dtype=torch.bfloat16)(extractor.forward)
        setattr(pipeline,attr,extractor)
    # Keep the native inline cascade and quantization. Only its completed SS
    # stage is supplied from an independently archived actual native sample.
    pipeline.image_cond_model_ss=object()
    pipeline.get_proj_cond_ss=lambda *args,**kwargs:{}
    def sparse_support(cond,resolution,num_samples,sampler_params):
        if resolution!=32 or num_samples!=1:raise ValueError('native single sample Shape512 grid32 required')
        return coords
    pipeline.sample_sparse_structure=sparse_support
    module.pil_views_to_tensor=prepared_views
    pipeline.to(torch.device('cuda:0'))
    camera=dict(camera_angle_x=row['camera_angle_x'],distance=[float(np.linalg.norm(np.asarray(t)[:3,3])) for t in row['transform_matrix']],
        transform_matrix=row['transform_matrix'],mesh_scale=row['mesh_scale'])
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        meshes,latents=pipeline.run(images,camera,seed=int(plan['seed']),preprocess_image=False,return_latent=True,
            pipeline_type='1024_cascade',max_num_tokens=32768)
    if len(meshes)!=1 or latents[2]!=1024:raise ValueError('one actual1024 cascade mesh required')
    mesh=meshes[0];shape,texture,resolution=latents
    if not len(mesh.vertices) or not len(mesh.faces):raise ValueError('empty generated mesh')
    np.savez_compressed(output/'prediction.npz',vertices=mesh.vertices.cpu().numpy(),faces=mesh.faces.cpu().numpy(),
        coords=mesh.coords.cpu().numpy(),attrs=mesh.attrs.cpu().numpy(),origin=mesh.origin.cpu().numpy(),
        voxel_size=np.asarray(mesh.voxel_size),voxel_shape=np.asarray(mesh.voxel_shape),
        shape_coords=shape.coords.cpu().numpy(),shape_feats=shape.feats.float().cpu().numpy(),
        texture_feats=texture.feats.float().cpu().numpy())
    generation_seconds=time.monotonic()-started
    # Evaluation-only geometry is first accessed AFTER all generated outputs.
    from data_toolkit.utils import transform_mesh
    with np.load(plan['source_geometry']) as source:
        source_vertices,faces=source['V'].copy(),source['F'].copy()
    target=np.load(plan['target_vertices'])
    if source_vertices.shape!=target.shape or target.shape[1]!=3:raise ValueError('paired source/target topology required')
    basis=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    transformed=transform_mesh(basis,dict(transform_matrix=row['transform_matrix'][0]))
    rotation=(transformed[1:]-transformed[0]).T.numpy()
    if not np.allclose(transformed[0].numpy(),0,atol=1e-5) or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5):
        raise ValueError('native anchor transform must be a pure rotation')
    prediction_world=mesh.vertices.float().cpu().numpy()@rotation/row['mesh_scale']
    center=(source_vertices.min(0)+source_vertices.max(0))/2
    extent=float(np.ptp(source_vertices,axis=0).max())
    pred_points=sample_surface((prediction_world-center)/extent,mesh.faces.cpu().numpy(),seed=plan['surface_seed'])
    target_points=sample_surface((target-center)/extent,faces,seed=plan['surface_seed'])
    geometry=compare_surfaces(pred_points,target_points,threshold=.02)
    np.savez_compressed(output/'geometry-evaluation.npz',prediction_world=prediction_world,target_world=target,
        target_faces=faces,source_center=center,source_extent=extent,anchor_rotation=rotation)
    result=dict(status='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION',sample_id=sample_id,
        generated_mesh=str(output/'prediction.npz'),actual_resolution=resolution,
        actual_SS32_tokens=len(coords),actual_HR_tokens=len(shape.coords),
        generation_seconds=generation_seconds,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        metrics=geometry,source_defined_normalization=True,per_output_alignment=False,
        SS_source_evaluation=plan['SS_evaluation'],SS_prediction_SHA256=entry['prediction_voxels_SHA256'],
        GT_geometry_or_coords_used_in_generation=False,GT_loaded_only_after_generation=True,
        fixed_shape_models=True,PBR_checkpoint=specs['pbr1024']['checkpoint'],
        additional_PBR_conditioner=False,full_evaluation_complete=False,
        raw_render_capture_not_implemented=True,multi_view_MIoU_not_measured=True,
        original_zero_shot_full_stack=True,conditioning_mode=protocol['conditioning_mode'],
        input_states=protocol['input_states'],input_camera_ids=protocol['input_camera_ids'],
        actual_RGB_input_files=[dict(path=path,sha256=sha(path)) for path in protocol['RGB_paths']],
        original_multiview_steps=dict(ss64=4210,shape512=4210,shape1024=16850,pbr1024=16850),
        PBR_checkpoint_origin='original_multiview_16850',deformation_finetuning_updates=0,
        visibility_mask_used=False,
        original_stage_configs_and_weights_unchanged=True,model_adopted=False)
    (output/'generation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
