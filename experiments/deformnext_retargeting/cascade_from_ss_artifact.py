"""Generate actual Shape512/1024 and texture from an archived native SS output.

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


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):d.update(b)
    return d.hexdigest()


def canonical_seed(value):
    payload=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return int(hashlib.sha256(payload).hexdigest()[:8],16)


def sample_surface_cuda(vertices,faces,requests):
    """Area-weighted samples with one CUDA area pass for several RNG seeds."""
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or not len(vertices)
            or faces.ndim!=2 or faces.shape[1]!=3 or not len(faces)):
        raise ValueError('nonempty indexed CUDA triangle mesh required')
    device=torch.device('cuda:0')
    vertices=vertices.to(device=device,dtype=torch.float64)
    faces=faces.to(device=device,dtype=torch.int64)
    areas=[]
    for start in range(0,len(faces),1_048_576):
        part=faces[start:start+1_048_576]
        triangle=vertices[part]
        area=torch.linalg.vector_norm(torch.cross(
            triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0],dim=1),dim=1)
        areas.append(area)
    area=torch.cat(areas)
    total=area.sum()
    if not torch.isfinite(area).all() or float(total)<=0:
        raise ValueError('generated mesh has no finite positive surface area')
    cdf=torch.cumsum(area,dim=0);cdf/=cdf[-1].clone()
    outputs={}
    for name,count,seed in requests:
        rng=np.random.default_rng(int(seed))
        # This is NumPy's weighted-choice inverse-CDF path, with the CDF
        # shared across all requested deterministic seeds.
        uniform=torch.from_numpy(rng.random(int(count))).to(device=device)
        chosen=torch.searchsorted(cdf,uniform,right=True)
        selected=faces[chosen]
        triangle=vertices[selected].cpu().numpy()
        u,v=rng.random((2,int(count)));root=np.sqrt(u)
        weights=np.stack([1-root,root*(1-v),root*v],axis=1)
        outputs[name]=(triangle*weights[:,:,None]).sum(1)
    return outputs


def load_stage(config,checkpoint):
    from pixal3d import models
    from pixal3d.modules.lora import inject_lora
    spec=config['models']['denoiser']
    with torch.device('meta'):
        model=getattr(models,spec['name'])(**spec['args'])
        model.convert_to(torch.bfloat16)
        inject_lora(model,**config['trainer']['args']['lora'])
    state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    expected=model.state_dict()
    if (set(state)!=set(expected) or any(state[k].shape!=expected[k].shape
            or state[k].dtype!=expected[k].dtype for k in state)):
        raise ValueError('complete saved native base+LoRA keys,shapes,dtypes required')
    model.load_state_dict(state,strict=True,assign=True)
    model.requires_grad_(False).eval()
    model.forward=torch.autocast('cuda',dtype=torch.bfloat16)(model.forward)
    return model


def run_geometry_1024(pipeline,module,images,camera,coords,seed,max_num_tokens):
    """Native 512->1024 shape cascade without the independent PBR stage."""
    from pixal3d.modules.sparse import SparseTensor
    images,cameras=module.normalize_calibrated_views(images,camera)
    camera_angle_x=cameras['camera_angle_x'];distance=cameras['distance']
    mesh_scale=cameras['mesh_scale'];transform_matrix=cameras['transform_matrix']
    torch.manual_seed(int(seed))
    cond_lr=pipeline.get_proj_cond_shape(pipeline.image_cond_model_shape_512,
        images,coords,camera_angle_x=camera_angle_x,distance=distance,
        mesh_scale=mesh_scale,transform_matrix=transform_matrix)
    lr_slat=pipeline.sample_shape_slat(cond_lr,
        pipeline.models['shape_slat_flow_model_512'],coords,{})
    del cond_lr;torch.cuda.empty_cache()
    if pipeline.low_vram:
        pipeline.models['shape_slat_decoder'].to(pipeline.device)
        pipeline.models['shape_slat_decoder'].low_vram=True
    hr_coords=pipeline.models['shape_slat_decoder'].upsample(lr_slat,upsample_times=4)
    if pipeline.low_vram:
        pipeline.models['shape_slat_decoder'].cpu()
        pipeline.models['shape_slat_decoder'].low_vram=False
    actual_resolution=1024;lr_resolution=512
    while True:
        grid_res=actual_resolution//16
        quant=torch.cat([hr_coords[:,:1],
            ((hr_coords[:,1:]+.5)/lr_resolution*(grid_res-1)).round().int()],dim=1)
        unique=quant.unique(dim=0)
        if len(unique)<max_num_tokens or actual_resolution==1024:break
        actual_resolution-=128
    del lr_slat,hr_coords,quant;torch.cuda.empty_cache()
    cond_hr=pipeline.get_proj_cond_shape(pipeline.image_cond_model_shape_1024,
        images,unique,camera_angle_x=camera_angle_x,distance=distance,
        mesh_scale=mesh_scale,transform_matrix=transform_matrix,
        grid_resolution_override=actual_resolution//16)
    flow=pipeline.models['shape_slat_flow_model_1024']
    noise=SparseTensor(feats=torch.randn(len(unique),flow.in_channels,
        device=pipeline.device),coords=unique)
    params={**pipeline.shape_slat_sampler_params}
    if pipeline.low_vram:flow.to(pipeline.device)
    sampled=pipeline.shape_slat_sampler.sample(flow,noise,**cond_hr,**params,
        verbose=True,tqdm_desc=f'Sampling HR shape SLat (proj, {actual_resolution})').samples
    if pipeline.low_vram:flow.cpu()
    std=torch.tensor(pipeline.shape_slat_normalization['std'])[None].to(sampled.device)
    mean=torch.tensor(pipeline.shape_slat_normalization['mean'])[None].to(sampled.device)
    shape=sampled*std+mean
    del cond_hr,noise,sampled,unique;torch.cuda.empty_cache()
    meshes,_=pipeline.decode_shape_slat(shape,actual_resolution)
    for mesh in meshes:mesh.fill_holes()
    return meshes,(shape,actual_resolution)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if gpu not in {'0','1','2','3','6','7'} or gpu not in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','):
        raise ValueError('one explicitly assigned authorized physical GPU required')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(plan['allocator_fraction'],0)
    output=Path(plan['output']);output.mkdir(exist_ok=False)
    row=plan['row'];sample_id=row['sample_id']
    scale_policy=plan.get('object_scale_policy','legacy-target')
    if scale_policy not in {'legacy-target','image-estimated-v1','native-unit'}:
        raise ValueError('unknown object scale policy')
    if scale_policy=='native-unit' and not np.isclose(float(row['mesh_scale']),1.,rtol=0,atol=1e-8):
        raise ValueError('native-unit control must use canonical scale 1.0')
    if scale_policy=='image-estimated-v1':
        source=Path(plan['image_scale_predictions'])
        receipt=json.loads(source.with_suffix('.receipt.json').read_text())
        if (receipt['status']!='PASS_IMAGE_ONLY_SCALE_PREDICTIONS'
                or receipt['output_sha256']!=sha(source)):
            raise ValueError('image-only scale prediction receipt mismatch')
        predictions=[json.loads(line) for line in source.read_text().splitlines() if line]
        found=[value for value in predictions if value['sample_id']==sample_id]
        if (len(found)!=1 or not np.isclose(float(row['mesh_scale']),
                found[0]['predicted_mesh_scale'],rtol=0,atol=1e-7)
                or found[0]['M0_or_M1_mesh_or_GT_scale_used_at_inference'] is not False):
            raise ValueError('case scale does not match image-only prediction')
    previous=json.loads(Path(plan['SS_evaluation']).read_text())
    entry=next(r for r in previous['native_SS_artifacts']['artifacts'] if r['sample_id']==sample_id)
    artifact=Path(plan['SS_evaluation']).parent/entry['artifact']
    if sha(artifact)!=entry['artifact_SHA256']:
        raise ValueError('actual SS archive bytes changed')
    with np.load(artifact) as z:
        if str(z['sample_id'])!=sample_id:raise ValueError('SS sample identity differs')
        occupancy=np.unpackbits(z['prediction_occupancy_packed'],count=64**3).reshape(64,64,64).astype(bool)
    coords=native_shape_support(occupancy).cuda()
    if plan['input_states']!=['M1','M0','M0','M0']:
        raise ValueError('this first smoke uses the explicit editedRGB four-view protocol')
    images=[Image.open(path).copy() for path in plan['RGB_paths']]
    if len(images)!=4:raise ValueError('exactly four calibrated RGB inputs required')
    from pixal3d import models
    from pixal3d.pipelines import pixal3d_image_to_3d as module
    from pixal3d.pipelines import samplers
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    from .frozen_projection_streaming import install_frozen_projection_streaming
    install_frozen_projection_streaming(max_assets=1)
    specs=plan['stages'];cfgs={name:json.loads(Path(s['config']).read_text()) for name,s in specs.items()}
    stage_keys={'shape512':'shape_slat_flow_model_512','shape1024':'shape_slat_flow_model_1024'}
    loaded={stage_keys[name]:load_stage(cfgs[name],spec['checkpoint'])
        for name,spec in specs.items() if name in stage_keys}
    for key,path in plan['decoders'].items():
        if key!='shape_slat_decoder':continue
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
    for stage,attr in [('shape512','image_cond_model_shape_512'),('shape1024','image_cond_model_shape_1024')]:
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
        meshes,latents=run_geometry_1024(pipeline,module,images,camera,coords,
            int(plan['seed']),32768)
    print(json.dumps({'phase':'CASCADE_PIPELINE_DONE','seconds':time.monotonic()-started}),flush=True)
    if len(meshes)!=1 or latents[1]!=1024:raise ValueError('one actual1024 cascade mesh required')
    mesh=meshes[0];shape,resolution=latents
    if not len(mesh.vertices) or not len(mesh.faces):raise ValueError('empty generated mesh')
    stable=canonical_seed(sample_id)
    sampled=sample_surface_cuda(mesh.vertices,mesh.faces,[
        ('inline',20000,int(plan['surface_seed'])),
        ('raw',20000,stable),
        ('registration',4096,stable+103),
    ])
    print(json.dumps({'phase':'CUDA_SURFACE_SAMPLES_DONE','seconds':time.monotonic()-started,
        'vertices':len(mesh.vertices),'faces':len(mesh.faces)}),flush=True)
    vertices=mesh.vertices.float().cpu().numpy();faces=mesh.faces.cpu().numpy()
    # Geometry ablations retain the actual generated mesh but avoid compressing
    # texture/latent arrays that are neither scored nor archived by this stage.
    np.savez(output/'prediction.npz',vertices=vertices,faces=faces)
    print(json.dumps({'phase':'MINIMAL_PREDICTION_SAVED','seconds':time.monotonic()-started}),flush=True)
    generation_seconds=time.monotonic()-started
    # Camera-frame conversion uses only inference inputs.  Evaluation-only GT
    # geometry is first accessed after generated mesh and samples are fixed.
    from data_toolkit.utils import transform_mesh
    basis=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    transformed=transform_mesh(basis,dict(transform_matrix=row['transform_matrix'][0]))
    rotation=(transformed[1:]-transformed[0]).T.numpy()
    if not np.allclose(transformed[0].numpy(),0,atol=1e-5) or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5):
        raise ValueError('native anchor transform must be a pure rotation')
    sampled_world={key:value@rotation/row['mesh_scale'] for key,value in sampled.items()}
    with np.load(plan['source_geometry']) as source:
        source_vertices,target_faces=source['V'].copy(),source['F'].copy()
    target=np.load(plan['target_vertices'])
    if source_vertices.shape!=target.shape or target.shape[1]!=3:raise ValueError('paired source/target topology required')
    center=(source_vertices.min(0)+source_vertices.max(0))/2
    extent=float(np.ptp(source_vertices,axis=0).max())
    pred_points=(sampled_world['inline']-center)/extent
    target_points=sample_surface((target-center)/extent,target_faces,seed=plan['surface_seed'])
    geometry=compare_surfaces(pred_points,target_points,threshold=.02)
    np.savez_compressed(output/'geometry-evaluation.npz',
        predicted_points=sampled_world['raw'],registration_points=sampled_world['registration'],
        target_world=target,target_faces=target_faces,source_center=center,
        source_extent=extent,anchor_rotation=rotation)
    print(json.dumps({'phase':'GEOMETRY_EVIDENCE_SAVED','seconds':time.monotonic()-started}),flush=True)
    result=dict(status='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_GEOMETRY_GENERATION',sample_id=sample_id,
        object_scale_policy=scale_policy,scale_used_in_generation=float(row['mesh_scale']),
        image_scale_predictions_sha256=sha(plan['image_scale_predictions']) if scale_policy=='image-estimated-v1' else None,
        M1_derived_scale_supplied_to_generation=scale_policy=='legacy-target',
        source_M0_mesh_used_in_generation=False,
        generated_mesh=str(output/'prediction.npz'),actual_resolution=resolution,
        actual_SS32_tokens=len(coords),actual_HR_tokens=len(shape.coords),
        generation_seconds=generation_seconds,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        metrics=geometry,source_defined_normalization=True,per_output_alignment=False,
        SS_source_evaluation=plan['SS_evaluation'],SS_prediction_SHA256=entry['prediction_voxels_SHA256'],
        GT_geometry_or_coords_used_in_generation=False,GT_loaded_only_after_generation=True,
        fixed_shape_models=True,PBR_checkpoint=None,geometry_stage_does_not_execute_PBR=True,
        additional_PBR_conditioner=False,full_evaluation_complete=False,
        raw_render_capture_not_implemented=True,multi_view_MIoU_not_measured=True,
        original_stage_configs_and_weights_unchanged=True,model_adopted=False)
    (output/'generation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
