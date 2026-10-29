"""Run the frozen V1 1024 cascade from GT-free BenchUp SS predictions."""
from __future__ import annotations
import argparse, hashlib, json, os, time
from contextlib import nullcontext
from pathlib import Path
os.environ.setdefault('ATTN_BACKEND','sdpa')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
from PIL import Image
import torch
from experiments.deformnext_retargeting.cascade_from_ss_artifact import load_stage
from experiments.deformnext_retargeting.cascade_artifact_inputs import native_shape_support,prepared_views

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('manifest','ss-root','template','output'):p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--gpu',required=True,choices=['0','1','2','3'])
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--ss-arm',choices=['v1','selected','mv-zero'],default='v1')
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu or not 0<=a.rank<a.world_size:raise ValueError('explicit GPU/shard required')
    torch.set_num_threads(2);torch.cuda.set_device(0)
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x]
    if len(rows)!=109 or any('mesh_scale' not in row for row in rows):raise ValueError('expected full inferred-scale manifest')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    template=json.loads(a.template.read_text());specs=template['stages']
    from pixal3d import models
    from pixal3d.pipelines import pixal3d_image_to_3d as module, samplers
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    from experiments.deformnext_retargeting.frozen_projection_streaming import install_frozen_projection_streaming
    install_frozen_projection_streaming(max_assets=1)
    cfgs={name:json.loads(Path(spec['config']).read_text()) for name,spec in specs.items()}
    keys={'shape512':'shape_slat_flow_model_512','shape1024':'shape_slat_flow_model_1024','pbr1024':'tex_slat_flow_model_1024'}
    loaded={key:load_stage(cfgs[name],specs[name]['checkpoint']) for name,key in keys.items()}
    for key,path in template['decoders'].items():loaded[key]=models.from_pretrained(path).eval().requires_grad_(False)
    pipeline_cfg=json.loads(Path(template['pipeline_config']).read_text())['args']
    constructors={};params={}
    for stage in ['sparse_structure','shape_slat','tex_slat']:
        spec=pipeline_cfg[stage+'_sampler'];constructors[stage+'_sampler']=getattr(samplers,spec['name'])(**spec['args']);params[stage+'_sampler_params']=spec['params']
    pipeline=module.Pixal3DImageTo3DPipeline(loaded,**constructors,**params,shape_slat_normalization=pipeline_cfg['shape_slat_normalization'],tex_slat_normalization=pipeline_cfg['tex_slat_normalization'],low_vram=True)
    for stage,attr in [('shape512','image_cond_model_shape_512'),('shape1024','image_cond_model_shape_1024'),('pbr1024','image_cond_model_tex_1024')]:
        ec=dict(cfgs[stage]['trainer']['args']['image_cond_model']['args']);ec['model_name']=template['DINO_local_path']
        extractor=DinoV3ProjFeatureExtractor(**ec).eval().requires_grad_(False)
        if extractor.use_naf_upsample:extractor._load_naf()
        extractor.forward=torch.autocast('cuda',dtype=torch.bfloat16)(extractor.forward);setattr(pipeline,attr,extractor)
    pipeline.image_cond_model_ss=object();pipeline.get_proj_cond_ss=lambda *args,**kwargs:{}
    module.pil_views_to_tensor=prepared_views;pipeline.to(torch.device('cuda:0'))
    selected=[(i,row) for i,row in enumerate(rows) if i%a.world_size==a.rank]
    if a.limit:selected=selected[:a.limit]
    for i,row in selected:
        sid=row['sample_id'];artifact=a.ss_root/(a.ss_arm+'-ss-shard-'+str(i%4))/'ss-artifacts'/(hashlib.sha256(sid.encode()).hexdigest()+'.npz')
        with np.load(artifact) as z:
            if str(z['sample_id'])!=sid:raise ValueError('SS identity mismatch')
            occ=np.unpackbits(z['prediction_occupancy_packed'],count=64**3).reshape(64,64,64).astype(bool)
        coords=native_shape_support(occ).cuda()
        def sparse_support(cond,resolution,num_samples,sampler_params):
            if resolution!=32 or num_samples!=1:raise ValueError('unexpected shape support')
            return coords
        pipeline.sample_sparse_structure=sparse_support
        images=[Image.open(path).copy() for path in [row['target_anchor'],*row['source_images']]]
        camera=dict(camera_angle_x=row['camera_angle_x'],distance=[float(np.linalg.norm(np.asarray(t)[:3,3])) for t in row['transform_matrix']],transform_matrix=row['transform_matrix'],mesh_scale=row['mesh_scale'])
        out=a.output/'cases'/sid;out.mkdir(parents=True,exist_ok=False)
        if a.ss_arm=='selected':
            from experiments.deformnext_retargeting.native_guide_capture_context import capture_native_guides
            capture=capture_native_guides(module.Pixal3DImageTo3DPipeline,out/'native-shape-guides.pt')
        else:capture=nullcontext(None)
        start=time.monotonic()
        with capture as events,torch.inference_mode():
            meshes,latents=pipeline.run(images,camera,seed=42+i,preprocess_image=False,return_latent=True,pipeline_type='1024_cascade',max_num_tokens=32768)
        if len(meshes)!=1 or latents[2]!=1024:raise ValueError('one 1024 cascade output required')
        mesh=meshes[0];shape,texture,resolution=latents
        if not len(mesh.vertices) or not len(mesh.faces):raise ValueError('empty mesh')
        if a.ss_arm=='selected' and len(events)!=1:raise ValueError('actual native shape subdivision guide capture required')
        np.savez_compressed(out/'prediction.npz',vertices=mesh.vertices.float().cpu().numpy(),faces=mesh.faces.cpu().numpy(),coords=mesh.coords.cpu().numpy(),attrs=mesh.attrs.cpu().numpy(),origin=mesh.origin.cpu().numpy(),voxel_size=np.asarray(mesh.voxel_size),voxel_shape=np.asarray(mesh.voxel_shape),shape_coords=shape.coords.cpu().numpy(),shape_feats=shape.feats.float().cpu().numpy(),texture_feats=texture.feats.float().cpu().numpy())
        result={'status':'V1_CASCADE_GENERATED_NO_TARGET_3D_GT' if a.ss_arm=='v1' else 'CASCADE_GENERATED_NO_TARGET_3D_GT','ss_arm':a.ss_arm,'sample_id':sid,'prediction_sha256':sha(out/'prediction.npz'),'ss_sha256':sha(artifact),'seed':42+i,'seconds':round(time.monotonic()-start,2),'camera':'estimated_from_source_mesh_and_views','scale':'image_estimated','resolution':resolution,'source_mesh_used_for_camera_estimation':True,'GT_target_geometry_used':False,'pbr_checkpoint':specs['pbr1024']['checkpoint'],'selected_ss_with_vanilla_pbr_is_not_selected_full':a.ss_arm=='selected'}
        if a.ss_arm=='selected':result['native_shape_guides_sha256']=sha(out/'native-shape-guides.pt')
        (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
        del mesh,meshes,latents,shape,texture,coords;torch.cuda.empty_cache()

if __name__=='__main__':main()
