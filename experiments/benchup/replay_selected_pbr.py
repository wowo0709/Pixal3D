"""Replay the frozen Top-1 PBR conditioner on selected generated geometry."""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .run_v1_cascade import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for x in ('manifest','shape-case','template','output'):p.add_argument('--'+x,type=Path,required=True)
    p.add_argument('--gpu',choices=['0','1','2','3'],required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu or a.output.exists():raise ValueError('GPU assignment and fresh output required')
    torch.set_num_threads(2);torch.cuda.set_device(0);os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    case=json.loads((a.shape_case/'receipt.json').read_text());sid=case['sample_id']
    if case['ss_arm']!='selected' or case['prediction_sha256']!=sha(a.shape_case/'prediction.npz') or case['native_shape_guides_sha256']!=sha(a.shape_case/'native-shape-guides.pt'):raise ValueError('actual selected SS shape and native guides required')
    rows=[x for x in map(json.loads,a.manifest.read_text().splitlines()) if x['sample_id']==sid]
    if len(rows)!=1:raise ValueError('one sample manifest required')
    row=rows[0];plan=json.loads(a.template.read_text())
    if plan['mode']!='top1-only' or plan['checkpoint_kind']!='trained-native1900':raise ValueError('frozen selected PBR required')
    from pixal3d import models
    from pixal3d.modules.sparse import SparseTensor
    from pixal3d.pipelines import pixal3d_image_to_3d as pipeline_module,samplers
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    from pixal3d.modules.r_mediated_pbr_source import RMediatedPBRSourceConditioner
    from pixal3d.trainers.flow_matching.r_mediated_pbr_source import pbr_source_kwargs
    from experiments.deformnext_retargeting.cascade_from_ss_artifact import load_stage
    from experiments.deformnext_retargeting.cascade_artifact_inputs import prepared_views
    from experiments.deformnext_retargeting.frozen_projection_streaming import install_frozen_projection_streaming
    from experiments.deformnext_retargeting.generated_r_mediated_pbr import generated_r_pbr_conditioning,image_only_row
    from experiments.deformnext_retargeting.archived_native_guides import load_guides
    from experiments.deformnext_retargeting.pbr_replay_artifacts import GEOMETRY_FIELDS,replayed_prediction
    install_frozen_projection_streaming(max_assets=1)
    cfg=json.loads(Path(plan['PBR_config']).read_text());pcfg=json.loads(Path(plan['pipeline_config']).read_text())['args'];spec=pcfg['tex_slat_sampler']
    if spec['params']!=plan['sampler_params']:raise ValueError('sampler mismatch')
    flow=load_stage(cfg,plan['PBR_checkpoint']);loaded={'tex_slat_flow_model_1024':flow}
    for name,path in plan['decoders'].items():loaded[name]=models.from_pretrained(path).eval().requires_grad_(False)
    pipe=pipeline_module.Pixal3DImageTo3DPipeline(loaded,tex_slat_sampler=getattr(samplers,spec['name'])(**spec['args']),tex_slat_sampler_params=spec['params'],shape_slat_normalization=pcfg['shape_slat_normalization'],tex_slat_normalization=pcfg['tex_slat_normalization'],low_vram=True)
    ec=dict(cfg['trainer']['args']['image_cond_model']['args']);ec['model_name']=plan['DINO_local_path']
    extractor=DinoV3ProjFeatureExtractor(**ec).eval().requires_grad_(False);extractor._load_naf();extractor.forward=torch.autocast('cuda',dtype=torch.bfloat16)(extractor.forward)
    pipe.image_cond_model_tex_1024=extractor;pipe.to(torch.device('cuda:0'))
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(cfg['training_seed']);conditioner=RMediatedPBRSourceConditioner(**pbr_source_kwargs(cfg['trainer']['args']['deform_conditioner']))
        conditioner.load_state_dict(torch.load(plan['conditioner_checkpoint'],map_location='cpu',weights_only=True),strict=True)
    conditioner.cuda().eval().requires_grad_(False)
    with np.load(a.shape_case/'prediction.npz') as z:original={key:z[key].copy() for key in z.files}
    shape=SparseTensor(feats=torch.from_numpy(original['shape_feats']).cuda(),coords=torch.from_numpy(original['shape_coords']).int().cuda())
    guides=load_guides(a.shape_case/'native-shape-guides.pt',shape,SparseTensor)
    images=[Image.open(x).copy() for x in [row['target_anchor'],*row['source_images']]]
    camera=dict(camera_angle_x=row['camera_angle_x'],distance=[float(np.linalg.norm(np.asarray(t)[:3,3])) for t in row['transform_matrix']],transform_matrix=row['transform_matrix'],mesh_scale=row['mesh_scale'],grid_resolution_override=64)
    events=[];pipeline_module.pil_views_to_tensor=prepared_views;start=time.monotonic()
    with generated_r_pbr_conditioning(pipe,conditioner=conditioner,row=image_only_row(row),path_map={},events=events),torch.inference_mode():
        torch.manual_seed(case['seed']);cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
        condition=pipe.get_proj_cond_shape(extractor,images,shape.coords,**camera)
        if not torch.equal(cpu_rng,torch.get_rng_state()) or not torch.equal(cuda_rng,torch.cuda.get_rng_state()):raise ValueError('conditioning changed sampler RNG')
        texture=pipe.sample_tex_slat(condition,flow,shape);texture_feats=texture.feats.float().cpu().numpy();del condition
        decoded=pipe.decode_tex_slat(texture,guides)
        if decoded.shape[0]!=1 or torch.any(decoded.coords[:,0]!=0):raise ValueError('invalid native PBR output')
        prediction=replayed_prediction(original,decoded.coords[:,1:].cpu().numpy(),decoded.feats.float().cpu().numpy(),texture_feats)
    if len(events)!=1 or any(prediction[k].tobytes()!=original[k].tobytes() for k in GEOMETRY_FIELDS):raise ValueError('texture must preserve native geometry')
    a.output.mkdir(parents=True);np.savez_compressed(a.output/'prediction.npz',**prediction)
    result={'status':'SELECTED_FULL_PBR_GENERATED_NO_TARGET_3D_GT','sample_id':sid,'prediction_sha256':sha(a.output/'prediction.npz'),'shape_sha256':case['prediction_sha256'],'guides_sha256':case['native_shape_guides_sha256'],'PBR_checkpoint_sha256':sha(plan['PBR_checkpoint']),'conditioner_sha256':sha(plan['conditioner_checkpoint']),'pbr_arm':'top1-only','geometry_byte_identity':True,'GT_geometry_used':False,'seed':case['seed'],'seconds':round(time.monotonic()-start,2),'events':events}
    (a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
