"""Sample new native R-source PBR on actual archived predicted geometry.

Supports explicitly separate cold engineering and completed native1900 pairs.
No GT sparse latent, mesh, projection map or old matching index enters inference.
The geometry evaluation file is copied only after the new prediction is saved.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from .cascade_from_ss_artifact import load_stage, sha
from .cascade_view_roles import ClosedInputs
from .pbr_replay_artifacts import GEOMETRY_FIELDS, replayed_prediction
from .r_pbr_replay_contract import validate_lineage, validate_generated_shape, validate_cold_comparison
from .replay_pbr_on_generated_shape import tensor_digest


def run(plan):
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not gpu.isdigit() or gpu not in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','):
        raise ValueError('one explicitly assigned physical GPU required')
    config = json.loads(Path(plan['PBR_config']).read_text())
    completion = (json.loads(Path(plan['stage_completion']).read_text())
                  if plan['checkpoint_kind'] == 'trained-native1900' else None)
    mode, kind = validate_lineage(plan, config, completion)
    from .generated_pbr_conditioning import resolve_path
    from .prepare_rgb_manifest import image_fingerprint
    resolve = lambda value: resolve_path(value,config['dataset']['args']['path_map'])
    if ([str(resolve(v))for v in [plan['row']['target_anchor'],*plan['row']['source_images']]] != plan['RGB_paths']
            or image_fingerprint(plan['row'],resolve) != plan['row']['conditioning_input_sha256']):
        raise ValueError('actual four RGB files/cameras must match the declared image-only input')
    torch.cuda.set_device(0); torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(plan['allocator_fraction'],0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    output = Path(plan['output']); output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    root = Path(plan['generated_shape_root'])
    original_report = json.loads((root/'generation.json').read_text())
    original_sha = sha(root/'prediction.npz')
    with np.load(root/'prediction.npz') as archive:
        original = {name:archive[name].copy() for name in archive.files}
    validate_generated_shape(plan,original_report,original)
    from pixal3d import models
    from pixal3d.modules.sparse import SparseTensor
    from pixal3d.pipelines import pixal3d_image_to_3d as pipeline_module, samplers
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    from .cascade_artifact_inputs import prepared_views
    from .frozen_projection_streaming import install_frozen_projection_streaming
    from .generated_r_mediated_pbr import generated_r_pbr_conditioning
    from .r_mediated_native_runtime import state_sha
    install_frozen_projection_streaming(max_assets=1)
    pipeline_cfg = json.loads(Path(plan['pipeline_config']).read_text())['args']
    spec = pipeline_cfg['tex_slat_sampler']
    if plan['sampler_params'] != spec['params']:
        raise ValueError('native pipeline texture sampler must be preserved exactly')
    flow = load_stage(config,plan['PBR_checkpoint'])
    loaded = {'tex_slat_flow_model_1024':flow}
    if set(plan['decoders']) != {'shape_slat_decoder','tex_slat_decoder'}:
        raise ValueError('the original explicit native Shape and texture VAE pair is required')
    for name,path in plan['decoders'].items():
        if not Path(path+'.json').is_file() or not Path(path+'.safetensors').is_file():
            raise ValueError('explicit local decoder config and safetensors required')
        loaded[name] = models.from_pretrained(path).eval().requires_grad_(False)
    pipe = pipeline_module.Pixal3DImageTo3DPipeline(loaded,
        tex_slat_sampler=getattr(samplers,spec['name'])(**spec['args']),tex_slat_sampler_params=spec['params'],
        shape_slat_normalization=pipeline_cfg['shape_slat_normalization'],
        tex_slat_normalization=pipeline_cfg['tex_slat_normalization'],low_vram=True)
    encoder_args = dict(config['trainer']['args']['image_cond_model']['args'])
    encoder_args['model_name'] = plan['DINO_local_path']
    if (encoder_args['image_size'],encoder_args['grid_resolution'],encoder_args['use_naf_upsample']) != (1024,64,True):
        raise ValueError('native PBR1024 grid64 DINO+NAF extractor required')
    extractor = DinoV3ProjFeatureExtractor(**encoder_args).eval().requires_grad_(False)
    extractor._load_naf()
    extractor.forward = torch.autocast('cuda',dtype=torch.bfloat16)(extractor.forward)
    pipe.image_cond_model_tex_1024 = extractor; pipe.to(torch.device('cuda:0'))
    conditioner = None; source_before = None
    if mode != 'vanilla':
        from pixal3d.modules.r_mediated_pbr_source import RMediatedPBRSourceConditioner
        from pixal3d.trainers.flow_matching.r_mediated_pbr_source import pbr_source_kwargs
        from pixal3d.trainers.flow_matching.member_feature_source_ray import verify_utility
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(config['training_seed'])
            conditioner = RMediatedPBRSourceConditioner(**pbr_source_kwargs(config['trainer']['args']['deform_conditioner']))
            if kind == 'trained-native1900':
                conditioner.load_state_dict(torch.load(plan['conditioner_checkpoint'],map_location='cpu',weights_only=True),strict=True)
            elif mode != 'top1-only':
                utility = config['trainer']['args']['ray_feature_utility']; verify_utility(utility)
                conditioner.load_utility(torch.load(utility['checkpoint'],map_location='cpu',weights_only=True))
        if sum(p.requires_grad for p in conditioner.parameters()) != (10 if mode=='top1-only' else 19):
            raise ValueError('own physical source10/19 checkpoint schema required')
        conditioner.cuda().eval().requires_grad_(False)
        source_before = state_sha(conditioner.state_dict().items())
    images = [Image.open(path).copy() for path in plan['RGB_paths']]
    shape = SparseTensor(feats=torch.from_numpy(original['shape_feats']).cuda(),
                         coords=torch.from_numpy(original['shape_coords']).int().cuda())
    row = plan['row']; events=[]
    camera = dict(camera_angle_x=row['camera_angle_x'],
        distance=[float(np.linalg.norm(np.asarray(t)[:3,3]))for t in row['transform_matrix']],
        transform_matrix=row['transform_matrix'],mesh_scale=row['mesh_scale'],grid_resolution_override=64)
    conditioning = (nullcontext() if conditioner is None else generated_r_pbr_conditioning(
        pipe,conditioner=conditioner,row=row,path_map=config['dataset']['args']['path_map'],events=events))
    previous_views = pipeline_module.pil_views_to_tensor
    pipeline_module.pil_views_to_tensor = prepared_views
    first_noise=[]
    try:
        with conditioning,torch.inference_mode():
            torch.manual_seed(plan['PBR_replay_seed'])
            cpu_rng,cuda_rng = torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
            condition = pipe.get_proj_cond_shape(extractor,images,shape.coords,**camera)
            if not torch.equal(cpu_rng,torch.get_rng_state()) or not torch.equal(cuda_rng,torch.cuda.get_rng_state()):
                raise ValueError('conditioning consumed the native paired PBR sampler noise stream')
            handle = flow.register_forward_pre_hook(lambda model,args:
                first_noise.append(tensor_digest(args[0].feats)) if not first_noise else None)
            try:
                texture = pipe.sample_tex_slat(condition,flow,shape)
            finally:
                handle.remove()
            texture_digest = tensor_digest(texture.feats)
            texture_feats = texture.feats.float().cpu().numpy()
            del condition
            if conditioner is not None:
                if state_sha(conditioner.state_dict().items()) != source_before:
                    raise ValueError('sampling changed saved source parameters/buffers')
                conditioner.cpu()
            torch.cuda.empty_cache()
            # Reuse the native predicted-shape VAE hierarchy only as the texture
            # guide. Its newly decoded mesh never replaces archived geometry.
            meshes,subs = pipe.decode_shape_slat(shape,1024); del meshes
            decoded = pipe.decode_tex_slat(texture,subs)
            if decoded.shape[0]!=1 or torch.any(decoded.coords[:,0]!=0):
                raise ValueError('one actual native decoded PBR sparse tensor required')
            decoded_coords = decoded.coords[:,1:].cpu().numpy()
            decoded_attrs = decoded.feats.float().cpu().numpy()
            prediction = replayed_prediction(original,decoded_coords,decoded_attrs,texture_feats)
    finally:
        pipeline_module.pil_views_to_tensor = previous_views
    checks = {name:prediction[name].dtype==original[name].dtype and prediction[name].tobytes()==original[name].tobytes()
              for name in GEOMETRY_FIELDS}
    if not all(checks.values()) or len(first_noise)!=1 or len(events)!=(0 if mode=='vanilla' else 1):
        raise ValueError('actual noise, one conditioning event and unchanged geometry required')
    np.savez_compressed(output/'prediction.npz',**prediction)
    shutil.copyfile(root/'geometry-evaluation.npz',output/'geometry-evaluation.npz')
    if sha(root/'prediction.npz')!=original_sha:
        raise ValueError('original generated geometry artifact changed')
    report=dict(status='PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE',sample_id=row['sample_id'],
        actual_resolution=1024,actual_HR_tokens=len(shape.coords),actual_SS32_tokens=original_report['actual_SS32_tokens'],
        generated_mesh=str(output/'prediction.npz'),geometry_array_byte_identity=checks,
        metrics=original_report['metrics'],raw_surface_metrics_unchanged=True,source_defined_normalization=True,
        per_output_alignment=False,fixed_shape_models=True,SS_source_evaluation=original_report['SS_source_evaluation'],
        SS_prediction_SHA256=original_report['SS_prediction_SHA256'],generated_shape_source=str(root),
        original_prediction_SHA256=original_sha,geometry_evaluation_SHA256=sha(output/'geometry-evaluation.npz'),
        GT_geometry_or_coords_used_in_generation=False,GT_sparse_latents_or_projection_cache_used=False,
        GT_evaluation_artifact_copied_only_after_prediction_saved=True,
        shape_decoder_used_only_for_native_texture_guide_hierarchy=True,
        PBR_arm=mode,checkpoint_kind=kind,engineering_only=plan['engineering_only'],PBR_checkpoint=plan['PBR_checkpoint'],
        PBR_checkpoint_SHA256=sha(plan['PBR_checkpoint']),conditioner_checkpoint=plan.get('conditioner_checkpoint'),
        actual_loaded_source_state_SHA256=source_before,source_weights_unchanged=True,
        PBR_replay_seed=plan['PBR_replay_seed'],actual_initial_texture_noise_SHA256=first_noise[0],
        actual_texture_latent_SHA256=texture_digest,
        actual_decoded_coordinates_SHA256=tensor_digest(decoded.coords[:,1:]),
        actual_decoded_attributes_SHA256=tensor_digest(decoded.feats.float()),
        PBR_noise_protocol='native CPU texture noise after paired CPU+CUDA seed reset; conditioning RNG unchanged',
        original_cascade_texture_noise_reused=False,sampler_class=spec['name'],sampler_params=spec['params'],
        transport_events=events,conditioning_preserves_CPU_and_CUDA_RNG=True,
        source_camera_ray_query=mode!='vanilla',generation_seconds=time.monotonic()-started,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,optimizer_updates=0,
        quality_gain_proven=False,model_adopted=False,full_evaluation_complete=False,full_goal_complete=False)
    if plan.get('cold_reference'):
        path=Path(plan['cold_reference']);check=ClosedInputs(plan['canonical_local_root'])
        check.check(path,path.parent/'file4-closure.json')
        validate_cold_comparison(report,json.loads(path.read_text()))
        report['actual_cold_native_texture_latent_and_decoded_arrays_exact']=True
    (output/'generation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],mode=mode,checkpoint_kind=kind,
        actual_HR_tokens=report['actual_HR_tokens'],generation_seconds=report['generation_seconds'])),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    check=ClosedInputs(plan['canonical_local_root'])
    def guard():
        check.check(args.plan,args.plan.parent/'file4-closure.json')
        for path,digest in {**plan['source_SHA256'],**plan['input_SHA256']}.items():
            if sha(path)!=digest:raise ValueError('bound replay source/input changed: '+path)
        for path,receipt in plan['closed_inputs'].items():check.check(path,receipt)
    guard();run(plan);guard()


if __name__=='__main__':main()
