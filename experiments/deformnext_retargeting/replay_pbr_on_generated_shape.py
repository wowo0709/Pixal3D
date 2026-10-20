"""Replay a complete native PBR checkpoint on an archived generated shape.

The original SS/Shape prediction and raw mesh are preserved. Texture sampling
and voxel decoding run again with an explicitly paired PBR-only noise seed.
No ground-truth sparse support, shape latent or projection map is consumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from .cascade_from_ss_artifact import load_stage, sha
from .pbr_replay_artifacts import GEOMETRY_FIELDS, replayed_prediction


def tensor_digest(tensor):
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(data.tobytes()).hexdigest()


def run(plan):
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (gpu not in {'0', '1', '2', '3'}
            or gpu not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(',')):
        raise ValueError('one explicitly assigned physical GPU0-3 required')
    if plan['arm'] not in {'vanilla', 'retargeting', 'direct'}:
        raise ValueError('explicit vanilla, physical retargeting-only or Direct arm required')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(plan['allocator_fraction'], 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    root = Path(plan['generated_shape_root'])
    original_report = json.loads((root/'generation.json').read_text())
    if (original_report['sample_id'] != plan['row']['sample_id']
            or original_report['actual_resolution'] != 1024
            or original_report['GT_geometry_or_coords_used_in_generation']
            or original_report['per_output_alignment']):
        raise ValueError('an unaligned actual generated1024 prediction for this sample required')
    original_sha = sha(root/'prediction.npz')
    with np.load(root/'prediction.npz') as archive:
        original = {name: archive[name].copy() for name in archive.files}
    from pixal3d import models
    from pixal3d.modules.sparse import SparseTensor
    from pixal3d.pipelines import pixal3d_image_to_3d as module
    from pixal3d.pipelines import samplers
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    from .cascade_artifact_inputs import prepared_views
    from .frozen_projection_streaming import install_frozen_projection_streaming
    from .generated_pbr_conditioning_v2 import install_generated_pbr_conditioning
    install_frozen_projection_streaming(max_assets=1)
    cfg = json.loads(Path(plan['PBR_config']).read_text())
    pipeline_cfg = json.loads(Path(plan['pipeline_config']).read_text())['args']
    spec = pipeline_cfg['tex_slat_sampler']
    if plan['sampler_params'] != spec['params']:
        raise ValueError('this replay protocol preserves the declared native pipeline PBR sampler')
    loaded = {'tex_slat_flow_model_1024': load_stage(cfg, plan['PBR_checkpoint'])}
    if set(plan['decoders']) != {'shape_slat_decoder', 'tex_slat_decoder'}:
        raise ValueError('both original explicitly pinned local native VAE decoders required')
    for name, path in plan['decoders'].items():
        if not Path(path+'.json').is_file() or not Path(path+'.safetensors').is_file():
            raise ValueError('local native decoder config and safetensors pair required')
        loaded[name] = models.from_pretrained(path).eval().requires_grad_(False)
    pipe = module.Pixal3DImageTo3DPipeline(loaded,
        tex_slat_sampler=getattr(samplers, spec['name'])(**spec['args']),
        tex_slat_sampler_params=spec['params'],
        shape_slat_normalization=pipeline_cfg['shape_slat_normalization'],
        tex_slat_normalization=pipeline_cfg['tex_slat_normalization'], low_vram=True)
    encoder_args = dict(cfg['trainer']['args']['image_cond_model']['args'])
    encoder_args['model_name'] = plan['DINO_local_path']
    if encoder_args['image_size'] != 1024 or encoder_args['grid_resolution'] != 64:
        raise ValueError('the native PBR1024 grid64 extractor is required')
    extractor = DinoV3ProjFeatureExtractor(**encoder_args).eval().requires_grad_(False)
    if extractor.use_naf_upsample:
        extractor._load_naf()
    extractor.forward = torch.autocast('cuda', dtype=torch.bfloat16)(extractor.forward)
    pipe.image_cond_model_tex_1024 = extractor
    pipe.to(torch.device('cuda:0'))
    module.pil_views_to_tensor = prepared_views
    events = []
    conditioner = None
    if plan['arm'] != 'vanilla':
        conditioner_cfg = json.loads(Path(plan['conditioner_config']).read_text())
        args = conditioner_cfg['trainer']['args']['deform_conditioner']
        if plan['arm'] == 'retargeting':
            from .retarget_only_pbr_contract import build_retarget_only
            conditioner, _ = build_retarget_only(args)
        else:
            from pixal3d.modules.direct_ray_retargeting import DirectRayRetargetingConditioner
            conditioner = DirectRayRetargetingConditioner(**args)
        conditioner.load_state_dict(torch.load(plan['conditioner_checkpoint'],
            map_location='cpu', weights_only=True), strict=True)
        conditioner.cuda().eval().requires_grad_(False)
        data = conditioner_cfg['dataset']['args']
        install_generated_pbr_conditioning(pipe, conditioner=conditioner,
            row=plan['row'], path_map=data['path_map'], match_root=data['match_root'],
            expected_q2c_sha256=data['expected_q2c_sha256'],
            physical_deformation_query=plan['arm']=='direct', events=events)
    images = [Image.open(path).copy() for path in plan['RGB_paths']]
    if len(images) != 4 or plan['input_states'] != ['M1', 'M0', 'M0', 'M0']:
        raise ValueError('the same explicit edited RGB plus three source inputs required')
    shape = SparseTensor(feats=torch.from_numpy(original['shape_feats']).cuda(),
                         coords=torch.from_numpy(original['shape_coords']).int().cuda())
    row = plan['row']
    camera = dict(camera_angle_x=row['camera_angle_x'],
        distance=[float(np.linalg.norm(np.asarray(t)[:3,3])) for t in row['transform_matrix']],
        transform_matrix=row['transform_matrix'], mesh_scale=row['mesh_scale'],
        grid_resolution_override=64)
    # Native sample_tex_slat draws CPU noise. All model initialization precedes
    # this reset, and neither native nor added conditioning may advance RNG.
    torch.manual_seed(int(plan['PBR_replay_seed']))
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    first_noise = []
    flow = loaded['tex_slat_flow_model_1024']
    with torch.inference_mode():
        condition = pipe.get_proj_cond_shape(extractor, images, shape.coords, **camera)
        if (not torch.equal(cpu_rng, torch.get_rng_state())
                or not torch.equal(cuda_rng, torch.cuda.get_rng_state())):
            raise ValueError('conditioning consumed the paired native PBR noise stream')
        handle = flow.register_forward_pre_hook(lambda model, arguments:
            first_noise.append(tensor_digest(arguments[0].feats)) if not first_noise else None)
        try:
            texture = pipe.sample_tex_slat(condition, flow, shape)
        finally:
            handle.remove()
        texture_feats = texture.feats.float().cpu().numpy()
        del condition
        if conditioner is not None:
            conditioner.cpu()
        torch.cuda.empty_cache()
        # The native shape decoder supplies the exact guide hierarchy for the
        # native texture VAE. Its newly extracted mesh is never substituted for
        # the archived raw mesh or used to compute a new alignment.
        decoded_meshes, subs = pipe.decode_shape_slat(shape, 1024)
        del decoded_meshes
        decoded = pipe.decode_tex_slat(texture, subs)
        if decoded.shape[0] != 1 or torch.any(decoded.coords[:,0] != 0):
            raise ValueError('one native decoded texture sparse tensor required')
        prediction = replayed_prediction(original, decoded.coords[:,1:].cpu().numpy(),
            decoded.feats.float().cpu().numpy(), texture_feats)
    np.savez_compressed(output/'prediction.npz', **prediction)
    # Only after the prediction is complete, carry forward the unchanged raw
    # geometric evaluation. No GT geometry is opened during inference.
    shutil.copyfile(root/'geometry-evaluation.npz', output/'geometry-evaluation.npz')
    if sha(root/'prediction.npz') != original_sha:
        raise ValueError('source generated artifact changed during texture replay')
    checks = {field: prediction[field].dtype == original[field].dtype
              and prediction[field].tobytes() == original[field].tobytes()
              for field in GEOMETRY_FIELDS}
    if not all(checks.values()) or len(first_noise) != 1:
        raise ValueError('raw geometry identity and actual sampler noise audit required')
    report = dict(status='PASS_ACTUAL_NATIVE_PBR_REPLAY_ON_GENERATED_SHAPE',
        sample_id=row['sample_id'], actual_resolution=1024,
        actual_HR_tokens=len(shape.coords), generated_mesh=str(output/'prediction.npz'),
        actual_SS32_tokens=original_report['actual_SS32_tokens'],
        geometry_array_byte_identity=checks, raw_surface_metrics_unchanged=True,
        metrics=original_report['metrics'], source_defined_normalization=True,
        per_output_alignment=False, fixed_shape_models=True,
        SS_source_evaluation=original_report['SS_source_evaluation'],
        SS_prediction_SHA256=original_report['SS_prediction_SHA256'],
        generated_shape_source=str(root), original_prediction_SHA256=original_sha,
        geometry_evaluation_SHA256=sha(output/'geometry-evaluation.npz'),
        GT_geometry_or_coords_used_in_generation=False,
        GT_sparse_latents_or_projection_cache_used=False,
        GT_evaluation_artifact_copied_only_after_prediction_saved=True,
        shape_decoder_used_only_for_native_texture_guide_hierarchy=True,
        PBR_arm=plan['arm'], PBR_checkpoint=plan['PBR_checkpoint'],
        PBR_checkpoint_SHA256=sha(plan['PBR_checkpoint']),
        conditioner_checkpoint=plan.get('conditioner_checkpoint'),
        additional_PBR_conditioner=conditioner is not None,
        PBR_replay_seed=plan['PBR_replay_seed'],
        PBR_noise_protocol='reset paired CPU+CUDA seed immediately before conditioning;native CPU texture noise',
        original_cascade_texture_noise_reused=False,
        actual_initial_texture_noise_SHA256=first_noise[0],
        sampler_class=spec['name'], sampler_params=spec['params'],
        transport_events=events, conditioning_preserves_CPU_and_CUDA_RNG=True,
        generation_seconds=time.monotonic()-started,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        scope=plan['scope'], training_updates=0, model_adopted=False,
        full_evaluation_complete=False, full_goal_complete=False)
    (output/'generation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ['status','PBR_arm','actual_HR_tokens',
        'generation_seconds','peak_allocated_GiB']}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.plan.read_text()))


if __name__ == '__main__':
    main()
