"""Engineering-only native PBR conditioning at real GT/predicted sparse support.

No denoiser, sampler, optimizer or quality metric is constructed. A temporary
R-output bias intervention exercises both DINO and NAF delivery, then is fully
restored. These are not trained weights and must never enter a scientific run.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
from PIL import Image
import torch

from .cascade_view_roles import ClosedInputs, file_sha256
from .evaluate_pbr_streaming import tensor_sha
from .generated_r_mediated_pbr import generated_r_pbr_conditioning, load_generated_r_transport
from .r_mediated_native_runtime import state_sha


@contextmanager
def output_bias_intervention(module):
    bias = module.retargeter.net[-1].bias
    before = bias.detach().clone()
    try:
        with torch.no_grad(): bias.fill_(1.)
        yield
    finally:
        with torch.no_grad(): bias.copy_(before)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if gpu not in {'0', '1', '2', '3'} or gpu not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(','):
        raise ValueError('one exclusively assigned physical GPU0-3 required')
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = json.loads(args.plan.read_text()); check = ClosedInputs(plan['canonical_local_root'])

    def guard():
        check.check(args.plan, args.plan.parent/'file4-closure.json')
        for path, digest in {**plan['source_SHA256'], **plan['input_SHA256']}.items():
            if file_sha256(path) != digest:
                raise ValueError('declared generated-support proof source/input changed: '+path)
        for path, receipt in plan['closed_inputs'].items(): check.check(path, receipt)

    guard()
    from pixal3d.modules.r_mediated_pbr_source import RMediatedPBRSourceConditioner
    from pixal3d.trainers.flow_matching.r_mediated_pbr_source import RMediatedPBRNativeTrainer, pbr_source_kwargs
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        DinoV3ProjFeatureExtractor, ImageConditionedProjMixin,
    )
    from pixal3d.pipelines import pixal3d_image_to_3d as pipeline_module
    from .cascade_artifact_inputs import prepared_views
    from .frozen_projection_streaming import install_frozen_projection_streaming
    output = Path(plan['output']); output.mkdir(parents=True, exist_ok=False)
    configs = {mode: json.loads(Path(path).read_text()) for mode, path in plan['source_configs'].items()}
    if set(configs) != {'top1-only', 'top1-plus-feature', 'raw-topk-plus-feature'}:
        raise ValueError('all three own native source modes required')
    reference = configs['top1-only']
    install_frozen_projection_streaming(max_assets=1)
    extractor = DinoV3ProjFeatureExtractor(**reference['trainer']['args']['image_cond_model']['args']).eval().requires_grad_(False)
    extractor._load_naf()
    extractor.cuda()
    extractor.forward = torch.autocast('cuda', dtype=torch.bfloat16)(extractor.forward)
    modules = {}
    for mode, cfg in configs.items():
        if cfg['trainer']['args']['image_cond_model'] != reference['trainer']['args']['image_cond_model']:
            raise ValueError('all source modes must share the original native encoder')
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(101)
            module = RMediatedPBRSourceConditioner(**pbr_source_kwargs(cfg['trainer']['args']['deform_conditioner']))
            if mode != 'top1-only':
                utility = cfg['trainer']['args']['ray_feature_utility']
                module.load_utility(torch.load(utility['checkpoint'], map_location='cpu', weights_only=True))
        modules[mode] = module.cuda().eval().requires_grad_(False)
    pipeline = SimpleNamespace(device=torch.device('cuda:0'), low_vram=False, image_cond_model_tex_1024=extractor)
    pipeline.get_proj_cond_shape = MethodType(pipeline_module.Pixal3DImageTo3DPipeline.get_proj_cond_shape, pipeline)
    previous_views = pipeline_module.pil_views_to_tensor
    pipeline_module.pil_views_to_tensor = prepared_views
    reports = []
    torch.cuda.reset_peak_memory_stats()
    try:
        for case in plan['cases']:
            row = json.loads(Path(case['image_only_row']).read_text())
            images = [Image.open(path).copy() for path in case['RGB_paths']]
            transform = torch.tensor(row['transform_matrix'], dtype=torch.float32, device='cuda')[None]
            camera = dict(transform_matrix=transform,
                distance=torch.linalg.vector_norm(transform[:, :, :3, 3], dim=-1),
                camera_angle_x=torch.tensor(row['camera_angle_x'], dtype=torch.float32, device='cuda')[None],
                mesh_scale=torch.tensor([row['mesh_scale']], dtype=torch.float32, device='cuda'))
            call = dict(camera_angle_x=row['camera_angle_x'], distance=camera['distance'][0].tolist(),
                mesh_scale=row['mesh_scale'], transform_matrix=row['transform_matrix'], grid_resolution_override=64)
            with np.load(case['support']) as archive:
                supports = {key: torch.from_numpy(archive[key].copy()).int().cuda()
                    for key in ['GT_training_parity_coords', 'actual_generated_coords'] if key in archive}
            for kind, coords in supports.items():
                cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
                native = pipeline.get_proj_cond_shape(extractor, images, coords, **call)
                transport = {k: v.cuda() for k, v in load_generated_r_transport(row, coords, path_map=plan['path_map']).items()}
                for mode, module in modules.items():
                    initial = state_sha(module.state_dict().items())
                    events = []
                    with generated_r_pbr_conditioning(pipeline, conditioner=module, row=row, path_map=plan['path_map'], events=events):
                        cold = pipeline.get_proj_cond_shape(extractor, images, coords, **call)
                        for role in ['cond', 'neg_cond']:
                            if (not torch.equal(cold[role]['global'], native[role]['global'])
                                    or not torch.equal(cold[role]['proj'].feats, native[role]['proj'].feats)
                                    or not torch.equal(cold[role]['proj'].coords, coords)):
                                raise ValueError('actual cold source pipeline differs from untouched native PBR')
                        with output_bias_intervention(module), torch.no_grad():
                            active = pipeline.get_proj_cond_shape(extractor, images, coords, **call)
                            facade = SimpleNamespace(image_cond_model=extractor, _image_cond_module=extractor,
                                image_attn_mode='proj', deform_conditioner=module)
                            facade.encode_image_proj = MethodType(ImageConditionedProjMixin.encode_image_proj, facade)
                            pixels = prepared_views(images, 1024, pipeline.device)
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                train_cond, train_negative = RMediatedPBRNativeTrainer._encode_deform_views(
                                    facade, pixels, dict(camera, coords=coords), transport)
                            if not torch.equal(active['cond']['proj'].feats, train_cond['proj']):
                                raise ValueError('actual pipeline and training source consumers differ under intervention')
                            if (not torch.equal(active['cond']['global'], train_cond['global'])
                                    or not torch.equal(active['cond']['global'], native['cond']['global'])):
                                raise ValueError('R-mediated source correction changed the native four-view global carrier')
                            if any(not torch.equal(active['neg_cond'][k].feats if k == 'proj' else active['neg_cond'][k],
                                    value) for k, value in train_negative.items()):
                                raise ValueError('actual negative native PBR conditioning differs')
                            delta = active['cond']['proj'].feats.float()-native['cond']['proj'].feats.float()
                            rms = [float(value.square().mean().sqrt()) for value in delta.split(1024, dim=-1)]
                            if not all(value > 0 for value in rms):
                                raise ValueError('actual DINO and NAF R delivery must both survive native consumer casting')
                    if initial != state_sha(module.state_dict().items()):
                        raise ValueError('temporary R proof intervention was not restored exactly')
                    reports.append(dict(case=case['label'], support=kind, mode=mode, actual_tokens=len(coords),
                        actual_coords_SHA256=tensor_sha(coords), cold_native_fields_exact=True,
                        nonzero_pipeline_and_native_training_consumer_exact=True,
                        intervention_R_output_bias=1., intervention_is_trained_weight=False,
                        DINO_NAF_consumer_delta_RMS=rms, source_state_restored=True, events=events))
                if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
                    raise ValueError('generated-support engineering conditioning consumed sampler RNG')
    finally:
        pipeline_module.pil_views_to_tensor = previous_views
    guard()
    audit = dict(status='PASS_ACTUAL_GENERATED_R_PBR_SUPPORT_CONDITIONING_ONLY',
        actual_cases=reports, actual_GPU=gpu,
        GT_support_loaded_only_for_explicit_training_parity=True,
        actual_generated_support_checked_separately=True, image_only_loader_has_no_GT_fields=True,
        native_pipeline_cold_identity=True, native_training_source_consumer_reused=True,
        nonzero_R_DINO_and_NAF_delivery_proven=True, temporary_intervention_restored=True,
        denoiser_loaded=False, sampler_calls=0, optimizer_updates=0,
        prototype_parameters_discarded=True, scientific_training_admitted=False,
        actual_PBR_replay_generation_proven=False, quality_gain_proven=False, full_goal_complete=False,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30, verified_inputs=check.inputs)
    (output/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(dict(status=audit['status'], actual_conditions=len(reports), full_goal_complete=False)), flush=True)


if __name__ == '__main__':
    main()
