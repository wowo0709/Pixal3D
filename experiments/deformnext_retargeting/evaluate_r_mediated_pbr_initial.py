"""Read-only generated32 identity for original MV PBR16850 and cold source arms.

This is an anchor-camera, fixed GT-shape stage check. It neither trains nor
estimates full-cascade/source-preservation quality. No discarded proof weight
or optimizer is loaded.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .cascade_view_roles import ClosedInputs, file_sha256
from .evaluate_pbr_streaming import aggregate_metrics, tensor_sha
from .probe_r_mediated_pbr_native import MODES, validate_config
from .r_mediated_native_runtime import rng_identity, state_sha


IDENTITY_FIELDS = (
    'sample_id', 'source_cluster_id', 'conditioning_input_sha256', 'seed',
    'actual_noise_SHA256', 'actual_RGB_input_SHA256',
    'actual_GT_PBR_latent_SHA256', 'actual_GT_Shape_condition_SHA256',
    'actual_camera_SHA256', 'actual_prediction_PBR_latent_SHA256',
)


def require_initial_identity(records):
    """Compare actual generation, not a panel name or its aggregate metric."""
    if set(records) != MODES:
        raise ValueError('all four actual PBR cold modes required')
    reference = records['vanilla']
    if len(reference) != 32 or len({r['sample_id'] for r in reference}) != 32:
        raise ValueError('exactly32 unique original-PBR generated assets required')
    for mode, rows in records.items():
        if len(rows) != len(reference):
            raise ValueError('incomplete PBR initial identity panel: '+mode)
        for left, right in zip(reference, rows):
            if any(left[key] != right[key] for key in IDENTITY_FIELDS):
                raise ValueError('actual PBR initial input/noise/prediction differs: '+mode)
            if left['metrics'] != right['metrics']:
                raise ValueError('actual PBR initial rendering metrics differ: '+mode)


@contextmanager
def audited_sampling(quick, trainer, events):
    """Capture actual sparse output and restore both hooks after any exception."""
    original_sample = quick._sample

    def sampled(actual_trainer, dataset, index, steps, seed):
        if actual_trainer is not trainer:
            raise ValueError('only the declared read-only trainer may generate')
        original_factory = trainer.get_sampler
        row = dataset.rows[index]
        record = {key: row[key] for key in
                  ['sample_id', 'source_cluster_id', 'conditioning_input_sha256']}
        record['seed'] = seed
        noises = []

        def factory():
            sampler = original_factory()
            original_call = sampler.sample

            def call(*args, **kwargs):
                noises.append(tensor_sha(kwargs['noise']))
                return original_call(*args, **kwargs)

            sampler.sample = call
            return sampler

        trainer.get_sampler = factory
        try:
            prediction, target, pack = original_sample(trainer, dataset, index, steps, seed)
        finally:
            trainer.get_sampler = original_factory
        if len(noises) != 1:
            raise ValueError('one actual initial PBR generation per asset required')
        record.update(actual_noise_SHA256=noises[0],
            actual_RGB_input_SHA256=tensor_sha(pack['cond']),
            actual_GT_PBR_latent_SHA256=tensor_sha(target),
            actual_GT_Shape_condition_SHA256=tensor_sha(pack['concat_cond']),
            actual_prediction_PBR_latent_SHA256=tensor_sha(prediction),
            actual_camera_SHA256={key: tensor_sha(pack[key]) for key in
                ['camera_angle_x', 'camera_distance', 'transform_matrix', 'mesh_scale']})
        events.append(record)
        return prediction, target, pack

    quick._sample = sampled
    try:
        yield
    finally:
        quick._sample = original_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if gpu not in {'0', '1', '2', '3'} or gpu not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(','):
        raise ValueError('one exclusively assigned physical GPU0-3 required')
    plan = json.loads(args.plan.read_text())
    check = ClosedInputs(plan['canonical_local_root'])

    def guard():
        check.check(args.plan, args.plan.parent/'file4-closure.json')
        for path, digest in {**plan['source_SHA256'], **plan['input_SHA256']}.items():
            if file_sha256(path) != digest:
                raise ValueError('initial PBR source/input changed: '+path)
        for path, receipt in plan['closed_inputs'].items():
            check.check(path, receipt)

    guard()
    cfg = json.loads(Path(plan['config']).read_text())
    original = json.loads(Path(plan['original_config']).read_text())
    validate_config(cfg, original, plan['mode'])
    validation = plan['validation']
    options = validation['quick_metrics']
    if (validation['dataset_name'] != cfg['dataset']['name']
            or validation['dataset_args'] != dict(cfg['dataset']['args'], expected_split='validation')
            or options['panel_size'] != 32 or options['sampling_steps'] != 12
            or options['render_resolution'] != 256 or not options['enabled']
            or not options.get('panel_file') or not options.get('seed_map_file')):
        raise ValueError('fixed native12/CFG3 32-asset PBR stage protocol required')
    from PIL import Image
    from pixal3d import models
    from pixal3d.datasets.deformnext import DeformNeXtPBR1024
    from pixal3d.datasets.deformnext_feature import DeformNeXtFeaturePBR1024
    from pixal3d.trainers.flow_matching.r_mediated_pbr_source import RMediatedPBRNativeTrainer
    from pixal3d.trainers.flow_matching.sparse_flow_matching import ImageConditionedProjSparseFlowMatchingCFGTrainer
    from train import setup_rng
    from . import quick_metrics as quick
    from .frozen_projection_streaming import install_frozen_projection_streaming

    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    setup_rng(0, cfg['training_seed'])
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    install_frozen_projection_streaming(max_assets=1, audit_directory=output/'projection-streaming-audit')
    Dataset = DeformNeXtPBR1024 if plan['mode'] == 'vanilla' else DeformNeXtFeaturePBR1024
    train = Dataset(cfg['data_dir'], **cfg['dataset']['args'])
    dataset = Dataset(validation['roots_file'], **validation['dataset_args'])
    train_ids = [row['sample_id'] for row in train.rows]
    if len(train_ids) != len(set(train_ids)) or len(train_ids) != 9139:
        raise ValueError('complete common eligible TRAIN9139 required at constructor')
    indices = quick.panel_indices(dataset, options)
    ids = [dataset.rows[i]['sample_id'] for i in indices]
    if len(ids) != len(set(ids)) or len(ids) != 32 or set(ids) & set(train_ids):
        raise ValueError('fixed32 distinct validation assets disjoint from TRAIN required')
    with torch.device('meta'):
        md = {name: getattr(models, spec['name'])(**spec['args']) for name, spec in cfg['models'].items()}
    Trainer = ImageConditionedProjSparseFlowMatchingCFGTrainer if plan['mode'] == 'vanilla' else RMediatedPBRNativeTrainer
    trainer = Trainer(md, train, **copy.deepcopy(cfg['trainer']['args']), output_dir=str(output/'native'),
        load_dir=None, step=None, wandb_run=None)
    source = {} if plan['mode'] == 'vanilla' else {
        name: value for name, value in trainer.deform_conditioner.named_parameters() if value.requires_grad}
    lora = {name: value for name, value in trainer.models['denoiser'].named_parameters() if value.requires_grad}
    expected = 0 if plan['mode'] == 'vanilla' else (10 if plan['mode'] == 'top1-only' else 19)
    if len(source) != expected or len(lora) != 360 or trainer.optimizer.state or trainer.step != 0:
        raise ValueError('original complete PBR360 and fresh physical source0/10/19 required')
    constructor_rng = rng_identity()
    initial_lora = state_sha(lora.items())
    common_r = state_sha((n, p) for n, p in source.items() if not n.startswith('ray_readout.')) if source else None
    before = {name: state_sha(model.state_dict().items()) for name, model in trainer.models.items()}
    for model in trainer.models.values():
        model.eval()
    trainer.p_uncond = 0.
    seeds = quick.panel_seeds(dataset, indices, options)
    events, numeric, panels = [], [], []
    began = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with audited_sampling(quick, trainer, events), torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for number, (index, seed) in enumerate(zip(indices, seeds)):
            metrics, panel = quick._pbr_metrics(trainer, dataset, [index], dict(options, seed=seed))
            if len(events) != number+1 or not all(math.isfinite(v) for v in metrics.values()):
                raise ValueError('one finite actual generated PBR asset required')
            numeric.append(metrics)
            events[-1]['metrics'] = metrics
            Image.fromarray(panel).save(output/f'case{number:03d}.png')
            if number < 8:
                panels.append(panel)
            with (output/'per-asset.partial.jsonl').open('a') as stream:
                stream.write(json.dumps(events[-1])+'\n')
            (output/'progress.json').write_text(json.dumps(dict(completed=number+1,
                expected=32, seconds=time.monotonic()-began))+'\n')
            print(json.dumps(dict(completed=number+1, expected=32, mode=plan['mode'])), flush=True)
    if trainer.optimizer.state or trainer.step != 0 or before != {
            name: state_sha(model.state_dict().items()) for name, model in trainer.models.items()}:
        raise ValueError('initial generated check must not update any model or optimizer state')
    guard()
    report = dict(status='PASS_ACTUAL_NATIVE_PBR_COLD_GENERATED32', mode=plan['mode'],
        physical_GPU=gpu, source_multiview_step=16850, source_tensors=expected,
        original_checkpoint_SHA256=file_sha256(cfg['trainer']['args']['continuation_ckpt']['denoiser']),
        initial_LoRA_SHA256=initial_lora, initial_common_R10_SHA256=common_r,
        post_constructor_RNG=constructor_rng, original_and_source_states_unchanged=True,
        actual_TRAIN_order_SHA256=hashlib.sha256(json.dumps(train_ids, separators=(',', ':')).encode()).hexdigest(),
        actual_TRAIN_assets=len(train), actual_validation_assets=len(dataset),
        actual_validation_dataset=validation['dataset_name'], quick_panel_ids=ids,
        per_asset=events, quick_metrics=aggregate_metrics(numeric),
        sampling_steps=12, guidance=3., generation_batch=1, ao_sampling_seed=0,
        geometry='fixed_GT_target_Shape', target_modality='rgb', input_states=['M1', 'M0', 'M0', 'M0'],
        evaluation_only=True, optimizer_updates=0, optimizer_states=0, full_cascade=False,
        source_preservation_metrics=False, actual_all_four_identity_proven=False,
        full_micro_capacity_proven=False, scientific_training_admitted=False,
        quality_gain_proven=False, full_goal_complete=False,
        seconds=time.monotonic()-began, peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        verified_inputs=check.inputs)
    (output/'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
    Image.fromarray(np.concatenate(panels, axis=0)).save(output/'panel.png')
    print(json.dumps(dict(status=report['status'], mode=plan['mode'], assets=len(events))), flush=True)


if __name__ == '__main__':
    main()
