"""Native one-asset fixed-use source-value proof; synthetic gates, no training.

This does not measure the quality of a trained source model. It verifies that
the saved-weight intervention can reach actual native projection and SS loss
while retaining the original source-use decision and matching readout weights.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    allowed = os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(',')
    if len(visible) != 1 or visible[0] not in allowed or visible[0] not in {'0', '1', '2', '3'}:
        raise ValueError('one explicitly assigned physical GPU0–3 required')
    from pixal3d import datasets, models
    from pixal3d.trainers.flow_matching.target_first_ray_v2 import TargetFirstRayImageConditionedProjFlowMatchingCFGTrainerV2 as Trainer
    from pixal3d.utils.data_utils import recursive_to_device
    from experiments.multiview_finetune.runtime import install_hooks
    from .quick_metrics import panel_indices
    from .target_first_value_control import fixed_use_value_control
    from train import setup_rng

    install_hooks()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    cfg = json.loads(args.config.read_text())
    if cfg['trainer']['name'] != Trainer.__name__ or cfg['training_seed'] != 101:
        raise ValueError('explicit corrected target-first V2 recipe required')
    setup_rng(0, 101)
    ds = getattr(datasets, cfg['dataset']['name'])(cfg['validation']['roots_file'],
                                                **cfg['validation']['dataset_args'])
    indices = panel_indices(ds, cfg['validation']['quick_metrics'])
    if len(indices) != 32:
        raise ValueError('fixed development32 population required, never held-out test')
    with torch.device('meta'):
        md = {name: getattr(models, spec['name'])(**spec['args'])
              for name, spec in cfg['models'].items()}
    kw = copy.deepcopy(cfg['trainer']['args'])
    kw.update(batch_size_per_gpu=1, batch_split=1, num_workers=0, max_steps=1, prefetch_data=False)
    args.output.mkdir(parents=True, exist_ok=False)
    trainer = Trainer(md, ds, **kw, output_dir=str(args.output / 'native-trainer'),
                      load_dir=None, step=None, wandb_run=None)
    for model in trainer.models.values():
        model.eval()
    trainer.p_uncond = 0
    module = trainer.deform_conditioner
    originals = {name: p.detach().clone() for name, p in module.named_parameters() if p.requires_grad}

    def frozen_digest():
        digest = hashlib.sha256()
        for name, value in module.q2c.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().contiguous().cpu().numpy().tobytes())
        return digest.hexdigest()

    frozen_before = frozen_digest()
    pack = ds.collate_fn([ds[indices[0]]])
    pack = pack if isinstance(pack, dict) else pack[0]
    pack = recursive_to_device(pack, trainer.device)
    pack.pop('_dataset_name', None)
    pack.pop('_sha256', None)
    target = pack.pop('x_0')
    camera = trainer._extract_camera_info(dict(pack))
    transport = trainer._pop_deform_transport(dict(pack))
    conditioning_events = []
    loss_events = []
    try:
        # The source model is initially zero. Nonzero synthetic parameters
        # exercise the causal boundary; they are restored and never saved as
        # a scientific checkpoint or counted as optimizer updates.
        with torch.no_grad(), torch.random.fork_rng(devices=[0]):
            torch.manual_seed(20261020)
            module.fusion_gate.fill_(.2)
            module.retargeter.net[-1].weight.normal_(std=.001)
            module.evidence_use.network[-1].weight.normal_(std=.03)
            if module.task_attention is not None:
                module.task_attention.content_output.weight.normal_(std=.002)
        with torch.no_grad(), torch.autocast('cuda', dtype=trainer.mix_precision_dtype):
            normal, negative = trainer._encode_deform_views(pack['cond'], camera, transport)
            use_normal = trainer._last_source_use_probability.clone()
            with fixed_use_value_control(module, conditioning_events):
                wrong, wrong_negative = trainer._encode_deform_views(pack['cond'], camera, transport)
                use_wrong = trainer._last_source_use_probability.clone()
            anchor, _ = trainer._encode_deform_views(pack['cond'], camera, None)
            empty_transport = dict(transport, source_valid=torch.zeros_like(transport['source_valid']))
            absent, _ = trainer._encode_deform_views(pack['cond'], camera, empty_transport)
        if (not torch.equal(normal['global'], wrong['global'])
                or not torch.equal(normal['global'], anchor['global'])
                or any(not torch.equal(negative[k], wrong_negative[k]) for k in negative)
                or any(not torch.equal(absent[k], anchor[k]) for k in anchor)):
            raise ValueError('source value control changed target/global/negative/gate or missing-source fallback')
        projection_difference = float((normal['proj'] - wrong['proj']).float().abs().max())
        if not np.isfinite(projection_difference) or projection_difference <= 0:
            raise ValueError('wrong source values must reach actual native SS projection')
        cpu_rng = torch.get_rng_state()
        gpu_rng = torch.cuda.get_rng_state()
        with torch.no_grad(), torch.autocast('cuda', dtype=trainer.mix_precision_dtype):
            normal_loss, _ = trainer.training_losses(x_0=target, **pack)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(gpu_rng)
            with fixed_use_value_control(module, loss_events):
                wrong_loss, _ = trainer.training_losses(x_0=target, **pack)
        loss_difference = abs(float(normal_loss['loss']) - float(wrong_loss['loss']))
        if not np.isfinite(loss_difference) or loss_difference <= 0:
            raise ValueError('fixed-use wrong source values must reach the native SS loss with the same noise')
    finally:
        with torch.no_grad():
            for name, value in module.named_parameters():
                if name in originals:
                    value.copy_(originals[name])
    if frozen_digest() != frozen_before or any(
            not torch.equal(value, originals[name]) for name, value in module.named_parameters() if name in originals):
        raise ValueError('all temporary parameters and frozen matching weights must be restored')
    result = dict(
        status='PASS_ACTUAL_NATIVE_FIXED_USE_SOURCE_VALUE_INTERVENTION',
        sample_id=ds.rows[indices[0]]['sample_id'], ray_mode=module.ray_mode,
        conditioning_events=conditioning_events, loss_events=loss_events,
        source_use_gate_exact_same_forward=all(event['source_use_probability_exact_same_forward']
            for event in conditioning_events + loss_events),
        source_use_gate_equal_across_separate_baseline_forward=bool(torch.equal(use_normal, use_wrong)),
        separate_baseline_forward_gate_max_abs_difference=float((use_normal-use_wrong).float().abs().max()),
        native_target_global_and_negative_exact=True,
        no_source_target_fallback_exact=True,
        native_projection_max_abs_difference=projection_difference,
        native_SS_loss_normal=float(normal_loss['loss']), native_SS_loss_wrong=float(wrong_loss['loss']),
        native_SS_loss_abs_difference=loss_difference, paired_SS_loss_noise=True,
        synthetic_nonzero_source_parameters=True, all_temporary_parameters_restored=True,
        frozen_matcher_state_exact=True, optimizer_updates=0,
        trained_model_source_utility_proven=False, trained_quality_gain_established=False,
        initial_V1_function_identity=False, heldout_test_measured=False,
        peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30, full_goal_complete=False)
    (args.output / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {'conditioning_events', 'loss_events'}}), flush=True)


if __name__ == '__main__':
    main()
