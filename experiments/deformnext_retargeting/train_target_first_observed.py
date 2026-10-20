"""Fresh native SS training for the four observed-query source readouts.

The external resolved recipe and its release evidence own the experiment.
This entry selects the explicit trainer in this process and leaves native
LoRA loading, optimization, sampling, validation and checkpointing intact.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


TRAINERS = {
    'top1': 'ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer',
    'weighted': 'ObservedQueryWeightedTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer',
    'attention': 'ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer',
    'spatial': 'ObservedQuerySpatialTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer',
}
MODES = {
    'top1': ('semantic-top1', 'semantic', 'top1'),
    'weighted': ('semantic-weighted', 'semantic', 'weighted'),
    'attention': ('semantic-attention', 'semantic', 'weighted'),
    'spatial': ('native-spatial-attention', 'native-spatial', 'weighted'),
}


def validate_recipe(cfg):
    """Reject accidental capacity, resume, input, batch or ablation changes."""
    experiment = cfg['experiment']
    arm = experiment['readout_arm']
    spec = cfg['trainer']['args']
    source = spec['deform_conditioner']
    if arm not in TRAINERS or cfg['trainer']['name'] != TRAINERS[arm]:
        raise ValueError('explicit observed-query readout trainer required')
    if (source['ray_mode'], source['correspondence_mode'], source['candidate_aggregation']) != MODES[arm]:
        raise ValueError('readout arm and correspondence recipe disagree')
    if (cfg.get('training_seed') not in (101, 202)
            or cfg.get('auto_retry') != 0 or cfg.get('load_ckpt') is not None
            or cfg.get('load_dir') != '' or cfg.get('ckpt') != 'none'
            or cfg.get('use_wandb') is not True
            or cfg.get('wandb_entity') != 'youngwoojeon-unist'
            or cfg.get('wandb_project') != 'pixal3d-deformnext-retargeting'
            or not cfg.get('wandb_id')
            or cfg.get('num_gpus') != 1 or cfg.get('num_nodes') != 1
            or cfg.get('tryrun') is not False or cfg.get('profile') is not False
            or cfg['dataset']['name'] != 'DeformNeXtSS64'
            or cfg['dataset']['args'].get('expected_split') != 'train'
            or spec['batch_size_per_gpu'] != 96 or spec['batch_split'] != 1
            or spec.get('max_steps') != 480 or spec.get('prefetch_data') is not False
            or spec.get('mix_precision_mode') != 'amp'
            or spec.get('mix_precision_dtype') != 'bfloat16'
            or spec.get('ema_rate') != [] or spec.get('microbatch_to_device') is not True
            or spec.get('lora') != {'rank': 16, 'alpha': 16, 'dropout': 0.0}
            or set(spec.get('continuation_ckpt', {})) != {'denoiser'}
            or spec.get('optimizer', {}).get('name') != 'AdamW'
            or spec['optimizer']['args']['lr'] != 1e-5
            or spec.get('model_learning_rates') != {'deform_conditioner': 1e-4}
            or spec.get('lr_scheduler') != {'name': 'LinearWarmupLRScheduler', 'args': {'warmup_steps': 25}}
            or spec.get('p_uncond') != .1 or spec.get('i_save') != 96
            or source.get('freeze_q2c') is not True or source.get('topk') != 64
            or source.get('score_temperature') != .05
            or source.get('ray_chunk_size') != 64
            or experiment.get('capacity_probe') is not False
            or experiment.get('physical_deformation_query_channels') != 0
            or experiment.get('target_modality') != 'rgb'
            or experiment.get('validate_at_start') is not True
            or experiment.get('training_assets') != 9149
            or experiment.get('updates_per_epoch') != 96
            or experiment.get('global_batch') != 96
            or experiment.get('world_size') != 1
            or experiment.get('microbatch_per_gpu') != 96):
        raise ValueError('fresh native LoRA/global96/micro96/accum1/five-epoch SS recipe required')
    validation = cfg['validation']
    quick = validation['quick_metrics']
    if (validation.get('dataset_name') != 'DeformNeXtSS64'
            or validation['dataset_args'].get('expected_split') != 'validation'
            or validation.get('every_steps') != 96 or validation.get('max_assets') != 32
            or quick.get('enabled') is not True or quick.get('panel_size') != 32
            or quick.get('sampling_steps') != 12 or quick.get('seed') != 42):
        raise ValueError('paired initial and epoch validation32 voxel protocol required')
    return arm


def initial_carrier_release(report):
    """Only release our own SS arms after the actual paired initial diagnostic."""
    if report.get('status') != 'PASS_ACTUAL_TARGET_FIRST_INITIAL32_CARRIER_COMPARISON':
        raise ValueError('actual initial carrier comparison required')
    panels = report['panels']
    for role in ('target-carrier', 'native-four-view-V1'):
        if len(panels[role]['per_asset']) != 32:
            raise ValueError('complete paired validation32 evidence required')
    for left, right in zip(panels['target-carrier']['per_asset'], panels['native-four-view-V1']['per_asset']):
        for key in ('sample_id', 'seed', 'target_voxels_sha256'):
            if left[key] != right[key]:
                raise ValueError('initial carriers must use paired assets, noise seeds and GT')
    delta = (panels['target-carrier']['metrics']['voxel_iou']
             - panels['native-four-view-V1']['metrics']['voxel_iou'])
    import math
    if not math.isfinite(delta) or delta != report['target_minus_native_V1_micro_IoU']:
        raise ValueError('initial carrier difference is inconsistent')
    return dict(status='PASS' if delta >= 0 else 'HOLD_INITIAL_SS_CARRIER_REGRESSION',
                target_minus_native_V1_micro_IoU=delta,
                target_carrier_is_native_V1=False,
                scope='Own SS readout training only; independent PBR science is unaffected',
                quality_gain_proven=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg_dict = json.loads(args.config.read_text())
    arm = validate_recipe(cfg_dict)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    allowed = os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(',')
    if visible not in {'0', '1', '2', '3'} or visible not in allowed:
        raise ValueError('one exclusively assigned physical GPU0–3 required')

    from easydict import EasyDict
    import torch
    from pixal3d import trainers
    from pixal3d.trainers.flow_matching.target_first_observed_query import (
        ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer,
    )
    from pixal3d.trainers.flow_matching.target_first_observed_weighted import (
        ObservedQueryWeightedTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer,
    )
    from pixal3d.trainers.flow_matching.target_first_observed_spatial import (
        ObservedQuerySpatialTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer,
    )
    from experiments.multiview_finetune.runtime import install_hooks
    import train as native

    choices = {cls.__name__: cls for cls in (
        ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer,
        ObservedQueryWeightedTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer,
        ObservedQuerySpatialTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer,
    )}
    selected = choices[cfg_dict['trainer']['name']]

    class Scientific(selected):
        def run(self):
            nn_parameters = [name for name, value in self.models['denoiser'].named_parameters() if value.requires_grad]
            source_count = sum(value.requires_grad for value in self.deform_conditioner.parameters())
            expected_source = 15 if arm in {'attention', 'spatial'} else 10
            if ((self.world_size, self.batch_size, self.batch_size_per_gpu, self.batch_split, self.step) != (1, 96, 96, 1, 0)
                    or len(nn_parameters) != 360 or any('lora_' not in name for name in nn_parameters)
                    or source_count != expected_source or len(self.optimizer.state) != 0):
                raise ValueError('fresh native LoRA and source-module optimizer required')
            report = dict(status='PASS_ACTUAL_FRESH_OBSERVED_QUERY_SS_STARTUP',
                          arm=arm, physical_GPU=visible, seed=cfg_dict['training_seed'],
                          world_size=1, global_batch=96, microbatch=96, accumulation=1,
                          actual_NN_LoRA_tensors=len(nn_parameters), actual_source_tensors=source_count,
                          actual_initial_optimizer_states=len(self.optimizer.state),
                          initial_function='Target carrier with zero source corrections',
                          native_four_view_V1_identity=False,
                          native_optimization_and_validation_lifecycle=True,
                          source_utility_or_final_RGB_preservation_not_proven=True)
            (Path(self.output_dir) / 'startup.json').write_text(json.dumps(report, indent=2) + '\n')
            return super().run()

    torch.cuda.set_device(0)
    install_hooks()
    torch.set_num_threads(2)
    setattr(trainers, cfg_dict['trainer']['name'], Scientific)
    output = Path(cfg_dict['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(cfg_dict, indent=2) + '\n')
    native.main(0, EasyDict(cfg_dict))


if __name__ == '__main__':
    main()
