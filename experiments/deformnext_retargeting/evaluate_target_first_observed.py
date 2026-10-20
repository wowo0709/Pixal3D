"""Pinned saved-weight SS evaluation of the four same-carrier source readouts.

Reuse the existing deterministic evaluator and its native sampling/voxel
metrics. The optional intervention changes only values and holds the original
matching probabilities and source-use decision fixed in the same forward.
It is a saved-weight diagnostic, never a retrained correspondence ablation.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import runpy
import sys

import torch

from .train_target_first_observed import validate_recipe


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--expected-assets', type=int, required=True)
    parser.add_argument('--source-value-control', choices=['normal', 'wrong-address-fixed-use'], default='normal')
    local, remaining = parser.parse_known_args()
    config = Path(remaining[remaining.index('--arm-config') + 1])
    cfg = json.loads(config.read_text())
    arm = validate_recipe(cfg)
    output = Path(remaining[remaining.index('--output') + 1])
    if output.exists():
        raise FileExistsError('fresh evaluation namespace required')
    if (remaining[remaining.index('--stage') + 1] != 'ss64'
            or any(option in remaining for option in ('--transport-control', '--task-value-control', '--r9-path-control', '--conditioning-mode'))):
        raise ValueError('native edited-RGB SS evaluation without combined interventions required')

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
    from .target_first_value_control import fixed_use_value_control

    choices = {cls.__name__: cls for cls in (
        ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer,
        ObservedQueryWeightedTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer,
        ObservedQuerySpatialTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer,
    )}
    selected = choices[cfg['trainer']['name']]
    captured, interventions, diagnostics = [], [], []

    class Evaluation(selected):
        def init_models_and_more(self, **kwargs):
            super().init_models_and_more(**kwargs)
            captured.append(self)

        def _encode_deform_views(self, *args, **kwargs):
            context = (fixed_use_value_control(self.deform_conditioner, interventions)
                       if local.source_value_control != 'normal' else contextlib.nullcontext())
            with context:
                result = super()._encode_deform_views(*args, **kwargs)
            last = getattr(self, '_last_deform_condition', None)
            if last is not None:
                diagnostics.append(dict(
                    fusion_gate=float(torch.tanh(self.deform_conditioner.fusion_gate.detach())),
                    query_valid_fraction=float(last.query_valid.float().mean()),
                    source_use_probability_mean=float(self._last_source_use_probability.float().mean()),
                    retarget_residual_RMS=float(last.retarget_residual.detach().float().square().mean().sqrt())))
            return result

    entry = Path(__file__).with_name('evaluate_deterministic_checkpoint.py')
    bound = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__), config, entry]}
    name = cfg['trainer']['name']
    prior = trainers.__dict__.get(name)
    setattr(trainers, name, Evaluation)
    previous = sys.argv
    try:
        sys.argv = [str(entry), *remaining]
        runpy.run_path(str(entry), run_name='__main__')
    finally:
        sys.argv = previous
        if prior is None:
            delattr(trainers, name)
        else:
            setattr(trainers, name, prior)
    path = output / 'evaluation.json'
    result = json.loads(path.read_text())
    if (len(captured) != 1 or len(result['per_asset']) != local.expected_assets
            or result['optimizer_updates_during_evaluation'] != 0
            or any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in bound.items())):
        raise ValueError('complete pinned read-only SS evaluation required')
    trainer = captured[0]
    if len(trainer.optimizer.state) != 0:
        raise ValueError('read-only evaluation cannot update native optimizer state')
    controlled = local.source_value_control != 'normal'
    if controlled and (not interventions or any(
            not row['source_use_probability_exact_same_forward']
            or not row['original_matching_logits_and_attention_weights_exact']
            for row in interventions)):
        raise ValueError('actual matching and source-use invariance evidence required')
    result['observed_query_target_first'] = dict(
        arm=arm, source_value_control=local.source_value_control,
        physical_deformation_query_channels=0, native_target_global_and_negative_condition=True,
        native_source_ray_projection=True, per_voxel_visibility_write_gate=False,
        source_registration='explicit process-local; frozen lazy registry unchanged',
        actual_optimizer_states=len(trainer.optimizer.state), source_SHA256=bound,
        actual_source_diagnostics=diagnostics, value_intervention_events=interventions,
        saved_weight_diagnostic_not_retrained_ablation=controlled,
        original_matching_score_weights_and_use_retained=controlled,
        original_correspondence_candidate_count=3 if arm == 'spatial' else 64,
        native_spatial_is_natural_three_view_not_fixed_K_contrast=arm == 'spatial',
        target_carrier_is_native_four_view_V1=False,
        no_final_RGB_preservation_or_independent_seed_gain_claim=True)
    result['diagnostic_only'] = result['diagnostic_only'] or controlled
    path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
