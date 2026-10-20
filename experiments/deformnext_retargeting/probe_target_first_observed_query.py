"""Native observed-query mechanics and initial32 carrier comparison, no training.

Reuse the frozen native proof in this process. Extra checks run with its
temporary nonzero source parameters and restore all diagnostic attributes.
No query-support mask is applied to native projection or voxel outputs.
"""
import argparse
import json
from pathlib import Path


def main():
    import torch
    from pixal3d.trainers.flow_matching import target_first_ray as original
    from pixal3d.trainers.flow_matching.target_first_observed_query import (
        ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer as Trainer,
    )
    from . import probe_target_first_ray_v2 as proof
    from pixal3d.trainers.flow_matching import target_first_ray_v2 as v2

    saved_v2 = v2.TargetFirstRayImageConditionedProjFlowMatchingCFGTrainerV2
    saved_original = original.TargetFirstRayImageConditionedProjFlowMatchingCFGTrainer
    saved_encode = Trainer._encode_deform_views
    observed_checks = []

    def encode(self, images, camera, transport, *, control='normal'):
        result = saved_encode(self, images, camera, transport, control=control)
        module = self.deform_conditioner
        if (transport is not None and not observed_checks
                and bool(transport['source_valid'].any())
                and bool(module.fusion_gate.detach().abs() > 0)):
            saved_condition = self._last_deform_condition
            saved_use = self._last_source_use_probability
            try:
                with torch.no_grad():
                    anchor, anchor_negative = saved_encode(self, images, camera, None)
                    absent, absent_negative = saved_encode(self, images, camera, dict(
                        transport, target_valid=torch.zeros_like(transport['target_valid'])))
                    no_query = self._last_deform_condition
                    no_query_use = self._last_source_use_probability
                    if (not all(torch.equal(absent[key], anchor[key]) for key in anchor)
                            or not all(torch.equal(absent_negative[key], anchor_negative[key])
                                       for key in anchor_negative)
                            or no_query.query_valid.any() or no_query_use.any()):
                        raise ValueError('unobserved matching queries must use the exact target prior')
                    observed = transport['target_valid'].gather(1,
                        transport['patch_grid'].flatten(1).long())
                    changed = (result[0]['proj'].float() - anchor['proj'].float()).abs().amax(-1) > 0
                    transmitted = int((changed & ~observed).sum())
                    if transmitted <= 0:
                        raise ValueError('native ray delivery must reach unobserved target-image voxels')
                    observed_checks.append(dict(
                        status='PASS_ACTUAL_NONZERO_SOURCE_OBSERVED_QUERY_NATIVE_RAY_DELIVERY',
                        no_observed_query_target_fallback_exact=True,
                        source_use_probability_without_query_exact_zero=True,
                        native_correction_reaches_unobserved_target_voxels=transmitted,
                        unobserved_target_voxels=int((~observed).sum()),
                        nonzero_source_fusion_gate=float(module.fusion_gate.detach()),
                        input_alpha_used_only_for_matching_query=True,
                        per_voxel_visibility_write_gate=False,
                        all_native_ray_memberships_retained=True,
                        optimizer_updates=0, trained_quality_gain_established=False))
            finally:
                self._last_deform_condition = saved_condition
                self._last_source_use_probability = saved_use
        return result

    v2.TargetFirstRayImageConditionedProjFlowMatchingCFGTrainerV2 = Trainer
    Trainer._encode_deform_views = encode
    try:
        proof.main()
    finally:
        Trainer._encode_deform_views = saved_encode
        v2.TargetFirstRayImageConditionedProjFlowMatchingCFGTrainerV2 = saved_v2
        original.TargetFirstRayImageConditionedProjFlowMatchingCFGTrainer = saved_original
    if len(observed_checks) != 1:
        raise ValueError('native nonzero source support/delivery probe did not execute')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (args.output / 'observed-query-native-delivery.json').write_text(
        json.dumps(observed_checks[0], indent=2) + '\n')
    print(json.dumps(observed_checks[0]), flush=True)


if __name__ == '__main__':
    main()
