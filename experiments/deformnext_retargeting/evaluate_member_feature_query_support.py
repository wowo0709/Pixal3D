"""Keep the existing deterministic SS evaluator; bind a utility-query probe."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

from .member_feature_query_support import MODES, utility_query_support


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--utility-query-control', choices=MODES, required=True)
    local, remaining = parser.parse_known_args()
    if '--source-value-control' in remaining and remaining[remaining.index('--source-value-control')+1] != 'normal':
        raise ValueError('uncombined utility-query probe required')
    from pixal3d.trainers.flow_matching.member_feature_source_ray import MemberFeatureSourceRayImageConditionedProjFlowMatchingCFGTrainer as selected
    from . import evaluate_member_feature_source_ray as native

    events, captured = [], []
    original = selected._encode_deform_views
    previous = selected.__dict__.get('_encode_deform_views')
    paths = [Path(__file__), Path(__file__).with_name('member_feature_query_support.py'), Path(native.__file__)]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    def encode(self, *args, **kwargs):
        if not captured:
            captured.append(self)
        elif captured[0] is not self:
            raise ValueError('one native evaluation trainer required')
        with utility_query_support(self.deform_conditioner, events, mode=local.utility_query_control):
            return original(self, *args, **kwargs)

    argv = sys.argv
    try:
        selected._encode_deform_views = encode
        sys.argv = [sys.argv[0], *remaining]
        native.main()
    finally:
        sys.argv = argv
        if previous is None:
            delattr(selected, '_encode_deform_views')
        else:
            selected._encode_deform_views = previous
    if (not events or len(captured) != 1 or len(captured[0].optimizer.state)
            or any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in hashes.items())
            or any(not row['original_member_prior_exact_same_forward']
                   or not row['source_use_probability_exact_same_forward'] for row in events)):
        raise ValueError('complete pinned read-only query intervention required')
    output = Path(remaining[remaining.index('--output')+1])/'evaluation.json'
    result = json.loads(output.read_text())
    if result['step'] != 480 or result['optimizer_updates_during_evaluation'] != 0:
        raise ValueError('final480 frozen weights with zero optimizer updates required')
    result['utility_query_support'] = dict(
        mode=local.utility_query_control, events=events, source_SHA256=hashes,
        changes_utility_input_for_both_score_selector_and_four_head_Q=True,
        member_posterior_R_context_source_use_and_native_delivery_unchanged=True,
        input_alpha_support_not_GT_surface_visibility=True,
        projected_correction_still_reaches_target_unobserved_voxels=True,
        training_updates=0, actual_optimizer_states=0,
        saved_weight_diagnostic_not_historical_target_patch_reproduction=True,
        no_independent_training_seed_test508_or_quality_gain_claim=True)
    result['diagnostic_only'] = True
    output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
