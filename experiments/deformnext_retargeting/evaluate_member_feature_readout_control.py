"""Native deterministic SS evaluation with a frozen readout intervention."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from .member_feature_readout_control import MODES, feature_readout_control


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--feature-readout-control', choices=MODES, required=True)
    local, remaining = parser.parse_known_args()
    if '--source-value-control' in remaining and remaining[remaining.index('--source-value-control')+1] != 'normal':
        raise ValueError('uncombined frozen readout diagnostic required')
    from pixal3d.trainers.flow_matching.member_feature_source_ray import MemberFeatureSourceRayImageConditionedProjFlowMatchingCFGTrainer as selected
    from . import evaluate_member_feature_source_ray as native

    events, captured = [], []
    original = selected._encode_deform_views
    previous = selected.__dict__.get('_encode_deform_views')
    paths = [Path(__file__), Path(__file__).with_name('member_feature_readout_control.py'),
             Path(__file__).with_name('member_feature_query_support.py'), Path(native.__file__)]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    def encode(self, *args, **kwargs):
        if not captured:
            captured.append(self)
        elif captured[0] is not self:
            raise ValueError('one native evaluation trainer required')
        with feature_readout_control(self.deform_conditioner, events, mode=local.feature_readout_control):
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
            or any(not row['original_candidates_Top1_R_target_and_source_use_exact_same_forward'] for row in events)):
        raise ValueError('complete pinned frozen readout intervention required')
    output = Path(remaining[remaining.index('--output')+1])/'evaluation.json'
    result = json.loads(output.read_text())
    if result['step'] != 480 or result['optimizer_updates_during_evaluation'] != 0:
        raise ValueError('final480 frozen weights with zero optimizer updates required')
    result['feature_readout_control'] = dict(mode=local.feature_readout_control, events=events,
        source_SHA256=hashes, actual_optimizer_states=0, training_updates=0,
        candidate_support_decoder_Top1_R_target_and_source_use_unchanged=True,
        readout_prior_removal_is_not_the_users_Top1_NoScore_arm=True,
        no_retrained_NoCorr_decoder_effect_or_independent_training_quality_claim=True)
    result['diagnostic_only'] = True
    output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
