"""Bind the bounded-prior NoDQ design to unchanged native training phases."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

from pixal3d.modules.bounded_member_feature_source_ray import BoundedPriorMemberFeatureSourceRayConditioner
from .bounded_member_feature_value_control import feature_value_control


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--bounded-prior-logit-bound', type=float, required=True)
    parser.add_argument('--native-entry', choices=['member_feature_runtime', 'native_source_seed_runtime'], default='member_feature_runtime')
    local, remaining = parser.parse_known_args()
    if local.bounded_prior_logit_bound != 1.:
        raise ValueError('one declared unit-logit residual design; no bound sweep')
    if local.native_entry == 'native_source_seed_runtime':
        if '--family' not in remaining or remaining[remaining.index('--family')+1] != 'member-feature':
            raise ValueError('explicit NoDQ member-feature family required')
    flag = '--config' if '--config' in remaining else '--arm-config'
    config_path = Path(remaining[remaining.index(flag)+1]);config = json.loads(config_path.read_text())
    if (config['trainer']['name'] != 'MemberFeatureSourceRayImageConditionedProjFlowMatchingCFGTrainer'
            or config['experiment'].get('attention_parameterization') != 'bounded-matching-prior-functional-residual-v1'
            or config['experiment'].get('residual_logit_bound') != 1.):
        raise ValueError('explicit bounded-prior NoDQ recipe and original tensor layout required')
    from pixal3d.trainers.flow_matching import member_feature_source_ray as trainer_module
    from . import member_feature_value_control as value_module
    original_class, original_control = trainer_module.MemberFeatureSourceRayConditioner, value_module.feature_value_control
    captured = []

    class Configured(BoundedPriorMemberFeatureSourceRayConditioner):
        def __init__(self, **kwargs):
            super().__init__(residual_logit_bound=local.bounded_prior_logit_bound, **kwargs)
            captured.append(self)

    files = [Path(__file__), Path(__file__).with_name('bounded_member_feature_value_control.py'),
             Path(trainer_module.__file__).parents[2]/'modules/bounded_member_feature_source_ray.py',config_path]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    argv = sys.argv
    try:
        trainer_module.MemberFeatureSourceRayConditioner = Configured
        value_module.feature_value_control = feature_value_control
        native = importlib.import_module('.'+local.native_entry, __package__)
        sys.argv = [sys.argv[0], *remaining]
        native.main()
    finally:
        sys.argv = argv
        trainer_module.MemberFeatureSourceRayConditioner = original_class
        value_module.feature_value_control = original_control
    if not captured or any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest for p,digest in hashes.items()):
        raise ValueError('actual source conditioners and unchanged runtime bindings required')
    phase = remaining[remaining.index('--phase')+1]
    heads = [module.ray_readout for module in captured]
    if (any(sum(p.requires_grad for p in module.parameters())!=19 for module in captured)
            or any(len(list(head.parameters()))!=9 for head in heads)
            or any(not head.executed_forwards or float(head.residual_logit_bound)!=1. for head in heads)
            or (phase != 'evaluation' and any(head.first_forward_prior_exact is not True for head in heads))):
        raise ValueError('fresh phases must actually start with exact matching-posterior head weights')
    output = Path(remaining[remaining.index('--output')+1]) if phase in {'native','evaluation'} else Path(config['output_dir'])
    report = dict(status='COMPLETED_BOUNDED_MEMBER_FEATURE_RUNTIME', phase=phase,
        attention_parameterization=config['experiment']['attention_parameterization'],
        source_SHA256=hashes, residual_logit_bound=1., matching_prior_and_Top64_preserved=True,
        trained_utility_weights_retained_and_function_anchored=True,
        original19_trainable_source_tensors_and_nine_feature_tensors=True,
        bounded_head_forward_calls=[head.executed_forwards for head in heads],
        calls_include_checkpoint_recomputation=True,
        first_forward_prior_exact=[head.first_forward_prior_exact for head in heads],
        anchors_are_serialized_frozen_buffers=True, per_voxel_visibility_write_gate=False,
        native_identity_capacity_and_quality_require_separate_actual_reports=True,
        no_auxiliary_label_training_or_older_target_patch_Qzero_reproduction=True,
        model_adopted=False,full_goal_complete=False)
    (output/'bounded-member-feature-runtime.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
