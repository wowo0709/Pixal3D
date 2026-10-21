"""Archive exact32 SS outputs using the bounded-prior checkpoint constructor.

The existing native subset producer still verifies every decoded count and
target/noise identity against the model's final128 reference. This wrapper
only binds the declared conditioner and its bounded-probability V verifier.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from pixal3d.modules.bounded_member_feature_source_ray import BoundedPriorMemberFeatureSourceRayConditioner
from .bounded_member_feature_value_control import feature_value_control


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--bounded-prior-logit-bound', type=float, required=True)
    local, remaining = parser.parse_known_args()
    if (local.bounded_prior_logit_bound != 1. or '--evaluation-module' not in remaining
            or remaining[remaining.index('--evaluation-module')+1] != 'member_feature_runtime'):
        raise ValueError('declared unit-bound NoDQ native subset producer required')
    config_path = Path(remaining[remaining.index('--arm-config')+1])
    config = json.loads(config_path.read_text())
    if (config['trainer']['name'] != 'MemberFeatureSourceRayImageConditionedProjFlowMatchingCFGTrainer'
            or config['experiment'].get('attention_parameterization') != 'bounded-matching-prior-functional-residual-v1'
            or config['experiment'].get('residual_logit_bound') != 1.):
        raise ValueError('bounded-prior scientific recipe and reference required')
    from pixal3d.trainers.flow_matching import member_feature_source_ray as trainer_module
    from . import member_feature_value_control as value_module
    from . import evaluate_member_feature_ss_artifact_subset as producer
    old_class, old_control = trainer_module.MemberFeatureSourceRayConditioner, value_module.feature_value_control
    captured = []

    class Configured(BoundedPriorMemberFeatureSourceRayConditioner):
        def __init__(self, **kwargs):
            super().__init__(residual_logit_bound=1., **kwargs)
            captured.append(self)

    files = [Path(__file__), Path(producer.__file__), config_path,
             Path(trainer_module.__file__).parents[2]/'modules/bounded_member_feature_source_ray.py',
             Path(__file__).with_name('bounded_member_feature_value_control.py')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    argv = sys.argv
    try:
        trainer_module.MemberFeatureSourceRayConditioner = Configured
        value_module.feature_value_control = feature_value_control
        sys.argv = [sys.argv[0], *remaining]
        producer.main()
    finally:
        sys.argv = argv
        trainer_module.MemberFeatureSourceRayConditioner = old_class
        value_module.feature_value_control = old_control
    if (not captured or any(not m.ray_readout.executed_forwards for m in captured)
            or any(sum(p.requires_grad for p in m.parameters()) != 19 for m in captured)
            or any(float(m.ray_readout.residual_logit_bound) != 1. for m in captured)
            or any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest for p,digest in hashes.items())):
        raise ValueError('actual unchanged bounded-prior model and producer required')
    output = Path(remaining[remaining.index('--output')+1]);path = output/'evaluation.json'
    result = json.loads(path.read_text())
    if (result['step'] != 480 or result['optimizer_updates_during_evaluation'] != 0
            or result['native_SS_artifacts']['archived_asset_count'] != 32):
        raise ValueError('actual final480 native32 prediction archive required')
    report = dict(status='PASS_ACTUAL_BOUNDED_PRIOR_NATIVE_SS32_ARCHIVE_BINDING',
        attention_parameterization=config['experiment']['attention_parameterization'],
        residual_logit_bound=1., source_SHA256=hashes,
        original_native_subset_producer_and_128_replay_checks_preserved=True,
        actual19_source_tensors_and_serialized_reference_buffers=True,
        optimizer_updates=0, model_adopted=False, full_goal_complete=False)
    result['bounded_prior_model_binding'] = report
    path.write_text(json.dumps(result,indent=2)+'\n')
    (output/'bounded-prior-archive-binding.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
