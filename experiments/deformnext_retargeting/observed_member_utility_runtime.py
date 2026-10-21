"""Bind one declared utility-query design to the unchanged NoDQ native runtime."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

from .observed_member_utility import observed_member_utility


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--utility-query-design', choices=['observed-input-target'], required=True)
    parser.add_argument('--native-entry', choices=['member_feature_runtime', 'native_source_seed_runtime'],
                        default='member_feature_runtime')
    local, remaining = parser.parse_known_args()
    if local.native_entry == 'native_source_seed_runtime':
        if '--family' not in remaining or remaining[remaining.index('--family')+1] != 'member-feature':
            raise ValueError('explicit NoDQ member-feature seed runtime required')
    config_flag = '--config' if '--config' in remaining else '--arm-config'
    config_path = Path(remaining[remaining.index(config_flag)+1])
    config = json.loads(config_path.read_text())
    if config['trainer']['name'] != 'MemberFeatureSourceRayImageConditionedProjFlowMatchingCFGTrainer':
        raise ValueError('original NoDQ trainer and identical parameter layout required')
    paths = [Path(__file__), Path(__file__).with_name('observed_member_utility.py'), config_path]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    native = importlib.import_module('.'+local.native_entry, __package__)
    argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining]
        with observed_member_utility() as statistics:
            native.main()
    finally:
        sys.argv = argv
    if (not statistics['forwards'] or statistics['forwards'] != statistics['utility_pool_calls']
            or statistics['forwards'] != statistics['target_pool_calls']
            or any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest for p, digest in hashes.items())):
        raise ValueError('complete unchanged original runtime and both pool calls required')
    phase = remaining[remaining.index('--phase')+1]
    output = (Path(remaining[remaining.index('--output')+1]) if phase in {'native', 'evaluation'}
              else Path(config['output_dir']))
    report = dict(status='COMPLETED_OBSERVED_MEMBER_UTILITY_RUNTIME', phase=phase,
        utility_query_design=local.utility_query_design, original_native_runtime=local.native_entry,
        source_SHA256=hashes, statistics=statistics, physical_module_or_parameter_layout_changed=False,
        observed_input_only_changes_utility_not_R_context_or_ray_write=True,
        native_identity_capacity_and_quality_require_their_separate_actual_reports=True,
        no_historical_target_patch_query_or_R96_reproduction_claim=True, model_adopted=False, full_goal_complete=False)
    (output/'observed-member-utility-runtime.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
