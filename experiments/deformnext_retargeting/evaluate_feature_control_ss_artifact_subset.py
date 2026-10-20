"""Exact32-prefix SS archives for physical Top1 and native spatial controls."""
import argparse
import json
from pathlib import Path
import sys

from .evaluate_ss_artifact_subset import capture_ss_artifacts, validate_subset_reference, sha
from .evaluate_member_feature_ss_artifact_subset import validate_archive_arguments
from .member_feature_control_recipe import NAME, SCHEMA, control_spec, validate_control_recipe
from .ss_pooled_support_metrics import artifact_support_metrics


def validate_control_replay(actual, reference, captured, spec):
    validate_subset_reference(actual, reference, captured)
    if (len(reference['per_asset']) != 128
            or len({row['sample_id'] for row in reference['per_asset']}) != 128
            or len({row['sample_id'] for row in actual['per_asset']}) != 32):
        raise ValueError('unique32 prefix of original unique128 control evaluation required')
    for result in [actual, reference]:
        source = result['native_prior_source_ray']
        if (result['step'] != 480 or result['diagnostic_only']
                or result['conditioning_mode'] != 'edited-rgb' or result['target_modality'] != 'rgb'
                or result['input_states'] != ['M1', 'M0', 'M0', 'M0']
                or result['input_camera_ids'] != [8, 9, 10, 11]
                or result['generation_batch_size'] != 8 or result['sampling_steps'] != 12 or result['guidance'] != 3.
                or source['arm'] != spec['arm'] or source['feature_control'] != spec['control']
                or source['source_value_control'] != 'normal'
                or source['physical_deformation_query_channels'] != 0
                or source['physical_retargeter_input_channels'] != 2048
                or source['actual_feature_head_tensors'] != spec['head_tensors']
                or source['QK_score_parameters_present'] != (spec['head_tensors'] == 9)
                or source['original_correspondence_candidate_count'] != spec['candidates']
                or source['actual_optimizer_states'] != 0
                or not source['target_carrier_is_native_four_view_V1']):
            raise ValueError('actual physical Top1/spatial control identity or native archive protocol differs')


def validate_runtime(marker, spec):
    if (marker['status'] != 'PASS_COMPLETED_MEMBER_FEATURE_CONTROL_RUNTIME_PHASE'
            or marker['phase'] != 'evaluation' or marker['trainer'] != NAME or marker['schema'] != SCHEMA
            or marker['arm'] != spec['arm'] or marker['physical_control'] != spec['control']
            or marker['actual_feature_head_tensors'] != spec['head_tensors']
            or marker['actual_source_tensors'] != spec['source_tensors']
            or marker['actual_correspondence_candidates'] != spec['candidates']
            or marker['physical_deformation_query_channels'] != 0):
        raise ValueError('completed actual physical Top1/spatial evaluation runtime required')


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--evaluation-module', choices=['member_feature_control_runtime'], required=True)
    parser.add_argument('--reference-evaluation', type=Path, required=True)
    local, remaining = parser.parse_known_args()
    validate_archive_arguments(remaining)
    config = Path(remaining[remaining.index('--arm-config') + 1]); cfg = json.loads(config.read_text())
    validate_control_recipe(cfg); spec = control_spec(cfg)
    from . import quick_metrics as quick
    from . import member_feature_control_runtime as entry
    from . import ss_pooled_support_metrics as support
    output = Path(remaining[remaining.index('--output') + 1])
    if output.exists():
        raise FileExistsError('fresh physical control prediction archive namespace required')
    reference = json.loads(local.reference_evaluation.read_text())
    bound = {str(Path(path).resolve()): sha(path) for path in [__file__, quick.__file__, entry.__file__,
             support.__file__, config, local.reference_evaluation]}
    captured = []; previous = sys.argv
    try:
        with capture_ss_artifacts(quick, output, captured):
            sys.argv = [entry.__file__, *remaining]; entry.main()
    finally:
        sys.argv = previous
    path = output / 'evaluation.json'; actual = json.loads(path.read_text())
    validate_control_replay(actual, reference, captured, spec)
    marker = json.loads((output / 'member-feature-control-runtime.json').read_text())
    validate_runtime(marker, spec)
    if any(sha(path) != digest for path, digest in bound.items()):
        raise ValueError('control archive runtime, recipe or exact reference changed')
    for artifact in captured:
        artifact['native_SS32_support'] = artifact_support_metrics(output / artifact['artifact'])
    actual['native_SS_artifacts'] = dict(status='PASS_ACTUAL_NATIVE_SS_ARCHIVE_AND_EXACT_REFERENCE_SUBSET_REPLAY',
        original_prefix32_all_records_and_occupancy_counts_exact=True, artifacts=captured,
        reference_evaluation=str(local.reference_evaluation), reference_SHA256=sha(local.reference_evaluation),
        source_SHA256=bound, actual_evaluation_runtime=marker, reference_asset_count=128, archived_asset_count=32,
        reference_full_panel_quick_metrics_not_compared_to_subset=True,
        original_reference_had_no_prediction_grid_hash_so_not_claimed=True,
        predicted_SS_is_available_for_full_cascade=True, GT_SS_is_for_scoring_only=True,
        decoded_GT_and_prediction_support32_metrics_recorded=True,
        full_cascade_generated=False, optimizer_updates=0)
    path.write_text(json.dumps(actual, indent=2) + '\n')
    print(json.dumps(dict(status=actual['native_SS_artifacts']['status'], assets=32,
                         physical_control=spec['control'], source_tensors=spec['source_tensors'])), flush=True)


if __name__ == '__main__':
    main()
