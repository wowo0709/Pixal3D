"""Archive original zero-shot SS predictions, with exact128-reference replay."""
import argparse
import json
from pathlib import Path
import sys

from . import evaluate_ss_artifact_subset as archive
from .ss_pooled_support_metrics import artifact_support_metrics


def validate_arguments(arguments):
    for flag, expected in {'--evaluation-module': 'evaluate_deterministic_checkpoint',
                           '--stage': 'ss64', '--step': '0', '--panel-size': '32',
                           '--generation-batch': '8'}.items():
        if (arguments.count(flag) != 1 or arguments.index(flag) + 1 >= len(arguments)
                or arguments[arguments.index(flag) + 1] != expected):
            raise ValueError('fixed original zero-shot archive protocol required: ' + flag)
    flag = '--conditioning-mode'
    if (arguments.count(flag) != 1 or arguments.index(flag) + 1 >= len(arguments)
            or arguments[arguments.index(flag) + 1] not in {'edited-rgb', 'source-only'}):
        raise ValueError('one explicit original zero-shot input mode required')
    if '--conditioner-checkpoint' in arguments:
        raise ValueError('zero-shot archive cannot load a deformation conditioner')


def main():
    arguments = sys.argv[1:]
    validate_arguments(arguments)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--reference-evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    local, _ = parser.parse_known_args(arguments)
    reference = json.loads(local.reference_evaluation.read_text())
    if (len(reference['per_asset']) != 128
            or len({row['sample_id'] for row in reference['per_asset']}) != 128
            or reference['step'] != 0 or reference['conditioner_checkpoint'] is not None):
        raise ValueError('original unique128 zero-shot reference required')
    bound = {str(Path(path).resolve()): archive.sha(path) for path in [__file__, archive.__file__, local.reference_evaluation]}
    archive.main()
    path = local.output / 'evaluation.json'
    actual = json.loads(path.read_text())
    report = actual['native_SS_artifacts']
    if (actual['conditioning_mode'] != arguments[arguments.index('--conditioning-mode') + 1]
            or actual['step'] != 0 or actual['conditioner_checkpoint'] is not None
            or actual['diagnostic_only'] or len(report['artifacts']) != 32
            or actual['input_camera_ids'] != [8, 9, 10, 11]
            or actual['target_modality'] != 'rgb'
            or len({row['sample_id'] for row in actual['per_asset']}) != 32):
        raise ValueError('actual original zero-shot32 archive protocol differs')
    for row in report['artifacts']:
        row['native_SS32_support'] = artifact_support_metrics(local.output / row['artifact'])
    if any(archive.sha(path) != digest for path, digest in bound.items()):
        raise ValueError('zero-shot archive wrapper or source changed')
    report.update(decoded_GT_and_prediction_support32_metrics_recorded=True,
                  original_reference_had_no_prediction_grid_hash_so_not_claimed=True,
                  original_multiview_SS_step=4210, deformation_finetuning_updates=0,
                  zero_shot_wrapper_source_SHA256=bound)
    path.write_text(json.dumps(actual, indent=2) + '\n')


if __name__ == '__main__':
    main()
