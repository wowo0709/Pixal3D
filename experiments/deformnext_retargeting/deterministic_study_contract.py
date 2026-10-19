"""Strict RGB/noise/GT and generated-count checks for versioned SS final panels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .summarize_ss_module_study import audit_final_replay


def validate_evaluation(record, *, assets, step, mode='edited-rgb', value_control='normal'):
    if value_control not in {'normal', 'wrong-address'}:
        raise ValueError('explicit supported value intervention required')
    if value_control != 'normal' and mode != 'edited-rgb':
        raise ValueError('value intervention retains edited RGB')
    expected = dict(status='PASS', stage='ss64', step=step, weights='raw',
        sampling_steps=12, guidance=3., generation_batch_size=8,
        optimizer_updates=0, optimizer_updates_during_evaluation=0,
        target_modality='rgb', conditioning_mode=mode, input_view_count=4,
        input_camera_ids=[8,9,10,11], diagnostic_only=value_control!='normal',
        transport_control='normal', r9_path_control='normal', task_value_control=value_control)
    if mode not in {'edited-rgb', 'source-only'}:
        raise ValueError('explicit four-RGB input protocol required')
    expected['input_states'] = ['M0']*4 if mode=='source-only' else ['M1','M0','M0','M0']
    if any(record.get(key)!=value for key,value in expected.items()):
        raise ValueError('actual SS evaluation protocol differs')
    rows = record['per_asset']
    if len(rows)!=assets or len({row['sample_id'] for row in rows})!=assets:
        raise ValueError('distinct fixed-panel assets required')
    for row in rows:
        if (not row.get('source_cluster_id') or not row.get('target_voxels_sha256')
                or row.get('target_voxel_shape')!=[64,64,64]):
            raise ValueError('actual GT/source-cluster identity missing')
        if any(not isinstance(row[key],int) or isinstance(row[key],bool) or row[key]<0
               for key in ['seed','tp','fp','fn']):
            raise ValueError('nonnegative integer noise/counts required')
    numeric = record.get('deterministic_evaluation', {})
    if (numeric.get('protocol')!='ss-deterministic-v1'
            or numeric.get('actual_deterministic_algorithms') is not True
            or numeric.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8'
            or numeric.get('model_data_trainer_sources_modified') is not False
            or numeric.get('optimizer_updates')!=0):
        raise ValueError('actual deterministic evaluator provenance missing')
    for key in ['entry_source_SHA256','adapter_SHA256']:
        value = numeric.get(key, '')
        if len(value)!=64 or any(char not in '0123456789abcdef' for char in value):
            raise ValueError('pinned evaluator/adapter source hash required')


def verify_replay(small, full, *, step=480, mode='edited-rgb'):
    validate_evaluation(small,assets=32,step=step,mode=mode)
    validate_evaluation(full,assets=128,step=step,mode=mode)
    for key in ['checkpoint','conditioner_checkpoint','arm_config','manifest_sha256','arm']:
        if small.get(key)!=full.get(key):
            raise ValueError('replay model/data/config identity differs: '+key)
    for key in ['entry_source_SHA256','adapter_SHA256']:
        if small['deterministic_evaluation'][key]!=full['deterministic_evaluation'][key]:
            raise ValueError('replay evaluator source differs: '+key)
    # This calls the original exact TP/FP/FN/GT/noise check. Its reference is
    # an explicitly separate deterministic saved480 replay, not a relabeled
    # legacy training validation. Legacy snapshots remain untouched.
    audit_final_replay({'validation':[{'quick_per_asset':small['per_asset']}]},full)
    return dict(status='PASS_DETERMINISTIC32_TO_128_EXACT_REPLAY',
        fixed_step=step, checked_assets=32, final_assets=128,
        exact_counts_GT_noise_and_model_identity=True,
        original_exact_replay_assertions_unchanged=True,
        legacy_training_epoch_relabelled=False,
        scope='Independent deterministic saved-checkpoint32 versus final128/noise0;does not assert bit-exact replay of legacy nondeterministic epoch validation.')


def legacy_epoch_comparison(legacy, deterministic):
    reference = {row['sample_id']:row for row in legacy['quick_per_asset']}
    differences = []
    for row in deterministic['per_asset']:
        before = reference[row['sample_id']]
        for key in ['seed','source_cluster_id','target_voxels_sha256','target_voxel_shape']:
            if before[key]!=row[key]:
                raise ValueError('legacy input/GT/noise identity differs: '+key)
        if any(before[key]!=row[key] for key in ['tp','fp','fn']):
            differences.append(dict(sample_id=row['sample_id'],
                legacy_counts={key:before[key] for key in ['tp','fp','fn']},
                deterministic_counts={key:row[key] for key in ['tp','fp','fn']}))
    return dict(GT_source_and_noise_exact=True, generated_counts_exact=not differences,
        different_generated_cases=len(differences), differences=differences,
        scope='Descriptive old-versus-new numerical-mode comparison;no legacy failed exact-replay gate is changed to PASS.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--small',type=Path,required=True)
    parser.add_argument('--full',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--legacy-validation',type=Path)
    args=parser.parse_args()
    small,full=[json.loads(path.read_text()) for path in [args.small,args.full]]
    report=verify_replay(small,full)
    if args.legacy_validation:
        rows=[json.loads(line) for line in args.legacy_validation.read_text().splitlines()]
        legacy=next(row for row in rows if row['step']==480)
        report['legacy_epoch_comparison']=legacy_epoch_comparison(legacy,small)
    paths=[args.small,args.full]+([args.legacy_validation] if args.legacy_validation else [])
    report['input_SHA256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
