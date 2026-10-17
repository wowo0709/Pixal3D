"""Require all24 replays and audit against original learned states/components."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .historical_q2c_statistics import aggregate_rows, contrast
from .reproduce_q2c import checked, read, reference, write_new


def summarize(root, historical, output):
    schedule = read(historical / 'exp4/_R9_SCHEDULE.json')
    frozen = read(historical / 'exp4/_R9_CHECKPOINT_FREEZE.json')
    by_tuple = {(r['tuple']['decoder_arm'], r['tuple']['fold'], r['tuple']['seed']): r
                for r in frozen['runs']}
    rows, audits, inputs = [], [], []
    for arm in ['D0-PAIRWISE', 'D1-Q2C']:
        for fold in range(4):
            for seed in [42017, 42018, 42019]:
                name = f'{arm}-fold{fold}-seed{seed}'
                path = root / name / 'summary.json'
                result = read(path)
                if (result['status'] != 'PASS' or result['completed_updates'] != 1200
                        or result['historical_loss_difference_max'] != 0
                        or not result['all_1200_sampled_pairs_and_query_hashes_match_historical']
                        or not result['evaluation_opened_after_fixed_final_freeze']):
                    raise ValueError('historical training was not exactly reproduced: ' + name)
                info = result['tuple']
                if (info['decoder_arm'], info['fold'], info['seed']) != (arm, fold, seed):
                    raise ValueError('reproduction tuple identity differs')
                original = by_tuple[arm, fold, seed]
                old = torch.load(checked(original['files']['checkpoint_final.pt']), map_location='cpu', weights_only=False)['model']
                current = torch.load(checked(result['checkpoint']), map_location='cpu', weights_only=True)['model']
                if set(old) != set(current) or any(not torch.equal(value, current[key]) for key, value in old.items()):
                    raise ValueError('historical learned model parameters differ: ' + name)
                old_path = historical / 'exp4/evaluation' / (original['run_id'] + '.json')
                old_rows = [r for r in read(old_path)['rows'] if r['split'] == 'DEV' and r['arm'] == arm]
                key = lambda r: (r['pair_id'], r['original_cue_id'], r['query_slot'])
                expected = {key(r): r for r in old_rows}
                actual = {key(r): r for r in result['component_rows']}
                if set(actual) != set(expected) or len(actual) != len(result['component_rows']):
                    raise ValueError('historical held-out cue/slot population differs: ' + name)
                error_fields = ['task_rgb_error', 'wrong_rgb_error', 'gt_rgb_error']
                max_error = max(abs(actual[k][field] - expected[k][field]) for k in actual for field in error_fields)
                if max_error != 0:
                    raise ValueError('historical held-out RGB error components differ: ' + name)
                rows.extend([{**r, 'backbone': info['backbone']} for r in result['component_rows']])
                audits.append({'arm': arm, 'fold': fold, 'seed': seed, 'optimizer_updates': 1200,
                    'all_sampled_pair_query_hashes_exact': True, 'all_training_losses_exact': True,
                    'all_learned_parameters_exact': True, 'held_out_RGB_components_max_abs': max_error,
                    'historical_evaluation': reference(old_path), 'replay': reference(path)})
                inputs.append(reference(path))
    aggregate = aggregate_rows(rows)
    if aggregate['pair_count'] != 63:
        raise ValueError('controlled-task OOF63 coverage differs')
    comparison = contrast(aggregate, 'D0-PAIRWISE', 'D1-Q2C')
    report = {'status': 'PASS_EXACT_HISTORICAL_REPRODUCTION', 'scope': 'controlled RGB task only',
              'training_tuples': 24, 'model_seeds': 3, 'OOF_folds': 4, 'pairs': 63,
              'all_28800_training_losses_exact': True, 'all_learned_states_and_held_out_components_exact': True,
              'statistics': aggregate, 'D1_Q2C_vs_D0_pairwise': comparison, 'tuple_audits': audits,
              'inputs': inputs, 'historical_schedule': reference(historical / 'exp4/_R9_SCHEDULE.json'),
              'current_DeformNeXt_SS_gain_claim': False, 'production_module_promotion': False}
    write_new(output, report)
    print(json.dumps({'status': report['status'], 'pairs': 63,
        'D0_RGB_L1': aggregate['arms']['D0-PAIRWISE']['task_rgb_error'],
        'D1_RGB_L1': aggregate['arms']['D1-Q2C']['task_rgb_error'], 'gain': comparison['mean'],
        'paired_CI95': comparison['ci95']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--historical', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    summarize(args.root, args.historical, args.output)


if __name__ == '__main__':
    main()
