"""Audit all144 archived G2 endpoint replays, including source-use controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .historical_geometry_statistics import _ci
from .reproduce_q2c import read, reference, write_new

METRICS = ['source_recall64', 'source_recall32', 'stale_leak64', 'stale_leak32',
           'visible_state_acc64', 'visible_state_acc32', 'global_f1_64', 'global_f1_32', 'regional_bce']


def leaf_errors(actual, expected):
    if set(actual) != set(expected):
        raise ValueError('historical metric keys differ')
    result = []
    for key, value in actual.items():
        if isinstance(value, dict):
            result.extend(leaf_errors(value, expected[key]))
        elif isinstance(value, (int, float)):
            result.append(abs(float(value) - float(expected[key])))
        else:
            raise ValueError('non-numeric metric leaf')
    return result


def summarize(inputs, root, output):
    authority = read(inputs / 'summary.json')
    expected = {(r['arm'], r['seed'], r['pair_id'], r['noise']): r
                for r in authority['historical_endpoints']}
    pair_ids = sorted(r['pair_id'] for r in authority['records'] if r['split'] == 'dev')
    if len(pair_ids) != 8 or len(expected) != 144:
        raise ValueError('original DEV8 x3model x3noise x2arm authority differs')
    observed, inputs_seen, maximum_error = {}, [], 0.
    for arm in ['r2_feat', 'r4_content']:
        for seed in [101, 202, 303]:
            path = root / arm / f'seed{seed}' / 'summary.json'
            run = read(path)
            if (run['status'] != 'PASS' or run['arm'] != arm or run['seed'] != seed
                    or len(run['rows']) != 24 or run['checkpoint'] != authority['models'][arm][str(seed)]):
                raise ValueError('missing or different learned reference checkpoint: ' + str(path))
            inputs_seen.append(reference(path))
            for row in run['rows']:
                key = (row['arm'], row['model_seed'], row['pair_id'], row['noise_seed'])
                if key in observed or key not in expected:
                    raise ValueError('duplicate or unexpected G2 replay row')
                cases = ['FACTORIAL', 'PROJ_SOURCE_OFF'] + (['GEOMETRY_SHUFFLED'] if arm == 'r4_content' else [])
                if set(row['metrics']) != set(cases):
                    raise ValueError('required source/query counterfactual is missing')
                for case in cases:
                    errors = leaf_errors(row['metrics'][case], expected[key]['metrics'][case])
                    if not np.isfinite(errors).all():
                        raise ValueError('nonfinite endpoint metric')
                    maximum_error = max(maximum_error, *errors)
                observed[key] = row
    if set(observed) != set(expected):
        raise ValueError('G2 endpoint replay is incomplete')
    metrics, contrasts = {}, {}
    for metric in METRICS:
        values = {}
        for arm in ['r2_feat', 'r4_content']:
            for case in ['FACTORIAL', 'PROJ_SOURCE_OFF'] + (['GEOMETRY_SHUFFLED'] if arm == 'r4_content' else []):
                key = arm + '/' + case
                values[key] = np.array([np.mean([observed[arm, seed, pair, noise]['metrics'][case][metric]
                    for seed in [101, 202, 303] for noise in [0, 1, 2]]) for pair in pair_ids])
                metrics.setdefault(key, {})[metric] = _ci(values[key], pair_ids)
        contrasts[metric] = {
            'content_minus_feature': _ci(values['r4_content/FACTORIAL'] - values['r2_feat/FACTORIAL'], pair_ids),
            'content_source_on_minus_projected_source_off': _ci(values['r4_content/FACTORIAL'] - values['r4_content/PROJ_SOURCE_OFF'], pair_ids),
            'content_geometry_on_minus_shuffled': _ci(values['r4_content/FACTORIAL'] - values['r4_content/GEOMETRY_SHUFFLED'], pair_ids)}
    result = {'status': 'PASS_ENDPOINT_REPLAY', 'rows': 144, 'pairs': 8, 'model_seeds': 3,
              'sampling_noise_seeds': 3, 'all_metric_leaves_exact': maximum_error == 0,
              'maximum_metric_leaf_abs_difference': maximum_error,
              'metrics': metrics, 'contrasts': contrasts, 'inputs': inputs_seen,
              'scope': 'replay of original trained checkpoints and original consumer/data, not fresh128-update training',
              'original_optimizer_trajectory_bit_exact_claim': False,
              'projected_source_off_keeps_native_multiview_global_and_negative': True,
              'r2_is_historical_zero_input_reference_not_production_physical_Noquery': True,
              'current_DeformNeXt_SS_gain_claim': False, 'production_promotion': False}
    write_new(output, result)
    print(json.dumps({'status': result['status'], 'rows': 144, 'all_metric_leaves_exact': maximum_error == 0,
        'source_recall32': contrasts['source_recall32'], 'global_f1_32': contrasts['global_f1_32']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.inputs, args.root, args.output)


if __name__ == '__main__':
    main()
