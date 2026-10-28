"""Paired source-cluster uncertainty for frozen V1 versus Selected on common TEST."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SOURCE = ('LPIPS', 'DreamSim', 'CLIP', 'CD', 'FScore')
TARGET = ('PSNR', 'SSIM', 'M_IoU')
METRICS = SOURCE + TARGET
DIRECTION = {'LPIPS': -1, 'DreamSim': -1, 'CLIP': 1, 'CD': -1,
             'FScore': 1, 'PSNR': 1, 'SSIM': 1, 'M_IoU': 1}


def read_method(directory: Path, method: str) -> dict[str, dict]:
    matches = []
    for path in directory.glob('*-results.json'):
        record = json.loads(path.read_text())
        if record['method'] == method:
            matches.append(record)
    if len(matches) != 1:
        raise ValueError(f'exactly one evaluator result required for {method}: {matches!r}')
    result = matches[0]
    rows = result['per_deformation_seed']
    indexed = {row['sample_id']: row for row in rows}
    if len(indexed) != len(rows) or any(row['seed'] != 0 for row in rows):
        raise ValueError('one seed-zero evaluation per common edit required: ' + method)
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ours-evaluation', type=Path, required=True)
    parser.add_argument('--ours-target8', type=Path, required=True)
    parser.add_argument('--common-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('raw', 'source_m0'), default='source_m0')
    parser.add_argument('--draws', type=int, default=20000)
    args = parser.parse_args()
    if args.output.exists() or args.draws < 4000:
        raise ValueError('new output and at least 4000 cluster bootstrap draws required')
    common = json.loads(args.common_manifest.read_text())
    target = json.loads(args.ours_target8.read_text())
    left_method, right_method = 'ours_v1_' + args.mode, 'ours_selected_' + args.mode
    ids = common['sample_ids']
    if (common['status'] != 'PASS_THIRTEEN_METHOD_COMMON_SUCCESS_BINDING'
            or target['status'] != 'PASS_OURS_V1_SELECTED_RAW_AND_M0_TARGET8_PAIRED'
            or not 1 <= len(ids) == common['samples'] == target['samples'] <= 502
            or len(set(ids)) != len(ids)):
        raise ValueError('same complete common-success population required')
    source = {method: read_method(args.ours_evaluation, method)
              for method in (left_method, right_method)}
    target_rows = {row['sample_id']: row for row in target['per_sample']}
    if (len(target_rows) != len(ids)
            or any(set(source[method]) != set(ids) for method in (left_method, right_method))
            or set(target_rows) != set(ids)):
        raise ValueError('paired V1/Selected source and target populations differ')
    clusters = []
    for sid in ids:
        a, b, t = source[left_method][sid], source[right_method][sid], target_rows[sid]
        if not (a['source_cluster_id'] == b['source_cluster_id']
                == t['source_cluster_id']):
            raise ValueError('source-cluster identity differs: ' + sid)
        clusters.append(a['source_cluster_id'])
    unique = sorted(set(clusters))
    cluster_indices = [np.asarray([i for i, c in enumerate(clusters) if c == cluster],
                                  dtype=np.int64) for cluster in unique]
    counts = np.asarray([len(indices) for indices in cluster_indices])
    draws = np.random.default_rng(20261028).integers(0, len(unique),
                                                    size=(args.draws, len(unique)))
    outcomes = {}
    for metric in METRICS:
        if metric in SOURCE:
            left = np.asarray([source[left_method][sid]['metrics'][metric] for sid in ids],
                              dtype=np.float64)
            right = np.asarray([source[right_method][sid]['metrics'][metric] for sid in ids],
                               dtype=np.float64)
        else:
            left = np.asarray([target_rows[sid]['methods'][left_method][metric] for sid in ids],
                              dtype=np.float64)
            right = np.asarray([target_rows[sid]['methods'][right_method][metric] for sid in ids],
                               dtype=np.float64)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError('nonfinite paired metric: ' + metric)
        delta = right - left
        sums = np.asarray([delta[indices].sum() for indices in cluster_indices])
        bootstrap = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        ci95 = np.quantile(bootstrap, [0.025, 0.975]).tolist()
        family8 = np.quantile(bootstrap, [0.05 / 16, 1 - 0.05 / 16]).tolist()
        family16 = np.quantile(bootstrap, [0.05 / 32, 1 - 0.05 / 32]).tolist()
        outcomes[metric] = {
            'v1_mean': float(left.mean()), 'selected_mean': float(right.mean()),
            'selected_minus_v1': float(delta.mean()),
            'cluster_ci95': [float(x) for x in ci95],
            'family8_bonferroni_ci95': [float(x) for x in family8],
            'family16_bonferroni_ci95': [float(x) for x in family16],
            'favorable_point': bool(DIRECTION[metric] * delta.mean() > 0),
            'family8_favorable_interval': bool(
                DIRECTION[metric] * family8[0] > 0 if DIRECTION[metric] > 0
                else DIRECTION[metric] * family8[1] > 0),
            'family16_favorable_interval': bool(
                DIRECTION[metric] * family16[0] > 0 if DIRECTION[metric] > 0
                else DIRECTION[metric] * family16[1] > 0),
        }
    report = {
        'status': ('PASS_GT_SCALE_FREE_OURS_V1_SELECTED_COMMON_PAIRED_EIGHT_ENDPOINTS'
                   if args.mode == 'source_m0' else
                   'PASS_GT_SCALE_FREE_OURS_V1_SELECTED_RAW_COMMON_PAIRED_EIGHT_ENDPOINTS'),
        'samples': len(ids), 'source_clusters': len(unique),
        'methods': [left_method, right_method], 'mode': args.mode, 'metrics': outcomes,
        'bootstrap': {'unit': 'source_cluster_id', 'draws': args.draws,
                      'seed': 20261028, 'family': 'eight predeclared endpoints',
                      'additional_family': 'two alignment conditions times eight endpoints'},
        'interpretation_limit': 'Frozen checkpoint comparison; not separate causal effects of modules.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'samples': len(ids),
                      'source_clusters': len(unique)}))


if __name__ == '__main__':
    main()
