"""Paired common-TEST uncertainty for frozen Ours versus official RGB1.

The two comparisons share one predeclared 16-endpoint Bonferroni family:
V1 versus official and Selected versus official, each on five source/surface
and three target-camera metrics. FID/KID are distribution summaries only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


OFFICIAL = 'pixal3d_official_rgb1_source_m0'
OURS = ('ours_v1_source_m0', 'ours_selected_source_m0')
SOURCE = ('LPIPS', 'DreamSim', 'CLIP', 'CD', 'FScore')
TARGET = ('PSNR', 'SSIM', 'M_IoU')
METRICS = SOURCE + TARGET
DIRECTION = {'LPIPS': -1, 'DreamSim': -1, 'CLIP': 1, 'CD': -1,
             'FScore': 1, 'PSNR': 1, 'SSIM': 1, 'M_IoU': 1}


def read_method(directory: Path, method: str, expected: set[str]) -> dict[str, dict]:
    matches = []
    for path in directory.glob('*-results.json'):
        report = json.loads(path.read_text())
        if report['method'] == method:
            matches.append(report)
    if len(matches) != 1:
        raise ValueError(f'exactly one scored result required: {method}')
    report = matches[0]
    rows = report['per_deformation_seed']
    indexed = {row['sample_id']: row for row in rows}
    if (len(rows) != len(expected) or set(indexed) != expected
            or any(row['seed'] != 0 for row in rows)
            or not report['summary']['full_metric_computation']):
        raise ValueError('method lacks one fully scored row per common edit: ' + method)
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-evaluation', type=Path, required=True)
    parser.add_argument('--ours-evaluation', type=Path, required=True)
    parser.add_argument('--official-target8', type=Path, required=True)
    parser.add_argument('--ours-target8', type=Path, required=True)
    parser.add_argument('--common-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=20_000)
    args = parser.parse_args()
    if args.output.exists() or args.draws < 4_000:
        raise ValueError('fresh output and at least 4,000 cluster draws required')
    common = json.loads(args.common_manifest.read_text())
    official_target = json.loads(args.official_target8.read_text())
    ours_target = json.loads(args.ours_target8.read_text())
    ids = common['sample_ids']
    n = common['samples']
    if (common['status'] != 'PASS_THIRTEEN_METHOD_COMMON_SUCCESS_BINDING'
            or not 1 <= len(ids) == n <= 502 or len(set(ids)) != n
            or official_target['status'] != 'PASS_OFFICIAL_RGB1_RAW_AND_M0_TARGET8_PAIRED'
            or ours_target['status'] != 'PASS_OURS_V1_SELECTED_RAW_AND_M0_TARGET8_PAIRED'
            or official_target['samples'] != n or ours_target['samples'] != n):
        raise ValueError('exact final common population required')
    expected = set(ids)
    source = {OFFICIAL: read_method(args.official_evaluation, OFFICIAL, expected)}
    source.update({method: read_method(args.ours_evaluation, method, expected)
                   for method in OURS})
    off_rows = {row['sample_id']: row for row in official_target['per_sample']}
    ours_rows = {row['sample_id']: row for row in ours_target['per_sample']}
    if (len(off_rows) != n or set(off_rows) != expected
            or len(ours_rows) != n or set(ours_rows) != expected):
        raise ValueError('target-camera population differs from source evaluator')
    clusters = []
    for sid in ids:
        cluster = source[OFFICIAL][sid]['source_cluster_id']
        if (any(source[method][sid]['source_cluster_id'] != cluster for method in OURS)
                or off_rows[sid]['source_cluster_id'] != cluster
                or ours_rows[sid]['source_cluster_id'] != cluster):
            raise ValueError('source cluster mismatch: ' + sid)
        clusters.append(cluster)
    unique = sorted(set(clusters))
    indices = [np.asarray([i for i, c in enumerate(clusters) if c == cluster],
                          dtype=np.int64) for cluster in unique]
    counts = np.asarray([len(x) for x in indices])
    draws = np.random.default_rng(20261028).integers(0, len(unique),
                                                     size=(args.draws, len(unique)))
    output = {}
    for method in OURS:
        effects = {}
        for metric in METRICS:
            if metric in SOURCE:
                left = np.asarray([source[OFFICIAL][sid]['metrics'][metric] for sid in ids],
                                  dtype=np.float64)
                right = np.asarray([source[method][sid]['metrics'][metric] for sid in ids],
                                   dtype=np.float64)
            else:
                left = np.asarray([off_rows[sid]['source_m0'][metric] for sid in ids],
                                  dtype=np.float64)
                right = np.asarray([ours_rows[sid]['methods'][method][metric] for sid in ids],
                                   dtype=np.float64)
            if not np.isfinite(left).all() or not np.isfinite(right).all():
                raise ValueError('nonfinite paired values: ' + method + '/' + metric)
            delta = right - left
            sums = np.asarray([delta[x].sum() for x in indices])
            bootstrap = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
            ci95 = np.quantile(bootstrap, [.025, .975])
            family16 = np.quantile(bootstrap, [.05 / 32, 1 - .05 / 32])
            favorable = (family16[0] > 0 if DIRECTION[metric] > 0
                         else family16[1] < 0)
            effects[metric] = {
                'official_mean': float(left.mean()), 'ours_mean': float(right.mean()),
                'ours_minus_official': float(delta.mean()),
                'cluster_ci95': [float(x) for x in ci95],
                'family16_bonferroni_ci95': [float(x) for x in family16],
                'favorable_point': bool(DIRECTION[metric] * delta.mean() > 0),
                'family16_favorable_interval': bool(favorable),
            }
        output[method] = effects
    report = {
        'status': 'PASS_COMMON_TEST_OURS_VERSUS_OFFICIAL_PAIRED_SIXTEEN_ENDPOINTS',
        'samples': n, 'source_clusters': len(unique),
        'left': OFFICIAL, 'right_methods': list(OURS), 'contrasts': output,
        'bootstrap': {'unit': 'source_cluster_id', 'draws': args.draws,
                      'seed': 20261028, 'family': 'two comparisons times eight endpoints'},
        'interpretation_limit': ('Official RGB1 receives T1+C; Ours receives T1+S3+C. '
                                 'The comparison measures complete systems with different '
                                 'available image evidence, not an isolated module effect.'),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'samples': n,
                      'source_clusters': len(unique), 'contrasts': len(output)}))


if __name__ == '__main__':
    main()
