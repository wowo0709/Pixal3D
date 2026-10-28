"""Derive guarded paper claims from the final common TEST table and paired CIs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


METRICS = ('LPIPS', 'DreamSim', 'CLIP', 'PSNR', 'SSIM', 'M_IoU', 'CD', 'FScore')
DIRECTION = {'LPIPS': -1, 'DreamSim': -1, 'CLIP': 1,
             'PSNR': 1, 'SSIM': 1, 'M_IoU': 1, 'CD': -1, 'FScore': 1}
V1 = 'ours_v1_source_m0'
SELECTED = 'ours_selected_source_m0'
OFFICIAL = 'pixal3d_official_rgb1_source_m0'


def classification(delta: float, interval: list[float], direction: int) -> str:
    low, high = map(float, interval)
    if not all(map(math.isfinite, (delta, low, high))) or low > high:
        raise ValueError('nonfinite or reversed uncertainty interval')
    if (low > 0 and direction > 0) or (high < 0 and direction < 0):
        return 'supported_favorable'
    if (high < 0 and direction > 0) or (low > 0 and direction < 0):
        return 'supported_unfavorable'
    return 'unresolved'


def compare_means(row: dict, left: float, right: float,
                  left_key: str, right_key: str, delta_key: str) -> float:
    actual_left = float(row[left_key])
    actual_right = float(row[right_key])
    delta = float(row[delta_key])
    if (not all(map(math.isfinite, (actual_left, actual_right, delta)))
            or max(abs(actual_left - left), abs(actual_right - right),
                   abs(delta - (right - left))) > 1e-6):
        raise ValueError('paired means and final table do not agree')
    return delta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-table-json', type=Path, required=True)
    parser.add_argument('--paired-json', type=Path, required=True)
    parser.add_argument('--official-contrasts-json', type=Path, required=True)
    parser.add_argument('--common-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('claim audit output must be new')
    table = json.loads(args.main_table_json.read_text())
    paired = json.loads(args.paired_json.read_text())
    official = json.loads(args.official_contrasts_json.read_text())
    common = json.loads(args.common_manifest.read_text())
    n = common['samples']
    if (common['status'] != 'PASS_THIRTEEN_METHOD_COMMON_SUCCESS_BINDING'
            or not 1 <= n <= 502 or len(common['sample_ids']) != n
            or len(set(common['sample_ids'])) != n
            or table['status'] != 'PASS_SCALE_UNGIVEN_COMMON_SUCCESS_MAIN_TABLE'
            or table['samples'] != n or len(table['methods']) != 10
            or set(table['metrics']) != set(METRICS) | {'FID', 'KID'}
            or paired['status']
               != 'PASS_GT_SCALE_FREE_OURS_V1_SELECTED_COMMON_PAIRED_EIGHT_ENDPOINTS'
            or paired['samples'] != n or set(paired['metrics']) != set(METRICS)
            or paired['source_clusters'] != official['source_clusters']
            or paired['bootstrap']['family'] != 'eight predeclared endpoints'
            or official['status']
               != 'PASS_COMMON_TEST_OURS_VERSUS_OFFICIAL_PAIRED_SIXTEEN_ENDPOINTS'
            or official['samples'] != n
            or official['bootstrap']['family']
               != 'two comparisons times eight endpoints'
            or set(official['contrasts']) != {V1, SELECTED}):
        raise ValueError('final common-population result contracts incomplete')
    results = {'selected_vs_v1': {}, 'ours_vs_official': {}}
    for metric in METRICS:
        row = paired['metrics'][metric]
        delta = compare_means(row, table['values'][V1][metric],
                              table['values'][SELECTED][metric],
                              'v1_mean', 'selected_mean', 'selected_minus_v1')
        state = classification(delta, row['family8_bonferroni_ci95'],
                               DIRECTION[metric])
        if row['family8_favorable_interval'] != (state == 'supported_favorable'):
            raise ValueError('selected/V1 favorable flag disagrees with interval')
        results['selected_vs_v1'][metric] = {
            'selected_minus_v1': delta,
            'family8_interval': row['family8_bonferroni_ci95'],
            'classification': state,
        }
    for method in (V1, SELECTED):
        results['ours_vs_official'][method] = {}
        rows = official['contrasts'][method]
        if set(rows) != set(METRICS):
            raise ValueError('official contrast lacks eight predeclared endpoints')
        for metric in METRICS:
            row = rows[metric]
            delta = compare_means(row, table['values'][OFFICIAL][metric],
                                  table['values'][method][metric],
                                  'official_mean', 'ours_mean',
                                  'ours_minus_official')
            state = classification(delta, row['family16_bonferroni_ci95'],
                                   DIRECTION[metric])
            if row['family16_favorable_interval'] != (state == 'supported_favorable'):
                raise ValueError('official favorable flag disagrees with interval')
            results['ours_vs_official'][method][metric] = {
                'ours_minus_official': delta,
                'family16_interval': row['family16_bonferroni_ci95'],
                'classification': state,
            }
    report = {
        'status': 'PASS_FAIR_TEST_CLAIM_AUDIT',
        'samples': n, 'source_clusters': paired['source_clusters'],
        'primary_metric_directions': DIRECTION,
        'comparisons': results,
        'distribution_metrics': {
            name: {method: table['values'][method][name]
                   for method in (OFFICIAL, V1, SELECTED)}
            for name in ('FID', 'KID')},
        'claim_rules': [
            'Use improves/worsens only for favorable/unfavorable adjusted paired intervals.',
            'An interval containing zero is unresolved, not equivalent.',
            'FID/KID and all cross-method ranks are descriptive point estimates.',
            'Official RGB1 uses T1+C; Ours uses T1+S3+C: system comparison only.',
            'Selected versus V1 bundles modules and does not identify an isolated module effect.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    counts = {state: sum(row['classification'] == state
                         for group in [results['selected_vs_v1'],
                                       *results['ours_vs_official'].values()]
                         for row in group.values())
              for state in ('supported_favorable', 'supported_unfavorable',
                            'unresolved')}
    print(json.dumps({'status': report['status'], 'samples': n,
                      'classifications': counts}))


if __name__ == '__main__':
    main()
