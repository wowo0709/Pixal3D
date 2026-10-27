"""Paired VAL31 raw/+M0 surface summary using the common benchmark metric."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARMS = ('V1', 'Retarget', 'Selected')
CONDITIONS = ('raw_target_M1', 'source_aligned_target_M1')
METRICS = ('CD', 'FScore')


def interval(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(1)
    return [float(v) for v in np.quantile(samples, [.025, .975])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for path in args.scores for line in path.read_text().splitlines() if line]
    by_arm = {arm: {row['sample_id']: row for row in rows if row['arm'] == arm} for arm in ARMS}
    ids = set(by_arm['V1'])
    if (len(rows) != 93 or len(ids) != 31 or
            any(set(by_arm[arm]) != ids for arm in ARMS) or
            len({by_arm['V1'][sid]['source_cluster_id'] for sid in ids}) != 31):
        raise ValueError('complete source-disjoint VAL31 × three-arm score closure required')
    ids = sorted(ids)
    for sid in ids:
        if (len({by_arm[arm][sid]['source_cluster_id'] for arm in ARMS}) != 1
                or len({by_arm[arm][sid]['source_geometry_sha256'] for arm in ARMS}) != 1
                or any(by_arm[arm][sid]['registration_reference'] != 'input_source_M0_only'
                       for arm in ARMS)):
            raise ValueError('paired source or registration binding differs')
    summary = {'status': 'PASS_PAIRED_VAL31_COMMON_SURFACE_RAW_AND_M0',
               'samples': 31, 'source_clusters': 31,
               'registration': 'input_source_M0_only; target_M1_after_correction',
               'conditions': {}}
    for condition in CONDITIONS:
        group = {}
        for arm in ARMS:
            record = {}
            for metric in METRICS:
                values = np.array([by_arm[arm][sid][condition][metric] for sid in ids])
                record[metric] = float(values.mean())
                if arm != 'V1':
                    base = np.array([by_arm['V1'][sid][condition][metric] for sid in ids])
                    difference = values - base
                    record[f'delta_{metric}_vs_V1'] = float(difference.mean())
                    record[f'delta_{metric}_CI95'] = interval(difference, 20261028)
                    record[f'{metric}_improved_cases'] = int(
                        (difference < 0).sum() if metric == 'CD' else (difference > 0).sum())
            group[arm] = record
        summary['conditions'][condition] = group
    summary['M0_postprocess'] = {}
    for arm in ARMS:
        record = {}
        for metric in METRICS:
            difference = np.array([by_arm[arm][sid]['source_aligned_target_M1'][metric]
                                   - by_arm[arm][sid]['raw_target_M1'][metric] for sid in ids])
            record[f'delta_{metric}_aligned_minus_raw'] = float(difference.mean())
            record[f'delta_{metric}_CI95'] = interval(difference, 20261029)
        scales = np.array([by_arm[arm][sid]['correction_scale'] for sid in ids])
        record['correction_scale_median'] = float(np.median(scales))
        record['correction_scale_range'] = [float(scales.min()), float(scales.max())]
        summary['M0_postprocess'][arm] = record
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': summary['status'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
