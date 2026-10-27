"""Summarize paired input-M0 registration on a fixed common-success set."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METHODS = ('trellis2', 'pixal3d', 'reconviagen', 'mvadapter', 'corradapter', 'shapeup', 'spacecontrol')


def summarize(rows: list[dict], *, draws: int = 2000) -> dict:
    ids = {row['sample_id'] for row in rows}
    if len(ids) != 502 or len(rows) != 502 * len(METHODS):
        raise ValueError('full paired common502 required')
    by_method = defaultdict(dict)
    for row in rows:
        method, sid = row['method'], row['sample_id']
        if method not in METHODS or sid in by_method[method]:
            raise ValueError('duplicate or unexpected method/sample')
        by_method[method][sid] = row
    if any(set(by_method[method]) != ids for method in METHODS):
        raise ValueError('method populations differ')
    clusters = defaultdict(list)
    for sid, row in by_method[METHODS[0]].items():
        clusters[row['source_cluster_id']].append(sid)
    if len(clusters) != 133:
        raise ValueError('expected 133 source clusters')
    cluster_ids = sorted(clusters)
    rng = np.random.default_rng(20261028)
    samples = rng.integers(0, len(cluster_ids), size=(draws, len(cluster_ids)))
    output = {'schema': 'deformnext-input-M0-Sim3-paired-common502-v1',
              'samples': len(ids), 'source_clusters': len(clusters), 'methods': {}}
    for method in METHODS:
        row_by_id = by_method[method]
        items = [row_by_id[sid] for sid in sorted(ids)]
        method_result = {
            'median_scale': float(np.median([row['correction_scale'] for row in items])),
            'median_source_fit_before': float(np.median([row['source_fit_before'] for row in items])),
            'median_source_fit_after': float(np.median([row['source_fit_after'] for row in items])),
        }
        for key in ('CD', 'FScore'):
            raw = {sid: row_by_id[sid]['raw_target_M1'][key] for sid in ids}
            aligned = {sid: row_by_id[sid]['source_aligned_target_M1'][key] for sid in ids}
            delta = {sid: aligned[sid] - raw[sid] for sid in ids}
            cluster_sums = np.asarray([sum(delta[sid] for sid in clusters[c]) for c in cluster_ids])
            cluster_counts = np.asarray([len(clusters[c]) for c in cluster_ids])
            bootstrap_delta = cluster_sums[samples].sum(axis=1) / cluster_counts[samples].sum(axis=1)
            method_result[key] = {
                'raw_mean': float(np.mean(list(raw.values()))),
                'source_aligned_mean': float(np.mean(list(aligned.values()))),
                'paired_delta': float(np.mean(list(delta.values()))),
                'paired_delta_cluster_ci95': [float(x) for x in np.quantile(bootstrap_delta, [.025, .975])],
                'fraction_improved': float(np.mean([delta[sid] < 0 if key == 'CD' else delta[sid] > 0 for sid in ids])),
            }
        output['methods'][method] = method_result
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    result = summarize(rows)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'status': 'PASS_SOURCE_SIM3_SUMMARY', 'output': str(args.output)}))


if __name__ == '__main__':
    main()
