"""Require source-Sim3 scorer's raw geometry to match frozen node16 scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = ('trellis2', 'pixal3d', 'reconviagen', 'mvadapter', 'corradapter', 'shapeup', 'spacecontrol')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, nargs='+', required=True)
    parser.add_argument('--frozen-results-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for path in args.scores for line in path.read_text().splitlines() if line.strip()]
    index = {(r['method'], r['sample_id']): r for r in rows}
    if len(rows) != len(index) or len(rows) != 502 * len(METHODS):
        raise ValueError('all paired common502 rows required')
    mismatches = []
    for method in METHODS:
        result_paths = list((args.frozen_results_root / f'{method}-full-test').glob('*-results.json'))
        if len(result_paths) != 1:
            raise ValueError('one frozen results file required: ' + method)
        frozen = json.loads(result_paths[0].read_text())
        frozen_index = {r['sample_id']: r for r in frozen['per_deformation_seed']}
        for sid in sorted(sample_id for m, sample_id in index if m == method):
            measured = index[method, sid]
            old = frozen_index[sid]
            for metric in ('CD', 'FScore', 'CD_squared_L2'):
                difference = abs(measured['raw_target_M1'][metric] - old['metrics'][metric])
                if difference > 1e-10:
                    mismatches.append((method, sid, metric, difference))
            if measured['prediction_mesh_sha256'] != old['prediction_mesh_sha256']:
                mismatches.append((method, sid, 'prediction_mesh_sha256', 'different'))
    if mismatches:
        raise ValueError(f'raw baseline parity failed: {len(mismatches)}; first {mismatches[:3]}')
    report = {'status': 'PASS_EXACT_FROZEN_RAW_SURFACE_PARITY', 'methods': list(METHODS),
              'samples': 502, 'comparisons': 502 * len(METHODS) * 4, 'threshold': 1e-10}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
