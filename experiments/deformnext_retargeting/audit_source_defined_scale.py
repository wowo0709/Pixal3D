"""Audit GT-free M0 mesh scale against the legacy M1-derived scale.

The M1 scale is read solely for a retrospective distribution check. No model
generation or hyperparameter search is performed by this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_toolkit.utils import sphere_normalize_torch, transform_mesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    result = []
    for index, row in enumerate(rows, 1):
        with np.load(row['source_geometry']) as archive:
            vertices = torch.from_numpy(np.asarray(archive['V']).copy()).float()
        normalized, _, radius = sphere_normalize_torch(vertices)
        transformed = transform_mesh(normalized, {'transform_matrix': row['transform_matrix'][0]})
        source_scale = .49999 / transformed.abs().max().item() / radius.item()
        target_scale = float(row['mesh_scale'])
        result.append({'sample_id': row['sample_id'], 'source_uid': row['source_uid'],
                       'source_defined_scale': source_scale,
                       'legacy_target_M1_scale': target_scale,
                       'relative_difference': source_scale / target_scale - 1,
                       'fixed_096_relative_difference': .96 / target_scale - 1})
        if index % 100 == 0:
            print(json.dumps({'audited': index, 'total': len(rows)}), flush=True)
    by = {field: np.asarray([row[field] for row in result]) for field in
          ('source_defined_scale', 'legacy_target_M1_scale', 'relative_difference',
           'fixed_096_relative_difference')}
    summary = {'schema': 'deformnext-source-M0-scale-audit-v1',
               'count': len(rows), 'manifest': str(args.manifest),
               'policy': 'M0 geometry plus supplied target camera only',
               'legacy_M1_used_for_generation': False,
               'statistics': {key: {'min': float(np.min(value)),
                                    'median': float(np.median(value)),
                                    'max': float(np.max(value)),
                                    'p05': float(np.quantile(value, .05)),
                                    'p95': float(np.quantile(value, .95)),
                                    'fraction_abs_over_5pct': float((np.abs(value) > .05).mean()),
                                    'fraction_abs_over_10pct': float((np.abs(value) > .10).mean())}
                              for key, value in by.items()},
               'rows': result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'PASS_SOURCE_DEFINED_SCALE_AUDIT', 'count': len(rows),
                      'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
