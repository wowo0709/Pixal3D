"""Check that the benchmark aggregate covers the complete frozen population."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


METHODS = {
    'V1', 'Selected', 'MV-zero', 'Official-RGB1', 'TRELLIS2',
    'ReconViaGen', 'ShapeUP', 'SpaceControl', 'MVAdapter-Recon',
    'CorrAdapter-Recon',
}
METRICS = {'PSNR', 'SSIM', 'M_IoU', 'source_LPIPS', 'source_DreamSim',
           'source_CLIP', 'FID', 'KID'}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--aggregate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest_bytes = a.manifest.read_bytes()
    rows = [json.loads(line) for line in manifest_bytes.splitlines() if line]
    population = [r['sample_id'] for r in rows]
    aggregate = json.loads(a.aggregate.read_text())
    errors = []
    if len(population) != 109 or len(set(population)) != 109:
        errors.append('manifest must contain 109 unique cases')
    if aggregate.get('population_manifest_sha256') != hashlib.sha256(manifest_bytes).hexdigest():
        errors.append('manifest hash mismatch')
    common = aggregate.get('common_sample_ids', [])
    if len(common) != len(population) or set(common) != set(population) or aggregate.get('common_count') != len(population):
        errors.append('common set does not cover the full manifest')
    counts = aggregate.get('available_counts', {})
    methods = aggregate.get('methods', {})
    if set(counts) != METHODS or set(methods) != METHODS:
        errors.append('aggregate method list differs from the required table')
    for method in sorted(METHODS):
        if counts.get(method) != len(population):
            errors.append(f'{method}: incomplete case coverage')
        for alignment in ('raw', 'source_m0'):
            values = methods.get(method, {}).get(alignment, {})
            def finite_metric(key: str) -> bool:
                value = values.get(key)
                if key == 'KID' and isinstance(value, dict):
                    value = value.get('KID')
                return isinstance(value, (int, float)) and math.isfinite(value)
            if not METRICS.issubset(values) or not all(finite_metric(key) for key in METRICS):
                errors.append(f'{method}/{alignment}: missing metric')
    result = {'status': 'PASS_FULL_COVERAGE' if not errors else 'INCOMPLETE_COVERAGE',
              'population_count': len(population), 'common_count': len(common),
              'available_counts': counts, 'errors': errors}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
