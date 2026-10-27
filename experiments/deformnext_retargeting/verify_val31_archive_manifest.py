"""Finish and verify the file4 archive binding for paired VAL31 scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pending', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.pending.read_text().splitlines() if line]
    if len(rows) != 93 or len({(r['arm'], r['sample_id']) for r in rows}) != 93:
        raise ValueError('paired VAL31 × three arms required')
    filled = 0
    for row in rows:
        archive = Path(row['file4_archive'])
        if not str(archive).startswith('/file4/youngwoo/deformation/experiments/retargeting-v1/scale-ungiven-val-20261028/'):
            raise ValueError('unexpected archive root')
        receipt = json.loads((archive / 'generation.json').read_text())
        if (receipt['sample_id'] != row['sample_id']
                or receipt['object_scale_policy'] != 'image-estimated-v1'
                or receipt['M1_derived_scale_supplied_to_generation']
                or receipt['source_M0_mesh_used_in_generation']
                or receipt['GT_geometry_or_coords_used_in_generation']):
            raise ValueError('archived generation provenance failed')
        for key, filename in [('prediction_sha256', 'prediction.npz'),
                              ('geometry_sha256', 'geometry-evaluation.npz')]:
            actual = sha(archive / filename)
            if row.get(key) is None:
                row[key] = actual
                filled += 1
            elif row[key] != actual:
                raise ValueError('archived SHA differs: ' + row['sample_id'] + '/' + filename)
    args.output.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    print(json.dumps({'status': 'PASS_VERIFIED_VAL31_FILE4_ARCHIVES', 'rows': len(rows),
                      'filled_missing_hashes': filled, 'output_sha256': sha(args.output)}), flush=True)


if __name__ == '__main__':
    main()
