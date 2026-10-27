"""Audit camera metadata and official MoGe estimates without scoring TEST."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stats(values: list[float]) -> dict:
    values = sorted(values)
    return {'min': values[0], 'median': statistics.median(values),
            'max': values[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--baseline-runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    if len(rows) != 508 or len({row['sample_id'] for row in rows}) != 508:
        raise ValueError('expected frozen TEST508 manifest')
    by_sample = {}
    for path in args.baseline_runs.glob('*__pixal3d__seed0*/01_generation/run_meta.json'):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if record.get('status') != 'ok' or 'camera_params' not in record:
            continue
        sample_id = record['sample_id']
        prior = by_sample.get(sample_id)
        if prior is None or record['completed_at_utc'] > prior['completed_at_utc']:
            by_sample[sample_id] = record
    gt_fov = [float(row['camera_angle_x'][0]) for row in rows]
    gt_distance = [math.sqrt(sum(float(row['transform_matrix'][0][i][3]) ** 2
                                 for i in range(3))) for row in rows]
    gt_scale = [float(row['mesh_scale']) for row in rows]
    matrix_templates = {tuple(round(float(value), 12)
                              for camera in row['transform_matrix']
                              for line in camera for value in line) for row in rows}
    predicted = [by_sample[row['sample_id']]['camera_params'] for row in rows
                 if row['sample_id'] in by_sample]
    result = {'status': 'PASS_TEST508_CAMERA_INPUT_AUDIT',
              'test_manifest_sha256': sha256(args.manifest),
              'test_samples': len(rows), 'official_camera_receipts': len(predicted),
              'fixed_rig_templates_at_1e-12': len(matrix_templates),
              'gt_fov_degrees': stats([v * 180 / math.pi for v in gt_fov]),
              'gt_target_distance': stats(gt_distance),
              'gt_mesh_scale': stats(gt_scale),
              'official_moge_fov_degrees': stats([
                  float(x['camera_angle_x']) * 180 / math.pi for x in predicted]),
              'official_moge_distance': stats([float(x['distance']) for x in predicted]),
              'official_mesh_scale': stats([float(x['mesh_scale']) for x in predicted]),
              'official_uses_dataset_camera_at_generation': False,
              'ours_uses_dataset_camera_and_scale_at_generation': True,
              'score_or_method_selection_performed': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
