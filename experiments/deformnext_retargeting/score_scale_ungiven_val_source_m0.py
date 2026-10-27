"""Raw and input-source-M0 Sim(3) surface scores on frozen image-scale VAL31.

Uses the common benchmark sampler and ICP implementation. The edited M1
target mesh is opened only after the source-only correction is fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from score_source_sim3_baselines import apply, register_m0


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError('new output and valid shard required')
    from deformnext_benchmark.io import canonical_hash, read_mesh
    from deformnext_benchmark.metrics import sample_surface, surface_metrics

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if (len(rows) != 31 * 3 or len({(r['arm'], r['sample_id']) for r in rows}) != len(rows)
            or {r['arm'] for r in rows} != {'V1', 'Retarget', 'Selected'}
            or len({r['source_cluster_id'] for r in rows}) != 31):
        raise ValueError('complete frozen paired VAL31 × three arms required')
    results = []
    for index, row in enumerate(rows[args.shard_index::args.shard_count], 1):
        sid, arm = row['sample_id'], row['arm']
        archive = Path(row['file4_archive'])
        prediction, evaluation = archive / 'prediction.npz', archive / 'geometry-evaluation.npz'
        if sha(prediction) != row['prediction_sha256'] or sha(evaluation) != row['geometry_sha256']:
            raise ValueError(f'file4 archive SHA differs: {arm} {sid}')
        with np.load(prediction, allow_pickle=False) as z:
            faces = z['faces'].copy()
        with np.load(evaluation, allow_pickle=False) as z:
            predicted_world = z['prediction_world'].copy()
        if (predicted_world.ndim != 2 or predicted_world.shape[1] != 3
                or faces.ndim != 2 or faces.shape[1] != 3
                or not np.isfinite(predicted_world).all() or not len(faces)
                or faces.min() < 0 or faces.max() >= len(predicted_world)):
            raise ValueError(f'invalid archived predicted geometry: {arm} {sid}')
        source_path = Path(row['source_geometry'])
        source_vertices, source_faces = read_mesh(source_path)
        stable = int(canonical_hash(sid)[:8], 16)
        source_points = sample_surface(source_vertices, source_faces, count=4096, seed=stable + 101)
        predicted_points = sample_surface(predicted_world, faces, count=20000, seed=stable)
        registration_points = sample_surface(predicted_world, faces, count=4096, seed=stable + 103)
        correction, fit_after, fit_before = register_m0(registration_points, source_points)
        # The target M1 mesh enters only after the correction has been fixed.
        target_vertices, target_faces = read_mesh(Path(row['target_geometry']))
        target_points = sample_surface(target_vertices, target_faces, count=20000, seed=stable + 1)
        results.append({
            'sample_id': sid, 'source_cluster_id': row['source_cluster_id'], 'arm': arm,
            'image_scale_generation': True, 'prediction_sha256': row['prediction_sha256'],
            'source_geometry_sha256': sha(source_path),
            'registration_reference': 'input_source_M0_only',
            'source_fit_before': fit_before, 'source_fit_after': fit_after,
            'correction_sim3': correction.tolist(),
            'correction_scale': float(np.cbrt(np.linalg.det(correction[:3, :3]))),
            'raw_target_M1': surface_metrics(predicted_points, target_points, threshold=.02),
            'source_aligned_target_M1': surface_metrics(apply(predicted_points, correction),
                                                        target_points, threshold=.02),
        })
        print(json.dumps({'index': index, 'arm': arm, 'sample_id': sid, 'status': 'PASS'}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        for item in results:
            handle.write(json.dumps(item, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'status': 'PASS_VAL31_SOURCE_M0_SURFACE_SHARD', 'rows': len(results),
                      'shard': args.shard_index, 'shards': args.shard_count}), flush=True)


if __name__ == '__main__':
    main()
