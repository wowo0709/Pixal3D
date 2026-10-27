"""Score frozen V1/Selected TEST meshes raw and after source-M0-only Sim(3)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from score_source_sim3_baselines import apply, register_m0


ARMS = {'V1': 'shape', 'SELECTED': 'top1-pbr'}
EXPECTED = {'V1': 'vanilla', 'SELECTED': 'top1-only'}


def sha(path: Path) -> str:
    state = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            state.update(block)
    return state.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--common-subset', type=Path, required=True)
    parser.add_argument('--benchmark-repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-samples', type=int, default=502)
    parser.add_argument('--sample-id', help='One-case mechanical pilot only')
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    if (args.output.exists() or args.shard_count < 1
            or not 0 <= args.shard_index < args.shard_count):
        raise ValueError('new output and valid shard required')
    sys.path.insert(0, str(args.benchmark_repo.resolve()))
    from deformnext_benchmark.io import canonical_hash, read_mesh
    from deformnext_benchmark.metrics import sample_surface, surface_metrics

    manifest = {row['sample_id']: row for row in rows(args.manifest)}
    subset = json.loads(args.common_subset.read_text())['sample_ids']
    if (len(subset) != args.expected_samples or len(set(subset)) != len(subset)
            or not set(subset) <= set(manifest)):
        raise ValueError('frozen common-success subset/manifest differs')
    if args.sample_id:
        if args.sample_id not in subset or args.shard_count != 1:
            raise ValueError('pilot sample must belong to common set')
        ids = [args.sample_id]
    else:
        ids = subset[args.shard_index::args.shard_count]
    results = []
    for index, sid in enumerate(ids, 1):
        row = manifest[sid]
        stable = int(canonical_hash(sid)[:8], 16)
        for arm, stage in ARMS.items():
            case = args.archive_root / arm / sid / stage
            receipt = json.loads((case / 'archive-receipt.json').read_text())
            export = json.loads((case / 'benchmark.json').read_text())
            prediction = case / 'prediction.npz'
            if (receipt['status'] != 'PASS_FILE4_ARCHIVE_CHECKSUM_VERIFIED'
                    or receipt['sha256']['prediction.npz'] != sha(prediction)
                    or receipt['sha256']['benchmark.glb'] != export['glb_sha256']
                    or export['status'] != 'PASS_OURS_FROZEN_GT_SCALE_FREE_1024_EXPORT'
                    or export['sample_id'] != sid
                    or export['texture_arm'] != EXPECTED[arm]
                    or export['prediction_sha256'] != receipt['sha256']['prediction.npz']
                    or export['GT_fitting_or_alignment']
                    or export['texture_size'] != 2048):
                raise ValueError('frozen archive/export binding differs: ' + arm + '/' + sid)
            shape_case = args.archive_root / arm / sid / 'shape'
            shape = json.loads((shape_case / 'generation.json').read_text())
            if (shape['status'] != 'PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION'
                    or shape['sample_id'] != sid
                    or shape['object_scale_policy'] != 'image-estimated-v1'
                    or shape['M1_derived_scale_supplied_to_generation']
                    or shape['GT_geometry_or_coords_used_in_generation']
                    or not np.isclose(export['image_estimated_scale'],
                                      shape['scale_used_in_generation'], rtol=0, atol=1e-8)):
                raise ValueError('frozen GT-free shape provenance differs: ' + arm + '/' + sid)
            if arm == 'SELECTED':
                texture = json.loads((case / 'generation.json').read_text())
                if (texture['status'] != 'PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE'
                        or texture['sample_id'] != sid or texture['PBR_arm'] != 'top1-only'
                        or not all(texture['geometry_array_byte_identity'].values())):
                    raise ValueError('paired Top-1 texture provenance differs: ' + sid)
            with np.load(prediction, allow_pickle=False) as raw:
                vertices, faces = raw['vertices'].copy(), raw['faces'].copy()
            if (vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2
                    or faces.shape[1] != 3 or not np.isfinite(vertices).all()
                    or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)):
                raise ValueError('invalid frozen raw prediction: ' + arm + '/' + sid)
            to_world = np.asarray(export['to_world'], dtype=np.float64)
            if (to_world.shape != (4, 4) or not np.allclose(to_world[3], [0, 0, 0, 1])
                    or not np.allclose(to_world[:3, :3].T @ to_world[:3, :3],
                                       np.eye(3) / export['image_estimated_scale'] ** 2,
                                       atol=1e-5)):
                raise ValueError('image-only camera/world transform differs: ' + arm + '/' + sid)
            world = apply(vertices, to_world)
            # Read M0 and M1 only after the frozen prediction/export checks.
            source_path, target_path = Path(row['source_geometry']), Path(row['target_geometry'])
            sv, sf = read_mesh(source_path)
            tv, tf = read_mesh(target_path)
            source_points = sample_surface(sv, sf, count=4096, seed=stable + 101)
            target_points = sample_surface(tv, tf, count=20000, seed=stable + 1)
            predicted_points = sample_surface(world, faces, count=20000, seed=stable)
            registration_points = sample_surface(world, faces, count=4096, seed=stable + 103)
            correction, fit_after, fit_before = register_m0(registration_points, source_points)
            raw_metrics = surface_metrics(predicted_points, target_points, threshold=.02)
            aligned_metrics = surface_metrics(apply(predicted_points, correction), target_points,
                                              threshold=.02)
            results.append({
                'sample_id': sid, 'source_cluster_id': row['source_cluster_id'],
                'method': 'ours_' + arm.lower() + '_image_scale_1024',
                'texture_arm': EXPECTED[arm], 'prediction_sha256': export['prediction_sha256'],
                'source_geometry_sha256': sha(source_path),
                'registration_reference': 'input_source_M0_only',
                'image_estimated_scale': export['image_estimated_scale'],
                'to_world': to_world.tolist(), 'source_fit_before': fit_before,
                'source_fit_after': fit_after, 'correction_sim3': correction.tolist(),
                'correction_scale': float(np.cbrt(np.linalg.det(correction[:3, :3]))),
                'raw_target_M1': raw_metrics, 'source_aligned_target_M1': aligned_metrics})
        if index % 25 == 0:
            print(f'{index}/{len(ids)} samples', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        for result in results:
            stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'status': 'PASS_OURS_V1_SELECTED_IMAGE_SCALE_SOURCE_M0_SURFACE',
                      'samples': len(ids), 'rows': len(results), 'output': str(args.output)}),
          flush=True)


if __name__ == '__main__':
    main()
