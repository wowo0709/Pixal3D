"""Score official RGB1 Pixal3D 1024 output in the common TEST frame.

The model receives target RGB and camera FOV/distance without object scale.
The input source M0 mesh is used only for optional post-generation Sim(3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from score_source_sim3_baselines import apply, register_m0


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path: Path) -> str:
    state = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            state.update(block)
    return state.hexdigest()


def anchor_to_world(camera_matrix: list) -> np.ndarray:
    """Use the same camera-frame transform as the frozen multiview evaluator."""
    import math
    import torch

    # Exact float32 matrix construction from data_toolkit.utils.transform_mesh;
    # duplicate it here because the node16 metric environment has no cv2.
    original = torch.tensor(camera_matrix, dtype=torch.float32)
    radius = original[:3, 3].norm().item()
    yaw, pitch = -math.pi / 2, 0.0
    eye = torch.tensor([radius * math.cos(yaw) * math.cos(pitch),
                        radius * math.sin(yaw) * math.cos(pitch),
                        radius * math.sin(pitch)], dtype=torch.float32)
    forward = -eye / eye.norm()
    right = torch.cross(forward, torch.tensor([0., 1., 0.]), dim=0)
    right /= right.norm()
    up = torch.cross(right, forward, dim=0)
    new = torch.eye(4)
    new[:3, 0], new[:3, 1], new[:3, 2], new[:3, 3] = right, up, -forward, eye
    rz90 = torch.tensor([[0., -1., 0., 0.], [1., 0., 0., 0.],
                         [0., 0., 1., 0.], [0., 0., 0., 1.]])
    init = torch.tensor([[1., 0., 0., 0.], [0., 0., -1., 0.],
                         [0., 1., 0., 0.], [0., 0., 0., 1.]])
    back = torch.tensor([[1., 0., 0., 0.], [0., 0., 1., 0.],
                         [0., -1., 0., 0.], [0., 0., 0., 1.]])
    ply = back
    transform = back @ (new @ rz90) @ torch.linalg.inv(original) @ ply @ init
    basis = torch.tensor([[0., 0., 0., 1.], [1., 0., 0., 1.],
                          [0., 1., 0., 1.], [0., 0., 1., 1.]])
    transformed = (transform @ basis.T).T[:, :3].numpy()
    rotation = (transformed[1:] - transformed[0]).T
    if (not np.allclose(transformed[0], 0, atol=1e-5)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)):
        raise ValueError('anchor transform is not a pure rotation')
    to_world = np.eye(4)
    to_world[:3, :3] = rotation.T  # mesh_scale is the native default 1.0
    return to_world


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--common-subset', type=Path, required=True)
    parser.add_argument('--expected-samples', type=int, default=502,
                        help='Intersection of frozen external common502 and official successes')
    parser.add_argument('--pixal3d-repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='Pilot only; zero scores all common cases')
    parser.add_argument('--sample-id', help='Pilot only; must belong to frozen common502')
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError('new output and valid shard required')

    import sys
    sys.path.insert(0, str(args.pixal3d_repo.resolve()))
    from deformnext_benchmark.io import canonical_hash, read_mesh
    from deformnext_benchmark.metrics import sample_surface, surface_metrics

    manifest = {row['sample_id']: row for row in read_rows(args.manifest)}
    subset = json.loads(args.common_subset.read_text())['sample_ids']
    if (args.expected_samples < 1 or len(subset) != args.expected_samples
            or len(set(subset)) != args.expected_samples or not set(subset) <= set(manifest)):
        raise ValueError('declared common-success subset/manifest required')
    if args.sample_id and (args.limit or args.shard_count != 1 or args.sample_id not in subset):
        raise ValueError('pilot sample must be one frozen common502 case with no other slicing')
    ids = ([args.sample_id] if args.sample_id else
           (subset[:args.limit] if args.limit else subset)[args.shard_index::args.shard_count])
    results = []
    for index, sid in enumerate(ids, 1):
        row = manifest[sid]
        case = args.official_root / 'cases' / sid
        prediction = case / 'prediction.npz'
        receipt = json.loads((case / 'generation.json').read_text())
        if (receipt['status'] != 'PASS_OFFICIAL_PIXAL3D_GIVEN1024_SCALE_UNGIVEN'
                or receipt['sample_id'] != sid or receipt['prediction_sha256'] != digest(prediction)
                or receipt['actual_resolution'] != 1024
                or receipt['mesh_scale_supplied'] or receipt['mesh_scale_effective_native_default'] != 1.0
                or 'mesh_scale' in receipt['camera_params']
                or receipt['GT_target_geometry_or_depth_used']
                or receipt['source_M0_mesh_used_in_generation']):
            raise ValueError('official generation provenance failed: ' + sid)
        with np.load(prediction, allow_pickle=False) as raw:
            vertices, faces = raw['vertices'].copy(), raw['faces'].copy()
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2
                or faces.shape[1] != 3 or not np.isfinite(vertices).all()
                or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError('invalid official mesh: ' + sid)
        to_world = anchor_to_world(row['transform_matrix'][0])
        world = apply(vertices, to_world)

        # M0 and M1 are opened only after the official prediction is complete.
        source_path, target_path = Path(row['source_geometry']), Path(row['target_geometry'])
        sv, sf = read_mesh(source_path)
        tv, tf = read_mesh(target_path)
        stable = int(canonical_hash(sid)[:8], 16)
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
            'method': 'pixal3d_official_given_camera_scale_ungiven_1024_rgb1',
            'prediction_sha256': receipt['prediction_sha256'],
            'source_geometry_sha256': digest(source_path),
            'registration_reference': 'input_source_M0_only',
            'native_mesh_scale': 1.0, 'to_world': to_world.tolist(),
            'source_fit_before': fit_before, 'source_fit_after': fit_after,
            'correction_sim3': correction.tolist(),
            'correction_scale': float(np.cbrt(np.linalg.det(correction[:3, :3]))),
            'raw_target_M1': raw_metrics, 'source_aligned_target_M1': aligned_metrics,
        })
        if index % 25 == 0:
            print(f'{index}/{len(ids)} samples', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        for result in results:
            stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'status': 'PASS_OFFICIAL_RGB1_SCALE_UNGIVEN_SOURCE_SIM3_SURFACE',
                      'samples': len(results), 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
