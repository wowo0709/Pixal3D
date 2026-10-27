"""Export a frozen Pixal3D prediction for the common Blender benchmark.

The raw prediction NPZ remains the metric surface. Only the archived native
render geometry is texture-baked into a GLB. The sole world transform comes
from the frozen camera anchor and mesh scale, without GT fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pixal3d-repo', type=Path, required=True)
    parser.add_argument('--benchmark-repo', type=Path, required=True)
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--render-geometry', type=Path, required=True)
    parser.add_argument('--anchor-evaluation', type=Path, required=True)
    parser.add_argument('--generation', type=Path, required=True)
    parser.add_argument('--mesh-scale', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--texture-size', type=int, default=2048)
    args = parser.parse_args()

    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('immutable export output already exists')
    if args.mesh_scale <= 0 or args.texture_size not in {2048, 4096}:
        raise ValueError('invalid mesh scale or texture size')
    sys.path.insert(0, str(args.pixal3d_repo.resolve()))
    sys.path.insert(0, str(args.benchmark_repo.resolve()))
    import torch
    import o_voxel
    from scipy.spatial import cKDTree
    from deformnext_benchmark.io import read_mesh
    from deformnext_benchmark.metrics import transform_vertices

    generation = json.loads(args.generation.read_text())
    resolution = int(generation['actual_resolution'])
    if resolution != 1024 or generation['GT_geometry_or_coords_used_in_generation']:
        raise ValueError('expected frozen 1024 prediction without GT-conditioned generation')
    with np.load(args.prediction, allow_pickle=False) as raw:
        if not {'vertices', 'faces', 'coords', 'attrs'} <= set(raw.files):
            raise ValueError('prediction missing raw geometry or PBR volume')
        raw_vertices = raw['vertices'].copy()
        raw_faces = raw['faces'].copy()
        coords = raw['coords'].copy()
        attrs = raw['attrs'].copy()
    with np.load(args.render_geometry, allow_pickle=False) as rendered:
        vertices = rendered['vertices'].copy()
        faces = rendered['faces'].copy()
    with np.load(args.anchor_evaluation, allow_pickle=False) as archived:
        rotation = archived['anchor_rotation'].copy()
    if (raw_vertices.ndim != 2 or raw_vertices.shape[1] != 3
            or raw_faces.ndim != 2 or raw_faces.shape[1] != 3
            or coords.ndim != 2 or coords.shape[1] != 3
            or attrs.shape != (len(coords), 6)
            or vertices.ndim != 2 or vertices.shape[1] != 3
            or faces.ndim != 2 or faces.shape[1] != 3
            or not np.isfinite(attrs).all()
            or not np.isfinite(vertices).all()
            or np.max(faces) >= len(vertices)
            or rotation.shape != (3, 3)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)):
        raise ValueError('invalid frozen geometry, PBR volume, or camera rotation')

    to_world = np.eye(4)
    to_world[:3, :3] = rotation.T / args.mesh_scale
    device = torch.device('cuda')
    glb = o_voxel.postprocess.to_glb(
        vertices=torch.from_numpy(vertices).to(device),
        faces=torch.from_numpy(faces).to(device),
        attr_volume=torch.from_numpy(attrs).to(device),
        coords=torch.from_numpy(coords).to(device),
        attr_layout={'base_color': slice(0, 3), 'metallic': slice(3, 4),
                     'roughness': slice(4, 5), 'alpha': slice(5, 6)},
        aabb=[[-.5, -.5, -.5], [.5, .5, .5]], grid_size=resolution,
        decimation_target=1_000_000, texture_size=args.texture_size,
        remesh=False, use_tqdm=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(args.output), extension_webp=True)
    exported_vertices, _ = read_mesh(args.output)
    actual = transform_vertices(exported_vertices, to_world)
    original = transform_vertices(vertices, to_world)
    bounds_error = float(np.max(np.abs(
        np.stack([actual.min(0), actual.max(0)])
        - np.stack([original.min(0), original.max(0)]))))
    # Baking may remove an isolated extremal vertex, so exact bbox parity is
    # not a valid file-format check. Compare the exported geometry against the
    # archived render geometry directly, after the benchmark's fixed GLB axis
    # conversion. No GT surface or per-output registration enters this test.
    rng = np.random.default_rng(42017)
    original_probe = original[rng.choice(len(original), min(4096, len(original)), replace=False)]
    exported_probe = actual[rng.choice(len(actual), min(4096, len(actual)), replace=False)]
    distances = [cKDTree(actual).query(original_probe, workers=1)[0],
                 cKDTree(original).query(exported_probe, workers=1)[0]]
    registration_q99 = float(max(np.quantile(d, .99) for d in distances))
    if not np.isfinite(registration_q99) or registration_q99 > .01:
        raise ValueError(f'exported GLB/render mesh differ in world space: {registration_q99}')
    report = {
        'status': 'PASS_COMMON_BENCHMARK_MESH_EXPORT',
        'prediction': str(args.prediction),
        'prediction_sha256': sha256(args.prediction),
        'render_geometry': str(args.render_geometry),
        'render_geometry_sha256': sha256(args.render_geometry),
        'glb': str(args.output), 'glb_sha256': sha256(args.output),
        'raw_surface_faces': len(raw_faces),
        'exported_render_faces': len(glb.faces) if hasattr(glb, 'faces') else None,
        'bounds_error_world': bounds_error,
        'render_geometry_registration_q99_world': registration_q99,
        'to_world': to_world.tolist(),
        'texture_size': args.texture_size,
        'remesh': False,
        'surface_metric_input_is_raw_prediction': True,
        'GT_fitting_or_alignment': False,
    }
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
