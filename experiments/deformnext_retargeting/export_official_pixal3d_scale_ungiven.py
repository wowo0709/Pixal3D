"""Texture-bake one official no-scale Pixal3D mesh for the common renderer.

Surface scores use the untouched prediction NPZ. This GLB is only a render
asset. Its world transform is the provided target camera anchor with native
mesh scale 1.0; no target geometry or per-output registration is consulted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from score_official_pixal3d_scale_ungiven import anchor_to_world


def sha(path: Path) -> str:
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
    parser.add_argument('--generation', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--texture-size', type=int, choices=(2048, 4096), default=2048)
    parser.add_argument('--pre-simplify', action='store_true',
                        help='Apply the native render-only face budget first')
    parser.add_argument('--render-face-budget', type=int,
                        choices=(250_000, 500_000, 1_000_000, 2_000_000,
                                 5_000_000), default=1_000_000)
    parser.add_argument('--simplifier', choices=('auto', 'native', 'fast'), default='auto')
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('immutable export output already exists')
    sys.path.insert(0, str(args.pixal3d_repo.resolve()))
    sys.path.insert(0, str(args.benchmark_repo.resolve()))
    import torch
    import o_voxel
    from scipy.spatial import cKDTree
    from deformnext_benchmark.io import read_mesh
    from deformnext_benchmark.metrics import transform_vertices

    generation = json.loads(args.generation.read_text())
    sid = generation['sample_id']
    if (generation['status'] != 'PASS_OFFICIAL_PIXAL3D_GIVEN1024_SCALE_UNGIVEN'
            or generation['actual_resolution'] != 1024
            or generation['mesh_scale_supplied']
            or generation['mesh_scale_effective_native_default'] != 1.0
            or 'mesh_scale' in generation['camera_params']
            or generation['GT_target_geometry_or_depth_used']
            or generation['source_M0_mesh_used_in_generation']
            or generation['prediction_sha256'] != sha(args.prediction)):
        raise ValueError('official GT-free generation provenance failed')
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    matches = [row for row in rows if row['sample_id'] == sid]
    if len(matches) != 1:
        raise ValueError('exactly one frozen TEST manifest case required')
    row = matches[0]
    to_world = anchor_to_world(row['transform_matrix'][0])
    with np.load(args.prediction, allow_pickle=False) as raw:
        vertices, faces = raw['vertices'].copy(), raw['faces'].copy()
        coords, attrs = raw['coords'].copy(), raw['attrs'].copy()
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3
            or coords.ndim != 2 or coords.shape[1] != 3 or attrs.shape != (len(coords), 6)
            or not len(vertices) or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)
            or not np.isfinite(vertices).all() or not np.isfinite(attrs).all()):
        raise ValueError('invalid official geometry or PBR attributes')
    device = torch.device('cuda')
    render_vertices = torch.from_numpy(vertices).to(device)
    render_faces = torch.from_numpy(faces).to(device)
    simplifier = ('fast' if args.simplifier == 'auto' and args.render_face_budget >= 1_000_000
                  else 'native' if args.simplifier == 'auto' else args.simplifier)
    if args.pre_simplify and len(faces) > args.render_face_budget:
        if simplifier == 'fast':
            sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor-fast-simplification'))
            from fast_simplification import simplify
            decimated_vertices, decimated_faces = simplify(
                vertices, faces, target_count=args.render_face_budget, agg=7)
            if len(decimated_faces) > args.render_face_budget:
                raise ValueError(f'fast simplifier produced {len(decimated_faces)} faces '
                                 f'for budget {args.render_face_budget}')
            render_vertices = torch.from_numpy(
                np.asarray(decimated_vertices, dtype=np.float32)).to(device)
            render_faces = torch.from_numpy(
                np.asarray(decimated_faces, dtype=np.int32)).to(device)
        else:
            from pixal3d.representations import Mesh
            render_mesh = Mesh(render_vertices, render_faces)
            render_mesh.simplify(target=args.render_face_budget)
            render_vertices, render_faces = render_mesh.vertices, render_mesh.faces
    original_cumesh = None
    if simplifier == 'fast':
        import cumesh
        original_cumesh = cumesh.CuMesh

        class PreserveTopologyMesh(original_cumesh):
            def simplify(self, *unused_args, **unused_kwargs):
                return None

            def fill_holes(self, *unused_args, **unused_kwargs):
                return None

            def remove_duplicate_faces(self, *unused_args, **unused_kwargs):
                return None

            def repair_non_manifold_edges(self, *unused_args, **unused_kwargs):
                return None

            def remove_small_connected_components(self, *unused_args, **unused_kwargs):
                return None

            def unify_face_orientations(self, *unused_args, **unused_kwargs):
                return None

        cumesh.CuMesh = PreserveTopologyMesh
    try:
        glb = o_voxel.postprocess.to_glb(
            vertices=render_vertices,
            faces=render_faces,
            attr_volume=torch.from_numpy(attrs).to(device),
            coords=torch.from_numpy(coords).to(device),
            attr_layout={'base_color': slice(0, 3), 'metallic': slice(3, 4),
                         'roughness': slice(4, 5), 'alpha': slice(5, 6)},
            aabb=[[-.5, -.5, -.5], [.5, .5, .5]], grid_size=1024,
            decimation_target=args.render_face_budget, texture_size=args.texture_size,
            remesh=False, use_tqdm=False)
    finally:
        if original_cumesh is not None:
            cumesh.CuMesh = original_cumesh
    args.output.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(args.output), extension_webp=True)
    exported_vertices, _ = read_mesh(args.output)
    original_world = transform_vertices(vertices, to_world)
    exported_world = transform_vertices(exported_vertices, to_world)
    rng = np.random.default_rng(42017)
    original_probe = original_world[rng.choice(len(original_world), min(4096, len(original_world)), replace=False)]
    exported_probe = exported_world[rng.choice(len(exported_world), min(4096, len(exported_world)), replace=False)]
    distances = [cKDTree(exported_world).query(original_probe, workers=1)[0],
                 cKDTree(original_world).query(exported_probe, workers=1)[0]]
    geometry_q99 = float(max(np.quantile(distances[0], .99), np.quantile(distances[1], .99)))
    if not np.isfinite(geometry_q99) or geometry_q99 > .01:
        args.output.unlink(missing_ok=True)
        raise ValueError(f'GLB/untouched metric mesh differ in world space: {geometry_q99}')
    result = {
        'status': 'PASS_OFFICIAL_RGB1_SCALE_UNGIVEN_RENDER_EXPORT',
        'sample_id': sid, 'prediction_sha256': generation['prediction_sha256'],
        'glb': str(args.output), 'glb_sha256': sha(args.output),
        'to_world': to_world.tolist(), 'native_scale': 1.0,
        'raw_surface_faces': len(faces),
        'pre_simplified_render_faces': int(len(render_faces)),
        'pre_simplify': bool(args.pre_simplify),
        'simplifier': simplifier if args.pre_simplify else 'none',
        'to_glb_cleanup': 'preserve_topology' if simplifier == 'fast' else 'standard',
        'render_face_budget': args.render_face_budget,
        'render_faces': len(glb.faces) if hasattr(glb, 'faces') else None,
        'render_geometry_registration_q99_world': geometry_q99,
        'surface_metrics_use_untouched_prediction': True,
        'GT_fitting_or_alignment': False,
        'texture_size': args.texture_size, 'remesh': False,
    }
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
