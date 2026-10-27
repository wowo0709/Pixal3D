"""Bake one frozen GT-scale-free Ours prediction for the common 1024 renderer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha(path: Path) -> str:
    state = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            state.update(block)
    return state.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pixal3d-repo', type=Path, required=True)
    parser.add_argument('--benchmark-repo', type=Path, required=True)
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--shape-generation', type=Path, required=True)
    parser.add_argument('--texture-generation', type=Path)
    parser.add_argument('--anchor-evaluation', type=Path, required=True)
    parser.add_argument('--texture-arm', choices=('vanilla', 'top1-only'), required=True)
    parser.add_argument('--render-face-budget', type=int,
                        choices=(100_000, 250_000, 500_000, 1_000_000, 2_000_000,
                                 5_000_000), default=1_000_000)
    parser.add_argument('--simplifier', choices=('auto', 'native', 'fast'), default='auto')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('immutable render export already exists')
    if (args.texture_arm == 'top1-only') != bool(args.texture_generation):
        raise ValueError('Top-1 texture requires its own frozen generation receipt')
    sys.path.insert(0, str(args.pixal3d_repo.resolve()))
    sys.path.insert(0, str(args.benchmark_repo.resolve()))
    import torch
    import o_voxel
    from scipy.spatial import cKDTree
    from deformnext_benchmark.io import read_mesh
    from deformnext_benchmark.metrics import transform_vertices

    shape = json.loads(args.shape_generation.read_text())
    if (shape['status'] != 'PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION'
            or shape['actual_resolution'] != 1024
            or shape['object_scale_policy'] != 'image-estimated-v1'
            or shape['M1_derived_scale_supplied_to_generation']
            or shape['GT_geometry_or_coords_used_in_generation']
            or shape['source_M0_mesh_used_in_generation']):
        raise ValueError('frozen GT-scale-free 1024 shape generation required')
    if args.texture_generation:
        texture = json.loads(args.texture_generation.read_text())
        if (texture['status'] != 'PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE'
                or texture['sample_id'] != shape['sample_id']
                or texture['PBR_arm'] != 'top1-only'
                or not all(texture['geometry_array_byte_identity'].values())
                or texture['GT_geometry_or_coords_used_in_generation']):
            raise ValueError('frozen paired Top-1 texture receipt required')
    else:
        texture = None
        if args.prediction.parent != args.shape_generation.parent:
            raise ValueError('vanilla render must use the frozen Shape/PBR prediction')
    scale = float(shape['scale_used_in_generation'])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('invalid image-estimated mesh scale')
    with np.load(args.prediction, allow_pickle=False) as raw:
        vertices = raw['vertices'].copy()
        faces = raw['faces'].copy()
        coords = raw['coords'].copy()
        attrs = raw['attrs'].copy()
    if (vertices.ndim != 2 or vertices.shape[1] != 3
            or faces.ndim != 2 or faces.shape[1] != 3
            or coords.ndim != 2 or coords.shape[1] != 3
            or attrs.shape != (len(coords), 6)
            or not len(vertices) or not len(faces)
            or faces.min() < 0 or faces.max() >= len(vertices)
            or not np.isfinite(vertices).all() or not np.isfinite(attrs).all()):
        raise ValueError('invalid frozen geometry or native PBR volume')
    with np.load(args.anchor_evaluation, allow_pickle=False) as archived:
        rotation = archived['anchor_rotation'].copy()
    if (rotation.shape != (3, 3)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)):
        raise ValueError('camera-only anchor rotation required')
    to_world = np.eye(4)
    to_world[:3, :3] = rotation.T / scale
    device = torch.device('cuda')
    render_vertices = torch.from_numpy(vertices).to(device)
    render_faces = torch.from_numpy(faces).to(device)
    simplifier = ('fast' if args.simplifier == 'auto' and args.render_face_budget >= 1_000_000
                  else 'native' if args.simplifier == 'auto' else args.simplifier)
    if len(faces) > args.render_face_budget:
        if simplifier == 'fast':
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
    # The native GLB baker normally cleans and simplifies the mesh a second
    # time.  On difficult surfaces that step can remove visible geometry even
    # when the already simplified input is faithful.  Preserve its topology
    # for the fast-simplification fallback; UV unwrapping and PBR baking still
    # run through the same standard baker.
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
            vertices=render_vertices, faces=render_faces,
            attr_volume=torch.from_numpy(attrs).to(device),
            coords=torch.from_numpy(coords).to(device),
            attr_layout={'base_color': slice(0, 3), 'metallic': slice(3, 4),
                         'roughness': slice(4, 5), 'alpha': slice(5, 6)},
            aabb=[[-.5, -.5, -.5], [.5, .5, .5]], grid_size=1024,
            decimation_target=args.render_face_budget, texture_size=2048,
            remesh=False, use_tqdm=False, verbose=args.verbose)
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
    distances = (cKDTree(exported_world).query(original_probe, workers=1)[0],
                 cKDTree(original_world).query(exported_probe, workers=1)[0])
    q99 = float(max(np.quantile(d, .99) for d in distances))
    if not np.isfinite(q99) or q99 > .01:
        args.output.unlink(missing_ok=True)
        raise ValueError(f'export/untouched metric mesh differ in world space: {q99}')
    report = {'status': 'PASS_OURS_FROZEN_GT_SCALE_FREE_1024_EXPORT',
              'sample_id': shape['sample_id'], 'texture_arm': args.texture_arm,
              'prediction_sha256': sha(args.prediction),
              'glb': str(args.output), 'glb_sha256': sha(args.output),
              'to_world': to_world.tolist(), 'image_estimated_scale': scale,
              'shape_generation_sha256': sha(args.shape_generation),
              'texture_generation_sha256': sha(args.texture_generation) if texture else None,
              'raw_surface_faces': len(faces),
              'render_face_budget': args.render_face_budget,
              'simplifier': simplifier if len(faces) > args.render_face_budget else 'none',
              'to_glb_cleanup': 'preserve_topology' if simplifier == 'fast' else 'standard',
              'pre_simplified_render_faces': int(len(render_faces)),
              'render_geometry_registration_q99_world': q99,
              'surface_metrics_use_untouched_prediction': True,
              'GT_fitting_or_alignment': False, 'texture_size': 2048,
              'remesh': False, 'training_updates': 0}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
