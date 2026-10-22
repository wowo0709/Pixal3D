"""Render a texture replay using the original prediction's native mesh LOD.

PBR isolation arms share one generated geometry. Reuse its already computed
native LOD so a fresh nondeterministic simplification cannot change masks
between texture arms. Cameras, rasterization, materials and AO stay native.
"""
from contextlib import contextmanager
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np


def array_sha(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def validate_geometry(vertices, faces):
    if (vertices.dtype != np.float32 or faces.dtype != np.int32
            or vertices.ndim != 2 or faces.ndim != 2
            or vertices.shape[1:] != (3,) or faces.shape[1:] != (3,)
            or not len(vertices) or not len(faces) or not np.isfinite(vertices).all()
            or faces.min() < 0 or faces.max() >= len(vertices)
            or len(faces) > 1000000):
        raise ValueError('finite native predicted LOD in decoder frame required')


@contextmanager
def reuse_predicted_lod(mesh_class, vertices, faces, raw_hashes, events):
    """Replace only simplification; verify the exact original predicted mesh."""
    validate_geometry(vertices, faces)
    previous = mesh_class.__dict__.get('simplify')

    def frozen(mesh, target=1000000, verbose=False, options=None):
        if target != 1000000 or verbose or options not in (None, {}) or events:
            raise ValueError('one unchanged native million-face simplification call required')
        actual = {k: array_sha(getattr(mesh, k).detach().cpu().numpy())
                  for k in ('vertices', 'faces')}
        if actual != raw_hashes:
            raise ValueError('LOD belongs to a different raw predicted mesh')
        import torch
        mesh.vertices = torch.from_numpy(vertices).to(mesh.device)
        mesh.faces = torch.from_numpy(faces).to(mesh.device)
        events.append(dict(original_raw_predicted_mesh_exact=True,
                           existing_native_LOD_reused=True, GT_geometry_used=False))

    mesh_class.simplify = frozen
    try:
        yield
    finally:
        if previous is None:
            delattr(mesh_class, 'simplify')
        else:
            mesh_class.simplify = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.plan.read_text())
    from .cascade_view_roles import file_sha256 as sha
    from pixal3d.representations import MeshWithVoxel
    native = Path(spec['native_plan'])
    source = Path(__file__).with_name('render_cascade_artifact_v2.py')
    geometry = Path(spec['frozen_LOD'])
    for path, expected in [(native, spec['native_plan_SHA256']),
                           (source, spec['native_source_SHA256']),
                           (geometry, spec['frozen_LOD_SHA256'])]:
        if sha(path) != expected:
            raise ValueError('unchanged native plan/source/predicted LOD required')
    with np.load(geometry, allow_pickle=False) as arrays:
        vertices, faces = arrays['vertices'].copy(), arrays['faces'].copy()
    validate_geometry(vertices, faces)
    plan = json.loads(native.read_text())
    root = Path(plan['generation_root'])
    with np.load(root/'prediction.npz', allow_pickle=False) as arrays:
        raw = {k: array_sha(arrays[k]) for k in ('vertices', 'faces')}
        raw_faces = len(arrays['faces'])
    if raw != spec['raw_mesh_array_SHA256']:
        raise ValueError('texture replay changed the source predicted geometry')
    events, previous = [], sys.argv
    try:
        with reuse_predicted_lod(MeshWithVoxel, vertices, faces, raw, events):
            sys.argv = ['render_cascade_artifact_v2', '--plan', str(native)]
            runpy.run_module('experiments.deformnext_retargeting.render_cascade_artifact_v2',
                             run_name='__main__')
    finally:
        sys.argv = previous
    if len(events) != int(raw_faces > 1000000) or sha(geometry) != spec['frozen_LOD_SHA256']:
        raise ValueError('exactly the native simplification branch must reuse the LOD')
    output = Path(plan['output'])
    with np.load(output/'render-geometry.npz', allow_pickle=False) as actual:
        if not np.array_equal(actual['vertices'], vertices) or not np.array_equal(actual['faces'], faces):
            raise ValueError('actual rendered geometry differs from the original native LOD')
    audit = dict(status='PASS_ACTUAL_PBR_REPLAY_SAME_NATIVE_PREDICTED_LOD',
        frozen_LOD=str(geometry), frozen_LOD_SHA256=spec['frozen_LOD_SHA256'],
        raw_mesh_array_SHA256=raw, simplification_events=events,
        native_one_time_simplification_per_generated_geometry=True,
        different_texture_arms_do_not_rerun_simplification=True,
        original_rasterization_materials_cameras_AO_metrics_unchanged=True,
        geometry_metrics_still_use_raw_generated_surface=True,
        no_GT_geometry_or_alignment_used_for_LOD=True, shared_class_restored=True,
        source_preservation_or_quality_improvement_not_implied=True, full_goal_complete=False)
    (output/'PBR-replay-LOD.json').write_text(json.dumps(audit, indent=2)+'\n')


if __name__ == '__main__':
    main()
