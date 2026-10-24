"""Freeze one generated rendering mesh for material-only PBR comparisons.

Native mesh simplification can produce different triangles on separate calls.
Raw generation archives remain authoritative for surface metrics. A separate
render-only input reuses the vanilla rendering mesh and each model's own PBR
attributes; the original renderer then has no simplification left to perform.
"""
import numpy as np

from .pbr_replay_artifacts import GEOMETRY_FIELDS, PREDICTION_FIELDS


FACE_BUDGET = 1_000_000


def _exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)


def render_only_prediction(model, common, frozen_mesh):
    """Return a derivative, without changing sampled geometry or material data."""
    if set(model) != set(PREDICTION_FIELDS) or set(common) != set(PREDICTION_FIELDS):
        raise ValueError('explicit original native prediction archives required')
    if any(not _exact(model[k], common[k]) for k in GEOMETRY_FIELDS):
        raise ValueError('PBR variants must share every raw geometry/support array')
    if set(frozen_mesh) != {'vertices', 'faces'}:
        raise ValueError('explicit vanilla render vertices and faces required')
    v, f = frozen_mesh['vertices'], frozen_mesh['faces']
    if (v.ndim != 2 or v.shape[1] != 3 or not len(v)
            or v.dtype != common['vertices'].dtype or v.dtype.kind != 'f'
            or not np.isfinite(v).all()
            or f.ndim != 2 or f.shape[1] != 3 or not 0 < len(f) <= FACE_BUDGET
            or f.dtype != common['faces'].dtype or f.dtype.kind not in 'iu'
            or f.min() < 0 or f.max() >= len(v)):
        raise ValueError('finite in-range native rendering mesh within face budget required')
    if len(common['faces']) <= FACE_BUDGET and any(
            not _exact(frozen_mesh[k], common[k]) for k in ('vertices', 'faces')):
        raise ValueError('already bounded original geometry must remain exact')
    return {**model, 'vertices': v, 'faces': f}


def verify_render_only_prediction(derived, model, frozen_mesh, rendered):
    """Reject material changes, a different mesh, or a second simplification."""
    if set(derived) != set(PREDICTION_FIELDS) or set(rendered) != {'vertices', 'faces'}:
        raise ValueError('explicit derivative and actual rendered mesh required')
    if any(not _exact(derived[k], model[k]) for k in PREDICTION_FIELDS
           if k not in ('vertices', 'faces')):
        raise ValueError('render-only preparation changed material or sparse support')
    if any(not _exact(derived[k], frozen_mesh[k])
           or not _exact(rendered[k], frozen_mesh[k]) for k in ('vertices', 'faces')):
        raise ValueError('actual rendering must use the exact frozen generated mesh')
    return dict(status='PASS_EXACT_FROZEN_GENERATED_PBR_RENDER_GEOMETRY',
                original_sampled_prediction_unchanged=True,
                original_surface_metrics_unchanged=True,
                model_material_and_sparse_support_exact=True,
                rendered_vertices_faces_exact=True,
                GT_geometry_used_for_prediction=False,
                new_generation_model_forwards=0,
                prediction_alpha_includes_learned_material_opacity=True)
