"""Evaluation-only deformation size in the existing source-bbox metric frame."""
import numpy as np


def surface_deformation_summary(source_vertices, source_faces, target_vertices,
                                *, count=20000, seed=20261022):
    """Measure paired vertex motion using source surface area, never target fit.

    RMS is the exact area integral of squared piecewise-linear displacement.
    Mean, quantiles and moved fractions use coupled source-surface Monte Carlo:
    each target point has the SAME face and barycentrics as its source point.
    Independently sampling source and target surfaces would measure a different
    quantity. Vertex density therefore does not determine the primary RMS.
    """
    source = np.asarray(source_vertices)
    target = np.asarray(target_vertices)
    faces = np.asarray(source_faces)
    if (source.ndim != 2 or source.shape[1:] != (3,) or len(source) < 3
            or target.shape != source.shape or source.dtype.kind != 'f'
            or target.dtype.kind != 'f' or not np.isfinite(source).all()
            or not np.isfinite(target).all() or faces.ndim != 2
            or faces.shape[1:] != (3,) or not len(faces)
            or faces.dtype.kind not in 'iu' or (faces < 0).any()
            or (faces >= len(source)).any() or type(count) is not int or count < 1
            or type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError('finite paired float vertices, valid source triangles and explicit MC count/seed required')
    # Compute extent before float64 conversion, matching the native source
    # normalization of stored float32 vertices; target is never re-centered.
    extent = float(np.ptp(source, axis=0).max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError('positive finite source bbox extent required')
    vertices = source.astype(np.float64)
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                   triangles[:, 2]-triangles[:, 0]), axis=-1)/2
    total_area = float(area.sum())
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError('positive finite source surface area required')
    displacement = (target.astype(np.float64)-vertices)/extent
    d = displacement[faces]
    squared = (np.square(d).sum(axis=(1, 2))
               + (d[:, 0]*d[:, 1]).sum(-1)
               + (d[:, 1]*d[:, 2]).sum(-1)
               + (d[:, 2]*d[:, 0]).sum(-1))/6
    rms = float(np.sqrt(max(float(area @ squared/total_area), 0.)))
    rng = np.random.default_rng(seed)
    sampled_faces = rng.choice(len(faces), size=count, p=area/total_area)
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, 0])
    bary = np.stack((1-root, root*(1-uv[:, 1]), root*uv[:, 1]), axis=-1)
    motion = np.linalg.norm((d[sampled_faces]*bary[..., None]).sum(axis=1), axis=-1)
    quantiles = np.quantile(motion, [.5, .9, .95, .99])
    return dict(schema='source-surface-correspondence-deformation-size-v1',
        normalization='source_bbox_max_extent; no target recentering/rescaling or alignment',
        vertices=len(vertices), source_faces=len(faces), source_bbox_extent=extent,
        source_surface_area=total_area, exact_area_RMS_displacement=rms,
        surface_MC_mean_displacement=float(motion.mean()),
        surface_MC_displacement_quantiles=dict(zip(['p50','p90','p95','p99'], quantiles.tolist())),
        surface_MC_fraction_above={str(t):float((motion > t).mean()) for t in [.005,.01,.02,.05,.1]},
        max_vertex_displacement=float(np.linalg.norm(displacement, axis=-1).max()),
        MC_samples=count, MC_seed=seed, shared_source_faces_and_barycentrics=True,
        displacement_is_not_Chamfer_distance_or_visibility=True)
