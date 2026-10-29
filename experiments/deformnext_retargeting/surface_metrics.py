"""Surface metrics in a caller-specified common frame; never fit an alignment."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def points(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not len(value) or not np.isfinite(value).all():
        raise ValueError('nonempty finite XYZ points required')
    return value


def sample_surface(vertices, faces, *, count=20000, seed):
    vertices = points(vertices)
    faces = np.asarray(faces)
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.dtype.kind not in 'iu' or faces.min()<0 or faces.max()>=len(vertices)
            or isinstance(count,bool) or not isinstance(count,int) or count<1):
        raise ValueError('nonempty indexed triangle mesh and explicit point count required')
    # Dense 1024 outputs can contain several million triangles.  Avoid the
    # (F,3,3) materialization: these edge buffers preserve the reference
    # arithmetic and RNG result with a much smaller peak working set.
    base = vertices[faces[:, 0]]
    edge1 = vertices[faces[:, 1]] - base
    edge2 = vertices[faces[:, 2]] - base
    del base
    area = np.linalg.norm(np.cross(edge1, edge2), axis=1)
    del edge1, edge2
    if not np.isfinite(area).all() or area.sum()<=0:
        raise ValueError('mesh has no finite positive surface area')
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(faces),size=count,p=area/area.sum())
    u,v = rng.random((2,count))
    root = np.sqrt(u)
    weights = np.stack([1-root,root*(1-v),root*v],axis=1)
    return (vertices[faces[chosen]]*weights[:,:,None]).sum(1)


def compare_surfaces(prediction, target, *, threshold=.02):
    prediction,target = points(prediction),points(target)
    if not np.isfinite(threshold) or threshold<=0:
        raise ValueError('positive predeclared distance threshold required')
    # One worker respects CPU quotas; no ICP, centering, or per-output scale fit.
    p_to_t = cKDTree(target).query(prediction,workers=1)[0]
    t_to_p = cKDTree(prediction).query(target,workers=1)[0]
    precision,recall = float((p_to_t<=threshold).mean()),float((t_to_p<=threshold).mean())
    return dict(chamfer_L1=float((p_to_t.mean()+t_to_p.mean())/2),
        chamfer_squared_L2=float(((p_to_t**2).mean()+(t_to_p**2).mean())/2),
        surface_precision=precision,surface_recall=recall,
        surface_FScore=2*precision*recall/(precision+recall) if precision+recall else 0.,
        threshold=float(threshold),prediction_samples=len(prediction),target_samples=len(target),
        per_output_alignment=False)
