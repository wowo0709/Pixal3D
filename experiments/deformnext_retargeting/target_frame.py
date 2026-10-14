"""Restore Blender's per-target dump normalization to the render world frame."""
from __future__ import annotations
import copy
import numpy as np

SCHEMA = 'deformnext-m1-render-world-v4'


def frame_metadata(vertices):
    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('invalid target vertices')
    minimum, maximum = vertices.min(0), vertices.max(0)
    extent = float((maximum - minimum).max())
    if extent <= 0:
        raise ValueError('degenerate target extent')
    return {'target_world_center': ((minimum + maximum) / 2).tolist(),
            'target_world_extent': extent, 'latent_geometry_schema': SCHEMA}


def restore_dump_world(dump, row):
    if row.get('latent_geometry_schema') != SCHEMA:
        raise ValueError('world-frame restoration requires explicit schema')
    extent = float(row['target_world_extent'])
    center = np.asarray(row['target_world_center'], dtype=np.float32)
    if extent <= 0 or not np.isfinite(extent) or center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('invalid dump inverse transform')
    output = copy.deepcopy(dump)
    for obj in output['objects']:
        obj['vertices'] = np.asarray(obj['vertices'], dtype=np.float32) * extent + center
    return output
