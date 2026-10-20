"""Exact sparse-coordinate joins for texture replay on archived predictions."""
from __future__ import annotations

import numpy as np


GEOMETRY_FIELDS = (
    'vertices', 'faces', 'coords', 'origin', 'voxel_size', 'voxel_shape',
    'shape_coords', 'shape_feats',
)
PREDICTION_FIELDS = (*GEOMETRY_FIELDS, 'attrs', 'texture_feats')


def _coordinates(value):
    value = np.asarray(value)
    if (value.ndim != 2 or value.shape[1] != 3 or not len(value)
            or value.dtype.kind not in 'iu' or (value < 0).any()
            or (value >= 1024).any()):
        raise ValueError('nonempty integer decoder coordinates in the1024 grid required')
    order = np.lexsort((value[:, 2], value[:, 1], value[:, 0]))
    ordered = value[order]
    if np.any(np.all(ordered[1:] == ordered[:-1], axis=1)):
        raise ValueError('duplicate sparse coordinates are not a valid texture join')
    return value, order, ordered


def align_voxel_attributes(expected_coords, decoded_coords, attributes):
    """Join by exact integer address; reject absent, added or duplicate voxels."""
    expected, expected_order, expected_sorted = _coordinates(expected_coords)
    decoded, decoded_order, decoded_sorted = _coordinates(decoded_coords)
    attributes = np.asarray(attributes)
    if (attributes.shape != (len(decoded), 6) or attributes.dtype.kind != 'f'
            or not np.isfinite(attributes).all()):
        raise ValueError('finite six-channel native PBR attributes required')
    if not np.array_equal(expected_sorted, decoded_sorted):
        raise ValueError('decoded voxel coordinate sets must exactly match archived geometry')
    result = np.empty((len(expected), 6), dtype=attributes.dtype)
    result[expected_order] = attributes[decoded_order]
    return result


def replayed_prediction(original, decoded_coords, attributes, texture_feats):
    """Replace texture only, preserving every raw geometry/support array byte."""
    if set(original) != set(PREDICTION_FIELDS):
        raise ValueError('the explicit native cascade prediction archive schema is required')
    texture_feats = np.asarray(texture_feats)
    if (texture_feats.shape != original['texture_feats'].shape
            or texture_feats.shape != original['shape_feats'].shape
            or texture_feats.ndim != 2 or texture_feats.shape[1] != 32
            or texture_feats.dtype.kind != 'f' or not np.isfinite(texture_feats).all()):
        raise ValueError('finite native32-channel texture latent on the archived support required')
    aligned = align_voxel_attributes(original['coords'], decoded_coords, attributes)
    if aligned.shape != original['attrs'].shape:
        raise ValueError('texture replay cannot change the native attribute layout')
    return {**{name: original[name] for name in GEOMETRY_FIELDS},
            'attrs': aligned, 'texture_feats': texture_feats}
