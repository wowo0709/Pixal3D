import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.deformnext_retargeting import historical_region_authority as authority
from experiments.deformnext_retargeting import cache_current_regions as current


def test_all_seven_native_support_ray_and_pool_functions_match_original_AST():
    path = Path(authority.__file__)
    expected = json.loads(path.with_name('historical_region_authority_sources.json').read_text())['functions']
    nodes = {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
    assert len(nodes) == len(expected) == 7
    for row in expected:
        assert hashlib.sha256(ast.dump(nodes[row['name']], include_attributes=False).encode()).hexdigest() == row['AST_sha256']
    assert authority.pool_support(np.array([[1, 2, 3], [1, 2, 2]], np.int32),
        source_resolution=64, target_resolution=32).tolist() == [[0, 1, 1]]


def test_outside_source_points_cannot_become_false_boundary_supervision(monkeypatch):
    monkeypatch.setattr(current, 'world_to_target_anchor_points', lambda points, *args, **kwargs: points)
    coords, dropped = current.visible_coords(np.array([[.6, 0, 0], [-.6, 0, 0], [0, 0, 0], [.5, 0, 0]], np.float32),
        {'transform_matrix': [np.eye(4)], 'mesh_scale': 1.})
    assert coords == {(32, 32, 32)}
    assert dropped == 3


def test_conservative_cube_free_label_never_contradicts_target_support():
    samples = np.zeros((3, 9, 3), np.float32)
    samples[:, :, 2] = -.5
    result = authority.classify_projected_voxel_cubes(samples,
        target_support=np.array([True, False, True]), visible_target_support=np.array([False, False, True]),
        raster_depth=np.zeros((2, 2), np.float32), raster_mask=np.ones((2, 2), bool), depth_tolerance=1e-4)
    assert result.tolist() == [authority.UNKNOWN, authority.FREE, authority.SURFACE]


def test_target_labels_keep_actual_SS_input_when_direct_mesh_misses_adjacent_cells():
    physical = np.array([[x, y, 0] for x in range(10) for y in range(10)], np.uint8)
    saved = np.concatenate([physical, [[10, 0, 0]]]).astype(np.uint8)
    locked, audit = current.lock_target_authority(physical, saved, hashlib.sha256(saved.tobytes()).hexdigest())
    assert current.coordinate_set(locked) == current.coordinate_set(saved)
    assert audit['encoder_only_target_voxels'] == 1
    assert audit['max_symmetric_Chebyshev_voxel_distance'] == 1
    with pytest.raises(ValueError, match='source hash'):
        current.lock_target_authority(physical, saved, 'wrong')
    moved = saved + 10
    with pytest.raises(ValueError, match='registration differs'):
        current.lock_target_authority(physical, moved, hashlib.sha256(moved.tobytes()).hexdigest())
