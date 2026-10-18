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


def test_actual_SS_input_has_exact_source_hash_and_physical_mesh_cell_bound():
    vertices = np.array([[-.5, -.5, 0], [.5, -.5, 0], [.5, .5, 0], [-.5, .5, 0]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    saved = np.array([[32, 32, 32]], np.uint8)
    locked, audit = current.lock_target_authority(saved, hashlib.sha256(saved.tobytes()).hexdigest(), vertices, faces)
    assert current.coordinate_set(locked) == current.coordinate_set(saved)
    assert audit['GT_voxels_outside_physical_bound'] == 0
    assert audit['all_actual_SS_input_voxels_physically_registered']
    with pytest.raises(ValueError, match='source hash'):
        current.lock_target_authority(saved, 'wrong', vertices, faces)
    moved = saved + 10
    with pytest.raises(ValueError, match='physical M1 mesh cell bound'):
        current.lock_target_authority(moved, hashlib.sha256(moved.tobytes()).hexdigest(), vertices, faces)
