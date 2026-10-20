import numpy as np
import pytest

from experiments.deformnext_retargeting.pbr_replay_artifacts import (
    GEOMETRY_FIELDS, align_voxel_attributes, replayed_prediction,
)


def test_texture_join_preserves_archived_coordinate_order():
    coords = np.array([[7, 8, 9], [1, 2, 3], [11, 12, 13]], dtype=np.int32)
    attrs = np.arange(18, dtype=np.float32).reshape(3, 6)
    permutation = [2, 0, 1]
    result = align_voxel_attributes(coords, coords[permutation], attrs[permutation])
    np.testing.assert_array_equal(result, attrs)


@pytest.mark.parametrize('bad_coords', [
    [[7, 8, 9], [7, 8, 9]],
    [[7, 8, 9]],
    [[7, 8, 9], [1, 2, 4]],
    [[7, 8, 9], [1, 2, 3], [4, 5, 6]],
])
def test_texture_join_rejects_duplicate_missing_or_extra_coordinates(bad_coords):
    expected = np.array([[7, 8, 9], [1, 2, 3]], dtype=np.int32)
    decoded = np.array(bad_coords, dtype=np.int32)
    with pytest.raises(ValueError, match='coordinate'):
        align_voxel_attributes(expected, decoded, np.zeros((len(decoded), 6), np.float32))


def test_replay_preserves_geometry_bytes_and_replaces_only_texture():
    original = dict(vertices=np.arange(9, dtype=np.float32).reshape(3, 3),
        faces=np.array([[0, 1, 2]], np.int32),
        coords=np.array([[7, 8, 9], [1, 2, 3]], np.int32),
        origin=np.array([-.5, -.5, -.5], np.float32),
        voxel_size=np.asarray(1/1024), voxel_shape=np.array([1, 6, 1024, 1024, 1024]),
        shape_coords=np.array([[0, 0, 0, 0]], np.int32),
        shape_feats=np.ones((1, 32), np.float32),
        attrs=np.zeros((2, 6), np.float32), texture_feats=np.zeros((1, 32), np.float32))
    result = replayed_prediction(original, original['coords'][::-1],
        np.ones((2, 6), np.float32), np.full((1, 32), .5, np.float32))
    for field in GEOMETRY_FIELDS:
        assert result[field].dtype == original[field].dtype
        assert result[field].tobytes() == original[field].tobytes()
    assert np.all(result['attrs'] == 1)
    assert np.all(result['texture_feats'] == .5)
    assert not original['attrs'].any()
    assert not original['texture_feats'].any()
