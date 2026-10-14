import numpy as np
import pytest
from experiments.deformnext_retargeting.target_frame import frame_metadata, restore_dump_world


def test_inverse_blender_normalization_preserves_world_camera_projection():
    vertices = np.array([[-.3, -.1, -.2], [.4, .2, .25], [.12, -.07, .13]], dtype=np.float32)
    metadata = frame_metadata(vertices)
    extent = metadata['target_world_extent']
    center = np.asarray(metadata['target_world_center'])
    normalized = (vertices - center) / extent
    dump = {'objects': [{'vertices': normalized, 'faces': np.array([[0, 1, 2]]), 'uvs': np.ones((1, 3, 2))}]}
    restored = restore_dump_world(dump, metadata)
    np.testing.assert_allclose(restored['objects'][0]['vertices'], vertices, atol=1e-7)
    # An off-origin target must retain its shift in a fixed original camera.
    def project(points):
        return points[:, :2] / (2 - points[:, 2, None])
    np.testing.assert_allclose(project(restored['objects'][0]['vertices']), project(vertices), atol=1e-7)
    assert not np.allclose(project(normalized), project(vertices))
    np.testing.assert_array_equal(dump['objects'][0]['vertices'], normalized)
    np.testing.assert_array_equal(restored['objects'][0]['uvs'], dump['objects'][0]['uvs'])


def test_restoration_rejects_missing_schema_and_degenerate_target():
    with pytest.raises(ValueError, match='schema'):
        restore_dump_world({'objects': []}, {})
    with pytest.raises(ValueError, match='degenerate'):
        frame_metadata(np.zeros((3, 3)))


def test_sparse_loader_rejects_mixed_frame_and_scale(tmp_path):
    from pixal3d.datasets.deformnext import _latent_arrays
    from experiments.deformnext_retargeting.target_frame import SCHEMA
    path = tmp_path / 'latent.npz'
    coords = np.zeros((1, 3), dtype=np.uint8)
    feats = np.ones((1, 32), dtype=np.float16)
    np.savez(path, coords=coords, feats=feats)
    with pytest.raises(ValueError, match='schema mismatch'):
        _latent_arrays(path, SCHEMA, 1.2)
    np.savez(path, coords=coords, feats=feats, schema=np.asarray(SCHEMA), mesh_scale=1.1)
    with pytest.raises(ValueError, match='scale mismatch'):
        _latent_arrays(path, SCHEMA, 1.2)
    _latent_arrays(path, SCHEMA, 1.1)
