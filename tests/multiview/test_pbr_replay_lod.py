import numpy as np
import pytest
import torch

from experiments.deformnext_retargeting.render_pbr_replay import (
    array_sha, reuse_predicted_lod, validate_geometry,
)


def geometry():
    return np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], np.float32), np.array([[0, 1, 2]], np.int32)


def test_validated_lod_rejects_invalid_indices_and_nonfinite_vertices():
    vertices, faces = geometry()
    validate_geometry(vertices, faces)
    with pytest.raises(ValueError):
        validate_geometry(vertices, np.array([[0, 1, 3]], np.int32))
    vertices[0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_geometry(vertices, faces)


def test_source_mesh_is_verified_and_original_method_restores_after_forward_error():
    vertices, faces = geometry()
    class Mesh:
        def simplify(self, target=1000000):
            raise AssertionError('native simplifier must not run in this unit test')
    original = Mesh.simplify
    mesh = Mesh()
    mesh.vertices, mesh.faces = torch.from_numpy(vertices.copy()), torch.from_numpy(faces.copy())
    mesh.device = torch.device('cpu')
    raw = {k: array_sha(v) for k, v in [('vertices', vertices), ('faces', faces)]}
    events = []
    with pytest.raises(RuntimeError):
        with reuse_predicted_lod(Mesh, vertices*.5, faces, raw, events):
            mesh.simplify(target=1000000)
            assert np.array_equal(mesh.vertices.numpy(), vertices*.5)
            raise RuntimeError('downstream render failure')
    assert Mesh.simplify is original and len(events) == 1


def test_lod_cannot_be_transferred_to_another_raw_prediction():
    vertices, faces = geometry()
    class Mesh:
        pass
    mesh = Mesh()
    mesh.vertices, mesh.faces = torch.from_numpy(vertices+1), torch.from_numpy(faces)
    mesh.device = torch.device('cpu')
    raw = dict(vertices=array_sha(vertices), faces=array_sha(faces))
    with reuse_predicted_lod(Mesh, vertices, faces, raw, []):
        with pytest.raises(ValueError, match='different raw predicted mesh'):
            mesh.simplify()
    assert 'simplify' not in Mesh.__dict__
