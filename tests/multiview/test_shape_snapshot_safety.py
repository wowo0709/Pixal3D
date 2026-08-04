import torch

from pixal3d.datasets.structured_latent_shape import SLatShapeVisMixin


def test_shape_snapshot_rejects_negative_face_indices_before_rendering():
    faces = torch.tensor([[0, 1, -1]], dtype=torch.long)

    assert not SLatShapeVisMixin._has_valid_face_indices(faces, num_vertices=3)


def test_shape_snapshot_accepts_in_range_face_indices():
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)

    assert SLatShapeVisMixin._has_valid_face_indices(faces, num_vertices=3)
