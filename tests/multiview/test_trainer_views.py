import torch

from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    anchor_camera_value,
    anchor_condition_image,
)


def test_anchor_condition_handles_single_and_multiview():
    single = torch.zeros(2, 3, 4, 4)
    multi = torch.arange(2 * 6 * 3 * 4 * 4).reshape(2, 6, 3, 4, 4)
    assert anchor_condition_image(single) is single
    assert torch.equal(anchor_condition_image(multi), multi[:, 0])


def test_anchor_camera_handles_scalar_and_view_vectors():
    single = torch.tensor([2.0, 3.0])
    multi = torch.tensor([[2.0, 4.0], [3.0, 5.0]])
    assert anchor_camera_value(single) is single
    assert torch.equal(anchor_camera_value(multi), torch.tensor([2.0, 3.0]))
