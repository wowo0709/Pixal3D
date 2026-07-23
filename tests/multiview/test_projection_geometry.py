import pytest
import torch

from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    ProjGrid,
    compute_multiview_projection_matrices,
)


RTOL = 1e-5
ATOL = 1e-5


def test_projection_matrices_keep_anchor_exact_and_non_anchor_relative():
    anchor = torch.tensor([
        [0.8660254, -0.5, 0.0, 1.25],
        [0.5, 0.8660254, 0.0, -2.5],
        [0.0, 0.0, 1.0, 3.75],
        [0.0, 0.0, 0.0, 1.0],
    ])
    second = torch.eye(4)
    second[:3, 3] = torch.tensor([-2.0, 1.5, 4.0])
    transforms = torch.stack([anchor, second])[None]
    distances = torch.tensor([[3.75, 4.72]])
    fixed = ProjGrid(2, 8).front_view_transform_matrix

    projection, relative = compute_multiview_projection_matrices(
        transforms, distances, fixed
    )

    expected_relative = torch.linalg.inv(anchor) @ second
    expected_fixed = fixed.clone()
    expected_fixed[1, 3] = -distances[0, 0]
    assert torch.equal(relative[:, 0], torch.eye(4)[None])
    assert torch.equal(projection[:, 0], expected_fixed[None])
    torch.testing.assert_close(
        relative[0, 1], expected_relative, rtol=RTOL, atol=ATOL
    )
    torch.testing.assert_close(
        projection[0, 1], expected_fixed @ expected_relative, rtol=RTOL, atol=ATOL
    )


def test_anchor_projection_equals_current_fixed_front_view():
    grid = ProjGrid(grid_resolution=2, image_resolution=8)
    transforms = torch.eye(4).reshape(1, 1, 4, 4)
    distances = torch.tensor([[2.5]])
    projection, relative = compute_multiview_projection_matrices(
        transforms, distances, grid.front_view_transform_matrix
    )
    expected = grid.front_view_transform_matrix.clone()
    expected[1, 3] = -2.5
    torch.testing.assert_close(relative, transforms, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(projection[0, 0], expected, rtol=RTOL, atol=ATOL)


def test_proj_grid_default_and_explicit_anchor_paths_match():
    grid = ProjGrid(grid_resolution=2, image_resolution=8)
    features = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)
    fov = torch.tensor([0.7])
    distance = torch.tensor([2.5])
    scale = torch.tensor([1.0])
    explicit = grid.front_view_transform_matrix[None].clone()
    explicit[:, 1, 3] = -distance
    expected = grid(features, fov, distance, scale)
    actual = grid(features, fov, distance, scale, explicit)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    ("transforms", "message"),
    [
        (torch.full((1, 2, 4, 4), float("nan")), "finite"),
        (torch.zeros(1, 2, 4, 4), "invertible"),
    ],
)
def test_projection_rejects_invalid_anchor_camera(transforms, message):
    with pytest.raises(ValueError, match=message):
        compute_multiview_projection_matrices(
            transforms, torch.ones(1, 2), torch.eye(4)
        )
