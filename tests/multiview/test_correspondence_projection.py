import math

import pytest
import torch

from pixal3d.experiments.correspondence.projection import (
    oracle_reliability,
    project_points_for_views,
    sample_oracle_masks,
)
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    project_points_to_image_batch,
)


def test_project_points_for_views_matches_per_view_projector_and_preserves_invalid_coordinates():
    """Would fail if views were flattened incorrectly or validity altered samples."""
    points = torch.tensor(
        [[0.0, 0.0, -1.0], [0.25, -0.25, -2.0], [0.0, 0.0, 1.0]]
    )
    transforms = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
    fovs = torch.tensor([[math.pi / 2.0, math.pi / 3.0]])
    resolution = 8

    result = project_points_for_views(points, transforms, fovs, resolution=resolution)
    expected_pixels = []
    expected_depths = []
    expected_validity = []
    for view_index in range(2):
        pixels, depth, valid = project_points_to_image_batch(
            points,
            transforms[:, view_index],
            fovs[:, view_index],
            resolution,
        )
        expected_pixels.append(pixels)
        expected_depths.append(depth)
        expected_validity.append(valid)
    expected_pixels = torch.stack(expected_pixels, dim=1)
    expected_depths = torch.stack(expected_depths, dim=1)
    expected_validity = torch.stack(expected_validity, dim=1)
    expected_norm = (expected_pixels + 0.5) / resolution * 2.0 - 1.0

    assert result.pixel_coordinates.shape == (1, 2, 3, 2)
    assert result.normalized_coordinates.shape == (1, 2, 3, 2)
    assert result.depth.shape == (1, 2, 3)
    assert result.valid_mask.shape == (1, 2, 3)
    assert result.valid_mask.dtype == torch.bool
    torch.testing.assert_close(result.pixel_coordinates, expected_pixels, rtol=0, atol=0)
    torch.testing.assert_close(result.normalized_coordinates, expected_norm, rtol=0, atol=0)
    torch.testing.assert_close(result.depth, expected_depths, rtol=0, atol=0)
    assert torch.equal(result.valid_mask, expected_validity)
    assert not result.valid_mask[0, :, 2].any()
    assert torch.isfinite(result.normalized_coordinates[0, :, 2]).all()


@pytest.mark.parametrize(
    ("points", "transforms", "fovs", "resolution", "error"),
    [
        (torch.ones(1, 2, 4), torch.eye(4).reshape(1, 1, 4, 4), torch.ones(1, 1), 8, "points"),
        (torch.ones(2, 2, 3), torch.eye(4).reshape(1, 1, 4, 4), torch.ones(1, 1), 8, "matching B"),
        (torch.ones(2, 3), torch.ones(1, 1, 3, 3), torch.ones(1, 1), 8, "transforms"),
        (torch.ones(2, 3), torch.eye(4).reshape(1, 1, 4, 4), torch.tensor([[0.0]]), 8, "FOV"),
        (torch.ones(2, 3), torch.eye(4).reshape(1, 1, 4, 4), torch.ones(1, 1), 0, "resolution"),
    ],
)
def test_project_points_for_views_rejects_invalid_geometry_inputs(
    points, transforms, fovs, resolution, error
):
    """Would fail if malformed geometry reached the projection primitive."""
    with pytest.raises((TypeError, ValueError), match=error):
        project_points_for_views(points, transforms, fovs, resolution=resolution)


def test_sample_oracle_masks_uses_pixel_center_coordinates_and_zero_padding():
    """Would fail if sampling used corner alignment, border padding, or validity."""
    masks = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
    masks[0, 0, 1, 1] = 1.0
    coordinates = torch.tensor([[[[0.0, 0.0], [-2.0 / 3.0, -2.0 / 3.0], [2.0, 2.0]]]])
    validity = torch.tensor([[[False, True, False]]])
    validity_before = validity.clone()

    sampled = sample_oracle_masks(masks, coordinates)

    assert sampled.shape == (1, 1, 3)
    assert sampled.dtype == torch.float32
    torch.testing.assert_close(sampled, torch.tensor([[[1.0, 0.0, 0.0]]]), rtol=0, atol=0)
    assert torch.equal(validity, validity_before)


def test_sample_oracle_masks_samples_the_corner_of_an_all_one_mask():
    """Would fail if valid edge pixels were treated as zero-padded outside queries."""
    masks = torch.ones((1, 1, 3, 3), dtype=torch.float32)
    coordinates = torch.tensor([[[[-2.0 / 3.0, -2.0 / 3.0]]]])

    sampled = sample_oracle_masks(masks, coordinates)

    torch.testing.assert_close(sampled, torch.ones((1, 1, 1)), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("masks", "coordinates", "error"),
    [
        (torch.ones(1, 1, 3, 3, dtype=torch.int64), torch.zeros(1, 1, 1, 2), "floating"),
        (torch.full((1, 1, 3, 3), 1.1), torch.zeros(1, 1, 1, 2), "between 0 and 1"),
        (torch.ones(1, 1, 3, 3), torch.zeros(1, 2, 1, 2), "matching"),
        (torch.ones(1, 1, 3, 3), torch.tensor([[[[float("nan"), 0.0]]]]), "finite"),
    ],
)
def test_sample_oracle_masks_rejects_invalid_inputs(masks, coordinates, error):
    """Would fail if invalid mask data silently yielded misleading reliability."""
    with pytest.raises((TypeError, ValueError), match=error):
        sample_oracle_masks(masks, coordinates)


def test_oracle_reliability_is_one_minus_the_validated_mask():
    """Would fail if reliability inverted incorrectly or changed floating dtype."""
    sampled = torch.tensor([0.0, 0.25, 1.0], dtype=torch.float32)

    reliability = oracle_reliability(sampled)

    assert reliability.dtype == torch.float32
    torch.testing.assert_close(reliability, torch.tensor([1.0, 0.75, 0.0]), rtol=0, atol=0)


@pytest.mark.parametrize("sampled", [torch.tensor([-0.1]), torch.tensor([1.1])])
def test_oracle_reliability_rejects_values_outside_the_mask_range(sampled):
    """Would fail if invalid sampled mask values produced negative reliability."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        oracle_reliability(sampled)
