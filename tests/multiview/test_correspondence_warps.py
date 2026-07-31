import pytest
import torch

from pixal3d.experiments.correspondence.warps import (
    affine_forward_field,
    affine_inverse_grid,
    invert_forward_field,
    normalized_to_pixel,
    pixel_to_normalized,
    smooth_random_forward_field,
    warp_affine,
    warp_with_forward_field,
)


def test_pixel_normalized_conversions_use_pixel_centers_and_are_inverse():
    """Would fail if conversion used corner alignment or lost invertibility."""
    points = torch.tensor(
        [[0.0, 0.0], [4.0, 2.0], [2.0, 1.0]],
        dtype=torch.float64,
    )
    expected = torch.tensor(
        [[-0.8, -2.0 / 3.0], [0.8, 2.0 / 3.0], [0.0, 0.0]],
        dtype=torch.float64,
    )

    normalized = pixel_to_normalized(points, height=3, width=5)
    reconstructed = normalized_to_pixel(normalized, height=3, width=5)

    assert normalized.dtype == points.dtype
    assert normalized.device == points.device
    torch.testing.assert_close(normalized, expected, rtol=0, atol=1e-15)
    torch.testing.assert_close(reconstructed, points, rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    ("conversion", "points", "height", "width", "error"),
    [
        (pixel_to_normalized, torch.zeros(2, dtype=torch.int64), 3, 5, "floating"),
        (normalized_to_pixel, torch.zeros(2, 3), 3, 5, "shape"),
        (pixel_to_normalized, torch.zeros(2), 0, 5, "height"),
        (normalized_to_pixel, torch.zeros(2), 3, True, "width"),
    ],
)
def test_pixel_normalized_conversions_reject_invalid_inputs(
    conversion, points, height, width, error
):
    """Would fail if malformed coordinates or sizes reached geometry code."""
    with pytest.raises((TypeError, ValueError), match=error):
        conversion(points, height=height, width=width)


def test_identity_forward_field_has_identity_inverse_and_preserves_image():
    """Would fail if identity introduced displacement or changed sampling."""
    image = torch.arange(25, dtype=torch.float32).reshape(1, 5, 5)
    affected_mask = torch.ones((5, 5), dtype=torch.bool)
    matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    expected_axis = torch.tensor([-0.8, -0.4, 0.0, 0.4, 0.8])
    expected_y, expected_x = torch.meshgrid(
        expected_axis, expected_axis, indexing="ij"
    )
    expected_grid = torch.stack((expected_x, expected_y), dim=-1)

    expected_invalid = torch.zeros((5, 5), dtype=torch.bool)

    forward_field = affine_forward_field(matrix, height=5, width=5)
    inverse_grid, invalid_mask = affine_inverse_grid(
        matrix, height=5, width=5
    )
    artifact = warp_affine(image, affected_mask, matrix)

    torch.testing.assert_close(
        forward_field, torch.zeros((5, 5, 2)), rtol=0, atol=0
    )
    torch.testing.assert_close(inverse_grid, expected_grid, rtol=0, atol=1e-7)
    assert torch.equal(invalid_mask, expected_invalid)
    torch.testing.assert_close(artifact.warped_image, image, rtol=0, atol=0)
    assert torch.equal(artifact.affected_mask, affected_mask)
    assert torch.equal(artifact.invalid_mask, invalid_mask)
    torch.testing.assert_close(
        artifact.inverse_grid_norm, expected_grid, rtol=0, atol=1e-7
    )


def test_translation_forward_field_moves_marker_and_records_forward_convention():
    """Would fail if a forward clean-to-destination field were used as a grid."""
    image = torch.zeros((1, 5, 5), dtype=torch.float32)
    image[0, 2, 1] = 1.0  # clean (x=1, y=2)
    affected_mask = torch.ones((5, 5), dtype=torch.bool)
    matrix = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]])
    expected_field = torch.empty((5, 5, 2))
    expected_field[..., 0] = 1.0
    expected_field[..., 1] = -1.0
    expected_invalid = torch.zeros((5, 5), dtype=torch.bool)
    expected_invalid[:, 0] = True
    expected_invalid[-1, :] = True
    expected_x = torch.tensor([-1.2, -0.8, -0.4, 0.0, 0.4])
    expected_y = torch.tensor([-0.4, 0.0, 0.4, 0.8, 1.2])
    expected_grid_y, expected_grid_x = torch.meshgrid(
        expected_y, expected_x, indexing="ij"
    )
    expected_grid = torch.stack((expected_grid_x, expected_grid_y), dim=-1)

    forward_field = affine_forward_field(matrix, height=5, width=5)
    inverse_grid, inverse_invalid = affine_inverse_grid(
        matrix, height=5, width=5
    )
    artifact = warp_affine(image, affected_mask, matrix)

    torch.testing.assert_close(forward_field, expected_field, rtol=0, atol=0)
    torch.testing.assert_close(inverse_grid, expected_grid, rtol=0, atol=1e-7)
    assert torch.equal(inverse_invalid, expected_invalid)
    torch.testing.assert_close(
        artifact.forward_field_px, expected_field, rtol=0, atol=0
    )
    torch.testing.assert_close(
        artifact.inverse_grid_norm, expected_grid, rtol=0, atol=1e-7
    )
    assert artifact.warped_image[0, 1, 2].item() == 1.0
    assert artifact.warped_image.sum().item() == 1.0
    assert torch.equal(artifact.invalid_mask, expected_invalid)
    assert not artifact.warped_image[:, expected_invalid].any()


def test_translation_field_inversion_matches_exact_affine_without_oscillation():
    """Would fail if zero padding made an invalid translation iterate oscillate."""
    matrix = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]])
    forward_field = torch.empty((5, 5, 2))
    forward_field[..., 0] = 1.0
    forward_field[..., 1] = -1.0
    expected_invalid = torch.zeros((5, 5), dtype=torch.bool)
    expected_invalid[:, 0] = True
    expected_invalid[-1, :] = True

    exact_grid, exact_invalid = affine_inverse_grid(
        matrix, height=5, width=5
    )
    numerical_grid, numerical_invalid = invert_forward_field(
        forward_field,
        iterations=12,
        convergence_tolerance_px=1e-3,
    )

    torch.testing.assert_close(numerical_grid, exact_grid, rtol=0, atol=1e-7)
    assert torch.equal(exact_invalid, expected_invalid)
    assert torch.equal(numerical_invalid, expected_invalid)


def test_affine_warp_uses_exact_inverse_for_nontranslation_transform():
    """Would fail if affine resampling were routed through fixed-point inversion."""
    image = torch.arange(25, dtype=torch.float32).reshape(1, 5, 5)
    affected_mask = torch.ones((5, 5), dtype=torch.bool)
    matrix = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

    artifact = warp_affine(image, affected_mask, matrix)

    assert not artifact.invalid_mask.any()
    assert artifact.warped_image[0, 2, 2].item() == 6.0
    assert artifact.warped_image[0, 4, 4].item() == 12.0
    torch.testing.assert_close(
        artifact.inverse_grid_norm[2, 2],
        torch.tensor([-0.4, -0.4]),
        rtol=0,
        atol=1e-7,
    )


def test_field_inversion_marks_in_bounds_nonconverged_sources_invalid():
    """Would fail if invalidity checked bounds but omitted fixed-point residual."""
    forward_field = torch.zeros((3, 3, 2))
    forward_field[:, -1, 0] = 1.0
    expected_invalid = torch.zeros((3, 3), dtype=torch.bool)
    expected_invalid[:, -1] = True

    inverse_grid, invalid_mask = invert_forward_field(
        forward_field,
        iterations=1,
        convergence_tolerance_px=0.1,
    )

    assert torch.isfinite(inverse_grid).all()
    assert torch.equal(invalid_mask, expected_invalid)


def test_identity_warp_preserves_pixels_outside_the_affected_destination():
    """Would fail if sampled pixels replaced unaffected destinations."""
    image = torch.arange(25, dtype=torch.float32).reshape(1, 5, 5)
    affected_mask = torch.zeros((5, 5), dtype=torch.bool)
    affected_mask[2, 2] = True
    forward_field = torch.empty((5, 5, 2))
    forward_field[..., 0] = 1.0
    forward_field[..., 1] = 0.0

    artifact = warp_with_forward_field(image, affected_mask, forward_field)

    assert torch.equal(
        artifact.warped_image[:, ~affected_mask], image[:, ~affected_mask]
    )
    assert artifact.warped_image[0, 2, 2].item() == image[0, 2, 1].item()


def test_identity_affine_inverse_rejects_a_singular_matrix():
    """Would fail if a non-invertible affine transform produced false geometry."""
    singular = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="singular"):
        affine_inverse_grid(singular, height=5, width=5)


@pytest.mark.parametrize(
    ("iterations", "tolerance", "error"),
    [
        (0, 1e-3, "iterations"),
        (12, 0.0, "tolerance"),
        (12, float("nan"), "tolerance"),
    ],
)
def test_identity_field_inversion_rejects_invalid_solver_controls(
    iterations, tolerance, error
):
    """Would fail if invalid controls silently yielded unreliable inverses."""
    with pytest.raises(ValueError, match=error):
        invert_forward_field(
            torch.zeros((5, 5, 2)),
            iterations=iterations,
            convergence_tolerance_px=tolerance,
        )


@pytest.mark.parametrize(
    ("image", "mask", "field", "error"),
    [
        (
            torch.zeros(1, 5, 5, dtype=torch.int64),
            torch.ones(5, 5, dtype=torch.bool),
            torch.zeros(5, 5, 2),
            "image",
        ),
        (
            torch.zeros(1, 5, 5),
            torch.ones(5, 5),
            torch.zeros(5, 5, 2),
            "bool",
        ),
        (
            torch.zeros(1, 5, 5),
            torch.ones(5, 5, dtype=torch.bool),
            torch.full((5, 5, 2), float("nan")),
            "finite",
        ),
        (
            torch.zeros(1, 5, 5),
            torch.ones(4, 5, dtype=torch.bool),
            torch.zeros(5, 5, 2),
            "shape",
        ),
    ],
)
def test_identity_warp_rejects_invalid_image_mask_or_field(
    image, mask, field, error
):
    """Would fail if malformed warp inputs silently produced an artifact."""
    with pytest.raises((TypeError, ValueError), match=error):
        warp_with_forward_field(image, mask, field)


def test_smooth_random_field_is_deterministic_masked_and_globally_bounded():
    """Would fail if randomness leaked globally or either field bound was missed."""
    affected_mask = torch.zeros((17, 19), dtype=torch.bool)
    affected_mask[2:-2, 3:-3] = True
    rng_before = torch.random.get_rng_state()

    first = smooth_random_forward_field(
        affected_mask,
        seed=20260731,
        max_displacement_px=1.25,
        coarse_size=5,
        blur_kernel_size=9,
        max_gradient=0.2,
    )
    second = smooth_random_forward_field(
        affected_mask,
        seed=20260731,
        max_displacement_px=1.25,
        coarse_size=5,
        blur_kernel_size=9,
        max_gradient=0.2,
    )
    rng_after = torch.random.get_rng_state()
    horizontal_gradient = torch.linalg.vector_norm(
        first[:, 1:] - first[:, :-1],
        dim=-1,
    )
    vertical_gradient = torch.linalg.vector_norm(
        first[1:] - first[:-1],
        dim=-1,
    )

    assert first.shape == (17, 19, 2)
    assert first.dtype == torch.float32
    assert first.device == affected_mask.device
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert torch.equal(rng_before, rng_after)
    assert torch.equal(
        first[~affected_mask],
        torch.zeros_like(first[~affected_mask]),
    )
    assert torch.linalg.vector_norm(first, dim=-1).max().item() <= 1.25 + 1e-5
    assert horizontal_gradient.max().item() <= 0.2 + 1e-5
    assert vertical_gradient.max().item() <= 0.2 + 1e-5


def test_smooth_random_field_is_exactly_zero_for_zero_displacement():
    """Would fail if normalization introduced values when displacement is zero."""
    affected_mask = torch.zeros((11, 11), dtype=torch.bool)
    affected_mask[2:-2, 2:-2] = True

    field = smooth_random_forward_field(
        affected_mask,
        seed=20260731,
        max_displacement_px=0.0,
    )

    assert torch.equal(field, torch.zeros((11, 11, 2), dtype=torch.float32))


@pytest.mark.parametrize(
    ("mask", "kwargs", "error"),
    [
        (
            torch.ones((11, 11), dtype=torch.bool),
            {"blur_kernel_size": 8},
            "blur_kernel_size",
        ),
        (
            torch.ones((11, 11), dtype=torch.bool),
            {"coarse_size": 1},
            "coarse_size",
        ),
        (
            torch.ones((11, 11), dtype=torch.bool),
            {"max_displacement_px": -0.1},
            "max_displacement",
        ),
        (
            torch.ones((11, 11), dtype=torch.bool),
            {"max_gradient": 0.0},
            "max_gradient",
        ),
        (
            torch.ones((11, 11), dtype=torch.bool),
            {"max_gradient": float("nan")},
            "max_gradient",
        ),
        (
            torch.zeros((11, 11), dtype=torch.bool),
            {},
            "empty",
        ),
    ],
)
def test_smooth_random_field_rejects_invalid_controls(mask, kwargs, error):
    """Would fail if malformed controls produced misleading controlled fields."""
    defaults = {
        "seed": 20260731,
        "max_displacement_px": 1.0,
    }
    defaults.update(kwargs)

    with pytest.raises((TypeError, ValueError), match=error):
        smooth_random_forward_field(mask, **defaults)


def test_smooth_field_warp_has_documented_artifact_and_preserves_unaffected_pixels(
):
    """Would fail if general-field warping changed unaffected destinations."""
    height, width = 17, 19
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    image = torch.stack((x, y), dim=0)
    affected_mask = torch.zeros((height, width), dtype=torch.bool)
    affected_mask[3:-3, 4:-4] = True
    field = smooth_random_forward_field(
        affected_mask,
        seed=20260731,
        max_displacement_px=0.15,
        max_gradient=0.05,
    )

    artifact = warp_with_forward_field(
        image,
        affected_mask,
        field,
        convergence_tolerance_px=1e-3,
    )

    assert artifact.warped_image.shape == (2, height, width)
    assert artifact.affected_mask.shape == (height, width)
    assert artifact.forward_field_px.shape == (height, width, 2)
    assert artifact.inverse_grid_norm.shape == (height, width, 2)
    assert artifact.invalid_mask.shape == (height, width)
    assert artifact.invalid_mask.dtype == torch.bool
    assert torch.isfinite(artifact.warped_image).all()
    assert torch.isfinite(artifact.forward_field_px).all()
    assert torch.isfinite(artifact.inverse_grid_norm).all()
    assert torch.equal(
        artifact.warped_image[:, ~affected_mask],
        image[:, ~affected_mask],
    )
