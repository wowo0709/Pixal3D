import math

import pytest
import torch

from pixal3d.experiments.correspondence.metrics import (
    branch_norms,
    feature_cosine_error,
    feature_l2_drift,
    weight_diagnostics,
)


def test_feature_cosine_error_uses_only_selected_voxels():
    """Would fail if masked voxels were included or cosine operands reversed."""
    reference = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    candidate = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    mask = torch.tensor([[False, True]])

    assert feature_cosine_error(candidate, reference, mask).item() == 1.0


def test_feature_l2_drift_is_mean_voxel_norm():
    """Would fail if the metric averaged channels instead of voxel norms."""
    reference = torch.zeros(1, 2, 2)
    candidate = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])

    assert feature_l2_drift(candidate, reference).item() == 2.5


def test_branch_norms_split_low_and_high_channel_halves():
    """Would fail if either branch used all channels or the wrong half."""
    features = torch.tensor([[[3.0, 4.0, 5.0, 12.0]]])

    low_norm, high_norm = branch_norms(features)

    assert low_norm.item() == 5.0
    assert high_norm.item() == 13.0


def test_feature_metrics_reject_an_empty_mask():
    """Would fail if an empty selection silently returned NaN or zero."""
    features = torch.ones(1, 2, 2)
    empty_mask = torch.tensor([[False, False]])

    with pytest.raises(ValueError, match="at least one"):
        feature_cosine_error(features, features, empty_mask)
    with pytest.raises(ValueError, match="at least one"):
        feature_l2_drift(features, features, empty_mask)


def test_feature_metrics_reject_shape_mismatch():
    """Would fail if tensors with incompatible feature geometry were broadcast."""
    with pytest.raises(ValueError, match="same shape"):
        feature_cosine_error(torch.ones(1, 2, 2), torch.ones(1, 3, 2))


def test_branch_norms_reject_odd_channel_count():
    """Would fail if the L/H boundary silently discarded a channel."""
    with pytest.raises(ValueError, match="even"):
        branch_norms(torch.ones(1, 2, 3))


def test_weight_diagnostics_rejects_weights_that_do_not_normalize_per_voxel():
    """Would fail if unnormalized view masses were accepted."""
    weights = torch.tensor([[[0.6], [0.6]]])

    with pytest.raises(ValueError, match="sum to one"):
        weight_diagnostics(weights)


def test_weight_diagnostics_reports_entropy_view_means_and_uniform_deviation():
    """Would fail if reductions used the wrong view or voxel axes."""
    weights = torch.tensor([[[1.0, 0.5], [0.0, 0.5]]])

    diagnostics = weight_diagnostics(weights)

    torch.testing.assert_close(
        diagnostics["entropy"], torch.tensor(math.log(2.0) / 2), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        diagnostics["normalized_entropy"], torch.tensor(0.5), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        diagnostics["mean_weight_per_view"], torch.tensor([0.75, 0.25]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        diagnostics["mean_abs_deviation_from_uniform"], torch.tensor(0.25),
        rtol=0,
        atol=0,
    )


def test_weight_diagnostics_reports_corrupt_view_mass_inside_and_outside_region():
    """Would fail if corrupt-view mass ignored or inverted the supplied region."""
    weights = torch.tensor([[[0.8, 0.2], [0.2, 0.8]]])
    corrupted_region_mask = torch.tensor([[True, False]])

    diagnostics = weight_diagnostics(
        weights,
        corrupted_region_mask=corrupted_region_mask,
        corrupted_view_index=1,
    )

    torch.testing.assert_close(
        diagnostics["corrupt_view_mass_inside_region"], torch.tensor(0.2),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        diagnostics["corrupt_view_mass_outside_region"], torch.tensor(0.8),
        rtol=0,
        atol=0,
    )
