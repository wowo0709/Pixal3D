import pytest
import torch

from pixal3d.pipelines.projection_aggregation import (
    ProjectionAggregationConfig,
    aggregate_global_features,
    aggregate_projected_features,
)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "invalid"}, "mode"),
        ({"alpha": -0.1}, "alpha"),
        ({"alpha": 1.1}, "alpha"),
        ({"temperature": 0.0}, "temperature"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"global_mode": "invalid"}, "global_mode"),
    ],
)
def test_projection_aggregation_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ProjectionAggregationConfig(**kwargs)


def test_mean_matches_fp32_reference_and_preserves_dtype():
    features = torch.tensor(
        [
            [[1.0, 3.0, 10.0, 12.0], [2.0, 4.0, 20.0, 22.0]],
            [[5.0, 7.0, 14.0, 16.0], [6.0, 8.0, 24.0, 26.0]],
        ],
        dtype=torch.bfloat16,
    )
    fused, diagnostics = aggregate_projected_features(
        features,
        ProjectionAggregationConfig(mode="mean"),
    )
    expected = features.float().mean(dim=0).to(torch.bfloat16)
    assert torch.equal(fused, expected)
    assert fused.dtype == features.dtype
    torch.testing.assert_close(
        diagnostics.weights,
        torch.full((2, 2), 0.5),
        rtol=0,
        atol=0,
    )


def test_alpha_zero_consensus_is_exact_equal_mean():
    generator = torch.Generator().manual_seed(20260731)
    features = torch.randn(4, 7, 8, generator=generator).to(torch.bfloat16)
    mean, _ = aggregate_projected_features(
        features,
        ProjectionAggregationConfig(mode="mean"),
    )
    residual, _ = aggregate_projected_features(
        features,
        ProjectionAggregationConfig(
            mode="consensus", alpha=0.0, temperature=0.2, chunk_size=3
        ),
    )
    assert torch.equal(residual, mean)


def test_alpha_zero_oracle_is_exact_equal_mean():
    generator = torch.Generator().manual_seed(20260731)
    features = torch.randn(4, 7, 8, generator=generator).to(torch.bfloat16)
    corruption = torch.randint(0, 2, (4, 7), generator=generator).float()
    mean, _ = aggregate_projected_features(
        features,
        ProjectionAggregationConfig(mode="mean"),
    )
    residual, _ = aggregate_projected_features(
        features,
        ProjectionAggregationConfig(mode="oracle", alpha=0.0, chunk_size=3),
        projected_corruption=corruption,
    )
    assert torch.equal(residual, mean)


def test_consensus_downweights_one_outlier_for_both_feature_halves():
    inlier = torch.tensor([1.0, 0.0, 1.0, 0.0])
    outlier = torch.tensor([0.0, 1.0, 0.0, 1.0])
    features = torch.stack([inlier, inlier, inlier, outlier])[:, None, :]
    _, diagnostics = aggregate_projected_features(
        features,
        ProjectionAggregationConfig(
            mode="consensus", alpha=1.0, temperature=0.05
        ),
    )
    assert diagnostics.weights[3, 0] < diagnostics.weights[:3, 0].min()
    torch.testing.assert_close(
        diagnostics.weights[:, 0].sum(),
        torch.tensor(1.0),
    )


def test_identical_and_k1_features_use_uniform_weights():
    identical = torch.ones(4, 3, 8)
    _, repeated = aggregate_projected_features(
        identical,
        ProjectionAggregationConfig(mode="consensus", alpha=1.0),
    )
    torch.testing.assert_close(repeated.weights, torch.full((4, 3), 0.25))

    one_view = torch.randn(1, 3, 8)
    fused, single = aggregate_projected_features(
        one_view,
        ProjectionAggregationConfig(mode="consensus", alpha=1.0),
    )
    assert torch.equal(fused, one_view[0])
    assert torch.equal(single.weights, torch.ones(1, 3))


def test_nonfinite_consensus_scores_fall_back_to_uniform_weights():
    features = torch.ones(4, 2, 8)
    features[3, :, 0] = float("nan")
    _, diagnostics = aggregate_projected_features(
        features,
        ProjectionAggregationConfig(mode="consensus", alpha=1.0),
    )
    torch.testing.assert_close(
        diagnostics.weights, torch.full((4, 2), 0.25)
    )


def test_global_features_default_to_fp32_arithmetic_mean():
    features = torch.arange(4 * 5 * 6).reshape(4, 5, 6).to(torch.bfloat16)
    weights = torch.tensor(
        [[0.7, 0.7], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]
    )
    actual = aggregate_global_features(features, weights, mode="mean")
    expected = features.float().mean(dim=0, keepdim=True).to(torch.bfloat16)
    assert torch.equal(actual, expected)


def test_projection_weight_global_mode_reduces_voxel_weights_to_view_weights():
    features = torch.tensor([[[1.0]], [[3.0]]])
    weights = torch.tensor([[0.75, 0.75], [0.25, 0.25]])
    actual = aggregate_global_features(
        features, weights, mode="projection_weights"
    )
    torch.testing.assert_close(actual, torch.tensor([[[1.5]]]))
