import pytest
import torch

from pixal3d.experiments.correspondence.aggregation import (
    aggregate_consensus_projection,
    aggregate_consensus_projection_naive,
    residual_to_uniform_weights,
)


def test_alpha_zero_returns_uniform_weights():
    scores = torch.tensor([[[3.0], [1.0], [-2.0], [0.5]]])

    diagnostics = residual_to_uniform_weights(
        scores, alpha=0.0, temperature=0.2
    )

    torch.testing.assert_close(
        diagnostics.weights,
        torch.full((1, 4, 1), 0.25),
        rtol=0,
        atol=0,
    )


def test_nonfinite_score_row_uses_uniform_fallback():
    scores = torch.tensor([[[1.0], [float("nan")], [0.0], [-1.0]]])

    diagnostics = residual_to_uniform_weights(
        scores, alpha=1.0, temperature=0.2
    )

    assert diagnostics.fallback_mask.item()
    torch.testing.assert_close(
        diagnostics.weights,
        torch.full((1, 4, 1), 0.25),
        rtol=0,
        atol=0,
    )


def test_one_opposite_outlier_receives_less_weight_than_three_agreeing_views():
    good = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
    outlier = torch.tensor([[[-1.0, 0.0, -1.0, 0.0]]])
    stacked = torch.stack([good, good, good, outlier], dim=1)

    result, diagnostics = aggregate_consensus_projection_naive(
        stacked, alpha=1.0, temperature=0.2
    )

    assert diagnostics.weights[0, 3, 0] < diagnostics.weights[0, 0, 0]
    assert result[0, 0, 0] > 0
    assert result[0, 0, 2] > 0


def test_identical_views_produce_uniform_weights_and_input_feature():
    feature = torch.tensor([[[2.0, -1.0, 0.5, 4.0]]])
    stacked = feature[:, None].repeat(1, 4, 1, 1)

    result, diagnostics = aggregate_consensus_projection_naive(
        stacked, alpha=1.0, temperature=0.2
    )

    torch.testing.assert_close(result, feature, rtol=0, atol=0)
    torch.testing.assert_close(
        diagnostics.weights,
        torch.full((1, 4, 1), 0.25),
        rtol=0,
        atol=0,
    )


def test_naive_mixed_degenerate_rows_use_exact_uniform_fallback():
    stacked = torch.tensor(
        [
            [
                [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]],
                [[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]],
                [[1.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 1.0, 0.0]],
            ]
        ]
    )

    _, diagnostics = aggregate_consensus_projection_naive(
        stacked, alpha=1.0, temperature=0.2
    )

    assert diagnostics.weights.dtype == torch.float32
    torch.testing.assert_close(
        diagnostics.weights,
        torch.full((1, 3, 2), 1.0 / 3.0, dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    assert torch.equal(
        diagnostics.fallback_mask,
        torch.ones((1, 2), dtype=torch.bool),
    )


def test_chunked_mixed_degenerate_rows_use_exact_uniform_fallback():
    stacked = torch.tensor(
        [
            [
                [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]],
                [[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]],
                [[1.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 1.0, 0.0]],
            ]
        ]
    )

    _, diagnostics = aggregate_consensus_projection(
        [stacked[:, view_index] for view_index in range(3)],
        alpha=1.0,
        temperature=0.2,
        chunk_size=1,
        compute_device="cpu",
        output_device="cpu",
    )

    assert diagnostics.weights.dtype == torch.float32
    torch.testing.assert_close(
        diagnostics.weights,
        torch.full((1, 3, 2), 1.0 / 3.0, dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    assert torch.equal(
        diagnostics.fallback_mask,
        torch.ones((1, 2), dtype=torch.bool),
    )


def test_k1_returns_sole_feature_and_trivial_diagnostics():
    feature = torch.tensor([[[2.0, -1.0, 0.5, 4.0]]])

    result, diagnostics = aggregate_consensus_projection_naive(
        feature[:, None], alpha=0.5, temperature=0.2
    )

    torch.testing.assert_close(result, feature, rtol=0, atol=0)
    torch.testing.assert_close(diagnostics.scores, torch.zeros((1, 1, 1)))
    torch.testing.assert_close(diagnostics.weights, torch.ones((1, 1, 1)))
    torch.testing.assert_close(diagnostics.entropy, torch.zeros((1, 1)))
    assert not diagnostics.fallback_mask.item()


def test_naive_rejects_odd_channels():
    with pytest.raises(ValueError, match="even"):
        aggregate_consensus_projection_naive(
            torch.ones(1, 2, 3, 5), alpha=0.5, temperature=0.2
        )


def test_naive_rejects_nonfloating_features():
    with pytest.raises(TypeError, match="floating"):
        aggregate_consensus_projection_naive(
            torch.ones(1, 2, 3, 4, dtype=torch.int64), alpha=0.5, temperature=0.2
        )


def test_naive_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        aggregate_consensus_projection_naive(
            torch.ones(1, 2, 3, 4), alpha=1.1, temperature=0.2
        )


def test_naive_rejects_invalid_temperature():
    with pytest.raises(ValueError, match="temperature"):
        aggregate_consensus_projection_naive(
            torch.ones(1, 2, 3, 4), alpha=0.5, temperature=0.0
        )


def test_sequence_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        aggregate_consensus_projection(
            [torch.ones(1, 3, 4), torch.ones(1, 2, 4)],
            alpha=0.5,
            temperature=0.2,
            chunk_size=1,
            compute_device="cpu",
            output_device="cpu",
        )


def test_chunked_aggregation_matches_naive_reference():
    generator = torch.Generator().manual_seed(20260731)
    stacked = torch.randn(2, 4, 11, 8, generator=generator)
    expected, expected_diag = aggregate_consensus_projection_naive(
        stacked, alpha=0.35, temperature=0.17
    )

    actual, actual_diag = aggregate_consensus_projection(
        [stacked[:, i].clone() for i in range(4)],
        alpha=0.35,
        temperature=0.17,
        chunk_size=3,
        compute_device="cpu",
        output_device="cpu",
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_diag.scores, expected_diag.scores)
    torch.testing.assert_close(actual_diag.weights, expected_diag.weights)
    torch.testing.assert_close(actual_diag.entropy, expected_diag.entropy)
    assert torch.equal(actual_diag.fallback_mask, expected_diag.fallback_mask)


def test_chunked_aggregation_rejects_nonpositive_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        aggregate_consensus_projection(
            [torch.ones(1, 3, 4)],
            alpha=0.5,
            temperature=0.2,
            chunk_size=0,
            compute_device="cpu",
            output_device="cpu",
        )
