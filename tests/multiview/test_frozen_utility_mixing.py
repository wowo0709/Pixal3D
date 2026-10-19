"""Guard the probability and pooled-value aggregates used in saved-weight diagnosis."""
import math

import pytest
import torch

from experiments.deformnext_retargeting.inspect_frozen_utility_mixing import (
    contrast, distribution_stats, quantiles, summarize_contrasts,
)


def test_distribution_statistics_distinguish_top1_from_uniform_support():
    hard = torch.tensor([[1., 0., 0., 0.]])
    uniform = torch.full((3, 4), .25)
    assert distribution_stats(hard)['mean_effective_candidates'] == 1.
    assert distribution_stats(uniform)['mean_effective_candidates'] == 4.
    assert distribution_stats(uniform)['mean_entropy_nats'] == pytest.approx(math.log(4))


@pytest.mark.parametrize('probability', [
    torch.tensor([[.1, .2]]), torch.tensor([[-.1, 1.1]]),
    torch.tensor([[float('nan'), 0.]]), torch.tensor([.5, .5]),
])
def test_invalid_probabilities_are_rejected(probability):
    with pytest.raises(ValueError):
        distribution_stats(probability)


def test_pooled_value_error_uses_global_squared_sums_and_query_weighted_statistics():
    reference = torch.tensor([[1., 0.], [1., 0.]])
    candidate = torch.tensor([[0., 1.], [1., 0.]])
    values = torch.tensor([[[1., 0.], [3., 0.]], [[2., 0.], [7., 0.]]])
    rows = [
        {'active_queries': 1, 'contrasts': {'a': contrast(candidate[:1], reference[:1], values[:1])}},
        {'active_queries': 1, 'contrasts': {'a': contrast(candidate[1:], reference[1:], values[1:])}},
    ]
    result = summarize_contrasts(rows, 'a')
    assert result['argmax_disagreement_fraction'] == .5
    assert result['mean_total_variation'] == .5
    assert result['value_aggregate_relative_RMS'] == pytest.approx(math.sqrt(4 / 5))
    direct = contrast(candidate, reference, values)
    for key in ['argmax_disagreement_fraction', 'mean_total_variation', 'value_mean_cosine']:
        assert result[key] == pytest.approx(direct[key])
    assert result['value_aggregate_relative_RMS'] != 1.  # Mean of per-case relative RMS would be wrong.


def test_identical_weights_produce_no_probability_or_value_drift():
    probability = torch.tensor([[.4, .6]])
    result = contrast(probability, probability, torch.tensor([[[1., 3.], [2., 4.]]]))
    assert result['argmax_disagreement_fraction'] == 0
    assert result['mean_total_variation'] == 0
    assert result['mean_KL_candidate_vs_reference'] == 0
    assert result['value_squared_error_sum'] == 0


def test_logit_quantiles_reject_empty_and_nonfinite_measurements():
    for value in [torch.empty(0), torch.tensor([float('inf')])]:
        with pytest.raises(ValueError):
            quantiles(value)
    assert quantiles(torch.arange(5.))['median'] == 2.
