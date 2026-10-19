import numpy as np
import pytest

from experiments.deformnext_retargeting.inspect_score_prior_support import probability, query_statistics


def test_softening_keeps_top1_and_support_but_increases_effective_candidates():
    scores = np.array([[.99, .98, .97, -2.], [.8, .7, .69, .1]], dtype=np.float32)
    valid = np.array([[True, True, True, False], [True, True, True, True]])
    views = np.array([[0, 1, 2, 2], [0, 1, 1, 2]])
    hard, soft = [probability(scores, valid, t) for t in [.005, .05]]
    assert np.array_equal(hard.argmax(1), soft.argmax(1))
    assert np.all(hard[~valid] == 0) and np.all(soft[~valid] == 0)
    assert np.all(query_statistics(soft, views)['effective_candidates'] > query_statistics(hard, views)['effective_candidates'])
    assert query_statistics(soft, views)['effective_views'][0] > query_statistics(hard, views)['effective_views'][0]


def test_equal_candidate_weights_do_not_imply_distinct_view_coverage():
    weights = np.full((1, 64), 1 / 64)
    one_view = query_statistics(weights, np.zeros((1, 64), dtype=int))
    assert one_view['effective_candidates'][0] == pytest.approx(64)
    assert one_view['effective_views'][0] == pytest.approx(1)


def test_inactive_query_and_nonfinite_active_score_are_rejected():
    scores = np.array([[1., 2.]])
    with pytest.raises(ValueError): probability(scores, np.zeros((1, 2), dtype=bool), .05)
    with pytest.raises(ValueError): probability(scores * np.inf, np.ones((1, 2), dtype=bool), .05)


def test_masked_nonfinite_score_never_receives_mass():
    weights = probability(np.array([[1., np.nan]]), np.array([[True, False]]), .05)
    assert np.array_equal(weights, [[1., 0.]])
