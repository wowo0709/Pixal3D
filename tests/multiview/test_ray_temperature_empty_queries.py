"""Two empty auxiliary assets remain in the cohort instead of being dropped."""
import numpy as np
import pytest
import torch

from experiments.deformnext_retargeting.calibrate_ray_matching_temperature_v2 import (
    join_probabilities, asset_metrics, aggregate)


def test_empty_query_prediction_and_labels_are_kept_beside_supervised_assets():
    empty = join_probabilities([])
    assert empty.shape == (0, 64) and empty.dtype == torch.float32
    labels = dict(relation=np.empty(0, np.int64), exact=np.zeros((0, 64), bool),
                  near1=np.zeros((0, 64), bool))
    row = asset_metrics(dict(sample_id='empty', source_cluster_id='a'), labels, empty)
    assert row['observable_rays'] == row['observable_relations'] == 0
    p = torch.zeros(1, 64)
    p[0, :2] = torch.tensor([.6, .4])
    positive = np.zeros((1, 64), bool)
    positive[0, 0] = True
    other = asset_metrics(dict(sample_id='valid', source_cluster_id='b'),
        dict(relation=np.array([0]), exact=positive, near1=positive), p)
    summary = aggregate([row, other])
    assert summary['assets'] == 2 and summary['assets_without_native_queries'] == 1
    assert summary['observable_rays'] == summary['observable_relations'] == 1
    assert summary['near1']['rays']['mean_probability_mass'] == pytest.approx(.6)
