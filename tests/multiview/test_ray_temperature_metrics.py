"""Confidence calibration uses unique rays, retaining relation diagnostics."""
import numpy as np
import pytest
import torch

from experiments.deformnext_retargeting.calibrate_ray_matching_temperature import asset_metrics, aggregate


def test_disjoint_visible_views_share_one_ray_and_proposal_misses_remain_in_mass_denominator():
    p = torch.zeros(3, 64)
    p[0, :3] = torch.tensor([.6, .3, .1])
    p[1, :2] = torch.tensor([.2, .8])
    p[2, :2] = torch.tensor([.5, .5])
    labels = np.zeros((4, 64), bool)
    labels[0, 0], labels[1, 1], labels[3, 1] = True, True, True
    y = dict(relation=np.array([0, 0, 1, 2]), exact=labels, near1=labels)
    record = asset_metrics(dict(sample_id='a', source_cluster_id='s'), y, p)
    result = aggregate([record])
    assert (result['observable_relations'], result['observable_rays']) == (4, 3)
    assert result['near1']['relations']['mean_probability_mass'] == pytest.approx(1.4/4)
    assert result['near1']['rays']['mean_probability_mass'] == pytest.approx(1.4/3)
    assert result['near1']['relations']['top1_coverage'] == .25
    assert result['near1']['rays']['top1_coverage'] == pytest.approx(1/3)
    assert result['near1']['rays']['proposal64_coverage'] == pytest.approx(2/3)
    assert result['near1']['rays']['covered_mean_NLL'] == pytest.approx(
        -(np.log(.9)+np.log(.5))/2)
