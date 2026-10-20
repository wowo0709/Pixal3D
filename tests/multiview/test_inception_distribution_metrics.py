import numpy as np
import pytest

pytest.importorskip('torch_fidelity')

from experiments.deformnext_retargeting.inception_distribution_metrics import official_distribution_metrics


def test_fid_translation_has_known_squared_distance_and_kid_is_reproducible():
    reference = np.random.default_rng(4).normal(size=(32, 4))
    prediction = reference + .25
    first = official_distribution_metrics(prediction, reference, subset_size=16, subsets=8)
    second = official_distribution_metrics(prediction, reference, subset_size=16, subsets=8)
    assert first == second
    assert first['frechet_inception_distance'] == pytest.approx(4*.25**2, abs=1e-12)
    assert first['images_per_pool'] == 32
    assert first['statistic_dtype'] == 'float64'


def test_unbiased_self_kid_can_be_negative_and_must_not_be_clamped():
    # For these two vectors the cubic Gram matrix is[[1,1],[1,3.375]].
    # Within-pool off-diagonal means are1 and the cross-pool mean is1.59375.
    values = np.array([[0.,0.],[1.,0.]])
    result = official_distribution_metrics(values,values,subset_size=2,subsets=4)
    assert result['kernel_inception_distance_mean'] == -1.1875
    assert result['kernel_inception_distance_std'] == 0
    assert not result['unbiased_KID_clamped']


@pytest.mark.parametrize('problem', ['counts','nonfinite','subset'])
def test_no_silent_pool_dropping_or_subset_resizing(problem):
    a = np.ones((4,2));b = a.copy();size = 4
    if problem == 'counts':
        b = b[:3]
    elif problem == 'nonfinite':
        a[0,0] = np.nan
    else:
        size = 5
    with pytest.raises(ValueError):
        official_distribution_metrics(a,b,subset_size=size)
