"""Official FID/KID reductions on explicitly paired, cached Inception features."""
from __future__ import annotations

import numpy as np


def official_distribution_metrics(prediction, reference, *, subset_size=1000,
                                  subsets=100, seed=42017):
    """Use float64 reductions; retain signed unbiased KID estimates unchanged.

    Low-dimensional fixtures exercise the mathematics only. Scientific callers
    must supply Inception-v3-compat2048 features and a complete declared pool.
    KID's subset standard deviation is not a source-cluster confidence interval.
    """
    import torch
    import torch_fidelity
    from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
    from torch_fidelity.metric_kid import kid_features_to_metric

    if torch_fidelity.__version__ != '0.3.0':
        raise ValueError('the pinned official torch-fidelity0.3.0 implementation is required')
    a, b = np.asarray(prediction), np.asarray(reference)
    if (a.ndim != 2 or b.shape != a.shape or min(a.shape) < 2
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('finite equal-count paired two-dimensional feature matrices required')
    if (isinstance(subset_size, bool) or not isinstance(subset_size, int)
            or not 2 <= subset_size <= len(a) or isinstance(subsets, bool)
            or not isinstance(subsets, int) or subsets < 1):
        raise ValueError('explicit valid KID subset size/count required;no automatic sample dropping')
    ta = torch.from_numpy(np.asarray(a, dtype=np.float64))
    tb = torch.from_numpy(np.asarray(b, dtype=np.float64))
    fid = fid_statistics_to_metric(fid_features_to_statistics(ta),
                                   fid_features_to_statistics(tb), verbose=False)
    kid = kid_features_to_metric(ta, tb, kid_subsets=subsets, kid_subset_size=subset_size,
        kid_degree=3, kid_gamma=None, kid_coef0=1., rng_seed=seed, verbose=False)
    values = {**fid, **kid}
    if not np.isfinite(list(values.values())).all():
        raise ValueError('official FID/KID returned nonfinite results')
    return dict(**values, images_per_pool=len(a), feature_dimension=a.shape[1],
        statistic_dtype='float64', kid_subsets=subsets, kid_subset_size=subset_size,
        seed=seed, unbiased_KID_clamped=False, KID_subset_std_is_not_CI=True,
        package='torch-fidelity0.3.0', unequal_pool_counts_allowed=False)
