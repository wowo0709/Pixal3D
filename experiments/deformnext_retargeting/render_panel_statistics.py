"""Paired rendered metrics with deformations kept inside source clusters."""
from __future__ import annotations

import numpy as np


MODELS = ('v1', 'retargeting', 'direct')
PAIRS = (('retargeting', 'v1'), ('direct', 'v1'), ('direct', 'retargeting'))


def paired_panel_statistics(records, metrics, *, draws=4000, seed=42017):
    """Report both deformation and equal-source means without changing units."""
    if tuple(records) != MODELS or not metrics or len(set(metrics)) != len(metrics):
        raise ValueError('ordered three-model records and distinct metrics required')
    if not isinstance(draws, int) or draws < 1:
        raise ValueError('positive bootstrap draw count required')
    ids = [row['sample_id'] for row in records['v1']]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('distinct nonempty paired sample IDs required')
    clusters = [row['source_cluster_id'] for row in records['v1']]
    if any(not isinstance(c, str) or not c for c in clusters):
        raise ValueError('source cluster identity is required')
    matrices = {}
    for model in MODELS:
        rows = records[model]
        if ([row['sample_id'] for row in rows] != ids
                or [row['source_cluster_id'] for row in rows] != clusters):
            raise ValueError('paired panel IDs, order or source clusters differ')
        values = np.asarray([[row[m] for m in metrics] for row in rows], dtype=np.float64)
        if values.shape != (len(ids), len(metrics)) or not np.isfinite(values).all():
            raise ValueError('complete finite paired metric values required')
        matrices[model] = values
    groups = [np.asarray([i for i, value in enumerate(clusters) if value == group])
              for group in sorted(set(clusters))]
    sizes = np.asarray([len(group) for group in groups])
    means = {}
    for model, values in matrices.items():
        means[model] = dict(
            deformation_weighted=dict(zip(metrics, values.mean(0).tolist())),
            source_weighted=dict(zip(metrics,
                np.stack([values[group].mean(0) for group in groups]).mean(0).tolist())))
    samples = np.random.default_rng(seed).integers(0, len(groups), size=(draws, len(groups)))
    contrasts = {}
    for candidate, baseline in PAIRS:
        delta = matrices[candidate] - matrices[baseline]
        grouped = np.stack([delta[group].sum(0) for group in groups])
        boot = grouped[samples].sum(1) / sizes[samples].sum(1)[:, None]
        source_boot = (grouped / sizes[:, None])[samples].mean(1)
        contrasts[candidate+'_minus_'+baseline] = dict(
            deformation_weighted_delta=dict(zip(metrics, delta.mean(0).tolist())),
            deformation_weighted_cluster_CI95={metric: np.quantile(boot[:, i], [.025, .975]).tolist()
                for i, metric in enumerate(metrics)},
            source_weighted_delta=dict(zip(metrics, (grouped/sizes[:, None]).mean(0).tolist())),
            source_weighted_cluster_CI95={metric: np.quantile(source_boot[:, i], [.025, .975]).tolist()
                for i, metric in enumerate(metrics)})
    return dict(assets=len(ids), source_clusters=len(groups), means=means, contrasts=contrasts,
                bootstrap_draws=draws, bootstrap_seed=seed)
