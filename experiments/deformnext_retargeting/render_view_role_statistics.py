"""Separate edited-input and source-camera performance on a paired panel.

Camera9/10 references are deformed M1 images, not the original M0 source
inputs. Camera roles do not establish surface visibility or occlusion.
"""
from __future__ import annotations

import numpy as np

from .render_panel_statistics import MODELS, paired_panel_statistics


ROLES = {'edited_target_camera8': (8,),
         'source_cameras9_10_M1_reference': (9, 10),
         'camera9': (9,), 'camera10': (10,), 'all_three_cameras': (8, 9, 10)}


def view_role_statistics(records, metrics, *, draws=4000, seed=42017):
    """Average cameras within each deformation before source-cluster bootstrap."""
    if tuple(records) != MODELS:
        raise ValueError('ordered three-model observations required')
    ids = list(dict.fromkeys(row['sample_id'] for row in records['v1']))
    if not ids:
        raise ValueError('nonempty observations required')
    indexed = {}
    clusters = {}
    for model, rows in records.items():
        by_view = {}
        for row in rows:
            key = row['sample_id'], row['camera_id']
            if key in by_view or row['camera_id'] not in (8, 9, 10):
                raise ValueError('distinct declared camera observations required')
            if (not np.isfinite([row[m] for m in metrics]).all()
                    or not isinstance(row['source_cluster_id'], str)
                    or not row['source_cluster_id']):
                raise ValueError('finite metrics and explicit source cluster required')
            by_view[key] = row
            if clusters.setdefault(row['sample_id'], row['source_cluster_id']) != row['source_cluster_id']:
                raise ValueError('source cluster differs across cameras or models')
        if set(by_view) != {(sid, cid) for sid in ids for cid in (8, 9, 10)}:
            raise ValueError('complete paired three-camera coverage required')
        indexed[model] = by_view

    panels = {}
    for name, cameras in ROLES.items():
        panel = {}
        for model in MODELS:
            panel[model] = [dict(sample_id=sid, source_cluster_id=clusters[sid],
                **{m:float(np.mean([indexed[model][sid, cid][m] for cid in cameras]))
                   for m in metrics}) for sid in ids]
        panels[name] = panel
    interaction = {}
    for model in MODELS:
        target = panels['edited_target_camera8'][model]
        source = panels['source_cameras9_10_M1_reference'][model]
        interaction[model] = [dict(sample_id=a['sample_id'], source_cluster_id=a['source_cluster_id'],
            **{m:a[m]-b[m] for m in metrics}) for a, b in zip(target, source)]
    statistics = {name:paired_panel_statistics(panel, metrics, draws=draws, seed=seed)
                  for name, panel in panels.items()}
    statistics['target_minus_source_camera_interaction'] = paired_panel_statistics(
        interaction, metrics, draws=draws, seed=seed)
    return dict(roles=statistics, per_view=records,
        source_camera_reference='deformed M1;not original M0 source RGB',
        source_camera11_not_evaluated=True, spatial_occlusion_not_measured=True,
        target_RGB_pixel_identity_not_implied=True,
        camera_views_resampled_as_independent_assets=False,
        posthoc_diagnostic=True, confirmatory_evidence=False)
