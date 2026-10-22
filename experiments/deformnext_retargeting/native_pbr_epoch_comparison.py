"""Paired fixed-four trends while a native PBR stage is still training.

Completed-stage checks remain stricter and separate. A published epoch is
never presented as completion of the five-epoch scientific run.
"""
import math
import statistics

from .native_pbr_stage_summary import (
    IDENTITY, METRICS, MODES, STEPS, aggregate, require, validate_curve,
)


def compare_epoch(source_rows, reference_rows, *, mode, reference_mode, step):
    require(mode in MODES[1:] and reference_mode in MODES[:MODES.index(mode)],
            'explicit source mode and earlier completed reference required')
    require(type(step) is int and step in STEPS, 'actual fixed epoch step required')
    reference = validate_curve(reference_rows)
    observed = [row['step'] for row in source_rows]
    require(observed == list(STEPS[:len(observed)]) and step in observed,
            'published actual source rows must form an ordered epoch prefix')
    prefix = source_rows[:STEPS.index(step)+1]
    require(len(prefix) == STEPS.index(step)+1, 'requested epoch has not been published')
    initial = reference_rows[0]['quick_per_asset']
    curves = {}
    for row in prefix:
        require(row.get('weights') == 'raw' and row.get('fixed_GT_target_shape') is True
                and row.get('full_cascade') is False
                and row.get('source_preservation_metrics') is False
                and row.get('optimizer_updates_during_evaluation') == 0,
                'actual read-only raw GT-shape-conditioned stage rows required')
        assets = row['quick_per_asset']
        require(len(assets) == 4 and row['quick_panel_ids'] == [a['sample_id'] for a in initial],
                'same four ordered actual validation assets required')
        for a, b in zip(assets, initial):
            require(all(a[k] == b[k] for k in IDENTITY),
                    'actual source/reference RGB/noise/GT/camera identity changed')
            if row['step'] == 0:
                require(a['actual_prediction_PBR_latent_SHA256'] == b['actual_prediction_PBR_latent_SHA256']
                        and a['metrics'] == b['metrics'], 'cold actual predictions must match')
        pooled = aggregate([a['metrics'] for a in assets])
        require(set(row['quick_metrics']) == METRICS and all(
            math.isclose(v, row['quick_metrics'][k], rel_tol=0, abs_tol=1e-12)
            for k, v in pooled.items()), 'per-asset measurements must reproduce pooled metrics')
        curves[row['step']] = dict(pooled_metrics=pooled,
            mean_asset_metrics={k: statistics.mean(a['metrics'][k] for a in assets) for k in METRICS})
    per_asset = [dict(sample_id=a['sample_id'], source_cluster_id=a['source_cluster_id'],
        delta={k:a['metrics'][k]-b['metrics'][k] for k in METRICS})
        for a,b in zip(prefix[-1]['quick_per_asset'],
                       reference_rows[STEPS.index(step)]['quick_per_asset'])]
    return dict(status='PASS_ACTUAL_NATIVE_PBR_MATCHED_EPOCH_FIXED4_TREND_ONLY',
        mode=mode, reference_mode=reference_mode, step=step, epoch=step//380,
        actual_observed_source_epoch_steps=observed, reported_source_epoch_steps=[r['step'] for r in prefix],
        assets=4, source_curve=curves, reference_curve_through_reported_epoch={k:reference[k] for k in curves},
        same_epoch_delta={kind:{k:curves[step][kind][k]-reference[step][kind][k] for k in METRICS}
                          for kind in ('pooled_metrics','mean_asset_metrics')},
        same_epoch_per_asset=per_asset, camera='target8', fixed_GT_shape=True,
        pooled_PSNR_is_not_mean_asset_dB=True, generation_or_optimizer_updates_by_comparison=0,
        full_stage_completion_proven=False, source_preservation=False, full_cascade=False,
        inferential_CI_or_model_adoption=False, quality_gain_proven=False, full_goal_complete=False)
