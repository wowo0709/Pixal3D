"""Paired fixed-panel native SS evidence at an actual shared checkpoint step."""
import numpy as np


METRICS = ('voxel_iou', 'voxel_precision', 'voxel_recall', 'voxel_f1')


def count_metrics(tp, fp, fn):
    eps = np.finfo(np.float64).eps
    p, r = tp/(tp+fp+eps), tp/(tp+fn+eps)
    return dict(zip(METRICS, [tp/(tp+fp+fn+eps), p, r, 2*p*r/(p+r+eps)]))


def summarize_native_epoch(records, *, step, draws=4000, seed=42192):
    allowed = {'vanilla', 'R_top1_only', 'R_feature', 'R_raw_topk'}
    if (step not in {96, 192, 288, 384, 480} or draws < 1
            or 'vanilla' not in records or not 2 <= len(records) <= 4 or not set(records) <= allowed):
        raise ValueError('actual shared native checkpoint epoch and declared source contrasts required')
    first = records['vanilla']['quick_per_asset']
    identity = ['sample_id', 'source_cluster_id', 'seed', 'target_voxels_sha256',
                'target_voxel_shape', 'target_representation']
    ids, clusters = [r['sample_id'] for r in first], [r['source_cluster_id'] for r in first]
    if len(ids) != len(set(ids)) or len(ids) != 32 or any(not c for c in clusters):
        raise ValueError('fixed32 unique asset and actual source cluster identities required')
    matrices = {}
    for name, record in records.items():
        rows = record['quick_per_asset']
        if (record['step'] != step or record['weights'] != 'raw' or record['quick_panel_ids'] != ids
                or [[r[k] for k in identity] for r in rows] != [[r[k] for k in identity] for r in first]):
            raise ValueError('actual same-step assets, source, noise seed and GT must pair')
        counts = np.asarray([[r[k] for k in ['tp', 'fp', 'fn']] for r in rows], dtype=np.float64)
        if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
            raise ValueError('actual finite nonnegative integer occupancy counts required')
        expected = np.asarray([[v for v in count_metrics(*row).values()] for row in counts])
        actual = np.asarray([[r[k] for k in METRICS] for r in rows], dtype=np.float64)
        pooled = count_metrics(*counts.sum(0))
        if (not np.isfinite(actual).all() or not np.allclose(actual, expected, atol=1e-12, rtol=1e-12)
                or not all(np.isclose(record['quick_metrics'][k], pooled[k], atol=1e-12, rtol=1e-12) for k in METRICS)):
            raise ValueError('per-asset and pooled native metrics must reproduce actual occupancy counts')
        matrices[name] = actual
    groups = [np.flatnonzero(np.asarray(clusters) == c) for c in sorted(set(clusters))]
    sizes = np.asarray([len(g) for g in groups])
    resamples = np.random.default_rng(seed).integers(0, len(groups), (draws, len(groups)))
    pairs = [(n, 'vanilla') for n in records if n != 'vanilla']
    pairs += [(a, b) for a, b in [('R_feature', 'R_top1_only'), ('R_raw_topk', 'R_feature'),
                                ('R_raw_topk', 'R_top1_only')] if a in records and b in records]
    contrasts = {}
    for a, b in pairs:
        delta = matrices[a]-matrices[b]
        sums = np.stack([delta[g].sum(0) for g in groups])
        boot = sums[resamples].sum(1)/sizes[resamples].sum(1)[:, None]
        contrasts[a+'_minus_'+b] = dict(mean_asset_delta=dict(zip(METRICS, delta.mean(0).tolist())),
            source_cluster_CI95={m: np.quantile(boot[:, k], [.025, .975]).tolist() for k, m in enumerate(METRICS)})
    return dict(status='PASS_ACTUAL_IN_PROCESS_NATIVE_PAIRED_EPOCH_DEV32_EVIDENCE',
        step=step, assets=32, source_clusters=len(groups), generation_noises=1, training_seed=101,
        actual_occupancy_counts_reproduce_all_metrics=True,
        native_global_occupancy_metrics={n: r['quick_metrics'] for n, r in records.items()},
        mean_asset_metrics={n: dict(zip(METRICS, m.mean(0).tolist())) for n, m in matrices.items()},
        exploratory_unadjusted_contrasts=contrasts,
        independent_checkpoint_GPU_replay_not_proven_by_this_summary=True,
        final128_or_test508_not_evaluated=True, quality_gain_proven=False,
        automatic_training_restart_or_model_adoption=False, full_goal_complete=False)
