"""Compare edited-target and adapted-source cameras in closed cascade results.

This module reads existing renders and official perceptual scores on the CPU.
References use original source materials on the edited M1 geometry. Source
camera scores therefore measure adaptation with preserved material identity;
they do not compare a generated image against an undeformed M0 image.
"""
from pathlib import Path
import hashlib
import json

import numpy as np


IMAGE_METRICS = (
    'PSNR', 'SSIM', 'LPIPS', 'M_IoU', 'foreground_PSNR',
    'DreamSim', 'CLIP', 'DreamSim_GTcrop', 'CLIP_GTcrop',
)
ALL_METRICS = IMAGE_METRICS + ('CD', 'FScore')
ROLE_VIEWS = {'target8': (8,), 'source9_10': (9, 10), 'all8_9_10': (8, 9, 10)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f'Input changed while hashing: {path}')
    return digest.hexdigest()


class ClosedInputs:
    """Check each read against its actual file4 SHA256/size receipt."""

    def __init__(self, canonical_local_root):
        self.root = Path(canonical_local_root).resolve()
        self.receipts = {}
        self.inputs = {}

    def check(self, path, receipt):
        path, receipt = Path(path).resolve(), Path(receipt).resolve()
        relative = str(path.relative_to(self.root))
        if receipt not in self.receipts:
            record = json.loads(receipt.read_text())
            require(record['status'] == 'PASS' and record['local_equals_actual_file4_SHA256'],
                    f'Unverified canonical receipt: {receipt}')
            entries = record['files']
            require(len({r['relative_path'] for r in entries}) == len(entries),
                    f'Duplicate receipt paths: {receipt}')
            self.receipts[receipt] = {r['relative_path']: r for r in entries}
        entry = self.receipts[receipt].get(relative)
        require(entry is not None, f'Input absent from receipt: {path}')
        digest = file_sha256(path)
        require(entry['sha256'] == digest and entry['bytes'] == path.stat().st_size,
                f'Canonical input identity mismatch: {path}')
        self.inputs[str(path)] = digest
        return digest


def summarize_view_roles(records, contrasts, *, draws=4000, seed=42017):
    """Pair by asset and bootstrap source clusters, including role interaction.

    Each record contains sample_id, source_cluster_id and a views mapping for
    cameras 8/9/10. CD/FScore belong to the geometry, so appear only in all-view
    output. The interaction is a paired difference of target and source effects,
    not a comparison of whether their separate confidence intervals cross zero.
    """
    require(bool(records) and draws > 0, 'Nonempty models and positive draws required')
    names = list(records)
    first = records[names[0]]
    ids = [r['sample_id'] for r in first]
    require(bool(ids) and len(ids) == len(set(ids)), 'Unique nonempty asset IDs required')
    clusters = {r['sample_id']: r['source_cluster_id'] for r in first}
    index = {sid: i for i, sid in enumerate(ids)}
    groups = [np.asarray([index[sid] for sid in ids if clusters[sid] == cluster])
              for cluster in sorted(set(clusters.values()))]
    sizes = np.asarray([len(group) for group in groups])
    resamples = np.random.default_rng(seed).integers(0, len(groups), size=(draws, len(groups)))
    matrices = {role: {} for role in ROLE_VIEWS}
    per_asset = {role: {} for role in ROLE_VIEWS}
    for name, rows in records.items():
        require([r['sample_id'] for r in rows] == ids, f'Asset order differs: {name}')
        for row in rows:
            require(row['source_cluster_id'] == clusters[row['sample_id']],
                    f'Source cluster differs: {name}/{row["sample_id"]}')
            require(set(row['views']) == {8, 9, 10}, 'Exactly cameras 8/9/10 required')
        for role, camera_ids in ROLE_VIEWS.items():
            metrics = ALL_METRICS if role == 'all8_9_10' else IMAGE_METRICS
            values = []
            for row in rows:
                values_by_metric = {
                    metric: float(np.mean([row['views'][cid][metric] for cid in camera_ids]))
                    for metric in IMAGE_METRICS
                }
                if role == 'all8_9_10':
                    values_by_metric.update(CD=row['CD'], FScore=row['FScore'])
                values.append(dict(sample_id=row['sample_id'],
                                   source_cluster_id=row['source_cluster_id'], **values_by_metric))
            matrix = np.asarray([[r[m] for m in metrics] for r in values], dtype=np.float64)
            require(bool(np.isfinite(matrix).all()), f'Nonfinite metrics: {name}/{role}')
            matrices[role][name] = matrix
            per_asset[role][name] = values

    def means(values, metrics):
        return dict(
            deformation_weighted=dict(zip(metrics, values.mean(0).tolist())),
            source_weighted=dict(zip(metrics, np.stack([values[g].mean(0) for g in groups]).mean(0).tolist())),
        )

    def contrast(delta, metrics):
        grouped = np.stack([delta[g].sum(0) for g in groups])
        boot = grouped[resamples].sum(1) / sizes[resamples].sum(1)[:, None]
        source_boot = (grouped / sizes[:, None])[resamples].mean(1)
        return dict(
            deformation_weighted_delta=dict(zip(metrics, delta.mean(0).tolist())),
            deformation_weighted_cluster_CI95={m: np.quantile(boot[:, i], [.025, .975]).tolist()
                                              for i, m in enumerate(metrics)},
            source_weighted_delta=dict(zip(metrics, (grouped / sizes[:, None]).mean(0).tolist())),
            source_weighted_cluster_CI95={m: np.quantile(source_boot[:, i], [.025, .975]).tolist()
                                         for i, m in enumerate(metrics)},
        )

    labels = [f'{a}_minus_{b}' for a, b in contrasts]
    require(len(labels) == len(set(labels)), 'Duplicate contrast labels')
    require(all(a in records and b in records and a != b for a, b in contrasts),
            'Contrasts require two distinct, present models')
    summary, interaction = {}, {}
    for role in ROLE_VIEWS:
        metrics = ALL_METRICS if role == 'all8_9_10' else IMAGE_METRICS
        summary[role] = dict(
            means={name: means(values, metrics) for name, values in matrices[role].items()},
            contrasts={f'{a}_minus_{b}': contrast(matrices[role][a] - matrices[role][b], metrics)
                       for a, b in contrasts},
        )
    for a, b in contrasts:
        delta = ((matrices['target8'][a] - matrices['target8'][b])
                 - (matrices['source9_10'][a] - matrices['source9_10'][b]))
        interaction[f'{a}_minus_{b}'] = contrast(delta, IMAGE_METRICS)
    return dict(assets=len(ids), source_clusters=len(groups), models=names, roles=summary,
                interaction_target_minus_source_effect=interaction, per_asset=per_asset,
                source_cluster_bootstrap_draws=draws, bootstrap_seed=seed)


def existing_statistic_difference(summary, existing):
    """Require every existing all-view mean and interval to be reproduced."""
    maximum = 0.0
    allview = summary['roles']['all8_9_10']
    for name, fields in existing['means'].items():
        for field, metrics in fields.items():
            for metric, value in metrics.items():
                maximum = max(maximum, float(np.max(np.abs(
                    np.asarray(allview['means'][name][field][metric]) - np.asarray(value)))))
    for name, fields in existing['contrasts'].items():
        for field, metrics in fields.items():
            for metric, value in metrics.items():
                maximum = max(maximum, float(np.max(np.abs(
                    np.asarray(allview['contrasts'][name][field][metric]) - np.asarray(value)))))
    require(maximum <= 1e-12, f'Existing all-view statistics changed: {maximum}')
    return maximum


def evaluate_plan(plan):
    """Read a frozen model/metric plan; return results without training/inference."""
    closed = ClosedInputs(plan['canonical_local_root'])
    models = plan['models']
    require(len(models) == plan['expected_models'], 'Unexpected model count')
    perceptual = {}
    for path, receipt in plan['perceptuals'].items():
        closed.check(path, receipt)
        report = json.loads(Path(path).read_text())
        require(report['status'] == 'PASS_ACTUAL_OFFICIAL_DREAMSIM_CLIP_RENDER_EVALUATION'
                and report['full_panel']
                and report['GPU_sanity']['status'] == 'PASS_ACTUAL_GPU_OFFICIAL_METRIC_SANITY',
                f'Incomplete official perceptual scores: {path}')
        for row in report['per_view']:
            key = (row['arm'], row['sample_id'], row['camera_id'])
            require(key not in perceptual, f'Duplicate perceptual row: {key}')
            perceptual[key] = row
    expected_keys, references, records = set(), {}, {}
    for name, rows in models.items():
        require(len(rows) == plan['expected_assets'], f'Unexpected asset count: {name}')
        records[name] = []
        for row in rows:
            sid, cluster = row['sample_id'], row['source_cluster_id']
            require(row['generation'] in row['closed_inputs'] and row['render'] in row['closed_inputs'],
                    f'Unreceipted generation/render input: {name}/{sid}')
            for path, receipt in row['closed_inputs'].items():
                closed.check(path, receipt)
            generation = json.loads(Path(row['generation']).read_text())
            render = json.loads(Path(row['render']).read_text())
            require(generation['sample_id'] == sid and generation['actual_resolution'] == 1024
                    and generation['source_defined_normalization'] and not generation['per_output_alignment']
                    and not generation['GT_geometry_or_coords_used_in_generation']
                    and generation['GT_loaded_only_after_generation'], f'Generation contract differs: {name}/{sid}')
            require(generation['fixed_shape_models'] and not generation['additional_PBR_conditioner']
                    and generation['PBR_checkpoint'] == plan['fixed_PBR_checkpoint'],
                    f'Fixed downstream stages differ: {name}/{sid}')
            status = render['status']
            require(status in {'PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE',
                               'PASS_ACTUAL_SHARED_RENDERER_MATERIAL_REFERENCE'},
                    f'Incomplete render: {name}/{sid}')
            if status == 'PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE':
                require(render['same_reference_pixels_for_every_model'], 'Different fixed reference')
            else:
                require(render['shared_renderer_lighting_camera_AO_seed'] and render['GT_VAE_reconstruction_not_used'],
                        'Different original renderer reference')
            views = {v['camera_id']: v for v in render['per_view']}
            require(len(render['per_view']) == len(views) == 3 and set(views) == {8, 9, 10},
                    f'Duplicate or missing render camera: {name}/{sid}')
            values = {}
            for cid, view in views.items():
                key = (name, sid, cid)
                expected_keys.add(key)
                p = perceptual[key]
                require(p['source_cluster_id'] == cluster, 'Perceptual cluster differs')
                image_audit = p['image_audit']
                pixels = Path(row['render']).parent / f'camera{cid:02d}.npz'
                digest = closed.check(pixels, pixels.parent / 'file4-closure.json')
                require(p['shared_npz_SHA256'] == digest, 'Perceptual scores belong to different pixels')
                with np.load(pixels, allow_pickle=False) as arrays:
                    identity = (hashlib.sha256(arrays['GT_shared_renderer'].tobytes()).hexdigest(),
                                hashlib.sha256(arrays['GT_alpha'].tobytes()).hexdigest())
                require(identity == (image_audit['original_float_GT_SHA256'],
                                     image_audit['original_GT_alpha_SHA256']), 'GT pixel identity differs')
                if 'fixed_GT_RGB_SHA256' in view:
                    require(identity == (view['fixed_GT_RGB_SHA256'], view['fixed_GT_alpha_SHA256']),
                            'Render GT pixel identity differs')
                reference_identity = identity + (tuple(image_audit['GT_only_crop_box']),
                    image_audit['converted_image_SHA256']['reference_full'],
                    image_audit['converted_image_SHA256']['reference_GTcrop'])
                require(references.setdefault((sid, cid), reference_identity) == reference_identity,
                        f'Cross-model GT/crop differs: {name}/{sid}/{cid}')
                values[cid] = {metric: view[metric] for metric in IMAGE_METRICS[:5]}
                scores = p['scores']
                values[cid].update(DreamSim=scores['DreamSim_fullframe'],
                                   CLIP=scores['CLIP_cosine_fullframe'],
                                   DreamSim_GTcrop=scores['DreamSim_GT_foreground_crop'],
                                   CLIP_GTcrop=scores['CLIP_cosine_GT_foreground_crop'])
            records[name].append(dict(sample_id=sid, source_cluster_id=cluster, views=values,
                                      CD=generation['metrics']['chamfer_L1'],
                                      FScore=generation['metrics']['surface_FScore']))
    require(set(perceptual) == expected_keys, 'Unexpected or missing perceptual view rows')
    summary = summarize_view_roles(records, plan['contrasts'])
    closed.check(plan['existing_summary'], plan['existing_summary_receipt'])
    previous = json.loads(Path(plan['existing_summary']).read_text())
    require(previous['assets'] == summary['assets'] and previous['development_panel']
            and previous['fixed_Shape512_Shape1024_V1PBR762'], 'Existing summary scope differs')
    summary.update(
        status='PASS_ACTUAL_CASCADE_TARGET_SOURCE_VIEW_ROLE_COMPARISON',
        actual_completed_cases=len(models) * summary['assets'], actual_perceptual_views=len(perceptual),
        maximum_existing_allview_statistic_difference=existing_statistic_difference(summary, previous),
        source_training_seed=101, sampling_noise_repeats=1, input_SHA256=closed.inputs,
        reference='Identical original source material on deformed M1 geometry; not undeformed M0 RGB',
        target8_interpretation='Edited target RGB/material preservation at camera8',
        source9_10_interpretation='Adaptation to M1 edit with original source material at source cameras9/10',
        fixed_original_Shape512_Shape1024_and_V1PBR762=True, trained_PBR_stage_effect_not_measured=True,
        development_panel=True, intervals_posthoc_uncorrected_for_multiple_comparisons=True,
        independent_training_replication=False, no_extra_inference_GPU_or_optimizer_updates=True,
        model_adopted=False, full_goal_complete=False,
    )
    return summary
