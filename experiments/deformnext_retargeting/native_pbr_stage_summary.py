"""Read-only checks for completed PBR stages and their fixed-four diagnostics.

These are target-camera, GT-shape-conditioned trend checks. They never stand
in for generated-geometry or source-preservation evaluation.
"""
import math
import statistics


MODES = ('vanilla', 'top1-only', 'top1-plus-feature', 'raw-topk-plus-feature')
STEPS = (0, 380, 760, 1140, 1520, 1900)
IDENTITY = ('sample_id', 'source_cluster_id', 'conditioning_input_sha256', 'seed',
            'actual_noise_SHA256', 'actual_RGB_input_SHA256',
            'actual_GT_PBR_latent_SHA256', 'actual_GT_Shape_condition_SHA256',
            'actual_camera_SHA256')
METRICS = frozenset(('foreground_fraction', 'foreground_psnr',
    'foreground_base_color_psnr', 'crop_ssim', 'crop_lpips', 'render_psnr',
    'render_ssim', 'render_lpips', 'base_color_psnr', 'base_color_ssim',
    'gt_shape_conditioned_alpha_iou'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def aggregate(per_asset):
    require(len(per_asset) == 4, 'exactly four actual assets required')
    for row in per_asset:
        require(set(row) == METRICS and all(math.isfinite(v) for v in row.values())
                and 0 < row['foreground_fraction'] <= 1, 'finite actual fixed-four metrics required')
    result = {key: statistics.mean(row[key] for row in per_asset) for key in METRICS}
    for key in ('render_psnr', 'base_color_psnr'):
        result[key] = -10*math.log10(statistics.mean(10**(-row[key]/10) for row in per_asset))
    total = sum(row['foreground_fraction'] for row in per_asset)
    for key in ('foreground_psnr', 'foreground_base_color_psnr'):
        result[key] = -10*math.log10(sum(row['foreground_fraction']*10**(-row[key]/10)
                                       for row in per_asset)/total)
    return result


def validate_curve(rows):
    require([r['step'] for r in rows] == list(STEPS), 'six actual fixed epoch rows through1900 required')
    initial = rows[0]['quick_per_asset']
    require(len(initial) == 4 and len({r['sample_id'] for r in initial}) == 4,
            'four distinct fixed validation assets required')
    curve = {}
    for row in rows:
        require(row.get('weights') == 'raw' and row.get('fixed_GT_target_shape') is True
                and row.get('full_cascade') is False and row.get('source_preservation_metrics') is False
                and row.get('optimizer_updates_during_evaluation') == 0,
                'read-only raw GT-shape stage protocol required')
        assets = row['quick_per_asset']
        require(len(assets) == 4 and row['quick_panel_ids'] == [r['sample_id'] for r in initial],
                'actual fixed ordered four assets required')
        for left, right in zip(initial, assets):
            require(all(left[k] == right[k] for k in IDENTITY), 'fixed RGB/noise/GT/Shape/camera identity changed')
        pooled = aggregate([r['metrics'] for r in assets])
        require(set(row['quick_metrics']) == METRICS
                and all(math.isclose(v, row['quick_metrics'][k], abs_tol=1e-12, rel_tol=0)
                        for k, v in pooled.items()), 'actual per-asset metrics must reproduce pooled PSNR')
        curve[row['step']] = dict(pooled_metrics=row['quick_metrics'],
            mean_asset_metrics={k: statistics.mean(r['metrics'][k] for r in assets) for k in METRICS})
    return curve


def validate_completion(mode, active, terminal, wrapper, ranks, progress):
    require(mode in MODES, 'explicit native PBR mode required')
    require(type(active.get('PID')) is int and active['PID'] > 0
            and isinstance(active.get('birth'), (int,float)) and active['birth'] > 0
            and active.get('mode') == mode and active.get('PID') == terminal.get('PID')
            and active.get('birth') == terminal.get('birth') and terminal.get('returncode') == 0
            and active.get('physical_GPUs') == terminal.get('physical_GPUs') == ['0','1','2','3'],
            'actual same owned child PID/birth terminal0 and assigned four GPUs required')
    require(wrapper.get('status') == 'PASS_ACTUAL_FRESH_NATIVE_PBR1900_SCIENCE'
            and wrapper.get('mode') == mode and wrapper.get('source_multiview_step') == 16850
            and wrapper.get('fresh_optimizer') is True and wrapper.get('world_size') == 4
            and wrapper.get('global_batch') == 24 and wrapper.get('microbatch') == 6
            and wrapper.get('accumulation') == 1 and wrapper.get('actual_optimizer_updates') == 1900
            and wrapper.get('actual_five_epoch_NN_source_Adam_canonical_archives_verified') is True,
            'own actual fresh1900 science completion required')
    require(progress.get('stage') == 'completed' and progress.get('current') == 1900,
            'actual native stage must finish its fixed budget')
    source = 0 if mode == 'vanilla' else 10 if mode == 'top1-only' else 19
    require(len(ranks) == 4 and all(type(r.get('rank')) is int for r in ranks)
            and {r['rank'] for r in ranks} == {0,1,2,3},
            'four actual rank completions required')
    for row in ranks:
        require(row.get('status') == 'PASS_ACTUAL_ORIGINAL_PBR16850_FRESH_NATIVE1900_STAGE'
                and row.get('mode') == mode and row.get('training_seed') == 101
                and row.get('world_size') == 4 and row.get('global_batch') == 24
                and row.get('microbatch') == 6 and row.get('accumulation') == 1
                and row.get('source_multiview_step') == 16850 and row.get('fresh_optimizer') is True
                and row.get('actual_optimizer_updates') == 1900
                and row.get('actual_optimizer_states') == 360+source
                and row.get('actual_training_assets') == 9139 and row.get('actual_dataloader_batches') == 380
                and row.get('source_tensors') == source
                and all(row.get(k) is True for k in ('initial4_identity_before_first_update',
                    'frozen_base_matcher_and_anchors_exact', 'post_training_DDP_parameters_exact',
                    'stage_training_complete')), 'actual own full-rank fresh1900 optimizer/data/frozen/DDP proof required')


def compare_snapshots(records):
    require(tuple(records) == MODES[:len(records)] and 2 <= len(records) <= 4,
            'ordered completed vanilla plus available source-stage prefix required')
    baseline = records['vanilla']
    baseline_rows = baseline['validation_rows']
    curves = {}
    for mode, record in records.items():
        require(record.get('status') == 'PASS_ACTUAL_NATIVE_PBR1900_FIXED4_STAGE_SNAPSHOT'
                and record.get('mode') == mode, 'actual own completed stage snapshot required')
        validate_completion(mode, **record['completion_proof'])
        curves[mode] = validate_curve(record['validation_rows'])
        require(record['training_trace_digests'] == baseline['training_trace_digests']
                and set(record['training_trace_digests']) == {'0','1','2','3'}
                and all(is_digest(v) for v in record['training_trace_digests'].values()),
                'all1900 actual per-rank asset/noise/timestep/native-global traces must pair')
        pixels = record['GT_PNG_pixel_SHA256']
        ids = record['validation_rows'][0]['quick_panel_ids']
        require(set(pixels) == set(map(str, STEPS))
                and all(set(row) == set(ids) and all(is_digest(v) for v in row.values())
                        for row in pixels.values()), 'all actual GT pixels from six four-case panels required')
        require(record['GT_PNG_pixel_SHA256'] == baseline['GT_PNG_pixel_SHA256'],
                'actual GT panel pixels must be identical across models')
        for row, original in zip(record['validation_rows'], baseline_rows):
            for a, b in zip(row['quick_per_asset'], original['quick_per_asset']):
                require(all(a[k] == b[k] for k in IDENTITY), 'same-epoch models changed actual inputs or GT')
                if row['step'] == 0:
                    require(a['actual_prediction_PBR_latent_SHA256'] == b['actual_prediction_PBR_latent_SHA256']
                            and a['metrics'] == b['metrics'], 'all cold initial predictions must match the original base')
    pairs = [(mode, 'vanilla') for mode in records if mode != 'vanilla']
    pairs += [(a,b) for a,b in [('top1-plus-feature','top1-only'),
        ('raw-topk-plus-feature','top1-plus-feature'),('raw-topk-plus-feature','top1-only')]
        if a in records and b in records]
    contrasts = {}
    for left, right in pairs:
        per_epoch = {}
        for step in STEPS:
            a, b = curves[left][step], curves[right][step]
            per_epoch[step] = {kind: {k: a[kind][k]-b[kind][k] for k in METRICS}
                              for kind in ('pooled_metrics','mean_asset_metrics')}
        per_asset = []
        for a, b in zip(records[left]['validation_rows'][-1]['quick_per_asset'],
                        records[right]['validation_rows'][-1]['quick_per_asset']):
            per_asset.append(dict(sample_id=a['sample_id'], source_cluster_id=a['source_cluster_id'],
                delta={k:a['metrics'][k]-b['metrics'][k] for k in METRICS}))
        contrasts[left+'_minus_'+right] = dict(per_epoch=per_epoch, final_per_asset=per_asset)
    return dict(status='PASS_ACTUAL_NATIVE_PBR1900_MATCHED_FIXED4_TREND_COMPARISON',
        models=list(records), actual_optimizer_updates=1900, source_multiview_step=16850,
        assets=4, camera='target8', fixed_GT_shape=True, generated_geometry=False,
        pooled_PSNR_is_not_mean_asset_dB=True, curves=curves, contrasts=contrasts,
        full1900_per_rank_training_traces_equal=True, initial_actual_predictions_exact=True,
        source_preservation=False, full_cascade=False, inferential_CI_or_model_adoption=False,
        tiny_panel_not_final_module_effect_evidence=True, quality_gain_proven=False, full_goal_complete=False)
