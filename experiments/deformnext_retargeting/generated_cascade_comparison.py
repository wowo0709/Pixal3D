"""Join original MV zero-shot and trained-PBR cascades without new inference.

These are deployment comparisons on a development panel. The original cascade
and PBR replay use different texture-noise stream protocols. Only the replay
isolation arms share geometry and initial texture noise; they support the
corresponding PBR-only comparison. No cross-protocol causal claim is made.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .cascade_view_roles import (
    ClosedInputs, IMAGE_METRICS, existing_statistic_difference,
    require, summarize_view_roles,
)


REPLAY_MODELS = ('full-v1', 'full-R', 'full-Direct', 'iso-R', 'iso-Direct')
ZERO_MODES = {'zero-source4': 'source-only', 'zero-edited-source3': 'edited-rgb'}


def validate_generation(name, generation, sample_id, expected_checkpoint_sha=None):
    """Enforce each producer's actual contract rather than assuming one lineage."""
    g = generation
    require(g['sample_id'] == sample_id and g['actual_resolution'] == 1024
            and g['source_defined_normalization'] and not g['per_output_alignment']
            and not g['GT_geometry_or_coords_used_in_generation']
            and g['fixed_shape_models'], f'Generation contract differs: {name}/{sample_id}')
    m = g['metrics']
    require(m['threshold'] == .02 and m['prediction_samples'] == 20000
            and m['target_samples'] == 20000 and not m['per_output_alignment'],
            'Raw CD/FScore protocols differ')
    if name in ZERO_MODES:
        mode = ZERO_MODES[name]
        states = ['M0'] * 4 if mode == 'source-only' else ['M1', 'M0', 'M0', 'M0']
        require(g['status'] == 'PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION'
                and g['original_zero_shot_full_stack']
                and g['conditioning_mode'] == mode and g['input_states'] == states
                and g['input_camera_ids'] == [8, 9, 10, 11]
                and g['GT_loaded_only_after_generation']
                and not g['additional_PBR_conditioner'] and not g['visibility_mask_used']
                and g['deformation_finetuning_updates'] == 0
                and g['original_stage_configs_and_weights_unchanged']
                and g['original_multiview_steps'] == dict(
                    ss64=4210, shape512=4210, shape1024=16850, pbr1024=16850),
                'Original four-RGB zero-shot protocol differs')
        inputs = g['actual_RGB_input_files']
        require(len(inputs) == 4 and len({r['path'] for r in inputs}) == 4
                and all(Path(r['path']).suffix.lower() in {'.png', '.jpg', '.jpeg'}
                        and len(r['sha256']) == 64 for r in inputs),
                'Four explicit RGB input identities required')
    else:
        require(name in REPLAY_MODELS, f'Undeclared model: {name}')
        require(g['status'] == 'PASS_ACTUAL_NATIVE_PBR_REPLAY_ON_GENERATED_SHAPE'
                and not g['GT_sparse_latents_or_projection_cache_used']
                and g['GT_evaluation_artifact_copied_only_after_prediction_saved']
                and g['raw_surface_metrics_unchanged']
                and g['geometry_array_byte_identity']
                and all(g['geometry_array_byte_identity'].values())
                and g['conditioning_preserves_CPU_and_CUDA_RNG']
                and g['training_updates'] == 0
                and not g['original_cascade_texture_noise_reused']
                and g['PBR_checkpoint_SHA256'] == expected_checkpoint_sha,
                'Trained PBR replay contract differs')
        arm = 'v1' if name == 'full-v1' else ('direct' if name.endswith('Direct') else 'retargeting')
        require(g['PBR_arm'] == arm and g['additional_PBR_conditioner'] == (arm != 'v1'),
                'PBR arm label differs from actual generation')


def validate_perceptual_reports(reports):
    """Require the same official weights, precision and primary image protocol."""
    identities, rows = [], {}
    for report in reports:
        require(report['status'] == 'PASS_ACTUAL_OFFICIAL_DREAMSIM_CLIP_RENDER_EVALUATION'
                and report['full_panel'] and report['fullframe_primary']
                and report['separate_GT_only_crop_supplementary']
                and report['GPU_sanity']['status'] == 'PASS_ACTUAL_GPU_OFFICIAL_METRIC_SANITY'
                and report['no_model_training'] and not report['global_environment_changed'],
                'Incomplete official perceptual report')
        identities.append((report['metric_registry_SHA256'], report['model_weights'],
                           report['model_precision']))
        for row in report['per_view']:
            key = (row['arm'], row['sample_id'], row['camera_id'])
            require(key not in rows, f'Duplicate perceptual row: {key}')
            rows[key] = row
    require(bool(identities) and all(i == identities[0] for i in identities),
            'Official metric weights/registry/precision differ across reports')
    return rows


def evaluate_plan(plan):
    """Read SHA-closed pixels and scores, validating references across all seven models."""
    require(set(plan['models']) == set(REPLAY_MODELS) | set(ZERO_MODES),
            'Five trained-PBR and both original zero-shot models required')
    closed = ClosedInputs(plan['canonical_local_root'])
    reports = []
    for path, receipt in plan['perceptuals'].items():
        closed.check(path, receipt)
        reports.append(json.loads(Path(path).read_text()))
    perceptual = validate_perceptual_reports(reports)
    expected, references, records, generations = set(), {}, {}, {}
    for name, rows in plan['models'].items():
        require(len(rows) == plan['expected_assets'], f'Unexpected asset count: {name}')
        records[name] = []
        for row in rows:
            sid, cluster = row['sample_id'], row['source_cluster_id']
            for path, receipt in row['closed_inputs'].items():
                closed.check(path, receipt)
            require(row['generation'] in row['closed_inputs'] and row['render'] in row['closed_inputs'],
                    'Generation and render need explicit canonical receipts')
            g = json.loads(Path(row['generation']).read_text())
            validate_generation(name, g, sid, plan['PBR_checkpoint_SHA256'].get(name))
            generations[name, sid] = g
            render = json.loads(Path(row['render']).read_text())
            require(render['status'] == 'PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE'
                    and render['sample_id'] == sid and render['same_reference_pixels_for_every_model'],
                    'Fixed original-material M1 render reference required')
            views = {v['camera_id']: v for v in render['per_view']}
            require(len(render['per_view']) == len(views) == 3 and set(views) == {8, 9, 10},
                    'Exactly three distinct evaluation cameras required')
            values = {}
            for cid, view in views.items():
                key = (name, sid, cid)
                expected.add(key)
                p = perceptual[key]
                require(p['source_cluster_id'] == cluster, 'Perceptual source cluster differs')
                pixels = Path(row['render']).parent / f'camera{cid:02d}.npz'
                digest = closed.check(pixels, pixels.parent / 'file4-closure.json')
                require(p['shared_npz_SHA256'] == digest, 'Scores belong to different rendered pixels')
                audit = p['image_audit']
                with np.load(pixels, allow_pickle=False) as z:
                    # Hash raw bytes using the producer's layout and dtype.
                    identity = tuple(hashlib.sha256(z[k].tobytes()).hexdigest()
                                     for k in ['GT_shared_renderer', 'GT_alpha'])
                require(identity == (audit['original_float_GT_SHA256'], audit['original_GT_alpha_SHA256'])
                        == (view['fixed_GT_RGB_SHA256'], view['fixed_GT_alpha_SHA256']),
                        'GT reference bytes differ from score/render audit')
                reference = identity + (tuple(audit['GT_only_crop_box']),
                    audit['converted_image_SHA256']['reference_full'],
                    audit['converted_image_SHA256']['reference_GTcrop'])
                require(references.setdefault((sid, cid), reference) == reference,
                        'Cross-model original-material M1 GT/crop differs')
                values[cid] = {metric: view[metric] for metric in IMAGE_METRICS[:5]}
                scores = p['scores']
                values[cid].update(DreamSim=scores['DreamSim_fullframe'], CLIP=scores['CLIP_cosine_fullframe'],
                    DreamSim_GTcrop=scores['DreamSim_GT_foreground_crop'],
                    CLIP_GTcrop=scores['CLIP_cosine_GT_foreground_crop'])
            records[name].append(dict(sample_id=sid, source_cluster_id=cluster, views=values,
                CD=g['metrics']['chamfer_L1'], FScore=g['metrics']['surface_FScore']))
    require(expected == set(perceptual), 'Unexpected or missing official perceptual rows')
    for row in records['full-v1']:
        sid = row['sample_id']
        baseline = generations['full-v1', sid]
        for name in REPLAY_MODELS:
            g = generations[name, sid]
            for field in ['sampler_class', 'sampler_params', 'PBR_replay_seed', 'PBR_noise_protocol']:
                require(g[field] == baseline[field], f'Replay sampler differs: {name}/{sid}/{field}')
        for name in ['iso-R', 'iso-Direct']:
            g = generations[name, sid]
            for field in ['geometry_evaluation_SHA256', 'metrics', 'SS_prediction_SHA256',
                          'actual_initial_texture_noise_SHA256', 'generated_shape_source']:
                require(g[field] == baseline[field], f'PBR isolation differs: {name}/{sid}/{field}')
    summary = summarize_view_roles(records, plan['contrasts'], draws=plan['draws'], seed=plan['seed'])
    closed.check(plan['existing_summary'], plan['existing_summary_receipt'])
    old = json.loads(Path(plan['existing_summary']).read_text())
    difference = existing_statistic_difference(summary, old['roles']['all8_9_10'])
    summary.update(status='PASS_ACTUAL_ORIGINAL_ZERO_AND_TRAINED_PBR_SEVEN_MODEL_COMPARISON',
        actual_perceptual_views=len(perceptual), input_SHA256=closed.inputs,
        maximum_previous_five_model_allview_statistic_difference=difference,
        same_original_material_on_M1_GT_reference=True, no_extra_GPU_inference_or_optimizer_updates=True,
        zero_vs_replay_texture_noise_streams_identical=False,
        cross_protocol_comparison='Deployment on one sampling noise; not isolated training/module causality',
        source_preservation=['LPIPS', 'DreamSim', 'CLIP'],
        target_alignment=['PSNR', 'SSIM', 'M_IoU', 'CD', 'FScore'],
        source_view_reference='Original source material on edited M1 geometry, not undeformed M0 RGB',
        FID_KID_scope='Retained for final508; not estimated from this development panel',
        independent_training_replication=False, intervals_posthoc_uncorrected=True,
        model_adopted=False, full_goal_complete=False)
    return summary
