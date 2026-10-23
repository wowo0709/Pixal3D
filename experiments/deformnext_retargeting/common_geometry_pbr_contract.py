"""Four native PBR models on the same generated V1 geometry and actual guides."""
import re

from .archived_native_guides import POLICY
from .pbr_replay_artifacts import GEOMETRY_FIELDS
from .r_pbr_generated_summary import summarize as summarize_images


MODES = ('vanilla', 'top1-only', 'top1-plus-feature', 'raw-topk-plus-feature')
CASE_STATUS = 'PASS_ACTUAL_CORE_PBR1900_COMMON_PREDICTED_NATIVE_GUIDE_CASE'


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_stage(cfg, mode, ranks, *, training_seed=101):
    require(mode in MODES and type(training_seed) is int and training_seed in (101, 202),
            'explicit native mode and training seed required')
    e, t = cfg['experiment'], cfg['trainer']['args']
    source_count = 0 if mode == 'vanilla' else 10 if mode == 'top1-only' else 19
    require(cfg['training_seed'] == training_seed and cfg['num_gpus'] == 4
        and cfg['trainer']['name'] == 'RMediatedPBRScienceTrainer'
        and e['PBR_source_mode'] == mode and e['source_multiview_step'] == 16850
        and e['native_task_objective'] == 'PBR_sparse_flow_mse' and e['source_teacher_updates'] == 0
        and e['engineering_only'] is False and t['max_steps'] == 1900
        and t['batch_size_per_gpu'] == 6 and t['batch_split'] == 1
        and len(ranks) == 4 and {r['rank'] for r in ranks} == {0, 1, 2, 3},
        'own fresh native1900/world4/global24 scientific stage required')
    spec = t.get('deform_conditioner')
    require(spec is None if mode == 'vanilla' else spec is not None and spec['appearance_mode'] == mode,
            'physical source mode must equal the saved configuration')
    for row in ranks:
        require(row.get('status') == 'PASS_ACTUAL_ORIGINAL_PBR16850_FRESH_NATIVE1900_STAGE'
            and type(row.get('rank')) is int and row['mode'] == mode
            and row['training_seed'] == training_seed and row['source_multiview_step'] == 16850
            and (row['world_size'], row['global_batch'], row['microbatch'], row['accumulation']) == (4, 24, 6, 1)
            and row['actual_optimizer_updates'] == 1900 and row['actual_optimizer_states'] == 360+source_count
            and row['actual_training_assets'] == 9139 and row['actual_dataloader_batches'] == 380
            and row['source_tensors'] == source_count
            and all(row.get(k) is True for k in ('fresh_optimizer', 'initial4_identity_before_first_update',
                'frozen_base_matcher_and_anchors_exact', 'post_training_DDP_parameters_exact', 'stage_training_complete')),
            'all four actual completed ranks and frozen base/source integrity required')
    return source_count


def validate_replay(report, base_generation, *, mode, model_SHA256, prediction_SHA256):
    require(mode in MODES and digest(model_SHA256) and digest(prediction_SHA256),
            'explicit actual own model and common prediction hashes required')
    require(base_generation['status'] == 'PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION'
        and base_generation['GT_geometry_or_coords_used_in_generation'] is False
        and base_generation['per_output_alignment'] is False
        and base_generation['texture_support_policy'] == POLICY
        and base_generation['actual_native_guide_logits_order_scale_and_dead_branches_preserved'] is True,
        'actual native predicted V1 geometry and original captured guides required')
    require(report['status'] == 'PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE'
        and report['sample_id'] == base_generation['sample_id'] and report['PBR_arm'] == mode
        and report['checkpoint_kind'] == 'trained-native1900' and report['engineering_only'] is False
        and report['PBR_checkpoint_SHA256'] == model_SHA256
        and report['PBR_checkpoint'].endswith('/ckpts/denoiser_step0001900.pt')
        and report['conditioner_checkpoint'] == (None if mode == 'vanilla' else
            report['PBR_checkpoint'].replace('/ckpts/denoiser_step0001900.pt', '/ckpts/deform_conditioner_step0001900.pt'))
        and report['original_prediction_SHA256'] == prediction_SHA256
        and report['SS_prediction_SHA256'] == base_generation['SS_prediction_SHA256']
        and report['SS_source_evaluation'] == base_generation['SS_source_evaluation']
        and report['metrics'] == base_generation['metrics']
        and set(report['geometry_array_byte_identity']) == set(GEOMETRY_FIELDS)
        and all(v is True for v in report['geometry_array_byte_identity'].values())
        and report['GT_geometry_or_coords_used_in_generation'] is False
        and report['GT_sparse_latents_or_projection_cache_used'] is False
        and report['per_output_alignment'] is False
        and report['source_weights_unchanged'] is True
        and report['conditioning_preserves_CPU_and_CUDA_RNG'] is True
        and report['raw_surface_metrics_unchanged'] is True
        and digest(report['actual_initial_texture_noise_SHA256'])
        and report['texture_support_policy'] == POLICY and report['shape_decoder_recomputed'] is False
        and report['actual_native_guide_archive_SHA256'] == base_generation['actual_native_guide_archive_SHA256']
        and report['actual_native_guide_logits_order_scale_and_dead_branches_preserved'] is True
        and report['numerical_entry_AST_device_literal_only']['all_other_AST_statements_unchanged'] is True,
        'one own native1900 replay on identical predicted geometry and actual original guides required')


def summarize(cases, generations, shared, official, *, sample_ids, clusters, pixel_SHA256,
              metric_registry_SHA256):
    require(set(cases) == set(MODES), 'four actual PBR modes required')
    common, weights = {}, {}
    for mode in MODES:
        require(set(cases[mode]) == set(sample_ids), 'all32 actual case receipts per mode required')
        for sid, case in cases[mode].items():
            g = generations[mode][sid]
            require(case['status'] == CASE_STATUS and case['mode'] == mode and case['sample_id'] == sid
                and case['source_cluster_id'] == clusters[sid]
                and case['new_SS_forwards'] == case['new_Shape_forwards'] == 0
                and case['new_PBR_sampling_calls'] == 1
                and len(case['actual_owned_terminals']) == 3
                and all(t['returncode'] == 0 and t['actual_session_drained'] for t in case['actual_owned_terminals'])
                and case['PBR_checkpoint_SHA256'] == g['PBR_checkpoint_SHA256']
                and case['common_prediction_SHA256'] == g['original_prediction_SHA256']
                and case['native_guide_SHA256'] == g['actual_native_guide_archive_SHA256']
                and g['texture_support_policy'] == POLICY and g['shape_decoder_recomputed'] is False
                and g['actual_native_guide_logits_order_scale_and_dead_branches_preserved'] is True,
                'complete actual replay/render/shared cases and original-guide identity required')
            identity = (case['common_prediction_SHA256'], case['native_guide_SHA256'])
            require(common.setdefault(sid, identity) == identity, 'common generated geometry/guide changed across modes')
            require(weights.setdefault(mode, g['PBR_checkpoint_SHA256']) == g['PBR_checkpoint_SHA256'],
                    'own actual checkpoint changed between cases')
    result = summarize_images(generations, shared, official, sample_ids=sample_ids, clusters=clusters,
                             pixel_SHA256=pixel_SHA256, metric_registry_SHA256=metric_registry_SHA256)
    result.update(status='PASS_ACTUAL_CORE4_PBR1900_FIXED_NATIVE_GUIDE32_EIGHT_METRICS',
        texture_support_policy=POLICY, actual_same_generated_geometry_and_native_guides=True,
        actual_paired_replay_initial_noise_sampler_and_raw_geometry_checked=True,
        PBR_checkpoint_SHA256=weights, reference='Original material on edited M1 geometry, not undeformed M0 RGB',
        original_cascade_noise_not_claimed_equal_to_seed_reset_replay=True,
        new_SS_or_Shape_forwards=0, original_support_not_reconstructed=True,
        no_legacy14_or_native8_or_old_reconstructed_support_dependency=True,
        FID_KID_and_independent_seed_and_final_TEST508_still_required=True,
        selected_model_adopted=False, full_goal_complete=False)
    return result
