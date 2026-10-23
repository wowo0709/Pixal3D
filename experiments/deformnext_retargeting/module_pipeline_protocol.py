"""One shared pipeline and explicit single-factor module comparisons.

This is a protocol, not a training launcher or evidence of improved quality.
Old target-patch/direct-C trajectories cannot be reused as these controls.
"""
from copy import deepcopy


FULL = dict(conditioning='target-first-native-source-map', correspondence='semantic',
    matching='D1-score', proposal='member', selection='topk64',
    value_transform=True, learned_attention=True, attention_prior='matching',
    query_unit='source-camera-ray', retargeting=True, delivery='R-only',
    deformation_channels=0)


def protocol():
    arms = {}

    def add(name, **changes):
        arms[name] = dict(FULL, **changes)

    add('FULL')
    add('C_DECODER', selection='top1', value_transform=False, learned_attention=False)
    add('C_RAW', selection='top1', value_transform=False, learned_attention=False, matching='RAW-cosine')
    add('C_SPATIAL', selection='top1', value_transform=False, learned_attention=False,
        matching='RAW-cosine', correspondence='native-spatial')
    add('C_TOP1_V', selection='top1', learned_attention=False)
    add('C_FIXED_TOPK', learned_attention=False)
    add('FULL_QK_ONLY', attention_prior='none')
    add('FULL_VOXEL_Q', query_unit='per-voxel-then-pool-before-R')
    add('C_DECODER_NO_R', selection='top1', value_transform=False, learned_attention=False,
        retargeting=False)
    add('FULL_MEAN_MATCH', proposal='mean')
    arms['V1'] = dict(conditioning='native-MV4', source_module_physically_absent=True)
    contrasts = [
        ('semantic_correspondence', 'C_RAW', 'C_SPATIAL', 'correspondence'),
        ('matching_decoder', 'C_DECODER', 'C_RAW', 'matching'),
        ('value_transform', 'C_TOP1_V', 'C_DECODER', 'value_transform'),
        ('TopK_combination', 'C_FIXED_TOPK', 'C_TOP1_V', 'selection'),
        ('learned_attention', 'FULL', 'C_FIXED_TOPK', 'learned_attention'),
        ('matching_prior_in_attention', 'FULL', 'FULL_QK_ONLY', 'attention_prior'),
        ('ray_query_unit', 'FULL', 'FULL_VOXEL_Q', 'query_unit'),
        ('retargeting', 'C_DECODER', 'C_DECODER_NO_R', 'retargeting'),
        ('member_proposals', 'FULL', 'FULL_MEAN_MATCH', 'proposal'),
    ]
    result = dict(schema='unified-module-pipeline-v1', status='PROTOCOL_ONLY_RUNTIME_BINDING_PENDING',
        arms=arms, module_contrasts=[dict(effect=e, candidate=a, reference=b, changed_factor=k)
                                   for e,a,b,k in contrasts],
        primary_pipeline_contrasts=[['FULL','V1'], ['FULL','C_DECODER']],
        common=dict(dataset='DeformNeXt', train_assets=9149, validation_assets=508,
            starting_SS_multiview_step=4210, epochs=5, total_updates=480,
            global_batch=96, microbatch_per_GPU=96, accumulation=1, world_size=1,
            training_seed=101, confirmation_seed=202, lora_rank=16, lora_alpha=16,
            denoiser_LR=1e-5, source_LR=1e-4, source_initialization='cold',
            unchanged_native_initial_function_required=True,
            input_camera_ids=[8,9,10,11], input_states=['M1','M0','M0','M0'], target_modality='rgb',
            direct_attention_correction_bypass=False, no_gt_geometry_depth_or_matching_at_inference=True,
            source_without_evidence='preserve_original_native_source_conditioning',
            native_global_branch_unchanged=True, original_target_view_conditioning_unchanged=True),
        evaluation=dict(SS_epoch_panel=32, SS_final_panel=128, SS_generation_noises=3,
            primary_SS_metric='mean-asset voxel IoU after per-asset three-noise average',
            secondary_SS_metric='pooled TP/FP/FN voxel IoU, separately named',
            paired_source_cluster_bootstrap_draws=4000,
            primary_pipeline_family_size=2, module_family_size=9,
            repeated_development_search_is_not_independent_test=True,
            source_preservation=['LPIPS','DreamSim','CLIP'],
            target_alignment=['PSNR','SSIM','M-IoU','CD','F-Score'],
            global_naturalness=['FID','KID'],
            cascade_development_panel=32, full_cascade_uses_predicted_geometry=True,
            cascade_SS_module_comparisons_share_same_PBR_weights=True,
            cascade_PBR_module_comparisons_share_same_generated_geometry_and_native_guides=True,
            final_TEST_assets=508, TEST_model_and_checkpoint_freeze_before_quality_read=True,
            final_Tests_do_not_drive_architecture_selection=True),
        implementation_requirements=dict(
            shared_source_model_and_forward_before_arm_launch=True,
            RAW_and_D1_share_point_RAW64_proposals=True,
            spatial_uses_original_camera_projection=True,
            spatial3_vs_semantic64_is_module_substitution_not_fixed_K_address_only_effect=True,
            fixed_TopK_and_learned_attention_share_V_output_initialization=True,
            fixed_TopK_uses_nonuniform_frozen_matching_prior=True,
            Top1_control_consumes_one_actual_correspondence_feature=True,
            QK_only_changes_attention_prior_not_candidates_or_Top1_carrier=True,
            voxel_query_control_pools_both_readout_outputs_before_common_R=True,
            R_effect_at_Top1_does_not_disconnect_an_attention_branch=True,
            disabled_modules_physically_absent_and_optimizer_groups_audited=True,
            initial_and_first_update_native_parity_and_capacity_required=True,
            all_existing_runs_require_config_data_source_and_checkpoint_identity_audit_for_reuse=True),
        optional_after_core=dict(deformation_query='one additional Full+physical-DQ arm after retargeting; no 2x2',
            no_teacher_feature_regression_or_weight_permutation_sweeps=True),
        full_is_candidate_not_adopted=True, actual_new_training_started=False, full_goal_complete=False)
    validate_protocol(result)
    return result


def validate_protocol(value):
    arms = value['arms']
    if len(arms) != 11 or len(value['module_contrasts']) != 9:
        raise ValueError('finite eleven-arm pipeline protocol and nine causal comparisons required')
    for c in value['module_contrasts']:
        a,b = arms[c['candidate']],arms[c['reference']]
        changed = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
        if changed != {c['changed_factor']}:
            raise ValueError('module comparison changes multiple design factors: '+c['effect'])
    c = value['common']
    if c['global_batch'] != c['microbatch_per_GPU']*c['world_size']*c['accumulation']:
        raise ValueError('declared SS batch contract differs')
    for name,spec in arms.items():
        if name == 'V1':
            if spec != dict(conditioning='native-MV4', source_module_physically_absent=True):
                raise ValueError('vanilla must have no auxiliary conditioner')
            continue
        if spec['conditioning'] != FULL['conditioning'] or spec['delivery'] != 'R-only':
            raise ValueError('all module comparisons require the same target-first native R-only carrier')
        if spec['deformation_channels'] != 0:
            raise ValueError('physical DQ is a separate later arm')
        if not spec['retargeting'] and (spec['selection']!='top1' or spec['value_transform']):
            raise ValueError('R ablation may not disconnect an active attention branch')
        if spec['selection']=='top1' and spec['learned_attention']:
            raise ValueError('actual Top1 control must not retain a dummy TopK attention head')
    return deepcopy(value)
