"""Paired DEV32 rendering/geometry evidence for the five core SS controls."""
from .archived_native_guides import POLICY
from .cascade_view_roles import IMAGE_METRICS,summarize_view_roles
from .common_pipeline_contract import ARMS,CASE_STATUS
from .generated_cascade_comparison import validate_perceptual_reports


def summarize_common_pipeline(cases,generations,shared,official,*,sample_ids,clusters,
                              pixel_SHA256,pixel_GT,common_PBR_checkpoint,
                              common_PBR_SHA256,metric_registry_SHA256,draws=4000):
    if (len(sample_ids)!=32 or len(set(sample_ids))!=32
            or any(set(d)!=set(ARMS) for d in [cases,generations,shared])
            or set(clusters)!=set(sample_ids)):
        raise ValueError('all five core groups on the same unique32 source-cluster panel required')
    if (official['metric_registry_SHA256']!=metric_registry_SHA256
            or official['model_precision']!='float32,TF32off,noAMP'
            or official['GPU_sanity']['CPU_model_forward_or_training_updates']!=0):
        raise ValueError('same official GPU weights and FP32 evaluation required')
    scores=validate_perceptual_reports([official])
    expected={(a,s,c) for a in ARMS for s in sample_ids for c in (8,9,10)}
    if set(scores)!=expected or set(pixel_SHA256)!=expected or set(pixel_GT)!=expected:
        raise ValueError('all480 actual pixel-matched official scores required')
    records={};references={}
    for arm in ARMS:
        if any(set(d[arm])!=set(sample_ids) for d in [cases,generations,shared]):
            raise ValueError('missing actual core cascade case')
        records[arm]=[]
        for sid in sample_ids:
            case,g,r=cases[arm][sid],generations[arm][sid],shared[arm][sid]
            if (case['status']!=CASE_STATUS or case['SS_arm']!=arm
                    or case['sample_id']!=sid or case['source_cluster_id']!=clusters[sid]
                    or case['actual_common_PBR_checkpoint_SHA256']!=common_PBR_SHA256
                    or case['actual_common_PBR_mode']!='vanilla'
                    or case['actual_common_PBR_updates']!=1900
                    or not case['original_Shape5124210_Shape102416850']
                    or case['GT_geometry_used_in_generation'] or case['per_output_alignment']
                    or case['new_SS_neural_forwards']!=0 or case['new_PBR_texture_calls']!=1
                    or len(case['actual_owned_terminals'])!=3
                    or any(t['returncode']!=0 or not t['actual_session_drained'] for t in case['actual_owned_terminals'])):
                raise ValueError('same actual trained V1 PBR on each own predicted SS/Shape required')
            if (g['status']!='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION'
                    or g['sample_id']!=sid or g['PBR_checkpoint']!=common_PBR_checkpoint
                    or g['SS_prediction_SHA256']!=case['SS_prediction_SHA256']
                    or g['GT_geometry_or_coords_used_in_generation'] or g['per_output_alignment']
                    or not g['GT_loaded_only_after_generation'] or not g['source_defined_normalization']
                    or g['texture_support_policy']!=POLICY
                    or not g['actual_native_guide_logits_order_scale_and_dead_branches_preserved']
                    or not g['numerical_entry_AST_device_literal_only']['all_other_AST_statements_unchanged']
                    or g['metrics']['per_output_alignment'] or g['metrics']['threshold']!=.02):
                raise ValueError('original native predicted-geometry cascade and surface convention required')
            if (r['status']!='PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE' or r['sample_id']!=sid
                    or not r['same_reference_pixels_for_every_model'] or len(r['per_view'])!=3):
                raise ValueError('same original material M1 reference pixels required')
            views={}
            for view in r['per_view']:
                cid=view['camera_id'];key=(arm,sid,cid)
                if cid not in (8,9,10) or cid in views:raise ValueError('unique target8/source9/10 cameras required')
                p=scores[key];audit=p['image_audit'];identity=(view['fixed_GT_RGB_SHA256'],view['fixed_GT_alpha_SHA256'])
                if (p['source_cluster_id']!=clusters[sid] or p['shared_npz_SHA256']!=pixel_SHA256[key]
                        or identity!=pixel_GT[key]
                        or identity!=(audit['original_float_GT_SHA256'],audit['original_GT_alpha_SHA256'])
                        or not view['identical_camera_matrices'] or not view['source_material_identity_preserved']):
                    raise ValueError('actual generated pixels, reference/camera and official score pairing differs')
                reference=identity+(tuple(audit['GT_only_crop_box']),audit['converted_image_SHA256']['reference_full'],audit['converted_image_SHA256']['reference_GTcrop'])
                if references.setdefault((sid,cid),reference)!=reference:
                    raise ValueError('reference pixels or crop differ between core models')
                views[cid]={k:view[k] for k in IMAGE_METRICS[:5]}
                v=p['scores'];views[cid].update(DreamSim=v['DreamSim_fullframe'],CLIP=v['CLIP_cosine_fullframe'],
                    DreamSim_GTcrop=v['DreamSim_GT_foreground_crop'],CLIP_GTcrop=v['CLIP_cosine_GT_foreground_crop'])
            records[arm].append(dict(sample_id=sid,source_cluster_id=clusters[sid],views=views,
                CD=g['metrics']['chamfer_L1'],FScore=g['metrics']['surface_FScore']))
    contrasts=[['FULL','V1'],['FULL','C_DECODER'],['C_TOP1_V','C_DECODER'],
               ['C_FIXED_TOPK','C_TOP1_V'],['FULL','C_FIXED_TOPK']]
    result=summarize_view_roles(records,contrasts,draws=draws,seed=20261023)
    result.update(status='PASS_ACTUAL_CORE5_COMMON_TRAINED_PBR_CASCADE32_EIGHT_METRICS',
        actual_official_views=480,common_PBR_checkpoint_SHA256=common_PBR_SHA256,
        texture_support_policy=POLICY,all_SS_effects_share_same_trained_V1_PBR=True,
        all_cases_use_own_predicted_geometry_and_native_guides=True,
        reference='Original material on edited M1 geometry, not undeformed M0 RGB',
        original_cascade_texture_rng_protocol_same=True,
        initial_texture_noise_not_claimed_equal_across_different_SS_supports=True,
        source_preservation=['LPIPS','DreamSim','CLIP'],target_alignment=['PSNR','SSIM','M_IoU','CD','FScore'],
        exploratory_DEV32_secondary_metrics_CIs_not_global_confirmation=True,
        primary_family_corrected_SS128_analysis_is_separate=True,
        FID_KID_and_independent_TRAIN_seed_and_final_TEST508_still_required=True,
        no_legacy14_or_native8_completion_dependency=True,model_adopted=False,full_goal_complete=False)
    return result
