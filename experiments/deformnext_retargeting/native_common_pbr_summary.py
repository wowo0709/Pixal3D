"""Separate SS effects from PBR effects using one common trained V1 PBR."""
from .archived_native_guides import POLICY
from .cascade_view_roles import IMAGE_METRICS, summarize_view_roles
from .generated_cascade_comparison import validate_perceptual_reports
from .legacy_native_pbr_bridge_summary import maximum_difference
from .native_fullcascade_summary import GROUPS as NATIVE_GROUPS, summarize as native_summary

VARIANTS = ('R_top1_only', 'R_feature', 'R_raw_topk')
COMMON_GROUPS = tuple(v+'-common-V1PBR' for v in VARIANTS)
GROUPS = NATIVE_GROUPS + COMMON_GROUPS
CASE_STATUS = 'PASS_ACTUAL_NATIVE_SS_COMMON_V1_PBR_CAPTURED_GUIDES_CASE'


def summarize(native_cases, common_cases, generations, shared, native_perceptual,
              common_perceptual, native_statistics, *, sample_ids, clusters,
              pixel_SHA256, pixel_GT, metric_registry_SHA256, original_PBR_SHA256):
    if (set(generations) != set(GROUPS) or set(shared) != set(GROUPS)
            or set(common_cases) != set(VARIANTS)
            or any(set(generations[g]) != set(sample_ids) or set(shared[g]) != set(sample_ids) for g in GROUPS)):
        raise ValueError('all eight native and three common-V1 groups on the same32 required')
    reproduced = native_summary(native_cases, {g:generations[g] for g in NATIVE_GROUPS},
        {g:shared[g] for g in NATIVE_GROUPS}, native_perceptual, sample_ids=sample_ids,
        clusters=clusters, pixel_SHA256=pixel_SHA256, metric_registry_SHA256=metric_registry_SHA256,
        original_PBR_SHA256=original_PBR_SHA256)
    difference = max(maximum_difference(reproduced[k], native_statistics[k])
                     for k in ('roles', 'interaction_target_minus_source_effect', 'per_asset'))
    if difference > 1e-12:
        raise ValueError('original native8 statistics changed')
    for report, groups in [(native_perceptual, NATIVE_GROUPS), (common_perceptual, COMMON_GROUPS)]:
        if (report['metric_registry_SHA256'] != metric_registry_SHA256
                or report['model_precision'] != 'float32,TF32off,noAMP'
                or report['GPU_sanity']['CPU_model_forward_or_training_updates'] != 0
                or {r['arm'] for r in report['per_view']} != set(groups)):
            raise ValueError('actual native768/common288 official GPU protocol required')
    official = validate_perceptual_reports([native_perceptual, common_perceptual])
    if set(official) != {(g,s,c) for g in GROUPS for s in sample_ids for c in (8,9,10)}:
        raise ValueError('exact1056 official image scores required')
    common_weight = reproduced['trained_PBR_checkpoint_SHA256']['vanilla']
    for variant in VARIANTS:
        if set(common_cases[variant]) != set(sample_ids):
            raise ValueError('all32 common-PBR cases required')
        for sid in sample_ids:
            case, base = common_cases[variant][sid], native_cases[variant][sid]
            g, reference = generations[variant+'-common-V1PBR'][sid], generations[variant+'-prior'][sid]
            if (case.get('status') != CASE_STATUS or case.get('SS_variant') != variant
                    or case.get('sample_id') != sid or case.get('source_cluster_id') != clusters[sid]
                    or case.get('actual_PBR_checkpoint_SHA256') != common_weight
                    or not case.get('original_native_geometry_and_guides_reused_read_only')
                    or not case.get('actual_original_predicted_LOD_reused')
                    or case.get('new_SS_or_Shape_forward_calls') != 0
                    or len(case.get('actual_owned_terminals', [])) != 3
                    or any(t.get('returncode') != 0 or not t.get('actual_session_drained')
                           for t in case['actual_owned_terminals'])):
                raise ValueError('actual completed common-PBR case on its own native SS geometry required')
            if (g.get('status') != 'PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE'
                    or g.get('sample_id') != sid or g.get('PBR_arm') != 'vanilla'
                    or g.get('checkpoint_kind') != 'trained-native1900' or g.get('engineering_only') is not False
                    or g.get('PBR_checkpoint_SHA256') != common_weight or g.get('conditioner_checkpoint') is not None
                    or not g.get('source_weights_unchanged') or not all(g['geometry_array_byte_identity'].values())
                    or g.get('SS_prediction_SHA256') != base['SS_prediction_SHA256']
                    or g.get('SS_source_evaluation') != base['source_SS_evaluation']
                    or g.get('GT_geometry_or_coords_used_in_generation') is not False
                    or g.get('GT_sparse_latents_or_projection_cache_used') is not False
                    or not g.get('conditioning_preserves_CPU_and_CUDA_RNG')):
                raise ValueError('own native SS and exact common trained V1 PBR required')
            for key in ('original_prediction_SHA256', 'geometry_evaluation_SHA256', 'SS_prediction_SHA256',
                        'actual_initial_texture_noise_SHA256', 'PBR_replay_seed', 'sampler_class',
                        'sampler_params', 'metrics', 'actual_native_guide_archive_SHA256'):
                if g[key] != reference[key]:
                    raise ValueError('within-SS common-PBR geometry/noise/guide identity differs: '+key)
    records, references = {}, {}
    for group in GROUPS:
        records[group] = []
        for sid in sample_ids:
            g, r = generations[group][sid], shared[group][sid]
            if (g.get('texture_support_policy') != POLICY or g.get('shape_decoder_recomputed') is not False
                    or r.get('status') != 'PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE'
                    or r.get('sample_id') != sid or not r.get('same_reference_pixels_for_every_model')
                    or len(r['per_view']) != 3):
                raise ValueError('same actual native guide policy and original M1 material reference required')
            views = {}
            for view in r['per_view']:
                cid = view['camera_id']; key = (group,sid,cid)
                if cid not in (8,9,10) or cid in views:
                    raise ValueError('unique target8/source9/10 required')
                p = official[key]; audit = p['image_audit']
                identity = (view['fixed_GT_RGB_SHA256'], view['fixed_GT_alpha_SHA256'])
                if (p['source_cluster_id'] != clusters[sid] or p['shared_npz_SHA256'] != pixel_SHA256[key]
                        or identity != pixel_GT[key]
                        or identity != (audit['original_float_GT_SHA256'], audit['original_GT_alpha_SHA256'])
                        or not view.get('identical_camera_matrices') or not view.get('source_material_identity_preserved')):
                    raise ValueError('actual score/render/camera/GT pixel identity mismatch')
                reference = identity + (tuple(audit['GT_only_crop_box']),
                    audit['converted_image_SHA256']['reference_full'], audit['converted_image_SHA256']['reference_GTcrop'])
                if references.setdefault((sid,cid), reference) != reference:
                    raise ValueError('cross-model original M1 reference/crop differs')
                views[cid] = {k:view[k] for k in IMAGE_METRICS[:5]}
                scores = p['scores']; views[cid].update(DreamSim=scores['DreamSim_fullframe'],
                    CLIP=scores['CLIP_cosine_fullframe'], DreamSim_GTcrop=scores['DreamSim_GT_foreground_crop'],
                    CLIP_GTcrop=scores['CLIP_cosine_GT_foreground_crop'])
            records[group].append(dict(sample_id=sid, source_cluster_id=clusters[sid], views=views,
                CD=g['metrics']['chamfer_L1'], FScore=g['metrics']['surface_FScore']))
    contrasts = [label.split('_minus_',1) for label in native_statistics['roles']['target8']['contrasts']]
    common = ['vanilla-matched', *COMMON_GROUPS]
    contrasts += [[a,b] for i,a in enumerate(common) for b in common[:i]]
    contrasts += [[v+'-matched', v+'-common-V1PBR'] for v in VARIANTS]
    contrasts += [[v+'-common-V1PBR', v+'-prior'] for v in VARIANTS]
    result = summarize_view_roles(records, contrasts, draws=4000, seed=20261022)
    result.update(status='PASS_ACTUAL_NATIVE11_COMMON_V1_PBR_COMPONENT_COMPARISON',
        native8_previous_statistics_max_difference=difference, assets=32, actual_official_views=1056,
        new_SS_or_Shape_forward_calls=0, additional_texture_replays=96,
        common_trained_PBR_checkpoint_SHA256=common_weight, texture_support_policy=POLICY,
        SS_effect_comparisons_share_trained_V1_PBR=True, PBR_effect_comparisons_share_actual_geometry_and_guides=True,
        initial_texture_noise_not_claimed_equal_across_different_SS_supports=True,
        exploratory_repeated_development_not_globally_corrected=True,
        independent_selected_model_confirmation_and_final_TEST508_still_required=True,
        model_adopted=False, full_goal_complete=False)
    return result
