"""Compare six legacy/common-PBR groups with eight actual native cascades."""
import math

from .cascade_view_roles import IMAGE_METRICS, summarize_view_roles
from .generated_cascade_comparison import validate_perceptual_reports
from .legacy_native_pbr_bridge import VARIANTS, validate_case, validate_pair
from .native_fullcascade_summary import GROUPS as NATIVE_GROUPS, summarize as summarize_native


LEGACY_GROUPS = tuple('legacy-'+v+'-'+k for v in VARIANTS for k in ('prior','matched'))
GROUPS = NATIVE_GROUPS + LEGACY_GROUPS


def maximum_difference(a, b):
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise ValueError('native statistics fields changed')
        return max((maximum_difference(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise ValueError('native statistics population changed')
        return max((maximum_difference(x,y) for x,y in zip(a,b)), default=0.)
    if isinstance(a, (float,int)) and not isinstance(a,bool):
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError('finite native statistics required')
        return abs(a-b)
    if a != b: raise ValueError('native statistics identity changed')
    return 0.


def summarize(native_cases, legacy_cases, original_geometries, generations, shared,
              native_perceptual, legacy_perceptual, native_statistics, *, sample_ids,
              clusters, pixel_SHA256, pixel_GT, metric_registry_SHA256, original_PBR_SHA256):
    if (set(generations) != set(GROUPS) or set(shared) != set(GROUPS)
            or set(legacy_cases) != set(VARIANTS) or set(original_geometries) != set(VARIANTS)
            or any(set(generations[g]) != set(sample_ids) or set(shared[g]) != set(sample_ids) for g in GROUPS)):
        raise ValueError('all14 groups on the same complete32 panel required')
    for report, groups in [(native_perceptual,NATIVE_GROUPS),(legacy_perceptual,LEGACY_GROUPS)]:
        if (report.get('metric_registry_SHA256') != metric_registry_SHA256
                or report.get('model_precision') != 'float32,TF32off,noAMP'
                or report.get('GPU_sanity',{}).get('CPU_model_forward_or_training_updates') != 0
                or {r['arm'] for r in report['per_view']} != set(groups)):
            raise ValueError('separate official native768 and legacy576 GPU metric reports required')
    official = validate_perceptual_reports([native_perceptual,legacy_perceptual])
    if set(official) != {(g,s,c) for g in GROUPS for s in sample_ids for c in (8,9,10)}:
        raise ValueError('exact1344 official image scores without duplicates required')
    reproduced = summarize_native(native_cases, {g:generations[g] for g in NATIVE_GROUPS},
        {g:shared[g] for g in NATIVE_GROUPS}, native_perceptual, sample_ids=sample_ids,
        clusters=clusters, pixel_SHA256=pixel_SHA256, metric_registry_SHA256=metric_registry_SHA256,
        original_PBR_SHA256=original_PBR_SHA256)
    difference = max(maximum_difference(reproduced[k],native_statistics[k])
                     for k in ('roles','interaction_target_minus_source_effect','per_asset'))
    if difference > 1e-12: raise ValueError('original native8 statistics not exactly reproduced')
    shapes = {k:reproduced['original_stage_checkpoint_SHA256'][k] for k in ('shape512','shape1024')}
    trained = reproduced['trained_PBR_checkpoint_SHA256']['vanilla']
    for variant in VARIANTS:
        if set(legacy_cases[variant]) != set(sample_ids) or set(original_geometries[variant]) != set(sample_ids):
            raise ValueError('complete actual legacy cases and original geometry metadata required')
        for sid in sample_ids:
            case = legacy_cases[variant][sid];validate_case(case)
            if (case['SS_variant'] != variant or case['sample_id'] != sid
                    or case['source_cluster_id'] != clusters[sid]
                    or case['actual_original_shape_checkpoint_SHA256'] != shapes
                    or case['actual_original_PBR_checkpoint_SHA256'] != original_PBR_SHA256
                    or case['actual_matched_PBR_checkpoint_SHA256'] != trained):
                raise ValueError('legacy groups must use common original Shapes and vanilla1900 PBR')
            validate_pair(generations['legacy-'+variant+'-prior'][sid],
                generations['legacy-'+variant+'-matched'][sid], original_geometries[variant][sid],
                sample_id=sid, original_PBR_SHA256=original_PBR_SHA256, trained_PBR_SHA256=trained,
                original_prediction_SHA256=case['original_geometry_inputs_SHA256'][case['original_geometry_root']+'/prediction.npz'])
    records, references = {}, {}
    for group in GROUPS:
        records[group] = []
        for sid in sample_ids:
            g, r = generations[group][sid], shared[group][sid]
            if (r.get('status') != 'PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE'
                    or r.get('sample_id') != sid or not r.get('same_reference_pixels_for_every_model')
                    or len(r['per_view']) != 3 or g['PBR_replay_seed'] != 20261020):
                raise ValueError('actual common M1 material reference and fixed texture seed required')
            views = {}
            for view in r['per_view']:
                cid = view['camera_id'];key = (group,sid,cid)
                if cid not in (8,9,10) or cid in views: raise ValueError('unique target8/source9/10 required')
                p = official[key];audit = p['image_audit']
                identity = (view['fixed_GT_RGB_SHA256'],view['fixed_GT_alpha_SHA256'])
                if (p['source_cluster_id'] != clusters[sid] or p['shared_npz_SHA256'] != pixel_SHA256[key]
                        or identity != pixel_GT[key]
                        or identity != (audit['original_float_GT_SHA256'],audit['original_GT_alpha_SHA256'])
                        or not view.get('identical_camera_matrices') or not view.get('source_material_identity_preserved')):
                    raise ValueError('actual score/render/camera/GT pixel identity mismatch')
                reference = identity + (tuple(audit['GT_only_crop_box']),
                    audit['converted_image_SHA256']['reference_full'],audit['converted_image_SHA256']['reference_GTcrop'])
                if references.setdefault((sid,cid),reference) != reference:
                    raise ValueError('cross-model original-material M1 reference or GT-only crop differs')
                views[cid] = {k:view[k] for k in IMAGE_METRICS[:5]}
                scores = p['scores'];views[cid].update(DreamSim=scores['DreamSim_fullframe'],
                    CLIP=scores['CLIP_cosine_fullframe'],DreamSim_GTcrop=scores['DreamSim_GT_foreground_crop'],
                    CLIP_GTcrop=scores['CLIP_cosine_GT_foreground_crop'])
            records[group].append(dict(sample_id=sid,source_cluster_id=clusters[sid],views=views,
                CD=g['metrics']['chamfer_L1'],FScore=g['metrics']['surface_FScore']))
    # All comparisons remain exploratory on a repeatedly used development panel.
    contrasts = [[a,b] for i,a in enumerate(GROUPS) for b in GROUPS[:i]]
    result = summarize_view_roles(records,contrasts,draws=4000,seed=20261023)
    result.update(status='PASS_ACTUAL_NATIVE8_AND_LEGACY6_COMMON_PBR_DEV32_COMPARISON',
        actual_generated_geometry_cases=224,actual_PBR_replay_cases=448,actual_official_perceptual_views=1344,
        existing_legacy_geometry_cases_reused_read_only=96,new_legacy_SS_or_Shape_inferences=0,
        native8_previous_statistics_max_difference=difference,
        original_shape_checkpoint_SHA256=shapes,original_PBR_checkpoint_SHA256=original_PBR_SHA256,
        legacy_common_trained_PBR_checkpoint_SHA256=trained,
        cross_SS_training_protocol_deployment_comparison_not_isolated_module_causality=True,
        source_preservation=['LPIPS','DreamSim','CLIP'],target_alignment=['PSNR','SSIM','M_IoU','CD','FScore'],
        paired_texture_noise_only_within_same_geometry=True,unadjusted_exploratory_development_CIs=True,
        independent_TRAIN_replication_and_final508_not_completed=True,FID_KID_measured=False,
        model_adopted=False,full_goal_complete=False)
    return result
