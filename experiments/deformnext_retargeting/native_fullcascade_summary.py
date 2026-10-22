"""Compare native SS480 cascades and paired original/trained PBR textures."""
import json

from .cascade_view_roles import summarize_view_roles
from .native_fullcascade_contract import PAIRS


GROUPS = tuple(variant+'-'+kind for variant in PAIRS for kind in ('prior','matched'))


def summarize(cases, generations, shared, perceptual, *, sample_ids, clusters,
              pixel_SHA256, metric_registry_SHA256, original_PBR_SHA256):
    if (set(cases) != set(PAIRS) or set(generations) != set(GROUPS) or set(shared) != set(GROUPS)
            or len(sample_ids) != 32 or len(set(sample_ids)) != 32 or set(clusters) != set(sample_ids)
            or perceptual.get('status') != 'PASS_ACTUAL_OFFICIAL_DREAMSIM_CLIP_RENDER_EVALUATION'
            or perceptual.get('metric_registry_SHA256') != metric_registry_SHA256
            or perceptual.get('model_precision') != 'float32,TF32off,noAMP'
            or perceptual.get('full_panel') is not True or not perceptual.get('no_model_training')
            or perceptual.get('global_environment_changed')
            or perceptual.get('GPU_sanity', {}).get('status') != 'PASS_ACTUAL_GPU_OFFICIAL_METRIC_SANITY'
            or perceptual['GPU_sanity'].get('CPU_model_forward_or_training_updates') != 0):
        raise ValueError('all four SS variants, eight groups and pinned GPU official metric protocol required')
    scores = {}
    for row in perceptual['per_view']:
        key = (row['arm'], row['sample_id'], row['camera_id'])
        if key in scores: raise ValueError('duplicate official image score')
        scores[key] = row
    if set(scores) != {(g,s,c) for g in GROUPS for s in sample_ids for c in (8,9,10)}:
        raise ValueError('all eight groups32 three-camera official768 coverage required')
    gt, original_stages, trained_weights, records = {}, None, {}, {}
    for variant, mode in PAIRS.items():
        if set(cases[variant]) != set(sample_ids): raise ValueError('complete actual cascade case panel required')
        for sid in sample_ids:
            case = cases[variant][sid]
            if (case.get('status') != 'PASS_ACTUAL_NATIVE_SS480_AND_MATCHED_PBR1900_FULLCASCADE_CASE'
                    or case.get('sample_id') != sid or case.get('source_cluster_id') != clusters[sid]
                    or case.get('SS_variant') != variant or case.get('PBR_mode') != mode
                    or not case.get('original_Shape5124210_Shape102416850')
                    or not case.get('paired_original_and_trained_texture_initial_noise_exact')
                    or not case.get('paired_raw_geometry_and_native_predicted_LOD_identical')
                    or not case.get('actual_prior_and_matched_three_camera_original_material_M1_shared_scores')
                    or not case.get('original_generated_geometry_not_aligned_or_replaced')
                    or len(case.get('terminals', [])) != 7
                    or any(t.get('returncode') != 0 for t in case['terminals'])):
                raise ValueError('actual native SS480 case, fixed stages and completed paired texture/render work required')
            stages = case['actual_original_stage_checkpoint_SHA256']
            if set(stages) != {'shape512','shape1024','pbr1024'} or stages['pbr1024'] != original_PBR_SHA256:
                raise ValueError('SS-only comparison must retain the original complete PBR16850')
            if original_stages is None: original_stages = stages
            elif stages != original_stages: raise ValueError('original Shape/PBR checkpoint changed across groups')
            pair = []
            for kind in ('prior','matched'):
                group = variant+'-'+kind
                if set(generations[group]) != set(sample_ids) or set(shared[group]) != set(sample_ids):
                    raise ValueError('complete own generated and shared-render panel required')
                g = generations[group][sid]
                expected_kind = 'original16850-engineering' if kind == 'prior' else 'trained-native1900'
                expected_mode = 'vanilla' if kind == 'prior' else mode
                if (g.get('status') != 'PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE'
                        or g.get('sample_id') != sid or g.get('PBR_arm') != expected_mode
                        or g.get('checkpoint_kind') != expected_kind
                        or g.get('engineering_only') is not (kind == 'prior')
                        or not g.get('source_weights_unchanged')
                        or g.get('GT_geometry_or_coords_used_in_generation') is not False
                        or not g.get('conditioning_preserves_CPU_and_CUDA_RNG')
                        or not g.get('raw_surface_metrics_unchanged')
                        or not all(g['geometry_array_byte_identity'].values())
                        or g.get('SS_prediction_SHA256') != case['SS_prediction_SHA256']
                        or g.get('SS_source_evaluation') != case['source_SS_evaluation']):
                    raise ValueError('own actual image-only PBR replay on this native predicted SS geometry required')
                if kind == 'prior':
                    if g.get('PBR_checkpoint_SHA256') != original_PBR_SHA256 or g.get('conditioner_checkpoint') is not None:
                        raise ValueError('original PBR16850 without a conditioner is required for SS isolation')
                else:
                    if (not g['PBR_checkpoint'].endswith('/ckpts/denoiser_step0001900.pt')
                            or g.get('PBR_checkpoint_SHA256') != case['actual_matched_PBR_checkpoint_SHA256']
                            or g.get('conditioner_checkpoint') != (None if mode == 'vanilla' else
                                g['PBR_checkpoint'].replace('/ckpts/denoiser_step0001900.pt',
                                    '/ckpts/deform_conditioner_step0001900.pt'))):
                        raise ValueError('own completed native1900 NN/source pair required')
                    if trained_weights.setdefault(mode,g['PBR_checkpoint_SHA256']) != g['PBR_checkpoint_SHA256']:
                        raise ValueError('trained checkpoint changed across cases')
                identity = tuple(g[k] for k in ['original_prediction_SHA256','geometry_evaluation_SHA256',
                    'SS_prediction_SHA256','actual_initial_texture_noise_SHA256','PBR_replay_seed','sampler_class'])
                pair.append(identity+(json.dumps(g['sampler_params'],sort_keys=True),json.dumps(g['metrics'],sort_keys=True)))
            if pair[0] != pair[1]: raise ValueError('prior/matched geometry, SS, initial noise, sampler or surface metrics differ')
    for group in GROUPS:
        records[group] = []
        for sid in sample_ids:
            g,r = generations[group][sid],shared[group][sid]
            if (r.get('status') != 'PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE' or r.get('sample_id') != sid
                    or not r.get('same_reference_pixels_for_every_model') or len(r['per_view']) != 3):
                raise ValueError('actual same M1 original-material reference required')
            views = {}
            for view in r['per_view']:
                cid=view['camera_id'];key=(group,sid,cid)
                if cid not in (8,9,10) or cid in views: raise ValueError('unique exact target/source cameras required')
                per=scores[key]
                if (per['source_cluster_id'] != clusters[sid] or per['shared_npz_SHA256'] != pixel_SHA256[key]
                        or not view.get('identical_camera_matrices') or not view.get('source_material_identity_preserved')):
                    raise ValueError('official metric belongs to different pixels, cluster or camera')
                identity=(view['fixed_GT_RGB_SHA256'],view['fixed_GT_alpha_SHA256'])
                if gt.setdefault((sid,cid),identity) != identity: raise ValueError('cross-model reference RGB/alpha changed')
                views[cid]={k:view[k] for k in ['PSNR','SSIM','LPIPS','M_IoU','foreground_PSNR']}
                p=per['scores'];views[cid].update(DreamSim=p['DreamSim_fullframe'],CLIP=p['CLIP_cosine_fullframe'],
                    DreamSim_GTcrop=p['DreamSim_GT_foreground_crop'],CLIP_GTcrop=p['CLIP_cosine_GT_foreground_crop'])
            records[group].append(dict(sample_id=sid,source_cluster_id=clusters[sid],views=views,
                CD=g['metrics']['chamfer_L1'],FScore=g['metrics']['surface_FScore']))
    variants=list(PAIRS)
    contrasts=[[a+'-'+kind,b+'-'+kind] for kind in ('prior','matched')
        for i,a in enumerate(variants) for b in variants[:i]]
    contrasts += [[v+'-matched',v+'-prior'] for v in variants]
    result=summarize_view_roles(records,contrasts,draws=4000,seed=20261022)
    result.update(original_stage_checkpoint_SHA256=original_stages,trained_PBR_checkpoint_SHA256=trained_weights,
        SS_isolation_groups=[v+'-prior' for v in variants],paired_texture_initial_noise_within_SS_variant=True,
        texture_noise_not_claimed_identical_across_different_generated_supports=True,
        primary_metric_families=dict(source_preservation=['LPIPS','DreamSim','CLIP'],
            target_alignment=['PSNR','SSIM','M_IoU','CD','FScore']),
        development_unadjusted_exploratory_CI=True,independent_training_seeds=1,FID_KID_measured=False)
    return result
