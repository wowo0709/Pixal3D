"""Ten final metrics for the actual deduplicated selected physical pipeline.

The original fixed-six reducers are preserved. Models and comparisons here
come from the actual DEV-only physical freeze, including a baseline winner.
"""
from copy import deepcopy

import numpy as np

from .cascade_view_roles import summarize_view_roles
from .metric_image_identity import validate_metric_image_join
from .selected_final_case_integrity import CASE_STATUS
from .selected_final_population import compose_models
from .selected_pipeline_freeze_v2 import METRIC_DIRECTIONS,exact
from .selected_pipeline_model_contract import require


EIGHT_STATUS='PASS_ACTUAL_SELECTED_PHYSICAL_TEST508_EIGHT_METRICS'
DIST_STATUS='PASS_ACTUAL_SELECTED_PHYSICAL_TEST508_OFFICIAL_FID_KID'
TEN_STATUS='PASS_ACTUAL_SELECTED_PHYSICAL_TEST508_ALL_TEN_METRICS'
PAIRWISE_ROLES=dict(LPIPS='source9_10',DreamSim='source9_10',CLIP='source9_10',
    PSNR='target8',SSIM='target8',M_IoU='target8',CD='all8_9_10',FScore='all8_9_10')


def metric_contract(parent,*,draws=4000):
    """Declare comparisons before TEST, retaining a genuinely merged winner."""
    selected=parent['selected_alias']
    require(selected in parent['models'] and type(draws) is int and draws>0,'actual selected model and positive declared bootstrap draws required')
    primary=[(selected,baseline) for baseline in ['V1','RETARGET'] if selected != baseline]
    descriptive=[('zero-edited4','zero-source4'),('V1','zero-edited4'),('RETARGET','V1')]
    contrasts=list(dict.fromkeys([*descriptive,*primary]))
    return dict(schema='selected-physical-final-ten-metric-contract-v1',
        pairwise_metric_roles=deepcopy(PAIRWISE_ROLES),metric_directions=deepcopy(METRIC_DIRECTIONS),
        primary_contrasts=[list(p) for p in primary],reported_contrasts=[list(p) for p in contrasts],
        source_cluster_bootstrap_draws=draws,source_cluster_bootstrap_seed=20261022,
        pairwise_primary_family_size=8*len(primary),distribution_metrics=['FID','KID'],
        distribution_CI_or_significance_claim=False,KID_subset_standard_deviation_is_not_CI=True,
        FID_KID=dict(subset_size=1000,subsets=100,seed=42017),
        fullframe_primary=True,GT_only_crop_supplementary=True,
        distribution_fullframe_cameras=[8,9,10],source_preservation_cameras=[9,10],target_alignment_camera=8)


def validate_contract(parent):
    contract=parent['evaluation_protocol']['selected_final_metric_contract']
    require(exact(contract,metric_contract(parent,draws=contract['source_cluster_bootstrap_draws'])),
        'all metric roles, directions, contrasts, resampling and official distribution settings must be frozen before TEST')
    return contract


def reference_binding(parent,controls):
    models=compose_models(parent,controls)
    return dict(schema='selected-physical-M1-original-material-reference-v1',
        population_manifest=deepcopy(parent['population']['manifest']),rendered_cameras=[8,9,10],
        original_materials_and_UVs=True,geometry='M1',reference_producer='zero-source4',
        original_control_binding=deepcopy(models['zero-source4']),GT_VAE_reconstruction_used=False)


def _primary_intervals(records,population,contract):
    ids=population['sample_ids'];clusters=population['source_clusters_by_sample'];row_index={sid:i for i,sid in enumerate(ids)}
    groups=[np.asarray([row_index[sid] for sid in ids if clusters[sid]==c]) for c in sorted(set(clusters.values()))]
    sizes=np.asarray([len(g) for g in groups]);draws=contract['source_cluster_bootstrap_draws']
    resamples=np.random.default_rng(contract['source_cluster_bootstrap_seed']).integers(0,len(groups),(draws,len(groups)))
    matrices={}
    for alias,rows in records.items():
        matrices[alias]=np.asarray([[r[m] if role=='all8_9_10' else np.mean([r['views'][cid][m]
            for cid in ([9,10] if role=='source9_10' else [8])]) for m,role in PAIRWISE_ROLES.items()] for r in rows],dtype=np.float64)
    alpha=.025/contract['pairwise_primary_family_size'];result={}
    for a,b in contract['primary_contrasts']:
        delta=matrices[a]-matrices[b];grouped=np.stack([delta[g].sum(0) for g in groups])
        boot=grouped[resamples].sum(1)/sizes[resamples].sum(1)[:,None];equal=(grouped/sizes[:,None])[resamples].mean(1)
        result[a+'_minus_'+b]=dict(endpoint_family_size=contract['pairwise_primary_family_size'],
            source_clusters=len(groups),deformation_weighted_delta=dict(zip(PAIRWISE_ROLES,delta.mean(0).tolist())),
            deformation_weighted_family_CI95={m:np.quantile(boot[:,j],[alpha,1-alpha]).tolist() for j,m in enumerate(PAIRWISE_ROLES)},
            source_weighted_family_CI95={m:np.quantile(equal[:,j],[alpha,1-alpha]).tolist() for j,m in enumerate(PAIRWISE_ROLES)})
    return result


def eight_metrics(*,parent,controls,cases,shared,perceptual):
    models=compose_models(parent,controls);population=parent['population'];contract=validate_contract(parent)
    ids=population['sample_ids'];clusters=population['source_clusters_by_sample'];registry=parent['evaluation_protocol']['metric_registry']['SHA256']
    require(set(cases)==set(shared)==set(perceptual)==set(models),'every distinct physical model and both original controls required')
    records={};observations={};input_hashes={};reference_pixels={};reference_surfaces={}
    for alias,model in models.items():
        require(list(cases[alias])==list(shared[alias])==ids,'complete ordered508 case and render population required')
        official=perceptual[alias]
        require(official.get('status')=='PASS_ACTUAL_OFFICIAL_DREAMSIM_CLIP_RENDER_EVALUATION'
            and official.get('metric_registry_SHA256')==registry and official.get('scope')=='final-test508'
            and official.get('model_precision')=='float32,TF32off,noAMP' and official.get('full_panel') is True
            and official.get('fullframe_primary') is True and official.get('no_model_training') is True
            and official.get('global_environment_changed') is False
            and official.get('GPU_sanity',{}).get('status')=='PASS_ACTUAL_GPU_OFFICIAL_METRIC_SANITY'
            and official['GPU_sanity'].get('CPU_model_forward_or_training_updates')==0,
            'actual same-registry full508 original FP32 GPU DreamSim and separate OpenAI CLIP required')
        scores={(r['sample_id'],r['camera_id']):r for r in official['per_view']}
        require(len(scores)==len(official['per_view'])==1524 and set(scores)=={(sid,cid) for sid in ids for cid in [8,9,10]},
            'all actual distinct1524 official views required')
        records[alias]=[];observations[alias]=[]
        for sid in ids:
            case=cases[alias][sid];report=shared[alias][sid];zero=model['kind']=='original-zero';reuse=case.get('geometry_reference') is not None
            phases=([] if reuse else ['geometry'])+([] if zero else ['PBR'])+['render','shared']
            require(case.get('status')==CASE_STATUS and case.get('actual_completed_native_case') is True
                and case.get('sample_id')==sid and case.get('source_cluster_id')==clusters[sid] and case.get('alias')==alias
                and exact(case.get('frozen_model_binding'),model) and case.get('GT_not_used_in_generation') is True
                and case.get('optimizer_updates')==0 and [t['phase'] for t in case['terminals']]==phases
                and all(t.get('returncode')==0 and t.get('actual_session_drained') is True for t in case['terminals']),
                'actual own frozen physical case and exact producer/reuse phase sequence required')
            surface=case['surface_reference_join'];metrics=case['raw_surface_metrics']
            require(surface.get('status')=='PASS_ACTUAL_LOCAL_NATIVE_GEOMETRY_SHARED_GT_SURFACE_JOIN'
                and surface.get('sample_id')==sid and surface.get('source_cluster_id')==clusters[sid]
                and exact(surface.get('native_metrics'),metrics) and surface.get('GT_reference_not_used_for_model_conditioning') is True
                and metrics.get('prediction_samples')==metrics.get('target_samples')==20000
                and metrics.get('threshold')==.02 and metrics.get('per_output_alignment') is False,
                'unchanged native20k/F@.02 scores and actual post-generation shared GT surface binding required')
            identity=(surface['GT_reference']['SHA256'],surface['target_points_SHA256'])
            require(reference_surfaces.setdefault(sid,identity)==identity,'all model surfaces require identical actual GT samples')
            expected='PASS_ACTUAL_SHARED_RENDERER_MATERIAL_REFERENCE' if alias=='zero-source4' else 'PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE'
            require(report.get('status')==expected and len(report['per_view'])==3,'original M1 material shared renderer required')
            per={r['camera_id']:r for r in report['per_view']};obs={r['camera_id']:r for r in case['scored_image_observations']}
            require(len(case['scored_image_observations'])==3 and set(per)==set(obs)=={8,9,10},'unique target8/source9/10 observations required')
            views={}
            for cid in [8,9,10]:
                view=per[cid];pixel=obs[cid];score=scores[sid,cid];audit=score['image_audit']
                require(pixel['sample_id']==sid and pixel['source_cluster_id']==clusters[sid]
                    and score['arm']==alias and score['source_cluster_id']==clusters[sid]
                    and score['shared_npz_SHA256']==pixel['archive_SHA256']
                    and audit['original_float_prediction_SHA256']==pixel['prediction_float_RGB_SHA256']
                    and audit['original_float_GT_SHA256']==pixel['reference_float_RGB_SHA256']
                    and audit['original_GT_alpha_SHA256']==pixel['reference_alpha_SHA256']
                    and audit.get('per_output_alignment') is False,'official perceptual scores must use these actual scored RGB/reference/alpha bytes')
                if alias!='zero-source4':
                    require(view.get('fixed_GT_RGB_SHA256')==pixel['reference_float_RGB_SHA256']
                        and view.get('fixed_GT_alpha_SHA256')==pixel['reference_alpha_SHA256']
                        and view.get('identical_camera_matrices') is True,'actual fixed-reference RGB/alpha and calibrated cameras differ')
                identity=tuple(pixel[k] for k in ['reference_float_RGB_SHA256','reference_CHW_uint8_SHA256','reference_alpha_SHA256'])
                require(reference_pixels.setdefault((sid,cid),identity)==identity,'cross-model reference RGB/alpha differs')
                require(pixel['archive'] not in input_hashes or input_hashes[pixel['archive']]==pixel['archive_SHA256'], 'conflicting actual archive hash')
                input_hashes[pixel['archive']]=pixel['archive_SHA256'];observations[alias].append(pixel)
                values=score['scores'];views[cid]={k:view[k] for k in ['PSNR','SSIM','LPIPS','M_IoU','foreground_PSNR']}
                views[cid].update(DreamSim=values['DreamSim_fullframe'],CLIP=values['CLIP_cosine_fullframe'],
                    DreamSim_GTcrop=values['DreamSim_GT_foreground_crop'],CLIP_GTcrop=values['CLIP_cosine_GT_foreground_crop'])
                require(np.isfinite(list(views[cid].values())).all()
                    and all(views[cid][k]>=-1e-6 for k in ['PSNR','LPIPS','foreground_PSNR'])
                    and -1-1e-6<=views[cid]['SSIM']<=1+1e-6 and 0<=views[cid]['M_IoU']<=1
                    and all(-1e-6<=views[cid][k]<=2+1e-6 for k in ['DreamSim','DreamSim_GTcrop'])
                    and all(-1-1e-6<=views[cid][k]<=1+1e-6 for k in ['CLIP','CLIP_GTcrop']),
                    'finite physical rendering and official perceptual metric ranges required')
            require(np.isfinite([metrics['chamfer_L1'],metrics['surface_FScore']]).all() and metrics['chamfer_L1']>=0
                and 0<=metrics['surface_FScore']<=1,'finite nonnegative CD and bounded surface FScore required')
            records[alias].append(dict(sample_id=sid,source_cluster_id=clusters[sid],views=views,CD=metrics['chamfer_L1'],FScore=metrics['surface_FScore']))
    stats=summarize_view_roles(records,contract['reported_contrasts'],draws=contract['source_cluster_bootstrap_draws'],seed=contract['source_cluster_bootstrap_seed'])
    return dict(status=EIGHT_STATUS,scope='final-test508',assets=508,source_clusters=134,models=models,statistics=stats,
        metric_contract=contract,primary_pairwise_family_intervals=_primary_intervals(records,population,contract),
        scored_image_observations=observations,input_SHA256=input_hashes,actual_completed_cases=508*len(models),
        actual_perceptual_views=1524*len(models),source_cluster_resampling_unit=True,
        independent_pipeline_training_seeds=1,independent_pipeline_seed_confirmation_proven=False,
        model_selection_from_test=False,FID_KID_not_yet_measured=True,full_goal_complete=False)


def combine_ten(eight,distribution,*,parent,controls):
    models=compose_models(parent,controls);pop=parent['population'];contract=validate_contract(parent)
    require(eight.get('status')==EIGHT_STATUS and distribution.get('status')==DIST_STATUS
        and exact(eight.get('models'),models) and exact(distribution.get('models'),models)
        and exact(eight.get('metric_contract'),contract) and set(distribution['distribution_metrics'])==set(models),
        'actual same frozen deduplicated4/5 physical model eight-metric and official distribution results required')
    join=validate_metric_image_join(eight,distribution,sample_ids=pop['sample_ids'],clusters=pop['source_clusters_by_sample'],models=models)
    table={}
    for alias in models:
        roles=eight['statistics']['roles'];src=roles['source9_10']['means'][alias]['deformation_weighted']
        tgt=roles['target8']['means'][alias]['deformation_weighted'];geom=roles['all8_9_10']['means'][alias]['deformation_weighted']
        d=distribution['distribution_metrics'][alias]
        require(d['images_per_pool']==1524 and d['feature_dimension']==2048 and d['package']=='torch-fidelity0.3.0'
            and d['statistic_dtype']=='float64' and d['KID_subset_std_is_not_CI'] is True
            and d['unbiased_KID_clamped'] is False and d['unequal_pool_counts_allowed'] is False
            and all(d[k]==contract['FID_KID'][field] for k,field in [('kid_subsets','subsets'),('kid_subset_size','subset_size'),('seed','seed')]),
            'same-count official complete fullframe FID/KID and original statistical settings required')
        table[alias]=dict(LPIPS=src['LPIPS'],DreamSim=src['DreamSim'],CLIP=src['CLIP'],
            FID=d['frechet_inception_distance'],KID=d['kernel_inception_distance_mean'],
            PSNR=tgt['PSNR'],SSIM=tgt['SSIM'],M_IoU=tgt['M_IoU'],CD=geom['CD'],FScore=geom['FScore'])
        require(np.isfinite(list(table[alias].values())).all(),'all ten final table metrics must be finite')
    return dict(status=TEN_STATUS,scope='final-test508',assets=508,source_clusters=134,models=models,
        primary_ten_metric_table=table,same_image_join=join,role_statistics=eight['statistics'],
        primary_pairwise_family_intervals=eight['primary_pairwise_family_intervals'],metric_contract=contract,
        distribution_metrics=distribution['distribution_metrics'],FID_KID_CI_or_significance_claim=False,
        KID_subset_std_not_source_cluster_CI=True,model_selection_from_test=False,
        independent_pipeline_training_seeds=1,independent_pipeline_seed_confirmation_proven=False,
        automatic_model_adoption=False,full_goal_complete=False)
