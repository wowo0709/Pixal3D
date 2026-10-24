"""Synthetic full-population metadata/statistics; no neural metric execution."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.deformnext_retargeting.selected_campaign_resources import geometry_owners
from experiments.deformnext_retargeting.selected_final_case_integrity import CASE_STATUS
from experiments.deformnext_retargeting.selected_final_metric_summary import (DIST_STATUS,EIGHT_STATUS,TEN_STATUS,
    combine_ten,eight_metrics,metric_contract,reference_binding,validate_contract)
from experiments.deformnext_retargeting.selected_final_population import compose_models
from test_selected_final_population import parents


def inputs(tmp_path,arm='E_LEARNED_RAW',mode='vanilla'):
    parent,controls=parents(tmp_path,arm,mode);parent['evaluation_protocol']['selected_final_metric_contract']=metric_contract(parent,draws=16)
    models=compose_models(parent,controls);pop=parent['population'];cases={};shared={};perceptual={};owners=geometry_owners(models)
    for ai,(alias,model) in enumerate(models.items()):
        cases[alias]={};shared[alias]={};rows=[];zero=model['kind']=='original-zero';reuse=owners[alias]!=alias
        phases=([] if reuse else ['geometry'])+([] if zero else ['PBR'])+['render','shared']
        for sid in pop['sample_ids']:
            cluster=pop['source_clusters_by_sample'][sid];metrics=dict(prediction_samples=20000,target_samples=20000,
                threshold=.02,per_output_alignment=False,chamfer_L1=.01+ai*.001,surface_FScore=.8-ai*.01)
            surface=dict(status='PASS_ACTUAL_LOCAL_NATIVE_GEOMETRY_SHARED_GT_SURFACE_JOIN',sample_id=sid,source_cluster_id=cluster,
                native_metrics=metrics,GT_reference_not_used_for_model_conditioning=True,GT_reference={'SHA256':'surface-'+sid},target_points_SHA256='points-'+sid)
            observations=[];views=[]
            for cid in [8,9,10]:
                archive='/fixture/'+alias+'/'+sid+f'/camera{cid:02d}.npz'
                pixel=dict(sample_id=sid,source_cluster_id=cluster,camera_id=cid,archive=archive,archive_SHA256='archive-'+archive,
                    prediction_float_RGB_SHA256='pred-'+archive,prediction_CHW_uint8_SHA256='pred8-'+archive,
                    reference_float_RGB_SHA256='GT-'+sid+str(cid),reference_CHW_uint8_SHA256='GT8-'+sid+str(cid),reference_alpha_SHA256='mask-'+sid+str(cid))
                observations.append(pixel)
                views.append(dict(camera_id=cid,PSNR=(10. if cid==8 else 20.)+ai,SSIM=.8,LPIPS=(.4 if cid==8 else .2)+ai*.01,
                    M_IoU=.7,foreground_PSNR=12.,fixed_GT_RGB_SHA256=pixel['reference_float_RGB_SHA256'],
                    fixed_GT_alpha_SHA256=pixel['reference_alpha_SHA256'],identical_camera_matrices=True))
                rows.append(dict(arm=alias,sample_id=sid,source_cluster_id=cluster,camera_id=cid,shared_npz_SHA256=pixel['archive_SHA256'],
                    image_audit=dict(original_float_prediction_SHA256=pixel['prediction_float_RGB_SHA256'],original_float_GT_SHA256=pixel['reference_float_RGB_SHA256'],
                        original_GT_alpha_SHA256=pixel['reference_alpha_SHA256'],per_output_alignment=False),
                    scores=dict(DreamSim_fullframe=.3+ai*.01,CLIP_cosine_fullframe=.8-ai*.01,DreamSim_GT_foreground_crop=.4,CLIP_cosine_GT_foreground_crop=.7)))
            cases[alias][sid]=dict(status=CASE_STATUS,actual_completed_native_case=True,alias=alias,sample_id=sid,source_cluster_id=cluster,
                frozen_model_binding=model,GT_not_used_in_generation=True,optimizer_updates=0,
                geometry_reference={'alias':owners[alias]} if reuse else None,terminals=[dict(phase=p,returncode=0,actual_session_drained=True) for p in phases],
                surface_reference_join=surface,raw_surface_metrics=metrics,scored_image_observations=observations)
            shared[alias][sid]=dict(status='PASS_ACTUAL_SHARED_RENDERER_MATERIAL_REFERENCE' if alias=='zero-source4' else 'PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE',per_view=views)
        perceptual[alias]=dict(status='PASS_ACTUAL_OFFICIAL_DREAMSIM_CLIP_RENDER_EVALUATION',metric_registry_SHA256=parent['evaluation_protocol']['metric_registry']['SHA256'],
            scope='final-test508',model_precision='float32,TF32off,noAMP',full_panel=True,fullframe_primary=True,no_model_training=True,global_environment_changed=False,
            GPU_sanity=dict(status='PASS_ACTUAL_GPU_OFFICIAL_METRIC_SANITY',CPU_model_forward_or_training_updates=0),per_view=rows)
    return dict(parent=parent,controls=controls,cases=cases,shared=shared,perceptual=perceptual)


def distribution(eight):
    observations={};reference=[]
    for alias,rows in eight['scored_image_observations'].items():
        observations[alias]=[dict(sample_id=r['sample_id'],camera_id=r['camera_id'],source_cluster_id=r['source_cluster_id'],archive=r['archive'],archive_SHA256=r['archive_SHA256'],
            array='prediction',original_float_RGB_SHA256=r['prediction_float_RGB_SHA256'],converted_CHW_uint8_SHA256=r['prediction_CHW_uint8_SHA256']) for r in rows]
        if alias=='zero-source4':
            reference=[dict(sample_id=r['sample_id'],camera_id=r['camera_id'],source_cluster_id=r['source_cluster_id'],archive=r['archive'],archive_SHA256=r['archive_SHA256'],
                array='GT_shared_renderer',original_float_RGB_SHA256=r['reference_float_RGB_SHA256'],converted_CHW_uint8_SHA256=r['reference_CHW_uint8_SHA256']) for r in rows]
    metrics={name:dict(images_per_pool=1524,feature_dimension=2048,package='torch-fidelity0.3.0',statistic_dtype='float64',
        KID_subset_std_is_not_CI=True,unbiased_KID_clamped=False,unequal_pool_counts_allowed=False,kid_subsets=100,kid_subset_size=1000,seed=42017,
        frechet_inception_distance=2.,kernel_inception_distance_mean=-.001) for name in eight['models']}
    return dict(status=DIST_STATUS,models=eight['models'],distribution_metrics=metrics,feature_image_observations=observations,reference_image_observations=reference)


@pytest.mark.parametrize('arm,mode,expected',[('E_LEARNED_RAW','vanilla',5),('V1','top1-only',5),('V1','vanilla',4),('C_DECODER','top1-only',4)])
def test_actual_physical_model_count_role_metrics_and_baseline_winner_are_retained(tmp_path,arm,mode,expected):
    args=inputs(tmp_path,arm,mode);before=deepcopy(args);eight=eight_metrics(**args)
    assert args==before and eight['status']==EIGHT_STATUS and len(eight['models'])==expected
    assert eight['actual_completed_cases']==expected*508 and eight['actual_perceptual_views']==expected*1524
    combined=combine_ten(eight,distribution(eight),parent=args['parent'],controls=args['controls'])
    assert combined['status']==TEN_STATUS and len(combined['primary_ten_metric_table'])==expected
    assert combined['primary_ten_metric_table']['zero-source4']['LPIPS']==pytest.approx(.2)
    assert combined['primary_ten_metric_table']['zero-source4']['PSNR']==10.
    assert combined['primary_ten_metric_table']['zero-source4']['KID']==-.001
    assert len(next(iter(combined['primary_ten_metric_table'].values())))==10
    assert not combined['independent_pipeline_seed_confirmation_proven'] and not combined['full_goal_complete']
    count=1 if expected==4 else 2
    assert len(combined['primary_pairwise_family_intervals'])==count
    assert all(v['endpoint_family_size']==8*count for v in combined['primary_pairwise_family_intervals'].values())
    reference=reference_binding(args['parent'],args['controls']);assert reference['reference_producer']=='zero-source4'


@pytest.mark.parametrize('damage',['wrong-physical-arm','missing-model','missing-case','failed-phase','missing-producer','extra-redecode','GT-conditioning',
    'surface-points','surface-metrics','duplicate-camera','official-registry','official-GPU','score-pixels','score-mask','score-alignment','render-GT','render-camera','nonfinite','invalid-range'])
def test_physical_model_surface_camera_pixel_and_metric_protocol_mismatches_are_rejected(tmp_path,damage):
    args=inputs(tmp_path,'V1','top1-only');sid=args['parent']['population']['sample_ids'][0];case=args['cases']['SELECTED'][sid]
    score=args['perceptual']['SELECTED']['per_view'][0]
    if damage=='wrong-physical-arm':case['frozen_model_binding']=deepcopy(args['cases']['RETARGET'][sid]['frozen_model_binding'])
    elif damage=='missing-model':args['cases'].pop('RETARGET')
    elif damage=='missing-case':args['cases']['SELECTED'].pop(sid)
    elif damage=='failed-phase':case['terminals'][0]['returncode']=1
    elif damage=='missing-producer':args['cases']['V1'][sid]['terminals'].pop(0)
    elif damage=='extra-redecode':case['terminals'].insert(0,dict(phase='geometry',returncode=0,actual_session_drained=True))
    elif damage=='GT-conditioning':case['GT_not_used_in_generation']=False
    elif damage=='surface-points':case['surface_reference_join']['target_points_SHA256']='wrong-points'
    elif damage=='surface-metrics':case['surface_reference_join']['native_metrics']={}
    elif damage=='duplicate-camera':case['scored_image_observations'][1]['camera_id']=8
    elif damage=='official-registry':args['perceptual']['SELECTED']['metric_registry_SHA256']='wrong-registry'
    elif damage=='official-GPU':args['perceptual']['SELECTED']['GPU_sanity']['CPU_model_forward_or_training_updates']=1
    elif damage=='score-pixels':score['image_audit']['original_float_prediction_SHA256']='wrong-pixels'
    elif damage=='score-mask':score['image_audit']['original_GT_alpha_SHA256']='wrong-mask'
    elif damage=='score-alignment':score['image_audit']['per_output_alignment']=True
    elif damage=='render-GT':args['shared']['SELECTED'][sid]['per_view'][0]['fixed_GT_RGB_SHA256']='wrong-GT'
    elif damage=='render-camera':args['shared']['SELECTED'][sid]['per_view'][0]['identical_camera_matrices']=False
    elif damage=='nonfinite':score['scores']['DreamSim_fullframe']=np.nan
    else:args['shared']['SELECTED'][sid]['per_view'][0]['M_IoU']=1.1
    with pytest.raises(ValueError):eight_metrics(**args)


@pytest.mark.parametrize('damage',['metric-role','contrast','distribution-setting','crop-primary'])
def test_metrics_and_comparisons_cannot_change_after_frozen_selection(tmp_path,damage):
    args=inputs(tmp_path);contract=args['parent']['evaluation_protocol']['selected_final_metric_contract']
    if damage=='metric-role':contract['pairwise_metric_roles']['LPIPS']='target8'
    elif damage=='contrast':contract['primary_contrasts']=[]
    elif damage=='distribution-setting':contract['FID_KID']['subsets']=1
    else:contract['fullframe_primary']=False
    with pytest.raises(ValueError):validate_contract(args['parent'])


@pytest.mark.parametrize('damage',['missing-model','other-reference','prediction-conversion','negative-KID-clamped','unequal-pool','weights-protocol'])
def test_final_ten_join_rejects_other_images_dropped_models_or_nonofficial_distribution_settings(tmp_path,damage):
    args=inputs(tmp_path,'V1','vanilla');eight=eight_metrics(**args);d=distribution(eight)
    if damage=='missing-model':d['distribution_metrics'].pop('RETARGET')
    elif damage=='other-reference':d['reference_image_observations'][0]['original_float_RGB_SHA256']='other-GT'
    elif damage=='prediction-conversion':d['feature_image_observations']['V1'][0]['converted_CHW_uint8_SHA256']='other-conversion'
    elif damage=='negative-KID-clamped':d['distribution_metrics']['V1']['unbiased_KID_clamped']=True
    elif damage=='unequal-pool':d['distribution_metrics']['V1']['images_per_pool']=1523
    else:d['distribution_metrics']['V1']['seed']=42018
    with pytest.raises(ValueError):combine_ten(eight,d,parent=args['parent'],controls=args['controls'])
