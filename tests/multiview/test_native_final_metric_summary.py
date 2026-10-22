from copy import deepcopy

import numpy as np
import pytest

from experiments.deformnext_retargeting.native_final_case_plans import ALIASES
from experiments.deformnext_retargeting.native_final_metric_summary import eight_metrics,combine_ten,validate_feature_pool
from test_native_final_case_plans import models


def inputs():
    parent,_,m=models();pop=parent['population'];ids=pop['sample_ids']
    cases={};shared={};perceptual={}
    for ai,alias in enumerate(ALIASES):
        cases[alias]={};shared[alias]={};score_rows=[]
        for sid in ids:
            cluster=pop['source_clusters_by_sample'][sid]
            metric=dict(prediction_samples=20000,target_samples=20000,threshold=.02,per_output_alignment=False,
                        chamfer_L1=.01+ai*.001,surface_FScore=.8-ai*.01)
            surface=dict(status='PASS_ACTUAL_LOCAL_NATIVE_GEOMETRY_SHARED_GT_SURFACE_JOIN',sample_id=sid,source_cluster_id=cluster,
                native_metrics=metric,GT_reference_not_used_for_model_conditioning=True,GT_reference={'SHA256':'surface-'+sid},target_points_SHA256='points-'+sid)
            observations=[];views=[]
            for cid in [8,9,10]:
                archive='/out/'+alias+'/'+sid+f'/camera{cid}.npz'
                observations.append(dict(sample_id=sid,camera_id=cid,source_cluster_id=cluster,archive=archive,archive_SHA256='archive-'+archive,
                    prediction_float_RGB_SHA256='prediction-'+archive,prediction_CHW_uint8_SHA256='converted-'+archive,
                    reference_float_RGB_SHA256='GT-'+sid+str(cid),reference_CHW_uint8_SHA256='GT8-'+sid+str(cid),reference_alpha_SHA256='alpha-'+sid+str(cid)))
                views.append(dict(camera_id=cid,PSNR=20.+ai,SSIM=.8+ai*.01,LPIPS=.2-ai*.01,M_IoU=.7+ai*.01,foreground_PSNR=12.+ai))
                score_rows.append(dict(arm=alias,sample_id=sid,source_cluster_id=cluster,camera_id=cid,shared_npz_SHA256='archive-'+archive,
                    scores=dict(DreamSim_fullframe=.3-ai*.01,CLIP_cosine_fullframe=.8+ai*.01,DreamSim_GT_foreground_crop=.4,CLIP_cosine_GT_foreground_crop=.7)))
            cases[alias][sid]=dict(status='PASS_ACTUAL_FROZEN_NATIVE_SIX_MODEL_TEST508_CASE',actual_completed_native_case=True,alias=alias,
                sample_id=sid,source_cluster_id=cluster,frozen_model_binding=m[alias],
                actual_SS_checkpoint_SHA256=m[alias]['ss64']['checkpoint']['SHA256'],actual_final_PBR_checkpoint_SHA256=m[alias]['pbr1024']['checkpoint']['SHA256'],
                GT_not_used_in_generation=True,terminals=[{'returncode':0}]*(3 if alias.startswith('zero-')else 4),
                surface_reference_join=surface,raw_surface_metrics=metric,scored_image_observations=observations)
            shared[alias][sid]=dict(status='PASS_ACTUAL_SHARED_RENDERER_MATERIAL_REFERENCE'if alias=='zero-source4'else'PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE',per_view=views)
        perceptual[alias]=dict(status='PASS_ACTUAL_OFFICIAL_DREAMSIM_CLIP_RENDER_EVALUATION',metric_registry_SHA256='registry',model_precision='float32,TF32off,noAMP',
            full_panel=True,no_model_training=True,global_environment_changed=False,
            GPU_sanity=dict(status='PASS_ACTUAL_GPU_OFFICIAL_METRIC_SANITY',CPU_model_forward_or_training_updates=0),per_view=score_rows)
    return dict(models=m,population=pop,cases=cases,shared=shared,perceptual=perceptual,metric_registry_SHA256='registry',draws=16)


def distribution(eight):
    observations={};reference=[]
    for alias,rows in eight['scored_image_observations'].items():
        observations[alias]=[dict(sample_id=r['sample_id'],camera_id=r['camera_id'],source_cluster_id=r['source_cluster_id'],
            archive=r['archive'],archive_SHA256=r['archive_SHA256'],array='prediction',original_float_RGB_SHA256=r['prediction_float_RGB_SHA256'],
            converted_CHW_uint8_SHA256=r['prediction_CHW_uint8_SHA256'])for r in rows]
        if alias=='zero-source4':reference=[dict(sample_id=r['sample_id'],camera_id=r['camera_id'],source_cluster_id=r['source_cluster_id'],
            archive=r['archive'],archive_SHA256=r['archive_SHA256'],array='GT_shared_renderer',original_float_RGB_SHA256=r['reference_float_RGB_SHA256'],
            converted_CHW_uint8_SHA256=r['reference_CHW_uint8_SHA256'])for r in rows]
    return dict(status='PASS_ACTUAL_NATIVE_SIX_MODEL_TEST508_FID_KID',models=eight['models'],feature_image_observations=observations,
        reference_image_observations=reference,distribution_metrics={a:dict(images_per_pool=1524,feature_dimension=2048,package='torch-fidelity0.3.0',
            KID_subset_std_is_not_CI=True,unbiased_KID_clamped=False,frechet_inception_distance=2.,kernel_inception_distance_mean=-.001)for a in ALIASES})


def test_all_ten_primary_metrics_keep_source_target_global_roles_and_same_pixels():
    args=inputs();eight=eight_metrics(**args);result=combine_ten(eight,distribution(eight),population=args['population'])
    assert result['status']=='PASS_ACTUAL_NATIVE_SIX_MODEL_TEST508_ALL_TEN_METRICS'
    assert len(result['primary_ten_metric_table'])==6
    assert result['primary_ten_metric_table']['vanilla']['KID']==-.001
    assert result['same_image_join']['prediction_images']==9144
    assert not result['automatic_model_adoption']and not result['full_goal_complete']


@pytest.mark.parametrize('damage',['missing-case','wrong-weights','failed-child','different-GT-pixels','different-GT-surface','wrong-official-view','wrong-camera','wrong-registry'])
def test_partial_or_wrong_model_cases_cannot_become_full508_metrics(damage):
    a=inputs();alias='R_top1_only';sid=a['population']['sample_ids'][0];case=a['cases'][alias][sid]
    if damage=='missing-case':a['cases'][alias].pop(sid)
    elif damage=='wrong-weights':case['actual_final_PBR_checkpoint_SHA256']='wrong'
    elif damage=='failed-child':case['terminals'][0]={'returncode':1}
    elif damage=='different-GT-pixels':case['scored_image_observations'][0]['reference_float_RGB_SHA256']='other'
    elif damage=='different-GT-surface':case['surface_reference_join']['target_points_SHA256']='other'
    elif damage=='wrong-official-view':a['perceptual'][alias]['per_view'][0]['shared_npz_SHA256']='other'
    elif damage=='wrong-camera':a['shared'][alias][sid]['per_view'][0]['camera_id']=11
    elif damage=='wrong-registry':a['perceptual'][alias]['metric_registry_SHA256']='other'
    with pytest.raises(ValueError):eight_metrics(**a)


def test_distribution_cannot_use_another_image_or_conversion():
    a=inputs();eight=eight_metrics(**a);d=distribution(eight)
    d['feature_image_observations']['vanilla'][0]['converted_CHW_uint8_SHA256']='other'
    with pytest.raises(ValueError):combine_ten(eight,d,population=a['population'])


def test_feature_pool_requires_actual_official_full2048_same_model_metadata():
    audit=dict(status='PASS_ACTUAL_OFFICIAL_INCEPTION2048_FEATURE_POOL',scope='final-test508',images=1524,assets=508,
        package='torch-fidelity0.3.0',feature_layer='2048',feature_dtype='float32',fullframe=True,TF32=False,autocast=False,
        same_batch_repeat_exact=True,optimizer_updates=0,model_binding={'NN':'actual'},weights_SHA256='weights',observations=['images'])
    values=np.ones((1524,2048),dtype=np.float32)
    validate_feature_pool(audit,values,observations=['images'],model_binding={'NN':'actual'},weights_SHA256='weights')
    with pytest.raises(ValueError):validate_feature_pool(audit,values[:100],observations=['images'],model_binding={'NN':'actual'},weights_SHA256='weights')
