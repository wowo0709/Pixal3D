import copy

import pytest

from experiments.deformnext_retargeting.archived_native_guides import POLICY
from experiments.deformnext_retargeting.common_pipeline_contract import ARMS,CASE_STATUS
from experiments.deformnext_retargeting.common_pipeline_summary import summarize_common_pipeline


def inputs():
    ids=[str(i) for i in range(32)];clusters={s:str(int(s)//2) for s in ids}
    cases={a:{} for a in ARMS};g={a:{} for a in ARMS};r={a:{} for a in ARMS};pixels={};gt={};official=[]
    for arm in ARMS:
        for sid in ids:
            cases[arm][sid]=dict(status=CASE_STATUS,SS_arm=arm,sample_id=sid,source_cluster_id=clusters[sid],
                actual_common_PBR_checkpoint_SHA256='PBR',actual_common_PBR_mode='vanilla',actual_common_PBR_updates=1900,
                original_Shape5124210_Shape102416850=True,GT_geometry_used_in_generation=False,per_output_alignment=False,
                new_SS_neural_forwards=0,new_PBR_texture_calls=1,SS_prediction_SHA256='SS'+arm+sid,
                actual_owned_terminals=[dict(returncode=0,actual_session_drained=True) for _ in range(3)])
            g[arm][sid]=dict(status='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION',
                sample_id=sid,PBR_checkpoint='/trained/PBR',SS_prediction_SHA256='SS'+arm+sid,
                GT_geometry_or_coords_used_in_generation=False,per_output_alignment=False,GT_loaded_only_after_generation=True,
                source_defined_normalization=True,texture_support_policy=POLICY,
                actual_native_guide_logits_order_scale_and_dead_branches_preserved=True,
                numerical_entry_AST_device_literal_only=dict(all_other_AST_statements_unchanged=True),
                metrics=dict(per_output_alignment=False,threshold=.02,chamfer_L1=.01,surface_FScore=.9))
            views=[]
            for cid in [8,9,10]:
                identity=('rgb'+sid+str(cid),'alpha'+sid+str(cid));pixel='prediction'+arm+sid+str(cid)
                pixels[arm,sid,cid]=pixel;gt[arm,sid,cid]=identity
                views.append(dict(camera_id=cid,fixed_GT_RGB_SHA256=identity[0],fixed_GT_alpha_SHA256=identity[1],
                    identical_camera_matrices=True,source_material_identity_preserved=True,
                    PSNR=25.,SSIM=.9,LPIPS=.1,M_IoU=.8,foreground_PSNR=20.))
                official.append(dict(arm=arm,sample_id=sid,camera_id=cid,source_cluster_id=clusters[sid],shared_npz_SHA256=pixel,
                    image_audit=dict(original_float_GT_SHA256=identity[0],original_GT_alpha_SHA256=identity[1],
                        GT_only_crop_box=[0,0,256,256],converted_image_SHA256=dict(reference_full='same',reference_GTcrop='samecrop')),
                    scores=dict(DreamSim_fullframe=.2,CLIP_cosine_fullframe=.9,DreamSim_GT_foreground_crop=.1,CLIP_cosine_GT_foreground_crop=.95)))
            r[arm][sid]=dict(status='PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE',sample_id=sid,
                same_reference_pixels_for_every_model=True,per_view=views)
    report=dict(status='PASS_ACTUAL_OFFICIAL_DREAMSIM_CLIP_RENDER_EVALUATION',metric_registry_SHA256='registry',
        model_precision='float32,TF32off,noAMP',model_weights=dict(official='weights'),full_panel=True,
        fullframe_primary=True,separate_GT_only_crop_supplementary=True,no_model_training=True,global_environment_changed=False,
        GPU_sanity=dict(status='PASS_ACTUAL_GPU_OFFICIAL_METRIC_SANITY',CPU_model_forward_or_training_updates=0),per_view=official)
    kw=dict(sample_ids=ids,clusters=clusters,pixel_SHA256=pixels,pixel_GT=gt,common_PBR_checkpoint='/trained/PBR',
        common_PBR_SHA256='PBR',metric_registry_SHA256='registry',draws=100)
    return cases,g,r,report,kw


def test_same_pixels_have_zero_effects_while_preserving_different_SS_predictions():
    c,g,r,p,kw=inputs();result=summarize_common_pipeline(c,g,r,p,**kw)
    assert len(result['models'])==5 and result['actual_official_views']==480
    assert result['no_legacy14_or_native8_completion_dependency']
    assert len(result['roles']['target8']['contrasts'])==5
    assert all(v==0 for d in result['roles']['all8_9_10']['contrasts'].values() for v in d['deformation_weighted_delta'].values())
    assert not result['model_adopted'] and not result['full_goal_complete']


@pytest.mark.parametrize('damage',['wrong-PBR','GT-geometry','wrong-SS','wrong-guides','GT-pixels','image-score',
    'unclosed-child','missing-view','wrong-camera','CPU-forward','wrong-registry','material'])
def test_rejects_false_end_to_end_module_attribution(damage):
    c,g,r,p,kw=inputs();a='C_TOP1_V';s='0';view=r[a][s]['per_view'][0]
    if damage=='wrong-PBR':c[a][s]['actual_common_PBR_checkpoint_SHA256']='other'
    elif damage=='GT-geometry':g[a][s]['GT_geometry_or_coords_used_in_generation']=True
    elif damage=='wrong-SS':g[a][s]['SS_prediction_SHA256']='other'
    elif damage=='wrong-guides':g[a][s]['texture_support_policy']='recomputed-different-support'
    elif damage=='GT-pixels':kw['pixel_GT'][a,s,8]=('other','alpha')
    elif damage=='image-score':p['per_view'][0]['shared_npz_SHA256']='other'
    elif damage=='unclosed-child':c[a][s]['actual_owned_terminals'][0]['actual_session_drained']=False
    elif damage=='missing-view':p['per_view'].pop()
    elif damage=='wrong-camera':view['identical_camera_matrices']=False
    elif damage=='CPU-forward':p['GPU_sanity']['CPU_model_forward_or_training_updates']=1
    elif damage=='wrong-registry':p['metric_registry_SHA256']='other'
    else:view['source_material_identity_preserved']=False
    with pytest.raises(ValueError):summarize_common_pipeline(c,g,r,p,**kw)
