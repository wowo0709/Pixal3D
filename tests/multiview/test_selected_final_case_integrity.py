"""Physical-family and real-owner counterexamples using synthetic native reports."""
from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.archived_native_guides import POLICY
from experiments.deformnext_retargeting.common_geometry_pbr_contract import MODES as PBR_MODES
from experiments.deformnext_retargeting.native_final_case_integrity import paired_seed
from experiments.deformnext_retargeting.pbr_replay_artifacts import GEOMETRY_FIELDS
from experiments.deformnext_retargeting.pipeline_population_contract_v2 import ARMS
from experiments.deformnext_retargeting.selected_final_case_integrity import CASE_STATUS,validate_case_reports
from experiments.deformnext_retargeting.selected_final_case_plans import case_plans
from test_selected_final_case_plans import inputs
from test_selected_final_population import evaluation


def reports(tmp_path,arm='E_LEARNED_RAW',mode='top1-only',alias=None):
    args=inputs(tmp_path,arm,mode,alias=alias);case=case_plans(**args);model=case['model'];sid=case['sample_id']
    zero=model['kind']=='original-zero';ss=evaluation(model,args['parent'],case['alias'])
    ss['native_SS_artifacts']['artifacts'][0]['prediction_voxels_SHA256']=args['SS_artifact']['prediction_voxels_SHA256']
    metrics=dict(prediction_samples=20000,target_samples=20000,threshold=.02,per_output_alignment=False,CD=.01,FScore=.9)
    geometry=dict(status='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION',sample_id=sid,
        SS_source_evaluation=case['SS_evaluation']['path'],SS_prediction_SHA256=args['SS_artifact']['prediction_voxels_SHA256'],
        GT_loaded_only_after_generation=True,GT_geometry_or_coords_used_in_generation=False,
        source_defined_normalization=True,per_output_alignment=False,actual_resolution=1024,fixed_shape_models=True,
        original_stage_configs_and_weights_unchanged=True,PBR_checkpoint=model['geometry_stages']['pbr1024']['checkpoint']['path'],
        texture_support_policy=POLICY,actual_native_guide_archive_SHA256='a'*64,
        actual_native_guide_logits_order_scale_and_dead_branches_preserved=True,shape_decoder_recomputed=True,
        actual_native_guide_events=[dict(actual_guide_count=4)],metrics=metrics)
    if zero:
        geometry.update(original_zero_shot_full_stack=True,deformation_finetuning_updates=0,
            original_multiview_steps=dict(ss64=4210,shape512=4210,shape1024=16850,pbr1024=16850))
        final=deepcopy(geometry)
    else:
        pbr=model['pbr1024'];final=dict(status='PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE',sample_id=sid,
            PBR_arm=model['PBR_mode'],checkpoint_kind='trained-native1900',engineering_only=False,
            PBR_checkpoint=pbr['checkpoint']['path'],PBR_checkpoint_SHA256=pbr['checkpoint']['SHA256'],
            conditioner_checkpoint=None if pbr['conditioner_checkpoint'] is None else pbr['conditioner_checkpoint']['path'],
            PBR_replay_seed=paired_seed(sid,'texture'),actual_initial_texture_noise_SHA256='c'*64,
            GT_geometry_or_coords_used_in_generation=False,GT_sparse_latents_or_projection_cache_used=False,
            GT_evaluation_artifact_copied_only_after_prediction_saved=True,source_weights_unchanged=True,
            conditioning_preserves_CPU_and_CUDA_RNG=True,optimizer_updates=0,raw_surface_metrics_unchanged=True,
            geometry_array_byte_identity=dict.fromkeys(GEOMETRY_FIELDS,True),
            SS_source_evaluation=geometry['SS_source_evaluation'],SS_prediction_SHA256=geometry['SS_prediction_SHA256'],
            generated_shape_source=case['geometry_source_root'],metrics=deepcopy(metrics),texture_support_policy=POLICY,
            actual_native_guide_archive_SHA256='a'*64,actual_native_guide_logits_order_scale_and_dead_branches_preserved=True,
            shape_decoder_recomputed=False)
    surface=dict(status='PASS_ACTUAL_LOCAL_NATIVE_GEOMETRY_SHARED_GT_SURFACE_JOIN',sample_id=sid,
        source_cluster_id=case['source_cluster_id'],native_metrics=deepcopy(metrics))
    render=dict(status='PASS_ACTUAL_CASCADE_MULTIVIEW_RGB_ALPHA_AND_CAMERA_AUDIT',sample_id=sid,
        per_view=[dict(camera_id=c,GT_geometry_RGBA_containment=1.) for c in [8,9,10]],
        prediction_geometry_postprocess=dict(same_budget_for_all_predictions=True,render_faces=1000000))
    first=case['alias']=='zero-source4'
    shared=dict(status='PASS_ACTUAL_SHARED_RENDERER_MATERIAL_REFERENCE' if first else 'PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE',
        per_view=[dict(camera_id=c,identical_camera_matrices=True) for c in [8,9,10]],
        original_materials_and_UVs=True,GT_VAE_reconstruction_not_used=True,shared_renderer_lighting_camera_AO_seed=True,
        GT_not_used_for_model_generation=True,maximum_GT_world_vertex_nearest_difference=0.,same_reference_pixels_for_every_model=True)
    if not first:shared['sample_id']=sid
    phases=([] if case['geometry_reference'] is not None else ['geometry'])+([] if zero else ['PBR'])+['render','shared']
    terminals=[dict(PID=i+1,birth=100.+i,returncode=0,phase=name) for i,name in enumerate(phases)]
    owner=None
    if case['geometry_reference'] is not None:
        owner_case,owner_kwargs=reports(tmp_path,arm,mode,alias=case['geometry_reference']['alias'])
        owner=validate_case_reports(owner_case,**owner_kwargs)
    kwargs=dict(parent=args['parent'],controls=args['controls'],SS_evaluation=ss,geometry=geometry,final=final,
        render=render,shared=shared,surface=surface,terminals=terminals,geometry_reference_audit=owner)
    return case,kwargs


@pytest.mark.parametrize('arm',ARMS)
@pytest.mark.parametrize('mode',PBR_MODES)
def test_each_actual_physical_composition_keeps_own_geometry_PBR_and_shared_reference(tmp_path,arm,mode):
    case,kwargs=reports(tmp_path,arm,mode);before=deepcopy((case,kwargs));result=validate_case_reports(case,**kwargs)
    assert before==(case,kwargs) and result['status']==CASE_STATUS
    assert result['frozen_model_binding']['SS_arm']==arm and result['frozen_model_binding']['PBR_mode']==mode
    assert result['full_goal_complete'] is False and result['quality_gain_proven'] is False


@pytest.mark.parametrize('alias',['zero-source4','zero-edited4'])
def test_original_zero_controls_are_distinct_original_four_stage_stacks(tmp_path,alias):
    case,kwargs=reports(tmp_path,alias=alias);result=validate_case_reports(case,**kwargs)
    assert result['actual_completed_child_phases']==3 and result['geometry_reference'] is None


@pytest.mark.parametrize('damage',['model-alias','PBR-family','PBR-weight','SS-prediction','guide-hash','guide-count',
    'Shape-redecode','missing-geometry-field','changed-geometry','GT-input','GT-copy-before-prediction',
    'noise','surface','failed-child','reused-PID','camera','budget','reference-pixels'])
def test_different_physical_modules_geometry_or_GT_protocol_cannot_pass_as_selected_pipeline(tmp_path,damage):
    case,kw=reports(tmp_path)
    if damage=='model-alias':case['model']['SS_arm']='FULL'
    elif damage=='PBR-family':kw['final']['PBR_arm']='vanilla'
    elif damage=='PBR-weight':kw['final']['PBR_checkpoint_SHA256']='f'*64
    elif damage=='SS-prediction':kw['geometry']['SS_prediction_SHA256']='f'*64
    elif damage=='guide-hash':kw['final']['actual_native_guide_archive_SHA256']='f'*64
    elif damage=='guide-count':kw['geometry']['actual_native_guide_events'][0]['actual_guide_count']=3
    elif damage=='Shape-redecode':kw['final']['shape_decoder_recomputed']=True
    elif damage=='missing-geometry-field':kw['final']['geometry_array_byte_identity'].pop('shape_feats')
    elif damage=='changed-geometry':kw['final']['geometry_array_byte_identity']['shape_feats']=False
    elif damage=='GT-input':kw['final']['GT_sparse_latents_or_projection_cache_used']=True
    elif damage=='GT-copy-before-prediction':kw['final']['GT_evaluation_artifact_copied_only_after_prediction_saved']=False
    elif damage=='noise':kw['final']['PBR_replay_seed']+=1
    elif damage=='surface':kw['surface']['native_metrics']['threshold']=.03
    elif damage=='failed-child':kw['terminals'][-1]['returncode']=1
    elif damage=='reused-PID':kw['terminals'][-1].update(PID=kw['terminals'][0]['PID'],birth=kw['terminals'][0]['birth'])
    elif damage=='camera':kw['render']['per_view'][-1]['camera_id']=11
    elif damage=='budget':kw['render']['prediction_geometry_postprocess']['render_faces']=1000001
    else:kw['shared']['same_reference_pixels_for_every_model']=False
    with pytest.raises(ValueError):validate_case_reports(case,**kw)


@pytest.mark.parametrize('damage',[None,'missing-owner','other-SS','other-weights','other-noise','other-sample',
    'unfinished-owner','failed-owner-child','other-source','other-guides'])
def test_shared_SS_reuses_only_real_completed_owner_geometry_and_native_guides(tmp_path,damage):
    case,kw=reports(tmp_path,'V1','top1-only');owner=kw['geometry_reference_audit']
    if damage=='missing-owner':kw['geometry_reference_audit']=None
    elif damage=='other-SS':owner['geometry_group']='FULL'
    elif damage=='other-weights':owner['frozen_model_binding']['ss64']['checkpoint']['SHA256']='f'*64
    elif damage=='other-noise':owner['actual_SS_prediction_SHA256']='f'*64
    elif damage=='other-sample':owner['sample_id']='test1'
    elif damage=='unfinished-owner':owner['actual_completed_native_case']=False
    elif damage=='failed-owner-child':owner['terminals'][0]['returncode']=1
    elif damage=='other-source':owner['source_cluster_id']='other'
    elif damage=='other-guides':owner['actual_native_guide_archive_SHA256']='f'*64
    if damage:
        with pytest.raises(ValueError):validate_case_reports(case,**kw)
    else:
        result=validate_case_reports(case,**kw)
        assert result['actual_completed_child_phases']==3
        assert result['geometry_reference']['alias']=='V1'
