from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.deformnext_retargeting.cascade_remote_results import array_identities
from experiments.deformnext_retargeting.native_final_case_integrity import admit_ss,local_surface_join,paired_seed,validate_case_reports
from experiments.deformnext_retargeting.native_final_case_plans import ALIASES
from test_native_final_case_plans import models


def ss_inputs(alias='R_top1_only'):
    parent,_,all_models=models();model=all_models[alias];ids=parent['population']['sample_ids']
    rows=[dict(sample_id=s,source_cluster_id=parent['population']['source_clusters_by_sample'][s],
        seed=parent['population']['generation_seeds'][s],target_voxels_sha256='GT',tp=3,fp=2,fn=1)for s in ids]
    artifacts=[dict(sample_id=r['sample_id'],seed=r['seed'],target_voxels_SHA256='GT',
        occupancy_counts=dict(tp=3,fp=2,fn=1))for r in rows]
    zero=alias.startswith('zero-');b=model['ss64']
    r=dict(status='PASS',stage='ss64',population_scope='final-test508',manifest_sha256='manifest',
        frozen_native_model_binding=b,checkpoint=b['checkpoint']['path'],step=0 if zero else 480,
        conditioner_checkpoint=None if b['conditioner_checkpoint']is None else b['conditioner_checkpoint']['path'],
        input_camera_ids=[8,9,10,11],input_states=['M0']*4 if alias=='zero-source4'else['M1','M0','M0','M0'],
        target_modality='rgb',optimizer_updates_during_evaluation=0,generation_batch_size=8,sampling_steps=12,guidance=3.,
        quick_panel_ids=ids,per_asset=rows,
        native_SS_artifacts=dict(status='PASS_ACTUAL_ORIGINAL_ZERO_SS_BOUND_POPULATION_ARCHIVE'if zero else
            'PASS_ACTUAL_NATIVE_BOUND_SS_POPULATION_ARCHIVE',archived_asset_count=508,artifacts=artifacts))
    if zero:r.update(deformation_finetuning_updates=0,original_multiview_SS_step=4210,
        actual_original_full_base_and_LoRA_validation=dict(original_full_base_and_LoRA_exact_before_evaluation=True))
    else:r.update(native_variant=alias,actual_source_tensors=0 if alias=='vanilla'else 10 if alias=='R_top1_only'else 19)
    return model,r,dict(parent=parent,manifest_SHA256='manifest')


@pytest.mark.parametrize('alias',ALIASES)
def test_admit_only_complete_same_frozen_SS508_archives(alias):
    model,result,kw=ss_inputs(alias)
    assert len(admit_ss(alias,model,result,**kw))==508


@pytest.mark.parametrize('damage',['old32','wrong-model','wrong-camera','different-seed','different-GT','wrong-source','different-manifest'])
def test_SS_population_admission_rejects_confounds(damage):
    m,r,kw=ss_inputs()
    if damage=='old32':r['native_SS_artifacts']['archived_asset_count']=32
    elif damage=='wrong-model':r['checkpoint']='/wrong.pt'
    elif damage=='wrong-camera':r['input_camera_ids']=[9,10,11]
    elif damage=='different-seed':r['per_asset'][0]['seed']+=1
    elif damage=='different-GT':r['native_SS_artifacts']['artifacts'][0]['target_voxels_SHA256']='wrong'
    elif damage=='wrong-source':r['actual_source_tensors']=19
    elif damage=='different-manifest':r['manifest_sha256']='validation'
    with pytest.raises(ValueError):admit_ss('R_top1_only',m,r,**kw)


def test_generation_noise_is_fixed_by_sample_and_purpose_not_model():
    assert paired_seed('asset','cascade')==paired_seed('asset','cascade')
    assert paired_seed('asset','cascade')!=paired_seed('asset','texture')
    assert 0<=paired_seed('asset','texture')<2**31


def surface_inputs(tmp_path):
    arrays=dict(target_world=np.arange(9,dtype=np.float32).reshape(3,3),target_faces=np.array([[0,1,2]]),
                source_center=np.zeros(3),source_extent=np.array(1.))
    path=tmp_path/'geometry.npz';np.savez(path,**arrays)
    calls=[]
    def expected(native):
        calls.append(1)
        return dict(source_cluster_id='cluster'),dict(path='/GT/ref.npz',SHA256='ref'),dict(
            native_GT_array_identities=array_identities(arrays),target_points_SHA256='points')
    refs=SimpleNamespace(spec={'path':'/GT/audit.json','SHA256':'audit'},expected=expected)
    native=dict(row=dict(sample_id='one'))
    gen=dict(status='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION',sample_id='one',
        GT_loaded_only_after_generation=True,GT_geometry_or_coords_used_in_generation=False,
        source_defined_normalization=True,per_output_alignment=False,actual_resolution=1024,
        metrics=dict(prediction_samples=20000,target_samples=20000,threshold=.02,per_output_alignment=False))
    return native,gen,path,refs,calls


def test_surface_reference_join_uses_actual_producer_GT_array_values(tmp_path):
    n,g,p,r,c=surface_inputs(tmp_path)
    result=local_surface_join(n,g,geometry_path=p,references=r)
    assert c==[1]and result['target_points_SHA256']=='points'and result['actual_existing_native_scores_unchanged']
    with np.load(p)as z:arrays={k:z[k]for k in z.files}
    arrays['target_world']=arrays['target_world']+1;np.savez(p,**arrays)
    with pytest.raises(ValueError,match='different GT'):local_surface_join(n,g,geometry_path=p,references=r)


@pytest.mark.parametrize('damage',['not-generated','GT-conditioned','aligned','wrong-resolution'])
def test_GT_is_not_accessed_before_successful_native_generation(tmp_path,damage):
    n,g,p,r,c=surface_inputs(tmp_path)
    if damage=='not-generated':g['status']='started'
    elif damage=='GT-conditioned':g['GT_geometry_or_coords_used_in_generation']=True
    elif damage=='aligned':g['per_output_alignment']=True
    elif damage=='wrong-resolution':g['actual_resolution']=512
    with pytest.raises(ValueError):local_surface_join(n,g,geometry_path=p,references=r)
    assert c==[]


def reports(alias='R_top1_only'):
    _,_,all_models=models();m=all_models[alias]
    g=dict(status='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION',sample_id='one',
        GT_loaded_only_after_generation=True,GT_geometry_or_coords_used_in_generation=False,
        per_output_alignment=False,metrics={'chamfer_L1':.01},SS_prediction_SHA256='SS')
    surface=dict(status='PASS_ACTUAL_LOCAL_NATIVE_GEOMETRY_SHARED_GT_SURFACE_JOIN',sample_id='one',native_metrics=g['metrics'])
    zero=alias.startswith('zero-')
    if zero:
        g.update(original_zero_shot_full_stack=True,deformation_finetuning_updates=0,
            PBR_checkpoint=m['pbr1024']['checkpoint']['path']);final=g
    else:
        final=dict(status='PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE',sample_id='one',PBR_arm=m['pbr1024']['mode'],
            checkpoint_kind='trained-native1900',engineering_only=False,PBR_checkpoint_SHA256=m['pbr1024']['checkpoint']['SHA256'],
            conditioner_checkpoint=None if m['pbr1024']['conditioner_checkpoint']is None else m['pbr1024']['conditioner_checkpoint']['path'],
            GT_geometry_or_coords_used_in_generation=False,GT_sparse_latents_or_projection_cache_used=False,
            source_weights_unchanged=True,conditioning_preserves_CPU_and_CUDA_RNG=True,raw_surface_metrics_unchanged=True,
            geometry_array_byte_identity={'vertices':True},metrics=g['metrics'],SS_prediction_SHA256='SS')
    shared=dict(status='PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE',sample_id='one',
        same_reference_pixels_for_every_model=True,per_view=[dict(camera_id=c,identical_camera_matrices=True)for c in [8,9,10]])
    if alias=='zero-source4':
        shared.pop('sample_id');shared.update(status='PASS_ACTUAL_SHARED_RENDERER_MATERIAL_REFERENCE',
            original_materials_and_UVs=True,GT_VAE_reconstruction_not_used=True,shared_renderer_lighting_camera_AO_seed=True,
            GT_not_used_for_model_generation=True,maximum_GT_world_vertex_nearest_difference=0.)
    terminals=[dict(PID=i+100,birth=1.,returncode=0,phase=p)for i,p in enumerate(
        ['geometry','render','shared']if zero else['geometry','PBR','render','shared'])]
    return m,g,final,shared,surface,terminals


@pytest.mark.parametrize('alias',ALIASES)
def test_case_release_requires_all_real_reports_and_completed_phases(alias):
    assert validate_case_reports(alias,*reports(alias))['status']=='PASS_ACTUAL_FROZEN_NATIVE_SIX_MODEL_TEST508_CASE'


@pytest.mark.parametrize('damage',['failed-phase','wrong-NN','changed-geometry','GT-conditioned','wrong-surface','different-camera'])
def test_partial_or_confounded_cases_cannot_be_published_or_retired(damage):
    m,g,f,s,gt,t=reports()
    if damage=='failed-phase':t[-1]['returncode']=1
    elif damage=='wrong-NN':f['PBR_checkpoint_SHA256']='wrong'
    elif damage=='changed-geometry':f['geometry_array_byte_identity']['vertices']=False
    elif damage=='GT-conditioned':f['GT_sparse_latents_or_projection_cache_used']=True
    elif damage=='wrong-surface':gt['native_metrics']={'chamfer_L1':.1}
    elif damage=='different-camera':s['per_view'][0]['identical_camera_matrices']=False
    with pytest.raises(ValueError):validate_case_reports('R_top1_only',m,g,f,s,gt,t)
