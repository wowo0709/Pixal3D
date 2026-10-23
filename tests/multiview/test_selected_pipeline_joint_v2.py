"""Selected joint pipeline must preserve actual families and measured pixels."""
from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.consistent_evidence_native_recipe import VARIANTS
from experiments.deformnext_retargeting.pipeline_population_contract_v2 import SCHEMA
from experiments.deformnext_retargeting.selected_pipeline_cascade_contract import CASE_STATUS
from experiments.deformnext_retargeting.selected_pipeline_joint_summary_v2 import compositions,summarize
from experiments.deformnext_retargeting.selected_pipeline_model_contract import validate_ss_evaluation
from test_joint_pipeline_summary import inputs as original_inputs


def inputs(arm='E_LEARNED_RAW'):
    b,c,g,base,r,p,kw=original_inputs(('FULL','vanilla'))
    for alias,binding in b.items():
        name=binding['ss64']['module_arm'];ev=kw['archives'][name]['record']
        ev.update(stage='ss64',actual_source_tensors=binding['ss64']['source_tensors'],actual_LoRA_tensors=360,
            target_modality='rgb',input_states=['M1','M0','M0','M0'],input_camera_ids=[8,9,10,11],
            generation_batch_size=8,sampling_steps=12,guidance=3.,deterministic_algorithms=True)
    if arm=='FULL':return b,c,g,base,r,p,kw
    ss=b['SELECTED']['ss64']
    ss.update(schema=SCHEMA,module_arm=arm,variant=arm,runtime_family='consistent_evidence',
        source_tensors=17 if arm=='E_LEARNED_RAW' else 10,evidence_mode=VARIANTS[arm],
        checkpoint_bundle_schema='consistent-raw-ray-evidence-SS-checkpoint-pair-v1',
        shared_carrier_use_retarget_evidence=True,learned_projection_tensors=0,
        learned_attention_head_tensors=7 if arm=='E_LEARNED_RAW' else 0)
    for key in ['native_config','checkpoint','conditioner_checkpoint']:
        ss[key]['path']=ss[key]['path'].replace('/FULL/','/'+arm+'/')
    archive=kw['archives'].pop('FULL');kw['archives'][arm]=archive
    ev=archive['record'];ev.update(native_variant=arm,actual_source_tensors=ss['source_tensors'],
        conditioning_schema='consistent-raw-ray-evidence-native-v1',
        followup_study_schema='consistent-raw-ray-evidence-followup-v1',
        shared_carrier_use_retarget_evidence=True,learned_projection_tensors=0,
        learned_attention_head_tensors=ss['learned_attention_head_tensors'],training_seed=101)
    for key,field in [('native_config','arm_config'),('checkpoint','checkpoint'),
                      ('conditioner_checkpoint','conditioner_checkpoint')]:ev[field]=ss[key]['path']
    ev['verified_inputs']={ss[key]['path']:ss[key]['SHA256'] for key in ['native_config','checkpoint','conditioner_checkpoint']}
    base[arm]=base.pop('FULL')
    for sid,value in base[arm].items():value['case'].update(SS_arm=arm,status=CASE_STATUS)
    kw['selected']=(arm,'vanilla')
    return b,c,g,base,r,p,kw


@pytest.mark.parametrize('arm',['FULL',*VARIANTS])
def test_actual_joint_pipeline_keeps_physical_selected_model_and_same_original_math(arm):
    b,c,g,base,r,p,kw=inputs(arm)
    out=summarize(b,c,g,base,r,p,**kw)
    assert out['actual_official_views']==288 and out['physical_selected_family_preserved']
    assert out['measured_compositions']['SELECTED']['ss64']==b['SELECTED']['ss64']
    assert not out['model_adopted'] and out['FID_KID_and_TEST508_still_required']


@pytest.mark.parametrize('damage',['alias-Full','old-schema','source-count','V','mode','source-bytes',
    'old-case','old-evaluation','wrong-epoch','wrong-camera','raw-count','guide','GTgeometry','PBRseed202'])
def test_new_structure_cannot_be_inferred_from_stage_or_relabelled_old_model(damage):
    b,c,g,base,r,p,kw=inputs();ss=b['SELECTED']['ss64'];ev=kw['archives']['E_LEARNED_RAW']['record']
    if damage=='alias-Full':ss['module_arm']='FULL'
    elif damage=='old-schema':ss['schema']='unified-module-SS480-frozen-binding-v1'
    elif damage=='source-count':ss['source_tensors']=19
    elif damage=='V':ss['learned_projection_tensors']=2
    elif damage=='mode':ss['evidence_mode']='fixed-prior'
    elif damage=='source-bytes':ev['verified_inputs'][ss['conditioner_checkpoint']['path']]='b'*64
    elif damage=='old-case':base['E_LEARNED_RAW']['0']['case']['status']='PASS_ACTUAL_CORE_SS_COMMON_TRAINED_V1_PBR_NATIVE_CASCADE_CASE'
    elif damage=='old-evaluation':ev['conditioning_schema']='unified-module-projected-native-v1'
    elif damage=='wrong-epoch':ev['step']=384
    elif damage=='wrong-camera':ev['input_camera_ids']=[9,10,11]
    elif damage=='raw-count':ev['actual_source_tensors']=19
    elif damage=='guide':g['SELECTED']['0']['actual_native_guide_archive_SHA256']='b'*64
    elif damage=='GTgeometry':g['SELECTED']['0']['GT_geometry_or_coords_used_in_generation']=True
    else:b['SELECTED']['pbr1024']['training_seed']=202
    with pytest.raises(ValueError):summarize(b,c,g,base,r,p,**kw)


def test_stage_only_model_changes_do_not_create_a_pipeline_candidate():
    pairs,alias=compositions(('C_DECODER','top1-only'))
    assert set(pairs)=={'V1','RETARGET'} and alias=='RETARGET'
    for value in [('R_feature','vanilla'),('E_LEARNED_RAW','untrained-mode')]:
        with pytest.raises(ValueError):compositions(value)


@pytest.mark.parametrize('arm',VARIANTS)
def test_selected_cascade_consumes_own_archived_prediction_without_new_SS_forward(arm):
    from experiments.deformnext_retargeting.selected_pipeline_cascade_contract import validate_archive
    b,_,_,_,_,_,kw=inputs(arm)
    ev=deepcopy(kw['archives'][arm]['record']);ref=deepcopy(ev)
    rows=[];artifacts=[]
    for i in range(32):
        rows.append(dict(sample_id=str(i),source_cluster_id=str(i),seed=i,target_voxels_sha256='a'*64,tp=1,fp=2,fn=3))
        artifacts.append(dict(sample_id=str(i),seed=i,target_voxels_SHA256='a'*64,occupancy_counts=dict(tp=1,fp=2,fn=3)))
    ev['per_asset']=deepcopy(rows);ref['per_asset']=rows
    ev['native_SS_artifacts'].update(archived_asset_count=32,artifacts=artifacts,
        predicted_SS_is_available_for_full_cascade=True,original_prefix32_all_records_and_occupancy_counts_exact=True)
    assert validate_archive(ev,arm,ref,binding=b['SELECTED']['ss64'])==list(map(str,range(32)))
    ev['native_SS_artifacts']['artifacts'][0]['occupancy_counts']['fp']=9
    with pytest.raises(ValueError):validate_archive(ev,arm,ref,binding=b['SELECTED']['ss64'])
