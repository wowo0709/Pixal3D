"""Physical composition/own geometry checks; no neural construction or forward."""
from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.pipeline_population_contract_v2 import (
    ARMS,ALIASES,SOURCE_COUNTS,family,schema_for,
)
from experiments.deformnext_retargeting.common_geometry_pbr_contract import MODES
from experiments.deformnext_retargeting.consistent_evidence_native_recipe import VARIANTS
from experiments.deformnext_retargeting.selected_pipeline_cascade_contract import CASE_STATUS
from experiments.deformnext_retargeting.selected_pipeline_execution import INDEX_STATUS,validate_index,validate_replay_base
from experiments.deformnext_retargeting.selected_pipeline_joint_summary_v2 import compositions
from test_selected_pipeline_joint_v2 import inputs


def binding(arm):
    b,*_=inputs('FULL')
    if arm=='V1':return b['V1']['ss64']
    if arm=='C_DECODER':return b['RETARGET']['ss64']
    value=b['SELECTED']['ss64']
    value.update(schema=schema_for(arm),module_arm=arm,variant=ALIASES.get(arm,arm),
        runtime_family=family(arm),source_tensors=SOURCE_COUNTS[arm])
    for key in ['native_config','checkpoint','conditioner_checkpoint']:
        value[key]['path']=value[key]['path'].replace('/FULL/','/'+arm+'/')
    if arm in VARIANTS:
        value.update(checkpoint_bundle_schema='consistent-raw-ray-evidence-SS-checkpoint-pair-v1',
            evidence_mode=VARIANTS[arm],shared_carrier_use_retarget_evidence=True,
            learned_projection_tensors=0,learned_attention_head_tensors=7 if arm=='E_LEARNED_RAW' else 0)
    return value


def index(selected=('E_LEARNED_RAW','raw-topk-plus-feature')):
    b,_,_,_,_,_,kw=inputs()
    pairs,alias=compositions(selected)
    ids=kw['sample_ids'];models={};bindings={};bases={};archives={}
    for name,(arm,mode) in pairs.items():
        ss=binding(arm);pbr=deepcopy(b['V1']['pbr1024'] if mode=='vanilla' else b['RETARGET']['pbr1024'])
        previous=pbr['mode'];pbr.update(mode=mode,source_tensors=0 if mode=='vanilla' else 10 if mode=='top1-only' else 19)
        for key in ['native_config','checkpoint','conditioner_checkpoint']:
            if pbr[key] is not None:pbr[key]['path']=pbr[key]['path'].replace('/'+previous+'/','/'+mode+'/')
        bindings[name]=dict(ss64=ss,pbr1024=pbr)
        models[name]=[dict(sample_id=sid,source_cluster_id=kw['clusters'][sid],case_name=name+'-'+sid,
            case_output='/new/'+name+'/'+sid,generation='/new/'+name+'/'+sid+'/generation',
            shared='/new/'+name+'/'+sid+'/shared') for sid in ids]
        bases[arm]=[dict(sample_id=sid,source_cluster_id=kw['clusters'][sid],case_output='/base/'+arm+'/'+sid,
            generation='/base/'+arm+'/'+sid+'/generation',geometry_plan='/plans/'+arm+'/'+sid+'.json') for sid in ids]
        archives[arm]=dict(path='/archives/'+arm+'/evaluation.json',receipt='/archives/'+arm+'/file4-closure.json')
    return dict(status=INDEX_STATUS,selected_pair=None if selected is None else list(selected),
        prospective_selected_alias=alias,models=models,compositions=bindings,base_cases=bases,
        SS_archives=archives,sample_ids=ids,original_shapes=kw['original_shapes'])


@pytest.mark.parametrize('arm',ARMS)
@pytest.mark.parametrize('mode',MODES)
def test_every_actual_physical_pair_uses_two_or_three_deduplicated_pipelines(arm,mode):
    value=index((arm,mode));pairs,alias=validate_index(value)
    assert len(pairs)==(2 if (arm,mode) in [('V1','vanilla'),('C_DECODER','top1-only')] else 3)
    assert alias in pairs and pairs[alias]==(arm,mode)
    assert set(value['base_cases'])=={pair[0] for pair in pairs.values()}


def test_baseline_only_index_does_not_select_a_model():
    pairs,alias=validate_index(index(None))
    assert alias is None and set(pairs)=={'V1','RETARGET'}


@pytest.mark.parametrize('damage',['alias-Full','wrong-pbr','source-count','seed202','wrong-order','tail',
    'cluster','other-pixels','missing-archive','extra-geometry','shape-step','duplicate-pair'])
def test_wrong_model_or_missing_composition_cannot_enter_official_metrics(damage):
    value=index();ss=value['compositions']['SELECTED']['ss64']
    if damage=='alias-Full':ss['module_arm']='FULL'
    elif damage=='wrong-pbr':value['compositions']['SELECTED']['pbr1024']['mode']='top1-only'
    elif damage=='source-count':ss['source_tensors']=19
    elif damage=='seed202':ss['training_seed']=202
    elif damage=='wrong-order':value['models']['SELECTED'].reverse()
    elif damage=='tail':value['models']['SELECTED'].pop()
    elif damage=='cluster':value['models']['SELECTED'][0]['source_cluster_id']='other'
    elif damage=='other-pixels':value['models']['SELECTED'][0]['shared']='/base/V1/other/shared'
    elif damage=='missing-archive':value['SS_archives'].pop('E_LEARNED_RAW')
    elif damage=='extra-geometry':value['base_cases']['R_feature']=[]
    elif damage=='shape-step':value['original_shapes']['shape512']['step']=16850
    else:value['selected_pair']=['C_DECODER','top1-only']
    with pytest.raises(ValueError):validate_index(value)


def replay_inputs(arm='E_LEARNED_RAW'):
    b,_,_,base,_,_,kw=inputs(arm)
    own=kw['archives'][arm];archive=own['record'];ref=deepcopy(archive)
    rows=[];artifacts=[]
    for i,sid in enumerate(kw['sample_ids']):
        artifact=deepcopy(archive['native_SS_artifacts']['artifacts'][i]);artifact.update(seed=i,
            target_voxels_SHA256='a'*64,occupancy_counts=dict(tp=1,fp=2,fn=3))
        artifacts.append(artifact)
        rows.append(dict(sample_id=sid,source_cluster_id=kw['clusters'][sid],seed=i,
            target_voxels_sha256='a'*64,tp=1,fp=2,fn=3))
    archive['per_asset']=deepcopy(rows);ref['per_asset']=rows
    archive['native_SS_artifacts'].update(archived_asset_count=32,artifacts=artifacts,
        predicted_SS_is_available_for_full_cascade=True,original_prefix32_all_records_and_occupancy_counts_exact=True)
    row=base[arm]['0'];case=row['case'];case.update(SS_evaluation=own['spec']['path'])
    if case['status']==CASE_STATUS:case['actual_SS_model_binding']=deepcopy(b['SELECTED']['ss64'])
    extra=dict(arm=arm,binding=b['SELECTED']['ss64'],archive_spec=own['spec'],
        prediction_SHA256=row['prediction_spec']['SHA256'],guide_SHA256=row['generation']['actual_native_guide_archive_SHA256'])
    return case,row['generation'],archive,ref,extra


@pytest.mark.parametrize('arm',['FULL',*VARIANTS])
def test_original_and_new_selected_case_read_own_ss_geometry_without_reconstruction(arm):
    c,g,a,r,kw=replay_inputs(arm)
    value=validate_replay_base(c,g,a,r,**kw)
    assert value['actual_physical_SS_arm']==arm and value['new_SS_or_Shape_forwards_for_replay']==0


@pytest.mark.parametrize('damage',['wrong-case','wrong-prediction','wrong-archive','archive-bytes','other-binding',
    'native-guide','GT-geometry','alignment','not-drained','other-noise','different-GT','occupancy-count'])
def test_another_models_geometry_cannot_be_relabelled_as_the_selected_pipeline(damage):
    c,g,a,r,kw=replay_inputs()
    if damage=='wrong-case':c['SS_arm']='FULL'
    elif damage=='wrong-prediction':g['SS_prediction_SHA256']='b'*64
    elif damage=='wrong-archive':c['SS_evaluation']='/other/evaluation.json'
    elif damage=='archive-bytes':c['verified_inputs'][kw['archive_spec']['path']]='b'*64
    elif damage=='other-binding':c['actual_SS_model_binding']['module_arm']='FULL'
    elif damage=='native-guide':g['actual_native_guide_archive_SHA256']='b'*64
    elif damage=='GT-geometry':g['GT_geometry_or_coords_used_in_generation']=True
    elif damage=='alignment':g['per_output_alignment']=True
    elif damage=='not-drained':c['actual_owned_terminals'][0]['actual_session_drained']=False
    elif damage=='other-noise':r['per_asset'][0]['seed']+=1
    elif damage=='different-GT':r['per_asset'][0]['target_voxels_sha256']='b'*64
    else:a['native_SS_artifacts']['artifacts'][0]['occupancy_counts']['fp']=8
    with pytest.raises(ValueError):validate_replay_base(c,g,a,r,**kw)
