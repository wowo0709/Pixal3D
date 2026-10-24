"""Synthetic physical bindings and508 records only; no held-out NN execution."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.deformnext_retargeting.common_geometry_pbr_contract import MODES as PBR_MODES
from experiments.deformnext_retargeting.original_zero_population import MODES, SCHEMA as ZERO_SCHEMA
from experiments.deformnext_retargeting.pipeline_population_contract_v2 import ARMS
from experiments.deformnext_retargeting.selected_final_population import (
    admit_all_ss, admit_ss, compose_models, make_original_controls, validate_parent,
)
from experiments.deformnext_retargeting.selected_pipeline_freeze_v2 import make_freeze
from experiments.deformnext_retargeting.selected_pipeline_joint_summary_v2 import compositions
from test_selected_pipeline_freeze_v2 import freeze_inputs
from test_selected_population_plan_factory import inputs as physical_inputs


def parents(tmp_path, arm='E_LEARNED_RAW', mode='vanilla'):
    # A metadata fixture, not an actual published DEV freeze.
    parent=make_freeze(**freeze_inputs())
    ss={m['ss64']['module_arm']:m['ss64'] for m in parent['models'].values()}
    if arm not in ss:ss[arm]=physical_inputs(tmp_path,arm)[0]['model_binding']
    pbr={m['pbr1024']['mode']:m['pbr1024'] for m in parent['models'].values()}
    for value in PBR_MODES:
        if value in pbr:continue
        binding=deepcopy(pbr['vanilla']);root='/native/pbr/'+value
        binding.update(mode=value,source_tensors=10 if value=='top1-only' else 19,
            native_config=dict(path=root+'/config.json',SHA256='a'*64),
            checkpoint=dict(path=root+'/ckpts/denoiser_step0001900.pt',SHA256='b'*64),
            conditioner_checkpoint=dict(path=root+'/ckpts/deform_conditioner_step0001900.pt',SHA256='c'*64))
        pbr[value]=binding
    pairs,alias=compositions((arm,mode));original=parent['original_stages']
    parent.update(selected_pair=[arm,mode],selected_alias=alias,models={name:dict(ss64=ss[a],pbr1024=pbr[m],
        shape512=original['shape512'],shape1024=original['shape1024']) for name,(a,m) in pairs.items()})
    parent['selection'].update(selected_SS_arm=arm,selected_PBR_mode=mode)
    originals={name:dict(schema=ZERO_SCHEMA,alias=name,conditioning_mode=input_mode,
        original_multiview_step=4210,deformation_finetuning_updates=0,
        native_config=dict(path='/zero-loader/config.json',SHA256='a'*64),
        checkpoint=original['ss64']['checkpoint'],conditioner_checkpoint=None) for name,input_mode in MODES.items()}
    controls=make_original_controls(parent,originals)
    return parent,controls


def evaluation(model, parent, alias):
    binding=model['ss64'];zero=model['kind']=='original-zero';ids=parent['population']['sample_ids'];pop=parent['population']
    rows=[dict(sample_id=sid,source_cluster_id=pop['source_clusters_by_sample'][sid],seed=pop['generation_seeds'][sid],
        target_voxels_sha256='b'*64,tp=1,fp=2,fn=3) for sid in ids]
    artifacts=[dict(sample_id=r['sample_id'],seed=r['seed'],artifact=f'ss-artifacts/{i}.npz',artifact_SHA256='a'*64,
        prediction_voxels_SHA256='a'*64,target_voxels_SHA256=r['target_voxels_sha256'],
        occupancy_counts={k:r[k] for k in ['tp','fp','fn']}) for i,r in enumerate(rows)]
    result=dict(status='PASS',stage='ss64',step=0 if zero else 480,weights='raw',population_scope='final-test508',
        diagnostic_only=False,manifest_sha256=pop['manifest']['SHA256'],frozen_native_model_binding=deepcopy(binding),
        arm_config=binding['native_config']['path'],checkpoint=binding['checkpoint']['path'],
        conditioner_checkpoint=None if binding['conditioner_checkpoint'] is None else binding['conditioner_checkpoint']['path'],
        input_states=['M0']*4 if alias=='zero-source4' else ['M1','M0','M0','M0'],input_camera_ids=[8,9,10,11],
        conditioning_mode='source-only' if alias=='zero-source4' else 'edited-rgb',target_modality='rgb',
        optimizer_updates_during_evaluation=0,generation_batch_size=8,sampling_steps=12,guidance=3.,
        quick_panel_ids=ids.copy(),per_asset=rows,native_SS_artifacts=dict(
            status='PASS_ACTUAL_ORIGINAL_ZERO_SS_BOUND_POPULATION_ARCHIVE' if zero else 'PASS_ACTUAL_NATIVE_BOUND_SS_POPULATION_ARCHIVE',
            archived_asset_count=508,artifacts=artifacts,predicted_SS_is_available_for_full_cascade=True,GT_SS_is_for_scoring_only=True))
    if zero:
        result.update(original_multiview_SS_step=4210,deformation_finetuning_updates=0,
            actual_original_full_base_and_LoRA_validation=dict(original_full_base_and_LoRA_exact_before_evaluation=True,
                actual_original_LoRA_tensors=360),deterministic_evaluation=dict(actual_deterministic_algorithms=True))
    else:
        result.update(native_variant=binding['variant'],actual_source_tensors=binding['source_tensors'],actual_LoRA_tensors=360,
            deterministic_algorithms=True,verified_inputs={binding[k]['path']:binding[k]['SHA256'] for k in
                ['native_config','checkpoint','conditioner_checkpoint'] if binding[k] is not None})
        if model['SS_arm'].startswith('E_'):
            result.update(conditioning_schema='consistent-raw-ray-evidence-native-v1',
                followup_study_schema='consistent-raw-ray-evidence-followup-v1',shared_carrier_use_retarget_evidence=True,
                learned_projection_tensors=0,learned_attention_head_tensors=binding['learned_attention_head_tensors'],training_seed=101)
    return result


@pytest.mark.parametrize('arm',ARMS)
@pytest.mark.parametrize('mode',PBR_MODES)
def test_every_selected_physical_pair_keeps_original_stages_and_both_zero_controls(tmp_path,arm,mode):
    parent,controls=parents(tmp_path,arm,mode);before=deepcopy((parent,controls));models=compose_models(parent,controls)
    assert (parent,controls)==before and len(models)==2+len(parent['models'])
    assert models[parent['selected_alias']]['ss64']['module_arm']==arm
    assert models[parent['selected_alias']]['PBR_mode']==mode
    assert all(m['geometry_stages']==controls['downstream_original_stages'] for m in models.values())
    assert models['zero-source4']['geometry_group']!=models['zero-edited4']['geometry_group']
    records={name:evaluation(m,parent,name) for name,m in models.items()}
    admitted=admit_all_ss(records,parent=parent,controls=controls)
    assert all(len(rows)==508 for rows in admitted.values())


@pytest.mark.parametrize('damage',['legacy-freeze','prior-test-read','selection','missing-control','wrong-stage','missing-evidence',
    'different-zero-config','zero-finetuned','different-SS-arm','same-arm-different-weights','incomplete508','float-noise'])
def test_wrong_freeze_controls_or_physical_compositions_never_admit_test(tmp_path,damage):
    p,c=parents(tmp_path)
    if damage=='legacy-freeze':p['schema']='legacy-four-model-freeze'
    elif damage=='prior-test-read':p['heldout_quality_read_before_freeze']=True
    elif damage=='selection':p['selection']['selected_SS_arm']='FULL'
    elif damage=='missing-control':c['models'].pop('zero-edited4')
    elif damage=='wrong-stage':c['downstream_original_stages']['shape1024']['checkpoint']['SHA256']='f'*64
    elif damage=='missing-evidence':p['development_evidence_roles'].pop('joint')
    elif damage=='different-zero-config':c['models']['zero-edited4']['native_config']['path']='/other/config.json'
    elif damage=='zero-finetuned':c['models']['zero-source4']['deformation_finetuning_updates']=480
    elif damage=='different-SS-arm':p['models']['SELECTED']['ss64']['module_arm']='FULL'
    elif damage=='same-arm-different-weights':
        p,c=parents(tmp_path,'V1','raw-topk-plus-feature');p['models']['SELECTED']['ss64']=deepcopy(p['models']['SELECTED']['ss64'])
        p['models']['SELECTED']['ss64']['checkpoint']['SHA256']='f'*64
    elif damage=='incomplete508':p['population']['sample_ids'].pop()
    else:p['population']['generation_seeds']['test0']=0.
    with pytest.raises(ValueError):compose_models(p,c)


@pytest.mark.parametrize('alias',['zero-source4','zero-edited4','V1','RETARGET','SELECTED'])
@pytest.mark.parametrize('damage',['prefix32','wrong-source','GT','noise','missing-archive','wrong-count','wrong-NN','wrong-mode','diagnostic','optimizer'])
def test_real508_archive_protocol_is_not_replaced_by_a_DEV_prefix_or_different_model(tmp_path,alias,damage):
    parent,controls=parents(tmp_path);models=compose_models(parent,controls);r=evaluation(models[alias],parent,alias)
    if damage=='prefix32':r['native_SS_artifacts']['archived_asset_count']=32
    elif damage=='wrong-source':r['per_asset'][0]['source_cluster_id']='other'
    elif damage=='GT':r['native_SS_artifacts']['artifacts'][0]['target_voxels_SHA256']='c'*64
    elif damage=='noise':r['per_asset'][-1]['seed']+=1
    elif damage=='missing-archive':r['native_SS_artifacts']['artifacts'].pop()
    elif damage=='wrong-count':r['native_SS_artifacts']['artifacts'][0]['occupancy_counts']['tp']+=1
    elif damage=='wrong-NN':r['checkpoint']='/different.pt'
    elif damage=='wrong-mode':r['input_states']=['M0']*3
    elif damage=='diagnostic':r['diagnostic_only']=True
    else:r['optimizer_updates_during_evaluation']=1
    with pytest.raises(ValueError):admit_ss(alias,models[alias],r,parent=parent,controls=controls,
        manifest_SHA256=parent['population']['manifest']['SHA256'])


def test_cross_model_GT_must_match_even_if_each_archive_is_internally_consistent(tmp_path):
    p,c=parents(tmp_path);models=compose_models(p,c);records={a:evaluation(m,p,a) for a,m in models.items()}
    r=records['SELECTED'];r['per_asset'][0]['target_voxels_sha256']='c'*64
    r['native_SS_artifacts']['artifacts'][0]['target_voxels_SHA256']='c'*64
    with pytest.raises(ValueError,match='actual GT'):admit_all_ss(records,parent=p,controls=c)


def test_original_zero_loader_retains_actual_cached_checkpoint_copy_and_config(tmp_path):
    p,c=parents(tmp_path);models=deepcopy(c['models'])
    for model in models.values():model['checkpoint']['path']='/actually-used-cache/denoiser_step0004210.pt'
    controls=make_original_controls(p,models);composed=compose_models(p,controls)
    for alias in MODES:
        assert composed[alias]['ss64']['checkpoint']==models[alias]['checkpoint']
        assert controls['original_SS_copy_lineage'][alias]['original']==p['original_stages']['ss64']['checkpoint']
        assert controls['original_SS_copy_lineage'][alias]['actual_used']==models[alias]['checkpoint']
        assert composed[alias]['ss64']['native_config']==models[alias]['native_config']


@pytest.mark.parametrize('damage',['wrong-copy-bytes','two-different-copies','original-lineage','copy-lineage','quality-claim'])
def test_copied_original_checkpoint_requires_actual_same_bytes_and_preserved_lineage(tmp_path,damage):
    p,c=parents(tmp_path)
    if damage=='wrong-copy-bytes':c['models']['zero-source4']['checkpoint']['SHA256']='f'*64
    elif damage=='two-different-copies':c['models']['zero-source4']['checkpoint']['path']='/other/denoiser_step0004210.pt'
    elif damage=='original-lineage':c['original_SS']['checkpoint']['path']='/other/denoiser_step0004210.pt'
    elif damage=='copy-lineage':c['original_SS_copy_lineage']['zero-source4']['actual_used']['SHA256']='f'*64
    else:c['no_candidate_or_control_quality_used']=False
    with pytest.raises(ValueError):compose_models(p,c)
