import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.deformnext_retargeting.semantic_topk_campaign_wandb import (
    conditional_scope, validate_noise_identity, publication_jobs, payload, COMPONENTS, validate_saved_qa, previous_c1_proof_files,
)


def records():
    es=[]
    for noise in range(3):
        rows=[{'sample_id':str(i),'source_cluster_id':str(i%93),'seed':1000*noise+i,
               'tp':2,'fp':1,'fn':2,'target_voxels_sha256':str(i).zfill(64),
               'target_voxel_shape':[64,64,64]} for i in range(128)]
        es.append({'per_asset':rows})
    return {'reference':es,'candidate':copy.deepcopy(es)}


def selected_reports():
    return [
        {'status':'PASS_FIXED_FINAL_SEED202_CONFIRMATION','selected_candidate':'task-full',
         'positive_CI_and_all_noise_direction_at_both_seeds':False},
        {'status':'PASS_ADDITIONAL121_FIXED_SELECTED_MODEL_VALIDATION','selected_candidate':'task-full',
         'positive_CI_and_all_noises_on_both_additional_training_seeds':False},
        {'status':'PASS_SELECTED_MODEL_SAME_POOL_VALUE_ADDRESS_DIAGNOSTIC','selected_candidate':'task-full'},
    ]


def test_actual_complete_same_panel_remains_same_when_model_predictions_differ():
    r=records();r['candidate'][1]['per_asset'][4].update(tp=3,fn=1,fp=0)
    assert validate_noise_identity(r,128,93)==[str(i) for i in range(128)]


@pytest.mark.parametrize('field,value',[
    ('source_cluster_id','other'),('seed',90000),
    ('target_voxels_sha256','b'*64),('target_voxel_shape',[32,32,32]),('fn',5),
])
def test_matching_metric_cannot_hide_different_population_noise_or_target(field,value):
    r=records();r['candidate'][2]['per_asset'][9][field]=value
    with pytest.raises(ValueError,match='identity'):validate_noise_identity(r,128,93)


def test_same_three_sampling_noise_files_cannot_be_repeated_to_claim_three_noises():
    r=records()
    for es in r.values():es[2]=copy.deepcopy(es[1])
    with pytest.raises(ValueError,match='distinct sampling'):validate_noise_identity(r,128,93)


def test_same_set_reordered_cannot_be_presented_as_same_case_panels():
    r=records();r['candidate'][1]['per_asset'].reverse()
    with pytest.raises(ValueError,match='order'):validate_noise_identity(r,128,93)


def test_source_population_is_checked_separately_from_asset_count():
    with pytest.raises(ValueError,match='cluster'):validate_noise_identity(records(),128,128)


@pytest.mark.parametrize('missing',[0,1,2])
def test_positive_selection_requires_all_selected_model_work_before_publication(missing):
    reports=selected_reports();reports[missing]=None
    with pytest.raises(ValueError,match='incomplete'):conditional_scope('task-full',*reports)


def test_pass_status_is_not_interpreted_as_positive_scientific_replication_or_adoption():
    scope=conditional_scope('task-full',*selected_reports())
    assert scope['seed_confirmation_completed'] and scope['additional121_completed']
    assert not scope['positive_in_both_training_seeds']
    assert not scope['positive_on_additional121_both_seeds']
    assert not scope['same_retargeter_Top1_replication_across_training_seeds']
    assert not scope['model_adopted'] and not scope['full_campaign_complete']


def test_a_different_selected_candidate_cannot_supply_confirmation():
    with pytest.raises(ValueError,match='differs'):conditional_scope('static-c1',*selected_reports())


def test_no_selected_candidate_is_preserved_as_negative_evidence_with_active_scope():
    scope=conditional_scope(None)
    assert scope['selected_candidate'] is None and not scope['model_adopted']
    assert not scope['seed_confirmation_completed'] and not scope['full_campaign_complete']
    with pytest.raises(ValueError,match='unselected'):conditional_scope(None,*selected_reports())


@pytest.mark.parametrize('candidate',[None,'task-full','static-c1'])
def test_reporting_waits_for_actual_models_and_auditors_without_allocating_a_gpu(tmp_path,candidate):
    jobs=publication_jobs(tmp_path,candidate,'select-model')
    assert [j['name'] for j in jobs]==['campaign-report-render','campaign-report-publish','campaign-report-readback']
    assert all(j['resource']=='cpu' and 'gpu' not in j and 'gpus' not in j for j in jobs)
    deps=jobs[0]['depends_on'];assert 'select-model' in deps
    assert 'functional-QA-s101-static-top1-c1' in deps
    assert 'functional-QA-s202-top1-reference' in deps
    assert 'summarize-firstwave-fusion-dose' in deps
    assert 'fresh-omega7-feature-cache-parity' in deps
    assert 'attention-statistics-s101-static-c1' in deps and 'attention-statistics-s101-task-full' in deps
    assert ('summarize-additional121-semantic-topk' in deps)==(candidate is not None)
    assert len(jobs[0]['requires'])==(9 if candidate is None else 10)*6+3
    assert jobs[1]['depends_on']==['campaign-report-render']
    assert jobs[2]['depends_on']==['campaign-report-publish']
    assert all(Path(output).is_relative_to(tmp_path) for j in jobs for output in j['outputs'])


def test_publication_preserves_panel_prefixes_and_all_named_module_comparisons(tmp_path,monkeypatch):
    contrast={'gain':.001,'paired_cluster_bootstrap_ci95':[-.001,.003]}
    data={'sets':{
        'development128':{'models':{'method':{'micro_voxel_iou':.4,'micro_voxel_f1':.57}},
                          'contrasts':{k:copy.deepcopy(contrast) for k in COMPONENTS.values()}},
        'additional121':{'models':{'method':{'micro_voxel_iou':.3,'micro_voxel_f1':.46}},'contrasts':{}},
    },'scope':conditional_scope('task-full',*selected_reports()),'files':[],
        'backbone_metrics':{'backbone_QA/case0/view0/FRAME/relative_RMS':.0002},
        'attention_metrics':{'attention_QA/task-full/mean_effective_candidates':3.5},
        'diagnostic_tables':{'diagnostics/matching_prior_drift/development128/component_swaps':{
            'columns':['inference_swap','argmax_disagreement_fraction'], 'data':[['saved-only',.45]]}}}
    folder=tmp_path/'campaign-wandb-v1/presentation';folder.mkdir(parents=True)
    (folder/'summary.json').write_text(json.dumps({'status':'PASS_COMPLETED_CAMPAIGN_PRESENTATION',
        'scope':data['scope'],'input_SHA256':{}}))
    monkeypatch.setattr('experiments.deformnext_retargeting.semantic_topk_campaign_wandb.collect',lambda _:data)
    spec=payload(tmp_path)
    assert spec['metrics']['development128/method/micro_voxel_iou']==.4
    assert spec['metrics']['additional121/method/micro_voxel_iou']==.3
    assert spec['metrics']['backbone_QA/case0/view0/FRAME/relative_RMS']==.0002
    assert spec['metrics']['attention_QA/task-full/mean_effective_candidates']==3.5
    assert spec['tables']['diagnostics/matching_prior_drift/development128/component_swaps']==data['diagnostic_tables']['diagnostics/matching_prior_drift/development128/component_swaps']
    rows=spec['tables']['development128/component_effects']['data']
    assert [r[0] for r in rows]==list(COMPONENTS)
    assert all(r[2]<0 for r in rows)  # Negative bounds are retained, never reformatted as improvement.
    assert not spec['scope']['full_campaign_complete'] and not spec['scope']['model_adopted']


@pytest.mark.parametrize('aggregation',['soft','top1'])
def test_frozen_c1_uses_its_actual_completed_qa_schema(aggregation):
    qa={'status':'PASS_CURRENT_C1_CACHE_AMP_CFG_CONTRACT','optimizer_updates':0,
        'actual_four_RGB_cameras_exact':True,'p1_all_branches_exact_zero':True,'sample_ids':['one','two']}
    validate_saved_qa(qa,{'arm':'c1','candidate_aggregation':aggregation})


def test_legacy_frozen_matcher_qa_cannot_certify_a_task_trainable_matching_head():
    qa={'status':'PASS_CURRENT_C1_CACHE_AMP_CFG_CONTRACT','optimizer_updates':0,
        'actual_four_RGB_cameras_exact':True,'p1_all_branches_exact_zero':True,'sample_ids':['one','two']}
    spec={'arm':'c1','task_attention':{'decoder':True,'score_prior':True}}
    with pytest.raises(ValueError,match='schema'):validate_saved_qa(qa,spec)
    qa['status']='PASS_TRAINED_SS_CACHE_AMP_CFG_CONTRACT';validate_saved_qa(qa,spec)


@pytest.fixture
def existing_c1_metadata(tmp_path):
    root=tmp_path/'search';base=tmp_path/'ss-seed-confirmation-v1'
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path
    def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    recipe=save(base/'configs/c1.json',{'training_seed':101,'trainer':{'args':{'deform_conditioner':{'arm':'c1'}}}})
    refs=[{'path':str(base/'runs_ss64/c1/ckpts'/name),'sha256':str(i)*64,'bytes':12}
          for i,name in enumerate(['denoiser_step0000480.pt','deform_conditioner_step0000480.pt'],1)]
    qa=save(tmp_path/'conditioning-contract-v2/summary.json',{
        'status':'PASS_CURRENT_C1_CACHE_AMP_CFG_CONTRACT','optimizer_updates':0,
        'actual_four_RGB_cameras_exact':True,'p1_all_branches_exact_zero':True,'sample_ids':['one','two'],
        'source_recipe':{'path':str(recipe),'sha256':digest(recipe)},'saved_frozen_head_exact_to_selected_checkpoint':True,
        'trained_denoiser':refs[0],'trained_conditioner':refs[1]})
    weights=save(base/'all-three-final-weight-audit-v1.json',{
        'status':'PASS_ALL_THREE_FINAL_MODEL_TENSORS_AND_FROZEN_BASE','B3_C1_all24_frozen_Q2C_tensors_exact':True,
        'actual_final_weights':[{'arm':'c1','actual_final_denoiser':refs[0],'actual_conditioner':refs[1],
            'all_denoiser_tensors_finite':True,'all_conditioner_tensors_finite':True,
            'all_frozen_base_tensors_dtype_shape_and_values_exact_to_SS4210':True,
            'frozen_base_tensor_keys':701,'lora_tensor_keys_changed':360,'frozen_Q2C_head_keys':24,
            'actual_retargeter_present':True,'RAW_deformation_query_channels':0}]})
    epoch=save(base/'epoch-audits-v2/c1/epoch05/audit.json',{'status':'PASS_EPOCH_CHECKPOINT_AUDIT',
        'training_seed':101,'step':480,'all_optimizer_steps':480,'all_optimizer_moments_finite':True,
        'global_batch':96,'world_size':2,'microbatch_per_rank':48,'accumulation':1})
    evaluations=[save(base/'evaluations/c1'/f'noise-{i}/evaluation.json',{'fixture':i}) for i in range(3)]
    for closure,paths in [
        (tmp_path/'provenance/current-C1-QA-v2-completed-file4.json',[qa]),
        (tmp_path/'provenance/all-three-final-weight-audit-v1-file4.json',[weights]),
        (tmp_path/'provenance/matched-all-three-final-quality-and-remote-v1-file4.json',[recipe,*evaluations]),
        (base/'epoch-audits-v2/c1/epoch05/file4-closure.json',[epoch])]:
        rows=[{'relative_path':str(p.relative_to(tmp_path)),'sha256':digest(p),'bytes':p.stat().st_size} for p in paths]
        if paths==[epoch]:rows += [{'relative_path':str(Path(r['path']).relative_to(tmp_path)),
                                  'sha256':r['sha256'],'bytes':r['bytes']} for r in refs]
        save(closure,{'status':'PASS','local_equals_actual_file4_SHA256':True,'files':rows})
    return root,{'recipe':recipe,'qa':qa,'weights':weights,'epoch':epoch}


def test_original_c1_proofs_retain_original_recipe_and_checkpoint_paths(existing_c1_metadata):
    root,paths=existing_c1_metadata;files=[];previous_c1_proof_files(root,files)
    assert paths['recipe'] in files and paths['qa'] in files and len(set(files))==11
    assert not (root/'configs/previous-c1.json').exists()


@pytest.mark.parametrize('changed',['checkpoint','recipe','optimizer','file4'])
def test_original_c1_cannot_be_certified_by_an_unrelated_or_changed_saved_proof(existing_c1_metadata,changed):
    root,paths=existing_c1_metadata
    key='epoch' if changed=='optimizer' else 'qa'
    p=paths[key];data=json.loads(p.read_text())
    if changed=='checkpoint':data['trained_denoiser']['sha256']='other-model'
    elif changed=='recipe':data['source_recipe']['path']='unrelated-recipe.json'
    elif changed=='optimizer':data['all_optimizer_moments_finite']=False
    else:data['sample_ids'].reverse()  # Still valid QA shape,but no longer the actual closed metadata bytes.
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError):previous_c1_proof_files(root,[])
