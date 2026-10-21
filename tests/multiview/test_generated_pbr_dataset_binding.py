"""Dataset-only PBR binding fixtures; no neural or actual test-quality work."""
import copy
import json
from pathlib import Path

import pytest

from experiments.deformnext_retargeting import bind_generated_pbr_dataset as binding
from experiments.deformnext_retargeting.frozen_cascade_metrics import verify_test_PBR_binding


def put(path,value,*,jsonl=False):
    path.write_text(''.join(json.dumps(r)+'\n' for r in value) if jsonl else json.dumps(value))
    return dict(path=str(path),SHA256=binding.sha(path))


def fixture(tmp_path):
    head=tmp_path/'TRAIN-matcher.pt';head.write_bytes(b'synthetic matching weights, not a trained model')
    digest=binding.sha(head)
    original=dict(models={'denoiser':{'frozen_base_and_LoRA':True}},seed=101,
        trainer={'args':{'lora':{'rank':16},'deform_conditioner':dict(q2c_checkpoint=str(head),freeze_q2c=True,
            topk=64,g_max=.3,score_temperature=.7,deformation_dim=2048,value_dim=2048)}},
        dataset={'name':'DeformNeXtDeformationQueryPBR1024','args':dict(resolution=1024,image_size=1024,
            path_map={'/file4':'/old/cache/file4'},match_root='/old/matches',expected_q2c_sha256=digest,
            pbr_slat_normalization={'mean':[1.],'std':[2.]})},
        validation={'roots_file':'/old/validation.jsonl'},sampler={'steps':12,'guidance':3.})
    recipe=put(tmp_path/'original.json',original)
    manifest=put(tmp_path/'test.jsonl',[dict(sample_id=f'test-{i}',split='test',source_cluster_id=f'source-{i%134}')
                                     for i in range(508)],jsonl=True)
    qp=put(tmp_path/'qa-plan.json',dict(published_manifest=manifest['path'],file4_cache='/new/cache/file4',
        match_root='/new/matches',projection_root='/new/maps'))
    qa=put(tmp_path/'qa.json',dict(status='PASS_ACTUAL_TEST508_NATIVE_CACHE_AND_LOADER_QA',assets=508,
        SS_PBR_cache_ready=True,no_test_asset_dropped=True,train_validation_identity_disjoint=True,
        all_input_RGB_fingerprints_verified=True,no_candidate_model_forward_or_test_quality_metric=True,
        matching_head_training_or_calibration_updates=0,Omega_completed_updates=23597,
        published_manifest_SHA256=manifest['SHA256'],published_manifest=manifest['path'],frozen_TRAIN_q2c_SHA256=digest,
        synthetic_unit_fixture_not_real_cache_QA=True))
    return dict(scope='final-test508',original_conditioner_config=recipe,manifest=manifest,cache_QA=qa,cache_QA_plan=qp)


def prepared(tmp_path):
    plan=fixture(tmp_path)
    record=binding.prepare(plan,tmp_path/'bound')
    spec=dict(path=str(tmp_path/'bound/binding.json'),SHA256=binding.sha(tmp_path/'bound/binding.json'))
    return plan,record,spec


def validate(plan,record,spec):
    return binding.validate_binding(spec,conditioner_config=record['conditioner_config'],
        cache_QA=plan['cache_QA'],cache_QA_plan=plan['cache_QA_plan'],manifest=plan['manifest'])


def test_actual_weight_bytes_and_every_non_data_field_are_preserved(tmp_path):
    plan,record,spec=prepared(tmp_path)
    original=json.loads(Path(plan['original_conditioner_config']['path']).read_text())
    actual=json.loads(Path(record['conditioner_config']['path']).read_text())
    original_bytes=Path(plan['original_conditioner_config']['path']).read_bytes()
    restored=copy.deepcopy(actual)
    for key in ['path_map','match_root']:restored['dataset']['args'][key]=original['dataset']['args'][key]
    assert restored==original
    assert actual['trainer']==original['trainer'] and actual['sampler']==original['sampler']
    assert actual['dataset']['args']['path_map']=={'/file4':'/new/cache/file4'}
    assert actual['dataset']['args']['match_root']=='/new/matches'
    assert validate(plan,record,spec)==record
    assert Path(plan['original_conditioner_config']['path']).read_bytes()==original_bytes


@pytest.mark.parametrize('problem',['g_max','temperature','topk','DQ-channels','LoRA','normalization','sampler','stale-root'])
def test_rehashed_config_cannot_change_the_scientific_method(tmp_path,problem):
    plan,record,spec=prepared(tmp_path)
    path=Path(record['conditioner_config']['path']);cfg=json.loads(path.read_text());args=cfg['trainer']['args']
    if problem=='g_max':args['deform_conditioner']['g_max']=.9
    if problem=='temperature':args['deform_conditioner']['score_temperature']=.3
    if problem=='topk':args['deform_conditioner']['topk']=1
    if problem=='DQ-channels':args['deform_conditioner']['deformation_dim']=0
    if problem=='LoRA':args['lora']['rank']=8
    if problem=='normalization':cfg['dataset']['args']['pbr_slat_normalization']['mean']=[0.]
    if problem=='sampler':cfg['sampler']['steps']=24
    if problem=='stale-root':cfg['dataset']['args']['match_root']='/old/matches'
    record['conditioner_config']=put(path,cfg)
    spec=put(Path(spec['path']),record)
    with pytest.raises(ValueError,match='constructor/model parameters|data roots'):
        validate(plan,record,spec)


@pytest.mark.parametrize('problem',['QA-incomplete','matching-updated','old-Omega','head-unfrozen','head-bytes-changed','head-digest-changed'])
def test_incomplete_or_changed_cache_head_is_rejected_before_output(tmp_path,problem):
    plan=fixture(tmp_path)
    qa_path=Path(plan['cache_QA']['path']);qa=json.loads(qa_path.read_text())
    recipe=Path(plan['original_conditioner_config']['path']);cfg=json.loads(recipe.read_text())
    if problem=='QA-incomplete':qa['SS_PBR_cache_ready']=False
    if problem=='matching-updated':qa['matching_head_training_or_calibration_updates']=1
    if problem=='old-Omega':qa['Omega_completed_updates']=0
    if problem=='head-unfrozen':cfg['trainer']['args']['deform_conditioner']['freeze_q2c']=False
    if problem=='head-bytes-changed':Path(cfg['trainer']['args']['deform_conditioner']['q2c_checkpoint']).write_bytes(b'changed')
    if problem=='head-digest-changed':qa['frozen_TRAIN_q2c_SHA256']='0'*64
    plan['cache_QA']=put(qa_path,qa);plan['original_conditioner_config']=put(recipe,cfg)
    with pytest.raises(ValueError):binding.prepare(plan,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_replay_constructor_and_actual_bound_path_are_frozen(tmp_path):
    plan,record,spec=prepared(tmp_path)
    model=dict(pbr_runtime_arm='retargeting',pbr_conditioner_checkpoint={'path':'fixture.pt'},
               pbr_dataset_binding=spec,pbr_conditioner_config=record['conditioner_config'])
    replay=dict(arm='retargeting',conditioner_config=record['conditioner_config']['path'])
    args={k:plan[k] for k in ['cache_QA','cache_QA_plan','manifest']}
    assert verify_test_PBR_binding(replay,model,**args)==record
    with pytest.raises(ValueError,match='constructor differs'):
        verify_test_PBR_binding(dict(replay,arm='direct'),model,**args)
    with pytest.raises(ValueError,match='config differs'):
        verify_test_PBR_binding(dict(replay,conditioner_config=plan['original_conditioner_config']['path']),model,**args)
    with pytest.raises(ValueError,match='test-cache binding'):
        verify_test_PBR_binding(replay,{k:v for k,v in model.items() if k!='pbr_dataset_binding'},**args)
    assert verify_test_PBR_binding({'arm':'vanilla'},dict(pbr_runtime_arm='vanilla',pbr_conditioner_checkpoint=None),**args) is None


def test_scope_and_prior_output_cannot_be_promoted_or_overwritten(tmp_path):
    plan=fixture(tmp_path)
    with pytest.raises(ValueError,match='final-test508'):binding.prepare(dict(plan,scope='engineering-validation32'),tmp_path/'out')
    out=tmp_path/'out';out.mkdir();(out/'keep').write_text('original')
    with pytest.raises(FileExistsError):binding.prepare(plan,out)
    assert (out/'keep').read_text()=='original'
