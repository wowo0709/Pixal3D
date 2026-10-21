"""Held-out data must not change model math or bypass freeze/coverage guards."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting import evaluate_bound_population as module


def rows(split, count):
    return [dict(sample_id=f'{split}-sample-{i}', source_uid=f'{split}-source-{i//4}',
                 source_cluster_id=f'{split}-cluster-{i//4}', split=split) for i in range(count)]


def populations():
    test, train, val = rows('test',508), rows('train',9149), rows('validation',508)
    panel=[row['sample_id'] for row in test]
    return test,train,val,panel,{sid:42+i for i,sid in enumerate(panel)}


def cfg():
    return dict(models={'denoiser':{'name':'unchanged','args':{'resolution':16}}},
                trainer={'args':{'lora':{'rank':16},'deform_conditioner':None}},
                validation=dict(roots_file='/old/validation.jsonl',dataset_name='DeformNeXtSS64',
                                dataset_args=dict(expected_split='validation',match_root=None,projection_root='/old/maps'),
                                quick_metrics=dict(panel_size=32,sampling_steps=12,seed=42)))


def test_actual_native_source_only_adds_explicit_data_projection(tmp_path):
    path=Path(module.__file__).with_name('evaluate_native_checkpoint.py')
    source=path.read_text()
    tree=module.native_data_tree(source)
    compile(tree,'<native-data-projection>','exec')
    original_calls=[ast.dump(node) for node in ast.walk(ast.parse(source)) if isinstance(node,ast.Call)]
    actual_calls=[ast.dump(node) for node in ast.walk(tree) if isinstance(node,ast.Call)
                  and not (isinstance(node.func,ast.Name) and node.func.id=='_bound_dataset')]
    assert actual_calls==original_calls
    selected=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
    position=next(i for i,node in enumerate(selected.body) if isinstance(node,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='cfg' for t in node.targets))
    file=tmp_path/'original.json'
    original=cfg()
    file.write_text(json.dumps(original))
    raw=file.read_bytes()
    namespace=dict(args=SimpleNamespace(arm_config=file),json=json,
                   _bound_dataset=lambda c:module.project_dataset(c,manifest='/new/test.jsonl',expected_split='test',
                                                                  match_root='/new/matches',projection_root='/new/maps'))
    exec(compile(ast.Module(body=selected.body[position:position+2],type_ignores=[]),str(path),'exec'),namespace)
    expected=copy.deepcopy(original)
    expected['validation']['roots_file']='/new/test.jsonl'
    expected['validation']['dataset_args'].update(expected_split='test',projection_root='/new/maps')
    assert namespace['cfg']==expected and file.read_bytes()==raw
    assert namespace['cfg']['trainer']==original['trainer'] and namespace['cfg']['models']==original['models']


def test_physical_RAW_dataset_keeps_online_matching_and_RAW_channels():
    original=cfg()
    original['validation']['dataset_name']='DeformNeXtMemberRawSS64'
    original['validation']['dataset_args'].update(deformation_query_channels=2048,task_matching_features=False)
    projected=module.project_dataset(original,manifest='/test.jsonl',expected_split='test',
                                      match_root='/unused-cached-matches',projection_root='/test/maps')
    assert projected['validation']['dataset_args']['match_root'] is None
    assert projected['validation']['dataset_args']['deformation_query_channels']==2048
    assert projected['validation']['dataset_args']['task_matching_features'] is False
    assert projected['validation']['dataset_name']=='DeformNeXtMemberRawSS64'


def test_cached_correspondence_mode_rebinds_its_existing_root_only():
    original=cfg()
    original['validation']['dataset_args']['match_root']='/old/matches'
    projected=module.project_dataset(original,manifest='/test.jsonl',expected_split='test',
                                      match_root='/test/matches',projection_root='/test/maps')
    assert projected['validation']['dataset_args']['match_root']=='/test/matches'
    assert original['validation']['dataset_args']['match_root']=='/old/matches'


def test_absent_optional_roots_do_not_enable_new_sidecar_inputs():
    original=cfg()
    original['validation']['dataset_args']={'expected_split':'validation'}
    projected=module.project_dataset(original,manifest='/test.jsonl',expected_split='test',
                                      match_root='/unused/matches',projection_root='/unused/maps')
    assert projected['validation']['dataset_args']=={'expected_split':'test'}


def test_native_recipe_read_site_drift_is_rejected():
    path=Path(module.__file__).with_name('evaluate_native_checkpoint.py')
    with pytest.raises(ValueError,match='exactly one'):
        module.native_data_tree(path.read_text()+"\ncfg=json.loads('{}')\n")


def test_complete_source_disjoint_test_population_accepted():
    test,train,val,panel,seeds=populations()
    assert module.validate_population(test,scope='final-test508',panel=panel,seeds=seeds,
                                      train_rows=train,validation_rows=val)=='test'


@pytest.mark.parametrize('field',module.IDENTITIES)
def test_any_test_train_identity_overlap_is_rejected(field):
    test,train,val,panel,seeds=populations()
    train[0][field]=test[0][field]
    with pytest.raises(ValueError,match='overlaps train'):
        module.validate_population(test,scope='final-test508',panel=panel,seeds=seeds,train_rows=train,validation_rows=val)


def test_missing_one_test_asset_is_not_a_successful_final_panel():
    test,train,val,panel,seeds=populations()
    seeds.pop(panel.pop())
    with pytest.raises(ValueError,match='all508'):
        module.validate_population(test[:-1],scope='final-test508',panel=panel,seeds=seeds,train_rows=train,validation_rows=val)


def test_disjointness_check_cannot_use_only_a_subset_of_training():
    test,train,val,panel,seeds=populations()
    with pytest.raises(ValueError,match='complete original9149'):
        module.validate_population(test,scope='final-test508',panel=panel,seeds=seeds,train_rows=train[:32],validation_rows=val)


def test_engineering_mode_cannot_be_used_to_peek_at_test_predictions():
    test,train,val,panel,seeds=populations()
    with pytest.raises(ValueError,match='identity and split'):
        module.validate_population(test,scope='engineering-validation32',panel=panel[:32],
                                    seeds={sid:seeds[sid] for sid in panel[:32]},train_rows=train,validation_rows=val)


@pytest.mark.parametrize('problem',['missing','bool','duplicate'])
def test_noise_and_panel_pairing_cannot_silently_change(problem):
    test,train,val,panel,seeds=populations()
    if problem=='missing':seeds.pop(panel[0])
    if problem=='bool':seeds[panel[0]]=True
    if problem=='duplicate':panel[-1]=panel[0]
    with pytest.raises(ValueError,match='matched unique panel'):
        module.validate_population(test,scope='final-test508',panel=panel,seeds=seeds,train_rows=train,validation_rows=val)


def test_final_test_requires_freeze_and_QA_before_any_paths_or_models_are_read():
    with pytest.raises(ValueError,match='frozen model set'):
        module.load_binding(dict(scope='final-test508'))


def engineering(tmp_path):
    def spec(path,payload):
        path.write_text(payload)
        return dict(path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
    original=cfg()
    recipe=spec(tmp_path/'arm.json',json.dumps(original))
    checkpoint=spec(tmp_path/'denoiser.pt','synthetic bytes, no model execution')
    source=Path(module.__file__).with_name('evaluate_deterministic_checkpoint.py')
    native=Path(module.__file__).with_name('evaluate_native_checkpoint.py')
    native_spec=dict(path=str(native),SHA256=module.sha(native))
    panel=spec(tmp_path/'panel.json',json.dumps({'sample_ids':[f'validation-sample-{i}' for i in range(32)]}))
    seeds=spec(tmp_path/'seeds.json',json.dumps({f'validation-sample-{i}':42+i for i in range(32)}))
    manifest=spec(tmp_path/'validation.jsonl',''.join(json.dumps(row)+'\n' for row in rows('validation',508)))
    training=spec(tmp_path/'train.jsonl',''.join(json.dumps(row)+'\n' for row in rows('train',9149)))
    model=dict(evaluation_module='evaluate_deterministic_checkpoint',step=480,conditioning_mode='edited-rgb',
                checkpoint=checkpoint,conditioner_checkpoint=None,arm_config=recipe,
                evaluator_source=dict(path=str(source),SHA256=module.sha(source)),
                source_SHA256={str(native):module.sha(native)})
    args=['--stage','ss64','--step','480','--output',str(tmp_path/'output'),
          '--checkpoint',checkpoint['path'],'--arm-config',recipe['path'],'--panel-file',panel['path'],
          '--seed-map-file',seeds['path'],'--panel-size','32','--generation-batch','8',
          '--expected-stage-root',str(native.parents[2]),'--expected-entry-sha',native_spec['SHA256']]
    return dict(scope='engineering-validation32',evaluation_module='evaluate_deterministic_checkpoint',
                native_arguments=args,output=str(tmp_path/'output'),assets=32,native_entry=native_spec,
                manifest=manifest,panel=panel,seeds=seeds,train_manifest=training,validation_manifest=manifest,
                dataset_roots=dict(match_root=None,projection_root='/old/maps'),engineering_model_binding=model)


def test_explicit_engineering_binding_checks_actual_weight_and_source_bytes(tmp_path):
    plan=engineering(tmp_path)
    actual,panel,seeds,split,model=module.load_binding(plan)
    assert len(actual)==508 and len(panel)==len(seeds)==32 and split=='validation'
    Path(model['checkpoint']['path']).write_text('changed bytes')
    with pytest.raises(ValueError,match='byte identity differs'):
        module.load_binding(plan)


def test_training_phase_and_alternate_sampling_batch_are_rejected(tmp_path):
    plan=engineering(tmp_path)
    plan['native_arguments']+=['--phase','science']
    with pytest.raises(ValueError,match='no training phase'):
        module.load_binding(plan)
    plan['native_arguments']=plan['native_arguments'][:-2]
    i=plan['native_arguments'].index('--generation-batch')+1
    plan['native_arguments'][i]='16'
    with pytest.raises(ValueError,match='argument contract differs'):
        module.load_binding(plan)


def final_plan(tmp_path):
    plan=engineering(tmp_path)
    def spec(name,value,jsonl=False):
        path=tmp_path/name
        path.write_text(''.join(json.dumps(row)+'\n' for row in value) if jsonl else json.dumps(value))
        return dict(path=str(path),SHA256=module.sha(path))
    test,_,_,panel,seeds=populations()
    plan.update(scope='final-test508',assets=508,manifest=spec('test.jsonl',test,True),
                panel=spec('all-panel.json',{'sample_ids':panel}),seeds=spec('all-seeds.json',seeds),
                dataset_roots=dict(match_root='/test/matches',projection_root='/test/maps'),model_alias='synthetic')
    for flag,value in [('--panel-size','508'),('--panel-file',plan['panel']['path']),('--seed-map-file',plan['seeds']['path'])]:
        plan['native_arguments'][plan['native_arguments'].index(flag)+1]=value
    model=plan.pop('engineering_model_binding')
    recipe=json.loads(Path(model['arm_config']['path']).read_text())
    recipe['validation']['dataset_args'].update(path_map={'/file4':'/cache/file4'},expected_q2c_sha256='0'*64)
    Path(model['arm_config']['path']).write_text(json.dumps(recipe))
    model['arm_config']['SHA256']=module.sha(model['arm_config']['path'])
    evidence=spec('development-evidence.json',{'unit_fixture':True,'not_model_quality_evidence':True})
    freeze=dict(status='FROZEN_FROM_DEVELOPMENT_ONLY',heldout_quality_read_before_freeze=False,
                population=dict(manifest_SHA256=plan['manifest']['SHA256'],seed_map_SHA256=plan['seeds']['SHA256'],sample_ids=panel),
                development_evidence=[evidence],models={'synthetic':{'ss64':model}})
    plan['model_freeze']=spec('freeze.json',freeze)
    qa=dict(status='PASS_ACTUAL_TEST508_NATIVE_CACHE_AND_LOADER_QA',assets=508,SS_PBR_cache_ready=True,
            no_test_asset_dropped=True,train_validation_identity_disjoint=True,all_input_RGB_fingerprints_verified=True,
            no_candidate_model_forward_or_test_quality_metric=True,matching_head_training_or_calibration_updates=0,
            Omega_completed_updates=23597,published_manifest_SHA256=plan['manifest']['SHA256'],
            published_manifest=plan['manifest']['path'],frozen_TRAIN_q2c_SHA256='0'*64)
    plan['cache_QA']=spec('qa.json',qa)
    plan['cache_QA_plan']=spec('qa-plan.json',dict(published_manifest=plan['manifest']['path'],
                                                file4_cache='/cache/file4',**plan['dataset_roots']))
    return plan


def test_all_final_binding_gates_accept_a_complete_prespecified_fixture(tmp_path):
    plan=final_plan(tmp_path)
    actual,panel,seeds,split,model=module.load_binding(plan)
    assert len(actual)==len(panel)==len(seeds)==508 and split=='test' and model['step']==480


@pytest.mark.parametrize('problem',['already-peeked','cache-incomplete','wrong-head'])
def test_final_metadata_flags_cannot_bypass_freeze_and_cache_guards(tmp_path,problem):
    plan=final_plan(tmp_path)
    key='model_freeze' if problem=='already-peeked' else 'cache_QA'
    path=Path(plan[key]['path'])
    value=json.loads(path.read_text())
    if problem=='already-peeked':value['heldout_quality_read_before_freeze']=True
    if problem=='cache-incomplete':value['SS_PBR_cache_ready']=False
    if problem=='wrong-head':value['frozen_TRAIN_q2c_SHA256']='1'*64
    path.write_text(json.dumps(value))
    plan[key]['SHA256']=module.sha(path)
    with pytest.raises(ValueError,match='development-only|native cache QA|matching-head binding'):
        module.load_binding(plan)
