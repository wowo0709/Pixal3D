"""Synthetic metadata fixtures only: no real freeze, NN, or quality evidence."""
import copy
import json
from pathlib import Path

import pytest

from experiments.deformnext_retargeting import evaluate_bound_population as bound
from experiments.deformnext_retargeting import prepare_frozen_ss_population as factory


def put(path, value, *, jsonl=False, raw=False):
    if raw:
        path.write_text(value)
    else:
        path.write_text(''.join(json.dumps(row)+'\n' for row in value) if jsonl else json.dumps(value))
    return dict(path=str(path), SHA256=bound.sha(path))


def fixture(tmp_path):
    def rows(split, count):
        return [dict(sample_id=f'{split}-sample-{i}', split=split,
                     source_uid=f'{split}-source-{i//4}',
                     source_cluster_id=f'{split}-cluster-{i%134}') for i in range(count)]
    test = rows('test',508)
    ids = [row['sample_id'] for row in test]
    manifest = put(tmp_path/'test.jsonl', test, jsonl=True)
    panel = put(tmp_path/'panel.json', {'sample_ids':ids})
    seeds = put(tmp_path/'seeds.json', {sid:42+i for i,sid in enumerate(ids)})
    native_path = Path(bound.__file__).with_name('evaluate_native_checkpoint.py')
    native = dict(path=str(native_path), SHA256=bound.sha(native_path))
    qa_plan = put(tmp_path/'qa-plan.json', dict(published_manifest=manifest['path'],
        file4_cache='/cache/file4', match_root='/test/matches', projection_root='/test/maps'))
    qa = put(tmp_path/'qa.json', dict(status='PASS_ACTUAL_TEST508_NATIVE_CACHE_AND_LOADER_QA',
        assets=508, SS_PBR_cache_ready=True, no_test_asset_dropped=True,
        train_validation_identity_disjoint=True, all_input_RGB_fingerprints_verified=True,
        no_candidate_model_forward_or_test_quality_metric=True, matching_head_training_or_calibration_updates=0,
        Omega_completed_updates=23597, published_manifest_SHA256=manifest['SHA256'],
        published_manifest=manifest['path'], frozen_TRAIN_q2c_SHA256='0'*64,
        unit_fixture=True, not_real_cache_QA=True))
    models = {}
    aliases = [('v1','evaluate_deterministic_checkpoint'),
               ('retarget','evaluate_consistent_ray_retargeting'),
               ('candidate','member_feature_deformation_runtime'),
               ('zero-source4','evaluate_deterministic_checkpoint'),
               ('zero-edited','evaluate_deterministic_checkpoint')]
    for alias, evaluator in aliases:
        zero = alias.startswith('zero-')
        source_only = alias == 'zero-source4'
        cfg = dict(models={'denoiser':{'unit_fixture':True}}, trainer={'args':{'lora':{'rank':16}}},
            validation=dict(roots_file='/old/validation.jsonl',
                dataset_name='DeformNeXtMemberRawSS64' if alias == 'candidate' else 'DeformNeXtSS64',
                dataset_args=dict(expected_split='validation', path_map={'/file4':'/cache/file4'},
                    match_root=None, projection_root='/old/maps', expected_q2c_sha256='0'*64),
                quick_metrics=dict(sampling_steps=12,panel_size=128)))
        recipe = put(tmp_path/(alias+'-recipe.json'),cfg)
        weight = put(tmp_path/(alias+'-SS.pt'),'synthetic checkpoint bytes; no NN execution',raw=True)
        condition = None if zero or alias == 'v1' else put(tmp_path/(alias+'-condition.pt'),'synthetic conditioner',raw=True)
        source = Path(bound.__file__).with_name(evaluator+'.py')
        mode = 'source-only' if source_only else 'edited-rgb'
        step = 4210 if zero else 480
        args = ['--stage','ss64','--step',str(step),'--output','/old/validation128',
                '--checkpoint',weight['path'],'--arm-config',recipe['path'],
                '--panel-file','/old/panel.json','--seed-map-file','/old/seeds.json',
                '--panel-size','128','--generation-batch','8','--conditioning-mode',mode,
                '--expected-stage-root',str(native_path.parents[2]),'--expected-entry-sha',native['SHA256']]
        if condition:
            args += ['--conditioner-checkpoint',condition['path']]
        if evaluator.startswith('member_'):
            args += ['--phase','evaluation','--expected-assets','128','--source-value-control','normal']
        ss = dict(evaluation_module=evaluator, evaluator_source=dict(path=str(source),SHA256=bound.sha(source)),
            source_SHA256={str(native_path):native['SHA256']}, step=step, conditioning_mode=mode,
            checkpoint=weight, conditioner_checkpoint=condition, arm_config=recipe, native_arguments=args)
        models[alias] = dict(ss64=ss, synthetic_unit_composition=True, not_a_real_selected_model=True)
    evidence = put(tmp_path/'development.json', {'unit_fixture':True,'not_real_development_quality_evidence':True})
    freeze = dict(status='FROZEN_FROM_DEVELOPMENT_ONLY',heldout_quality_read_before_freeze=False,
        development_evidence=[evidence], models=models, selected_candidate='candidate',
        comparison_roles=dict(vanilla='v1',retargeting='retarget',candidate='candidate'),
        contrasts=[['candidate','v1'],['candidate','retarget'],['v1','zero-source4'],['v1','zero-edited']],
        reference_binding={'synthetic_unit_reference':True},
        population=dict(sample_ids=ids,source_clusters_by_sample={r['sample_id']:r['source_cluster_id'] for r in test},
            manifest_SHA256=manifest['SHA256'],seed_map_SHA256=seeds['SHA256'],rendered_cameras=[8,9,10]))
    plan = dict(scope='final-test508',model_freeze=put(tmp_path/'freeze.json',freeze),cache_QA=qa,cache_QA_plan=qa_plan,
        manifest=manifest,panel=panel,seeds=seeds,native_entry=native,
        train_manifest=put(tmp_path/'train.jsonl',rows('train',9149),jsonl=True),
        validation_manifest=put(tmp_path/'validation.jsonl',rows('validation',508),jsonl=True))
    return plan


def mutate(plan, key, change):
    path = Path(plan[key]['path'])
    value = json.loads(path.read_text())
    change(value)
    plan[key] = put(path,value)


def test_prepare_retains_all_five_models_and_relocates_only_population_arguments(tmp_path):
    plan = fixture(tmp_path)
    freeze = json.loads(Path(plan['model_freeze']['path']).read_text())
    recipes = {alias:Path(m['ss64']['arm_config']['path']).read_bytes() for alias,m in freeze['models'].items()}
    result = factory.prepare(plan,tmp_path/'prepared')
    assert result['models'] == list(freeze['models']) and len(result['evaluations']) == 5
    assert result['assets'] == 508 and result['source_clusters'] == 134
    for record in result['evaluations'].values():
        native = json.loads(bound.pinned(record['native_bound_plan']).read_text())
        model = freeze['models'][record['alias']]['ss64']
        assert native['model_alias'] == record['alias'] and native['assets'] == 508
        assert native['evaluation_module'] == model['evaluation_module']
        assert native['dataset_roots'] == {'match_root':'/test/matches','projection_root':'/test/maps'}
        restored = copy.deepcopy(native['native_arguments'])
        for flag in record['native_argument_relocations']:
            restored[restored.index(flag)+1] = bound.argument(model['native_arguments'],flag)
        assert restored == model['native_arguments']
        assert Path(model['arm_config']['path']).read_bytes() == recipes[record['alias']]
        assert not Path(record['output']).exists()


def test_validation_scope_cannot_generate_final_test_jobs(tmp_path):
    with pytest.raises(ValueError,match='development-only freeze'):
        factory.prepare({'scope':'engineering-validation32'},tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('problem',['already-peeked','incomplete-cache','wrong-head','missing-control','changed-cluster','bad-last-weight'])
def test_every_model_and_population_guard_runs_before_any_output(tmp_path,problem):
    plan = fixture(tmp_path)
    if problem == 'already-peeked':
        mutate(plan,'model_freeze',lambda d:d.update(heldout_quality_read_before_freeze=True))
    if problem == 'incomplete-cache':
        mutate(plan,'cache_QA',lambda d:d.update(SS_PBR_cache_ready=False))
    if problem == 'wrong-head':
        mutate(plan,'cache_QA',lambda d:d.update(frozen_TRAIN_q2c_SHA256='1'*64))
    if problem == 'missing-control':
        mutate(plan,'model_freeze',lambda d:d['models'].pop('retarget'))
    if problem == 'changed-cluster':
        def change(d):
            a,b = d['population']['sample_ids'][:2]
            c = d['population']['source_clusters_by_sample']
            c[a],c[b] = c[b],c[a]
        mutate(plan,'model_freeze',change)
    if problem == 'bad-last-weight':
        freeze = json.loads(Path(plan['model_freeze']['path']).read_text())
        Path(freeze['models']['zero-edited']['ss64']['checkpoint']['path']).write_text('changed bytes')
    with pytest.raises(ValueError):
        factory.prepare(plan,tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('problem',['training','wrong-microbatch','wrong-SS-step','score-intervention','wrong-sampling-steps','missing-template'])
def test_native_method_is_not_changed_during_population_binding(tmp_path,problem):
    plan = fixture(tmp_path)
    def change(d):
        ss = d['models']['candidate']['ss64']
        args = ss['native_arguments']
        if problem == 'training':args[args.index('--phase')+1] = 'science'
        if problem == 'wrong-microbatch':args[args.index('--generation-batch')+1] = '16'
        if problem == 'wrong-SS-step':args[args.index('--step')+1] = '384'
        if problem == 'score-intervention':args[args.index('--source-value-control')+1] = 'wrong-feature-values-fixed-use'
        if problem == 'wrong-sampling-steps':
            path = Path(ss['arm_config']['path'])
            cfg = json.loads(path.read_text());cfg['validation']['quick_metrics']['sampling_steps'] = 8
            ss['arm_config'] = put(path,cfg)
        if problem == 'missing-template':ss.pop('native_arguments')
    mutate(plan,'model_freeze',change)
    with pytest.raises(ValueError):
        factory.prepare(plan,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_existing_output_and_duplicate_population_flags_are_rejected(tmp_path):
    plan = fixture(tmp_path)
    out = tmp_path/'out';out.mkdir();(out/'preserve').write_text('original')
    with pytest.raises(FileExistsError):factory.prepare(plan,out)
    assert (out/'preserve').read_text() == 'original'
    args = ['--output','/old','--panel-size','128','--panel-file','/old/panel','--seed-map-file','/old/seeds',
            '--panel-file','/second/panel']
    with pytest.raises(ValueError,match='one explicit argument'):
        factory.population_arguments(args,output='/new',panel={'path':'/panel'},seeds={'path':'/seeds'})
