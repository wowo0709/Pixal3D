"""Own-family population plans and held-out gates; no neural forward."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.deformnext_retargeting.pipeline_population_contract_v2 import ARMS, ENGINEERING, FINAL
from experiments.deformnext_retargeting.selected_population_plan_factory import SCHEMA, make_plan, sources
from experiments.deformnext_retargeting.selected_population_plan_factory import read_input
from experiments.deformnext_retargeting.cascade_view_roles import ClosedInputs
from test_pipeline_followup_completion_v2 import bound_inputs
from test_unified_population_contract import inputs as original_inputs


def spec(path):
    return dict(path=path, SHA256=hashlib.sha256(path.encode()).hexdigest(), receipt='/inputs/file4-closure.json')


def inputs(tmp_path, arm='E_LEARNED_RAW', scope=ENGINEERING):
    values = bound_inputs(tmp_path, arm, scope) if arm.startswith('E_') else original_inputs(tmp_path, arm, scope)
    p, bundle, cfg, rows, ids, seeds, kw = values[:7]
    binding = kw['binding']; population = {k: spec(v['path']) for k, v in p['population_inputs'].items()}
    population.update(train_manifest=spec('/train.jsonl'), validation_manifest=spec('/validation.jsonl'))
    kw['cache_QA']['published_manifest_SHA256'] = population['manifest']['SHA256']
    kw['freeze'].update(schema='selected-physical-pipeline-development-freeze-v2')
    kw['freeze']['development_evidence'] = [{k: v for k, v in spec('/dev/audit.json').items() if k != 'receipt'}]
    bspec = spec('/bundles/'+arm+'/bundle.json'); bspec['receipt'] = '/bundles/'+arm+'/file4-closure.json'
    request = dict(schema=SCHEMA, name='own-'+arm, module_arm=arm, model_binding=binding,
        model_alias='Top1', bundle=bspec, population_scope=scope, population_inputs=population,
        dataset_roots=p['dataset_roots'], assigned_GPUs=['0', '6'], output='/outputs/'+arm+'/native',
        wrapper_output='/outputs/'+arm+'/wrapper', repo='/repo', python='/python', canonical_local_root='/outputs',
        source_SHA256={path:'a'*64 for path in sources(arm)}, frozen_inputs=[])
    kwargs = dict(bundle=bundle, config=cfg, rows=rows, panel=ids, seeds=seeds,
        train_rows=kw['train_rows'], validation_rows=kw['validation_rows'])
    if scope == FINAL:
        for name in ['model_freeze', 'cache_QA', 'cache_QA_plan', 'engineering_parity']:
            request[name] = spec('/inputs/'+name+'.json')
        kwargs.update(freeze=kw['freeze'], cache_QA=kw['cache_QA'], cache_QA_plan=kw['cache_QA_plan'], parity=kw['parity'])
    else:
        request['reference_evaluation'] = spec('/reference/'+arm+'.json')
        ev = dict(status='PASS', stage='ss64', step=480, weights='raw', native_variant=binding['variant'],
            actual_source_tensors=binding['source_tensors'], actual_LoRA_tensors=360,
            optimizer_updates_during_evaluation=0, target_modality='rgb', input_states=['M1','M0','M0','M0'],
            input_camera_ids=[8,9,10,11], generation_batch_size=8, sampling_steps=12, guidance=3.,
            deterministic_algorithms=True, evaluation_mode='final128', native_task_checkpoint_bundle=bspec['path'],
            verified_inputs={binding[k]['path']:binding[k]['SHA256'] for k in
                ['native_config','checkpoint','conditioner_checkpoint'] if binding[k] is not None})
        for key, field in [('native_config','arm_config'), ('checkpoint','checkpoint'),
                           ('conditioner_checkpoint','conditioner_checkpoint')]:
            ev[field] = None if binding[key] is None else binding[key]['path']
        ev.update(quick_panel_ids=[r['sample_id'] for r in rows[:128]],
            per_asset=[dict(sample_id=r['sample_id'], seed=100+i) for i,r in enumerate(rows[:128])])
        if arm.startswith('E_'):
            ev.update(conditioning_schema='consistent-raw-ray-evidence-native-v1',
                followup_study_schema='consistent-raw-ray-evidence-followup-v1',
                shared_carrier_use_retarget_evidence=True, learned_projection_tensors=0,
                learned_attention_head_tensors=binding['learned_attention_head_tensors'], training_seed=101)
        kwargs['reference'] = ev
    return request, kwargs


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('scope', [ENGINEERING, FINAL])
def test_all14_families_have_own_population_plan_without_changing_recipe(tmp_path, arm, scope):
    request, kwargs = inputs(tmp_path, arm, scope); before = deepcopy((request, kwargs))
    result = make_plan(request, **kwargs)
    assert (request, kwargs) == before
    assert result['module_arm'] == arm and result['model_binding'] == kwargs.get('parity', {}).get('model_binding', request['model_binding'])
    assert result['mode'] == 'bound-population' and result['generation_batch_size'] == 8
    assert result['deterministic_algorithms'] and result['expected_assets'] == (36 if scope == ENGINEERING else 508)
    assert all(result['closed_inputs'].get(v['path']) == v['receipt'] for v in request['population_inputs'].values())
    assert result['bundle'] == request['bundle']['path']


@pytest.mark.parametrize('damage', ['wrong-binding','other-bundle','seed-prefix','panel-prefix','reference-mode',
    'reference-model','source-pin','missing-receipt','same-output','implicit-GPU','duplicate-GPU','legacy-freeze',
    'no-freeze','prior-test-read','overlap','QA','one-load','q2c','wrong-family','missing-frozen-inputs','frozen-receipt'])
def test_wrong_model_or_incomplete_evidence_cannot_publish_test_or_validation_plan(tmp_path, damage):
    final = damage in ['legacy-freeze','no-freeze','prior-test-read','overlap','QA','one-load','q2c']
    request, kwargs = inputs(tmp_path, scope=FINAL if final else ENGINEERING)
    if damage == 'wrong-binding':request['model_binding']['learned_projection_tensors'] = 2
    elif damage == 'other-bundle':request['bundle']['path'] = '/bundles/other/bundle.json'
    elif damage == 'seed-prefix':kwargs['seeds'][kwargs['panel'][0]] += 1
    elif damage == 'panel-prefix':kwargs['panel'][0],kwargs['panel'][1] = kwargs['panel'][1],kwargs['panel'][0]
    elif damage == 'reference-mode':kwargs['reference']['evaluation_mode'] = 'archive32'
    elif damage == 'reference-model':kwargs['reference']['checkpoint'] = '/other.pt'
    elif damage == 'source-pin':request['source_SHA256'].pop(sources('E_LEARNED_RAW')[0])
    elif damage == 'missing-receipt':request['population_inputs']['panel'].pop('receipt')
    elif damage == 'same-output':request['wrapper_output'] = request['output']
    elif damage == 'implicit-GPU':request['assigned_GPUs'] = []
    elif damage == 'duplicate-GPU':request['assigned_GPUs'] = ['0','0']
    elif damage == 'legacy-freeze':kwargs['freeze']['schema'] = 'legacy-four-model-freeze'
    elif damage == 'no-freeze':kwargs['freeze'] = None
    elif damage == 'prior-test-read':kwargs['freeze']['heldout_quality_read_before_freeze'] = True
    elif damage == 'overlap':kwargs['rows'][0]['source_cluster_id'] = kwargs['train_rows'][0]['source_cluster_id']
    elif damage == 'QA':kwargs['cache_QA']['SS_PBR_cache_ready'] = False
    elif damage == 'one-load':kwargs['parity']['independent_model_loads'] = 1
    elif damage == 'q2c':kwargs['cache_QA']['frozen_TRAIN_q2c_SHA256'] = 'other'
    elif damage == 'wrong-family':request['module_arm'] = 'FULL'
    elif damage == 'missing-frozen-inputs':request.pop('frozen_inputs')
    else:request['frozen_inputs'] = [dict(path='/frozen/head.pt',SHA256='a'*64)]
    with pytest.raises(ValueError): make_plan(request, **kwargs)


def test_original_manifest_path_kept_with_explicit_byte_identical_canonical_copy(tmp_path):
    request, kwargs = inputs(tmp_path)
    original = request['population_inputs']['manifest']['path']
    request['canonical_input_aliases'] = {original:'/canonical/validation.jsonl'}
    result = make_plan(request, **kwargs)
    assert result['population_inputs']['manifest']['path'] == original
    assert result['input_SHA256'][original] == request['population_inputs']['manifest']['SHA256']
    assert original not in result['closed_inputs']
    assert result['closed_inputs']['/canonical/validation.jsonl'] == request['population_inputs']['manifest']['receipt']


@pytest.mark.parametrize('damage',[None,'changed-original','changed-copy','absent-from-receipt'])
def test_real_bytes_and_receipt_are_required_for_external_recipe_manifest(tmp_path,damage):
    original = tmp_path/'external/manifest.jsonl'; original.parent.mkdir()
    original.write_text('{"sample_id":"one"}\n')
    root = tmp_path/'canonical'; root.mkdir(); copy = root/'copy.jsonl'; copy.write_bytes(original.read_bytes())
    digest = hashlib.sha256(copy.read_bytes()).hexdigest(); receipt = root/'file4-closure.json'
    receipt.write_text(json.dumps(dict(status='PASS',local_equals_actual_file4_SHA256=True,
        files=[dict(relative_path='copy.jsonl',sha256=digest,bytes=copy.stat().st_size)])))
    item = dict(path=str(original),SHA256=digest,receipt=str(receipt)); aliases={str(original):str(copy)}
    if damage == 'changed-original':original.write_text('{}\n')
    elif damage == 'changed-copy':copy.write_text('{}\n')
    elif damage == 'absent-from-receipt':aliases[str(original)] = str(root/'other.jsonl')
    if damage:
        with pytest.raises((ValueError,FileNotFoundError)):read_input(item,ClosedInputs(root),aliases,lines=True)
    else:assert read_input(item,ClosedInputs(root),aliases,lines=True) == [dict(sample_id='one')]
