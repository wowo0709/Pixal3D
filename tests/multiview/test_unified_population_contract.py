"""Population/model binding tests; no CPU or GPU neural forward."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting.evaluate_unified_bound_population import dataset_binding
from experiments.deformnext_retargeting.module_pipeline_protocol_v2 import protocol
from experiments.deformnext_retargeting.unified_population_contract import (
    ALIASES, ENGINEERING, FINAL, SCHEMA, SOURCE_COUNTS, bundle_validator,
    evaluator_binding, family, model_binding, required_sources, validate_plan,
)
from test_module_confirmation_recipe import inputs as recipe_inputs
from test_native_population_contract import inputs as population_inputs


def inputs(tmp_path, arm='C_FIXED_TOPK', scope=FINAL):
    cfg, _ = recipe_inputs(tmp_path, arm)
    cfg['validation'].update(roots_file='/validation.jsonl')
    cfg['validation']['dataset_args'].update(match_root=None, projection_root='/val-proj',
                                           expected_q2c_sha256='D1')
    if arm not in ('V1', 'FULL'):
        p = Path(cfg['experiment']['reference_science_config'])
        ref = json.loads(p.read_text())
        ref['validation'] = deepcopy(cfg['validation'])
        p.write_text(json.dumps(ref))
        cfg['experiment']['reference_science_config_SHA256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    core = bundle_validator(arm)
    schema = (core.BUNDLE_SCHEMA if arm != 'C_DECODER'
              else 'cold-native-task-Top1-SS-checkpoint-pair-v1')
    root = cfg['output_dir']
    bundle = dict(schema=schema, variant=ALIASES.get(arm, arm), step=480,
        native_config=root+'/config.json', checkpoint=root+'/ckpts/denoiser_step0000480.pt',
        conditioner_checkpoint=None if arm == 'V1' else root+'/ckpts/deform_conditioner_step0000480.pt',
        source_teacher_updates=0, native_task_objective='SS_flow_mse', native_validation=dict(step=480))
    hashes = {p: 'a'*64 for k, p in bundle.items()
              if k in ('native_config', 'checkpoint', 'conditioner_checkpoint') and p is not None}
    own = model_binding(bundle, cfg, hashes, arm=arm)
    plan, _, _, rows, ids, seeds, kw = population_inputs(scope)
    plan.update(module_arm=arm, model_binding=own)
    kw['binding'] = own
    kw['freeze']['models']['Top1']['ss64'] = own
    kw['parity']['model_binding'] = own
    return plan, bundle, cfg, rows, ids, seeds, kw, hashes


@pytest.mark.parametrize('arm', protocol()['arms'])
@pytest.mark.parametrize('scope', [ENGINEERING, FINAL])
def test_every_physical_family_keeps_original_model_and_native_population_guards(tmp_path, arm, scope):
    p, b, c, rows, ids, seeds, kw, _ = inputs(tmp_path, arm, scope)
    before = deepcopy(c)
    result = validate_plan(p, b, c, rows, ids, seeds, **kw)
    assert c == before
    assert kw['binding']['schema'] == SCHEMA
    assert kw['binding']['source_tensors'] == SOURCE_COUNTS[arm]
    assert kw['binding']['runtime_family'] == family(arm)
    if scope == FINAL:
        assert result['validation']['dataset_args']['expected_split'] == 'test'
        assert result['validation']['dataset_args']['projection_root'] == '/test-proj'
        assert result['validation']['dataset_args']['match_root'] is None
        result['validation'] = before['validation']
        assert result == before
    else:
        assert result == before


@pytest.mark.parametrize('damage', ['different-family', 'schema', 'source-path', 'denoiser-path',
    'step384', 'float-seed', 'missing-SHA', 'non-hex-SHA', 'fake-source-count', 'binding-step',
    'wrong-arm', 'float-batch', 'float-population', 'no-freeze', 'prior-test-read',
    'overlap', 'QA', 'q2c', 'one-parity-load', 'tail-parity', 'wrong-freeze-model'])
def test_wrong_weights_or_incomplete_evidence_cannot_release_any_test_forward(tmp_path, damage):
    p, b, c, rows, ids, seeds, kw, hashes = inputs(tmp_path)
    if damage == 'different-family':
        b['variant'] = 'FULL'
    elif damage == 'schema':
        b['schema'] = 'cold-native-task-SS-checkpoint-pair-v1'
    elif damage == 'source-path':
        b['conditioner_checkpoint'] = '/other-source.pt'
    elif damage == 'denoiser-path':
        b['checkpoint'] = '/other-denoiser.pt'
    elif damage == 'step384':
        b['step'] = 384
    elif damage == 'float-seed':
        c['training_seed'] = 101.0
    elif damage in ('missing-SHA', 'non-hex-SHA'):
        kw['binding']['checkpoint']['SHA256'] = '' if damage == 'missing-SHA' else 'g'*64
    elif damage == 'fake-source-count':
        kw['binding']['source_tensors'] = 19
    elif damage == 'binding-step':
        kw['binding']['step'] = 96
    elif damage == 'wrong-arm':
        p['module_arm'] = 'FULL'
    elif damage == 'float-batch':
        p['generation_batch_size'] = 8.0
    elif damage == 'float-population':
        p['expected_assets'] = 508.0
    elif damage == 'no-freeze':
        kw['freeze'] = None
    elif damage == 'prior-test-read':
        kw['freeze']['heldout_quality_read_before_freeze'] = True
    elif damage == 'overlap':
        rows[0]['source_cluster_id'] = kw['train_rows'][0]['source_cluster_id']
    elif damage == 'QA':
        kw['cache_QA']['all508_paired_coverage'] = False
    elif damage == 'q2c':
        kw['cache_QA']['frozen_TRAIN_q2c_SHA256'] = 'other'
    elif damage == 'one-parity-load':
        kw['parity']['independent_model_loads'] = 1
    elif damage == 'tail-parity':
        kw['parity']['tail_generation_batch'] = 0
    elif damage == 'wrong-freeze-model':
        kw['freeze']['models']['Top1']['ss64'] = dict(kw['binding'], module_arm='FULL')
    with pytest.raises(ValueError):
        validate_plan(p, b, c, rows, ids, seeds, **kw)


@pytest.mark.parametrize('arm', protocol()['arms'])
def test_each_entry_is_the_existing_family_and_is_restored_even_after_exception(arm):
    original = importlib.import_module('experiments.deformnext_retargeting.evaluate_r_native_checkpoint')
    previous = original.validate_bundle, original.cold_source_binding
    with pytest.raises(RuntimeError, match='owned'):
        with evaluator_binding(arm) as core:
            assert callable(core.main) and callable(core.validate_panel)
            assert core.__name__.endswith({
                'r': 'evaluate_r_native_checkpoint', 'r_top1': 'evaluate_r_native_checkpoint',
                'projected': 'evaluate_projected_native_checkpoint',
                'raw_correspondence': 'evaluate_raw_correspondence_checkpoint',
                'unified_design': 'evaluate_unified_design_checkpoint'}[family(arm)])
            raise RuntimeError('owned failure')
    assert (original.validate_bundle, original.cold_source_binding) == previous
    paths = required_sources(arm)
    assert paths and len(paths) == len(set(paths)) and all(Path(p).is_file() for p in paths)
    assert any(p.endswith('/evaluate_unified_bound_population.py') for p in paths)


def test_dataset_binding_restores_original_evaluator_and_only_temporarily_changes_validation():
    original = dict(validation=dict(roots_file='validation'), trainer=dict(actual='original'))
    rebound = deepcopy(original)
    rebound['validation']['roots_file'] = 'test'
    old = lambda *_: None
    core = SimpleNamespace(validate_panel=old)
    with pytest.raises(RuntimeError, match='owned'):
        with dataset_binding(core, lambda *_: (['sample'], {'sample': 1}, rebound)):
            assert core.validate_panel({}, {}, original) == (['sample'], {'sample': 1})
            assert original['validation']['roots_file'] == 'test'
            raise RuntimeError('owned failure')
    assert original['validation']['roots_file'] == 'validation' and core.validate_panel is old


def test_legacy_or_unplanned_models_cannot_be_aliased_into_the_finite12_study():
    with pytest.raises(ValueError, match='twelve'):
        family('R_feature')
