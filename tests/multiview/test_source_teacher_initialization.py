import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from experiments.deformnext_retargeting.source_teacher_initialization import (
    MARKER, load_teacher_initialization, source_input_binding,
)
from pixal3d.modules.member_feature_source_ray import MemberFeatureSourceRayConditioner
from test_source_contrast_member_feature import fixture


KWARGS = dict(arm='c1', implementation='r9-native-v5', key_dim=1024, value_dim=1024,
    query_dim=1024, hidden_dim=256, topk=64, score_temperature=.5,
    projection_mode='native-bilinear', ray_mode='semantic-top1', ray_chunk_size=4,
    candidate_aggregation='top1')


def saved(tmp_path, module, variant):
    audit = dict(status='PASS_ACTUAL_SOURCE_FEATURE_TEACHER_TRAIN', variant=variant,
        actual_source_parameters=19, epochs=7, optimizer_updates=112, TRAIN_assets=1536,
        source_clusters=1536, validation_source_overlap=0, global_batch=96,
        microbatch=96, accumulation=1, matcher_unchanged=True,
        actual_original_SS_or_PBR_training=False, denoiser_not_loaded_or_updated=True)
    cp, ap = tmp_path/'source.pt', tmp_path/'audit.json'
    torch.save(module.state_dict(), cp); ap.write_text(json.dumps(audit))
    return dict(variant=variant, checkpoint=str(cp), audit=str(ap),
        checkpoint_SHA256=hashlib.sha256(cp.read_bytes()).hexdigest(),
        audit_SHA256=hashlib.sha256(ap.read_bytes()).hexdigest())


def replace_checkpoint(spec, state):
    torch.save(state, spec['checkpoint'])
    spec['checkpoint_SHA256'] = hashlib.sha256(Path(spec['checkpoint']).read_bytes()).hexdigest()


@pytest.mark.parametrize('variant', ['original-NoDQ', 'source-contrast-NoDQ'])
def test_teacher_loads_after_utility_and_before_fresh_optimizer(tmp_path, variant):
    old, contrast, _ = fixture(); teacher = old if variant == 'original-NoDQ' else contrast
    with torch.no_grad():
        for value in teacher.parameters():
            if value.requires_grad: value.add_(.03)
    spec = saved(tmp_path, teacher, variant)
    utility = {k: torch.zeros_like(v) for k,v in teacher.ray_readout.score_selector.state_dict().items()}
    trainer = SimpleNamespace(MemberFeatureSourceRayConditioner=MemberFeatureSourceRayConditioner)
    with source_input_binding(trainer, variant=variant, teacher=spec) as instances:
        module = trainer.MemberFeatureSourceRayConditioner(**KWARGS)
        # Match the test teacher's frozen q2c, independent of constructor RNG.
        module.q2c.load_state_dict(teacher.q2c.state_dict())
        rng = torch.random.get_rng_state()
        module.load_utility(utility)
        assert torch.equal(rng, torch.random.get_rng_state())
        assert instances == [module]
        assert all(torch.equal(v, teacher.state_dict()[k]) for k,v in module.state_dict().items())
        assert module._source_input_initialization['whole_conditioner_loaded_after_utility']
        optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=1e-4)
        assert not optimizer.state and len(optimizer.param_groups[0]['params']) == 19
        with pytest.raises(ValueError, match='only once'): module.load_utility(utility)
    assert trainer.MemberFeatureSourceRayConditioner is MemberFeatureSourceRayConditioner


@pytest.mark.parametrize('damage', ['matcher', 'nonfinite', 'missing', 'marker'])
def test_invalid_state_rejected_before_any_parameter_changes(tmp_path, damage):
    _, module, _ = fixture(); spec = saved(tmp_path, module, 'source-contrast-NoDQ')
    before = {k:v.clone() for k,v in module.state_dict().items()}
    state = {k:v.clone() for k,v in before.items()}
    # An early valid changed tensor must not be copied before a later failure.
    state['fusion_gate'].add_(.1)
    if damage == 'matcher':
        key = next(k for k,v in state.items() if k.startswith('q2c.') and v.is_floating_point())
        state[key].add_(1)
    elif damage == 'nonfinite': state['retargeter.net.2.weight'].flatten()[0] = float('nan')
    elif damage == 'missing': state.pop('retargeter.net.2.bias')
    else: state[MARKER].fill_(0)
    replace_checkpoint(spec, state)
    with pytest.raises(ValueError): load_teacher_initialization(module, spec)
    assert all(torch.equal(v,before[k]) for k,v in module.state_dict().items())


def test_rejects_wrong_variant_hash_or_validation_overlap(tmp_path):
    old, contrast, _ = fixture(); spec = saved(tmp_path, contrast, 'source-contrast-NoDQ')
    with pytest.raises(ValueError, match='actual nineteen-parameter variant'): load_teacher_initialization(old, spec)
    with pytest.raises(ValueError, match='SHA256'):
        load_teacher_initialization(contrast, dict(spec, checkpoint_SHA256='0'*64))
    audit = json.loads(Path(spec['audit']).read_text()); audit['validation_source_overlap'] = 1
    Path(spec['audit']).write_text(json.dumps(audit))
    spec['audit_SHA256'] = hashlib.sha256(Path(spec['audit']).read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='matched TRAIN'): load_teacher_initialization(contrast, spec)


def test_cold_factory_and_exception_restore_binding_without_shared_state():
    trainer = SimpleNamespace(MemberFeatureSourceRayConditioner=MemberFeatureSourceRayConditioner)
    with pytest.raises(RuntimeError, match='owned failure'):
        with source_input_binding(trainer, variant='source-contrast-NoDQ'):
            module = trainer.MemberFeatureSourceRayConditioner(**KWARGS)
            module.load_utility(module.ray_readout.score_selector.state_dict())
            assert int(module.state_dict()[MARKER].item()) == 1
            assert not module._source_input_initialization['additional_TRAIN_M1_source9_10_supervision']
            raise RuntimeError('owned failure')
    assert trainer.MemberFeatureSourceRayConditioner is MemberFeatureSourceRayConditioner


@pytest.mark.parametrize('case', ['warm-native-identity', 'resume', 'undeclared-teacher'])
def test_joint_entry_rejects_ambiguous_or_false_initial_identity_before_gpu(tmp_path, monkeypatch, case):
    from experiments.deformnext_retargeting.source_input_joint_runtime import main
    config = dict(load_dir='', load_ckpt=None, experiment=dict(
        source_input_design='source-contrast-NoDQ', additional_source_teacher_supervision=True))
    if case == 'resume': config['load_dir'] = '/existing-run'
    elif case == 'undeclared-teacher': config['experiment']['additional_source_teacher_supervision'] = False
    cp = tmp_path/'config.json'; cp.write_text(json.dumps(config))
    monkeypatch.setattr('sys.argv', ['entry', '--source-input-design', 'source-contrast-NoDQ',
        '--source-teacher-init-spec', str(tmp_path/'intentionally-not-read.json'),
        '--phase', 'native' if case == 'warm-native-identity' else 'capacity', '--config', str(cp)])
    with pytest.raises(ValueError): main()


@pytest.mark.parametrize('teacher_enabled', [False, True])
def test_science_validation_preserves_honest_metadata_and_all_native_settings(teacher_enabled):
    from experiments.deformnext_retargeting.source_input_joint_runtime import (
        science_recipe_binding, COLD_INITIAL_FUNCTION, TEACHER_INITIAL_FUNCTION,
    )
    variant = 'source-contrast-NoDQ'
    actual = dict(experiment=dict(initial_function=(TEACHER_INITIAL_FUNCTION if teacher_enabled else COLD_INITIAL_FUNCTION),
        source_input_design=variant, additional_source_teacher_supervision=teacher_enabled),
        trainer=dict(args=dict(batch_size_per_gpu=96, optimizer=dict(name='AdamW', args=dict(eps=1e-8)))))
    before = json.dumps(actual, sort_keys=True)
    def sealed_validator(config):
        assert config is not actual and config['experiment']['initial_function'] == COLD_INITIAL_FUNCTION
        assert config['trainer'] == actual['trainer']
        return 'feature-direct-nodq'
    training = SimpleNamespace(validate_feature_recipe=sealed_validator)
    with science_recipe_binding(training, variant=variant, teacher_enabled=teacher_enabled):
        assert training.validate_feature_recipe(actual) == 'feature-direct-nodq'
    assert json.dumps(actual, sort_keys=True) == before
    assert training.validate_feature_recipe is sealed_validator


def test_warm_science_cannot_record_cold_function_and_validator_restores_on_failure():
    from experiments.deformnext_retargeting.source_input_joint_runtime import (
        science_recipe_binding, COLD_INITIAL_FUNCTION,
    )
    def sealed_validator(_): raise AssertionError('must reject before sealed validator')
    training = SimpleNamespace(validate_feature_recipe=sealed_validator)
    with pytest.raises(ValueError, match='honest source'):
        with science_recipe_binding(training, variant='source-contrast-NoDQ', teacher_enabled=True):
            training.validate_feature_recipe(dict(experiment=dict(initial_function=COLD_INITIAL_FUNCTION,
                source_input_design='source-contrast-NoDQ', additional_source_teacher_supervision=True)))
    assert training.validate_feature_recipe is sealed_validator
