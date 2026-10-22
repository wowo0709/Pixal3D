import copy
import hashlib
import json

import pytest

from experiments.deformnext_retargeting.r_top1_native_recipe import NAME, SCHEMA, VARIANT, validate_config
from test_r_mediated_native_recipe import recipe


def control(phase='science'):
    reference = recipe('R_feature')
    cfg = copy.deepcopy(reference)
    cfg['trainer']['name'] = NAME
    cfg['trainer']['args'].pop('ray_feature_utility')
    cfg['use_wandb'] = phase == 'science'
    cfg['experiment'].update(native_variant=VARIANT, conditioning_schema=SCHEMA,
        appearance_mode='single-semantic-Top1-feature', actual_source_tensors=10,
        learned_attention_head_tensors=0, physical_deformation_query_channels=0,
        engineering_only=phase != 'science')
    return cfg, reference


@pytest.mark.parametrize('phase', ['proof', 'capacity', 'science'])
def test_physical_top1_control_keeps_the_full_native_training_contract(phase):
    cfg, reference = control(phase)
    assert validate_config(cfg, phase, reference) == VARIANT


@pytest.mark.parametrize('damage', ['uniform', 'head', 'utility', 'batch', 'lr', 'manifest', 'source-lr', 'teacher', 'DQ', 'matching'])
def test_no_head_or_changed_training_and_data_contract_can_hide_in_the_top1_control(damage):
    cfg, reference = control()
    t, e = cfg['trainer']['args'], cfg['experiment']
    if damage == 'uniform': e['appearance_mode'] = 'uniform'
    elif damage == 'head': e['learned_attention_head_tensors'] = 9
    elif damage == 'utility': t['ray_feature_utility'] = reference['trainer']['args']['ray_feature_utility']
    elif damage == 'batch': t['batch_size_per_gpu'] = 48
    elif damage == 'lr': t['optimizer']['args']['lr'] = 1e-4
    elif damage == 'manifest': cfg['data_dir'] = 'different assets'
    elif damage == 'source-lr': t['model_learning_rates']['deform_conditioner'] = 1e-3
    elif damage == 'teacher': e['source_teacher_updates'] = 112
    elif damage == 'DQ': e['physical_deformation_query_channels'] = 2048
    else: t['deform_conditioner']['score_temperature'] = .5
    with pytest.raises(ValueError): validate_config(cfg, 'science', reference)


def test_top1_checkpoint_pair_and_process_bindings_are_own_and_restore_on_failure(tmp_path):
    from experiments.deformnext_retargeting import evaluate_r_native_checkpoint as common
    from experiments.deformnext_retargeting import r_mediated_native_runtime as runtime
    from experiments.deformnext_retargeting.evaluate_r_top1_native_checkpoint import validate_bundle, top1_evaluation_binding
    cfg, reference = control()
    path = tmp_path / 'reference.json'
    path.write_text(json.dumps(reference))
    cfg['experiment'].update(reference_science_config=str(path),
        reference_science_config_SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
    cfg['output_dir'] = '/actual/Top1/native'
    bundle = dict(schema='cold-native-task-Top1-SS-checkpoint-pair-v1', variant=VARIANT, step=96,
        checkpoint='/actual/Top1/native/ckpts/denoiser_step0000096.pt',
        conditioner_checkpoint='/actual/Top1/native/ckpts/deform_conditioner_step0000096.pt',
        native_config='/actual/Top1/native/config.json', source_teacher_updates=0,
        native_task_objective='SS_flow_mse', native_validation=dict(step=96))
    assert validate_bundle(bundle, cfg) == VARIANT
    previous = common.validate_bundle, common.cold_source_binding, runtime.selected_trainer, runtime.parameter_groups
    with pytest.raises(RuntimeError, match='owned'):
        with top1_evaluation_binding():
            assert common.validate_bundle is validate_bundle
            raise RuntimeError('owned read-only failure')
    assert previous == (common.validate_bundle, common.cold_source_binding, runtime.selected_trainer, runtime.parameter_groups)
    bundle['conditioner_checkpoint'] = '/actual/R_feature/native/ckpts/deform_conditioner_step0000096.pt'
    with pytest.raises(ValueError): validate_bundle(bundle, cfg)
