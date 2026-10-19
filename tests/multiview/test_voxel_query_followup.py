"""New query recipes retain the actual study's batch/head/initialization rules."""
import copy

import pytest

from experiments.deformnext_retargeting.frozen_utility_strength import recipe as strength_recipe
from experiments.deformnext_retargeting.voxel_query_followup import (
    NAME, NAMESPACE, TRAINER, recipe, standalone_directory, validate_profile,
)


def source(root):
    cfg = {'ckpt': 'none', 'load_dir': '', 'num_gpus': 2, 'training_seed': 101,
        'models': {'denoiser': {'native': True}},
        'default_output_dir': str(root / 'runs_ss64/frozen-utility-c1'),
        'experiment': {'arm': 'frozen-utility-c1', 'required_microbatch_per_gpu': 48},
        'trainer': {'name': 'ImageConditionedProjFlowMatchingCFGTrainer', 'args': {
            'max_steps': 480, 'batch_size_per_gpu': 48, 'batch_split': 1,
            'continuation_ckpt': {'denoiser': 'originalSS4210.pt'},
            'optimizer': {'name': 'AdamW', 'args': {'lr': 1e-5, 'eps': 1e-8}},
            'deform_conditioner': {'implementation': 'r9-native-v5', 'arm': 'c1', 'score_temperature': .005,
                'task_attention': {'decoder': True, 'score_prior': True, 'train_decoder': False,
                    'train_score': False, 'content_mode': 'residual-zero', 'chunk_size': 64}}}}}
    return strength_recipe(cfg, root)


def test_recipe_changes_only_query_trainer_and_voxel_chunk_in_model_training_contract(tmp_path):
    original = source(tmp_path / 'utility-strength-followup-v1')
    before = copy.deepcopy(original)
    root = tmp_path / NAMESPACE
    cfg = recipe(original, root)
    assert original == before
    validate_profile(cfg, root)
    assert cfg['trainer']['name'] == TRAINER
    args = copy.deepcopy(cfg['trainer']['args'])
    assert args['deform_conditioner'].pop('voxel_chunk_size') == 128
    assert args == original['trainer']['args']
    assert cfg['models'] == original['models']
    assert cfg['config'] == cfg['experiment']['recipe_path'] == str(root / 'configs' / (NAME + '.json'))
    assert cfg['output_dir'] == cfg['default_output_dir'] == str(root / 'runs_ss64' / NAME)
    assert cfg['master_port'] == '30446' and cfg['wandb_id'] == 'ss-voxel-query-c1-s101'


@pytest.mark.parametrize('change', ['trainer', 'seed', 'scale', 'score', 'decoder', 'microbatch', 'chunks', 'output'])
def test_profile_rejects_uncontrolled_model_and_optimization_changes(tmp_path, change):
    root = tmp_path / NAMESPACE
    cfg = recipe(source(tmp_path / 'utility-strength-followup-v1'), root)
    args = cfg['trainer']['args']; spec = args['deform_conditioner']
    if change == 'trainer':
        cfg['trainer']['name'] = 'ImageConditionedProjFlowMatchingCFGTrainer'
    elif change == 'seed':
        cfg['training_seed'] = 202
    elif change == 'scale':
        spec['task_attention']['initial_content_scale'] = 4.
    elif change == 'score':
        spec['score_temperature'] = .05
    elif change == 'decoder':
        spec['task_attention']['train_decoder'] = True
    elif change == 'microbatch':
        args['batch_split'] = 2
    elif change == 'chunks':
        spec['voxel_chunk_size'] = 32
    else:
        cfg['output_dir'] = str(tmp_path / 'existing-run')
    with pytest.raises(ValueError):
        validate_profile(cfg, root)


def test_standalone_proof_cannot_overwrite_production_or_previous_proof(tmp_path):
    parent = tmp_path / 'verification' / NAME
    selected = standalone_directory(tmp_path)
    assert selected == parent / 'standalone-native-v1'
    for path in [parent, tmp_path / 'elsewhere']:
        with pytest.raises(ValueError): standalone_directory(tmp_path, path)
    selected.mkdir(parents=True)
    (selected / 'native-proof.json').write_text('{"synthetic":true}')
    with pytest.raises(FileExistsError): standalone_directory(tmp_path)
    assert standalone_directory(tmp_path, parent / 'standalone-native-v2') == parent / 'standalone-native-v2'
