import copy

import pytest
import torch

from experiments.deformnext_retargeting.frozen_utility_strength import recipe as strength_recipe
from experiments.deformnext_retargeting.utility_softprior_followup import NAME, NAMESPACE, recipe, validate_profile
from pixal3d.modules.task_matching_attention import TaskCandidateAttention


def source(root):
    return strength_recipe({'ckpt': 'none', 'load_dir': '', 'num_gpus': 2, 'training_seed': 101,
        'default_output_dir': str(root / 'runs_ss64/frozen-utility-c1'),
        'experiment': {'arm': 'frozen-utility-c1', 'required_microbatch_per_gpu': 48},
        'trainer': {'name': 'ImageConditionedProjFlowMatchingCFGTrainer', 'args': {
            'max_steps': 480, 'batch_size_per_gpu': 48, 'batch_split': 1,
            'continuation_ckpt': {'denoiser': 'SS4210.pt'},
            'optimizer': {'name': 'AdamW', 'args': {'lr': 1e-5, 'eps': 1e-8}},
            'deform_conditioner': {'implementation': 'r9-native-v5', 'arm': 'c1', 'score_temperature': .005,
                'task_attention': {'decoder': True, 'score_prior': True, 'train_decoder': False,
                    'train_score': False, 'content_mode': 'residual-zero', 'chunk_size': 64}}}}}, root)


def test_only_model_change_is_frozen_prior_temperature(tmp_path):
    old = source(tmp_path / 'utility-strength-followup-v1'); before = copy.deepcopy(old)
    root = tmp_path / NAMESPACE; cfg = recipe(old, root)
    assert old == before
    validate_profile(cfg, root)
    new_trainer = copy.deepcopy(cfg['trainer'])
    assert new_trainer['args']['deform_conditioner']['score_temperature'] == .05
    new_trainer['args']['deform_conditioner']['score_temperature'] = .005
    assert new_trainer == old['trainer']
    assert cfg['output_dir'] == cfg['default_output_dir'] == str(root / 'runs_ss64' / NAME)
    assert cfg['config'] == cfg['experiment']['recipe_path'] == str(root / 'configs' / (NAME + '.json'))
    assert cfg['master_port'] == '30447'


@pytest.mark.parametrize('change', ['temperature', 'scale', 'voxel', 'seed', 'decoder', 'accumulation', 'trainer'])
def test_uncontrolled_changes_are_rejected(tmp_path, change):
    root = tmp_path / NAMESPACE; cfg = recipe(source(tmp_path / 'utility-strength-followup-v1'), root)
    spec = cfg['trainer']['args']['deform_conditioner']
    if change == 'temperature': spec['score_temperature'] = .5
    elif change == 'scale': spec['task_attention']['initial_content_scale'] = 4.
    elif change == 'voxel': spec['voxel_chunk_size'] = 128
    elif change == 'seed': cfg['training_seed'] = 202
    elif change == 'decoder': spec['task_attention']['train_decoder'] = True
    elif change == 'accumulation': cfg['trainer']['args']['batch_split'] = 2
    else: cfg['trainer']['name'] = 'VoxelQueryImageConditionedProjFlowMatchingCFGTrainer'
    with pytest.raises(ValueError): validate_profile(cfg, root)


def test_zero_output_utility_retains_each_prior_but_soft_prior_is_not_uniform():
    attention = TaskCandidateAttention(width=4, hidden=3, content_mode='residual-zero', initial_content_scale=16.)
    values = torch.eye(4)[None]; query = values[:, 0]; valid = torch.ones(1, 4, dtype=torch.bool)
    scores = torch.tensor([[.1, .08, .06, 0.]])
    with torch.no_grad():
        hard = attention(query, query, values, valid, scores / .005)[1]
        soft = attention(query, query, values, valid, scores / .05)[1]
    torch.testing.assert_close(hard, (scores / .005).softmax(-1))
    torch.testing.assert_close(soft, (scores / .05).softmax(-1))
    assert soft.max() < hard.max() and not torch.equal(soft, torch.full_like(soft, .25))
