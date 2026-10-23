"""Config-only continuation tests; no neural forwards or training."""
from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.pbr_confirmation_recipe import make_confirmation, validate_confirmation
from test_r_mediated_pbr_science import recipe


def inputs(mode='top1-only', world=4):
    original, proof, multiview = recipe(mode, world=world)
    confirmed = make_confirmation(original, proof=proof, original_multiview=multiview,
        output_dir='/new-seed202/native', run_id='pbr-'+mode+'-s202',
        recipe_path='/new-seed202/config.json', master_port=47023)
    return original, confirmed, proof, multiview


@pytest.mark.parametrize('mode', ['vanilla', 'top1-only', 'top1-plus-feature', 'raw-topk-plus-feature'])
@pytest.mark.parametrize('world', [2, 4])
def test_independent_seed_retains_actual_native_initialization_batch_optimizer_data_and_validation(mode, world):
    original, confirmed, proof, multiview = inputs(mode, world)
    report = validate_confirmation(original, confirmed, proof=proof, original_multiview=multiview)
    assert confirmed['trainer'] == original['trainer']
    assert confirmed['models'] == original['models']
    assert confirmed['dataset'] == original['dataset']
    assert confirmed['validation'] == original['validation']
    assert report['microbatch']*world == 24
    assert report['seed101_engineering_architecture_capacity_reference_reused']
    assert report['actual_seed202_engineering_proof_or_capacity_not_claimed']
    assert not report['new_training_queued'] and not report['model_selected']


@pytest.mark.parametrize('damage', ['world', 'accumulation', 'LR', 'initializer', 'source-mode',
    'source-LR', 'max-tokens', 'dataset', 'validation', 'float-batch', 'float-seed',
    'same-seed', 'same-output', 'same-wandb', 'resume', 'teacher', 'invalid-port', 'relative-recipe'])
def test_a_second_seed_cannot_hide_an_architecture_or_numerical_protocol_change(damage):
    original, c, proof, multiview = inputs()
    if damage == 'world':
        c['num_gpus'] = 2
    elif damage == 'accumulation':
        c['trainer']['args']['batch_split'] = 2
    elif damage == 'LR':
        c['trainer']['args']['optimizer']['args']['lr'] = 1e-5
    elif damage == 'initializer':
        c['trainer']['args']['continuation_ckpt']['denoiser'] = '/trained-seed101/1900.pt'
    elif damage == 'source-mode':
        c['trainer']['args']['deform_conditioner']['appearance_mode'] = 'top1-plus-feature'
    elif damage == 'source-LR':
        c['trainer']['args']['model_learning_rates']['deform_conditioner'] = .001
    elif damage == 'max-tokens':
        c['dataset']['args']['max_tokens'] = 16000
    elif damage == 'dataset':
        c['data_dir'] = '/different-subset'
    elif damage == 'validation':
        c['validation']['quick_metrics']['seed_map_file'] = '/different-noises'
    elif damage == 'float-batch':
        c['trainer']['args']['batch_size_per_gpu'] = 6.0
    elif damage == 'float-seed':
        c['training_seed'] = 202.0
    elif damage == 'same-seed':
        c['training_seed'] = 101
    elif damage == 'same-output':
        c['output_dir'] = original['output_dir']
    elif damage == 'same-wandb':
        c['wandb_id'] = c['wandb_name'] = original['wandb_id']
    elif damage == 'resume':
        c['load_dir'] = original['output_dir']
    elif damage == 'teacher':
        c['experiment']['source_teacher_updates'] = 112
    elif damage == 'invalid-port':
        c['master_port'] = 'no-port'
    elif damage == 'relative-recipe':
        c['config'] = 'unknown-relative-config.json'
    with pytest.raises(ValueError):
        validate_confirmation(original, c, proof=proof, original_multiview=multiview)


def test_factory_leaves_the_actual_proof_and_scientific_run_immutable():
    original, _, proof, multiview = inputs()
    before = deepcopy((original, proof, multiview))
    make_confirmation(original, proof=proof, original_multiview=multiview,
        output_dir='/another-seed202', run_id='another', recipe_path='/another/config.json', master_port=47321)
    assert (original, proof, multiview) == before


@pytest.mark.parametrize('port', [True, 47000.0, 0, 65536])
def test_port_and_run_metadata_must_be_explicit_without_changing_GPU_assignment(port):
    original, _, proof, multiview = inputs()
    with pytest.raises(ValueError):
        make_confirmation(original, proof=proof, original_multiview=multiview,
            output_dir='/new', run_id='new', recipe_path='/new/config.json', master_port=port)
