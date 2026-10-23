from copy import deepcopy
import hashlib
import json

import pytest

from experiments.deformnext_retargeting.module_confirmation_recipe import (
    make_confirmation, make_reference, validate_confirmation,
)
from experiments.deformnext_retargeting.module_pipeline_protocol_v2 import protocol
from test_projected_native_recipe import control as projected_control
from test_r_mediated_native_recipe import recipe
from test_r_top1_native_recipe import control as top1_control
from test_raw_correspondence_native import control as correspondence_control
from test_unified_design_controls import control as design_control


def write(path, value):
    path.write_text(json.dumps(value))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(tmp_path, arm, phase='science'):
    if arm in ('V1', 'FULL'):
        original, reference = recipe('vanilla' if arm == 'V1' else 'R_raw_topk'), None
    elif arm == 'C_DECODER':
        original, reference = top1_control()
    elif arm in ('C_TOP1_V', 'C_FIXED_TOPK'):
        original, reference = projected_control(arm)
    elif arm in ('C_RAW', 'C_SPATIAL', 'C_D1_COS'):
        original, reference = correspondence_control(arm)
    else:
        original, reference = design_control(arm)
    original.update(output_dir='/original/' + arm, wandb_id='original-' + arm)
    kwargs = {}
    if reference is not None:
        path, digest = write(tmp_path/'original-reference.json', reference)
        original['experiment'].update(reference_science_config=str(path), reference_science_config_SHA256=digest)
        new_reference = make_reference(reference, output_dir='/recipe-only/new-reference',
            run_id='ref202', recipe_path=str(tmp_path/'new-reference.json'))
        path, digest = write(tmp_path/'new-reference.json', new_reference)
        kwargs = dict(reference_path=path, reference_SHA256=digest)
    confirmed = make_confirmation(original, arm=arm, phase=phase, output_dir='/confirmation/' + arm,
        run_id=arm + '-s202-' + phase, recipe_path=str(tmp_path/(arm + '-' + phase + '.json')), **kwargs)
    return original, confirmed


@pytest.mark.parametrize('arm', protocol()['arms'])
@pytest.mark.parametrize('phase', ['proof', 'capacity', 'science'])
def test_all_finite_physical_families_support_independent_training_without_new_designs(tmp_path, arm, phase):
    original, confirmed = inputs(tmp_path, arm, phase)
    report = validate_confirmation(original, confirmed, arm=arm, phase=phase)
    assert confirmed['training_seed'] == 202
    assert confirmed['trainer'] == original['trainer']
    assert confirmed['dataset'] == original['dataset']
    assert confirmed['validation'] == original['validation']
    assert confirmed['use_wandb'] is (phase == 'science')
    assert not report['candidate_selected'] and not report['new_training_queued']
    assert report['actual_GPU_parity_capacity_still_required']


@pytest.mark.parametrize('damage', ['FFN', 'LR', 'initial-checkpoint', 'data', 'matcher',
    'float-type', 'new-config-field', 'reference-bytes', 'reference-model', 'same-seed', 'float-seed'])
def test_independent_seed_cannot_hide_a_changed_numerical_or_reference_contract(tmp_path, damage):
    original, confirmed = inputs(tmp_path, 'C_FIXED_TOPK')
    t = confirmed['trainer']['args']
    if damage == 'FFN':
        t['ffn_chunk_assets'] = 12
    elif damage == 'LR':
        t['optimizer']['args']['lr'] = 1e-4
    elif damage == 'initial-checkpoint':
        t['continuation_ckpt']['denoiser'] = 'warm-start-from-trained480'
    elif damage == 'data':
        confirmed['dataset']['args']['expected_split'] = 'test'
    elif damage == 'matcher':
        t['deform_conditioner']['freeze_q2c'] = False
    elif damage == 'float-type':
        t['batch_size_per_gpu'] = 96.0
    elif damage == 'new-config-field':
        confirmed['test_quality_already_read'] = False
    elif damage in ('reference-bytes', 'reference-model'):
        path = tmp_path/'new-reference.json'
        ref = json.loads(path.read_text())
        ref['trainer']['args']['deform_conditioner']['q2c_checkpoint'] = 'other-matcher'
        _, digest = write(path, ref)
        if damage == 'reference-model':
            # Updating the SHA is insufficient: both the control and its
            # reference must preserve the original scientific initialization.
            confirmed['experiment']['reference_science_config_SHA256'] = digest
            t['deform_conditioner']['q2c_checkpoint'] = 'other-matcher'
    else:
        confirmed['training_seed'] = 101 if damage == 'same-seed' else 202.0
    with pytest.raises(ValueError):
        validate_confirmation(original, confirmed, arm='C_FIXED_TOPK', phase='science')


def test_original_artifacts_are_not_mutated_and_cannot_be_overwritten(tmp_path):
    original, confirmed = inputs(tmp_path, 'FULL')
    before = deepcopy(original)
    with pytest.raises(ValueError, match='new absolute'):
        make_confirmation(original, arm='FULL', phase='science', output_dir=original['output_dir'],
            run_id='new', recipe_path=str(tmp_path/'new.json'))
    assert original == before and confirmed['trainer'] == before['trainer']


def test_recipe_only_reference_is_not_a_confirmation_or_an_extra_learning_arm(tmp_path):
    ref = make_reference(recipe('R_feature'), output_dir='/reference-only', run_id='reference',
                         recipe_path=str(tmp_path/'reference.json'))
    assert ref['experiment']['native_variant'] == 'R_feature'
    assert ref['training_seed'] == 202
    with pytest.raises(ValueError, match='twelve'):
        make_confirmation(ref, arm='R_feature', phase='science', output_dir='/no-extra-arm',
                          run_id='not-a-core-arm', recipe_path=str(tmp_path/'wrong.json'))
