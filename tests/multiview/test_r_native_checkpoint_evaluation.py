import copy
import json

import pytest

from experiments.deformnext_retargeting.evaluate_r_native_checkpoint import (
    BUNDLE_SCHEMA, check_epoch_replay, validate_bundle, validate_panel,
)
from test_r_mediated_native_recipe import recipe


def bundle_fixture(variant='R_raw_topk', step=96):
    cfg = recipe(variant)
    cfg['output_dir'] = '/campaign/' + variant + '/native'
    root = cfg['output_dir']
    bundle = dict(schema=BUNDLE_SCHEMA, variant=variant, step=step,
        checkpoint=f'{root}/ckpts/denoiser_step{step:07d}.pt',
        conditioner_checkpoint=None if variant == 'vanilla' else
            f'{root}/ckpts/deform_conditioner_step{step:07d}.pt',
        native_config=root + '/config.json', source_teacher_updates=0,
        native_task_objective='SS_flow_mse', native_validation=dict(step=step))
    return cfg, bundle


@pytest.mark.parametrize('variant', ['vanilla', 'R_feature', 'R_raw_topk'])
def test_own_actual_pair_is_valid(variant):
    cfg, bundle = bundle_fixture(variant)
    assert validate_bundle(bundle, cfg) == variant


@pytest.mark.parametrize('damage', ['variant', 'denoiser', 'source', 'epoch', 'teacher', 'config'])
def test_cross_variant_epoch_teacher_and_stale_original_weights_are_rejected(damage):
    cfg, bundle = bundle_fixture()
    if damage == 'variant': bundle['variant'] = 'R_feature'
    elif damage == 'denoiser': bundle['checkpoint'] = 'SS4210.pt'
    elif damage == 'source': bundle['conditioner_checkpoint'] = '/campaign/R_feature/native/ckpts/deform_conditioner_step0000096.pt'
    elif damage == 'epoch': bundle['native_validation']['step'] = 480
    elif damage == 'teacher': bundle['source_teacher_updates'] = 112
    else: bundle['native_config'] = '/legacy/config.json'
    with pytest.raises(ValueError, match='checkpoint pair'):
        validate_bundle(bundle, cfg)


def test_vanilla_cannot_silently_acquire_a_source_checkpoint():
    cfg, bundle = bundle_fixture('vanilla')
    bundle['conditioner_checkpoint'] = 'unexpected-source.pt'
    with pytest.raises(ValueError): validate_bundle(bundle, cfg)


def panel_fixture(tmp_path):
    cfg, bundle = bundle_fixture()
    ids = [f'asset{i}' for i in range(32)]
    panel, seeds = tmp_path / 'panel.json', tmp_path / 'seeds.json'
    panel.write_text(json.dumps(dict(sample_ids=ids)))
    seeds.write_text(json.dumps(dict(zip(ids, range(32)))))
    cfg['validation']['quick_metrics'].update(panel_file=str(panel), seed_map_file=str(seeds))
    plan = dict(panel=str(panel), seeds=str(seeds), expected_assets=32,
        generation_batch_size=8, deterministic_algorithms=False, mode='epoch-replay')
    return cfg, bundle, plan


def test_epoch_replay_must_keep_the_actual_training_validation_protocol(tmp_path):
    cfg, bundle, plan = panel_fixture(tmp_path)
    assert len(validate_panel(plan, bundle, cfg)[0]) == 32
    plan['deterministic_algorithms'] = True
    with pytest.raises(ValueError, match='epoch replay'): validate_panel(plan, bundle, cfg)
    plan['deterministic_algorithms'] = False
    other = tmp_path / 'other.json'
    other.write_text(json.dumps(dict(sample_ids=[f'other{i}' for i in range(32)])))
    plan['panel'] = str(other)
    with pytest.raises(ValueError): validate_panel(plan, bundle, cfg)


def test_same_means_are_insufficient_for_actual_saved_pair_replay():
    reference = dict(step=96, weights='raw', quick_panel_ids=['a'],
        quick_metrics=dict(voxel_iou=.4), quick_per_asset=[dict(sample_id='a', seed=3, tp=4)])
    actual = {**reference, 'per_asset': copy.deepcopy(reference['quick_per_asset'])}
    check_epoch_replay(actual, reference)
    actual['per_asset'][0]['tp'] = 5
    with pytest.raises(ValueError, match='exactly reproduce'): check_epoch_replay(actual, reference)


def test_final_protocol_rejects_early_epoch_and_missing_integer_seeds(tmp_path):
    cfg, bundle, plan = panel_fixture(tmp_path)
    plan.update(mode='archive32', deterministic_algorithms=True)
    with pytest.raises(ValueError, match='final480'): validate_panel(plan, bundle, cfg)
    bundle['step'] = 480
    validate_panel(plan, bundle, cfg)
    seeds = json.loads((tmp_path / 'seeds.json').read_text()); seeds['asset0'] = '0'
    (tmp_path / 'seeds.json').write_text(json.dumps(seeds))
    with pytest.raises(ValueError, match='integer seeds'): validate_panel(plan, bundle, cfg)
