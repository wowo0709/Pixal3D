import copy
from pathlib import Path

import pytest

from experiments.deformnext_retargeting.sampler_sensitivity import evaluation_config, validate_pair


def pair():
    rows = [{'sample_id': str(i), 'source_cluster_id': str(i), 'seed': i+100,
             'tp': 4, 'fp': 2, 'fn': 1, 'target_voxels_sha256': f'{i:064x}',
             'target_voxel_shape': [64,64,64]} for i in range(32)]
    left = dict(status='PASS', stage='ss64', step=480, weights='raw', sampling_steps=12,
        guidance=3., generation_batch_size=8, optimizer_updates=0, optimizer_updates_during_evaluation=0,
        conditioning_mode='edited-rgb', target_modality='rgb', transport_control='normal',
        r9_path_control='normal', input_view_count=4, input_states=['M1','M0','M0','M0'],
        input_camera_ids=[8,9,10,11], checkpoint='same.pt', conditioner_checkpoint='same-conditioner.pt',
        manifest_sha256='same', panel_file='same-panel.json', seed_map_file='same-seeds.json',
        quick_panel_ids=[r['sample_id'] for r in rows], per_asset=rows)
    right = copy.deepcopy(left);right['sampling_steps'] = 50
    return left, right


def test_intentional_step_difference_retains_paired_gt_noise_checks():
    left,right = pair()
    for row in right['per_asset']: row.update(tp=5, fn=0)
    result = validate_pair(left,right)
    assert result['baseline_iou'] == 4/7
    assert result['candidate_iou'] == 5/7
    assert result['paired_target_hashes_verified'] == 32
    assert result['selection_use'] is False


@pytest.mark.parametrize('key,value', [
    ('checkpoint','other.pt'), ('guidance',7.5), ('optimizer_updates_during_evaluation',1),
    ('sampling_steps',24), ('input_camera_ids',[8,9,9,11]),
    ('conditioning_mode','source-only'), ('diagnostic_only',True),
])
def test_rejects_unmatched_weights_or_generation_contract(key,value):
    left,right = pair();right[key] = value
    with pytest.raises(ValueError): validate_pair(left,right)


@pytest.mark.parametrize('key,value', [('seed',999), ('target_voxels_sha256','different'), ('target_voxels_sha256',None)])
def test_rejects_changed_noise_or_gt(key,value):
    left,right = pair();right['per_asset'][0][key] = value
    with pytest.raises(ValueError): validate_pair(left,right)


def test_recipe_changes_only_bounded_sampling_diagnostic():
    source = {'models': {'frozen': True}, 'trainer': {'args': {'lr': 1e-5}},
              'dataset': {'manifest': 'unchanged'}, 'validation': {'dataset_args': {'camera': 'unchanged'},
              'quick_metrics': {'sampling_steps': 12, 'other': 'retained'}}}
    before = copy.deepcopy(source);result = evaluation_config(source,Path('/assigned'),50)
    assert source == before
    assert all(result[k] == source[k] for k in ['models','trainer','dataset'])
    assert result['validation']['dataset_args'] == source['validation']['dataset_args']
    assert result['validation']['quick_metrics']['sampling_steps'] == 50
    assert result['validation']['quick_metrics']['other'] == 'retained'
