"""Restore the unnamed native V1 Adam group and preserve global sampler offset."""
from __future__ import annotations

import copy

import torch


def convert_pbr_sampler(state, *, source_world, source_batch, target_world):
    if (set(state) != {'epoch', 'idx'} or source_world != 3 or source_batch != 8
            or target_world not in {1,2,4}
            or any(not isinstance(v,int) or isinstance(v,bool) or v < 0 for v in state.values())
            or state['idx'] % source_batch):
        raise ValueError('declared native world3/batch8 sampler state required')
    position = state['idx'] * source_world
    if position % target_world:
        raise ValueError('global sampler offset cannot be represented in target ranks')
    return dict(epoch=state['epoch'], idx=position // target_world)


def restore_v1_adam(optimizer, source, *, names, checkpoint_names, expected_step):
    """Validate every common tensor before replacing any live optimizer state."""
    current = optimizer.state_dict()
    if (current['state'] or len(source['param_groups']) != 1
            or source['param_groups'][0].get('model_name') not in {None,'default'}
            or names != checkpoint_names or not names or len(set(names)) != len(names)):
        raise ValueError('fresh optimizer and exact native NN parameter order required')
    common = [g for g in current['param_groups'] if g.get('model_name') == 'default']
    live = [g for g in optimizer.param_groups if g.get('model_name') == 'default']
    if len(common) != 1 or len(live) != 1:
        raise ValueError('exactly one common NN group required')
    old = source['param_groups'][0]
    if (len(old['params']) != len(names) or len(common[0]['params']) != len(names)
            or len(set(old['params'])) != len(names) or set(old['params']) != set(source['state'])):
        raise ValueError('every saved V1 NN state must map exactly once')
    for old_id,new_id,parameter in zip(old['params'],common[0]['params'],live[0]['params']):
        state = source['state'][old_id]
        if (set(state) != {'step','exp_avg','exp_avg_sq'} or float(state['step']) != expected_step
                or state['exp_avg'].shape != parameter.shape or state['exp_avg_sq'].shape != parameter.shape
                or not torch.isfinite(state['exp_avg']).all() or not torch.isfinite(state['exp_avg_sq']).all()):
            raise ValueError('saved NN moment shape, step or finiteness differs')
        current['state'][new_id] = copy.deepcopy(state)
    ids = common[0]['params']
    common[0].clear()
    common[0].update(copy.deepcopy(old), params=ids, model_name='default')
    optimizer.load_state_dict(current)
    return len(names)
