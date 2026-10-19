from itertools import islice

import pytest
import torch

from experiments.deformnext_retargeting.pbr_continuation_state import convert_pbr_sampler, restore_v1_adam
from pixal3d.utils import data_utils


@pytest.mark.parametrize('world', [2,4])
def test_balanced_native_sampler_preserves_next_global24_after_world_change(monkeypatch,world):
    class Dataset:
        loads = [(i%17)*3+1 for i in range(113)]
        def __len__(self): return len(self.loads)
    dataset = Dataset()

    def global_batch(size,state):
        batch=[]
        for rank in range(size):
            monkeypatch.setattr(data_utils.dist,'is_initialized',lambda:True)
            monkeypatch.setattr(data_utils.dist,'get_world_size',lambda:size)
            monkeypatch.setattr(data_utils.dist,'get_rank',lambda:rank)
            sampler=data_utils.BalancedResumableSampler(dataset,batch_size=24//size)
            sampler.load_state_dict(state)
            batch.extend(islice(iter(sampler),24//size))
        return sorted(batch)

    source=dict(epoch=2,idx=16)
    assert global_batch(3,source)==global_batch(world,convert_pbr_sampler(
        source,source_world=3,source_batch=8,target_world=world))


@pytest.mark.parametrize('world', [2,4])
def test_original_unbalanced_sampler_preserves_next_global24(monkeypatch,world):
    # The actual step762 source config explicitly sets balanced_sampler=False.
    dataset = list(range(9139))

    def global_batch(size,state):
        batch=[]
        for rank in range(size):
            monkeypatch.setattr(data_utils.dist,'is_initialized',lambda:True)
            monkeypatch.setattr(data_utils.dist,'get_world_size',lambda:size)
            monkeypatch.setattr(data_utils.dist,'get_rank',lambda:rank)
            sampler=data_utils.ResumableSampler(dataset,shuffle=True)
            sampler.load_state_dict(state)
            batch.extend(islice(iter(sampler),24//size))
        return sorted(batch)

    source=dict(epoch=2,idx=16)
    assert global_batch(3,source)==global_batch(world,convert_pbr_sampler(
        source,source_world=3,source_batch=8,target_world=world))


def test_unnamed_V1_moments_restore_without_touching_new_parameters():
    source_params=[torch.nn.Parameter(torch.randn(2,3)),torch.nn.Parameter(torch.randn(2,3))]
    source=torch.optim.AdamW(source_params,lr=5e-6)
    sum(p.square().sum() for p in source_params).backward();source.step()
    saved=source.state_dict()
    for state in saved['state'].values():state['step'].fill_(762)
    current_params=[torch.nn.Parameter(p.detach().clone()) for p in source_params]
    fresh=torch.nn.Parameter(torch.ones(2,3))
    current=torch.optim.AdamW([dict(params=current_params,model_name='default'),
                              dict(params=[fresh],model_name='new')],lr=5e-6)
    names=['block0.lora_A','block1.lora_A']
    with pytest.raises(ValueError):
        restore_v1_adam(current,saved,names=names,checkpoint_names=names[::-1],expected_step=762)
    assert not current.state
    assert restore_v1_adam(current,saved,names=names,checkpoint_names=names,expected_step=762)==2
    assert fresh not in current.state
    for p,old in zip(current_params,saved['state'].values()):
        assert torch.equal(current.state[p]['exp_avg'],old['exp_avg'])
        assert torch.equal(current.state[p]['exp_avg_sq'],old['exp_avg_sq'])
        assert float(current.state[p]['step'])==762
