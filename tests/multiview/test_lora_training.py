import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from experiments.multiview_finetune.configure import stage_config
from pixal3d.trainers.basic import BasicTrainer
from test_lora import TinyFlow


class Samples(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {'x': torch.full((4,), (index + 1) / 8),
                'target': torch.zeros(4), 'cond': torch.zeros(1, 3, 1, 1)}


class TinyTrainer(BasicTrainer):
    def training_losses(self, x, target, **kwargs):
        pred = self.training_models['denoiser'](x)
        return {'loss': (pred.float() - target).square().mean()}, {}


def make_trainer(path, base_path, *, resume=None):
    return TinyTrainer({'denoiser': TinyFlow()}, Samples(),
        output_dir=str(path), load_dir=str(path) if resume is not None else None, step=resume,
        batch_size_per_gpu=4, batch_split=4, max_steps=3,
        optimizer={'name': 'AdamW', 'args': {'lr': .01}},
        lr_scheduler={'name': 'StepLR', 'args': {'step_size': 1, 'gamma': .9}},
        grad_clip=None, ema_rate=[], num_workers=0,
        mix_precision_mode=None, mix_precision_dtype='bfloat16',
        lora={'rank': 2, 'alpha': 2, 'dropout': 0.0}, microbatch_to_device=True,
        finetune_ckpt={'denoiser': str(base_path)})


def test_load_data_retains_cpu_packs_until_forward():
    trainer = object.__new__(BasicTrainer)
    trainer.models = {'denoiser': nn.Linear(1, 1, device='meta')}
    trainer.microbatch_to_device = True
    trainer.prefetch_data = False
    trainer.batch_split = 2
    trainer.data_iterator = iter([{'x': torch.ones(2, 1), 'cond': torch.zeros(2, 1, 3, 1, 1)}])
    packs = trainer.load_data()
    assert len(packs) == 2
    assert all(pack['x'].device.type == 'cpu' for pack in packs)


def test_sparse_transfer_does_not_store_device_caches_back_on_cpu_pack():
    from pixal3d.modules.sparse import SparseTensor
    from pixal3d.utils.data_utils import recursive_to_device
    source = SparseTensor(feats=torch.zeros(1, 2), coords=torch.zeros(1, 4, dtype=torch.int32))
    source._shape = torch.Size([1, 2])
    source.register_spatial_cache('layout', [slice(0, 1)])
    source.register_spatial_cache('cum_seqlen', torch.tensor([0, 1], dtype=torch.int32))
    moved = recursive_to_device(source, torch.device('meta'))
    assert moved.get_spatial_cache('cum_seqlen').device.type == 'meta'
    moved.register_spatial_cache('device_only', torch.ones(1, device='meta'))
    assert source.get_spatial_cache('device_only') is None
    assert source.get_spatial_cache('cum_seqlen').device.type == 'cpu'


def test_official_initialization_and_full_checkpoint_resume_preserve_frozen_base(tmp_path):
    base = TinyFlow()
    base_path = tmp_path / 'official.pt'
    torch.save(base.state_dict(), base_path)
    trainer = make_trainer(tmp_path / 'run', base_path)
    assert len(trainer.model_params) == 12
    assert all(p.dtype == torch.float32 for p in trainer.model_params)
    for name, p in trainer.models['denoiser'].named_parameters():
        if not name.endswith(('lora_A', 'lora_B')):
            torch.testing.assert_close(p, base.state_dict()[name].bfloat16(), rtol=0, atol=0)
    for _ in range(2):
        trainer.run_step(trainer.load_data())
        trainer.step += 1
    trainer.save()
    saved = {k: v.clone() for k, v in trainer.models['denoiser'].state_dict().items()}
    resumed = make_trainer(tmp_path / 'run', base_path, resume=2)
    assert resumed.step == 2
    assert resumed.lr_scheduler.state_dict() == trainer.lr_scheduler.state_dict()
    assert resumed.data_sampler.state_dict() == trainer.data_sampler.state_dict()
    assert all(float(state['step']) == 2 for state in resumed.optimizer.state.values())
    for key, value in resumed.models['denoiser'].state_dict().items():
        torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
    resumed.run_step(resumed.load_data())
    for name, p in resumed.models['denoiser'].named_parameters():
        if not name.endswith(('lora_A', 'lora_B')):
            torch.testing.assert_close(p, base.state_dict()[name].bfloat16(), rtol=0, atol=0)


def test_lora_recipe_resolves_microbatch_one_without_changing_full_finetune():
    for stage, ebs in [('ss64', 96), ('shape512', 96), ('shape1024', 24), ('pbr1024', 24)]:
        config = stage_config(stage, 21314, method='lora')
        args = config['trainer']['args']
        assert args['batch_size_per_gpu'] == args['batch_split'] == ebs
        assert args['microbatch_to_device'] is True
        assert args['i_save']==500
        assert config['validation']['every_steps']==21314//ebs
        assert args['elastic'] is None and args['ema_rate'] == []
        assert args['optimizer']['args']['lr'] == 1e-4
        assert config['models']['denoiser']['args']['use_checkpoint'] is True
        assert args['lora'] == {'rank': 16, 'alpha': 16, 'dropout': 0.0}
        full = stage_config(stage, 21314)
        assert full['trainer']['args']['ema_rate'] == [0.9999]
        assert 'lora' not in full['trainer']['args']


def test_microbatch_transfer_accumulates_mean_gradient_and_steps_scheduler_once():
    trainer = object.__new__(BasicTrainer)
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.)
    trainer.models = trainer.training_models = {'denoiser': model}
    trainer.model_params = trainer.master_params = list(model.parameters())
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1, gamma=.5)
    trainer.lr_scheduler_config = {'name': 'StepLR'}
    trainer.world_size = 1
    trainer.microbatch_to_device = True
    trainer.mix_precision_mode = None
    trainer.mix_precision_dtype = torch.float32
    trainer.elastic_controller_config = trainer.grad_clip = None
    trainer.log_param_stats = trainer.is_master = trainer.debug = False
    trainer.training_losses = lambda x, **_: ({'loss': model(x).square().mean()}, {})
    packs = [{'x': torch.tensor([[v]]), 'cond': torch.zeros(1, 1, 3, 1, 1)} for v in [1., 2.]]
    trainer.run_step(packs)
    torch.testing.assert_close(model.weight, torch.tensor([[.5]]))
    assert trainer.optimizer.param_groups[0]['lr'] == .05
