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


def make_continuation_trainer(path, checkpoint):
    return TinyTrainer({'denoiser': TinyFlow()}, Samples(),
        output_dir=str(path), load_dir=None, step=None,
        batch_size_per_gpu=4, batch_split=4, max_steps=3,
        optimizer={'name': 'AdamW', 'args': {'lr': .01}},
        lr_scheduler={'name': 'StepLR', 'args': {'step_size': 1, 'gamma': .9}},
        grad_clip=None, ema_rate=[], num_workers=0,
        mix_precision_mode=None, mix_precision_dtype='bfloat16',
        lora={'rank': 2, 'alpha': 2, 'dropout': 0.0}, microbatch_to_device=True,
        continuation_ckpt={'denoiser': str(checkpoint)})


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


def test_lora_continuation_strict_loads_combined_weights_with_fresh_state(tmp_path):
    base_path = tmp_path / 'official.pt'
    torch.save(TinyFlow().state_dict(), base_path)
    first = make_trainer(tmp_path / 'first', base_path)
    first.run_step(first.load_data())
    expected = {key: value.detach().clone() for key, value in first.models['denoiser'].state_dict().items()}
    checkpoint = tmp_path / 'combined.pt'
    torch.save(expected, checkpoint)

    continued = make_continuation_trainer(tmp_path / 'continued', checkpoint)
    assert continued.step == 0
    assert continued.optimizer.state == {}
    assert continued.data_sampler.state_dict()['idx'] == 0
    for key, value in continued.models['denoiser'].state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_lora_continuation_rejects_incomplete_adapter_state(tmp_path):
    checkpoint = tmp_path / 'base-only.pt'
    torch.save(TinyFlow().state_dict(), checkpoint)
    with pytest.raises(RuntimeError, match='exact base\\+LoRA state dict'):
        make_continuation_trainer(tmp_path / 'bad', checkpoint)


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


def test_inference_reconstructs_saved_adapter_and_dtype(tmp_path, monkeypatch):
    from pixal3d import models
    from importlib.util import find_spec
    assert find_spec('experiments.multiview_finetune.model_loading') is not None
    from experiments.multiview_finetune.model_loading import load_run_denoiser
    base_path = tmp_path / 'official.pt'
    torch.save(TinyFlow().state_dict(), base_path)
    trainer = make_trainer(tmp_path / 'run', base_path)
    trainer.run_step(trainer.load_data())
    trainer.step = 1
    trainer.save()
    monkeypatch.setattr(models, 'TinyFlow', TinyFlow, raising=False)
    config = {'models': {'denoiser': {'name': 'TinyFlow', 'args': {}}},
              'trainer': {'args': {'lora': {'rank': 2, 'alpha': 2, 'dropout': 0.0}}}}
    loaded = load_run_denoiser(config, tmp_path / 'run/ckpts/denoiser_step0000001.pt')
    x = torch.ones(2, 4)
    torch.testing.assert_close(loaded(x), trainer.models['denoiser'](x), rtol=0, atol=0)
    assert loaded.dtype == torch.bfloat16 and not loaded.training


def test_full_finetune_inference_retains_native_half_precision_torso(tmp_path, monkeypatch):
    from pixal3d import models
    from experiments.multiview_finetune.model_loading import load_run_denoiser
    monkeypatch.setattr(models, 'TinyFlow', TinyFlow, raising=False)
    base = TinyFlow()
    checkpoint = tmp_path / 'full.pt'
    torch.save(base.state_dict(), checkpoint)
    config = {'models': {'denoiser': {'name': 'TinyFlow', 'args': {}}}, 'trainer': {'args': {}}}
    loaded = load_run_denoiser(config, checkpoint)
    assert loaded.dtype == torch.float16


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


def test_model_specific_learning_rate_builds_distinct_optimizer_group(tmp_path):
    trainer = TinyTrainer(
        {"denoiser": nn.Linear(4, 4), "deform_conditioner": nn.Linear(4, 4)},
        Samples(), output_dir=str(tmp_path), load_dir=None, step=None,
        batch_size_per_gpu=2, max_steps=1,
        optimizer={"name": "AdamW", "args": {"lr": 1e-5}},
        model_learning_rates={"deform_conditioner": 1e-4},
        ema_rate=[], num_workers=0, mix_precision_mode=None,
    )
    groups = {group["model_name"]: group["lr"] for group in trainer.optimizer.param_groups}
    assert groups == {"default": 1e-5, "deform_conditioner": 1e-4}


@pytest.mark.parametrize('mutate_base', [False, True])
def test_checkpoint_verifier_checks_adapters_against_the_cast_frozen_base(tmp_path, mutate_base):
    root = tmp_path / 'ss64'
    (root / 'ckpts').mkdir(parents=True)
    official = tmp_path / 'official.pt'
    torch.save({'weight': torch.tensor([1.2345])}, official)
    raw = {'weight': torch.tensor([1.5 if mutate_base else 1.2345]).bfloat16(),
           'block.lora_A': torch.ones(2, 1), 'block.lora_B': torch.ones(1, 2)}
    torch.save(raw, root / 'ckpts/denoiser_step0000002.pt')
    config = {'trainer': {'args': {'ema_rate': [], 'finetune_ckpt': {'denoiser': str(official)},
              'lora': {'rank': 2, 'alpha': 2, 'dropout': 0.0}, 'batch_size_per_gpu': 24,
              'optimizer': {'args': {'lr': .1}}, 'lr_scheduler': {'args': {'warmup_steps': 10}}}},
              'experiment': {'updates_per_epoch': 100}}
    (root / 'config.json').write_text(json.dumps(config))
    misc = {'step': 2, 'lr_scheduler': {'last_epoch': 2}, 'data_sampler': {'epoch': 0, 'idx': 48},
            'optimizer': {'param_groups': [{'lr': .03}], 'state': {0: {'exp_avg': torch.ones(1)}}}}
    torch.save(misc, root / 'ckpts/misc_step0000002.pt')
    result = subprocess.run([sys.executable, 'experiments/multiview_finetune/verify_checkpoints.py',
                             '--result', str(tmp_path), '--stage', 'ss64', '--step', '2'],
                            text=True, capture_output=True)
    if mutate_base:
        assert result.returncode != 0, 'Verifier accepted a changed frozen base'
    else:
        assert result.returncode == 0, result.stderr
        report = json.loads((root / 'checkpoint_integrity.json').read_text())
        assert report['lora']['frozen_base_unchanged'] is True
        assert report['lora']['adapter_tensors'] == 2
