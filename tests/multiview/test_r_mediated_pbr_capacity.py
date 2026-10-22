from contextlib import nullcontext
from copy import deepcopy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.deformnext_retargeting import probe_r_mediated_pbr_capacity as capacity
from test_r_mediated_pbr_native_probe import fixture


def plan(cfg, world):
    return dict(mode=cfg['experiment']['PBR_source_mode'], world_size=world,
        microbatch=24//world, global_batch=24, accumulation=1, max_tokens=31900,
        temporary_updates=4, validation=dict(dataset_name=cfg['dataset']['name'],
            dataset_args=dict(cfg['dataset']['args'], expected_split='validation'),
            quick_metrics=dict(panel_size=4, sampling_steps=12, render_resolution=256,
                panel_file='declared-panel4.json', seed_map_file='declared-seeds.json')))


@pytest.mark.parametrize('mode', ['vanilla', 'top1-only', 'top1-plus-feature', 'raw-topk-plus-feature'])
@pytest.mark.parametrize('world', [2, 4])
def test_capacity_keeps_global24_native_dropout_and_original_fresh_lineage(mode, world):
    cfg, original = fixture(mode)
    spec = plan(cfg, world)
    capacity.validate_capacity(spec, cfg, original)
    for key, value in [('microbatch', 1), ('accumulation', 2), ('global_batch', 12),
                       ('max_tokens', 2000), ('temporary_updates', 3), ('world_size', 3)]:
        bad = deepcopy(spec); bad[key] = value
        with pytest.raises(ValueError): capacity.validate_capacity(bad, cfg, original)
    bad = deepcopy(cfg); bad['trainer']['args']['p_uncond'] = 0.
    with pytest.raises(ValueError): capacity.validate_capacity(spec, bad, original)
    bad = deepcopy(spec); bad['validation']['dataset_args']['expected_split'] = 'train'
    with pytest.raises(ValueError): capacity.validate_capacity(bad, cfg, original)
    bad = deepcopy(spec); bad['validation']['quick_metrics']['panel_file'] = None
    with pytest.raises(ValueError): capacity.validate_capacity(bad, cfg, original)


@pytest.mark.parametrize('fail', [False, True])
def test_rank0_validation_uses_no_collective_forward_and_restores_modes_rng_dropout(monkeypatch, fail):
    original_fork = torch.random.fork_rng
    monkeypatch.setattr(capacity.torch.random, 'fork_rng', lambda devices: original_fork(devices=[]))
    monkeypatch.setattr(capacity.torch, 'autocast', lambda *a, **kw: nullcontext())
    models = dict(denoiser=torch.nn.Linear(2, 2), source=torch.nn.Linear(2, 2).eval())
    ddp_wrappers = {'collective-forward-must-not-be-called': object()}
    trainer = SimpleNamespace(models=models, training_models=ddp_wrappers, p_uncond=.1)
    py_before, np_before, torch_before = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    try:
        with capacity.unwrapped_validation(trainer, device=0):
            assert trainer.training_models is models
            assert not any(m.training for m in models.values()) and trainer.p_uncond == 0.
            random.random(); np.random.rand(); torch.randn(10)
            if fail:
                raise RuntimeError('render failed')
    except RuntimeError:
        assert fail
    assert trainer.training_models is ddp_wrappers
    assert models['denoiser'].training and not models['source'].training
    assert trainer.p_uncond == .1 and random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0] and np.array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)
