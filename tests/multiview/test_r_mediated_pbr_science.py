from copy import deepcopy
import json
import sys
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.deformnext_retargeting.evaluate_pbr_streaming import aggregate_metrics
from experiments.deformnext_retargeting.evaluate_r_mediated_pbr_initial import IDENTITY_FIELDS
from experiments.deformnext_retargeting.r_mediated_native_runtime import state_sha
from experiments.deformnext_retargeting.r_mediated_pbr_science_recipe import (
    make_science_config, validate_initial4, validate_science,
)
from experiments.deformnext_retargeting.r_mediated_pbr_science_metrics import log_validation
from experiments.deformnext_retargeting import train_r_mediated_pbr_native as entry
from test_r_mediated_pbr_native_probe import fixture


def recipe(mode='vanilla', world=2, seed=101):
    proof, original = fixture(mode)
    proof['data_dir'] = '/declared/train.jsonl'
    validation = dict(dataset_name=proof['dataset']['name'],
        dataset_args=dict(proof['dataset']['args'], expected_split='validation'),
        roots_file='/declared/validation.jsonl', every_steps=380, max_assets=4, assets_per_source=None,
        quick_metrics=dict(enabled=True, panel_size=4, generation_batch_size=1,
            sampling_steps=12, render_resolution=256, panel_file='/declared/panel4.json',
            seed_map_file='/declared/seeds.json'))
    cfg = make_science_config(proof, original, validation, world=world, seed=seed,
        run_id='native-PBR-'+mode, output_dir='/declared/output', port=47951)
    return cfg, proof, original


@pytest.mark.parametrize('mode', ['vanilla', 'top1-only', 'top1-plus-feature', 'raw-topk-plus-feature'])
@pytest.mark.parametrize('world', [2, 4])
@pytest.mark.parametrize('seed', [101, 202])
def test_science_preserves_actual_proof_lineage_and_fixed_budget(mode, world, seed):
    cfg, proof, original = recipe(mode, world, seed)
    assert validate_science(cfg, proof, original) == mode
    assert cfg['trainer']['args']['batch_size_per_gpu']*world == 24
    assert cfg['trainer']['args']['max_checkpoints'] is None
    for key, value in [('batch_split', 2), ('p_uncond', 0.), ('max_steps', 1905),
                       ('i_save', 381), ('max_checkpoints', 4), ('num_workers', 8)]:
        wrong = deepcopy(cfg); wrong['trainer']['args'][key] = value
        with pytest.raises(ValueError): validate_science(wrong, proof, original)
    wrong = deepcopy(cfg); wrong['dataset']['args']['max_tokens'] = 16000
    with pytest.raises(ValueError): validate_science(wrong, proof, original)
    for key, value in [('source_multiview_step', 762), ('source_teacher_updates', 112), ('validate_at_start', False)]:
        wrong = deepcopy(cfg); wrong['experiment'][key] = value
        with pytest.raises(ValueError): validate_science(wrong, proof, original)


def initial_rows():
    metrics = dict(render_psnr=25., base_color_psnr=26., foreground_psnr=20.,
        foreground_base_color_psnr=21., foreground_fraction=.4, crop_lpips=.1)
    rows = [{**{key: str(i)+'-'+key for key in IDENTITY_FIELDS}, 'metrics': deepcopy(metrics)} for i in range(32)]
    reference = dict(status='PASS_ACTUAL_NATIVE_PBR_COLD_GENERATED32', source_multiview_step=16850,
        per_asset=rows, optimizer_updates=0, original_and_source_states_unchanged=True)
    selected = deepcopy(rows[:4])
    actual = dict(step=0, quick_per_asset=selected, quick_panel_ids=[r['sample_id'] for r in selected],
        quick_metrics=aggregate_metrics([r['metrics'] for r in selected]), seconds=1.)
    return actual, reference


def test_initial_native_guard_requires_prediction_hashes_and_pixel_error_pooling():
    actual, reference = initial_rows()
    validate_initial4(actual, reference)
    wrong = deepcopy(actual); wrong['quick_per_asset'][1]['actual_prediction_PBR_latent_SHA256'] = 'wrong'
    with pytest.raises(ValueError, match='before first update'): validate_initial4(wrong, reference)
    wrong = deepcopy(actual); wrong['quick_metrics']['render_psnr'] += .01
    with pytest.raises(ValueError, match='pixel-error pooling'): validate_initial4(wrong, reference)
    wrong = deepcopy(reference); wrong['optimizer_updates'] = 3
    with pytest.raises(ValueError): validate_initial4(actual, wrong)


def test_scientific_validation_logs_both_actual_metrics_and_GT_prediction_images(monkeypatch):
    images, logs = [], []
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(Image=lambda panel, caption:
        images.append((panel, caption)) or 'image'))
    trainer = SimpleNamespace(wandb_run=SimpleNamespace(log=lambda payload, step: logs.append((payload, step))))
    actual, _ = initial_rows()
    for row in actual['quick_per_asset']: row['flow_loss'] = .25
    panel = np.zeros((8, 8, 3), dtype=np.uint8)
    log_validation(trainer, actual, panel)
    assert logs[0][1] == 0 and logs[0][0]['validation/render_psnr'] == actual['quick_metrics']['render_psnr']
    assert logs[0][0]['validation/K4_loss'] == .25 and logs[0][0]['validation/quick_panel'] == 'image'
    assert images[0][0] is panel and 'Fixed GT Shape' in images[0][1]


def synthetic_trainer(tmp_path, monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapters = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1)) for _ in range(360)])
        def forward(self, x, t, cond): return x
    model = Model()
    reference = tmp_path/'reference.json'
    reference.write_text(json.dumps(dict(initial_LoRA_SHA256=state_sha(model.named_parameters()))))
    class Base:
        def __init__(self):
            self.models = {'denoiser': model}; self.dataset = range(9139); self.dataloader = range(380)
            self.optimizer = SimpleNamespace(state={}); self.step = 0
            self.world_size, self.batch_size, self.batch_size_per_gpu, self.batch_split = 2, 24, 12, 1
            self.rank, self.is_master, self.device = 0, True, torch.device('cpu')
            self.data_sampler = SimpleNamespace(state_dict=lambda: dict(epoch=0, idx=0))
            self.lr_scheduler = SimpleNamespace(state_dict=lambda: dict(last_epoch=self.step))
            self.output_dir = str(tmp_path); self.saves = self.updates = 0
        def save(self, non_blocking=False): self.saves += 1
        def run_step(self, packs):
            self.updates += 1
            self.models['denoiser'](torch.zeros(1), torch.zeros(1), {'global': torch.zeros(1)})
            return {'loss': {'loss': .1}}
    cfg, _, _ = recipe()
    plan = dict(mode='vanilla', initial_reference=str(reference), canonical_local_root=str(tmp_path),
        checkpoint_archive_command=['declared-archiver'], checkpoint_archive_root=str(tmp_path/'archives'))
    cls, _ = entry.science_trainer(Base, cfg, plan)
    monkeypatch.setattr(entry.torch.distributed, 'broadcast', lambda flag, src: None)
    return cls()


def test_first_optimizer_step_is_blocked_until_actual_initial_validation_admission(tmp_path, monkeypatch):
    trainer = synthetic_trainer(tmp_path, monkeypatch)
    packs = [{'_sha256': [str(i) for i in range(12)]}]
    with pytest.raises(ValueError, match='before the first'): trainer.run_step(packs)
    assert trainer.updates == 0
    trainer._PBR_initial_guard_passed = True
    trainer.run_step(packs)
    assert trainer.updates == 1
    row = json.loads((tmp_path/'training-noise-trace.jsonl').read_text())
    assert row['asset_ids'] == packs[0]['_sha256'] and row['step'] == 1


def test_duplicate_epoch_save_cannot_overwrite_closed_weights_or_hide_changes(tmp_path, monkeypatch):
    trainer = synthetic_trainer(tmp_path, monkeypatch)
    archives = []
    monkeypatch.setattr(entry.subprocess, 'run', lambda command, check: archives.append(command))
    monkeypatch.setattr(entry, 'ClosedInputs', lambda root: SimpleNamespace(check=lambda path, receipt: None))
    trainer.step = 380
    trainer.save(); trainer.save()
    assert trainer.saves == 1 and len(archives) == 1
    with torch.no_grad(): trainer.models['denoiser'].adapters[0].add_(1.)
    with pytest.raises(ValueError, match='previously archived step'): trainer.save()


def test_archive_failure_does_not_mark_a_checkpoint_verified(tmp_path, monkeypatch):
    trainer = synthetic_trainer(tmp_path, monkeypatch)
    def fail(command, check): raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(entry.subprocess, 'run', fail)
    trainer.step = 380
    with pytest.raises(subprocess.CalledProcessError): trainer.save()
    assert trainer.saves == 1 and trainer._PBR_last_archived_step is None
