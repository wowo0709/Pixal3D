import json
import shutil
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.multiview_finetune.configure import STAGES


def test_configure_resolves_external_paths_without_rewriting_recipes(tmp_path):
    data = tmp_path / 'external corpus'
    output = tmp_path / 'external runs'
    configs = tmp_path / 'resolved configs'
    weights = tmp_path / 'converted weights'
    dino = tmp_path / 'local dino'
    rows = []
    for source in ['ObjaverseXL_sketchfab', 'ABO', '3D-FUTURE', 'HSSD']:
        for split in ['train', 'validation']:
            row = dict(source=source, shard='shard', batch=split, split=split)
            rows.append(row)
            root = data / source / 'shard' / split
            root.mkdir(parents=True)
            assets = [f'{source}-{split}-{i}' for i in range(100 if split == 'train' else 2)]
            (root / 'metadata_report.json').write_text(json.dumps({'eligible_assets': assets}))
    (data / 'selection.json').write_text(json.dumps({'batches': rows}))
    checkout = tmp_path / 'checkout with spaces'
    shutil.copytree('experiments/multiview_finetune', checkout / 'experiments/multiview_finetune',
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree('configs/gen/multiview_finetune', checkout / 'configs/gen/multiview_finetune')
    for _, native_config, _ in STAGES.values():
        shutil.copyfile(Path('configs/gen') / native_config, checkout / 'configs/gen' / native_config)
    script = checkout / 'experiments/multiview_finetune/configure.py'
    recipes = {p: p.read_bytes() for p in (checkout / 'configs/gen/multiview_finetune').glob('*.json')}
    result = subprocess.run([sys.executable, str(script),
        '--data-root', str(data), '--weights-root', str(weights), '--dino', str(dino),
        '--output-root', str(output), '--output-config-dir', str(configs)],
        text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert all(path.read_bytes() == content for path, content in recipes.items())
    assert not (data / 'dataset_summary.json').exists()
    summary = json.loads((configs / 'dataset_summary.json').read_text())
    assert summary['training_assets'] == 400
    assert summary['validation_assets'] == 8
    assert set(summary['sources']['train']) == {'ObjaverseXL_sketchfab', 'ABO', '3D-FUTURE', 'HSSD'}
    for stage, (ebs, _, stem) in STAGES.items():
        cfg = json.loads((configs / f'{stage}.json').read_text())
        trainer = cfg['trainer']['args']
        assert trainer['finetune_ckpt']['denoiser'] == str(weights / f'{stem}.pt')
        assert trainer['image_cond_model']['args']['model_name'] == str(dino)
        assert cfg['default_output_dir'] == str(output / stage)
        assert trainer['max_steps'] == 5 * (400 // ebs)
        assert trainer['ema_rate'] == [0.9999]
        assert trainer['mix_precision_dtype'] == 'bfloat16'
        mappings = json.loads(cfg['data_dir'])
        assert len(mappings) == 4
        assert all(str(data) in roots['render_cond'] for roots in mappings.values())
        validation = Path(cfg['validation']['roots_file'])
        assert validation.parent == configs
        assert len(json.loads(validation.read_text())) == 4
    # A mistaken output directory must not replace the tracked recipe templates.
    refused = subprocess.run([sys.executable, str(script), '--data-root', str(data),
        '--output-config-dir', str(checkout / 'configs/gen/multiview_finetune')],
        text=True, capture_output=True)
    assert refused.returncode != 0
    assert 'recipe' in refused.stderr
    assert all(path.read_bytes() == content for path, content in recipes.items())


@pytest.mark.parametrize('visible', ['7', 'GPU-0123', 'MIG-0123'])
def test_single_visible_gpu_supports_arbitrary_assignment(monkeypatch, visible):
    from experiments.multiview_finetune.paths import single_visible_device
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    assert single_visible_device() == visible


@pytest.mark.parametrize('visible', [None, '', '0,1'])
def test_single_visible_gpu_requires_explicit_one_device(monkeypatch, visible):
    from experiments.multiview_finetune.paths import single_visible_device
    if visible is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    with pytest.raises(ValueError, match='exactly one'):
        single_visible_device()


def test_snapshot_lookup_uses_report_file_paths_not_cache_layout(tmp_path):
    from experiments.multiview_finetune.paths import official_snapshot, checkpoint_path
    snapshot = tmp_path / 'snapshot elsewhere'
    report = {'files': {STAGES['ss64'][2]: {'safetensors': str(snapshot / 'ckpts' / 'ss.safetensors')}}}
    path = tmp_path / 'download_report.json'
    path.write_text(json.dumps(report))
    assert official_snapshot(None, path) == snapshot
    assert official_snapshot(tmp_path / 'explicit', path) == tmp_path / 'explicit'
    assert checkpoint_path(tmp_path, 12, '0.9999').name == 'denoiser_ema0.9999_step0000012.pt'


def test_checkpoint_verifier_accepts_later_step_and_completed_warmup(tmp_path):
    import torch
    stage = tmp_path / 'ss64'
    (stage / 'ckpts').mkdir(parents=True)
    pretrained = tmp_path / 'official.pt'
    torch.save({'weight': torch.zeros(2)}, pretrained)
    config = {'trainer': {'args': {
        'finetune_ckpt': {'denoiser': str(pretrained)}, 'ema_rate': [0.9999],
        'batch_size_per_gpu': 96, 'optimizer': {'args': {'lr': 5e-5}},
        'lr_scheduler': {'args': {'warmup_steps': 2}}}},
        'experiment': {'updates_per_epoch': 4}}
    (stage / 'config.json').write_text(json.dumps(config))
    for name in ['denoiser', 'denoiser_ema0.9999']:
        torch.save({'weight': torch.ones(2)}, stage / 'ckpts' / f'{name}_step0000007.pt')
    torch.save({'step': 7, 'lr_scheduler': {'last_epoch': 7},
        'data_sampler': {'epoch': 1, 'idx': 3 * 96},
        'optimizer': {'param_groups': [{'lr': 5e-5}], 'state': {
            0: {'exp_avg': torch.ones(2), 'step': torch.tensor(7)}}}},
        stage / 'ckpts' / 'misc_step0000007.pt')
    result = subprocess.run([sys.executable, 'experiments/multiview_finetune/verify_checkpoints.py',
        '--result', str(tmp_path), '--stage', 'ss64', '--step', '7'], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((stage / 'checkpoint_integrity.json').read_text())
    assert report['checkpoint_step'] == 7
    assert report['scheduler_last_epoch'] == 7
    assert report['lr_next_update'] == 5e-5


@pytest.mark.parametrize(('visible', 'gpu_args', 'message'), [
    (None, [], 'exactly one'),
    ('', [], 'exactly one'),
    ('0,1', [], 'exactly one'),
    ('GPU-assigned', ['--num_gpus', '2'], 'invalid choice'),
    ('GPU-assigned', ['--num_gpus=-1'], 'invalid choice'),
])
def test_training_rejects_assignment_before_runtime_import(tmp_path, visible, gpu_args, message):
    import os
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    if visible is None:
        env.pop('CUDA_VISIBLE_DEVICES', None)
    else:
        env['CUDA_VISIBLE_DEVICES'] = visible
    # Intercept the first runtime import, so a failed guard cannot launch CUDA.
    (tmp_path / 'torch.py').write_text("raise RuntimeError('RUNTIME_IMPORT_REACHED')\n")
    result = subprocess.run([sys.executable, 'experiments/multiview_finetune/train.py',
        '--config', 'unused.json', *gpu_args], env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert message in result.stderr
    assert 'RUNTIME_IMPORT_REACHED' not in result.stderr


def test_training_defaults_to_one_worker_before_runtime_import(tmp_path):
    import os
    (tmp_path / 'torch.py').write_text(
        "import json, sys\nprint(json.dumps(sys.argv), flush=True)\n"
        "raise RuntimeError('RUNTIME_IMPORT_REACHED')\n")
    env = dict(os.environ, PYTHONPATH=str(tmp_path), CUDA_VISIBLE_DEVICES='GPU-assigned')
    result = subprocess.run([sys.executable, 'experiments/multiview_finetune/train.py',
        '--config', 'unused.json'], env=env, text=True, capture_output=True)
    assert 'RUNTIME_IMPORT_REACHED' in result.stderr
    native_argv = json.loads(result.stdout)
    assert native_argv[native_argv.index('--num_gpus') + 1] == '1'


def test_training_help_works_without_device_assignment():
    import os
    env = dict(os.environ)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    result = subprocess.run([sys.executable, 'experiments/multiview_finetune/train.py',
        '--help'], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '--num_gpus' in result.stdout
