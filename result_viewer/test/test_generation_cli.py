"""CPU checks for a portable generation entrypoint, without loading models."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]


def test_help_needs_no_site_packages_or_visible_gpu(tmp_path):
    result = subprocess.run([sys.executable, '-S', str(REPO / 'result_viewer/generate_cases.py'), '--help'],
                            cwd=tmp_path, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for option in ['--weights-root', '--official-root', '--finetuned-root', '--checkpoint-step',
                   '--memory-budget-gib', '--selection', '--output-dir']:
        assert option in result.stdout


@pytest.mark.parametrize('visible', ['3', 'GPU-12345678', 'MIG-12345678'])
def test_single_visible_gpu_is_logical_zero(monkeypatch, visible):
    from result_viewer.generate_cases import require_single_visible_gpu
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    assert require_single_visible_gpu() == 'cuda:0'


@pytest.mark.parametrize('visible', ['', '0,1', '-1'])
def test_generation_requires_explicit_single_gpu(monkeypatch, visible):
    from result_viewer.generate_cases import require_single_visible_gpu
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    with pytest.raises(RuntimeError, match='CUDA_VISIBLE_DEVICES'):
        require_single_visible_gpu()


def test_weight_paths_preserve_provenance_and_accept_explicit_snapshot(tmp_path, monkeypatch):
    from result_viewer.generate_cases import resolve_weight_paths, STAGE_MODELS
    weights = tmp_path / 'checkpoints'
    weights.mkdir()
    revisions = {'TencentARC/Pixal3D': 'official-revision',
                 'camenduru/dinov3-vitl16-pretrain-lvd1689m': 'dino-revision',
                 'microsoft/TRELLIS-image-large': 'ss-revision',
                 'microsoft/TRELLIS.2-4B': 'decoder-revision'}
    report = {'revisions': revisions, 'files': {}, 'dino': '/old-machine/missing/dino'}
    for stage, (_, stem) in STAGE_MODELS.items():
        report['files'][stem] = {'safetensors': '/old-machine/missing/' + stem + '.safetensors'}
    for stage in ['ss', 'shape', 'pbr']:
        report['files'][stage + '_decoder'] = '/old-machine/missing/decoder'
    (weights / 'download_report.json').write_text(json.dumps(report))
    monkeypatch.setenv('HF_HOME', str(tmp_path / 'hf'))
    monkeypatch.delenv('HUGGINGFACE_HUB_CACHE', raising=False)
    monkeypatch.delenv('HF_HUB_CACHE', raising=False)
    official, resolved = resolve_weight_paths(weights)
    assert official == tmp_path / 'hf/hub/models--TencentARC--Pixal3D/snapshots/official-revision'
    assert resolved['dino'] == '/old-machine/missing/dino'
    assert resolved['files']['ss_decoder'] == '/old-machine/missing/decoder'
    supplied = tmp_path / 'official'
    official, resolved = resolve_weight_paths(weights, supplied)
    assert official == supplied
    for _, stem in STAGE_MODELS.values():
        assert Path(resolved['files'][stem]['safetensors']) == supplied / 'ckpts' / (stem + '.safetensors')


def test_relative_weight_report_paths_follow_weights_root(tmp_path):
    from result_viewer.generate_cases import resolve_weight_paths, STAGE_MODELS
    report = {'revisions': {'TencentARC/Pixal3D': 'revision'},
              'dino': 'dino', 'files': {'ss_decoder': 'decoders/ss',
                                      'shape_decoder': 'decoders/shape', 'pbr_decoder': 'decoders/pbr'}}
    for _, stem in STAGE_MODELS.values():
        report['files'][stem] = {'safetensors': 'single_view/' + stem + '.safetensors'}
    (tmp_path / 'download_report.json').write_text(json.dumps(report))
    _, resolved = resolve_weight_paths(tmp_path, tmp_path / 'official')
    assert Path(resolved['dino']) == tmp_path / 'dino'
    assert Path(resolved['files']['shape_decoder']) == tmp_path / 'decoders/shape'


def test_local_decoder_matches_native_config_weights_and_fp16(tmp_path):
    import torch
    from safetensors.torch import save_file
    from pixal3d import models
    from result_viewer.generate_cases import load_local_decoder
    prefix = tmp_path / 'decoder'
    config = {'name': 'SparseStructureDecoder', 'args': {
        'out_channels': 1, 'latent_channels': 2, 'num_res_blocks': 1,
        'num_res_blocks_middle': 1, 'channels': [4], 'use_fp16': True}}
    Path(str(prefix) + '.json').write_text(json.dumps(config))
    source = models.SparseStructureDecoder(**config['args'])
    save_file(source.state_dict(), str(prefix) + '.safetensors')
    expected = models.from_pretrained(str(prefix)).eval().requires_grad_(False)
    actual = load_local_decoder(prefix)
    assert actual.dtype == expected.dtype == torch.float16
    assert actual.use_fp16 and not actual.training
    assert all(not parameter.requires_grad for parameter in actual.parameters())
    for key, value in expected.state_dict().items():
        assert actual.state_dict()[key].dtype == value.dtype
        torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize('missing', ['json', 'safetensors'])
def test_missing_local_decoder_fails_on_local_file(tmp_path, missing):
    from result_viewer.generate_cases import load_local_decoder
    prefix = tmp_path / 'decoder'
    if missing == 'safetensors':
        Path(str(prefix) + '.json').write_text(json.dumps({
            'name': 'SparseStructureDecoder', 'args': {'out_channels': 1, 'latent_channels': 2,
            'num_res_blocks': 1, 'num_res_blocks_middle': 1, 'channels': [4], 'use_fp16': True}}))
    with pytest.raises(FileNotFoundError) as error:
        load_local_decoder(prefix)
    assert str(prefix) + '.' + missing in str(error.value)
