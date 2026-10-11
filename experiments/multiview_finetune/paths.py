"""Explicit run paths and a single caller-assigned visible CUDA device."""
import json
import os
from pathlib import Path


def single_visible_device():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible.strip() or len(visible.split(',')) != 1:
        raise ValueError('Set CUDA_VISIBLE_DEVICES to exactly one assigned GPU index or UUID')
    return visible


def checkpoint_path(root, step, ema=None):
    name = 'denoiser' if ema is None else f'denoiser_ema{ema}'
    return Path(root) / 'ckpts' / f'{name}_step{step:07d}.pt'


def official_snapshot(snapshot, download_report):
    if snapshot is not None:
        return Path(snapshot)
    report = json.loads(Path(download_report).read_text())
    path = report['files']['ss_flow_img_dit_1_3B_64_bf16']['safetensors']
    return Path(path).parent.parent


def assigned_gpu_count(requested=None):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    devices = [value.strip() for value in visible.split(',') if value.strip()]
    if not devices or (requested is not None and requested != len(devices)):
        raise ValueError('num_gpus must match the explicitly assigned visible GPUs')
    return len(devices)
