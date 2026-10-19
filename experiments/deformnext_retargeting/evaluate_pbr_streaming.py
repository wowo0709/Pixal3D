"""Evaluate saved PBR arms one asset at a time with paired input/noise audits.

This reuses the existing anchor renderer and GT Shape protocol. It is a bounded
texture-stage diagnostic, not novel-view or full-cascade evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


def tensor_sha(tensor):
    if hasattr(tensor, 'feats'):
        return hashlib.sha256((tensor_sha(tensor.feats) + tensor_sha(tensor.coords)).encode()).hexdigest()
    digest = hashlib.sha256(str((tensor.shape, tensor.dtype)).encode())
    digest.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def aggregate_metrics(rows):
    """Pool squared errors before PSNR, matching the original batched metric."""
    if not rows:
        raise ValueError('nonempty per-asset PBR metrics required')
    result = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    for key in ['render_psnr', 'base_color_psnr']:
        result[key] = -10 * math.log10(float(np.mean([10**(-row[key]/10) for row in rows])))
    denominator = sum(row['foreground_fraction'] for row in rows)
    if denominator <= 0:
        raise ValueError('nonempty foreground required')
    for key in ['foreground_psnr', 'foreground_base_color_psnr']:
        mse = sum(row['foreground_fraction'] * 10**(-row[key]/10) for row in rows) / denominator
        result[key] = -10 * math.log10(mse)
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--expected-assets', type=int, required=True)
    local, remaining = parser.parse_known_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (visible not in {'0', '1', '2', '3'}
            or visible not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(',')):
        raise ValueError('one explicitly assigned physical GPU0-3 is required')
    output = Path(remaining[remaining.index('--output')+1])
    if output.exists():
        raise ValueError('fresh PBR output namespace required')
    if remaining[remaining.index('--stage')+1] != 'pbr1024':
        raise ValueError('this wrapper only evaluates PBR1024')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.28, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    from PIL import Image
    from . import quick_metrics as quick
    original_pbr, original_sample = quick._pbr_metrics, quick._sample
    source_paths = [Path(__file__).resolve(), Path(quick.__file__).resolve(),
                    Path(__file__).with_name('evaluate_native_checkpoint.py')]
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    cases, audits = [], []
    started = time.monotonic()

    def audited_sample(trainer, dataset, index, steps, seed):
        original_get = trainer.get_sampler
        record = dict(sample_id=dataset.rows[index]['sample_id'], seed=seed,
            source_cluster_id=dataset.rows[index]['source_cluster_id'],
            conditioning_input_sha256=dataset.rows[index]['conditioning_input_sha256'])

        def get_sampler():
            sampler = original_get()
            sample = sampler.sample

            def sampled(*args, **kwargs):
                record['actual_noise_SHA256'] = tensor_sha(kwargs['noise'])
                return sample(*args, **kwargs)
            sampler.sample = sampled
            return sampler

        trainer.get_sampler = get_sampler
        try:
            prediction, target, pack = original_sample(trainer, dataset, index, steps, seed)
        finally:
            trainer.get_sampler = original_get
        record.update(actual_RGB_input_SHA256=tensor_sha(pack['cond']),
            actual_GT_PBR_latent_SHA256=tensor_sha(target),
            actual_GT_Shape_condition_SHA256=tensor_sha(pack['concat_cond']))
        audits.append(record)
        return prediction, target, pack

    def streaming(trainer, dataset, indices, config):
        if len(indices) != local.expected_assets:
            raise ValueError('declared fixed PBR panel size differs')
        seeds = quick.panel_seeds(dataset, indices, config)
        panels, numeric = [], []
        for number, (index, seed) in enumerate(zip(indices, seeds)):
            one = dict(config, seed=seed)
            metrics, panel = original_pbr(trainer, dataset, [index], one)
            if len(audits) != number + 1:
                raise ValueError('exactly one actual generation/input audit per asset required')
            record = dict(audits[-1], **metrics)
            if not all(math.isfinite(value) for value in metrics.values()):
                raise ValueError('nonfinite rendering metric')
            cases.append(record)
            numeric.append(metrics)
            name = f'case{number:03d}.png'
            Image.fromarray(panel).save(output/name)
            record['render_panel'] = name
            if number < 8:
                panels.append(panel)
            (output/'streaming-progress.json').write_text(json.dumps(dict(completed=len(cases),
                expected=local.expected_assets, seconds=time.monotonic()-started,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30), indent=2)+'\n')
            with (output/'per-asset.partial.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            print(json.dumps(dict(PBR_completed=len(cases), expected=local.expected_assets,
                foreground_psnr=metrics['foreground_psnr'])), flush=True)
        trainer._quick_per_asset = cases
        return aggregate_metrics(numeric), np.concatenate(panels, axis=0)

    quick._sample, quick._pbr_metrics = audited_sample, streaming
    old_arguments = sys.argv
    try:
        from . import evaluate_native_checkpoint as entry
        sys.argv = [str(entry.__file__), *remaining]
        entry.main()
    finally:
        quick._sample, quick._pbr_metrics = original_sample, original_pbr
        sys.argv = old_arguments
    path = output/'evaluation.json'
    result = json.loads(path.read_text())
    if len(cases) != local.expected_assets or len(result['per_asset']) != local.expected_assets:
        raise ValueError('complete actual PBR panel required')
    if any(hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest
           for name,digest in source_hashes.items()):
        raise ValueError('PBR evaluator source changed during execution')
    result.update(streaming_PBR=dict(actual_assets=len(cases), fixed_GT_target_Shape=True,
        render_view='input target anchor camera08', novel_view=False,
        source_preservation_LPIPS=False, target_crop_LPIPS_diagnostic=True,
        source_SHA256=source_hashes, GPU=visible, allocator_fraction=.28,
        seconds=time.monotonic()-started, numerical_mode='BF16 native;TF32 disabled;paired noise/AO;exact replay not yet proven',
        model_adopted=False, full_goal_complete=False))
    path.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
