import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

from configure import REPO as SOURCE, STAGES
from paths import single_visible_device

parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=STAGES)
parser.add_argument('--sizes', default='1,2,3,4,6,8,12')
parser.add_argument('--stress', action='store_true')
parser.add_argument('--config', type=Path, required=True, help='Matching resolved native stage config')
parser.add_argument('--roots', type=Path, help='Override training roots; default: config data_dir')
parser.add_argument('--headroom-gib', type=float, default=0, help='Required free-memory margin beyond measured peak')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
gpu = single_visible_device()
sys.path.insert(0, str(SOURCE))
import numpy as np
import torch
from pixal3d import models, trainers, datasets
from pixal3d.utils.data_utils import recursive_to_device

torch.set_num_threads(4)
torch.manual_seed(20261004)
np.random.seed(20261004)
config = json.loads(args.config.read_text())
ebs = config['trainer']['args']['batch_size_per_gpu']
data_dir = args.roots.read_text() if args.roots else config['data_dir']
free_start, total = torch.cuda.mem_get_info()
dataset = getattr(datasets, config['dataset']['name'])(data_dir, **config['dataset']['args'])
dataset.min_condition_views = dataset.max_condition_views = 6
if args.stage != 'ss64':
    # Select the largest eligible asset to bound denoiser tokens per instance.
    index = int(np.argmax(dataset.loads))
else:
    index = 0
if args.stress and args.stage != 'ss64':
    from unittest.mock import patch
    root, sha, _ = dataset.instances[index]
    key = 'pbr_latent' if args.stage == 'pbr1024' else 'shape_latent'
    counts = [len(np.load(Path(root[key]) / sha / f'view{anchor:02d}.npz')['coords']) for anchor in (0, 1)]
    with patch.object(np.random, 'randint', return_value=int(np.argmax(counts))):
        item = dataset[index]
else:
    item = dataset[index]
if args.stage != 'ss64':
    print('HIGH_TOKEN_ASSET', index, len(item['coords']), flush=True)
model = getattr(models, config['models']['denoiser']['name'])(**config['models']['denoiser']['args']).cuda()
out = args.output
out.mkdir(parents=True, exist_ok=True)
trainer = getattr(trainers, config['trainer']['name'])(
    {'denoiser': model}, dataset, **config['trainer']['args'],
    output_dir=str(out), load_dir=None, step=None)
if args.stress and hasattr(trainer, 'elastic_controller'):
    # Test the highest activation retention allowed during these ten updates,
    # rather than a lucky random Elastic checkpointing draw.
    trainer.elastic_controller.get_mem_ratio = lambda input_size: trainer.elastic_controller._max_mem_ratio
report = {'stage': args.stage, 'gpu': gpu, 'ebs': ebs, 'K': 6, 'stress':args.stress,
    'eligible_dataset_assets':len(dataset), 'config':str(args.config), 'dataset_roots_file':str(args.roots) if args.roots else None,
    'headroom_gib':args.headroom_gib,
    'selected_asset':dataset.instances[index][1], 'anchor':int(item['view_idx']),
    'free_start_gib': free_start / 2**30, 'total_gib': total / 2**30, 'candidates': []}
# Keep the full EBS input-image allocation resident, as BasicTrainer.load_data does.
# The tested pack is a repeated high-token real asset, with its real camera/latent data.
image_size = config['dataset']['args']['image_size']
resident_inputs = None if config['trainer']['args'].get('microbatch_to_device') else torch.empty(
    (ebs, 6, 3, image_size, image_size), device='cuda', dtype=torch.float32)
for size in map(int, args.sizes.split(',')):
    torch.cuda.empty_cache()
    row = {'microbatch_assets': size}
    batch = dataset.collate_fn([item] * size)
    pack = batch if isinstance(batch, dict) else batch[0]
    if not config['trainer']['args'].get('microbatch_to_device'):
        pack = recursive_to_device(pack, torch.device('cuda:0'))
    if args.stage != 'ss64':
        row['tokens'] = int(pack['x_0'].feats.shape[0])
    torch.cuda.synchronize()
    began = time.monotonic()
    peaks = []
    try:
        # Two complete optimizer updates include initialized Adam state on pass 2.
        for repeat in range(2):
            torch.cuda.reset_peak_memory_stats()
            if repeat == 1:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                    result = trainer.run_step([pack])
                names = sorted({event.key for event in profile.key_averages() if 'flash' in event.key.lower()})
                row['flash_attention_events'] = names
            else:
                result = trainer.run_step([pack])
            torch.cuda.synchronize()
            assert np.isfinite(result['loss']['loss'])
            peaks.append(torch.cuda.max_memory_allocated() / 2**30)
        row.update(ok=True, loss=result['loss'], peak_allocated_gib=max(peaks),
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            elapsed_seconds=time.monotonic()-began,
            free_after_gib=torch.cuda.mem_get_info()[0] / 2**30)
        # Caller-selected margin includes memory unavailable to this process.
        row['safe'] = max(peaks) < free_start / 2**30 - args.headroom_gib
    except torch.OutOfMemoryError as error:
        row.update(ok=False, safe=False, error=str(error))
        report['candidates'].append(row)
        (out / 'memory_report.json').write_text(json.dumps(report, indent=2) + '\n')
        print('PROBE_OOM', json.dumps(row), flush=True)
        raise
    report['candidates'].append(row)
    (out / 'memory_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PROBE_RESULT', json.dumps(row), flush=True)
    del pack, batch
    gc.collect()
    if not row['safe']:
        break
print('PROBE_COMPLETE', flush=True)
