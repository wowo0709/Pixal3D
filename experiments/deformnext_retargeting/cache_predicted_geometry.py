"""Frozen mixed385 RGB-only O/D geometry queries for a fixed current-data pilot.

No mesh, target vertex array, SS ground-truth latent, or topology label is
opened. The original nine-channel geometry function consumes model logits.
This is query preparation, not evidence of a trained SS improvement.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .historical_reproduction import geometry_from_logits
from .reproduce_q2c import read, reference, write_new

SCHEMA = 'deformnext-predicted-OD-geometry9-v1'
CAMERA_KEYS = ('camera_angle_x', 'camera_distance', 'transform_matrix', 'view_indices')


def rgb_packs(dataset, indices, load_image):
    """Read only the audited RGB/camera path; never dataset[index]/x_0."""
    original, desired = [], []
    for index in indices:
        row = dataset.rows[index]
        if (row['latent_geometry_schema'] != 'deformnext-m1-render-world-v4'
                or row['conditioning_camera_ids'] != [8, 9, 10, 11]
                or not np.allclose(row['transform_matrix'][0], row['source_anchor_transform_matrix'], atol=1e-6, rtol=0)
                or abs(row['camera_angle_x'][0] - row['source_anchor_camera_angle_x']) > 1e-6):
            raise ValueError('O/D must share the fixed world-v4 target08 camera frame')
        base = dataset._common(row)
        if set(base) != {'cond', 'camera_angle_x', 'camera_distance', 'transform_matrix',
                         'mesh_scale', 'view_indices', '_dataset_name', '_sha256'}:
            raise ValueError('RGB-only query input has unexpected transport or target data')
        if base['cond'].shape != (4, 3, 512, 512):
            raise ValueError('expected native512px four-view RGB')
        source = dict(base)
        source['cond'] = base['cond'].clone()
        source['cond'][0] = load_image(dataset._resolve(row['source_anchor_rgb']), dataset.image_size)
        target = {key: value[:1] if key in ('cond', *CAMERA_KEYS) else value
                  for key, value in base.items()}
        original.append(source)
        desired.append(target)
    result = [dataset.collate_fn(rows) for rows in [original, desired]]
    for pack in result:
        pack.pop('_dataset_name'); pack.pop('_sha256')
    if result[0]['cond'].shape[1] != 4 or result[1]['cond'].shape[1] != 1:
        raise ValueError('expected O=four undeformed RGB views, D=one deformed target RGB')
    return result


def noise_seed(sample_id):
    return int.from_bytes(hashlib.sha256(('current-geometry-query-v1|' + sample_id).encode()).digest()[:8], 'big') % (2**63 - 1)


def rgb_lineage(dataset, row):
    paths = [row['source_anchor_rgb'], row['target_anchor'], *row['source_images']]
    return {'sample_id': row['sample_id'], 'rgb_files': [reference(dataset._resolve(path)) for path in paths],
            'camera_ids': row['conditioning_camera_ids'], 'transforms': row['transform_matrix'],
            'fov': row['camera_angle_x'], 'mesh_scale': row['mesh_scale'],
            'conditioning_input_sha256': row['conditioning_input_sha256']}


def selected_ids(args):
    if args.task_population:
        population = read(args.task_population)
        if population['status'] != 'PASS' or not population['full_population'] or len(population['records']) != 320:
            raise ValueError('current task pilot population has not passed')
        ids = [row['sample_id'] for row in population['records']]
    else:
        ids = read(args.panel_file)['sample_ids']
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('fixed population has duplicate or empty sample IDs')
    return ids[:1] if args.preflight_only else ids


def run(args):
    from pixal3d import datasets, models, trainers
    from pixal3d.datasets.deformnext import _load_image
    from pixal3d.utils.data_utils import recursive_to_device
    cfg = read(args.config)
    ds_args = copy.deepcopy(cfg['dataset']['args'])
    ds_args.update(transport_arm='v1', expected_split=args.split)
    for key in ['match_root', 'projection_root', 'deformation_root', 'expected_q2c_sha256']:
        ds_args.pop(key, None)
    dataset = getattr(datasets, cfg['dataset']['name'])(str(args.manifest), **ds_args)
    lookup = {row['sample_id']: i for i, row in enumerate(dataset.rows)}
    ids = selected_ids(args)
    if len(lookup) != len(dataset) or not set(ids) <= lookup.keys():
        raise ValueError('fixed query population is not in the explicitly supplied manifest')
    if not (0 <= args.rank < args.world_size):
        raise ValueError('invalid explicit cache rank')
    assigned = ids[args.rank::args.world_size]
    if not assigned:
        raise ValueError('empty query-cache shard')
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / f'shard-{args.rank}.json'
    if summary_path.exists():
        raise FileExistsError('immutable query shard is already complete')
    with torch.device('meta'):
        model_dict = {name: getattr(models, spec['name'])(**spec['args']) for name, spec in cfg['models'].items()}
    kw = copy.deepcopy(cfg['trainer']['args'])
    kw.update(continuation_ckpt={'denoiser': str(args.checkpoint)}, deform_conditioner=None,
              batch_size_per_gpu=1, batch_split=1, num_workers=0, prefetch_data=False, max_steps=1)
    kw.get('model_learning_rates', {}).pop('deform_conditioner', None)
    trainer = getattr(trainers, cfg['trainer']['name'])(model_dict, dataset, **kw,
        output_dir=str(args.output / f'runtime-rank{args.rank}'), load_dir=None, step=None, wandb_run=None)
    trainer.models['denoiser'].eval().requires_grad_(False)
    trainer.p_uncond = 0
    manifest_ref, consumer = reference(args.manifest), reference(args.checkpoint)
    proof = None
    if not args.preflight_only:
        if args.preflight_summary is None:
            raise ValueError('full query cache requires an actual GPU preflight certificate')
        preflight = read(args.preflight_summary)
        if (preflight['status'] != 'PASS_PREFLIGHT_ONLY' or len(preflight['records']) != 1
                or preflight['protocol']['consumer'] != consumer
                or preflight['protocol']['config'] != reference(args.config)):
            raise ValueError('GPU preflight differs from the frozen current consumer')
        proof = reference(args.preflight_summary)
    seed_map = read(args.seed_map_file) if args.seed_map_file else None
    protocol = {'schema': SCHEMA, 'consumer': consumer, 'config': reference(args.config),
        'manifest': manifest_ref, 'population': reference(args.task_population or args.panel_file),
        'noise_protocol': reference(args.seed_map_file) if args.seed_map_file else 'current-geometry-query-v1 SHA256 pair seed',
        'sampling_steps': 12, 'guidance_strength': 3., 'same_O_D_initial_noise': True,
        'O_states': ['M0'] * 4, 'D_states': ['M1'], 'O_cameras': [8, 9, 10, 11], 'D_cameras': [8],
        'input_GT_geometry_or_latent_opened': False, 'adapter_training_performed': False,
        'production_promotion': False, 'SS_gain_claim': False, 'GPU_preflight': proof}
    records, started = [], time.monotonic()
    run_log = None
    if args.wandb_id:
        import wandb
        run_log = wandb.init(project='pixal3d-deformnext-retargeting', entity='youngwoojeon-unist',
            id=args.wandb_id, resume='never', dir=str(args.output), config=protocol)
    for start in range(0, len(assigned), args.generation_batch):
        batch_ids = assigned[start:start+args.generation_batch]
        rows = [dataset.rows[lookup[sid]] for sid in batch_ids]
        lineages = [rgb_lineage(dataset, row) for row in rows]
        seeds = [int(seed_map[sid]) if seed_map is not None else noise_seed(sid) for sid in batch_ids]
        paths = [args.output / 'data' / (sid + '.npz') for sid in batch_ids]
        if any(path.exists() for path in paths):
            raise FileExistsError('preserve partial cache; use a fresh output namespace')
        packs = rgb_packs(dataset, [lookup[sid] for sid in batch_ids], _load_image)
        # Match current common-panel RNG: one per-asset CUDA randn; no GT shape/values.
        noises = []
        for seed in seeds:
            torch.manual_seed(seed); torch.cuda.manual_seed(seed)
            noises.append(torch.randn(1, 8, 16, 16, 16, device=trainer.device))
        noise = torch.cat(noises)
        logits = []
        with torch.inference_mode(), torch.autocast('cuda', dtype=trainer.mix_precision_dtype):
            for pack in packs:
                condition = trainer.get_inference_cond(**recursive_to_device(pack, trainer.device))
                prediction = trainer.get_sampler().sample(trainer.models['denoiser'], noise=noise.clone(),
                    **condition, steps=12, guidance_strength=3., verbose=False).samples
                # Public dataset decoder returns raw, unthresholded native logits.
                decoded = dataset.decode_latent(prediction, batch_size=2)
                if decoded.shape != (len(batch_ids), 1, 64, 64, 64):
                    raise ValueError('predicted SS decoder must return native raw logits64')
                logits.append(decoded[:, 0].float().cpu().numpy())
        for index, (sid, row, path, seed, lineage) in enumerate(zip(batch_ids, rows, paths, seeds, lineages, strict=True)):
            geometry = geometry_from_logits(logits[0][index], logits[1][index]).numpy()[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name('.' + path.name + f'.{os.getpid()}.tmp.npz')
            np.savez(temporary, schema=SCHEMA, sample_id=sid, source_cluster_id=row['source_cluster_id'],
                seed=seed, original_logit64=logits[0][index], desired_logit64=logits[1][index], geometry9=geometry,
                rgb_lineage=json.dumps(lineage, sort_keys=True), protocol=json.dumps(protocol, sort_keys=True))
            os.replace(temporary, path)
            record = {'sample_id': sid, 'source_cluster_id': row['source_cluster_id'], 'seed': seed,
                'artifact': reference(path), 'original_occupied_voxels': int((logits[0][index] > 0).sum()),
                'desired_occupied_voxels': int((logits[1][index] > 0).sum()),
                'geometry9_rms': float(np.sqrt(np.mean(geometry ** 2)))}
            records.append(record)
        print(json.dumps({'rank': args.rank, 'completed': len(records), 'of': len(assigned),
            'elapsed_seconds': time.monotonic()-started, 'peak_GPU_allocated_gib': torch.cuda.max_memory_allocated()/1024**3}), flush=True)
        if run_log:
            run_log.log({'cache/completed': len(records), 'cache/elapsed_seconds': time.monotonic()-started})
    write_new(summary_path, {'status': 'PASS_PREFLIGHT_ONLY' if args.preflight_only else 'PASS',
        'rank': args.rank, 'world_size': args.world_size, 'population_size': len(ids), 'records': records,
        'seconds': time.monotonic()-started, 'protocol': protocol,
        'peak_GPU_allocated_gib': torch.cuda.max_memory_allocated()/1024**3})
    if run_log:
        run_log.summary.update({'status': 'PASS_EXECUTION', 'production_promotion': False})
        run_log.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'checkpoint', 'manifest', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    population = parser.add_mutually_exclusive_group(required=True)
    population.add_argument('--task-population', type=Path)
    population.add_argument('--panel-file', type=Path)
    parser.add_argument('--split', choices=['train', 'validation'], required=True)
    parser.add_argument('--seed-map-file', type=Path)
    parser.add_argument('--generation-batch', type=int, default=8)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--preflight-summary', type=Path)
    parser.add_argument('--wandb-id')
    args = parser.parse_args()
    if args.generation_batch < 1:
        parser.error('positive generation batch required')
    torch.set_num_threads(2)
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    torch.cuda.set_device(0)
    run(args)


if __name__ == '__main__':
    main()
