"""Freeze current native RGB conditions on the preselected TRAIN32/cal32 pilot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from .current_geometry_runtime import load_current_trainer, capture_native
from .cache_predicted_geometry import rgb_lineage
from .reproduce_q2c import read, checked, reference, write_new

SCHEMA = 'deformnext-current-native-voxel-condition-v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'checkpoint', 'manifest', 'population', 'queries', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--generation-batch', type=int, default=8)
    args = parser.parse_args()
    from pixal3d.utils.data_utils import recursive_to_device
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    if (args.output/'summary.json').exists():
        raise FileExistsError('immutable current RGB condition cache exists')
    population = read(args.population)
    if population['status'] != 'PASS' or not population['full_population'] or len(population['records']) != 320:
        raise ValueError('expected frozen256/64 current population')
    audit = read(args.queries/'audit-summary.json')
    if (audit['status'] != 'PASS_CACHE_AUDIT' or not audit['original_geometry9_function_exact_for_all352_assets']
            or not audit['source_only_O_occupied_counts_equal_baseline']):
        raise ValueError('current RGB-predicted geometry audit has not passed')
    chosen = sum(([row for row in population['records'] if row['role'] == role][:32]
                  for role in ['train', 'calibration']), [])
    query_rows = {row['sample_id']: row for rank in [0, 1]
                  for row in read(args.queries/'pilot320'/f'shard-{rank}.json')['records']}
    cfg = read(args.config)
    trainer, dataset = load_current_trainer(cfg, args.checkpoint, args.manifest, args.output/'runtime')
    lookup = {row['sample_id']: i for i, row in enumerate(dataset.rows)}
    consumer = reference(args.checkpoint)
    if audit['protocol_pilot']['consumer'] != consumer:
        raise ValueError('predicted query and native condition use different consumers')
    records, started = [], time.monotonic()
    for start in range(0, len(chosen), args.generation_batch):
        selected = chosen[start:start+args.generation_batch]
        rows = [dataset.rows[lookup[item['sample_id']]] for item in selected]
        samples = [dataset._common(row) for row in rows]
        if any('x_0' in sample or any(key.startswith('transport_') for key in sample) for sample in samples):
            raise ValueError('RGB condition cache must not read GT latent or transport values')
        pack = dataset.collate_fn(samples)
        pack.pop('_dataset_name'); pack.pop('_sha256')
        with torch.inference_mode(), torch.autocast('cuda', dtype=trainer.mix_precision_dtype):
            native, target = capture_native(trainer, recursive_to_device(pack, trainer.device))
        for index, (item, row) in enumerate(zip(selected, rows, strict=True)):
            if (row['conditioning_input_sha256'] != item['conditioning_input_sha256']
                    or query_rows[row['sample_id']]['source_cluster_id'] != row['source_cluster_id']):
                raise ValueError('fixed RGB/query asset identity differs')
            path = args.output/'data'/(row['sample_id']+'.pt')
            if path.exists():
                raise FileExistsError('preserve earlier native condition output')
            path.parent.mkdir(parents=True, exist_ok=True)
            value = {'schema': SCHEMA, 'sample_id': row['sample_id'], 'role': item['role'],
                'source_cluster_id': row['source_cluster_id'], 'consumer': consumer,
                'target_proj': target[index:index+1].detach().cpu().contiguous(),
                'native_condition': {branch: {key: tensor[index:index+1].detach().cpu().contiguous()
                    for key, tensor in values.items()} for branch, values in native.items()},
                'rgb_lineage': rgb_lineage(dataset, row)}
            temporary = path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
            torch.save(value, temporary); os.replace(temporary, path)
            records.append({'sample_id': row['sample_id'], 'source_cluster_id': row['source_cluster_id'],
                'role': item['role'], 'artifact': reference(path), 'geometry_query': query_rows[row['sample_id']]['artifact'],
                'conditioning_input_sha256': row['conditioning_input_sha256'],
                'native_projection_replayed_exact': True})
        print(json.dumps({'completed': len(records), 'of': len(chosen), 'seconds': time.monotonic()-started}), flush=True)
    write_new(args.output/'summary.json', {'status': 'PASS', 'schema': SCHEMA, 'records': records,
        'population': reference(args.population), 'query_audit': reference(args.queries/'audit-summary.json'),
        'config': reference(args.config), 'consumer': consumer, 'seconds': time.monotonic()-started,
        'GT_geometry_or_latent_values_opened': False, 'input_views': ['M1', 'M0', 'M0', 'M0'],
        'target_proj_source': 'actual first per-view native projection; same anchor as the multiview mean',
        'performance_claim': False, 'SS_training_started': False})


if __name__ == '__main__':
    main()
