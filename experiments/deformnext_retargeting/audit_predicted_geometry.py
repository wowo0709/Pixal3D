"""Audit fixed RGB-only query shards and original source-only count parity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cache_predicted_geometry import SCHEMA, noise_seed
from .historical_reproduction import geometry_from_logits
from .reproduce_q2c import checked, read, reference, write_new


def check_arrays(data):
    if (str(data['schema']) != SCHEMA or data['geometry9'].shape != (4096, 9)
            or not np.isfinite(data['geometry9']).all()):
        raise ValueError('invalid current predicted-geometry contract')
    expected = geometry_from_logits(data['original_logit64'], data['desired_logit64']).numpy()[0]
    if not np.array_equal(data['geometry9'], expected):
        raise ValueError('stored query differs from original O/D9-channel function')


def audit_group(root, ids, seed_for, world_size):
    records, protocol, artifacts = {}, None, []
    for rank in range(world_size):
        path = root / f'shard-{rank}.json'
        shard = read(path)
        if (shard['status'] != 'PASS' or (shard['rank'], shard['world_size']) != (rank, world_size)
                or shard['population_size'] != len(ids)
                or [r['sample_id'] for r in shard['records']] != ids[rank::world_size]):
            raise ValueError('query shard is not the fixed disjoint population partition')
        current = shard['protocol']
        if protocol is not None and current != protocol:
            raise ValueError('query shards use different frozen protocols')
        protocol = current
        if (current['input_GT_geometry_or_latent_opened'] or current['adapter_training_performed']
                or current['SS_gain_claim'] or current['production_promotion']
                or current['O_states'] != ['M0'] * 4 or current['D_states'] != ['M1']
                or current['O_cameras'] != [8, 9, 10, 11] or current['D_cameras'] != [8]):
            raise ValueError('query cache changed its RGB-only/claim boundary')
        artifacts.append(reference(path))
        for row in shard['records']:
            sid = row['sample_id']
            if sid in records or row['seed'] != seed_for(sid):
                raise ValueError('duplicate query identity or different fixed noise')
            with np.load(checked(row['artifact']), allow_pickle=False) as data:
                check_arrays(data)
                lineage = json.loads(str(data['rgb_lineage']))
                if (str(data['sample_id']) != sid or int(data['seed']) != row['seed']
                        or json.loads(str(data['protocol'])) != current or lineage['sample_id'] != sid
                        or len(lineage['rgb_files']) != 5 or lineage['camera_ids'] != [8, 9, 10, 11]
                        or str(data['source_cluster_id']) != row['source_cluster_id']):
                    raise ValueError('query artifact identity/RGB lineage differs')
                for label, key in [('original', 'original_logit64'), ('desired', 'desired_logit64')]:
                    if int((data[key] > 0).sum()) != row[label + '_occupied_voxels']:
                        raise ValueError('cached raw-logit occupancy summary differs')
            records[sid] = row
    if set(records) != set(ids):
        raise ValueError('query cache union differs from frozen population')
    return records, protocol, artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'population', 'panel-file', 'seed-map-file', 'source-only-baseline', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    import os
    import torch
    torch.set_num_threads(2)
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    population = read(args.population)
    if population['status'] != 'PASS' or not population['full_population'] or len(population['records']) != 320:
        raise ValueError('expected frozen320 current-task pilot identities')
    ids = [r['sample_id'] for r in population['records']]
    pilot, pilot_protocol, artifacts = audit_group(args.root / 'pilot320', ids, noise_seed, 2)
    if any(pilot[r['sample_id']]['source_cluster_id'] != r['source_cluster_id'] for r in population['records']):
        raise ValueError('pilot query cluster identities differ')
    panel = read(args.panel_file)['sample_ids']
    seeds = read(args.seed_map_file)
    common, common_protocol, extra = audit_group(args.root / 'common32', panel, lambda sid: int(seeds[sid]), 1)
    if common_protocol['consumer'] != pilot_protocol['consumer']:
        raise ValueError('current common32 and pilot320 have different frozen SS consumers')
    baseline = read(args.source_only_baseline)
    if (baseline['status'] != 'PASS' or baseline['case'] != 'source-only'
            or baseline['generation_batch_size'] != 8 or baseline['sampling_steps'] != 12 or baseline['guidance'] != 3.):
        raise ValueError('source-only count control has a different sampling protocol')
    lookup = {r['sample_id']: r for r in baseline['per_asset']}
    differences = []
    for sid in panel:
        previous = lookup[sid]
        if previous['seed'] != common[sid]['seed']:
            raise ValueError('source-only control has a different noise')
        differences.append(common[sid]['original_occupied_voxels'] - previous['tp'] - previous['fp'])
    result = {'status': 'PASS_CACHE_AUDIT', 'schema': SCHEMA, 'pilot_assets': 320, 'common_panel_assets': len(panel),
        'original_geometry9_function_exact_for_all352_assets': True, 'all_artifact_SHA256_checked': True,
        'pilot_original_zero_observable_match_queries_retained': 8,
        'source_only_O_occupied_counts_equal_baseline': not any(differences),
        'source_only_O_occupied_count_differences': dict(zip(panel, differences)),
        'source_only_parity_scope': 'occupied counts and seeds only; old baseline lacks prediction voxel SHA, not a full tensor parity claim',
        'protocol_pilot': pilot_protocol, 'protocol_common': common_protocol,
        'inputs': artifacts + extra + [reference(args.population), reference(args.source_only_baseline)],
        'SS_adapter_training_started': False, 'SS_generation_improvement_claim': False,
        'production_promotion': False}
    write_new(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if not isinstance(value, (list, dict))}), flush=True)


if __name__ == '__main__':
    main()
