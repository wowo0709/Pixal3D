"""Check actual ray main initialization and measured DDP48 startup without CUDA."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time

from .matching_v2_ss import audit_memory_probe
from .ray_group_study import ARMS, source_record, validate_profile
from .semantic_topk_search import write_new


def read(path):
    return json.loads(path.read_text())


def lines(path):
    value = path.read_text()
    complete = value.splitlines()
    if not value.endswith('\n'):
        complete = complete[:-1]
    return [json.loads(line) for line in complete if line]


def verify_initial(candidate, reference):
    if candidate['step'] != 0 or reference['step'] != 0:
        raise ValueError('actual pre-update validation required')
    ids = candidate['quick_panel_ids']
    if len(ids) != 32 or len(set(ids)) != 32 or ids != reference['quick_panel_ids']:
        raise ValueError('fixed initial common32 order differs')
    if (candidate['quick_metrics'] != reference['quick_metrics']
            or candidate['quick_per_asset'] != reference['quick_per_asset']):
        raise ValueError('initial native predictions/GT/noise/metrics differ')
    if any(not row.get('target_voxels_sha256') for row in candidate['quick_per_asset']):
        raise ValueError('actual initial target hashes required')


def verify_ranks(byrank, *, first_steps):
    if len(byrank) != 2:
        raise ValueError('two actual rank records required')
    for rank, rows in enumerate(byrank):
        if [row['step'] for row in rows] != first_steps:
            raise ValueError('actual rank steps differ: ' + str(rank))
        for row in rows:
            if (row['rank'] != rank or row['world_size'] != 2 or row['effective_batch_size'] != 96
                    or row['accumulation_packs'] != 1 or row['microbatch_assets'] != [48]
                    or row['K'] != 4 or not math.isfinite(row['loss']['loss'])
                    or not 0 < row['peak_allocated_gib'] <= 91
                    or not math.isfinite(row['compute_seconds']) or row['compute_seconds'] <= 0):
                raise ValueError('actual main DDP/microbatch/finiteness contract differs')


def audit(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('startup audit is CPU-only;use CUDA_VISIBLE_DEVICES empty')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != args.expected_sha:
        raise ValueError('declared startup auditor source changed')
    root, name = args.root, args.arm
    run = root / 'runs_ss64' / name
    folder = root / 'startup-audits-v1' / name
    if (folder / 'summary.json').exists():
        raise FileExistsError('preserve actual startup audit')
    paths = [run / ('runtime_batch.jsonl' if rank == 0 else 'runtime_batch.rank1.jsonl') for rank in range(2)]
    while True:
        rows = [lines(path) for path in paths]
        if all(len(values) >= 3 for values in rows):
            break
        job = read(args.work_root / 'queue-state.json').get('train-' + name, {})
        if job.get('status') in {'FAILED', 'HELD', 'completed'}:
            raise RuntimeError('actual main is terminal without three saved updates: ' + str(job))
        print('AWAITING_FIRST_THREE_ACTUAL_MAIN_UPDATES', [len(v) for v in rows], flush=True)
        time.sleep(10)
    cfg = read(root / 'configs' / (name + '.json'))
    validate_profile(cfg, root)
    entry = read(root / 'verification' / name / 'entry-source-contract.json')
    if entry['component_SHA256'] != source_record()['component_SHA256']:
        raise ValueError('executed numerical ray source differs')
    contract = read(getattr(args, 'prepared_source_contract', None)
                    or root / 'provenance/prelaunch-v3/source-contract.json')
    archived = [Path(row['archived_path']) for row in contract['runtime_components']
                if Path(row['runtime_path']) == root / 'configs' / (name + '.json')]
    if len(archived) != 1:
        raise ValueError('exact archived prelaunch recipe required')
    prepared = read(archived[0])
    for key in ['models', 'dataset', 'trainer', 'validation', 'data_dir', 'training_seed', 'num_gpus']:
        if cfg[key] != prepared[key]:
            raise ValueError('numerical main recipe differs from prepared contract: ' + key)
    first = [values[:3] for values in rows]
    verify_ranks(first, first_steps=[1,2,3])
    native = read(root / 'verification' / name / 'native-proof.json')
    if native['status'] != 'PASS' or any(x != 0 for x in native['condition_max_abs_errors'].values()):
        raise ValueError('actual launch native conditioning proof missing')
    bench = root / 'memory-bench' / name / 'split1'
    measured = [lines(bench / ('runtime_batch.jsonl' if rank == 0 else 'runtime_batch.rank1.jsonl')) for rank in range(2)]
    verify_ranks(measured, first_steps=[1,2])
    if not audit_memory_probe(measured,48,2,1):
        raise ValueError('actual ray DDP48 memory proof failed')
    initial = lines(run / 'validation.jsonl')[0]
    references = [root.parent.parent / 'ss-seed-confirmation-v1/runs_ss64/v1/validation.jsonl',
                  root.parent / 'runs_ss64/static-top1-c1/validation.jsonl']
    for path in references:
        verify_initial(initial, lines(path)[0])
    write_new(folder / 'initial-validation-snapshot.json', initial)
    for rank in range(2):
        write_new(folder / f'main-rank{rank}-first3.json', first[rank])
        write_new(folder / f'benchmark-rank{rank}-two-updates.json', measured[rank])
    write_new(folder / 'summary.json', {
        'status':'PASS_ACTUAL_RAY_MAIN_NATIVE_INITIAL32_AND_DDP48_STARTUP',
        'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arm':name,
        'initial_common32_per_asset_GT_noise_and_four_metrics_exact_to_V1_and_old_Top1R':True,
        'initial_IoU':initial['quick_metrics']['voxel_iou'],'global_batch':96,'world_size':2,
        'microbatch_per_rank':48,'accumulation':1,'actual_main_steps_checked':[1,2,3],
        'actual_DDP_benchmark_steps_checked':[1,2],
        'actual_benchmark_peak_allocated_GiB':[max(r['peak_allocated_gib'] for r in values) for values in measured],
        'actual_main_first3_peak_allocated_GiB':[max(r['peak_allocated_gib'] for r in values) for values in first],
        'actual_main_first3_compute_seconds':[[r['compute_seconds'] for r in values] for values in first],
        'actual_prepared_model_data_training_recipe_equal':True,
        'numerical_component_SHA256':entry['component_SHA256'],
        'driver_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'optimizer_updates_by_auditor':0,'GPU_used_by_auditor':False,
        'scope':'Actual launch/main startup only;not future OOM guarantee or quality improvement.',
        'model_adopted':False})
    print('PASS_ACTUAL_RAY_STARTUP',name,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--work-root',type=Path,required=True)
    p.add_argument('--arm',choices=list(ARMS),required=True)
    p.add_argument('--expected-sha',required=True)
    p.add_argument('--prepared-source-contract',type=Path)
    audit(p.parse_args())


if __name__=='__main__':main()
