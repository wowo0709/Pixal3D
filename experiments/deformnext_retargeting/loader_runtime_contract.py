"""Allow a smaller fresh-run loader only with its real data/order proof."""
import hashlib
import json
from pathlib import Path


def validate_loader_runtime(cfg):
    workers = cfg['trainer']['args'].get('num_workers', 8)
    if workers == 8:
        if cfg['experiment'].get('loader_runtime_contract') is not None:
            raise ValueError('loader metadata must match the actual worker count')
        return
    declared = cfg['experiment'].get('loader_runtime_contract', {})
    if workers != 2 or declared.get('reference_workers') != 8 or declared.get('workers') != 2:
        raise ValueError('fresh loader must retain8 workers or use the proved2-worker runtime')
    path = Path(declared['proof'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != declared.get('proof_SHA256'):
        raise ValueError('actual loader data/order proof changed')
    proof = json.loads(path.read_text())
    if (proof.get('status') != 'PASS_CPU_DATASET_DETERMINISM_AND_WORKER8_VS2_ORDER_PREFLIGHT'
            or proof.get('dataset_samples') != 9149 or proof.get('stochastic_calls_in_dataset_source') != []
            or proof.get('both_ranks_all480_batch_indices_consumed_sampler_and_main_CPU_RNG_exact') is not True
            or proof.get('optimizer_updates') != 0 or proof.get('GPU_used') is not False):
        raise ValueError('real9149 deterministic data and both-rank480-update order proof required')
    checks = proof.get('real_item_repeat_checks', [])
    if len(checks) != 2 or not all(row.get('CPU_torch_numpy_python_RNG_unchanged') is True for row in checks):
        raise ValueError('actual repeated input checks must preserve CPU RNG')
    repo = Path(__file__).resolve().parents[2]
    for source, field in [('pixal3d/datasets/deformnext.py', 'dataset_source_SHA256'),
                          ('pixal3d/utils/data_utils.py', 'sampler_source_SHA256')]:
        if hashlib.sha256((repo / source).read_bytes()).hexdigest() != proof.get(field):
            raise ValueError('dataset/sampler changed after loader parity proof')
    rows = {(row['rank'], row['workers']): row for row in proof['index_only_DataLoader_experiments']}
    if set(rows) != {(rank, worker) for rank in [0, 1] for worker in [8, 2]}:
        raise ValueError('both explicit ranks and worker settings required')
    for rank in [0, 1]:
        for key in ['all480_batch_index_SHA256', 'all480_consumed_sampler_state_SHA256', 'final_main_CPU_RNG_SHA256']:
            if rows[rank, 8][key] != rows[rank, 2][key]:
                raise ValueError('fresh loader changes consumed data order or CPU RNG')
