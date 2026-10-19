"""Validate loader proof declarations; synthetic metadata is not model evidence."""
import hashlib
import json
from pathlib import Path

import pytest

from experiments.deformnext_retargeting.loader_runtime_contract import validate_loader_runtime


def config(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    rows = []
    for rank in [0, 1]:
        for workers in [8, 2]:
            rows.append({'rank': rank, 'workers': workers, 'all480_batch_index_SHA256': 'indices',
                'all480_consumed_sampler_state_SHA256': 'sampler', 'final_main_CPU_RNG_SHA256': 'rng'})
    value = {'status': 'PASS_CPU_DATASET_DETERMINISM_AND_WORKER8_VS2_ORDER_PREFLIGHT',
        'dataset_samples': 9149, 'stochastic_calls_in_dataset_source': [],
        'both_ranks_all480_batch_indices_consumed_sampler_and_main_CPU_RNG_exact': True,
        'optimizer_updates': 0, 'GPU_used': False, 'index_only_DataLoader_experiments': rows,
        'real_item_repeat_checks': [{'CPU_torch_numpy_python_RNG_unchanged': True}] * 2,
        'dataset_source_SHA256': hashlib.sha256((repo / 'pixal3d/datasets/deformnext.py').read_bytes()).hexdigest(),
        'sampler_source_SHA256': hashlib.sha256((repo / 'pixal3d/utils/data_utils.py').read_bytes()).hexdigest()}
    path = tmp_path / 'synthetic-proof.json'; path.write_text(json.dumps(value))
    cfg = {'trainer': {'args': {'num_workers': 2}}, 'experiment': {'loader_runtime_contract': {
        'reference_workers': 8, 'workers': 2, 'proof': str(path),
        'proof_SHA256': hashlib.sha256(path.read_bytes()).hexdigest()}}}
    return cfg, value, path


def test_proved_runtime_and_unchanged_original_workers_are_allowed(tmp_path):
    cfg, _, _ = config(tmp_path); validate_loader_runtime(cfg)
    validate_loader_runtime({'trainer': {'args': {'num_workers': 8}}, 'experiment': {}})


@pytest.mark.parametrize('change', ['hash', 'order', 'source', 'stochastic', 'worker_metadata', 'real_inputs'])
def test_mismatched_runtime_or_proof_is_rejected(tmp_path, change):
    cfg, value, path = config(tmp_path)
    if change == 'hash':
        path.write_text('{}')
    elif change == 'worker_metadata':
        cfg['trainer']['args']['num_workers'] = 8
    else:
        if change == 'order': value['index_only_DataLoader_experiments'][-1]['all480_batch_index_SHA256'] = 'changed'
        elif change == 'source': value['dataset_source_SHA256'] = 'changed'
        elif change == 'real_inputs': value['real_item_repeat_checks'] = []
        else: value['stochastic_calls_in_dataset_source'] = ['random.random']
        path.write_text(json.dumps(value));cfg['experiment']['loader_runtime_contract']['proof_SHA256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError): validate_loader_runtime(cfg)
