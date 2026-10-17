"""Original matching math stays cue-free with continuous patch RGB supervision."""
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.deformnext_retargeting.cache_task_rgb_transfer import SCHEMA
from experiments.deformnext_retargeting.train_task_rgb_transfer import load_record, make_model, sample_batch, forward, step


def example():
    rng = np.random.default_rng(6)
    return {'query_L12': rng.normal(size=(3, 1024)).astype(np.float32),
            'source_L12': rng.normal(size=(64, 1024)).astype(np.float32),
            'candidates': np.tile(np.arange(64), (3, 1)),
            'raw_scores': rng.normal(0, .01, size=(3, 64)).astype(np.float32),
            'eligible_queries': np.ones(3, bool),
            'source_rgb': rng.uniform(size=(16, 64, 3)).astype(np.float32),
            'target_rgb': rng.uniform(size=(16, 3, 3)).astype(np.float32)}


def test_both_models_share_original_head_initial_scores_loss_and_query_sampler():
    records = {'pair': example()}
    models = [make_model(arm, 42017, 'cpu') for arm in ['D0-PAIRWISE', 'D1-Q2C']]
    batches = [sample_batch(records, np.random.default_rng(42017), 'cpu') for _ in models]
    assert batches[0]['sampled_query_sha256'] == batches[1]['sampled_query_sha256']
    for name, value in models[0][0].fusion_head.state_dict().items():
        assert torch.equal(value, models[1][0].fusion_head.state_dict()[name])
    results = [step(model, optimizer, batch, check_initial=True)
               for (model, optimizer), batch in zip(models, batches)]
    assert results[0]['loss'] == results[1]['loss']
    assert all(r['initial_D0_D1_RAW_scores_exact'] for r in results)


def test_changing_rgb_changes_supervision_without_affecting_descriptor_scores():
    model, _ = make_model('D1-Q2C', 42017, 'cpu')
    batch = sample_batch({'pair': example()}, np.random.default_rng(3), 'cpu')
    original = forward(model, batch)['scores']
    changed = {**batch, 'source_rgb': batch['source_rgb'] * 0, 'target_rgb': batch['target_rgb'] + .2}
    assert torch.equal(original, forward(model, changed)['scores'])
    assert all(not p.requires_grad for p in batch['query'].values())


def test_training_loader_never_opens_evaluator_bank_and_rejects_calibration_values(tmp_path):
    path = tmp_path / 'input.npz'
    def save(role):
        np.savez(path, schema=SCHEMA, sample_id='pair', role=role, supervision_only=True,
            query_L12=np.zeros((1024, 1024), np.float16), source_L12=np.zeros((3072, 1024), np.float16),
            candidates=np.tile(np.arange(64), (1024, 1)), raw_scores=np.zeros((1024, 64), np.float32),
            eligible_queries=np.ones(1024, bool), label_source_rgb=np.zeros((16, 3072, 3), np.float16),
            label_target_rgb=np.zeros((16, 1024, 3), np.float16))
        return {'sample_id': 'pair', 'role': role,
                'artifact': {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
                'evaluation_artifact': {'path': str(tmp_path / 'MUST_NOT_OPEN'), 'sha256': 'unused'}}
    record = save('train')
    loaded = load_record(record, with_training_values=True)
    assert loaded['source_rgb'].shape == (16, 3072, 3)
    assert not (tmp_path / 'MUST_NOT_OPEN').exists()
    record = save('calibration')
    with pytest.raises(ValueError, match='forbidden in training'):
        load_record(record, with_training_values=True)
    loaded = load_record(record, with_training_values=False)
    assert 'source_rgb' not in loaded and 'target_rgb' not in loaded
