import hashlib

import numpy as np
import pytest
import torch

from experiments.deformnext_retargeting.train_current_geometry_endpoint import training_order, load_pair, score
from experiments.deformnext_retargeting.cache_current_regions import SCHEMA


def test_original_hash_schedule_covers_all32_pairs_each_epoch_with_paired_noise():
    rows = [{'sample_id': f'pair{i:02d}'} for i in range(32)]
    for epoch in range(4):
        schedule = [training_order(rows, 101, update) for update in range(epoch*32, (epoch+1)*32)]
        assert len({row['sample_id'] for row, _, _ in schedule}) == 32
        assert all(one_epoch == epoch for _, one_epoch, _ in schedule)
    assert training_order(rows, 101, 29) == training_order(list(reversed(rows)), 101, 29)
    assert training_order(rows, 101, 29)[2] != training_order(rows, 202, 29)[2]


def test_training_rejects_calibration_before_opening_any_files():
    with pytest.raises(ValueError, match='role or identity'):
        load_pair({'role': 'calibration'}, {'role': 'calibration'}, role='train',
            with_query=True, consumer={}, device='cpu')


def test_no_query_loader_does_not_open_query_and_metrics_keep_empty_regions_explicit(tmp_path):
    consumer = {'sha256': 'fixed'}
    condition_path, label_path = tmp_path/'condition.pt', tmp_path/'label.npz'
    torch.save({'consumer': consumer, 'role': 'train', 'sample_id': 'pair', 'source_cluster_id': 'cluster',
        'target_proj': torch.zeros(1, 4096, 4), 'native_condition': {
            'cond': {'global': torch.zeros(1, 2, 4), 'proj': torch.zeros(1, 4096, 4)},
            'neg_cond': {'global': torch.zeros(1, 2, 4), 'proj': torch.zeros(1, 4096, 4)}}}, condition_path)
    one, empty = np.array([[1, 1, 1]], np.int32), np.zeros((0, 3), np.int32)
    np.savez(label_path, schema=SCHEMA, role='train', sample_id='pair', supervision_only=True, frame_contract='frame',
        source64=one, target64=one, surface64=one, free64=np.array([[3, 3, 3]], np.int32),
        matched_source_exclusive64=one, matched_source_exclusive32=one//2,
        stale_visible64=empty, stale_visible32=empty)
    ref = lambda p: {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    row = {'role': 'train', 'sample_id': 'pair', 'artifact': ref(condition_path),
        'geometry_query': {'path': str(tmp_path/'QUERY_MUST_NOT_OPEN'), 'sha256': 'unused'}}
    labels = {'role': 'train', 'sample_id': 'pair', 'source_cluster_id': 'cluster',
              'frame_contract': 'frame', 'artifact': ref(label_path)}
    pair = load_pair(row, labels, role='train', with_query=False, consumer=consumer, device='cpu')
    assert pair['geometry'] is None
    logits = torch.full((1, 1, 64, 64, 64), -1.)
    logits[0, 0, 1, 1, 1] = 1.
    values = score(logits, pair)
    assert values['voxel_iou'] == pytest.approx(1)
    assert (values['tp'], values['fp'], values['fn']) == (1, 0, 0)
    assert values['stale_leak64'] is None
    assert values['predicted_voxels_sha256'] == values['target_voxels_sha256']
