import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from experiments.deformnext_retargeting.cascade_view_roles import file_sha256
from experiments.deformnext_retargeting.train_source_feature_teacher_pilot import TeacherPilotDataset


def dataset(tmp_path):
    # Exercise a real serialized teacher, native cache and world-v4 override;
    # construct only this one row, bypassing the fixed1536 catalog constructor.
    ds = TeacherPilotDataset.__new__(TeacherPilotDataset)
    ds.prefix = tmp_path/'mirror'; ds.root = tmp_path/'teacher'
    ds.projection_root = tmp_path/'world-v4'
    ds.root.mkdir(); ds.projection_root.mkdir(); (ds.prefix/'file4').mkdir(parents=True)
    row = dict(sample_id='source_d1', source_cluster_id='source', conditioning_input_sha256='bound-input',
        transport_cache='/file4/transport.npz', target_anchor='/file4/target.png',
        transform_matrix=np.tile(np.eye(4), (4, 1, 1)).tolist(), camera_angle_x=[.7]*4, mesh_scale=1.)
    ds.rows = [row]
    metadata = dict(schema=np.asarray('source-feature-teacher-TRAIN-v1'), split=np.asarray('train'),
        not_for_training=np.bool_(False), sample_id=np.asarray(row['sample_id']),
        source_cluster_id=np.asarray('source'), conditioning_input_sha256=np.asarray('bound-input'),
        camera_ids=np.asarray([9, 10]), teacher_source9_10_FP16=np.ones((2, 1024, 1024), np.float16))
    path = ds.root/'one.npz'; np.savez(path, **metadata)
    ds.index = [dict(relative_path=path.name, SHA256=file_sha256(path))]
    np.savez(ds.local(row['transport_cache']), conditioning_input_sha256=np.asarray('bound-input'),
        source_values=np.zeros((3, 1024, 1024), np.float16), source_valid=np.ones((3, 1024), bool),
        target_query=np.zeros((1024, 1024), np.float16),
        target_keys=np.concatenate([np.ones((1024, 1024)), np.full((1024, 1024), 9)], -1).astype(np.float16),
        source_keys=np.concatenate([np.ones((3, 1024, 1024)), np.full((3, 1024, 1024), 9)], -1).astype(np.float16),
        patch_grid=np.zeros((16, 16, 16), np.int64))
    np.savez(ds.projection_root/'source_d1.npz', schema=np.asarray('deformnext-anchor-projection-v2'),
        patch_grid=np.full((16, 16, 16), 17, np.int64))
    rgba = np.zeros((512, 512, 4), np.uint8); rgba[8, 8, 3] = 255
    Image.fromarray(rgba).save(ds.local(row['target_anchor']))
    return ds, metadata


def test_real_dataset_uses_FRAME_worldv4_input_alpha_and_teacher_separately(tmp_path):
    ds, _ = dataset(tmp_path); row = ds[0]
    assert row['patch_grid'].eq(17).all()  # Reject obsolete transport.patch_grid.
    assert row['target_keys'].eq(1).all() and row['source_keys'].eq(1).all()
    assert row['target_valid'].sum() == 1 and row['target_valid'][0]
    assert row['source_values'].eq(0).all() and row['teacher'].eq(1).all()


@pytest.mark.parametrize('field,value', [('split', 'validation'), ('sample_id', 'other'),
    ('conditioning_input_sha256', 'wrong'), ('source_cluster_id', 'heldout'), ('not_for_training', True)])
def test_dataset_rejects_a_foreign_teacher_even_with_a_valid_file_hash(tmp_path, field, value):
    ds, metadata = dataset(tmp_path); metadata[field] = np.asarray(value)
    path = ds.root/'one.npz'; np.savez(path, **metadata); ds.index[0]['SHA256'] = file_sha256(path)
    with pytest.raises(ValueError, match='TRAIN-only teacher identity'):
        ds[0]
