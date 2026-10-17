from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from experiments.deformnext_retargeting.cache_predicted_geometry import rgb_packs, noise_seed


class RGBOnlyDataset:
    image_size = 512
    collate_fn = staticmethod(default_collate)
    _resolve = staticmethod(Path)

    def __init__(self):
        self.rows = [dict(sample_id='pair', latent_geometry_schema='deformnext-m1-render-world-v4',
            conditioning_camera_ids=[8, 9, 10, 11], transform_matrix=np.tile(np.eye(4), (4, 1, 1)).tolist(),
            source_anchor_transform_matrix=np.eye(4).tolist(), camera_angle_x=[.6] * 4,
            source_anchor_camera_angle_x=.6, source_anchor_rgb='M0_cam08.png')]

    def __getitem__(self, index):
        raise AssertionError('GT latent dataset path must never be entered')

    def _common(self, row):
        return dict(cond=torch.stack([torch.full((3, 512, 512), i/4) for i in range(4)]),
            camera_angle_x=torch.tensor(row['camera_angle_x']), camera_distance=torch.ones(4) * 2,
            transform_matrix=torch.tensor(row['transform_matrix']), mesh_scale=torch.tensor(1.),
            view_indices=torch.arange(4), _dataset_name='DeformNeXt', _sha256='pair')


def test_query_input_is_four_source_rgb_and_one_target_rgb_with_identical_anchor_frame():
    ds = RGBOnlyDataset()
    loaded = []
    def image(path, size):
        loaded.append((path, size))
        return torch.ones(3, size, size)
    original, desired = rgb_packs(ds, [0], image)
    assert loaded == [(Path('M0_cam08.png'), 512)]
    assert original['cond'].shape == (1, 4, 3, 512, 512)
    assert desired['cond'].shape == (1, 1, 3, 512, 512)
    assert original['cond'][0, 0].eq(1).all()
    assert desired['cond'][0, 0].eq(0).all()
    assert original['cond'][0, 1, 0, 0, 0] == .25
    assert torch.equal(original['transform_matrix'][:, :1], desired['transform_matrix'])
    assert torch.equal(original['mesh_scale'], desired['mesh_scale'])
    assert 'x_0' not in original and 'x_0' not in desired
    assert desired['camera_angle_x'].shape == (1, 1)


def test_geometry_queries_reject_camera_frame_drift_and_transport_inputs():
    ds = RGBOnlyDataset()
    ds.rows[0]['source_anchor_transform_matrix'][0][3] = .01
    with pytest.raises(ValueError, match='camera frame'):
        rgb_packs(ds, [0], lambda *a: None)
    ds = RGBOnlyDataset()
    original = ds._common
    ds._common = lambda row: {**original(row), 'x_0': torch.zeros(8, 16, 16, 16)}
    with pytest.raises(ValueError, match='unexpected transport or target'):
        rgb_packs(ds, [0], lambda *a: None)


def test_query_noise_is_pair_specific_and_independent_of_label_random_state():
    state = deepcopy(np.random.get_state())
    seed = noise_seed('pair')
    assert seed == noise_seed('pair') and seed != noise_seed('other')
    assert 0 <= seed < 2**63-1
    assert all(np.array_equal(a, b) for a, b in zip(state, np.random.get_state()))
