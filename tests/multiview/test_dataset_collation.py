from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from pixal3d import datasets
from pixal3d.datasets.components import (
    MultiViewImageConditionedMixin,
    slice_condition_views,
)
from pixal3d.datasets.sparse_structure_latent import (
    MultiViewImageConditionedSparseStructureLatentView,
)
from pixal3d.datasets.structured_latent_shape import (
    MultiViewImageConditionedSLatShapeView,
)
from pixal3d.datasets.structured_latent_svpbr import (
    MultiViewImageConditionedSLatPbrView,
)


VIEW_KEYS = (
    "cond", "camera_angle_x", "camera_distance", "transform_matrix", "view_indices"
)


def condition_sample(offset):
    return {
        "cond": torch.arange(offset, offset + 24).reshape(8, 3, 1, 1).float(),
        "camera_angle_x": torch.arange(8).float() + offset,
        "camera_distance": torch.arange(8).float() + 2.0,
        "transform_matrix": torch.eye(4).repeat(8, 1, 1),
        "view_indices": torch.arange(8),
        "mesh_scale": torch.tensor(1.0),
    }


def selector():
    value = SimpleNamespace(min_condition_views=2, max_condition_views=6)
    value.select_batch_condition_views = MethodType(
        MultiViewImageConditionedMixin.select_batch_condition_views, value
    )
    return value


def test_slice_is_batchwide_and_does_not_mutate_sources():
    source = [condition_sample(0), condition_sample(100)]
    sliced = slice_condition_views(source, 4)
    for item in sliced:
        for key in VIEW_KEYS:
            assert item[key].shape[0] == 4
    assert source[0]["cond"].shape[0] == 8
    assert sliced[0]["mesh_scale"].ndim == 0


@pytest.mark.parametrize("num_views", [2, 6])
def test_dense_shape_and_pbr_collators_keep_one_endpoint_k(monkeypatch, num_views):
    monkeypatch.setattr(np.random, "randint", lambda low, high: num_views)
    dense = condition_sample(0)
    dense["x_0"] = torch.zeros(2, 2, 2, 2)
    shape = condition_sample(0)
    shape.update({
        "coords": torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int32),
        "feats": torch.zeros(2, 32),
    })
    pbr = condition_sample(0)
    pbr.update({
        "coords": torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int32),
        "pbr_feats": torch.zeros(2, 32),
        "shape_feats": torch.ones(2, 32),
    })
    dense_pack = MultiViewImageConditionedSparseStructureLatentView.collate_fn(
        selector(), [dense, dense]
    )
    shape_pack = MultiViewImageConditionedSLatShapeView.collate_fn(
        selector(), [shape, shape]
    )
    pbr_pack = MultiViewImageConditionedSLatPbrView.collate_fn(
        selector(), [pbr, pbr]
    )
    assert dense_pack["cond"].shape[:2] == (2, num_views)
    assert shape_pack["cond"].shape[:2] == (2, num_views)
    assert pbr_pack["cond"].shape[:2] == (2, num_views)
    assert shape_pack["x_0"].shape[0] == 2
    assert pbr_pack["concat_cond"].shape[0] == 2


def test_all_multiview_dataset_names_are_registered():
    expected = {
        "MultiViewImageConditionedSparseStructureLatentView":
            MultiViewImageConditionedSparseStructureLatentView,
        "MultiViewImageConditionedSLatShapeView":
            MultiViewImageConditionedSLatShapeView,
        "MultiViewImageConditionedSLatPbrView":
            MultiViewImageConditionedSLatPbrView,
    }
    for name, class_object in expected.items():
        assert getattr(datasets, name) is class_object
