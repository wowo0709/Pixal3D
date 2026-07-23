import torch

from pixal3d.trainers.flow_matching.flow_matching import (
    ImageConditionedProjFlowMatchingCFGTrainer,
)
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    anchor_camera_value,
    anchor_condition_image,
)
from pixal3d.trainers.flow_matching.sparse_flow_matching import (
    ImageConditionedProjSparseFlowMatchingCFGTrainer,
)


def test_anchor_condition_handles_single_and_multiview():
    single = torch.zeros(2, 3, 4, 4)
    multi = torch.arange(2 * 6 * 3 * 4 * 4).reshape(2, 6, 3, 4, 4)
    assert anchor_condition_image(single) is single
    assert torch.equal(anchor_condition_image(multi), multi[:, 0])


def test_anchor_camera_handles_scalar_and_view_vectors():
    single = torch.tensor([2.0, 3.0])
    multi = torch.tensor([[2.0, 4.0], [3.0, 5.0]])
    assert anchor_camera_value(single) is single
    assert torch.equal(anchor_camera_value(multi), torch.tensor([2.0, 3.0]))


class _ShapeStyleRecordingDataset:
    def visualize_sample(
        self,
        x_0,
        *,
        camera_angle_x=None,
        camera_distance=None,
        mesh_scale=None,
    ):
        self.received = {
            "x_0": x_0,
            "camera_angle_x": camera_angle_x,
            "camera_distance": camera_distance,
            "mesh_scale": mesh_scale,
        }
        return self.received


class _PbrStyleRecordingDataset:
    def visualize_sample(self, sample):
        self.received = sample
        return sample


def _multiview_snapshot_sample():
    return {
        "x_0": object(),
        "concat_cond": object(),
        "camera_angle_x": torch.tensor([[0.7, 0.8], [0.9, 1.0]]),
        "camera_distance": torch.tensor([[2.0, 2.5], [3.0, 3.5]]),
        "mesh_scale": torch.tensor([1.0, 1.5]),
    }


def _assert_anchor_camera_snapshot(received, caller_sample):
    torch.testing.assert_close(
        received["camera_angle_x"], torch.tensor([0.7, 0.9])
    )
    torch.testing.assert_close(
        received["camera_distance"], torch.tensor([2.0, 3.0])
    )
    assert received["mesh_scale"] is caller_sample["mesh_scale"]
    assert received["x_0"] is caller_sample["x_0"]
    torch.testing.assert_close(
        caller_sample["camera_angle_x"],
        torch.tensor([[0.7, 0.8], [0.9, 1.0]]),
    )
    torch.testing.assert_close(
        caller_sample["camera_distance"],
        torch.tensor([[2.0, 2.5], [3.0, 3.5]]),
    )


def test_dense_projection_visualize_sample_anchors_camera_without_mutation():
    dataset = _ShapeStyleRecordingDataset()
    trainer = object.__new__(ImageConditionedProjFlowMatchingCFGTrainer)
    trainer.dataset = dataset
    caller_sample = _multiview_snapshot_sample()

    trainer.visualize_sample(caller_sample)

    _assert_anchor_camera_snapshot(dataset.received, caller_sample)


def test_sparse_shape_visualize_sample_anchors_camera_without_mutation():
    dataset = _ShapeStyleRecordingDataset()
    trainer = object.__new__(ImageConditionedProjSparseFlowMatchingCFGTrainer)
    trainer.dataset = dataset
    caller_sample = _multiview_snapshot_sample()

    trainer.visualize_sample(caller_sample)

    _assert_anchor_camera_snapshot(dataset.received, caller_sample)


def test_sparse_pbr_visualize_sample_anchors_camera_without_mutation():
    dataset = _PbrStyleRecordingDataset()
    trainer = object.__new__(ImageConditionedProjSparseFlowMatchingCFGTrainer)
    trainer.dataset = dataset
    caller_sample = _multiview_snapshot_sample()

    trainer.visualize_sample(caller_sample)

    assert dataset.received is not caller_sample
    _assert_anchor_camera_snapshot(dataset.received, caller_sample)
    assert dataset.received["concat_cond"] is caller_sample["concat_cond"]
