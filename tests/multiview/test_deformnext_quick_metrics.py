import pytest
import torch

from experiments.deformnext_retargeting.quick_metrics import (
    binary_metrics,
    stratified_indices,
)


class _Dataset:
    rows = [
        {"category": "B"},
        {"category": "A"},
        {"category": "B"},
        {"category": "A"},
        {"category": "C"},
    ]

    def __len__(self):
        return len(self.rows)


def test_stratified_indices_are_stable_category_round_robin():
    assert stratified_indices(_Dataset(), 4) == [1, 0, 4, 3]


def test_binary_metrics_report_overlap_without_background_accuracy():
    prediction = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
    target = torch.tensor([1, 0, 1, 0], dtype=torch.bool)

    metrics = binary_metrics(prediction, target)

    assert metrics == pytest.approx({
        "voxel_iou": 1 / 3,
        "voxel_precision": 1 / 2,
        "voxel_recall": 1 / 2,
        "voxel_f1": 1 / 2,
    })
