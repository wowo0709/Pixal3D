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


def test_pbr_decode_and_render_disable_training_autocast(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from experiments.deformnext_retargeting import quick_metrics as qm
    from pixal3d.modules import sparse as sp
    from pixal3d.utils import loss_utils

    active = [True]

    @contextmanager
    def autocast(device_type, enabled=True, **kwargs):
        previous = active[0]
        active[0] = enabled
        try:
            yield
        finally:
            active[0] = previous

    latent = torch.ones(1, 2, dtype=torch.bfloat16)
    pack = dict(concat_cond=latent, camera_angle_x=torch.ones(1, 1),
                camera_distance=torch.ones(1, 1), mesh_scale=torch.ones(1))
    monkeypatch.setattr(qm, '_sample', lambda *args: (latent, latent, pack))
    monkeypatch.setattr(sp, 'sparse_cat', lambda values: torch.cat(values))
    monkeypatch.setattr(torch, 'autocast', autocast)

    def decode(pbr, shape, batch_size):
        assert active[0] is False
        assert pbr.dtype == shape.dtype == torch.float32
        return [object(), object()]

    def render(*args):
        assert active[0] is False
        return {key: torch.ones(2, 1 if key == 'alpha' else 3, 16, 16)
                for key in ('shaded', 'base_color', 'alpha')}

    monkeypatch.setattr(qm, '_render_pbr_anchor', render)
    def metric(*args):
        assert active[0] is False
        return torch.tensor(1.0)

    for name in ('psnr', 'ssim', 'lpips'):
        monkeypatch.setattr(loss_utils, name, metric)
    metrics, panel = qm._pbr_metrics(
        None, SimpleNamespace(decode_latent=decode), [0], {'sampling_steps': 12}
    )
    assert metrics['gt_shape_conditioned_alpha_iou'] == pytest.approx(1.0)
    assert panel.shape == (16, 48, 3)
    assert active[0] is True
