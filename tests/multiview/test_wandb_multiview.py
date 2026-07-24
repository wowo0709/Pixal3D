from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pixal3d.trainers import basic
from pixal3d.trainers.basic import BasicTrainer, batch_multiview_k
from pixal3d.trainers.flow_matching import flow_matching, sparse_flow_matching
from pixal3d.trainers.flow_matching.flow_matching import (
    ImageConditionedProjFlowMatchingCFGTrainer,
)
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    ImageConditionedProjMixin,
    format_multiview_metadata,
    make_anchor_marked_view_grid,
)
from pixal3d.trainers.flow_matching.sparse_flow_matching import (
    ImageConditionedProjSparseFlowMatchingCFGTrainer,
)


def test_ordered_grid_marks_only_anchor_and_keeps_view_order():
    condition = torch.zeros(1, 3, 3, 6, 6)
    condition[:, 0] = 0.1
    condition[:, 1] = 0.2
    condition[:, 2] = 0.3
    original = condition.clone()

    grid = make_anchor_marked_view_grid(condition, border=1)

    assert torch.equal(condition, original)
    assert grid.shape == (1, 3, 6, 18)
    assert torch.all(grid[:, 0, 0, :6] == 1.0)
    assert torch.all(grid[:, 1:, 0, :6] == 0.0)
    assert torch.allclose(
        grid[:, :, 1:-1, 7:11], torch.full((1, 3, 4, 4), 0.2)
    )
    assert torch.allclose(
        grid[:, :, 1:-1, 13:17], torch.full((1, 3, 4, 4), 0.3)
    )


def test_anchor_marked_grid_returns_legacy_4d_input_unchanged():
    condition = torch.zeros(2, 3, 6, 6)
    assert make_anchor_marked_view_grid(condition) is condition


def test_batch_multiview_k_returns_one_shared_k():
    data = [{"cond": torch.zeros(2, 4, 3, 8, 8)}]
    assert batch_multiview_k(data) == 4


def test_batch_multiview_k_returns_none_without_5d_conditioning():
    data = [{"cond": torch.zeros(2, 3, 8, 8)}]
    assert batch_multiview_k(data) is None


def test_batch_multiview_k_rejects_mixed_micro_batch_counts():
    data = [
        {"cond": torch.zeros(1, 2, 3, 8, 8)},
        {"cond": torch.zeros(1, 6, 3, 8, 8)},
    ]
    with pytest.raises(ValueError, match="share K"):
        batch_multiview_k(data)


def test_run_step_adds_exact_multiview_k_without_changing_existing_logs():
    trainer = object.__new__(BasicTrainer)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    trainer.model_params = [parameter]
    trainer.master_params = [parameter]
    trainer.optimizer = torch.optim.SGD([parameter], lr=0.1)
    trainer.training_models = {}
    trainer.world_size = 1
    trainer.mix_precision_mode = None
    trainer.mix_precision_dtype = torch.float32
    trainer.elastic_controller_config = None
    trainer.grad_clip = None
    trainer.lr_scheduler_config = None
    trainer.log_param_stats = False
    trainer.is_master = False
    trainer.debug = False
    trainer.training_losses = lambda **_: (
        {"loss": parameter.square(), "legacy_loss": parameter + 2},
        {"legacy_status": parameter + 3},
    )
    data = [{"cond": torch.zeros(1, 4, 3, 8, 8)}]

    step_log = BasicTrainer.run_step(trainer, data)

    assert step_log["multiview"] == {"k": 4}
    assert set(step_log["loss"]) == {"loss", "legacy_loss"}
    assert set(step_log["status"]) == {"legacy_status"}


def test_run_step_rejects_mixed_k_before_optimizer_or_ema_mutation():
    trainer = object.__new__(BasicTrainer)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    trainer.model_params = [parameter]
    trainer.master_params = [parameter]
    trainer.optimizer = torch.optim.Adam([parameter], lr=0.1)
    trainer.training_models = {}
    trainer.world_size = 1
    trainer.mix_precision_mode = None
    trainer.mix_precision_dtype = torch.float32
    trainer.elastic_controller_config = None
    trainer.grad_clip = None
    trainer.lr_scheduler_config = None
    trainer.log_param_stats = False
    trainer.is_master = True
    trainer.debug = False
    ema_calls = []
    trainer.update_ema = lambda: ema_calls.append(True)
    trainer.training_losses = lambda **_: (
        {"loss": parameter.square()},
        {"legacy_status": parameter + 3},
    )
    data = [
        {"cond": torch.zeros(1, 2, 3, 8, 8)},
        {"cond": torch.zeros(1, 6, 3, 8, 8)},
    ]
    parameter_before = parameter.detach().clone()

    with pytest.raises(ValueError, match="share K"):
        BasicTrainer.run_step(trainer, data)

    assert torch.equal(parameter, parameter_before)
    assert trainer.optimizer.state == {}
    assert ema_calls == []


class _FakeWandbRun:
    def __init__(self):
        self.calls = []

    def log(self, payload, step):
        self.calls.append((payload, step))


def test_save_logs_keeps_latest_multiview_k_exact_and_averages_legacy_scalars(
    tmp_path,
):
    trainer = object.__new__(BasicTrainer)
    trainer.log = [
        (1, {"loss": {"legacy": 1.0}, "multiview": {"k": 2}}),
        (2, {"loss": {"legacy": 3.0}, "multiview": {"k": 3}}),
    ]
    trainer._log_file = str(tmp_path / "train.log")
    trainer.writer = None
    trainer.wandb_run = _FakeWandbRun()
    trainer.step = 2

    BasicTrainer.save_logs(trainer)

    payload, step = trainer.wandb_run.calls[-1]
    assert step == 2
    assert payload["loss/legacy"] == 2.0
    assert payload["multiview/k"] == 3
    assert type(payload["multiview/k"]) is int


def test_metadata_contains_stage_dataset_sha_k_anchor_and_order():
    caption = format_multiview_metadata(
        "shape512", "ABO", "a" * 64, torch.tensor([1, 7, 4, 0])
    )
    assert caption == (
        "stage=shape512 dataset=ABO sha=" + "a" * 64
        + " K=4 anchor=view01 views=[1,7,4,0]"
    )


class _InitSink:
    def __init__(self, *args, **kwargs):
        self.sink_args = args
        self.sink_kwargs = kwargs


class _ProjectionInitHarness(ImageConditionedProjMixin, _InitSink):
    pass


def test_projection_mixin_stores_stage_without_forwarding_it():
    harness = _ProjectionInitHarness(
        "sentinel",
        image_cond_model={"name": "unused"},
        multiview_stage="pbr1024",
        legacy_option=True,
    )
    assert harness.multiview_stage == "pbr1024"
    assert harness.sink_args == ("sentinel",)
    assert harness.sink_kwargs == {"legacy_option": True}


def test_projection_visualization_preserves_legacy_image_and_adds_multiview_keys():
    harness = object.__new__(ImageConditionedProjMixin)
    single = torch.rand(2, 3, 6, 6)
    multiview = single[:, None].repeat(1, 3, 1, 1, 1)

    legacy = ImageConditionedProjMixin.vis_cond(harness, single)
    result = ImageConditionedProjMixin.vis_cond(harness, multiview)

    assert set(legacy) == {"image"}
    assert legacy["image"]["value"] is single
    assert set(result) == {"image", "input_views", "anchor"}
    assert torch.equal(result["image"]["value"], multiview[:, 0])
    assert torch.equal(result["anchor"]["value"], multiview[:, 0])
    assert result["input_views"]["value"].shape == (2, 3, 6, 18)
    assert all(value["type"] == "image" for value in result.values())


def test_wandb_offline_serializes_exact_scalar_and_image_keys(tmp_path, monkeypatch):
    import wandb

    monkeypatch.setenv("WANDB_MODE", "offline")
    run = wandb.init(
        project="pixal3d-multiview",
        name="serialization-test",
        dir=str(tmp_path),
        config={"stage": "ss64", "dataset": "ABO"},
    )
    run_dir = Path(run.dir).parent
    try:
        run.log({
            "multiview/k": 2,
            "samples/input_views": wandb.Image(
                np.zeros((8, 16, 3), dtype=np.uint8)
            ),
            "samples/anchor": wandb.Image(
                np.zeros((8, 8, 3), dtype=np.uint8)
            ),
        }, step=1)
    finally:
        run.finish()

    media = sorted(run_dir.glob("files/media/images/**/*.png"))
    assert len(media) == 2
    assert {path.name.split("_0_")[0] for path in media} == {
        "input_views",
        "anchor",
    }
    assert len(list(run_dir.glob("run-*.wandb"))) == 1
    assert wandb.run is None


def _snapshot_wandb_payload(tmp_path, monkeypatch, rendered_views):
    fake_run = _FakeWandbRun()
    monkeypatch.setattr(
        basic.wandb,
        "Image",
        lambda image, caption: {"image": image, "caption": caption},
    )
    trainer = object.__new__(BasicTrainer)
    trainer.log = [(7, {"multiview": {"k": 4}})]
    trainer._log_file = str(tmp_path / "train.log")
    trainer.writer = None
    trainer.wandb_run = fake_run
    trainer.step = 7
    trainer.is_master = True
    trainer.world_size = 1
    trainer.output_dir = str(tmp_path)
    trainer.dataset = SimpleNamespace(
        value_range=(0.0, 1.0),
        visualize_sample=lambda sample: rendered_views[sample],
    )
    trainer.mix_precision_mode = None
    trainer.mix_precision_dtype = torch.float32
    trainer.run_snapshot = lambda *_args, **_kwargs: {
        "image": {
            "value": torch.zeros(2, 3, 8, 8),
            "type": "image",
        },
        "sample": {
            "value": "generated",
            "type": "sample",
        },
        "sample_gt": {
            "value": "ground_truth",
            "type": "sample",
        },
    }

    BasicTrainer.save_logs(trainer)
    BasicTrainer.snapshot(trainer, suffix="wiring", num_samples=2, batch_size=2)

    scalar_payload = fake_run.calls[0][0]
    assert scalar_payload["multiview/k"] == 4
    return fake_run.calls[-1][0]


def test_shape_snapshot_wandb_uses_anchor_view_names(tmp_path, monkeypatch):
    rendered_views = {
        "generated": {"anchor_view": torch.zeros(2, 3, 8, 8)},
        "ground_truth": {"anchor_view": torch.ones(2, 3, 8, 8)},
    }

    image_payload = _snapshot_wandb_payload(
        tmp_path, monkeypatch, rendered_views
    )

    assert "samples/sample_anchor_view" in image_payload
    assert "samples/sample_gt_anchor_view" in image_payload
    assert "samples/combined_anchor_views" in image_payload
    assert "samples/sample_gt_view" not in image_payload
    assert "samples/sample_gt_gt_view" not in image_payload
    assert "samples/combined_views" not in image_payload


def test_pbr_snapshot_wandb_uses_anchor_view_attribute_names(
    tmp_path, monkeypatch
):
    rendered_views = {
        "generated": {
            "anchor_view_base_color": torch.zeros(2, 3, 8, 8),
        },
        "ground_truth": {
            "anchor_view_base_color": torch.ones(2, 3, 8, 8),
        },
    }

    image_payload = _snapshot_wandb_payload(
        tmp_path, monkeypatch, rendered_views
    )

    assert "samples/sample_anchor_view_base_color" in image_payload
    assert "samples/sample_gt_anchor_view_base_color" in image_payload
    assert "samples/combined_anchor_views_base_color" in image_payload
    assert "samples/sample_gt_view_base_color" not in image_payload
    assert "samples/sample_gt_gt_view_base_color" not in image_payload
    assert "samples/combined_views_base_color" not in image_payload


class _SnapshotDataset:
    pass


class _Sampler:
    def sample(self, _model, *, noise, **_kwargs):
        return SimpleNamespace(samples=noise)


class _FakeSparse:
    def __init__(self, feats=None):
        self.feats = (
            torch.zeros(2, 1, dtype=torch.float32)
            if feats is None
            else feats
        )
        self.coords = torch.tensor(
            [[0, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.int32
        )

    def __getitem__(self, _index):
        return self

    def replace(self, feats):
        return _FakeSparse(feats)


def _metadata_batch(include_view_indices=True):
    batch = {
        "x_0": torch.ones(2, 1, 2, 2),
        "cond": torch.zeros(2, 2, 3, 4, 4),
        "_dataset_name": ["ABO", "Objaverse"],
        "_sha256": ["a" * 64, "b" * 64],
    }
    if include_view_indices:
        batch["view_indices"] = torch.tensor([[1, 7], [0, 3]])
    return batch


def _configure_snapshot_trainer(trainer, stage):
    trainer.step = 5
    trainer.dataset = _SnapshotDataset()
    trainer.multiview_stage = stage
    trainer.models = {"denoiser": object()}
    trainer.get_sampler = lambda: _Sampler()
    trainer.vis_cond = lambda **_kwargs: {}
    trainer.get_inference_cond = lambda **_kwargs: {}


def test_dense_snapshot_uses_exact_multiview_metadata(monkeypatch):
    data = _metadata_batch()
    monkeypatch.setattr(
        flow_matching, "DataLoader", lambda *_args, **_kwargs: [data]
    )
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    trainer = object.__new__(ImageConditionedProjFlowMatchingCFGTrainer)
    _configure_snapshot_trainer(trainer, "ss64")

    result = ImageConditionedProjFlowMatchingCFGTrainer.run_snapshot(
        trainer, num_samples=2, batch_size=2
    )

    assert result["_metadata"] == [
        format_multiview_metadata("ss64", "ABO", "a" * 64, [1, 7]),
        format_multiview_metadata("ss64", "Objaverse", "b" * 64, [0, 3]),
    ]


def test_dense_snapshot_preserves_legacy_metadata_without_view_indices(monkeypatch):
    data = _metadata_batch(include_view_indices=False)
    monkeypatch.setattr(
        flow_matching, "DataLoader", lambda *_args, **_kwargs: [data]
    )
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    trainer = object.__new__(ImageConditionedProjFlowMatchingCFGTrainer)
    _configure_snapshot_trainer(trainer, "ss64")

    result = ImageConditionedProjFlowMatchingCFGTrainer.run_snapshot(
        trainer, num_samples=2, batch_size=2
    )

    assert result["_metadata"] == ["ABO/" + "a" * 64, "Objaverse/" + "b" * 64]


def test_dense_snapshot_bounds_metadata_to_partial_batch(monkeypatch):
    data = _metadata_batch()
    monkeypatch.setattr(
        flow_matching, "DataLoader", lambda *_args, **_kwargs: [data]
    )
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    trainer = object.__new__(ImageConditionedProjFlowMatchingCFGTrainer)
    _configure_snapshot_trainer(trainer, "ss64")

    result = ImageConditionedProjFlowMatchingCFGTrainer.run_snapshot(
        trainer, num_samples=1, batch_size=2
    )

    assert result["_metadata"] == [
        format_multiview_metadata("ss64", "ABO", "a" * 64, [1, 7])
    ]


def _run_sparse_snapshot(monkeypatch, data, *, num_samples=2):
    data["x_0"] = _FakeSparse()
    monkeypatch.setattr(
        sparse_flow_matching, "DataLoader", lambda *_args, **_kwargs: [data]
    )
    monkeypatch.setattr(
        sparse_flow_matching,
        "recursive_to_device",
        lambda value, _device: value,
    )
    monkeypatch.setattr(
        sparse_flow_matching.sp, "sparse_cat", lambda values: values[0]
    )
    trainer = object.__new__(ImageConditionedProjSparseFlowMatchingCFGTrainer)
    _configure_snapshot_trainer(trainer, "pbr1024")
    return ImageConditionedProjSparseFlowMatchingCFGTrainer.run_snapshot(
        trainer, num_samples=num_samples, batch_size=2
    )


def test_projection_sparse_snapshot_uses_exact_multiview_metadata(monkeypatch):
    result = _run_sparse_snapshot(monkeypatch, _metadata_batch())
    assert result["_metadata"] == [
        format_multiview_metadata("pbr1024", "ABO", "a" * 64, [1, 7]),
        format_multiview_metadata("pbr1024", "Objaverse", "b" * 64, [0, 3]),
    ]


def test_projection_sparse_snapshot_preserves_legacy_metadata(monkeypatch):
    result = _run_sparse_snapshot(
        monkeypatch, _metadata_batch(include_view_indices=False)
    )
    assert result["_metadata"] == ["ABO/" + "a" * 64, "Objaverse/" + "b" * 64]


def test_projection_sparse_snapshot_bounds_metadata_to_partial_batch(monkeypatch):
    data = _metadata_batch()
    data["_dataset_name"] = data["_dataset_name"][:1]
    data["_sha256"] = data["_sha256"][:1]
    data["view_indices"] = data["view_indices"][:1]

    result = _run_sparse_snapshot(monkeypatch, data, num_samples=3)

    assert result["_metadata"] == [
        format_multiview_metadata("pbr1024", "ABO", "a" * 64, [1, 7])
    ]
