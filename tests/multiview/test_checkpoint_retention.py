from pathlib import Path
import threading

import pytest
import torch

import pixal3d.trainers.basic as basic
from pixal3d.trainers.basic import BasicTrainer


class _Stateful:
    def state_dict(self):
        return {}


def _checkpoint_paths(ckpt_dir, step):
    suffix = f"_step{step:07d}.pt"
    return [
        ckpt_dir / f"denoiser{suffix}",
        ckpt_dir / f"denoiser_ema0.9{suffix}",
        ckpt_dir / f"misc{suffix}",
    ]


def _write_complete_checkpoint(ckpt_dir, step):
    for path in _checkpoint_paths(ckpt_dir, step):
        torch.save({}, path)


def _make_trainer(tmp_path, step, max_checkpoints=5):
    trainer = object.__new__(BasicTrainer)
    trainer.is_master = True
    trainer.output_dir = str(tmp_path)
    trainer.step = step
    trainer.max_checkpoints = max_checkpoints
    trainer.master_params = []
    trainer.ema_params = [[]]
    trainer.ema_rate = [0.9]
    trainer.optimizer = _Stateful()
    trainer.data_sampler = _Stateful()
    trainer.mix_precision_mode = None
    trainer.mix_precision_dtype = None
    trainer.lr_scheduler_config = None
    trainer.elastic_controller_config = None
    trainer.grad_clip = None
    trainer._master_params_to_state_dicts = lambda params: {
        "denoiser": {"weight": torch.tensor(1)}
    }
    (tmp_path / "ckpts").mkdir()
    return trainer


def test_retention_keeps_latest_five_complete_checkpoints(tmp_path):
    trainer = _make_trainer(tmp_path, step=6)
    ckpt_dir = tmp_path / "ckpts"
    for step in range(1, 6):
        _write_complete_checkpoint(ckpt_dir, step)

    trainer.save()

    assert not any(path.exists() for path in _checkpoint_paths(ckpt_dir, 1))
    for step in range(2, 7):
        assert all(path.exists() for path in _checkpoint_paths(ckpt_dir, step))


def test_failed_misc_save_does_not_publish_a_retention_marker(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, step=6)
    ckpt_dir = tmp_path / "ckpts"
    for step in range(1, 6):
        _write_complete_checkpoint(ckpt_dir, step)

    def write_partial_misc_then_fail(value, path, *args, **kwargs):
        filename = Path(path).name
        if filename == "misc_step0000006.pt" or filename.startswith(
            ".misc_step0000006.pt."
        ):
            Path(path).write_bytes(b"partial")
            raise OSError("simulated save failure")
        torch.serialization.save(value, path, *args, **kwargs)

    monkeypatch.setattr(torch, "save", write_partial_misc_then_fail)

    with pytest.raises(OSError, match="simulated save failure"):
        trainer.save()

    trainer.step = 7
    trainer.save()

    assert not (ckpt_dir / "misc_step0000006.pt").exists()
    assert not list(ckpt_dir.glob("*.tmp"))
    for step in range(2, 6):
        assert all(path.exists() for path in _checkpoint_paths(ckpt_dir, step))
    assert all(path.exists() for path in _checkpoint_paths(ckpt_dir, 7))


def test_duplicate_completed_step_is_a_noop_without_reserializing_checkpoint(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, step=6)
    ckpt_dir = tmp_path / "ckpts"
    _write_complete_checkpoint(ckpt_dir, 6)
    original_model = torch.load(ckpt_dir / "denoiser_step0000006.pt", weights_only=True)
    save_calls = []

    def record_duplicate_save_attempt(*args, **kwargs):
        save_calls.append(args)

    monkeypatch.setattr(torch, "save", record_duplicate_save_attempt)

    trainer.save()

    assert save_calls == []
    assert torch.load(ckpt_dir / "denoiser_step0000006.pt", weights_only=True) == original_model
    assert all(path.exists() for path in _checkpoint_paths(ckpt_dir, 6))


def test_abort_after_scheduled_retained_save_reaches_abort_flow_without_reserializing(
    tmp_path, monkeypatch, capsys
):
    trainer = _make_trainer(tmp_path, step=6)
    trainer.mix_precision_mode = "inflat_all"
    trainer.mix_precision_dtype = torch.float16
    trainer.log_scale = -1
    trainer.world_size = 2
    trainer.save()
    save_calls = []
    log_calls = []
    barriers = []

    def record_duplicate_save_attempt(*args, **kwargs):
        save_calls.append(args)

    monkeypatch.setattr(torch, "save", record_duplicate_save_attempt)
    monkeypatch.setattr(trainer, "save_logs", lambda: log_calls.append(trainer.step))
    monkeypatch.setattr(basic.dist, "barrier", lambda: barriers.append(True))

    with pytest.raises(ValueError, match="ABORT: log_scale"):
        trainer.check_abort()

    assert "ABORT: log_scale" in capsys.readouterr().out
    assert save_calls == []
    assert log_calls == [6]
    assert barriers == [True]


def test_incomplete_step_is_not_a_retention_marker(tmp_path):
    trainer = _make_trainer(tmp_path, step=7)
    ckpt_dir = tmp_path / "ckpts"
    for step in range(1, 6):
        _write_complete_checkpoint(ckpt_dir, step)
    for path in _checkpoint_paths(ckpt_dir, 6)[:2]:
        torch.save({}, path)

    trainer.save()

    assert not any(path.exists() for path in _checkpoint_paths(ckpt_dir, 1))
    assert all(path.exists() for path in _checkpoint_paths(ckpt_dir, 6)[:2])
    for step in (2, 3, 4, 5, 7):
        assert all(path.exists() for path in _checkpoint_paths(ckpt_dir, step))


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5, "5"])
def test_max_checkpoints_requires_a_positive_integer(invalid, tmp_path):
    with pytest.raises(ValueError, match="max_checkpoints"):
        BasicTrainer(
            models={},
            dataset=None,
            output_dir=str(tmp_path),
            load_dir=None,
            step=None,
            max_steps=1,
            batch_size=1,
            optimizer={},
            max_checkpoints=invalid,
        )


def test_legacy_non_blocking_save_starts_threads_and_never_prunes(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, step=6, max_checkpoints=None)
    ckpt_dir = tmp_path / "ckpts"
    for step in range(1, 6):
        _write_complete_checkpoint(ckpt_dir, step)
    started = []

    class RecordingThread:
        def __init__(self, *, target, args):
            self.target = target
            self.args = args

        def start(self):
            started.append(self.args[1])

    monkeypatch.setattr(threading, "Thread", RecordingThread)
    monkeypatch.setattr(
        trainer,
        "_prune_checkpoints",
        lambda: pytest.fail("legacy save must not prune checkpoints"),
    )

    trainer.save()

    assert len(started) == 3
    assert all(path.exists() for path in _checkpoint_paths(ckpt_dir, 1))


def test_legacy_blocking_save_uses_synchronous_torch_save(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, step=6, max_checkpoints=None)
    saved_paths = []

    def record_save(value, path, *args, **kwargs):
        saved_paths.append(Path(path).name)

    monkeypatch.setattr(torch, "save", record_save)
    monkeypatch.setattr(
        threading,
        "Thread",
        lambda *args, **kwargs: pytest.fail("blocking save must not start a thread"),
    )
    monkeypatch.setattr(
        trainer,
        "_prune_checkpoints",
        lambda: pytest.fail("legacy save must not prune checkpoints"),
    )

    trainer.save(non_blocking=False)

    assert saved_paths == [
        "denoiser_step0000006.pt",
        "denoiser_ema0.9_step0000006.pt",
        "misc_step0000006.pt",
    ]
