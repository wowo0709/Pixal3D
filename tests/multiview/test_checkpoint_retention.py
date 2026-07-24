from pathlib import Path

import pytest
import torch

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


def test_failed_save_keeps_all_previous_complete_checkpoints(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, step=6)
    ckpt_dir = tmp_path / "ckpts"
    for step in range(1, 6):
        _write_complete_checkpoint(ckpt_dir, step)
    original_save = torch.save

    def fail_on_new_misc(value, path, *args, **kwargs):
        if Path(path).name == "misc_step0000006.pt":
            raise OSError("simulated save failure")
        return original_save(value, path, *args, **kwargs)

    monkeypatch.setattr(torch, "save", fail_on_new_misc)

    with pytest.raises(OSError, match="simulated save failure"):
        trainer.save()

    for step in range(1, 6):
        assert all(path.exists() for path in _checkpoint_paths(ckpt_dir, step))


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
