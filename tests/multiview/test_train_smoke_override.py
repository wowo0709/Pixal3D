from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from easydict import EasyDict as edict
import pytest

from train import apply_smoke_overrides


REPO_ROOT = Path(__file__).resolve().parents[2]
RESOLVED_CONFIG_MARKER = f"Config:\n{'=' * 80}\n"


def test_ten_step_smoke_logs_every_step_samples_five_and_ten_and_saves_ten():
    config = edict({"trainer": {"args": {
        "max_steps": 1_000_000,
        "i_log": 5,
        "i_sample": 250,
        "i_save": 1000,
        "snapshot_batch_size": 8,
        "snapshot_num_samples": 64,
        "num_workers": 14,
        "prefetch_data": True,
    }}})
    apply_smoke_overrides(config, 10)
    assert config.trainer.args.max_steps == 10
    assert config.trainer.args.i_log == 1
    assert config.trainer.args.i_sample == 5
    assert config.trainer.args.i_save == 10
    assert config.trainer.args.snapshot_batch_size == 1
    assert config.trainer.args.snapshot_num_samples == 1
    assert config.trainer.args.num_workers == 0
    assert config.trainer.args.prefetch_data is False


def test_absent_smoke_steps_is_a_true_no_op():
    config = edict({"trainer": {"args": {
        "max_steps": 1_000_000,
        "i_log": 5,
        "i_sample": 250,
        "i_save": 1000,
    }}})
    original = deepcopy(config)

    result = apply_smoke_overrides(config, None)

    assert result is config
    assert config == original


def test_invalid_direct_smoke_steps_are_rejected():
    config = edict({"trainer": {"args": {}}})

    with pytest.raises(ValueError, match="smoke_steps must be 1 or 10"):
        apply_smoke_overrides(config, 5)


def test_cli_smoke_steps_override_conflicting_json_before_resolved_config(tmp_path):
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps({
        "node_rank": 0,
        "smoke_steps": 1,
        "trainer": {"args": {
            "max_steps": 1_000_000,
            "i_log": 5,
            "i_sample": 250,
            "i_save": 1000,
            "snapshot_batch_size": 8,
            "snapshot_num_samples": 64,
            "num_workers": 14,
            "prefetch_data": True,
        }},
    }))

    result = subprocess.run(
        [
            sys.executable,
            "train.py",
            "--config",
            str(config_path),
            "--output_dir",
            str(config_path),
            "--smoke_steps",
            "10",
            "--num_gpus",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode != 0
    assert "FileExistsError" in result.stderr
    assert RESOLVED_CONFIG_MARKER in result.stdout
    resolved = json.loads(result.stdout.split(RESOLVED_CONFIG_MARKER, 1)[1])
    assert resolved["smoke_steps"] == 1
    assert resolved["trainer"]["args"] == {
        "max_steps": 10,
        "i_log": 1,
        "i_sample": 5,
        "i_save": 10,
        "snapshot_batch_size": 1,
        "snapshot_num_samples": 1,
        "num_workers": 0,
        "prefetch_data": False,
    }


def test_one_step_online_gate_samples_and_saves_step_one():
    config = edict({"trainer": {"args": {
        "max_steps": 1_000_000,
        "i_log": 5,
        "i_sample": 250,
        "i_save": 1000,
        "snapshot_batch_size": 8,
        "snapshot_num_samples": 64,
        "num_workers": 14,
        "prefetch_data": True,
    }}})
    apply_smoke_overrides(config, 1)
    assert config.trainer.args.max_steps == 1
    assert config.trainer.args.i_log == 1
    assert config.trainer.args.i_sample == 1
    assert config.trainer.args.i_save == 1
    assert config.trainer.args.snapshot_batch_size == 1
    assert config.trainer.args.snapshot_num_samples == 1
    assert config.trainer.args.num_workers == 0
    assert config.trainer.args.prefetch_data is False
