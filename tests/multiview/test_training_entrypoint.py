import json
from pathlib import Path
import runpy
import sys

import pytest
import torch

from data_toolkit.pipeline import training_manifest
from tests.multiview.test_training_manifest import _write_source


REPO_ROOT = Path(__file__).resolve().parents[2]
RESOLVED_CONFIG_MARKER = f"Config:\n{'=' * 80}\n"


def test_entrypoint_accepts_standalone_hssd_before_cuda(
    tmp_path, monkeypatch, capsys
):
    hssd_path = _write_source(
        tmp_path, "HSSD", schema_version=2, count=3
    )
    output_path = tmp_path / "existing-output"
    output_path.write_text("block output directory creation")
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "default_output_dir": str(output_path),
                "trainer": {"args": {"multiview_stage": "ss64"}},
                "node_rank": 0,
            }
        )
    )
    original_resolve = training_manifest.resolve_training_input
    events = []

    def resolve(config, cli_data_dir, cli_training_data):
        result = original_resolve(
            config, cli_data_dir, cli_training_data
        )
        events.append(result[1])
        return result

    def device_count():
        assert events[0]["source_counts"] == {"HSSD": 3}
        assert events[0]["total_count"] == 3
        return 1

    monkeypatch.setattr(
        training_manifest, "resolve_training_input", resolve
    )
    monkeypatch.setattr(torch.cuda, "device_count", device_count)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--training_data",
            str(hssd_path),
        ],
    )
    with pytest.raises(FileExistsError):
        runpy.run_path(str(REPO_ROOT / "train.py"), run_name="__main__")
    resolved_config = json.loads(
        capsys.readouterr().out.split(RESOLVED_CONFIG_MARKER, 1)[1]
    )
    assert list(json.loads(resolved_config["data_dir"])) == ["HSSD"]
    assert resolved_config["training_evidence"]["source_counts"] == {
        "HSSD": 3
    }


def test_entrypoint_rejects_manifest_before_cuda_for_unknown_stage(
    tmp_path, monkeypatch, capsys
):
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "default_output_dir": str(tmp_path / "output"),
                "multiview_stage": "unknown",
            }
        )
    )

    def fail_if_called():
        pytest.fail("CUDA queried before manifest resolution")

    monkeypatch.setattr(torch.cuda, "device_count", fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--training_data",
            str(tmp_path / "combined.json"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(REPO_ROOT / "train.py"), run_name="__main__")
    assert error.value.code == 2
    assert "unknown multiview_stage" in capsys.readouterr().err


def test_entrypoint_rejects_both_training_interfaces_before_cuda(
    tmp_path, monkeypatch, capsys
):
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "default_output_dir": str(tmp_path / "output"),
                "multiview_stage": "ss64",
            }
        )
    )

    def fail_if_called():
        pytest.fail("CUDA queried before training input validation")

    monkeypatch.setattr(torch.cuda, "device_count", fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--data_dir",
            "{}",
            "--training_data",
            str(tmp_path / "combined.json"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(REPO_ROOT / "train.py"), run_name="__main__")
    assert error.value.code == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_entrypoint_persists_resolved_training_evidence_before_cuda(
    tmp_path, monkeypatch, capsys
):
    output_path = tmp_path / "existing-output"
    output_path.write_text("block output directory creation")
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "default_output_dir": str(output_path),
                "multiview_stage": "ss64",
                "data_dir": "/stale/config/data",
                "node_rank": 0,
            }
        )
    )
    evidence = {
        "training_data": {
            "path": "/verified/combined.json",
            "sha256": "a" * 64,
        },
        "stage": "ss64",
    }
    events = []

    def resolve(config, cli_data_dir, cli_training_data):
        events.append(("resolved", cli_data_dir, cli_training_data))
        return '{"ABO":{},"3D-FUTURE":{}}', evidence

    def device_count():
        assert events == [("resolved", None, "/verified/combined.json")]
        return 1

    monkeypatch.setattr(
        training_manifest, "resolve_training_input", resolve
    )
    monkeypatch.setattr(torch.cuda, "device_count", device_count)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--training_data",
            "/verified/combined.json",
        ],
    )
    with pytest.raises(FileExistsError):
        runpy.run_path(str(REPO_ROOT / "train.py"), run_name="__main__")
    stdout = capsys.readouterr().out
    resolved_config = json.loads(
        stdout.split(RESOLVED_CONFIG_MARKER, 1)[1]
    )
    assert resolved_config["data_dir"] == '{"ABO":{},"3D-FUTURE":{}}'
    assert resolved_config["training_evidence"] == evidence


def test_legacy_entrypoint_removes_spoofed_training_evidence(
    tmp_path, monkeypatch, capsys
):
    output_path = tmp_path / "existing-output"
    output_path.write_text("block output directory creation")
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "default_output_dir": str(output_path),
                "node_rank": 0,
                "training_evidence": {
                    "stage": "spoofed",
                    "training_data": {
                        "path": "/unverified/input.json",
                        "sha256": "0" * 64,
                    },
                },
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--num_gpus",
            "1",
        ],
    )
    with pytest.raises(FileExistsError):
        runpy.run_path(str(REPO_ROOT / "train.py"), run_name="__main__")
    stdout = capsys.readouterr().out
    resolved_config = json.loads(
        stdout.split(RESOLVED_CONFIG_MARKER, 1)[1]
    )
    assert "training_evidence" not in resolved_config
