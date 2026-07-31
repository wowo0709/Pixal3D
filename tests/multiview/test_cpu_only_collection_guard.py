import os
from pathlib import Path
import subprocess
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cpu_only_pytest_collection_imports_flex_gemm_dependents():
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/multiview/test_pipeline_inputs.py",
            "--collect-only",
            "-q",
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_cpu_only_harness_does_not_mask_unrelated_collection_cuda_calls(
    tmp_path,
):
    probe = tmp_path / "test_unrelated_cuda_probe.py"
    probe.write_text(
        "import torch\n"
        "\n"
        "torch.cuda.get_device_name()\n"
        "\n"
        "def test_probe():\n"
        "    pass\n"
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests.multiview.conftest",
            str(probe),
            "--collect-only",
            "-q",
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2, result.stdout + result.stderr
    assert "RuntimeError: No CUDA GPUs are available" in result.stdout


def test_cpu_only_collection_patch_is_restored_before_test_execution():
    assert torch.cuda.get_device_name.__module__ == "torch.cuda"
    assert torch.cuda.get_device_name.__name__ == "get_device_name"
