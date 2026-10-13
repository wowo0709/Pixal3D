#!/usr/bin/env python3
"""Launch one generated retargeting config with explicit assigned GPUs."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpus", required=True, help="comma-separated physical GPU IDs")
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke-steps", type=int, choices=(1, 10))
    args, extra = parser.parse_known_args()
    gpu_ids = [part.strip() for part in args.gpus.split(",") if part.strip()]
    if not gpu_ids or any(gpu not in {"0", "1", "2", "3"} for gpu in gpu_ids):
        raise ValueError("this experiment is restricted to physical GPUs 0-3")
    command = [
        sys.executable, str(REPO / "train.py"), "--config", str(args.config),
        "--num_gpus", str(len(gpu_ids)), "--multiview_runtime",
    ]
    if args.output_dir:
        command += ["--output_dir", args.output_dir]
    if args.smoke_steps:
        command += ["--smoke_steps", str(args.smoke_steps)]
    command += extra
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids)}
    # PyTorch SDPA dispatches to its fused CUDA kernels on supported GPUs and
    # is the compatible fast path when the optional flash-attn package is not
    # installed in the selected environment.
    if importlib.util.find_spec("flash_attn") is None:
        env.setdefault("ATTN_BACKEND", "sdpa")
    raise SystemExit(subprocess.call(command, cwd=REPO, env=env))


if __name__ == "__main__":
    main()
