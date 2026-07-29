#!/usr/bin/env python3
"""Plan or execute CPU-only Node16 training-data preparation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def _require_cpu_environment() -> None:
    if (
        "CUDA_VISIBLE_DEVICES" not in os.environ
        or os.environ["CUDA_VISIBLE_DEVICES"] != ""
    ):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be explicitly set to empty"
        )


_require_cpu_environment()

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_toolkit.pipeline.node16_training_prepare import (  # noqa: E402
    PreparationPaths,
    plan_node16_training,
    prepare_node16_training,
)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Node16-local HSSD and three-source training data."
    )
    parser.add_argument(
        "--data2-root",
        type=Path,
        default=Path("/file2/youngwoo/pixal3d"),
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=Path("/home/youngwoo/data/pixal3d"),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/home/youngwoo/Pixal3D-training-hssd"),
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    paths = PreparationPaths.from_roots(
        args.data2_root, args.local_root, args.repo_root
    )
    if args.execute:
        report_path = prepare_node16_training(paths)
        print(json.dumps({"report": str(report_path)}, indent=2))
    else:
        print(json.dumps(plan_node16_training(paths), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
