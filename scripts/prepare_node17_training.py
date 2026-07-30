#!/usr/bin/env python3
"""Plan or execute CPU-only Node17 training-data preparation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def _require_cpu_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be explicitly set to empty"
        )
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError(
            "PYTHONDONTWRITEBYTECODE must be explicitly set to 1"
        )


_require_cpu_environment()

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_toolkit.pipeline.node17_training_prepare import (  # noqa: E402
    Node17PreparationPaths,
    plan_node17_training,
    prepare_node17_training,
)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or create Node17 HSSD and three-source training inputs."
        )
    )
    parser.add_argument(
        "--data2-root",
        type=Path,
        default=Path("/root/data2/pixal3d"),
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=Path("/root/node17/data/pixal3d"),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(
            "/root/dev/Pixal3D/.worktrees/multiview-model-extension"
        ),
    )
    parser.add_argument(
        "--source-host", default="youngwoo@n16.unist.info"
    )
    parser.add_argument("--source-port", type=int, default=55555)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "/home/youngwoo/data/pixal3d/train/production/hssd"
        ),
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    paths = Node17PreparationPaths.from_roots(
        args.data2_root,
        args.local_root,
        args.repo_root,
        source_host=args.source_host,
        source_port=args.source_port,
        source_root=args.source_root,
    )
    if args.execute:
        report = prepare_node17_training(paths)
        result = {"report": str(report)}
    else:
        result = plan_node17_training(paths)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
