#!/usr/bin/env python3
"""Publish the verified ABO + 3D-FUTURE training manifest."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_toolkit.pipeline.training_manifest import (
    CANONICAL_SOURCES,
    STAGES,
    publish_combined_training_data,
)


DEFAULT_ABO = Path(
    "/root/node17/data/pixal3d/train/production/abo/training_data.json"
)
DEFAULT_3D_FUTURE = Path(
    "/root/node17/data/pixal3d/train/production/3d-future/training_data.json"
)
DEFAULT_OUTPUT = Path(
    "/root/node17/data/pixal3d/train/production/"
    "abo-3d-future/training_data.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a strict ABO + 3D-FUTURE training manifest."
    )
    parser.add_argument("--abo", type=Path, default=DEFAULT_ABO)
    parser.add_argument(
        "--3d-future", dest="future", type=Path, default=DEFAULT_3D_FUTURE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = publish_combined_training_data(
        {
            CANONICAL_SOURCES[0]: args.abo,
            CANONICAL_SOURCES[1]: args.future,
        },
        args.output,
    )
    manifest = json.loads(output.read_text())
    result = {
        "output": str(output),
        "sha256": sha256(output.read_bytes()).hexdigest(),
        "stages": {
            stage: {
                "source_counts": manifest["stages"][stage][
                    "source_counts"
                ],
                "total_count": manifest["stages"][stage]["total_count"],
            }
            for stage in STAGES
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
