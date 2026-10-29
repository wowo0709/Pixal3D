"""Render common-501 appearance controls on GPUs 0,1,2,3,6,7."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,6,7")
    parser.add_argument("--threads-per-gpu", type=int, default=2)
    args = parser.parse_args()
    if (args.gpus != "0,1,2,3,6,7" or not args.blender.is_file()
            or not args.protocol.is_file() or args.threads_per_gpu < 1):
        raise ValueError("exact authorized GPUs, renderer and protocol required")
    campaign = json.loads((args.plan_root / "campaign.json").read_text())
    chunks = sorted(args.plan_root.glob("chunk[0-9][0-9][0-9]"))
    if (campaign.get("status") != "PASS_COMMON501_APPEARANCE_SOURCE_RENDER_PLAN"
            or campaign.get("samples") != 501 or campaign.get("jobs") != 2004
            or len(chunks) != campaign.get("chunks")):
        raise ValueError("complete common-501 appearance render plan required")
    scripts = Path(__file__).parent
    for position, chunk in enumerate(chunks, 1):
        completed = chunk / "completed.json"
        if completed.is_file():
            record = json.loads(completed.read_text())
            if record.get("status") != "PASS_COMMON501_APPEARANCE_RENDER_CHUNK":
                raise ValueError("invalid existing appearance render chunk")
            continue
        subprocess.run([
            sys.executable, str(scripts / "render_source_sim3_chunk.py"),
            "--chunk", str(chunk), "--stage-root", str(chunk / "stage"),
            "--blender", str(args.blender), "--batch-script",
            str(scripts / "render_source_sim3_blender_batch.py"),
            "--benchmark-root", str(args.benchmark_root),
            "--protocol", str(args.protocol), "--gpus", args.gpus,
            "--threads-per-gpu", str(args.threads_per_gpu),
        ], check=True)
        plan = json.loads((chunk / "plan.json").read_text())
        record = {
            "status": "PASS_COMMON501_APPEARANCE_RENDER_CHUNK",
            "chunk": chunk.name, "samples": len(plan["sample_ids"]),
            "jobs": plan["jobs"], "position": position, "chunks": len(chunks),
        }
        completed.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    result = {
        "status": "PASS_COMMON501_APPEARANCE_SOURCE_RENDER_CAMPAIGN",
        "samples": 501, "jobs": 2004, "chunks": len(chunks),
        "physical_gpus": args.gpus.split(","),
    }
    (args.plan_root / "complete.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
