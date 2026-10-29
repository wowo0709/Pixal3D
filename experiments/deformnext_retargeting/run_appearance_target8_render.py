"""Render the retained appearance camera-8 ablation on authorized GPUs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-root", type=Path, required=True)
    p.add_argument("--blender", type=Path, required=True)
    p.add_argument("--benchmark-root", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--gpus", default="0,1,2,3,6,7")
    p.add_argument("--threads-per-gpu", type=int, default=2)
    a = p.parse_args()
    campaign = json.loads((a.plan_root / "campaign.json").read_text())
    chunks = sorted(a.plan_root.glob("chunk[0-9][0-9][0-9]"))
    if a.gpus != "0,1,2,3,6,7" or campaign.get("jobs") != 1002 or len(chunks) != 16:
        raise ValueError("complete target8 plan and authorized GPUs required")
    script = Path(__file__).parent
    for i, chunk in enumerate(chunks, 1):
        done = chunk / "completed.json"
        if done.is_file():
            continue
        subprocess.run([sys.executable, str(script / "render_source_sim3_chunk.py"),
            "--chunk", str(chunk), "--stage-root", str(chunk / "stage"),
            "--blender", str(a.blender), "--batch-script", str(script / "render_source_sim3_blender_batch.py"),
            "--benchmark-root", str(a.benchmark_root), "--protocol", str(a.protocol),
            "--gpus", a.gpus, "--threads-per-gpu", str(a.threads_per_gpu)], check=True)
        result = {"status": "PASS_APPEARANCE_TARGET8_RENDER_CHUNK", "position": i,
                  "chunks": len(chunks), "jobs": json.loads((chunk / "plan.json").read_text())["jobs"]}
        done.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    result = {"status": "PASS_APPEARANCE_TARGET8_RENDER_CAMPAIGN", "samples": 501, "jobs": 1002}
    (a.plan_root / "complete.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
