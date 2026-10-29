"""Generate, Top-1 texture, and export one geometry arm on one GPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from .run_minimal_test_cascade_worker import retryable_render_failure, sha


def execute(command: list[str], log: Path, env: dict, timeout: int | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        subprocess.run(command, check=True, env=env, timeout=timeout,
                       stdout=stream, stderr=subprocess.STDOUT)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--runtime-python", type=Path, required=True)
    p.add_argument("--physical-gpu", required=True)
    p.add_argument("--allowed-gpus", default="0,1,2,3,6,7")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    a = p.parse_args()
    campaign = json.loads(a.campaign.read_text())
    if (campaign.get("status") != "PREPARED_COMMON501_SIX_ARM_FULL_RENDER_ABLATION"
            or a.arm not in campaign["arms"] or a.physical_gpu not in a.allowed_gpus.split(",")):
        raise ValueError("frozen campaign arm and authorized GPU required")
    all_entries = [r for r in campaign["plans"] if r["arm"] == a.arm]
    if (len(all_entries) != 501 or a.world_size < 1
            or not 0 <= a.rank < a.world_size):
        raise ValueError("complete arm required")
    entries = all_entries[a.rank::a.world_size]
    root = a.campaign.parent
    repo = Path(__file__).resolve().parents[2]
    benchmark = Path("/root/dev/.cache/deformnext-benchmark-runtime")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.physical_gpu,
               MATCHING_V2_ALLOWED_GPUS=a.allowed_gpus,
               BENCHMARK_ASSIGNED_GPU=a.physical_gpu, ATTN_BACKEND="sdpa",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", OPENCV_IO_ENABLE_OPENEXR="1",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    for position, entry in enumerate(entries, 1):
        sid = entry["sample_id"]
        shape_plan = Path(entry["shape_plan"]); shape = json.loads(shape_plan.read_text())
        shape_root = Path(shape["output"])
        pbr_plan = Path(entry["top1_pbr_plan"]); pbr = json.loads(pbr_plan.read_text())
        pbr_root = Path(pbr["output"]); glb = pbr_root / "benchmark.glb"
        if glb.is_file() and glb.with_suffix(".json").is_file():
            continue
        if not (shape_root / "generation.json").is_file():
            if shape_root.exists():
                raise FileExistsError(shape_root)
            shape_root.parent.mkdir(parents=True, exist_ok=True)
            execute([str(a.runtime_python), "-u", "-m",
                "experiments.deformnext_retargeting.cascade_from_ss_artifact",
                "--plan", str(shape_plan)], root / "logs" / "shape" / a.arm / f"{sid}.log", env)
        shape_result = json.loads((shape_root / "generation.json").read_text())
        if shape_result.get("status") != "PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION":
            raise ValueError("full shape generation failed: " + sid)
        if not (pbr_root / "generation.json").is_file():
            if pbr_root.exists():
                raise FileExistsError(pbr_root)
            code = ("import json,sys;from pathlib import Path;"
                    "from experiments.deformnext_retargeting.replay_r_pbr_on_generated_shape import run;"
                    "run(json.loads(Path(sys.argv[1]).read_text()))")
            execute([str(a.runtime_python), "-u", "-c", code, str(pbr_plan)],
                    root / "logs" / "pbr" / a.arm / f"{sid}.log", env)
        attempts = ((250_000, None), (500_000, None), (1_000_000, None),
                    (2_000_000, None), (5_000_000, None),
                    (5_000_000, "native"))
        for budget, simplifier in attempts:
            if glb.is_file() and glb.with_suffix(".json").is_file():
                break
            command = [str(a.runtime_python), "-u", "-m",
                "experiments.deformnext_retargeting.export_minimal_test_mesh",
                "--pixal3d-repo", str(repo), "--benchmark-repo", str(benchmark),
                "--prediction", str(pbr_root / "prediction.npz"),
                "--shape-generation", str(shape_root / "generation.json"),
                "--texture-generation", str(pbr_root / "generation.json"),
                "--anchor-evaluation", str(pbr_root / "geometry-evaluation.npz"),
                "--texture-arm", "top1-only", "--render-face-budget", str(budget),
                "--output", str(glb)]
            if simplifier:
                command.extend(("--simplifier", simplifier))
            suffix = f"-{simplifier}" if simplifier else ""
            log = root / "logs" / "export" / a.arm / f"{sid}-{budget}{suffix}.log"
            try:
                execute(command, log, env, 3600)
            except subprocess.CalledProcessError:
                if (budget, simplifier) == attempts[-1] or not retryable_render_failure(log.read_text(errors="replace")):
                    raise
        report = json.loads(glb.with_suffix(".json").read_text())
        if report.get("glb_sha256") != sha(glb):
            raise ValueError("export checksum differs: " + sid)
        # Full-resolution prediction archives are no longer needed after the
        # immutable GLB and compact surface evidence exist.
        (shape_root / "prediction.npz").unlink(missing_ok=True)
        (pbr_root / "prediction.npz").unlink(missing_ok=True)
        print(json.dumps({"phase": "EXPORTED", "arm": a.arm, "rank": a.rank, "position": position,
                          "total": len(entries), "sample_id": sid}), flush=True)
    print(json.dumps({"status": "PASS_COMMON501_GEOMETRY_ARM_EXPORTED", "arm": a.arm,
                      "samples": len(entries), "physical_gpu": a.physical_gpu,
                      "rank": a.rank, "world_size": a.world_size}), flush=True)


if __name__ == "__main__":
    main()
