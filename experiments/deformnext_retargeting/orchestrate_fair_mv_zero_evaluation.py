"""Wait for cascade closure, then render and evaluate MV-Pixal3D zero-shot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run(command: list[str], log: Path, *, cwd: Path, env: dict | None = None) -> None:
    print(json.dumps({"phase": "RUN", "log": str(log), "command": command}), flush=True)
    with log.open("x") as stream:
        subprocess.run(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cascade-complete", type=Path, required=True)
    parser.add_argument("--template-plan-root", type=Path, required=True)
    parser.add_argument("--gt-prep-root", type=Path, required=True)
    parser.add_argument("--evaluation-cache", type=Path, required=True)
    parser.add_argument("--common-subset", type=Path, required=True)
    parser.add_argument("--original-references", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--benchmark-runtime", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--metric-registry", type=Path, required=True)
    parser.add_argument("--metric-python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.output_root.exists() or args.poll_seconds < 10:
        raise ValueError("fresh output root and bounded polling required")
    args.output_root.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[2]
    logs = args.output_root / "logs"
    logs.mkdir()
    while not args.cascade_complete.is_file():
        print(json.dumps({"phase": "WAIT_FAIR_MV_ZERO_CASCADE"}), flush=True)
        time.sleep(args.poll_seconds)
    cascade = json.loads(args.cascade_complete.read_text())
    if (cascade.get("status") != "PASS_FAIR_MV_PIXAL3D_ZERO_SHOT_TEST508_ARCHIVED"
            or cascade.get("samples") != 508):
        raise ValueError("complete checksum-verified TEST508 cascade required")

    module = "experiments.deformnext_retargeting."
    render_plan = args.output_root / "render-plan"
    run([sys.executable, "-m", module + "build_fair_mv_zero_render_jobs",
         "--template-plan-root", str(args.template_plan_root),
         "--gt-prep-root", str(args.gt_prep_root),
         "--evaluation-cache", str(args.evaluation_cache),
         "--common-subset", str(args.common_subset),
         "--output-root", str(render_plan), "--chunk-samples", "32"],
        logs / "01-build-render-plan.log", cwd=repo)
    run([sys.executable, "-m", module + "run_fair_mv_zero_render_campaign",
         "--plan-root", str(render_plan), "--blender", str(args.blender),
         "--benchmark-root", str(args.benchmark_runtime),
         "--protocol", str(args.protocol), "--gpus", "0,1,2,3,6,7",
         "--threads-per-gpu", "2"],
        logs / "02-render.log", cwd=repo)

    metric_env = dict(os.environ)
    metric_env["PYTHONPATH"] = str(args.benchmark_runtime)
    binding = args.output_root / "binding"
    run([str(args.metric_python), "-m",
         module + "build_fair_mv_zero_eval_manifests",
         "--render-plan-root", str(render_plan),
         "--common-subset", str(args.common_subset),
         "--original-references", str(args.original_references),
         "--output-root", str(binding), "--expected-samples", "501"],
        logs / "03-build-binding.log", cwd=repo, env=metric_env)
    target8 = args.output_root / "target8.json"
    run([str(args.metric_python), "-m", module + "score_fair_mv_zero_target8",
         "--render-plan-root", str(render_plan),
         "--common-subset", str(args.common_subset),
         "--original-references", str(args.original_references),
         "--output", str(target8), "--expected-samples", "501"],
        logs / "04-score-target8.log", cwd=repo, env=metric_env)

    evaluation = args.output_root / "metric-evaluation"
    gpus = [gpu for gpu in ("0", "1", "2", "3", "6", "7") for _ in range(12)]
    run([sys.executable, "-m", module + "run_local_sharded_ours_v1_m0",
         "--binding", str(binding), "--reference-root", str(args.reference_root),
         "--protocol", str(args.protocol), "--metric-registry", str(args.metric_registry),
         "--metric-python", str(args.metric_python),
         "--benchmark-root", str(args.benchmark_runtime / "deformnext_benchmark"),
         "--method", "mv_pixal3d_zero_shot_source_m0", "--standalone-one-method",
         "--output-root", str(evaluation), "--gpus", *gpus,
         "--cpu-threads", "1", "--balance-by-mesh-bytes",
         "--expected-samples", "501"],
        logs / "05-learned-and-geometry-metrics.log", cwd=repo)
    row = args.output_root / "main-table-row.json"
    run([sys.executable, "-m", module + "compose_fair_mv_zero_row",
         "--evaluation-summary",
         str(evaluation / "evaluation-one-method" / "summary.json"),
         "--target8", str(target8), "--output", str(row)],
        logs / "06-compose-row.log", cwd=repo)
    result = json.loads(row.read_text())
    completion = {
        "status": "PASS_FAIR_MV_PIXAL3D_ZERO_SHOT_COMMON501_COMPLETE",
        "samples": 501, "metrics": result["metrics"], "row": str(row),
        "cascade": str(args.cascade_complete), "render": str(render_plan / "complete.json"),
        "evaluation": str(evaluation / "complete.json"), "target8": str(target8),
    }
    (args.output_root / "complete.json").write_text(
        json.dumps(completion, indent=2, allow_nan=False) + "\n")
    print(json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
