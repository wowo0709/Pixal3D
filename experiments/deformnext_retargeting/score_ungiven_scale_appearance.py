"""Score and summarize common-501 appearance-stage ablation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ARMS = ("vanilla", "raw-topk-plus-feature")
METRICS = ("LPIPS", "DreamSim", "CLIP")
GPUS = ("0", "1", "2", "3", "6", "7")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def weighted(samples: list[dict], metric: str, uid: bool) -> float:
    if not uid:
        return float(np.mean([row["metrics"][metric] for row in samples]))
    groups = {}
    for row in samples:
        groups.setdefault(row["source_uid"], []).append(row["metrics"][metric])
    return float(np.mean([np.mean(values) for values in groups.values()]))


def cluster_interval(differences: np.ndarray, clusters: list[str], *, seed: int,
                     draws: int = 10000) -> dict:
    unique = sorted(set(clusters))
    members = {key: np.flatnonzero(np.asarray(clusters) == key) for key in unique}
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        chosen = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([members[key] for key in chosen])
        values[draw] = differences[indices].mean()
    return {"mean": float(differences.mean()),
            "ci95": [float(np.quantile(values, .025)),
                     float(np.quantile(values, .975))],
            "draws": draws, "seed": seed, "cluster_count": len(unique)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-plan-root", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--metric-registry", type=Path, required=True)
    parser.add_argument("--metric-python", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--top1-results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    args = parser.parse_args()
    if (args.output_root.exists() or args.workers_per_gpu < 1
            or not args.metric_python.is_file()):
        raise ValueError("fresh output, installed metric runtime and positive workers required")
    complete = json.loads((args.render_plan_root / "complete.json").read_text())
    if (complete.get("status") != "PASS_COMMON501_APPEARANCE_SOURCE_RENDER_CAMPAIGN"
            or complete.get("samples") != 501 or complete.get("jobs") != 2004):
        raise ValueError("complete appearance render campaign required")
    references = rows(args.references)
    refs = {row["sample_id"]: row for row in references}
    if len(references) != 501 or len(refs) != 501:
        raise ValueError("frozen common-501 references required")

    tasks = []
    for chunk in sorted(args.render_plan_root.glob("chunk[0-9][0-9][0-9]")):
        plan = json.loads((chunk / "plan.json").read_text())
        done = json.loads((chunk / "completed.json").read_text())
        if (plan.get("status") != "PREPARED_COMMON501_APPEARANCE_SOURCE_RENDER_CHUNK"
                or done.get("status") != "PASS_COMMON501_APPEARANCE_RENDER_CHUNK"):
            raise ValueError("appearance render chunk closure differs")
        for job in rows(chunk / "jobs.jsonl"):
            method = job["method"]
            arm = ("vanilla" if method == "appearance_vanilla" else
                   "raw-topk-plus-feature" if method == "appearance_raw_topk_plus_feature"
                   else None)
            if arm is None or job["camera_id"] not in (9, 10):
                raise ValueError("unexpected appearance render job")
            reference = refs[job["sample_id"]]
            ref_views = {view["camera_id"]: view for view in reference["views"]}
            view = ref_views[job["camera_id"]]
            tasks.append({
                "arm": arm, "sample_id": job["sample_id"],
                "camera_id": job["camera_id"],
                "source_uid": reference["source_uid"],
                "source_cluster_id": reference["source_cluster_id"],
                "category": reference["category"],
                "camera_sha256": view["camera_sha256"],
                "prediction_rgba": str(chunk / "stage" / job["output_rgba"]),
                "reference_rgba": view["rgba"],
            })
    if (len(tasks) != 2004 or len({(row["arm"], row["sample_id"], row["camera_id"])
                                  for row in tasks}) != 2004):
        raise ValueError("two arms times common501 times two source cameras required")
    args.output_root.mkdir(parents=True)
    task_file = args.output_root / "tasks.jsonl"
    task_file.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in tasks))

    gpus = [gpu for gpu in GPUS for _ in range(args.workers_per_gpu)]
    logs = args.output_root / "logs"
    shards = args.output_root / "shards"
    logs.mkdir(); shards.mkdir()
    children = []
    env = dict(os.environ, PYTHONPATH=str(args.benchmark_root))
    for rank, gpu in enumerate(gpus):
        output = shards / f"rank{rank:02d}.jsonl"
        log_path = logs / f"rank{rank:02d}-gpu{gpu}.log"
        stream = log_path.open("w")
        command = [str(args.metric_python), "-u", "-m",
                   "experiments.deformnext_retargeting.score_ungiven_scale_appearance_worker",
                   "--tasks", str(task_file), "--reference-root", str(args.reference_root),
                   "--protocol", str(args.protocol),
                   "--metric-registry", str(args.metric_registry),
                   "--physical-gpu", gpu, "--allowed-gpus", ",".join(GPUS),
                   "--rank", str(rank), "--world-size", str(len(gpus)),
                   "--output", str(output)]
        process = subprocess.Popen(command, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT)
        children.append((rank, gpu, output, log_path, process, stream))
        print(json.dumps({"phase": "LAUNCH", "rank": rank, "gpu": gpu,
                          "pid": process.pid}), flush=True)
    failures = []
    for rank, gpu, output, log_path, process, stream in children:
        code = process.wait(); stream.close()
        print(json.dumps({"phase": "JOIN", "rank": rank, "gpu": gpu,
                          "returncode": code}), flush=True)
        if code or not output.is_file():
            failures.append({"rank": rank, "gpu": gpu, "returncode": code,
                             "log": str(log_path)})
    if failures:
        raise RuntimeError("appearance metric shards failed: " + json.dumps(failures))
    scored = [row for _, _, output, _, _, _ in children for row in rows(output)]
    index = {(row["arm"], row["sample_id"], row["camera_id"]): row for row in scored}
    if len(scored) != 2004 or len(index) != 2004:
        raise ValueError("appearance metric shard merge differs")

    per_sample = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for sid in refs:
            views = [index[arm, sid, camera] for camera in (9, 10)]
            per_sample[arm].append({
                "sample_id": sid, "source_uid": refs[sid]["source_uid"],
                "source_cluster_id": refs[sid]["source_cluster_id"],
                "category": refs[sid]["category"],
                "metrics": {metric: float(np.mean([view["metrics"][metric]
                                                    for view in views]))
                            for metric in METRICS},
            })
    top1 = json.loads(args.top1_results.read_text())
    if (top1.get("method") != "ours_selected_source_m0" or top1.get("failures")
            or len(top1.get("per_deformation_seed", [])) != 501):
        raise ValueError("complete selected Top-1 common-501 result required")
    top1_rows = top1["per_deformation_seed"]
    top1_by_id = {row["sample_id"]: row for row in top1_rows}
    if set(top1_by_id) != set(refs):
        raise ValueError("Top-1 and appearance common populations differ")

    summaries = {}
    for arm, sample_rows in (*per_sample.items(), ("top1-only", top1_rows)):
        summaries[arm] = {
            "deformation_weighted": {metric: weighted(sample_rows, metric, False)
                                     for metric in METRICS},
            "UID_weighted": {metric: weighted(sample_rows, metric, True)
                             for metric in METRICS},
            "samples": len(sample_rows),
        }
    aligned = {
        "vanilla": {row["sample_id"]: row for row in per_sample["vanilla"]},
        "raw-topk-plus-feature": {row["sample_id"]: row
                                  for row in per_sample["raw-topk-plus-feature"]},
        "top1-only": top1_by_id,
    }
    contrasts = {}
    for left, right in (("top1-only", "vanilla"),
                        ("raw-topk-plus-feature", "vanilla"),
                        ("raw-topk-plus-feature", "top1-only")):
        key = left + "_minus_" + right
        contrasts[key] = {}
        for offset, metric in enumerate(METRICS):
            diff = np.asarray([aligned[left][sid]["metrics"][metric]
                               - aligned[right][sid]["metrics"][metric]
                               for sid in refs], dtype=np.float64)
            clusters = [refs[sid]["source_cluster_id"] for sid in refs]
            contrasts[key][metric] = cluster_interval(
                diff, clusters, seed=20261029 + offset)
    report = {
        "status": "PASS_COMMON501_APPEARANCE_ABLATION_THREE_METRICS",
        "samples": 501, "source_views": [9, 10],
        "selected_geometry_fixed": True,
        "source_registration": "input_source_M0_only",
        "rows": summaries, "paired_cluster_bootstrap": contrasts,
        "metric_directions": {"LPIPS": "lower", "DreamSim": "lower",
                              "CLIP": "higher"},
        "training_updates": 0,
    }
    (args.output_root / "appearance-ablation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output_root / "per-sample.json").write_text(
        json.dumps({**per_sample, "top1-only": top1_rows}, indent=2,
                   allow_nan=False) + "\n")
    (args.output_root / "complete.json").write_text(json.dumps({
        "status": "PASS_COMMON501_APPEARANCE_ABLATION_COMPLETE",
        "samples": 501, "result": str(args.output_root / "appearance-ablation.json"),
        "workers": len(gpus), "physical_gpus": list(GPUS),
    }, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
