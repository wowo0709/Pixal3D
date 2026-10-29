"""Score common-501 MV-Pixal3D zero-shot target-camera renders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHOD = "mv_pixal3d_zero_shot_source_m0"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-plan-root", type=Path, required=True)
    parser.add_argument("--common-subset", type=Path, required=True)
    parser.add_argument("--original-references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=501)
    args = parser.parse_args()
    if args.output.exists() or args.expected_samples < 1:
        raise ValueError("fresh output and positive sample count required")

    from deformnext_benchmark.io import canonical_hash, digest, read_rgba
    from deformnext_benchmark.metrics import image_metrics

    subset = json.loads(args.common_subset.read_text())["sample_ids"]
    if len(subset) != args.expected_samples or len(set(subset)) != len(subset):
        raise ValueError("frozen common-success population differs")
    refs = {row["sample_id"]: row for row in rows(args.original_references)}
    if not set(subset) <= refs.keys():
        raise ValueError("target GT/reference population differs")

    jobs: dict[str, tuple[Path, dict]] = {}
    gt_jobs: dict[str, tuple[Path, dict]] = {}
    for chunk in sorted(args.render_plan_root.glob("chunk[0-9][0-9][0-9]")):
        complete = json.loads((chunk / "completed.json").read_text())
        if complete.get("status") != "PASS_FAIR_MV_ZERO_RENDER_CHUNK":
            raise ValueError("fair zero render chunk is incomplete")
        for job in rows(chunk / "jobs.jsonl"):
            if job["method"] == "gt_target8" and job["camera_id"] == 8:
                sid = job["sample_id"]
                if sid in gt_jobs:
                    raise ValueError("duplicate target8 GT job: " + sid)
                gt_jobs[sid] = chunk / "stage", job
                continue
            if (job["method"] != "mv_pixal3d_zero_shot_source_m0"
                    or job["camera_id"] != 8):
                continue
            sid = job["sample_id"]
            if sid in jobs:
                raise ValueError("duplicate source-M0 target8 job: " + sid)
            jobs[sid] = chunk / "stage", job
    if set(jobs) != set(subset) or set(gt_jobs) != set(subset):
        raise ValueError("common prediction/GT target8 render closure differs")

    scored = []
    for sid in subset:
        gt_stage, gt_job = gt_jobs[sid]
        target_path = gt_stage / gt_job["output_rgba"]
        target_receipt_path = target_path.with_suffix(".receipt.json")
        if not target_path.is_file() or not target_receipt_path.is_file():
            raise FileNotFoundError("standard target8 GT render missing: " + sid)
        target_receipt = json.loads(target_receipt_path.read_text())
        if (target_receipt.get("job_sha256") != canonical_hash(gt_job)
                or target_receipt.get("output_sha256") != digest(target_path)):
            raise ValueError("standard target8 GT receipt differs: " + sid)
        target, target_alpha = read_rgba(
            target_path, resolution=512, background=[1, 1, 1])
        stage, job = jobs[sid]
        prediction = stage / job["output_rgba"]
        receipt_path = prediction.with_suffix(".receipt.json")
        if not prediction.is_file() or not receipt_path.is_file():
            raise FileNotFoundError("fair zero target8 render missing: " + sid)
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("job_sha256") != canonical_hash(job)
                or receipt.get("output_sha256") != digest(prediction)):
            raise ValueError("fair zero target8 receipt differs: " + sid)
        rgb, alpha = read_rgba(
            prediction, resolution=512, background=[1, 1, 1])
        scored.append({
            "sample_id": sid,
            "source_cluster_id": refs[sid]["source_cluster_id"],
            "metrics": image_metrics(rgb, target, alpha, target_alpha),
        })

    clusters = sorted({row["source_cluster_id"] for row in scored})
    cluster_indices = {
        cluster: [index for index, row in enumerate(scored)
                  if row["source_cluster_id"] == cluster]
        for cluster in clusters
    }
    rng = np.random.default_rng(20261028)
    draws = rng.integers(0, len(clusters), size=(20_000, len(clusters)))
    summary = {}
    for metric in ("PSNR", "SSIM", "M_IoU"):
        values = np.asarray([row["metrics"][metric] for row in scored])
        sums = np.asarray([values[cluster_indices[cluster]].sum()
                           for cluster in clusters])
        counts = np.asarray([len(cluster_indices[cluster]) for cluster in clusters])
        bootstrap = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        summary[metric] = {
            "mean": float(values.mean()),
            "cluster_bootstrap_ci95": [float(value) for value in
                                       np.quantile(bootstrap, [.025, .975])],
        }
    result = {
        "status": "PASS_FAIR_MV_ZERO_SOURCE_M0_TARGET8",
        "method": METHOD, "samples": len(scored),
        "source_clusters": len(clusters), "sample_ids": subset,
        "summary": summary, "per_sample": scored,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "samples": len(scored)}),
          flush=True)


if __name__ == "__main__":
    main()
