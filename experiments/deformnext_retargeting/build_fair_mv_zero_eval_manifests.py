"""Bind local common-501 MV-Pixal3D zero-shot renders to the frozen evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


METHOD = "mv_pixal3d_zero_shot_source_m0"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-plan-root", type=Path, required=True)
    parser.add_argument("--common-subset", type=Path, required=True)
    parser.add_argument("--original-references", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=501)
    args = parser.parse_args()
    if args.output_root.exists() or args.expected_samples < 1:
        raise ValueError("fresh binding and positive sample count required")

    from deformnext_benchmark.io import canonical_hash, digest

    subset = json.loads(args.common_subset.read_text())["sample_ids"]
    if len(subset) != args.expected_samples or len(set(subset)) != len(subset):
        raise ValueError("frozen common-success population differs")
    originals = {row["sample_id"]: row for row in rows(args.original_references)}
    if not set(subset) <= originals.keys():
        raise ValueError("source reference population does not cover common subset")

    jobs: dict[tuple[str, int], tuple[Path, dict]] = {}
    chunks = sorted(args.render_plan_root.glob("chunk[0-9][0-9][0-9]"))
    for chunk in chunks:
        plan = json.loads((chunk / "plan.json").read_text())
        complete = json.loads((chunk / "completed.json").read_text())
        if (plan.get("status") != "PREPARED_FAIR_MV_ZERO_RAW_AND_M0_RENDER"
                or complete.get("status") != "PASS_FAIR_MV_ZERO_RENDER_CHUNK"
                or complete.get("jobs") != plan.get("jobs")):
            raise ValueError("fair zero render chunk is not complete: " + chunk.name)
        for job in rows(chunk / "jobs.jsonl"):
            if job["method"] != "mv_pixal3d_zero_shot_source_m0":
                continue
            key = job["sample_id"], job["camera_id"]
            if key in jobs:
                raise ValueError("duplicate fair zero render job: " + repr(key))
            jobs[key] = chunk / "stage", job
    if len(jobs) != args.expected_samples * 3:
        raise ValueError("source-M0 camera8/9/10 render closure differs")

    refs, predictions = [], []
    for sid in subset:
        reference = originals[sid]
        refs.append(reference)
        ref_views = {view["camera_id"]: view for view in reference["views"]}
        if set(ref_views) != {9, 10}:
            raise ValueError("camera9/10 source references required")
        bound_views = []
        first_job = None
        for camera in (8, 9, 10):
            stage, job = jobs[sid, camera]
            if first_job is None:
                first_job = job
            elif (job["to_world"] != first_job["to_world"]
                  or job["metric_mesh"] != first_job["metric_mesh"]):
                raise ValueError("camera-specific prediction geometry differs: " + sid)
            image = stage / job["output_rgba"]
            receipt_path = image.with_suffix(".receipt.json")
            metric_mesh = stage / job["metric_mesh"]
            glb = stage / job["mesh"]
            if not all(path.is_file() for path in
                       (image, receipt_path, metric_mesh, glb)):
                raise FileNotFoundError("local fair zero render binding missing: " + sid)
            receipt = json.loads(receipt_path.read_text())
            expected = {
                "job_sha256": canonical_hash(job),
                "metric_mesh_sha256": digest(metric_mesh),
                "input_sha256": digest(glb),
                "to_world_sha256": canonical_hash(job["to_world"]),
                "output_sha256": digest(image),
                "protocol_sha256": job["protocol_sha256"],
            }
            if any(receipt.get(key) != value for key, value in expected.items()):
                raise ValueError("fair zero render receipt differs: " + str(image))
            if camera in ref_views:
                view = ref_views[camera]
                if (receipt.get("camera_sha256") != view["camera_sha256"]
                        or receipt.get("render_sha256") != view["render_sha256"]):
                    raise ValueError("source camera/renderer differs: " + sid)
                bound_views.append({
                    "camera_id": camera,
                    "rgba": str(image).lstrip("/"),
                    "camera_sha256": view["camera_sha256"],
                    "render_sha256": view["render_sha256"],
                })
        stage, _ = jobs[sid, 8]
        predictions.append({
            "method": METHOD, "sample_id": sid, "seed": 0, "status": "ok",
            "mesh": str(stage / first_job["metric_mesh"]).lstrip("/"),
            "to_world": first_job["to_world"], "frame_id": first_job["frame_id"],
            "protocol_sha256": first_job["protocol_sha256"], "views": bound_views,
        })

    args.output_root.mkdir(parents=True)
    (args.output_root / "references.jsonl").write_text("".join(
        json.dumps(row, sort_keys=True) + "\n" for row in refs))
    (args.output_root / "predictions.jsonl").write_text("".join(
        json.dumps(row, sort_keys=True) + "\n" for row in predictions))
    report = {
        "status": "PASS_FAIR_MV_ZERO_SOURCE_M0_EVALUATOR_BINDING",
        "samples": len(refs), "predictions": len(predictions), "method": METHOD,
        "source_views": [9, 10], "target_camera8_rendered_separately": True,
        "source_registration_reference": "input_source_M0_only",
        "target_M1_used_for_registration": False, "sample_ids": subset,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "samples", "predictions")}), flush=True)


if __name__ == "__main__":
    main()
