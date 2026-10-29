"""Build source-camera render jobs for fixed-geometry appearance ablation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance-campaign", type=Path, required=True)
    parser.add_argument("--generation-complete", type=Path, required=True)
    parser.add_argument("--template-plan-root", type=Path, required=True)
    parser.add_argument("--selected-binding", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chunk-samples", type=int, default=32)
    args = parser.parse_args()
    if args.output_root.exists() or args.chunk_samples < 1:
        raise ValueError("fresh appearance render root and positive chunk size required")
    campaign = json.loads(args.appearance_campaign.read_text())
    complete = json.loads(args.generation_complete.read_text())
    if (campaign.get("status")
            != "PREPARED_COMMON501_PAIRED_NOISE_APPEARANCE_ABLATION_ON_SELECTED_GEOMETRY"
            or campaign.get("paired_texture_noise") is not True
            or complete.get("status")
            != "PASS_COMMON501_APPEARANCE_GENERATION_ARCHIVED"
            or campaign.get("samples") != 501 or complete.get("samples") != 501):
        raise ValueError("complete common-501 appearance generation required")
    entries = {row["sample_id"]: row for row in campaign["plans"]}
    ids = list(entries)

    templates = {}
    for job_file in sorted(args.template_plan_root.glob("chunk[0-9][0-9][0-9]/jobs.jsonl")):
        for job in rows(job_file):
            if job.get("method") == "ours_selected_source_m0" and job.get("camera_id") in (9, 10):
                key = job["sample_id"], job["camera_id"]
                if key in templates:
                    raise ValueError("duplicate selected source-camera template")
                templates[key] = job
    selected_rows = [row for row in rows(args.selected_binding / "predictions.jsonl")
                     if row.get("method") == "ours_selected_source_m0"]
    selected = {row["sample_id"]: row for row in selected_rows}
    if (len(selected_rows) != 501 or len(selected) != 501 or set(ids) != set(selected)
            or any((sid, camera) not in templates for sid in ids for camera in (9, 10))):
        raise ValueError("selected common-501 templates/binding differ")

    args.output_root.mkdir(parents=True)
    jobs_total = 0
    for start in range(0, len(ids), args.chunk_samples):
        group = ids[start:start + args.chunk_samples]
        chunk = args.output_root / f"chunk{start // args.chunk_samples:03d}"
        stage = chunk / "stage"
        stage.mkdir(parents=True)
        jobs, inputs = [], []
        for sid in group:
            entry = entries[sid]
            transform = selected[sid]["to_world"]
            for arm, key in (("vanilla", "vanilla_root"),
                             ("raw-topk-plus-feature", "topk_root")):
                source = Path(entry[key]) / "benchmark.glb"
                source_report = source.with_suffix(".json")
                archive = Path(entry[key]) / "archive-receipt.json"
                if not all(path.is_file() for path in (source, source_report, archive)):
                    raise FileNotFoundError("appearance GLB/archive missing: " + sid + "/" + arm)
                relative = Path("inputs") / arm / sid / "benchmark.glb"
                target = stage / relative
                target.parent.mkdir(parents=True)
                os.link(source, target)
                inputs.append(str(relative))
                for camera in (9, 10):
                    job = dict(templates[sid, camera],
                               method="appearance_" + arm.replace("-", "_"),
                               mesh=str(relative), metric_mesh=str(relative),
                               to_world=transform,
                               output_rgba=(f"renders/{arm}/{sid}/"
                                            f"view{camera:02d}.png"))
                    jobs.append(job)
        if len(jobs) != len(group) * 4:
            raise ValueError("two appearance arms times source cameras9/10 required")
        (chunk / "jobs.jsonl").write_text("".join(
            json.dumps(row, sort_keys=True) + "\n" for row in jobs))
        (chunk / "input-files.txt").write_text("".join(
            path + "\n" for path in sorted(set(inputs))))
        plan = {
            "status": "PREPARED_COMMON501_APPEARANCE_SOURCE_RENDER_CHUNK",
            "sample_ids": group, "jobs": len(jobs),
            "input_files": len(set(inputs)),
            "arms": ["vanilla", "raw-topk-plus-feature"],
            "cameras": [9, 10], "selected_geometry_fixed": True,
            "source_registration": "input_source_M0_only",
        }
        (chunk / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        jobs_total += len(jobs)
    result = {
        "status": "PASS_COMMON501_APPEARANCE_SOURCE_RENDER_PLAN",
        "samples": 501, "jobs": jobs_total,
        "chunks": (len(ids) + args.chunk_samples - 1) // args.chunk_samples,
        "arms": ["vanilla", "raw-topk-plus-feature"], "cameras": [9, 10],
    }
    (args.output_root / "campaign.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
