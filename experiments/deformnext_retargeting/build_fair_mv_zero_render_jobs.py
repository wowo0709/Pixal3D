"""Build local raw/+source-M0 render jobs for fair MV-Pixal3D zero-shot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-plan-root", type=Path, required=True)
    parser.add_argument("--gt-prep-root", type=Path, required=True)
    parser.add_argument("--evaluation-cache", type=Path, required=True)
    parser.add_argument("--common-subset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chunk-samples", type=int, default=32)
    args = parser.parse_args()
    if args.output_root.exists() or args.chunk_samples < 1:
        raise ValueError("fresh render plan and positive chunk size required")
    ids = json.loads(args.common_subset.read_text())["sample_ids"]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("distinct common-success sample IDs required")
    templates = {}
    for chunk in sorted(args.template_plan_root.glob("chunk[0-9][0-9][0-9]")):
        for job in rows(chunk / "jobs.jsonl"):
            if job.get("method") == "ours_v1_raw":
                key = job["sample_id"], job["camera_id"]
                if key in templates:
                    raise ValueError("duplicate common camera template")
                templates[key] = job
    if any((sid, camera) not in templates for sid in ids for camera in (8, 9, 10)):
        raise ValueError("V1 common camera template does not cover requested population")
    inputs_with_meshes = {
        row["sample_id"]: row for row in
        rows(args.gt_prep_root / "inference-inputs-with-meshes.jsonl")
    }
    gt9 = {
        row["sample_id"]: row for row in
        rows(args.gt_prep_root / "GT-render-jobs.jsonl")
        if row["camera_id"] == 9
    }
    if not set(ids) <= inputs_with_meshes.keys() or not set(ids) <= gt9.keys():
        raise ValueError("frozen target GT preparation does not cover common population")
    args.output_root.mkdir(parents=True)
    total = 0
    for start in range(0, len(ids), args.chunk_samples):
        group = ids[start:start + args.chunk_samples]
        chunk = args.output_root / f"chunk{start // args.chunk_samples:03d}"
        stage = chunk / "stage"
        stage.mkdir(parents=True)
        jobs, inputs = [], []
        for sid in group:
            case = args.evaluation_cache / sid
            cache_receipt = json.loads((case / "receipt.json").read_text())
            score = json.loads((case / "surface-score.json").read_text())
            if (cache_receipt.get("status") != "PASS_LOCAL_ZERO_EVALUATION_CACHE"
                    or score.get("status") != "PASS_FAIR_MV_ZERO_RAW_AND_SOURCE_M0_SURFACE"
                    or score.get("sample_id") != sid
                    or score.get("registration_reference") != "input_source_M0_only"):
                raise ValueError("fair zero local cache/surface score differs: " + sid)
            relative = Path("inputs") / sid
            (stage / relative).mkdir(parents=True)
            for name in ("benchmark.glb", "prediction.npz"):
                source, target = case / name, stage / relative / name
                if not source.is_file():
                    raise FileNotFoundError(source)
                os.link(source, target)
                inputs.append(str(relative / name))
            gt_relative = Path("inputs") / "gt" / sid
            (stage / gt_relative).mkdir(parents=True)
            gt_base = gt9[sid]
            gt_material_source = args.gt_prep_root / gt_base["trusted_material_dump"]
            gt_mesh_source = args.gt_prep_root / gt_base["metric_mesh"]
            gt_material = stage / gt_relative / "material.pickle"
            gt_mesh = stage / gt_relative / "target.npz"
            if not gt_material_source.is_file() or not gt_mesh_source.is_file():
                raise FileNotFoundError("frozen target GT asset missing: " + sid)
            os.link(gt_material_source, gt_material)
            os.link(gt_mesh_source, gt_mesh)
            inputs.extend([str(gt_relative / "material.pickle"),
                           str(gt_relative / "target.npz")])
            raw = np.asarray(score["to_world"], dtype=np.float64)
            correction = np.asarray(score["correction_sim3"], dtype=np.float64)
            if (raw.shape != (4, 4) or correction.shape != (4, 4)
                    or not np.allclose(raw[3], [0, 0, 0, 1])
                    or not np.allclose(correction[3], [0, 0, 0, 1])):
                raise ValueError("invalid fair zero raw/Sim3 transform: " + sid)
            for mode, transform in (("raw", raw), ("source_m0", correction @ raw)):
                for camera_id in (8, 9, 10):
                    template = templates[sid, camera_id]
                    job = dict(template,
                        method=f"mv_pixal3d_zero_shot_{mode}",
                        mesh=str(relative / "benchmark.glb"),
                        metric_mesh=str(relative / "prediction.npz"),
                        trusted_material_dump=None, camera_id=camera_id,
                        to_world=transform.tolist(),
                        output_rgba=(f"renders/mv_zero/{mode}/{sid}/"
                                     f"view{camera_id:02d}.png"))
                    jobs.append(job)
            gt_job = dict(gt_base,
                method="gt_target8", camera_id=8,
                camera=inputs_with_meshes[sid]["conditioning_cameras"][0],
                mesh=None, metric_mesh=str(gt_relative / "target.npz"),
                trusted_material_dump=str(gt_relative / "material.pickle"),
                output_rgba=f"renders/gt/{sid}/view08.png")
            jobs.append(gt_job)
        if len(jobs) != len(group) * 7:
            raise ValueError("raw/M0 three-camera predictions plus one target8 GT required")
        (chunk / "jobs.jsonl").write_text("".join(
            json.dumps(job, sort_keys=True) + "\n" for job in jobs))
        (chunk / "input-files.txt").write_text("".join(
            path + "\n" for path in sorted(set(inputs))))
        plan = {"status": "PREPARED_FAIR_MV_ZERO_RAW_AND_M0_RENDER",
                "sample_ids": group, "jobs": len(jobs),
                "input_files": len(set(inputs)),
                "conditions": ["raw", "source_m0"], "cameras": [8, 9, 10],
                "target8_gt_jobs": len(group)}
        (chunk / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        total += len(jobs)
    campaign = {"status": "PASS_FAIR_MV_ZERO_RAW_AND_M0_RENDER_PLAN",
                "samples": len(ids), "jobs": total,
                "chunks": (len(ids) + args.chunk_samples - 1) // args.chunk_samples}
    (args.output_root / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")
    print(json.dumps(campaign), flush=True)


if __name__ == "__main__":
    main()
