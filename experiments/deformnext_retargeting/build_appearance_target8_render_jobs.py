"""Build camera-8 render jobs for the two retained appearance ablations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--appearance-campaign", type=Path, required=True)
    p.add_argument("--template-plan-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--aligned-source-plan-root", type=Path)
    p.add_argument("--chunk-samples", type=int, default=32)
    a = p.parse_args()
    if a.output_root.exists() or a.chunk_samples < 1:
        raise ValueError("fresh output and positive chunk size required")
    campaign = json.loads(a.appearance_campaign.read_text())
    entries = {r["sample_id"]: r for r in campaign["plans"]}
    if campaign.get("samples") != 501 or len(entries) != 501:
        raise ValueError("common-501 appearance campaign required")
    templates = {}
    for path in a.template_plan_root.glob("chunk[0-9][0-9][0-9]/jobs.jsonl"):
        for job in rows(path):
            if job.get("camera_id") == 8 and job.get("method") == "mv_pixal3d_zero_shot_source_m0":
                templates[job["sample_id"]] = job
    if set(templates) != set(entries):
        raise ValueError("camera-8 template population differs")
    aligned = {}
    if a.aligned_source_plan_root:
        for path in a.aligned_source_plan_root.glob("chunk*/jobs.jsonl"):
            for job in rows(path):
                key = (job["method"], job["sample_id"])
                if key in aligned and aligned[key] != job["to_world"]:
                    raise ValueError("source camera transforms disagree")
                aligned[key] = job["to_world"]
        if len(aligned) != 1002:
            raise ValueError("aligned appearance source transforms incomplete")
    a.output_root.mkdir(parents=True)
    ids = list(entries)
    total = 0
    for start in range(0, len(ids), a.chunk_samples):
        group = ids[start:start + a.chunk_samples]
        chunk = a.output_root / f"chunk{start // a.chunk_samples:03d}"
        stage = chunk / "stage"
        stage.mkdir(parents=True)
        jobs, inputs = [], []
        for sid in group:
            entry = entries[sid]
            for arm, key in (("vanilla", "vanilla_root"), ("raw-topk-plus-feature", "topk_root")):
                source = Path(entry[key]) / "benchmark.glb"
                if not source.is_file():
                    raise FileNotFoundError(source)
                relative = Path("inputs") / arm / sid / "benchmark.glb"
                target = stage / relative
                target.parent.mkdir(parents=True)
                os.link(source, target)
                inputs.append(str(relative))
                job = dict(templates[sid], method="appearance_" + arm.replace("-", "_"),
                           mesh=str(relative), metric_mesh=str(relative),
                           to_world=(aligned[("appearance_" + arm.replace("-", "_"), sid)]
                                     if a.aligned_source_plan_root else
                                     json.loads((Path(entry[key]) / "benchmark.json").read_text())["to_world"]),
                           trusted_material_dump=None,
                           output_rgba=f"renders/{arm}/{sid}/view08.png")
                jobs.append(job)
        (chunk / "jobs.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in jobs))
        (chunk / "input-files.txt").write_text("".join(x + "\n" for x in sorted(set(inputs))))
        (chunk / "plan.json").write_text(json.dumps({"status": "PREPARED_APPEARANCE_TARGET8_CHUNK",
            "sample_ids": group, "jobs": len(jobs), "cameras": [8]}, indent=2) + "\n")
        total += len(jobs)
    result = {"status": "PASS_APPEARANCE_TARGET8_RENDER_PLAN", "samples": 501,
              "jobs": total, "chunks": (len(ids) + a.chunk_samples - 1) // a.chunk_samples,
              "arms": ["vanilla", "raw-topk-plus-feature"], "cameras": [8]}
    (a.output_root / "campaign.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
