"""Prepare original four-stage MV-Pixal3D TEST508 image-scale cascades."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--scale-predictions", type=Path, required=True)
    parser.add_argument("--zero-ss", type=Path, required=True)
    parser.add_argument("--shape-template", type=Path, required=True)
    parser.add_argument("--file4-local-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or not args.file4_local_root.is_dir():
        raise ValueError("fresh plan root and local file4 mirror required")
    freeze = json.loads(args.freeze.read_text())
    zero = json.loads((args.zero_ss / "evaluation.json").read_text())
    contract = json.loads((args.zero_ss / "scale-contract.json").read_text())
    if (zero.get("arm") != "MV-Pixal3D-zero-shot" or zero.get("quick_panel_ids")
            != freeze["population"]["sample_ids"]
            or contract.get("status")
            != "PASS_TEST508_MV_PIXAL3D_ZERO_SHOT_SS_IMAGE_ONLY_SCALE"):
        raise ValueError("complete fair original SS508 artifacts required")
    predictions = [json.loads(line) for line in args.scale_predictions.read_text().splitlines()
                   if line.strip()]
    by_scale = {row["sample_id"]: row for row in predictions}
    receipt = json.loads(args.scale_predictions.with_suffix(".receipt.json").read_text())
    if (len(by_scale) != 508 or receipt["status"] != "PASS_IMAGE_ONLY_SCALE_PREDICTIONS"
            or receipt["output_sha256"] != sha(args.scale_predictions)):
        raise ValueError("frozen TEST508 image-only scale predictions required")
    population = freeze["population"]
    manifest = Path(population["manifest"]["path"])
    rows = {row["sample_id"]: row for row in
            map(json.loads, manifest.read_text().splitlines())}
    if set(rows) != set(population["sample_ids"]) or set(by_scale) != set(rows):
        raise ValueError("TEST508 row/scale identity differs")
    original = freeze["original_stages"]
    for stage in ("ss64", "shape512", "shape1024", "pbr1024"):
        for key in ("config", "checkpoint"):
            spec = original[stage][key]
            if sha(Path(spec["path"])) != spec["SHA256"]:
                raise ValueError("original stage bytes differ: " + stage + "/" + key)
    template = json.loads(args.shape_template.read_text())

    def local(path: str) -> str:
        if not path.startswith("/file4/"):
            raise ValueError("canonical file4 path required: " + path)
        result = args.file4_local_root / path[len("/file4/"):]
        if not result.is_file():
            raise FileNotFoundError(result)
        return str(result)

    plans = []
    for sid in population["sample_ids"]:
        row = rows[sid]
        prediction = by_scale[sid]
        if prediction["M0_or_M1_mesh_or_GT_scale_used_at_inference"] is not False:
            raise ValueError("geometry-derived scale is forbidden")
        bound = dict(row, mesh_scale=float(prediction["predicted_mesh_scale"]))
        plan = copy.deepcopy(template)
        plan.update(
            status="PREPARED_FAIR_MV_PIXAL3D_ZERO_SHOT_TEST508",
            row=bound, object_scale_policy="image-estimated-v1",
            image_scale_predictions=str(args.scale_predictions),
            SS_evaluation=str(args.zero_ss / "evaluation.json"),
            RGB_paths=[local(row["target_anchor"]),
                       *(local(path) for path in row["source_images"])],
            source_geometry=local(row["source_geometry"]),
            target_vertices=local(row["target_vertices"]),
            evaluation_target_geometry=local(row["target_geometry"]),
            seed=population["generation_seeds"][sid],
            output=str(args.output_root / "cases" / sid / "shape"),
            frozen_model_set=str(args.freeze), TEST508_seed_frozen=True,
            M1_derived_scale_supplied_to_generation=False,
            model="MV-Pixal3D-zero-shot", deformation_finetuning_updates=0,
        )
        plan.pop("pilot_rebased_from", None)
        plan["stages"] = {
            stage: {"config": original[stage]["config"]["path"],
                    "checkpoint": original[stage]["checkpoint"]["path"],
                    "original_multiview_step": original[stage]["step"]}
            for stage in ("shape512", "shape1024", "pbr1024")
        }
        plan["original_stage_checkpoints"] = {
            stage: plan["stages"][stage]["checkpoint"] for stage in plan["stages"]}
        plan_path = args.output_root / "plans" / (sid + ".json")
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        plans.append({"sample_id": sid, "plan": str(plan_path)})
    campaign = {
        "status": "PREPARED_FAIR_MV_PIXAL3D_ZERO_SHOT_TEST508",
        "samples": 508, "method": "MV-Pixal3D (zero-shot)",
        "input": "T1+S3+C", "plans": plans,
        "freeze_sha256": sha(args.freeze),
        "scale_prediction_sha256": sha(args.scale_predictions),
        "SS_evaluation_sha256": sha(args.zero_ss / "evaluation.json"),
        "original_steps": {stage: original[stage]["step"] for stage in original},
        "deformation_finetuning_updates": 0,
    }
    (args.output_root / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")
    print(json.dumps({key: campaign[key] for key in
                      ("status", "samples", "method", "input")}), flush=True)


if __name__ == "__main__":
    main()
