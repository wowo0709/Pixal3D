"""Bind six matched SS ablation arms to the frozen TEST508 cascade."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ARMS = ("M_FIXED_SPATIAL", "M_FIXED_RAW", "M_FIXED_D1_COS", "C_DECODER",
        "A_FIXED_UNIFORM", "A_FIXED_NO_R")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--ss-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise ValueError("fresh geometry campaign root required")
    source = json.loads(args.source_campaign.read_text())
    selection = json.loads(args.selection.read_text())
    if (source.get("status") != "PREPARED_FROZEN_V1_SELECTED_TEST508_IMAGE_SCALE"
            or source.get("samples") != 508
            or tuple(selection.get("additional_matched_SS_controls", ()))
                != tuple(arm for arm in ARMS if arm != "C_DECODER")
            or selection.get("actual_TEST_forward_or_quality_read") is not False):
        raise ValueError("frozen TEST508 template and development-only control selection required")
    templates = [row for row in source["plans"] if row["arm"] == "V1"]
    if len(templates) != 508 or len({row["sample_id"] for row in templates}) != 508:
        raise ValueError("exactly one V1 cascade template per TEST asset required")
    for arm in ARMS:
        receipt = args.ss_root / arm / "complete.json"
        evaluation = args.ss_root / arm / "evaluation.json"
        if not receipt.is_file() or not evaluation.is_file():
            raise FileNotFoundError(f"complete TEST508 SS artifacts required for {arm}")
        complete = json.loads(receipt.read_text())
        if (complete.get("status") != "PASS_TEST508_PREREGISTERED_ABLATION_IMAGE_ONLY_SCALE"
                or complete.get("arm") != arm or complete.get("samples") != 508):
            raise ValueError("SS completion receipt differs for " + arm)

    plans = []
    for template in templates:
        sid = template["sample_id"]
        original = json.loads(Path(template["shape_plan"]).read_text())
        if (original.get("object_scale_policy") != "image-estimated-v1"
                or original.get("M1_derived_scale_supplied_to_generation") is not False):
            raise ValueError("source cascade template is not the frozen image-scale protocol")
        for arm in ARMS:
            plan = copy.deepcopy(original)
            plan.update(
                status="PREPARED_TEST508_PREREGISTERED_ABLATION_GEOMETRY",
                output=str(args.output_root / "cases" / arm / sid / "shape"),
                SS_evaluation=str(args.ss_root / arm / "evaluation.json"),
                frozen_model_set=str(args.selection),
                ablation_arm=arm,
                ablation_selection_SHA256=sha(args.selection),
                downstream_shape_and_vanilla_PBR_identical=True,
                render_metrics_requested=False,
            )
            destination = args.output_root / "plans" / arm / f"{sid}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(plan, indent=2) + "\n")
            plans.append({"sample_id": sid, "arm": arm, "plan": str(destination)})
    if len(plans) != 508 * len(ARMS):
        raise ValueError("six-arm TEST508 geometry plan closure failed")
    result = {
        "status": "PREPARED_TEST508_PREREGISTERED_ABLATION_GEOMETRY",
        "samples": 508,
        "arms": list(ARMS),
        "plans": plans,
        "selection": str(args.selection),
        "selection_SHA256": sha(args.selection),
        "source_campaign": str(args.source_campaign),
        "source_campaign_SHA256": sha(args.source_campaign),
        "scale_policy": "image-estimated-v1",
        "registration": "input-source-M0-only Sim(3)",
        "training_updates": 0,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "campaign.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "plans": len(plans)}), flush=True)


if __name__ == "__main__":
    main()
