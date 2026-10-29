"""Prepare common-501 appearance-only controls on frozen Selected geometry."""
from __future__ import annotations

import argparse
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
    parser.add_argument("--selected-campaign", type=Path, required=True)
    parser.add_argument("--common-subset", type=Path, required=True)
    parser.add_argument("--vanilla-root", type=Path, required=True)
    parser.add_argument("--raw-topk-root", type=Path, required=True)
    parser.add_argument("--remote-shape-base", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise ValueError("fresh appearance plan root required")
    if not args.remote_shape_base.startswith(
        "/file4/youngwoo/deformation/experiments/retargeting-v1/"
    ):
        raise ValueError("checksum-archived file4 Selected shape root required")

    campaign = json.loads(args.selected_campaign.read_text())
    if (campaign.get("status") != "PREPARED_FROZEN_V1_SELECTED_TEST508_IMAGE_SCALE"
            or campaign.get("samples") != 508):
        raise ValueError("frozen image-scale TEST508 cascade plan required")
    selected = {row["sample_id"]: row for row in campaign["plans"]
                if row.get("arm") == "SELECTED"}
    ids = json.loads(args.common_subset.read_text())["sample_ids"]
    if len(ids) != 501 or len(set(ids)) != 501 or not set(ids) <= selected.keys():
        raise ValueError("frozen common-501 Selected population required")

    def bundle(native: Path, mode: str) -> dict[str, Path]:
        files = {
            "config": native / "config.json",
            "checkpoint": native / "ckpts/denoiser_step0001900.pt",
            "completion": native / "stage-completed.rank0.json",
        }
        if mode != "vanilla":
            files["conditioner"] = (
                native / "ckpts/deform_conditioner_step0001900.pt"
            )
        if not all(path.is_file() for path in files.values()):
            raise FileNotFoundError(f"complete {mode} native1900 bundle required")
        config = json.loads(files["config"].read_text())
        completion = json.loads(files["completion"].read_text())
        if (config.get("output_dir") != str(native)
                or config.get("experiment", {}).get("PBR_source_mode") != mode
                or completion.get("status")
                    != "PASS_ACTUAL_ORIGINAL_PBR16850_FRESH_NATIVE1900_STAGE"
                or completion.get("mode") != mode
                or completion.get("actual_optimizer_updates") != 1900
                or not completion.get("stage_training_complete")):
            raise ValueError(f"{mode} completed training lineage differs")
        return files

    vanilla_files = bundle(args.vanilla_root, "vanilla")
    topk_files = bundle(args.raw_topk_root, "raw-topk-plus-feature")

    args.output_root.mkdir(parents=True)
    plans = []
    for sid in ids:
        source = selected[sid]
        template_path = Path(source["top1_pbr_plan"])
        template = json.loads(template_path.read_text())
        if (template.get("mode") != "top1-only"
                or template.get("checkpoint_kind") != "trained-native1900"
                or template.get("row", {}).get("sample_id") != sid):
            raise ValueError("frozen Top-1 replay template differs: " + sid)
        vanilla_shape = args.output_root / "work" / "vanilla" / sid / "shape"
        topk_shape = args.output_root / "work" / "topk" / sid / "shape"
        vanilla_output = args.output_root / "cases" / "vanilla" / sid / "replay"
        topk_output = args.output_root / "cases" / "topk" / sid / "replay"
        stripped = dict(template)
        for key in ("source_SHA256", "input_SHA256", "closed_inputs",
                    "canonical_local_root", "cold_reference", "native_guides"):
            stripped.pop(key, None)
        vanilla = dict(stripped)
        vanilla.update(
            name="common501-vanilla-paired-appearance-replay",
            mode="vanilla",
            generated_shape_root=str(vanilla_shape),
            PBR_config=str(vanilla_files["config"]),
            PBR_checkpoint=str(vanilla_files["checkpoint"]),
            conditioner_checkpoint=None,
            stage_completion=str(vanilla_files["completion"]),
            output=str(vanilla_output),
        )
        vanilla_plan = args.output_root / "plans" / "vanilla" / f"{sid}.json"
        vanilla_plan.parent.mkdir(parents=True, exist_ok=True)
        vanilla_plan.write_text(json.dumps(vanilla, indent=2) + "\n")
        topk = dict(stripped)
        topk.update(
            name="common501-raw-topk-plus-feature-appearance-replay",
            mode="raw-topk-plus-feature",
            generated_shape_root=str(topk_shape),
            PBR_config=str(topk_files["config"]),
            PBR_checkpoint=str(topk_files["checkpoint"]),
            conditioner_checkpoint=str(topk_files["conditioner"]),
            stage_completion=str(topk_files["completion"]),
            output=str(topk_output),
        )
        plan_path = args.output_root / "plans" / "topk" / f"{sid}.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(topk, indent=2) + "\n")
        plans.append({
            "sample_id": sid,
            "vanilla_shape_root": str(vanilla_shape),
            "topk_shape_root": str(topk_shape),
            "remote_shape": f"{args.remote_shape_base}/SELECTED/{sid}/shape",
            "vanilla_root": str(args.output_root / "cases" / "vanilla" / sid),
            "topk_root": str(args.output_root / "cases" / "topk" / sid),
            "vanilla_plan": str(vanilla_plan),
            "topk_plan": str(plan_path),
            "top1_template_sha256": sha(template_path),
        })
    result = {
        "status": "PREPARED_COMMON501_PAIRED_NOISE_APPEARANCE_ABLATION_ON_SELECTED_GEOMETRY",
        "samples": 501,
        "arms": ["vanilla", "raw-topk-plus-feature"],
        "selected_geometry_fixed": True,
        "source_registration": "input_source_M0_only",
        "source_cameras": [9, 10],
        "plans": plans,
        "paired_texture_noise": True,
        "vanilla_bundle": {key: {"path": str(path), "sha256": sha(path)}
                           for key, path in vanilla_files.items()},
        "raw_topk_bundle": {key: {"path": str(path), "sha256": sha(path)}
                            for key, path in topk_files.items()},
        "selected_campaign_sha256": sha(args.selected_campaign),
        "common_subset_sha256": sha(args.common_subset),
        "training_updates": 0,
    }
    (args.output_root / "campaign.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({key: result[key] for key in ("status", "samples", "arms")}),
          flush=True)


if __name__ == "__main__":
    main()
