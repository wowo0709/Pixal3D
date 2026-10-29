"""Evaluate a preregistered SS ablation checkpoint on TEST508.

This is the matched TEST counterpart of the development-only v554 control
selection.  It never resumes training, never supplies target scale, and
captures the exact SS prediction used by the downstream geometry cascade.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("ATTN_BACKEND", "sdpa")

import numpy as np
import torch


ALLOWED_CONTROLS = {
    "M_FIXED_SPATIAL": "fixed-correspondence",
    "M_FIXED_RAW": "fixed-correspondence",
    "M_FIXED_D1_COS": "fixed-correspondence",
    "A_FIXED_UNIFORM": "shared-evidence-ablation",
    "A_FIXED_NO_R": "shared-evidence-ablation",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runtime-family", choices=tuple(sorted(set(ALLOWED_CONTROLS.values()))),
                        required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--panel-file", type=Path, required=True)
    parser.add_argument("--seed-map-file", type=Path, required=True)
    parser.add_argument("--scale-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generation-batch", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.generation_batch <= 8:
        raise ValueError("fresh output and bounded native generation batch required")

    selection = json.loads(args.selection.read_text())
    if (selection.get("status") != "SELECTED_FROM_ACTUAL_DEVELOPMENT_ONLY"
            or selection.get("heldout_quality_read") is not False
            or selection.get("actual_TEST_forward_or_quality_read") is not False):
        raise ValueError("development-only v554 selection required")
    controls = selection.get("additional_matched_SS_controls")
    if controls != list(ALLOWED_CONTROLS):
        raise ValueError("the five preregistered matched controls changed")

    bundle = json.loads(args.bundle.read_text())
    config_path = Path(bundle["native_config"])
    checkpoint = Path(bundle["checkpoint"])
    conditioner = Path(bundle["conditioner_checkpoint"])
    for path in (config_path, checkpoint, conditioner, args.test_manifest,
                 args.panel_file, args.seed_map_file, args.scale_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text())

    if args.runtime_family == "fixed-correspondence":
        from .evaluate_fixed_correspondence_checkpoint import validate_bundle
        from .fixed_correspondence_native_runtime import parameter_groups, selected_trainer, state_sha
    else:
        from .evaluate_shared_evidence_ablation_checkpoint import validate_bundle
        from .shared_evidence_ablation_native_runtime import parameter_groups, selected_trainer, state_sha
    variant = validate_bundle(bundle, config)
    if variant not in ALLOWED_CONTROLS or ALLOWED_CONTROLS[variant] != args.runtime_family:
        raise ValueError("bundle is not the requested preregistered runtime/control")

    population = json.loads(args.panel_file.read_text())["sample_ids"]
    seeds = json.loads(args.seed_map_file.read_text())
    rows = {row["sample_id"]: row for row in map(json.loads,
                                                   args.test_manifest.read_text().splitlines())}
    if (len(population) != 508 or len(population) != len(set(population))
            or set(population) != set(rows) or not set(population) <= set(seeds)
            or any(type(seeds[sid]) is not int for sid in population)):
        raise ValueError("exact TEST508 population and generation seeds required")

    import joblib
    from .probe_image_estimated_scale import features
    scale_estimator = joblib.load(args.scale_model)
    if (scale_estimator.get("feature_schema") != "RGBA128-mask-bbox-area-camera-v1"
            or set(population) & set(scale_estimator["training_sample_ids"])):
        raise ValueError("frozen image-only scale estimator or TRAIN/TEST separation differs")
    path_prefix = Path(config["validation"]["dataset_args"]["path_map"]["/file4"])
    vectors = []
    for sid in population:
        row = rows[sid]
        permitted = {key: row[key] for key in
                     ("target_anchor", "source_images", "transform_matrix", "camera_angle_x")}
        vectors.append(features(permitted, path_prefix))
    predicted_scales = np.exp(scale_estimator["estimator"].predict(
        np.asarray(vectors, dtype=np.float32)))
    image_scales = {sid: float(value) for sid, value in zip(population, predicted_scales)}
    if any(not .4 < value < 1.6 for value in image_scales.values()):
        raise ValueError("image-only scale prediction outside contract")

    from PIL import Image
    from pixal3d import datasets, models
    from train import setup_rng
    from . import quick_metrics as quick
    from .evaluate_ss_artifact_subset import capture_ss_artifacts
    from .projection_maps import anchor_patch_indices

    original = datasets.DeformNeXtSS64
    grid = np.indices((16, 16, 16)).reshape(3, -1).T
    events: list[str] = []

    class ImageEstimatedScaleSS(original):
        def __getitem__(self, index):
            item = super().__getitem__(index)
            row = self.rows[index]
            sid = row["sample_id"]
            scale = image_scales[sid]
            item["mesh_scale"] = torch.tensor(scale, dtype=torch.float32)
            canonical_row = dict(row, mesh_scale=scale)
            patch = anchor_patch_indices(canonical_row, grid, 16).reshape(16, 16, 16)
            item["transport_patch_grid"] = torch.from_numpy(patch).long()
            events.append(sid)
            return item

    setup_rng(0, config["training_seed"])
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    dataset_args = copy.deepcopy(config["validation"]["dataset_args"])
    dataset_args["expected_split"] = "test"
    test_projection = args.test_manifest.parents[2] / "projection_maps_world_v4"
    if any(not (test_projection / f"{sid}.npz").is_file() for sid in population):
        raise FileNotFoundError("frozen TEST508 projection cache is incomplete")
    dataset_args["projection_root"] = str(test_projection)
    dataset = ImageEstimatedScaleSS(args.test_manifest, **dataset_args)

    validation = copy.deepcopy(config["validation"])
    validation["roots_file"] = str(args.test_manifest)
    validation["dataset_args"] = dataset_args
    validation["quick_metrics"].update(
        panel_size=508, panel_file=str(args.panel_file), seed_map_file=str(args.seed_map_file),
        generation_batch_size=args.generation_batch, log_progress=True)
    with torch.device("meta"):
        model_defs = {name: getattr(models, spec["name"])(**spec["args"])
                      for name, spec in config["models"].items()}
    trainer_args = copy.deepcopy(config["trainer"]["args"])
    trainer_args.update(continuation_ckpt={"denoiser": str(checkpoint)},
                        batch_size_per_gpu=1, batch_split=1, num_workers=0,
                        prefetch_data=False, max_steps=1)
    args.output.mkdir(parents=True, exist_ok=False)
    trainer = selected_trainer(variant)(model_defs, dataset, **trainer_args,
        output_dir=str(args.output / "loader"), load_dir=None, step=None, wandb_run=None)
    trainer.deform_conditioner.load_state_dict(
        torch.load(conditioner, map_location="cpu", weights_only=True), strict=True)
    lora, source = parameter_groups(trainer, variant)
    before = state_sha(list(lora.items()) + list(source.items()))
    if trainer.optimizer.state or trainer.step != 0:
        raise ValueError("evaluation must not resume optimizer or update weights")
    for loaded in trainer.models.values():
        loaded.eval()
    trainer.deform_conditioner.eval()
    trainer.p_uncond = 0.

    captured: list[dict] = []
    with capture_ss_artifacts(quick, args.output, captured), torch.no_grad(), \
            torch.autocast("cuda", dtype=trainer.mix_precision_dtype):
        metrics, panel_image, actual_ids = quick.evaluate_quick(trainer, dataset, validation)
    if (actual_ids != population or len(captured) != 508 or trainer.optimizer.state
            or before != state_sha(list(lora.items()) + list(source.items()))
            or set(events) != set(population)):
        raise ValueError("TEST evaluation changed weights, population, or artifact coverage")

    result = {
        "status": "PASS_TEST508_PREREGISTERED_ABLATION_IMAGE_ONLY_SCALE",
        "arm": variant,
        "runtime_family": args.runtime_family,
        "quick_metrics": metrics,
        "quick_panel_ids": actual_ids,
        "per_asset": trainer._quick_per_asset,
        "optimizer_updates_during_evaluation": 0,
        "checkpoint": str(checkpoint),
        "checkpoint_SHA256": sha(checkpoint),
        "conditioner_checkpoint": str(conditioner),
        "conditioner_SHA256": sha(conditioner),
        "arm_config": str(config_path),
        "arm_config_SHA256": sha(config_path),
        "native_bundle": str(args.bundle),
        "native_bundle_SHA256": sha(args.bundle),
        "preregistered_selection": str(args.selection),
        "preregistered_selection_SHA256": sha(args.selection),
        "scale_model": str(args.scale_model),
        "scale_model_SHA256": sha(args.scale_model),
        "instance_scale_supplied_to_generation": False,
        "target_M1_scale_not_used_for_inference": True,
        "source_M0_mesh_used_in_generation": False,
        "canonical_projection_recomputed": True,
        "matching_cache_unchanged": True,
        "image_estimated_scales": image_scales,
        "native_SS_artifacts": {
            "artifacts": captured,
            "archived_asset_count": len(captured),
            "GT_SS_is_for_scoring_only": True,
        },
    }
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    Image.fromarray(panel_image).save(args.output / "panel.png")
    (args.output / "complete.json").write_text(json.dumps({
        "status": result["status"], "arm": variant, "samples": 508,
        "voxel_iou": metrics["voxel_iou"], "optimizer_updates": 0,
    }, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "arm": variant,
                      "samples": 508, "voxel_iou": metrics["voxel_iou"]}), flush=True)


if __name__ == "__main__":
    main()
