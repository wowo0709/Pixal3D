"""Generate fair TEST508 SS artifacts for the original MV-Pixal3D checkpoint.

The model sees exactly the edited target RGB at camera 8 and the three source
RGBs at cameras 9--11.  Object scale is supplied only by the frozen image-only
estimator used by the FT and Full rows; M0/M1 geometry is never read for
inference.  This is evaluation-only and performs no optimizer update.
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


def sha(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def tensor_state_sha(named_tensors) -> str:
    state = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        value = tensor.detach().cpu().contiguous()
        state.update((name + str(value.dtype) + repr(tuple(value.shape))).encode())
        state.update(value.view(torch.uint8).numpy().tobytes())
    return state.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--zero-config", type=Path, required=True)
    parser.add_argument("--scale-model", type=Path, required=True)
    parser.add_argument("--scale-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generation-batch", type=int, default=8)
    args = parser.parse_args()
    if (args.output.exists() or not 1 <= args.generation_batch <= 8
            or os.environ.get("BENCHMARK_ASSIGNED_GPU")
            != os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise ValueError("fresh output, bounded batch, and explicit physical GPU required")

    freeze = json.loads(args.freeze.read_text())
    if freeze.get("status") != "FROZEN_FROM_DEVELOPMENT_ONLY":
        raise ValueError("development-only frozen population required")
    population = freeze["population"]
    manifest = Path(population["manifest"]["path"])
    panel = population["sample_ids"]
    seeds = population["generation_seeds"]
    if (sha(manifest) != population["manifest"]["SHA256"]
            or len(panel) != 508 or len(set(panel)) != 508):
        raise ValueError("frozen TEST508 population/seed closure differs")
    args.output.mkdir(parents=True, exist_ok=False)
    panel_file = args.output / "population-panel.json"
    seed_file = args.output / "population-seeds.json"
    panel_file.write_text(json.dumps({"sample_ids": panel}, indent=2) + "\n")
    seed_file.write_text(json.dumps(seeds, indent=2) + "\n")

    original = freeze["original_stages"]["ss64"]
    config_path = Path(original["config"]["path"])
    checkpoint = Path(original["checkpoint"]["path"])
    for spec in (original["config"], original["checkpoint"]):
        if not Path(spec["path"]).is_file() or sha(Path(spec["path"])) != spec["SHA256"]:
            raise ValueError("original MV-Pixal3D SS input differs: " + spec["path"])
    if sha(args.zero_config) != "42496e6ed6abbb55ef315914650caeb55beb13ec2ad4fac383744e1d9e8b4dca":
        raise ValueError("validated DeformNeXt zero-shot adapter config differs")

    config = json.loads(args.zero_config.read_text())
    if (config["trainer"]["name"] != "ImageConditionedProjFlowMatchingCFGTrainer"
            or config["trainer"]["args"].get("deform_conditioner") is not None
            or config["dataset"]["name"] != "DeformNeXtSS64"):
        raise ValueError("unaltered original MV trainer and vanilla DeformNeXt adapter required")

    import joblib
    from .probe_image_estimated_scale import features

    estimator = joblib.load(args.scale_model)
    if (estimator.get("feature_schema") != "RGBA128-mask-bbox-area-camera-v1"
            or set(panel) & set(estimator["training_sample_ids"])):
        raise ValueError("frozen image-only scale estimator or TEST separation differs")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    by_id = {row["sample_id"]: row for row in rows}
    if len(rows) != 508 or set(by_id) != set(panel):
        raise ValueError("TEST508 manifest identity differs")
    path_prefix = Path(config["validation"]["dataset_args"]["path_map"]["/file4"])
    vectors = []
    for sid in panel:
        row = by_id[sid]
        permitted = {key: row[key] for key in
                     ("target_anchor", "source_images", "transform_matrix", "camera_angle_x")}
        vectors.append(features(permitted, path_prefix))
    values = np.exp(estimator["estimator"].predict(np.asarray(vectors, dtype=np.float32)))
    image_scales = {sid: float(value) for sid, value in zip(panel, values)}
    if any(not .4 < value < 1.6 for value in image_scales.values()):
        raise ValueError("image-only scale prediction outside frozen contract")
    declared = [json.loads(line) for line in args.scale_predictions.read_text().splitlines()
                if line.strip()]
    declared_by_id = {row["sample_id"]: row for row in declared}
    declared_receipt = json.loads(
        args.scale_predictions.with_suffix(".receipt.json").read_text())
    if (len(declared_by_id) != 508 or set(declared_by_id) != set(panel)
            or declared_receipt.get("status") != "PASS_IMAGE_ONLY_SCALE_PREDICTIONS"
            or declared_receipt.get("output_sha256") != sha(args.scale_predictions)
            or any(row["M0_or_M1_mesh_or_GT_scale_used_at_inference"] is not False
                   for row in declared)
            or any(not np.isclose(image_scales[sid],
                                  float(declared_by_id[sid]["predicted_mesh_scale"]),
                                  rtol=0, atol=1e-7) for sid in panel)):
        raise ValueError("recomputed and published TEST508 image-only scales differ")

    from pixal3d import datasets, models, trainers
    from PIL import Image
    from train import setup_rng
    from . import quick_metrics as quick
    from .evaluate_ss_artifact_subset import capture_ss_artifacts

    base_dataset = datasets.DeformNeXtSS64
    scale_events = []

    class FairZeroSS(base_dataset):
        def __getitem__(self, index):
            item = super().__getitem__(index)
            sid = self.rows[index]["sample_id"]
            item["mesh_scale"] = torch.tensor(image_scales[sid], dtype=torch.float32)
            scale_events.append(sid)
            return item

    setup_rng(0, config["training_seed"])
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    dataset_args = copy.deepcopy(config["validation"]["dataset_args"])
    dataset_args["expected_split"] = "test"
    dataset = FairZeroSS(manifest, **dataset_args)
    validation = copy.deepcopy(config["validation"])
    validation["roots_file"] = str(manifest)
    validation["dataset_args"] = dataset_args
    validation["quick_metrics"].update(
        panel_size=508, panel_file=str(panel_file),
        seed_map_file=str(seed_file),
        generation_batch_size=args.generation_batch, log_progress=True)
    with torch.device("meta"):
        model_defs = {name: getattr(models, spec["name"])(**spec["args"])
                      for name, spec in config["models"].items()}
    trainer_args = copy.deepcopy(config["trainer"]["args"])
    trainer_args.update(continuation_ckpt={"denoiser": str(checkpoint)},
                        batch_size_per_gpu=1, batch_split=1, num_workers=0,
                        prefetch_data=False, max_steps=1)
    trainer = getattr(trainers, config["trainer"]["name"])(
        model_defs, dataset, **trainer_args, output_dir=str(args.output / "loader"),
        load_dir=None, step=None, wandb_run=None)
    if trainer.optimizer.state or trainer.step != 0 or trainer.deform_conditioner is not None:
        raise ValueError("zero-shot evaluation must start without optimizer state or conditioner")
    model = trainer.models["denoiser"]
    model.eval()
    trainer.p_uncond = 0.
    lora = [(name, value) for name, value in model.state_dict().items() if "lora_" in name]
    if not lora:
        raise ValueError("original multi-view LoRA tensors required")
    before = tensor_state_sha(lora)
    captured = []
    with capture_ss_artifacts(quick, args.output, captured), torch.no_grad(), \
            torch.autocast("cuda", dtype=trainer.mix_precision_dtype):
        metrics, panel_image, actual_ids = quick.evaluate_quick(
            trainer, dataset, validation)
    after = tensor_state_sha([(name, value) for name, value in model.state_dict().items()
                              if "lora_" in name])
    if (trainer.optimizer.state or trainer.step != 0 or before != after
            or actual_ids != panel or len(captured) != 508
            or not set(panel) <= set(scale_events)):
        raise ValueError("zero-shot evaluation changed weights, scale coverage, or population")

    result = {
        "status": "PASS", "stage": "ss64", "step": 0,
        "arm": "MV-Pixal3D-zero-shot", "split": "test",
        "checkpoint": str(checkpoint), "conditioner_checkpoint": None,
        "arm_config": str(args.zero_config), "conditioning_mode": "edited-rgb",
        "target_modality": "rgb", "input_states": ["M1", "M0", "M0", "M0"],
        "input_camera_ids": [8, 9, 10, 11], "diagnostic_only": False,
        "quick_metrics": metrics, "quick_panel_ids": actual_ids,
        "per_asset": trainer._quick_per_asset, "optimizer_updates": 0,
        "optimizer_updates_during_evaluation": 0,
        "generation_batch_size": args.generation_batch,
        "manifest_sha256": sha(manifest), "original_multiview_step": 4210,
        "deformation_finetuning_updates": 0,
        "native_SS_artifacts": {"artifacts": captured,
            "archived_asset_count": len(captured), "GT_SS_is_for_scoring_only": True},
    }
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    Image.fromarray(panel_image).save(args.output / "panel.png")
    contract = {
        "status": "PASS_TEST508_MV_PIXAL3D_ZERO_SHOT_SS_IMAGE_ONLY_SCALE",
        "samples": 508, "input": "T1+S3+C", "model": "original-MV-Pixal3D",
        "frozen_checkpoint_sha256": original["checkpoint"]["SHA256"],
        "image_estimator_checkpoint": str(args.scale_model),
        "image_estimator_sha256": sha(args.scale_model),
        "published_scale_predictions": str(args.scale_predictions),
        "published_scale_predictions_sha256": sha(args.scale_predictions),
        "image_estimated_scales": image_scales,
        "target_M1_scale_not_used_for_inference": True,
        "source_M0_mesh_used_in_generation": False,
        "legacy_target_latent_used_as_score_label_only": True,
        "optimizer_updates": 0,
    }
    (args.output / "scale-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    completion = {"status": contract["status"], "samples": 508,
                  "voxel_iou": metrics["voxel_iou"], "training_updates": 0}
    (args.output / "complete.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
