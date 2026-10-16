#!/usr/bin/env python3
"""Generate the six frozen DeformNeXt arm configs from production configs."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARMS = ("v1", "b1", "b2", "b3", "c1", "c2")
STAGES = {
    "ss64": {
        "dataset": "DeformNeXtSS64", "world_size": 1,
        "global_batch": 96, "per_gpu": 96, "batch_split": 1,
        "lora_lr": 1e-5, "value_dim": 1024,
        "checkpoint": "/file4/jaehyeok/results/Pixal3D_youngwoo/multiview/train/mixed385-lora-ddp-20261006-151618/ss64/ckpts/denoiser_step0004210.pt",
    },
    "pbr1024": {
        "dataset": "DeformNeXtPBR1024", "world_size": 3,
        "global_batch": 24, "per_gpu": 8, "batch_split": 2,
        "lora_lr": 5e-6, "value_dim": 2048,
        "checkpoint": "/file4/jaehyeok/results/Pixal3D_youngwoo/multiview/train/mixed385-lora-ddp-20261006-151618/pbr1024/ckpts/denoiser_step0016850.pt",
    },
}


def arm_conditioner(arm: str, value_dim: int, q2c: str | None) -> dict:
    config = {
        "arm": arm, "key_dim": 1024, "value_dim": value_dim,
        "implementation": "r9-native-v5",
        "score_temperature": 0.005,
        "query_dim": 1024, "hidden_dim": 256, "topk": 64, "g_max": 0.25,
    }
    if arm == "c2":
        config["deformation_dim"] = 2048
    if arm in {"b2", "b3", "c1", "c2"} and q2c:
        config.update(q2c_checkpoint=q2c, freeze_q2c=True)
    return config


def build(stage: str, arm: str, args) -> dict:
    spec = STAGES[stage]
    training_assets = (getattr(args, "pbr_training_assets", None) or args.training_assets) if stage == "pbr1024" else args.training_assets
    config = json.loads((args.base_config_root / f"{stage}.json").read_text())
    config["_comment"] = (
        f"DeformNeXt-10k retargeting arm {arm}; exact production LoRA continuation, "
        "edited M1 RGB anchor plus three M0 RGB sources; Omega epoch-7 L12 "
        f"addresses and native Pixal values; five epochs on {training_assets:,} retained train assets."
    )
    config["default_output_dir"] = str(args.output_root / stage / arm)
    dataset_args = copy.deepcopy(config["dataset"]["args"])
    for key in (
        "min_aesthetic_score", "num_views", "condition_num_views",
        "min_condition_views", "max_condition_views", "skip_aesthetic_score_datasets",
        "full_pbr",
    ):
        dataset_args.pop(key, None)
    dataset_args.update(
        expected_split="train", verify_files=True, transport_arm=arm
    )
    if arm != "v1":
        dataset_args["transport_implementation"] = "r9-native-v5"
        dataset_args["match_root"] = str(args.data_root / ("matches_raw_top1_v5" if arm == 'b1' else "matches_r9_v5"))
        if arm != 'b1' and args.q2c_checkpoint and Path(args.q2c_checkpoint).is_file():
            import hashlib
            with Path(args.q2c_checkpoint).open('rb') as stream:
                dataset_args["expected_q2c_sha256"] = hashlib.file_digest(stream, 'sha256').hexdigest()
        dataset_args["projection_root"] = str(args.data_root / "projection_maps_world_v4")
    if args.file4_mirror_root is not None:
        dataset_args["path_map"] = {"/file4": str(args.file4_mirror_root)}
    config["dataset"] = {"name": spec["dataset"], "args": dataset_args}
    manifest_root = ((getattr(args, "ss_manifest_root", None) if stage == "ss64" else None)
                     or getattr(args, "manifest_root", None) or args.data_root / "manifests_world_v4")
    if stage == "ss64" and getattr(args, "ss_latent_schema", None):
        config["dataset"]["args"]["expected_ss_schema"] = args.ss_latent_schema
        dataset_args["expected_ss_schema"] = args.ss_latent_schema
    config["data_dir"] = str(manifest_root / "train.jsonl")
    trainer = config["trainer"]["args"]
    updates_per_epoch = math.ceil(training_assets / spec["global_batch"])
    trainer.update(
        max_steps=updates_per_epoch * args.epochs,
        batch_size_per_gpu=spec["per_gpu"], batch_split=spec["batch_split"],
        continuation_ckpt={"denoiser": (
            args.ss_checkpoint if stage == "ss64" else args.pbr_checkpoint
        )},
        finetune_ckpt=None, ema_rate=[], elastic=None,
        deform_conditioner=(None if arm == "v1" else arm_conditioner(arm, spec["value_dim"], args.q2c_checkpoint)),
        num_workers=args.num_workers, prefetch_data=False,
        snapshot_dataset_on_start=False, i_sample=-1,
        i_save=updates_per_epoch if stage == "ss64" else min(64, updates_per_epoch),
        max_checkpoints=args.epochs,
    )
    trainer["optimizer"]["args"]["lr"] = spec["lora_lr"]
    if arm != "v1":
        trainer["model_learning_rates"] = {"deform_conditioner": 1e-4}
    else:
        trainer.pop("model_learning_rates", None)
    trainer["lr_scheduler"] = {
        "name": "LinearWarmupLRScheduler",
        "args": {"warmup_steps": math.ceil(0.05 * updates_per_epoch * args.epochs)},
    }
    trainer["image_cond_model"]["args"]["model_name"] = args.dino
    config["validation"] = {
        "dataset_name": spec["dataset"],
        "dataset_args": {**dataset_args, "expected_split": "validation"},
        "roots_file": str(manifest_root / "validation.jsonl"),
        "assets_per_source": None,
        "max_assets": 32 if stage == "ss64" else 16,
        "every_steps": updates_per_epoch,
        "quick_metrics": {
            "enabled": True,
            "panel_size": 4 if stage == "ss64" else 2,
            "sampling_steps": 12,
            "seed": 42,
            **({"render_resolution": 256} if stage == "pbr1024" else {}),
        },
    }
    config["experiment"] = {
        "dataset": "DeformNeXt-10k", "arm": arm, "epochs": args.epochs,
        "target_modality": "rgb",
        "conditioning_schema": "native-mv-anchor-frame-v2" if arm == 'v1' else "r9-native-residual-v5",
        "release_status": "native" if arm == 'v1' else "audit-hold",
        "validate_at_start": True,
        "training_assets": training_assets,
        "updates_per_epoch": updates_per_epoch,
        "global_batch": spec["global_batch"], "world_size": spec["world_size"],
        "microbatch_per_gpu": spec["per_gpu"] // spec["batch_split"],
        "omega_checkpoint": args.omega_checkpoint,
        "q2c_checkpoint": args.q2c_checkpoint,
    }
    config["num_gpus"] = spec["world_size"]
    if args.wandb_project:
        config.update(
            use_wandb=True,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_name=f"deformnext-{stage}-{arm}",
        )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config-root", type=Path, default=HERE / "base_configs")
    parser.add_argument("--data-root", type=Path, default=Path("/file4/youngwoo/deformation/experiments/retargeting-v1"))
    parser.add_argument("--output-root", type=Path, default=Path("/file4/youngwoo/deformation/experiments/retargeting-v1/runs_r9_v5"))
    parser.add_argument("--config-root", type=Path, default=HERE / "configs")
    parser.add_argument("--ss-manifest-root", type=Path, help="Versioned manifest directory for corrected SS supervision")
    parser.add_argument("--manifest-root", type=Path, help="Shared world-v4 manifest directory")
    parser.add_argument("--ss-latent-schema", default="deformnext-ss64-m1-anchor-v3", help="Required schema for SS manifest and NPZ labels")
    parser.add_argument("--q2c-checkpoint")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--omega-checkpoint", default="/file4/youngwoo/pixal3d/omega-mixed385-20261012/runs/extension/checkpoints/step-023597.pt")
    parser.add_argument("--dino", default="/root/.cache/huggingface/hub/models--camenduru--dinov3-vitl16-pretrain-lvd1689m/snapshots/3c276edd87d6f6e569ff0c4400e086807d0f3881")
    parser.add_argument("--training-assets", type=int, default=9149)
    parser.add_argument("--pbr-training-assets", type=int, help="PBR assets retained after max_tokens filtering")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--file4-mirror-root", type=Path)
    parser.add_argument("--ss-checkpoint", default=STAGES["ss64"]["checkpoint"])
    parser.add_argument("--pbr-checkpoint", default=STAGES["pbr1024"]["checkpoint"])
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity", default="")
    args = parser.parse_args()
    if any(arm in {'b2','b3','c1','c2'} for arm in args.arms):
        if not args.q2c_checkpoint:
            parser.error("semantic recipes require the shared verified R9 --q2c-checkpoint")
        import torch
        from pixal3d.modules.r9_matching import PromotedR9Matcher
        PromotedR9Matcher().load_checkpoint(torch.load(args.q2c_checkpoint, map_location='cpu', weights_only=True))
    args.config_root.mkdir(parents=True, exist_ok=True)
    for stage in STAGES:
        for arm in args.arms:
            config = build(stage, arm, args)
            path = args.config_root / f"{stage}-{arm}.json"
            path.write_text(json.dumps(config, indent=2) + "\n")
            print(path)


if __name__ == "__main__":
    main()
