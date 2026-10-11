"""Resolve native configs and directory mappings for the agreed five-epoch recipe."""
import argparse
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.multiview_finetune.prepare import LATENTS
DATA = REPO / "data/pixal3d/mixed100"
PRETRAINED = REPO / "data/pixal3d/pretrained/single_view"
DINO = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
OUTPUT = REPO / "results/multiview"
STAGES = {
    "ss64": (96, "ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json", "ss_flow_img_dit_1_3B_64_bf16"),
    "shape512": (96, "slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json", "slat_flow_img2shape_dit_1_3B_512_bf16"),
    "shape1024": (24, "slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json", "slat_flow_img2shape_dit_1_3B_1024_bf16"),
    "pbr1024": (24, "slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json", "slat_flow_imgshape2tex_dit_1_3B_1024_bf16"),
}


def stage_config(stage, training_assets, microbatch=1, *, weights_root=PRETRAINED, dino=DINO, output_root=OUTPUT,
                 output_config_dir=None):
    ebs, filename, stem = STAGES[stage]
    recipe = json.loads((REPO / "configs/gen/multiview_finetune" / f"{stage}.json").read_text())
    config = json.loads((REPO / "configs/gen" / recipe["source_config"]).read_text())
    output_config_dir = Path(output_config_dir) if output_config_dir else Path(output_root) / "configs"
    ebs = recipe["effective_batch_size"]
    updates_per_epoch = training_assets // ebs
    total = recipe["epochs"] * updates_per_epoch
    config["_comment"] = ("Our mixed-source single-view-to-multi-view fine-tune; derived from "
        f"{filename} at 3ff6fca. Official SV initialization; K=1..6; "
        "LR halved, 5% linear warmup then constant; original objective/EMA/BF16 preserved. "
        "Depends on prepared common-eligible mixed-source splits and official frozen encoders/decoders.")
    config["dataset"]["args"]["min_condition_views"] = 1
    args = config["trainer"]["args"]
    args.update(max_steps=total, batch_size_per_gpu=ebs, batch_split=ebs // microbatch,
                num_workers=2, prefetch_data=False, snapshot_dataset_on_start=False,
                i_save=updates_per_epoch, i_sample=-1, max_checkpoints=5)
    args["optimizer"]["args"]["lr"] *= recipe["lr_scale"]
    args["lr_scheduler"] = {"name": "LinearWarmupLRScheduler",
                            "args": {"warmup_steps": math.ceil(recipe["warmup_fraction"] * total)}}
    args["finetune_ckpt"] = {"denoiser": str(Path(weights_root) / (stem + ".pt"))}
    args["image_cond_model"]["args"]["model_name"] = str(dino)
    config["default_output_dir"] = str(Path(output_root) / stage)
    config["validation"] = {"dataset_name": config["dataset"]["name"],
                            "dataset_args": config["dataset"]["args"].copy(),
                            "roots_file": str(output_config_dir / f"validation_{stage}_roots.json"),
                            "assets_per_source": None}
    config["experiment"] = {"epochs": recipe["epochs"], "training_assets": training_assets,
                            "updates_per_epoch": updates_per_epoch, "effective_batch_size": ebs,
                            "scope": "recipe only; main training has not been launched"}
    return {"_comment": config["_comment"], **config}


def roots_for_stage(root, stage):
    roots = {"render_cond": str(root / "renders_cond")}
    latent_stage = "shape1024" if stage == "pbr1024" else stage
    roots["ss_latent" if stage == "ss64" else "shape_latent"] = str(root / LATENTS[latent_stage])
    if stage == "pbr1024":
        roots["pbr_latent"] = str(root / LATENTS[stage])
    return roots


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-root", "--data", dest="data", type=Path, default=DATA)
    parser.add_argument("--selection", type=Path, help="External selection profile; default: DATA_ROOT/selection.json")
    parser.add_argument("--weights-root", type=Path, default=PRETRAINED,
                        help="Directory containing converted official denoiser .pt files")
    parser.add_argument("--dino", default=DINO, help="Hugging Face identifier or local DINO snapshot")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--output-config-dir", type=Path,
                        help="Resolved JSON/mappings directory; default: OUTPUT_ROOT/configs")
    parser.add_argument("--microbatches", type=int, nargs=4, default=None,
                        metavar="B", help="Nominal assets per pack: SS64 Shape512 Shape1024 PBR1024")
    args = parser.parse_args()
    args.data = args.data.resolve()
    args.weights_root = args.weights_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.microbatches is None:
        args.microbatches = [json.loads((REPO / "configs/gen/multiview_finetune" / f"{stage}.json").read_text())["microbatch_assets"] for stage in STAGES]
    selection = json.loads((args.selection or args.data / "selection.json").read_text())
    if args.selection and Path(selection["destination_root"]).resolve() != args.data:
        raise ValueError("Selection destination must match --data-root")
    counts = {"train": {}, "validation": {}}
    mappings = {split: {stage: {} for stage in STAGES} for split in counts}
    assets = {"train": set(), "validation": set()}
    for row in selection["batches"]:
        root = args.data / row["source"] / row["shard"] / row["batch"]
        report = json.loads((root / "metadata_report.json").read_text())
        split, source = row["split"], row["source"]
        sha = set(report["eligible_assets"])
        if sha & (assets["train"] | assets["validation"]):
            raise ValueError(f"Repeated asset across selected batches: {root}")
        assets[split].update(sha)
        counts[split][source] = counts[split].get(source, 0) + len(sha)
        key = f"{source}:{row['shard']}:{row['batch']}"
        for stage in STAGES:
            mappings[split][stage][key] = roots_for_stage(root, stage)
    config_root = (args.output_config_dir or args.output_root / "configs").resolve()
    if config_root == (REPO / "configs/gen/multiview_finetune").resolve():
        raise ValueError("Resolved output-config-dir must not overwrite source recipe templates")
    config_root.mkdir(parents=True, exist_ok=True)
    for stage, microbatch in zip(STAGES, args.microbatches):
        config = stage_config(stage, len(assets["train"]), microbatch=microbatch,
                              weights_root=args.weights_root, dino=args.dino,
                              output_root=args.output_root, output_config_dir=config_root)
        config["data_dir"] = json.dumps(mappings["train"][stage])
        config["validation"]["roots_file"] = str(config_root / f"validation_{stage}_roots.json")
        (config_root / (stage + ".json")).write_text(json.dumps(config, indent=2) + "\n")
        for split in mappings:
            (config_root / (split + "_" + stage + "_roots.json")).write_text(json.dumps(mappings[split][stage], indent=2) + "\n")
    summary = {"sources": counts, "training_assets": len(assets["train"]),
               "validation_assets": len(assets["validation"]), "asset_overlap": 0,
               "config_root": str(config_root)}
    (config_root / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
