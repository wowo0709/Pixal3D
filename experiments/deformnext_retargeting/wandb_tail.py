#!/usr/bin/env python3
"""Backfill and follow DeformNeXt runtime JSONL metrics in W&B."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import wandb


def flatten(prefix: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result = {}
    for key, child in value.items():
        name = f"{prefix}/{key}" if prefix else key
        result.update(flatten(name, child))
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def train_metrics(row: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "train/step": row["step"],
        "train/epoch": row["step"] / ARGS.steps_per_epoch,
        "train/effective_batch_size": row["effective_batch_size"],
        "train/load_seconds": row["load_seconds"],
        "train/compute_seconds": row["compute_seconds"],
        "train/peak_allocated_gib": row["peak_allocated_gib"],
        "train/reserved_gib": row["reserved_gib"],
        "train/lr": row["lr_next_update"],
    }
    payload.update(flatten("train", row["loss"]))
    return payload


def validation_metrics(row: dict[str, Any]) -> dict[str, Any]:
    sources = row["sources"]
    total = sum(source["assets"] for source in sources)
    payload = {
        "validation/step": row["step"],
        "validation/epoch": row["step"] / ARGS.steps_per_epoch,
        "validation/assets": total,
    }
    for source in sources:
        payload[f"validation/{source['source']}/K4_loss"] = source["K4_loss"]
        if source.get("K1_loss") is not None:
            payload[f"validation/{source['source']}/K1_loss"] = source["K1_loss"]
    if total:
        payload["validation/K4_loss"] = sum(
            source["K4_loss"] * source["assets"] for source in sources
        ) / total
        k1_sources = [source for source in sources if source.get("K1_loss") is not None]
        k1_assets = sum(source["assets"] for source in k1_sources)
        if k1_assets:
            payload["validation/K1_loss"] = sum(
                source["K1_loss"] * source["assets"] for source in k1_sources
            ) / k1_assets
    payload.update({
        f"validation/{key}": value
        for key, value in row.get("quick_metrics", {}).items()
    })
    return payload


def sync(run: wandb.sdk.wandb_run.Run, state: dict[str, int]) -> bool:
    changed = False
    for row in read_jsonl(ARGS.run_dir / "runtime_batch.jsonl"):
        if row["step"] <= state["train_step"]:
            continue
        run.log(train_metrics(row))
        state["train_step"] = row["step"]
        changed = True
    for row in read_jsonl(ARGS.run_dir / "validation.jsonl"):
        if row["step"] <= state["validation_step"]:
            continue
        run.log(validation_metrics(row))
        state["validation_step"] = row["step"]
        changed = True
    if changed:
        ARGS.state_file.write_text(json.dumps(state, indent=2) + "\n")
    return changed


def main() -> None:
    config_path = ARGS.run_dir / "config.json"
    config = json.loads(config_path.read_text())
    state = {"train_step": 0, "validation_step": 0}
    if ARGS.state_file.exists():
        state.update(json.loads(ARGS.state_file.read_text()))
    run = wandb.init(
        entity=ARGS.entity,
        project=ARGS.project,
        name=ARGS.name,
        id=ARGS.run_id,
        resume="allow",
        group=ARGS.group,
        job_type=ARGS.stage,
        tags=["DeformNeXt-10k", ARGS.stage, ARGS.arm],
        config=config,
        dir=str(ARGS.wandb_dir),
    )
    run.define_metric("train/step")
    run.define_metric("train/*", step_metric="train/step")
    run.define_metric("validation/step")
    run.define_metric("validation/*", step_metric="validation/step")
    try:
        while True:
            sync(run, state)
            if ARGS.once:
                break
            progress_path = ARGS.run_dir / "progress.json"
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
            if progress.get("stage") in {"completed", "stopped"}:
                sync(run, state)
                break
            time.sleep(ARGS.poll_seconds)
    finally:
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--steps-per-epoch", type=int, required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--group", default="deformnext-six-arm")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--once", action="store_true", help="Backfill available rows and exit")
    parser.add_argument("--wandb-dir", type=Path, default=Path("/root/dev/.cache/wandb"))
    parser.add_argument("--state-file", type=Path)
    ARGS = parser.parse_args()
    ARGS.wandb_dir.mkdir(parents=True, exist_ok=True)
    if ARGS.state_file is None:
        ARGS.state_file = ARGS.run_dir / ".wandb-tail-state.json"
    main()
