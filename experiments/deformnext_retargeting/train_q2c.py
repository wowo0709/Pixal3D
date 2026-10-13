#!/usr/bin/env python3
"""Warm up the one-layer Q2C decoder from topology-exact patch labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from pixal3d.modules.deform_conditioner import DeformationAwareConditioner


def remap(path: str, mappings: list[tuple[str, str]]) -> Path:
    for source, target in mappings:
        if path == source or path.startswith(source.rstrip("/") + "/"):
            return Path(target + path[len(source):])
    return Path(path)


def loss_for_cache(
    model, path: Path, topk: int
) -> tuple[torch.Tensor, int, int, int, int]:
    with np.load(path) as cache:
        target = torch.from_numpy(np.asarray(cache["target_keys"]).copy()).cuda().float()
        source = torch.from_numpy(np.asarray(cache["source_keys"]).copy()).cuda().float()
        valid = torch.from_numpy(np.asarray(cache["source_valid"]).copy()).cuda().bool()
        positives = torch.from_numpy(np.asarray(cache["spatial_indices"]).copy()).cuda().long()
    views, sites, dim = source.shape
    flat = source.reshape(views * sites, dim)
    similarity = F.normalize(target, dim=-1) @ F.normalize(flat, dim=-1).T
    similarity = similarity.masked_fill(~valid.reshape(-1)[None], -torch.inf)
    cosine, indices = similarity.topk(topk, dim=-1)
    candidates = flat[indices]
    logits = model._q2c_logits(
        target[None], candidates[None], cosine[None]
    ).squeeze(0)
    positive_flat = positives.T + torch.arange(views, device="cuda")[None] * sites
    mask = (indices[..., None] == positive_flat[:, None]).any(dim=-1)
    eligible = mask.any(dim=-1)
    total = int(mask.shape[0])
    raw_correct = int(mask[:, 0].sum())
    if not bool(eligible.any()):
        return logits.sum() * 0.0, 0, 0, raw_correct, total
    selected = logits[eligible]
    selected_mask = mask[eligible]
    numerator = torch.logsumexp(selected.masked_fill(~selected_mask, -torch.inf), dim=-1)
    denominator = torch.logsumexp(selected, dim=-1)
    prediction = selected.argmax(dim=-1, keepdim=True)
    correct = int(selected_mask.gather(1, prediction).sum())
    return (
        (denominator - numerator).mean(), int(eligible.sum()), correct,
        raw_correct, total,
    )


@torch.no_grad()
def validate(model, rows, mappings, topk: int) -> dict[str, float | int]:
    model.eval()
    loss_sum = 0.0
    eligible_sum = 0
    correct_sum = 0
    raw_correct_sum = 0
    total_sum = 0
    for row in rows:
        cache = remap(row["transport_cache"], mappings)
        loss, queries, correct, raw_correct, total = loss_for_cache(
            model, cache, topk
        )
        loss_sum += float(loss) * queries
        eligible_sum += queries
        correct_sum += correct
        raw_correct_sum += raw_correct
        total_sum += total
    model.train()
    return {
        "loss": loss_sum / max(eligible_sum, 1),
        "q2c_top1_accuracy": correct_sum / max(total_sum, 1),
        "raw_cosine_top1_accuracy": raw_correct_sum / max(total_sum, 1),
        "topk_recall": eligible_sum / max(total_sum, 1),
        "eligible_queries": eligible_sum,
        "total_queries": total_sum,
        "samples": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-map", action="append", default=[], metavar="FROM=TO")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    mappings = [tuple(value.split("=", 1)) for value in args.path_map]
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    validation_rows = [] if args.validation_manifest is None else [
        json.loads(line)
        for line in args.validation_manifest.read_text().splitlines() if line
    ]
    model = DeformationAwareConditioner(
        arm="b2", key_dim=2048, value_dim=1024, query_dim=1024,
        hidden_dim=256, topk=args.topk,
    ).cuda()
    optimizer = torch.optim.AdamW(model.q2c.parameters(), lr=args.lr, weight_decay=0.01)
    generator = np.random.default_rng(args.seed)
    events = args.output.with_suffix(".jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    if validation_rows:
        initial = {"epoch": 0, "validation": validate(
            model, validation_rows, mappings, args.topk
        )}
        history.append(initial)
        with events.open("a") as output:
            output.write(json.dumps(initial) + "\n")
        print(json.dumps(initial), flush=True)
    for epoch in range(args.epochs):
        for index in generator.permutation(len(rows)):
            cache = remap(rows[int(index)]["transport_cache"], mappings)
            loss, queries, correct, _, _ = loss_for_cache(model, cache, args.topk)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.q2c.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 100 == 0:
                with events.open("a") as output:
                    output.write(json.dumps({
                        "step": step, "epoch": epoch + 1,
                        "loss": float(loss.detach()), "eligible_queries": queries,
                        "top1_correct": correct,
                    }) + "\n")
                print(step, float(loss.detach()), queries, flush=True)
        metrics = validate(model, validation_rows, mappings, args.topk) if validation_rows else {
            "loss": float("nan"), "q2c_top1_accuracy": float("nan"),
            "raw_cosine_top1_accuracy": float("nan"), "topk_recall": float("nan"),
            "eligible_queries": 0, "total_queries": 0, "samples": 0,
        }
        record = {"epoch": epoch + 1, "validation": metrics}
        history.append(record)
        with events.open("a") as output:
            output.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if not validation_rows or metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.q2c.state_dict().items()
            }
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    torch.save({
        "q2c": best_state, "epochs": args.epochs, "steps": step,
        "topk": args.topk, "seed": args.seed, "best_epoch": best_epoch,
        "validation": history,
    }, temporary)
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
