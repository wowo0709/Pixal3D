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


def loss_for_cache(model, path: Path, topk: int) -> tuple[torch.Tensor, int]:
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
    query = target[:, None].expand_as(candidates)
    logits = model.q2c(torch.cat((query, candidates, cosine[..., None]), dim=-1)).squeeze(-1)
    positive_flat = positives.T + torch.arange(views, device="cuda")[None] * sites
    mask = (indices[..., None] == positive_flat[:, None]).any(dim=-1)
    eligible = mask.any(dim=-1)
    if not bool(eligible.any()):
        return logits.sum() * 0.0, 0
    selected = logits[eligible]
    selected_mask = mask[eligible]
    numerator = torch.logsumexp(selected.masked_fill(~selected_mask, -torch.inf), dim=-1)
    denominator = torch.logsumexp(selected, dim=-1)
    return (denominator - numerator).mean(), int(eligible.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
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
    model = DeformationAwareConditioner(
        arm="b2", key_dim=2048, value_dim=1024, query_dim=1024,
        hidden_dim=256, topk=args.topk,
    ).cuda()
    optimizer = torch.optim.AdamW(model.q2c.parameters(), lr=args.lr, weight_decay=0.01)
    generator = np.random.default_rng(args.seed)
    events = args.output.with_suffix(".jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    for epoch in range(args.epochs):
        for index in generator.permutation(len(rows)):
            cache = remap(rows[int(index)]["transport_cache"], mappings)
            loss, queries = loss_for_cache(model, cache, args.topk)
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
                    }) + "\n")
                print(step, float(loss.detach()), queries, flush=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    torch.save({
        "q2c": model.q2c.state_dict(), "epochs": args.epochs, "steps": step,
        "topk": args.topk, "seed": args.seed,
    }, temporary)
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
