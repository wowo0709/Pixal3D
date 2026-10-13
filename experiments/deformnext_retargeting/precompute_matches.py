#!/usr/bin/env python3
"""Cache frozen semantic addresses, Q2C scores, and C2 deformation context."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pixal3d.modules.deform_conditioner import DeformationAwareConditioner


SCHEMA = "deformnext-match-v1"


def remap(path: str, mappings: list[tuple[str, str]]) -> Path:
    for source, target in mappings:
        if path == source or path.startswith(source.rstrip("/") + "/"):
            return Path(target + path[len(source):])
    return Path(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_cache(path: Path, checkpoint_hash: str, topk: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path) as cache:
            return (
                str(cache["schema"]) == SCHEMA
                and str(cache["q2c_sha256"]) == checkpoint_hash
                and int(cache["topk"]) == topk
                and cache["semantic_indices"].shape == (1024, topk)
                and cache["matching_logits"].shape == (1024, topk)
                and cache["candidate_valid"].shape == (1024, topk)
                and cache["deformation"].shape == (1024, 2048)
            )
    except Exception:
        return False


@torch.inference_mode()
def compute(model, transport: Path, topk: int) -> dict[str, np.ndarray]:
    with np.load(transport) as cache:
        target = torch.from_numpy(np.asarray(cache["target_keys"])).cuda().float()
        source = torch.from_numpy(np.asarray(cache["source_keys"])).cuda().float()
        source_valid = torch.from_numpy(np.asarray(cache["source_valid"])).cuda().bool()
    views, sites, dim = source.shape
    flat = source.reshape(views * sites, dim)
    valid_flat = source_valid.reshape(-1)
    similarity = F.normalize(target, dim=-1) @ F.normalize(flat, dim=-1).T
    similarity = similarity.masked_fill(~valid_flat[None], -torch.inf)
    cosine, indices = similarity.topk(topk, dim=-1)
    valid = torch.isfinite(cosine)
    candidates = flat[indices]
    logits = model._q2c_logits(
        target[None], candidates[None], cosine.nan_to_num(-1.0)[None]
    ).squeeze(0).masked_fill(~valid, -torch.inf)

    stable = logits - logits.amax(dim=-1, keepdim=True).clamp(min=-1e4)
    exponent = torch.exp(stable).masked_fill(~valid, 0.0)
    view_index = torch.div(indices, sites, rounding_mode="floor")
    by_view_den = logits.new_zeros(len(target), views)
    by_view_den.scatter_add_(1, view_index, exponent)
    confidence = logits.new_zeros(len(target), views)
    confidence.scatter_reduce_(
        1, view_index, torch.sigmoid(logits).masked_fill(~valid, 0.0),
        reduce="amax", include_self=True,
    )
    confidence = confidence.masked_fill(by_view_den == 0, 0.0)
    view_weights = confidence / confidence.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    candidate_weights = exponent / by_view_den.gather(1, view_index).clamp_min(1e-8)
    combined = candidate_weights * view_weights.gather(1, view_index)
    source_context = (candidates * combined[..., None]).sum(dim=1)
    deformation = target - source_context
    return {
        "semantic_indices": indices.cpu().numpy().astype(np.uint16),
        "matching_logits": logits.cpu().numpy().astype(np.float32),
        "candidate_valid": valid.cpu().numpy(),
        "deformation": deformation.cpu().numpy().astype(np.float16),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--q2c-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--path-map", action="append", default=[], metavar="FROM=TO")
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()
    mappings = [tuple(value.split("=", 1)) for value in args.path_map]
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    rows = [row for index, row in enumerate(rows) if index % args.world_size == args.rank]
    checkpoint_hash = sha256(args.q2c_checkpoint)
    model = DeformationAwareConditioner(
        arm="b2", key_dim=2048, value_dim=1024, query_dim=1024,
        hidden_dim=256, topk=args.topk,
        q2c_checkpoint=str(args.q2c_checkpoint), freeze_q2c=True,
    ).eval().cuda()
    args.output_root.mkdir(parents=True, exist_ok=True)
    failure_path = args.output_root.parent / f"match-failures-rank{args.rank:03d}.jsonl"
    completed = skipped = failed = 0
    for row in rows:
        output = args.output_root / f"{row['sample_id']}.npz"
        if valid_cache(output, checkpoint_hash, args.topk):
            skipped += 1
            continue
        try:
            arrays = compute(model, remap(row["transport_cache"], mappings), args.topk)
            temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.npz")
            np.savez(
                temporary, schema=np.asarray(SCHEMA),
                q2c_sha256=np.asarray(checkpoint_hash), topk=np.asarray(args.topk),
                **arrays,
            )
            if not valid_cache(temporary, checkpoint_hash, args.topk):
                raise ValueError("temporary match cache failed schema validation")
            os.replace(temporary, output)
            completed += 1
        except Exception as error:
            failed += 1
            with failure_path.open("a") as stream:
                stream.write(json.dumps({
                    "sample_id": row["sample_id"],
                    "error": f"{type(error).__name__}: {error}",
                }) + "\n")
        if (completed + skipped + failed) % 25 == 0:
            print(json.dumps({
                "rank": args.rank, "completed": completed,
                "skipped": skipped, "failed": failed,
            }), flush=True)
    print(json.dumps({
        "rank": args.rank, "assigned": len(rows), "completed": completed,
        "skipped": skipped, "failed": failed,
    }), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
