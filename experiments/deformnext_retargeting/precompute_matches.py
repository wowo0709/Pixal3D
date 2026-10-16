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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
from pixal3d.modules.r9_matching import MATCHER_CONTRACT, ATTENTION_CONTRACT, frame_raw, frame_topk, categorical_weights


SCHEMA = "deformnext-match-r9-v5"


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


def valid_cache(path: Path, checkpoint_hash: str, topk: int, input_hash=None) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path) as cache:
            return (
                str(cache["schema"]) == SCHEMA
                and (input_hash is None or ('conditioning_input_sha256' in cache and str(cache['conditioning_input_sha256'])==input_hash))
                and json.loads(str(cache['matcher_contract'])) == MATCHER_CONTRACT
                and str(cache['aggregation']) == 'global-categorical-v1'
                and json.loads(str(cache['attention_contract'])) == ATTENTION_CONTRACT
                and str(cache['proposal_policy']) == 'GLOBAL64_VALID_OR_NATIVE'
                and str(cache["q2c_sha256"]) == checkpoint_hash
                and int(cache["topk"]) == topk
                and cache["semantic_indices"].shape == (1024, topk)
                and cache["matching_logits"].shape == (1024, topk)
                and cache["candidate_valid"].shape == (1024, topk)
                and cache["deformation"].shape == (1024, 2048)
                and np.isfinite(cache['deformation']).all()
                and np.array_equal(np.isfinite(cache['matching_logits']), cache['candidate_valid'])
                and not np.isnan(cache['matching_logits']).any()
                and (cache['semantic_indices'] < 3072).all()
                and (cache['semantic_indices'] >= 0).all()
                and (np.diff(np.sort(cache['semantic_indices'].astype(np.int32), axis=-1), axis=-1) > 0).all()
            )
    except Exception:
        return False


@torch.inference_mode()
def compute(model, transport: Path, topk: int) -> dict[str, np.ndarray]:
    with np.load(transport) as cache:
        target_context = torch.from_numpy(np.asarray(cache["target_keys"])).pin_memory().cuda(non_blocking=True).float()
        source_context = torch.from_numpy(np.asarray(cache["source_keys"])).pin_memory().cuda(non_blocking=True).float()
        target, source = frame_raw(target_context), frame_raw(source_context)
        source_valid = torch.from_numpy(np.asarray(cache["source_valid"])).cuda().bool()
    views, sites, dim = source.shape
    flat = source.reshape(views * sites, dim)
    cosine, indices = frame_topk(target[None], source[None], source_valid[None], topk)
    cosine, indices = cosine[0], indices[0]
    valid = torch.isfinite(cosine)
    candidates = flat[indices]
    logits = model._q2c_logits(
        target[None], candidates[None], cosine.nan_to_num(nan=-1.0,posinf=-1.0,neginf=-1.0)[None]
    ).squeeze(0).masked_fill(~valid, -torch.inf)

    combined=categorical_weights(logits,valid)
    source_summary = (source_context.flatten(0, 1)[indices] * combined[..., None]).sum(dim=1)
    deformation = target_context - source_summary
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
        arm="b2", key_dim=1024, value_dim=1024, query_dim=1024,
        hidden_dim=256, topk=args.topk,
        q2c_checkpoint=str(args.q2c_checkpoint), freeze_q2c=True,
    ).eval().cuda()
    args.output_root.mkdir(parents=True, exist_ok=True)
    failure_path = args.output_root.parent / f"match-failures-rank{args.rank:03d}.jsonl"
    completed = skipped = failed = 0
    for row in rows:
        output = args.output_root / f"{row['sample_id']}.npz"
        if valid_cache(output, checkpoint_hash, args.topk, row.get('conditioning_input_sha256')):
            skipped += 1
            continue
        try:
            if row.get('conditioning_input_sha256') is not None:
                with np.load(remap(row['transport_cache'], mappings)) as transport:
                    if str(transport['conditioning_input_sha256']) != row['conditioning_input_sha256']:
                        raise ValueError('matching transport RGB input hash differs')
            arrays = compute(model, remap(row["transport_cache"], mappings), args.topk)
            temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.npz")
            np.savez(
                temporary, schema=np.asarray(SCHEMA),
                matcher_contract=np.asarray(json.dumps(MATCHER_CONTRACT, sort_keys=True)),
                aggregation=np.asarray('global-categorical-v1'),
                attention_contract=np.asarray(json.dumps(ATTENTION_CONTRACT,sort_keys=True)),
                proposal_policy=np.asarray('GLOBAL64_VALID_OR_NATIVE'),
                q2c_sha256=np.asarray(checkpoint_hash), topk=np.asarray(args.topk),
                conditioning_input_sha256=np.asarray(row.get('conditioning_input_sha256','')),
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
