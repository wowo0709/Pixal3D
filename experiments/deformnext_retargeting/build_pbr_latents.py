#!/usr/bin/env python3
"""Voxelize deformed target PBR dumps and encode aligned PBR1024 latents."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import o_voxel
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data_toolkit.encode_shape_latent_view import _coordinates_to_uint8
from data_toolkit.dual_grid_view import transform_mesh
from data_toolkit.utils import sphere_normalize_torch
from data_toolkit.voxelize_pbr_view import prepare_pbr_dump, transform_pbr_dump
from pixal3d import models
from pixal3d.modules import sparse as sp


def remap(path: str, mappings: list[tuple[str, str]]) -> Path:
    for source, target in mappings:
        if path == source or path.startswith(source.rstrip("/") + "/"):
            return Path(target + path[len(source):])
    return Path(path)


def atomic_latent(path: Path, feats: np.ndarray, coords: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez(temporary, feats=feats, coords=coords)
    with np.load(temporary) as check:
        if check["coords"].shape[0] != check["feats"].shape[0]:
            raise ValueError("temporary PBR latent has mismatched arrays")
    os.replace(temporary, path)


def encode_shape(shape_encoder, dump: dict, frame: dict):
    vertices_list = []
    faces_list = []
    offset = 0
    for obj in dump["objects"]:
        if obj["vertices"].size == 0 or obj["faces"].size == 0:
            continue
        vertices_list.append(obj["vertices"])
        faces_list.append(obj["faces"] + offset)
        offset += len(obj["vertices"])
    vertices = torch.from_numpy(np.concatenate(vertices_list)).float().contiguous()
    faces = torch.from_numpy(np.concatenate(faces_list)).long().contiguous()
    vertices, _, _ = sphere_normalize_torch(vertices)
    vertices = transform_mesh(vertices, frame)
    vertices = vertices * (0.49999 / vertices.abs().max().item())
    coords, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices.float().contiguous(), faces=faces,
        grid_size=1024,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2,
        regularization_weight=1e-2, timing=False,
    )
    dual_vertices = torch.clamp(dual_vertices.float() * 1024 - coords.float(), 0, 1)
    sparse_vertices = sp.SparseTensor(
        dual_vertices,
        torch.cat((torch.zeros_like(coords[:, :1]), coords), dim=-1),
    ).cuda()
    sparse_intersected = sparse_vertices.replace(intersected.bool().cuda())
    with torch.inference_mode():
        return shape_encoder(sparse_vertices, sparse_intersected)


def encode_one(shape_encoder, pbr_encoder, row: dict, dump_root: Path, mappings) -> int:
    dump_path = dump_root / "pbr_dumps" / f"{row['sample_id']}.pickle"
    with dump_path.open("rb") as stream:
        raw_dump = pickle.load(stream)
    frame = {"transform_matrix": row["transform_matrix"][0]}
    shape_output = remap(row["shape_latent"], mappings)
    if shape_output.is_file():
        with np.load(shape_output) as shape:
            shape_coords = np.asarray(shape["coords"]).copy()
    else:
        shape_latent = encode_shape(shape_encoder, raw_dump, frame)
        if not torch.isfinite(shape_latent.feats).all():
            raise ValueError("shape encoder produced non-finite features")
        shape_coords = _coordinates_to_uint8(
            shape_latent.coords[:, 1:].cpu().numpy(), 1024
        )
        atomic_latent(
            shape_output,
            shape_latent.feats.cpu().numpy().astype(np.float16),
            shape_coords,
        )
    dump = prepare_pbr_dump(raw_dump)
    transformed, _ = transform_pbr_dump(dump, frame)
    coords, attrs = o_voxel.convert.blender_dump_to_volumetric_attr(
        transformed,
        grid_size=1024,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        mip_level_offset=0,
        verbose=False,
        timing=False,
    )
    channels = ("base_color", "metallic", "roughness", "alpha")
    features = torch.cat([attrs[name] for name in channels], dim=-1) / 255.0 * 2.0 - 1.0
    voxels = sp.SparseTensor(
        features.float(),
        torch.cat((torch.zeros_like(coords[:, :1]), coords), dim=-1),
    ).cuda()
    with torch.inference_mode():
        latent = pbr_encoder(voxels)
    if not torch.isfinite(latent.feats).all():
        raise ValueError("PBR encoder produced non-finite features")
    latent_coords = _coordinates_to_uint8(latent.coords[:, 1:].cpu().numpy(), 1024)
    if not np.array_equal(latent_coords, shape_coords):
        raise ValueError(
            f"PBR/shape support mismatch: pbr={latent_coords.shape} shape={shape_coords.shape}"
        )
    output = remap(row["pbr_latent"], mappings)
    atomic_latent(
        output,
        latent.feats.cpu().numpy().astype(np.float16),
        latent_coords,
    )
    return len(latent_coords)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dump-root", type=Path, required=True)
    parser.add_argument("--shape-encoder", default="microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16")
    parser.add_argument("--pbr-encoder", default="microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16")
    parser.add_argument("--path-map", action="append", default=[], metavar="FROM=TO")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    mappings = [tuple(value.split("=", 1)) for value in args.path_map]
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    selected = [row for index, row in enumerate(rows) if index % args.world_size == args.rank]
    if args.limit is not None:
        selected = selected[:args.limit]
    shape_encoder = models.from_pretrained(args.shape_encoder).eval().cuda()
    pbr_encoder = models.from_pretrained(args.pbr_encoder).eval().cuda()
    completed = skipped = failed = 0
    failure_path = args.dump_root / f"latent-failures-rank{args.rank:03d}.jsonl"
    for row in selected:
        output = remap(row["pbr_latent"], mappings)
        if output.is_file():
            skipped += 1
            continue
        try:
            tokens = encode_one(
                shape_encoder, pbr_encoder, row, args.dump_root, mappings
            )
            completed += 1
            if completed % 10 == 0:
                print(json.dumps({
                    "rank": args.rank, "completed": completed,
                    "skipped": skipped, "failed": failed, "last_tokens": tokens,
                }), flush=True)
        except Exception as error:
            failed += 1
            with failure_path.open("a") as stream:
                stream.write(json.dumps({
                    "sample_id": row["sample_id"],
                    "error": f"{type(error).__name__}: {error}",
                }) + "\n")
    print(json.dumps({
        "rank": args.rank, "assigned": len(selected), "completed": completed,
        "skipped": skipped, "failed": failed,
    }), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
