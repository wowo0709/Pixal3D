#!/usr/bin/env python3
"""Cache Omega-L12 addresses and native Pixal values for DeformNeXt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.deformnext_retargeting.omega_loader import load_epoch7, layer12_patch_tokens
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    DinoV3ProjFeatureExtractor,
    ProjGrid,
    project_points_to_image_batch,
)


def remap(path: str, mappings: list[tuple[str, str]]) -> Path:
    for source, target in mappings:
        if path == source or path.startswith(source.rstrip("/") + "/"):
            return Path(target + path[len(source):])
    return Path(path)


def load_rgba(path: Path, size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    with Image.open(path) as source:
        rgba = np.asarray(source.convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)).copy()
    tensor = torch.from_numpy(rgba).float() / 255.0
    alpha = tensor[..., 3]
    return tensor[..., :3].permute(2, 0, 1) * alpha, alpha


def pixal_patches(extractor, images: torch.Tensor) -> torch.Tensor:
    normalized = extractor.transform(images)
    hidden = extractor.extract_features(normalized)
    registers = getattr(extractor.model.config, "num_register_tokens", 4)
    return hidden[:, 1 + registers:].float()


def pbr_pixal_patches(extractor, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 32x32 DINO/NAF values aligned with Omega's 512px patch sites."""
    normalized = extractor.transform(images)
    hidden = extractor.extract_features(normalized)
    registers = getattr(extractor.model.config, "num_register_tokens", 4)
    low = hidden[:, 1 + registers:].reshape(len(images), 64, 64, -1)
    low_bchw = low.permute(0, 3, 1, 2)
    low_32 = F.avg_pool2d(low_bchw, 2, 2)
    extractor._load_naf()
    # A four-view 1024 NAF tensor has 4 * 1024 * 1024 * 1024 elements,
    # which exceeds the CUDA pooling kernel's signed-int indexing limit.
    # Preserve the production 1024px NAF path, but pool one view at a time.
    high_32 = torch.cat([
        F.adaptive_avg_pool2d(
            extractor.naf_model(
                images[index:index + 1], low_bchw[index:index + 1], (1024, 1024)
            ),
            (32, 32),
        )
        for index in range(len(images))
    ])
    values = torch.cat((low_32, high_32), dim=1)
    values = values.permute(0, 2, 3, 1).reshape(len(images), 1024, -1)
    query = low_32.permute(0, 2, 3, 1).reshape(len(images), 1024, -1)
    return values.float(), query.float()


def patch_visibility(alpha: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    return F.avg_pool2d(alpha[:, None], patch_size, patch_size).flatten(1) > 0.05


def unproject_patch_centers(depth: np.ndarray, transform: torch.Tensor, fov: float) -> torch.Tensor:
    patch = 16
    centers = torch.arange(patch / 2, 512, patch)
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    depth_t = torch.from_numpy(depth).float()[yy.long(), xx.long()]
    focal = 512.0 / (2.0 * np.tan(float(fov) / 2.0))
    x = (xx - 256.0) * depth_t / focal
    y = -(yy - 256.0) * depth_t / focal
    camera = torch.stack((x, y, -depth_t, torch.ones_like(depth_t)), dim=-1)
    world = camera.reshape(-1, 4) @ transform.T
    return world[:, :3]


def spatial_indices(row: dict, native_arrays: Path) -> torch.Tensor:
    with np.load(native_arrays) as archive:
        depth = np.asarray(archive["depth_0"], dtype=np.float32)
    transforms = torch.tensor(row["transform_matrix"], dtype=torch.float32)
    points = unproject_patch_centers(depth, transforms[0], row["camera_angle_x"][0])
    outputs = []
    for view in range(1, 4):
        pixels, _, valid = project_points_to_image_batch(
            points[None], transforms[view][None],
            torch.tensor([row["camera_angle_x"][view]]), 512,
        )
        xy = torch.floor(pixels[0] / 16).long().clamp(0, 31)
        index = xy[:, 1] * 32 + xy[:, 0]
        index[~valid[0]] = 0
        outputs.append(index)
    return torch.stack(outputs)


def ss_patch_grid(row: dict) -> torch.Tensor:
    projector = ProjGrid(grid_resolution=16, image_resolution=512)
    points = projector.grid_points / float(row["mesh_scale"]) / 2.0
    transform = torch.tensor(row["transform_matrix"][0], dtype=torch.float32)
    pixels, _, _ = project_points_to_image_batch(
        points[None], transform[None],
        torch.tensor([row["camera_angle_x"][0]]), 512,
    )
    xy = torch.floor(pixels[0] / 16).long().clamp(0, 31)
    return (xy[:, 1] * 32 + xy[:, 0]).reshape(16, 16, 16)


def pbr_patch_ids(row: dict, latent_path: Path) -> torch.Tensor:
    with np.load(latent_path) as archive:
        coords = torch.from_numpy(np.asarray(archive["coords"]).copy()).long()
    if coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError(f"invalid sparse coordinates: {latent_path}")
    if coords.shape[1] == 4:
        if torch.any(coords[:, 0] != 0):
            raise ValueError(f"stored sparse batch index is not zero: {latent_path}")
        coords = coords[:, 1:]
    if torch.any((coords < 0) | (coords >= 64)):
        raise ValueError(f"PBR sparse coordinate outside [0,64): {latent_path}")
    one_dim = torch.linspace(-1, 1, 64)
    points = torch.stack(torch.meshgrid(one_dim, one_dim, one_dim, indexing="ij"), dim=-1)
    rotation = torch.tensor(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
    points = (points @ rotation.T)[coords[:, 0], coords[:, 1], coords[:, 2]]
    points = points / float(row["mesh_scale"]) / 2.0
    transform = torch.tensor(row["transform_matrix"][0], dtype=torch.float32)
    pixels, _, _ = project_points_to_image_batch(
        points[None], transform[None],
        torch.tensor([row["camera_angle_x"][0]]), 512,
    )
    xy = torch.floor(pixels[0] / 16).long().clamp(0, 31)
    return xy[:, 1] * 32 + xy[:, 0]


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vggt-root", type=Path, default=Path("/root/dev/vggt-omega"))
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--epoch7-checkpoint", type=Path)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--stage", choices=("ss64", "pbr1024"), required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--path-map", action="append", default=[], metavar="FROM=TO")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    mappings = [tuple(value.split("=", 1)) for value in args.path_map]
    if args.stage == "ss64":
        if args.base_checkpoint is None or args.epoch7_checkpoint is None:
            parser.error("ss64 requires --base-checkpoint and --epoch7-checkpoint")
        omega, checkpoint = load_epoch7(
            args.vggt_root, args.base_checkpoint, args.epoch7_checkpoint
        )
        dino = DinoV3ProjFeatureExtractor(
            model_name=args.dino, image_size=512, grid_resolution=16
        ).cuda().eval()
        cache_field = "transport_cache"
    else:
        omega = checkpoint = None
        dino = DinoV3ProjFeatureExtractor(
            model_name=args.dino, image_size=1024, grid_resolution=64,
            use_naf_upsample=True, naf_target_size=1024,
        ).cuda().eval()
        cache_field = "transport_cache_pbr"
    all_rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    selected = [row for index, row in enumerate(all_rows) if index % args.world_size == args.rank]
    if args.limit is not None:
        selected = selected[:args.limit]
    completed = skipped = 0
    for row in selected:
        output = remap(row[cache_field], mappings)
        if output.is_file():
            skipped += 1
            continue
        paths = [row["target_anchor"], *row["source_images"]]
        with torch.inference_mode():
            if args.stage == "ss64":
                omega_loaded = [load_rgba(remap(path, mappings), 512) for path in paths]
                omega_images = torch.stack([item[0] for item in omega_loaded]).cuda()
                alpha = torch.stack([item[1] for item in omega_loaded]).cuda()
                omega_tokens = layer12_patch_tokens(omega, omega_images[None])[0]
                dino_tokens = pixal_patches(dino, omega_images)
                source_values = dino_tokens[1:]
                target_query = dino_tokens[0]
            else:
                pbr_images = torch.stack([
                    load_rgba(remap(path, mappings), 1024)[0] for path in paths
                ]).cuda()
                dino_tokens, dino_query = pbr_pixal_patches(dino, pbr_images)
                source_values = dino_tokens[1:]
                target_query = dino_query[0]
        if args.stage == "ss64":
            valid = patch_visibility(alpha[1:]).cpu()
            native_arrays = remap(row["target_anchor_source"], mappings)
            spatial = spatial_indices(row, native_arrays)
            atomic_npz(
                output,
                schema=np.array("DEFORMNEXT_OMEGA_L12_PIXAL_VALUES_SS64_V1"),
                omega_completed_updates=np.int64(checkpoint["completed_updates"]),
                target_keys=omega_tokens[0].cpu().half().numpy(),
                source_keys=omega_tokens[1:].cpu().half().numpy(),
                source_values=source_values.cpu().half().numpy(),
                source_valid=valid.numpy(),
                target_query=target_query.cpu().half().numpy(),
                spatial_indices=spatial.numpy(),
                patch_grid=ss_patch_grid(row).numpy(),
            )
        else:
            shared = remap(row["transport_cache"], mappings)
            if not shared.is_file():
                raise FileNotFoundError(f"shared SS transport cache is missing: {shared}")
            atomic_npz(
                output,
                schema=np.array("DEFORMNEXT_PIXAL_VALUES_PBR1024_V1"),
                source_values=source_values.cpu().half().numpy(),
                target_query=target_query.cpu().half().numpy(),
                patch_ids=pbr_patch_ids(
                    row, remap(row["shape_latent"], mappings)
                ).numpy(),
            )
        completed += 1
        if completed % 10 == 0:
            print(json.dumps({"rank": args.rank, "completed": completed, "skipped": skipped}), flush=True)
    print(json.dumps({"rank": args.rank, "completed": completed, "skipped": skipped, "assigned": len(selected)}))


if __name__ == "__main__":
    main()
