#!/usr/bin/env python3
"""Resolve DeformNeXt into edited-RGB plus original-RGB training manifests."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image


SPLIT_ROOT = Path("/file4/youngwoo/deformation/exports/pixal3d-split-90-5-5-v1")
SNAPSHOT_ROOT = Path("/file4/youngwoo/deformation/exports/pixal3d-final-approved-10165-20261012")
OUTPUT_ROOT = Path("/file4/youngwoo/deformation/experiments/retargeting-v1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def atomic_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    image.save(temporary, format="PNG", compress_level=4)
    os.replace(temporary, path)


def depth_anchor(native_arrays: Path, output: Path) -> None:
    """Encode fixed-range camera depth without any target texture channels."""
    with np.load(native_arrays) as archive:
        depth = np.asarray(archive["depth_0"], dtype=np.float32)
        mask = np.asarray(archive["mask_0"], dtype=np.float32) > 0.5
    if depth.shape != (512, 512) or mask.shape != depth.shape or mask.sum() < 64:
        raise ValueError(f"invalid target depth/mask: {native_arrays}")
    value = np.clip((3.0 - depth) / 2.0, 0.0, 1.0)
    value[~mask] = 0.0
    rgb = np.repeat(np.round(value[..., None] * 255).astype(np.uint8), 3, axis=-1)
    alpha = (mask[..., None] * 255).astype(np.uint8)
    atomic_image(Image.fromarray(np.concatenate((rgb, alpha), axis=-1), "RGBA"), output)


def raster_depth(vertices: np.ndarray, faces: np.ndarray, c2w: np.ndarray, fov: float) -> np.ndarray:
    """Small deterministic CPU z-buffer used only for legacy rows lacking native arrays."""
    size = 512
    w2c = np.linalg.inv(c2w)
    xyz = vertices.astype(np.float64) @ w2c[:3, :3].T + w2c[:3, 3]
    z = -xyz[:, 2]
    focal = size / (2.0 * math.tan(fov / 2.0))
    xy = np.stack((xyz[:, 0] * focal / np.maximum(z, 1e-12) + size / 2,
                   -xyz[:, 1] * focal / np.maximum(z, 1e-12) + size / 2), axis=-1)
    output = np.full((size, size), np.inf, dtype=np.float32)
    for face in faces:
        q, depth = xy[face], z[face]
        if np.any(depth <= 1e-4):
            continue
        left = max(0, int(np.floor(q[:, 0].min())))
        right = min(size - 1, int(np.ceil(q[:, 0].max())))
        top = max(0, int(np.floor(q[:, 1].min())))
        bottom = min(size - 1, int(np.ceil(q[:, 1].max())))
        if left > right or top > bottom:
            continue
        denominator = ((q[1, 1] - q[2, 1]) * (q[0, 0] - q[2, 0])
                       + (q[2, 0] - q[1, 0]) * (q[0, 1] - q[2, 1]))
        if abs(denominator) < 1e-10:
            continue
        yy, xx = np.mgrid[top:bottom + 1, left:right + 1]
        xx, yy = xx + 0.5, yy + 0.5
        b0 = ((q[1, 1] - q[2, 1]) * (xx - q[2, 0])
              + (q[2, 0] - q[1, 0]) * (yy - q[2, 1])) / denominator
        b1 = ((q[2, 1] - q[0, 1]) * (xx - q[2, 0])
              + (q[0, 0] - q[2, 0]) * (yy - q[2, 1])) / denominator
        b2 = 1.0 - b0 - b1
        inside = (b0 >= -1e-6) & (b1 >= -1e-6) & (b2 >= -1e-6)
        perspective = np.stack((b0 / depth[0], b1 / depth[1], b2 / depth[2]), axis=-1)
        bary = perspective / np.maximum(perspective.sum(axis=-1, keepdims=True), 1e-12)
        pixel_depth = (bary * depth).sum(axis=-1)
        current = output[top:bottom + 1, left:right + 1]
        take = inside & (pixel_depth < current)
        current[take] = pixel_depth[take]
    return output


def fallback_anchor_arrays(sample: dict, frame: dict, output: Path) -> None:
    if output.is_file():
        return
    with np.load(sample["source_geometry"]["path"]) as source:
        faces = np.asarray(source["F"], dtype=np.int32)
    with np.load(sample["final_geometry"]["path"]) as target:
        vertices = np.asarray(target["V"], dtype=np.float32)
    depth = raster_depth(
        vertices, faces, np.asarray(frame["transform_matrix"], dtype=np.float64),
        float(frame["camera_angle_x"]),
    )
    mask = np.isfinite(depth)
    if mask.sum() < 64:
        raise ValueError("fallback target raster is empty")
    depth[~mask] = 0.0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.npz")
    np.savez(temporary, depth_0=depth, mask_0=mask.astype(np.float32))
    os.replace(temporary, output)


def native_dir(sample: dict) -> Path | None:
    native = sample.get("native")
    if isinstance(native, dict) and native.get("path"):
        return Path(native["path"]).parent
    for container in (sample.get("final_adjudication", {}), sample):
        roots = container.get("native_artifacts") or container.get("native")
        if isinstance(roots, dict) and roots.get("path"):
            return Path(roots["path"]).parent
    return None


def resolve(split_row: dict, sample: dict, output_root: Path, create_anchor: bool) -> dict:
    if split_row["sample_id"] != sample["sample_id"]:
        raise ValueError("split/sample identity mismatch")
    storage = Path(split_row["storage_root"])
    sample_id = split_row["sample_id"]
    render = storage / "train/render_cond" / sample_id
    transforms_path = render / "transforms.json"
    transforms = json.loads(transforms_path.read_text())
    frames = transforms["frames"]
    if len(frames) != 8 or frames[0].get("mesh_state") != "M1":
        raise ValueError(f"unexpected render role manifest: {transforms_path}")
    if any(frames[index].get("mesh_state") != "M0" for index in (1, 2, 3, 4)):
        raise ValueError(f"source roles are not M0: {transforms_path}")
    native = native_dir(sample)
    if native is None:
        native_arrays = output_root / "anchor_arrays" / f"{sample_id}.npz"
        if create_anchor:
            fallback_anchor_arrays(sample, frames[0], native_arrays)
        anchor_source_kind = "canonical_cpu_raster"
        target_vertices = Path(sample["final_geometry"]["path"])
    else:
        native_arrays = native / "M1_native_render_arrays.npz"
        anchor_source_kind = "native_render_arrays"
        target_vertices = native.parent / "rgb" / "final_vertices.npy"
    anchor = output_root / "anchors" / f"{sample_id}.png"
    if create_anchor and not anchor.exists():
        depth_anchor(native_arrays, anchor)
    ss_dir = storage / "train/ss_latent" / sample_id
    shape_dir = storage / "train/shape_latent" / sample_id
    scale = json.loads((ss_dir / "view00_scale.json").read_text())["total_scale"]
    camera_frames = [frames[index] for index in (0, 2, 3, 4)]
    rgb_result = None
    candidate_native = sample.get("native", {}).get("path") if isinstance(sample.get("native"), dict) else None
    if candidate_native:
        candidate_root = Path(candidate_native).parent.parent
        candidate_rgb = candidate_root / "rgb/result.json"
        if candidate_rgb.is_file():
            rgb_result = json.loads(candidate_rgb.read_text())
    evaluation = []
    if rgb_result:
        evaluation = [entry["M1"]["path"] for entry in rgb_result.get("images", [])]
    result = {
        "sample_id": sample_id,
        "pair_id": split_row["pair_id"],
        "split": split_row["logical_split"],
        "source_uid": split_row["source_uid"],
        "source_cluster_id": split_row["source_cluster_id"],
        "category": split_row["category"],
        "target_anchor": str(render / frames[0]["file_path"]),
        "target_anchor_kind": "rgb",
        "target_rgb_input": str(render / frames[0]["file_path"]),
        "target_depth_auxiliary": str(anchor),
        "target_anchor_source": str(native_arrays),
        "target_anchor_source_kind": anchor_source_kind,
        "source_images": [str(render / frames[index]["file_path"]) for index in (2, 3, 4)],
        "source_anchor_rgb": str(render / frames[1]["file_path"]),
        "source_anchor_transform_matrix": frames[1]["transform_matrix"],
        "source_anchor_camera_angle_x": frames[1]["camera_angle_x"],
        "conditioning_camera_ids": [frame["camera_id"] for frame in camera_frames],
        "target_rgb_supervision": str(render / frames[0]["file_path"]),
        "evaluation_target_images": evaluation,
        "camera_angle_x": [float(frame["camera_angle_x"]) for frame in camera_frames],
        "transform_matrix": [frame["transform_matrix"] for frame in camera_frames],
        "mesh_scale": float(scale),
        "ss_latent": str(ss_dir / "view00.npz"),
        "shape512_latent": str(shape_dir / "view00.npz"),
        "shape_latent": str(output_root / "shape1024_latents" / f"{sample_id}.npz"),
        "pbr_latent": str(output_root / "pbr_latents" / f"{sample_id}.npz"),
        "transport_cache": str(output_root / "transport" / f"{sample_id}.npz"),
        "transport_cache_pbr": str(output_root / "transport_pbr" / f"{sample_id}.npz"),
        "source_geometry": sample["source_geometry"]["path"],
        "target_geometry": sample["final_geometry"]["path"],
        "target_vertices": str(target_vertices),
        "topology_correspondence": True,
        "frame_extent": float(sample["frame_extent"]),
    }
    from .prepare_rgb_manifest import SCHEMA, image_fingerprint
    result['conditioning_input_schema']=SCHEMA
    result['conditioning_input_sha256']=image_fingerprint(result,Path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=SPLIT_ROOT)
    parser.add_argument("--snapshot-root", type=Path, default=SNAPSHOT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--no-create-anchors", action="store_true")
    parser.add_argument("--allow-unresolved", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    samples = {row["sample_id"]: row for row in rows(args.snapshot_root / "samples.jsonl")}
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = {"status": "PASS", "counts": {}, "unresolved": []}
    for split in args.splits:
        output = args.output_root / "manifests" / f"{split}.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        resolved = []
        split_rows = list(rows(args.split_root / f"{split}.jsonl"))

        def work(split_row):
            try:
                return resolve(
                    split_row, samples[split_row["sample_id"]], args.output_root,
                    not args.no_create_anchors,
                ), None
            except Exception as error:
                return None, {
                    "split": split, "sample_id": split_row["sample_id"],
                    "error": f"{type(error).__name__}: {error}",
                }

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for row, error in executor.map(work, split_rows):
                if row is not None:
                    resolved.append(row)
                else:
                    report["unresolved"].append(error)
        temporary = output.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in resolved))
        os.replace(temporary, output)
        report["counts"][split] = len(resolved)
        report.setdefault("manifest_sha256", {})[split] = sha256(output)
    if report["unresolved"]:
        report["status"] = "INCOMPLETE"
    (args.output_root / "manifest-build-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["unresolved"] and not args.allow_unresolved:
        raise SystemExit(f"unresolved samples: {len(report['unresolved'])}")


if __name__ == "__main__":
    main()
