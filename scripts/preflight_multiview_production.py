from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Sequence
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch


SOURCE = "ABO"
RENDER_ROOT = "renders_cond"
COMPONENTS = {
    "ss64": (("ss_latents/ss_enc_conv3d_16l8_fp16_64_view", "ss", (
        "ss_latent_view_scale00_encoded", "ss_latent_view_scale01_encoded",
    )),),
    "shape512": (("shape_latents/shape_enc_next_dc_f16c32_fp16_512_view", "shape", (
        "shape_latent_view00_encoded", "shape_latent_view01_encoded",
    )),),
    "shape1024": (("shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view", "shape", (
        "shape_latent_view00_encoded", "shape_latent_view01_encoded",
    )),),
    "pbr1024": (
        ("shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view", "shape", (
            "shape_latent_view00_encoded", "shape_latent_view01_encoded",
        )),
        ("pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix", "pbr", (
            "pbr_latent_view00_encoded", "pbr_latent_view01_encoded",
        )),
    ),
}
CONFIGS = {
    "ss64": Path("configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json"),
    "shape512": Path("configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json"),
    "shape1024": Path("configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json"),
    "pbr1024": Path("configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json"),
}


@dataclass(frozen=True)
class StagePreflight:
    stage: str
    root: Path
    asset_count: int
    asset_scope_sha256: str
    anchors_checked: int
    validation_counts: dict[str, int]


def _error(stage: str, asset: str | None, message: str, anchor: str | None = None) -> ValueError:
    parts = [f"source={SOURCE}", f"stage={stage}"]
    if asset is not None:
        parts.append(f"asset={asset}")
    if anchor is not None:
        parts.append(f"anchor={anchor}")
    return ValueError(" ".join(parts) + f": {message}")


def _regular(path: Path, stage: str, asset: str, label: str, anchor: str | None = None) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise _error(stage, asset, f"missing or unsafe {label}: {path}", anchor)


def _metadata_assets(stage: str, root: Path, relative: str, fields: tuple[str, ...], expected: set[str]) -> None:
    component_root = root / relative
    if component_root.is_symlink() or not component_root.is_dir():
        raise _error(stage, None, f"missing or unsafe component root: {relative}")
    path = component_root / "metadata.csv"
    _regular(path, stage, None, f"metadata for {relative}")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = ["sha256", *fields]
        rows = list(reader)
        if reader.fieldnames != required:
            asset = rows[0].get("sha256") if rows else None
            raise _error(stage, asset, f"metadata columns for {relative} must be {required}")
    actual = set()
    for row in rows:
        asset = row["sha256"]
        if not asset or asset in actual:
            raise _error(stage, asset or None, f"invalid metadata asset in {relative}")
        actual.add(asset)
        for field in fields:
            if row[field] != "True":
                raise _error(stage, asset, f"metadata {field} must be literal True")
    if actual != expected:
        missing, extra = expected - actual, actual - expected
        asset = sorted(missing or extra)[0] if missing or extra else None
        raise _error(stage, asset, f"metadata scope mismatch for {relative}")
    entries = {entry.name for entry in component_root.iterdir()}
    allowed = {"metadata.csv", *expected}
    if entries != allowed:
        extra = entries - allowed
        asset = sorted(extra)[0] if extra else sorted(allowed - entries)[0]
        raise _error(stage, asset, f"component directory scope mismatch for {relative}")
    for asset in expected:
        asset_root = component_root / asset
        if asset_root.is_symlink() or not asset_root.is_dir():
            raise _error(stage, asset, f"component asset directory is unsafe for {relative}")


def _validate_render(stage: str, root: Path, asset: str) -> int:
    directory = root / RENDER_ROOT / asset
    if directory.is_symlink() or not directory.is_dir():
        raise _error(stage, asset, "missing render directory")
    names = {entry.name for entry in directory.iterdir()}
    required = {"transforms.json", *(f"{index:03d}.png" for index in range(8))}
    if names != required:
        raise _error(stage, asset, "render directory must contain exactly eight PNGs and transforms.json")
    manifest_path = directory / "transforms.json"
    _regular(manifest_path, stage, asset, "transforms.json")
    try:
        manifest = json.loads(manifest_path.read_text())
        frames = manifest["frames"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise _error(stage, asset, "invalid transforms.json") from error
    if not isinstance(frames, list) or len(frames) != 8:
        raise _error(stage, asset, "transforms must contain exactly eight frames")
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise _error(stage, asset, "frame must be an object")
        file_path = frame.get("file_path")
        expected_name = f"{index:03d}.png"
        if not isinstance(file_path, str) or file_path != expected_name:
            raise _error(stage, asset, f"unsafe or unordered frame path: {file_path!r}")
        entry_path = directory / file_path
        _regular(entry_path, stage, asset, "render PNG")
        image_path = entry_path.resolve()
        if not image_path.is_relative_to(directory.resolve()):
            raise _error(stage, asset, f"unsafe frame path: {file_path!r}")
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                if image.mode != "RGBA" or image.size != (512, 512):
                    raise _error(stage, asset, "render PNG must be RGBA 512x512")
        except ValueError:
            raise
        except Exception as error:
            raise _error(stage, asset, f"invalid render PNG: {image_path}") from error
        try:
            angle = np.asarray(frame["camera_angle_x"], dtype=np.float32).item()
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise _error(stage, asset, "camera_angle_x must be finite and positive") from error
        if not np.isfinite(angle) or angle <= 0:
            raise _error(stage, asset, "camera_angle_x must be finite and positive")
        try:
            transform = np.asarray(frame["transform_matrix"], dtype=np.float32)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise _error(stage, asset, "invalid transform_matrix") from error
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise _error(stage, asset, "transform_matrix must be finite [4, 4]")
        rotation = transform[:3, :3]
        if not np.isfinite(np.linalg.det(rotation)) or np.linalg.det(rotation) == 0:
            raise _error(stage, asset, "transform rotation must be nonsingular")
        distance = float(np.linalg.norm(transform[:3, 3]))
        if not np.isfinite(distance) or distance <= 0:
            raise _error(stage, asset, "camera distance must be finite and positive")
    return 8


def _scale(stage: str, asset: str, path: Path, anchor: str) -> np.float32:
    _regular(path, stage, asset, "scale JSON", anchor)
    try:
        value = json.loads(path.read_text())["total_scale"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("total_scale must be a JSON numeric scalar")
        scale = np.float32(value)
    except (KeyError, TypeError, ValueError, OverflowError, json.JSONDecodeError) as error:
        raise _error(stage, asset, "total_scale must be finite and positive after float32 conversion", anchor) from error
    if not np.isfinite(scale) or scale <= 0:
        raise _error(stage, asset, "total_scale must be finite and positive after float32 conversion", anchor)
    return scale


def _numeric_finite(value: np.ndarray) -> bool:
    return value.dtype.kind in "fiu" and np.isfinite(np.asarray(value, dtype=np.float32)).all()


def _latent(stage: str, asset: str, component: str, directory: Path, anchor_index: int) -> tuple[np.ndarray | None, np.float32]:
    anchor = f"view{anchor_index:02d}"
    npz_path = directory / asset / f"{anchor}.npz"
    _regular(npz_path, stage, asset, "latent NPZ", anchor)
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            keys = set(data.files)
            if component == "ss":
                if keys != {"z"}:
                    raise _error(stage, asset, "SS latent requires exactly z", anchor)
                z = np.asarray(data["z"])
                if z.shape != (8, 16, 16, 16) or z.dtype.kind not in "fiu" or not _numeric_finite(z):
                    raise _error(stage, asset, "SS z must be finite float-compatible [8, 16, 16, 16]", anchor)
                coords = None
            else:
                if keys != {"coords", "feats"}:
                    raise _error(stage, asset, "sparse latent requires exactly coords and feats", anchor)
                coords = np.asarray(data["coords"])
                feats = np.asarray(data["feats"])
                if coords.ndim != 2 or coords.shape[1:] != (3,):
                    raise _error(stage, asset, "coords must have shape [N, 3]", anchor)
                if feats.ndim != 2 or feats.shape[1:] != (32,):
                    raise _error(stage, asset, "feats must have shape [N, 32]", anchor)
                if len(coords) != len(feats):
                    raise _error(stage, asset, "coords and feats row counts must match", anchor)
                if not _numeric_finite(coords) or not _numeric_finite(feats):
                    raise _error(stage, asset, "coords and feats must be finite numeric arrays", anchor)
                integral = np.equal(coords, np.floor(coords))
                if not integral.all():
                    raise _error(stage, asset, "coords must be integral", anchor)
                grid = 32 if stage == "shape512" else 64
                integer_coords = coords.astype(np.int64)
                if (integer_coords < 0).any() or (integer_coords >= grid).any():
                    raise _error(stage, asset, "coords exceed stage grid bounds", anchor)
                if len(np.unique(integer_coords, axis=0)) != len(integer_coords):
                    raise _error(stage, asset, "coords must be unique", anchor)
                maximum = 8192 if stage == "shape512" else 32768
                if len(integer_coords) > maximum:
                    raise _error(stage, asset, "sparse token count exceeds stage maximum", anchor)
                coords = integer_coords
    except ValueError:
        raise
    except Exception as error:
        raise _error(stage, asset, "invalid latent NPZ", anchor) from error
    scale = _scale(stage, asset, directory / asset / f"{anchor}_scale.json", anchor)
    return coords, scale


def validate_stage_structure(stage: str, root: Path, expected_assets: Sequence[str]) -> dict[str, int]:
    """Validate all published files for one stage without invoking a dataset loader."""
    if stage not in COMPONENTS:
        raise ValueError(f"unknown stage: {stage}")
    root = Path(root)
    expected = set(expected_assets)
    if len(expected) != len(expected_assets) or not expected:
        raise _error(stage, None, "expected asset scope must be non-empty and unique")
    _metadata_assets(stage, root, RENDER_ROOT, ("cond_rendered",), expected)
    for relative, _component, fields in COMPONENTS[stage]:
        _metadata_assets(stage, root, relative, fields, expected)
    renders = latents = scales = 0
    for asset in expected_assets:
        renders += _validate_render(stage, root, asset)
        values: dict[str, tuple[np.ndarray | None, np.float32]] = {}
        for relative, component, _fields in COMPONENTS[stage]:
            for anchor in (0, 1):
                values[f"{component}:{anchor}"] = _latent(stage, asset, component, root / relative, anchor)
                latents += 1
                scales += 1
        if stage == "pbr1024":
            for anchor in (0, 1):
                shape_coords, shape_scale = values[f"shape:{anchor}"]
                pbr_coords, pbr_scale = values[f"pbr:{anchor}"]
                anchor_name = f"view{anchor:02d}"
                if not np.array_equal(shape_coords, pbr_coords):
                    raise _error(stage, asset, "PBR and Shape coordinates must match", anchor_name)
                if shape_scale != pbr_scale:
                    raise _error(stage, asset, "PBR and Shape float32 total_scale must match", anchor_name)
    return {"assets": len(expected_assets), "renders": renders, "latents": latents, "scales": scales}


def stage_data_dir(stage: str, root: Path) -> dict[str, dict[str, str]]:
    values = {"base": str(root), "render_cond": str(root / "renders_cond")}
    if stage == "ss64":
        values["ss_latent"] = str(root / "ss_latents/ss_enc_conv3d_16l8_fp16_64_view")
    elif stage == "shape512":
        values["shape_latent"] = str(root / "shape_latents/shape_enc_next_dc_f16c32_fp16_512_view")
    elif stage == "shape1024":
        values["shape_latent"] = str(root / "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view")
    elif stage == "pbr1024":
        values["shape_latent"] = str(root / "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view")
        values["pbr_latent"] = str(root / "pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix")
    else:
        raise ValueError(f"unknown stage: {stage}")
    return {SOURCE: values}


def _require_tensor(stage: str, asset: str, anchor: int, pack: dict, key: str, shape: tuple[int, ...], *, positive: bool = False) -> torch.Tensor:
    value = pack.get(key)
    anchor_name = f"view{anchor:02d}"
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or tuple(value.shape) != shape or not torch.isfinite(value).all():
        raise _error(stage, asset, f"loader {key} must be finite float32 with shape {shape}", anchor_name)
    if positive and not (value > 0).all():
        raise _error(stage, asset, f"loader {key} must be positive", anchor_name)
    return value


def _validate_loader_pack(stage: str, asset: str, anchor: int, pack: dict, image_size: int) -> None:
    anchor_name = f"view{anchor:02d}"
    if pack.get("view_idx") != anchor:
        raise _error(stage, asset, "loader selected the wrong anchor", anchor_name)
    view_indices = pack.get("view_indices")
    if not isinstance(view_indices, torch.Tensor) or view_indices.dtype != torch.int64 or tuple(view_indices.shape) != (8,) or view_indices[0].item() != anchor or sorted(view_indices.tolist()) != list(range(8)):
        raise _error(stage, asset, "loader view_indices must be anchor-first permutation", anchor_name)
    _require_tensor(stage, asset, anchor, pack, "cond", (8, 3, image_size, image_size))
    _require_tensor(stage, asset, anchor, pack, "camera_angle_x", (8,), positive=True)
    _require_tensor(stage, asset, anchor, pack, "camera_distance", (8,), positive=True)
    _require_tensor(stage, asset, anchor, pack, "transform_matrix", (8, 4, 4))
    mesh_scale = pack.get("mesh_scale")
    if not isinstance(mesh_scale, torch.Tensor) or mesh_scale.dtype != torch.float32 or mesh_scale.ndim != 0 or not torch.isfinite(mesh_scale) or mesh_scale <= 0:
        raise _error(stage, asset, "loader mesh_scale must be a finite positive float32 scalar", anchor_name)
    if stage == "ss64":
        _require_tensor(stage, asset, anchor, pack, "x_0", (8, 16, 16, 16))
        return
    coords = pack.get("coords")
    if not isinstance(coords, torch.Tensor) or coords.dtype not in (torch.int32, torch.int64) or coords.ndim != 2 or coords.shape[1:] != (3,):
        raise _error(stage, asset, "loader coords must be integer [N, 3]", anchor_name)
    if stage in ("shape512", "shape1024"):
        _require_tensor(stage, asset, anchor, pack, "feats", (coords.shape[0], 32))
    else:
        _require_tensor(stage, asset, anchor, pack, "pbr_feats", (coords.shape[0], 32))
        _require_tensor(stage, asset, anchor, pack, "shape_feats", (coords.shape[0], 32))


def validate_direct_loader(stage: str, root: Path, expected_assets: Sequence[str], config_path: Path) -> int:
    """Call the configured real dataset directly for both anchors of every asset."""
    if stage not in CONFIGS:
        raise ValueError(f"unknown stage: {stage}")
    try:
        config = json.loads(Path(config_path).read_text())
        dataset_config = config["dataset"]
        dataset_name = dataset_config["name"]
        dataset_args = dataset_config["args"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise _error(stage, None, f"invalid dataset config: {config_path}") from error
    try:
        # flex_gemm selects import-time Triton kernels by eagerly querying CUDA.
        # Its A100 table is sufficient to import dataset definitions; this preflight
        # never invokes those kernels or initializes CUDA.
        with patch.object(torch.cuda, "get_device_name", return_value="A100"):
            from pixal3d import datasets
            dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(json.dumps(stage_data_dir(stage, Path(root))), **dataset_args)
    except Exception as error:
        raise _error(stage, None, f"failed to construct configured dataset {dataset_name}") from error
    expected = set(expected_assets)
    instances = list(dataset.instances)
    actual = {asset for _root_record, asset, source in instances if source == SOURCE}
    if len(instances) != len(expected_assets) or actual != expected or any(source != SOURCE for _root_record, _asset, source in instances):
        raise _error(stage, None, "configured dataset instance set does not exactly match materialized scope")
    by_asset = {asset: root_record for root_record, asset, _source in instances}
    image_size = int(dataset_args["image_size"])
    checked = 0
    for asset in expected_assets:
        root_record = by_asset[asset]
        for anchor in (0, 1):
            dataset._current_dataset_name = SOURCE
            try:
                with patch.object(np.random, "randint", return_value=anchor):
                    pack = dataset.get_instance(root_record, asset)
            except Exception as error:
                raise RuntimeError(f"{error} stage={stage}") from error
            _validate_loader_pack(stage, asset, anchor, pack, image_size)
            checked += 1
    return checked


def _materialization_scope(stage: str, root: Path) -> tuple[tuple[str, ...], str]:
    path = Path(root) / "materialization.json"
    _regular(path, stage, None, "materialization.json")
    try:
        evidence = json.loads(path.read_text())
        assets = evidence["stage_scope"]
        digest = evidence["stage_scope_sha256"]
        count = evidence["asset_count"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise _error(stage, None, "invalid materialization evidence") from error
    if "stage_root" not in evidence:
        raise _error(stage, None, "materialization stage root identity is missing")
    stage_root = evidence["stage_root"]
    if evidence.get("stage") != stage or not isinstance(assets, list) or not all(isinstance(asset, str) for asset in assets):
        raise _error(stage, None, "materialization stage identity is invalid")
    canonical_root = str(Path(root).resolve())
    if not isinstance(stage_root, str) or stage_root != canonical_root:
        raise _error(stage, None, "materialization stage root identity mismatch")
    if assets != sorted(assets) or len(set(assets)) != len(assets) or count != len(assets):
        raise _error(stage, None, "materialization asset scope is not canonical")
    computed = sha256("\n".join(assets).encode()).hexdigest()
    if digest != computed:
        raise _error(stage, None, "materialization asset scope digest mismatch")
    return tuple(assets), digest


def preflight_stage(stage: str, root: Path, config_path: Path) -> StagePreflight:
    assets, digest = _materialization_scope(stage, root)
    counts = validate_stage_structure(stage, root, assets)
    anchors = validate_direct_loader(stage, root, assets, config_path)
    return StagePreflight(stage, Path(root), len(assets), digest, anchors, counts)
