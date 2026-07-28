"""Source-aware strict validation and create-only training handoff publication."""

from __future__ import annotations

import csv
import json
import os
import stat
import tempfile
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from data_toolkit.pipeline.training_eligibility import (  # noqa: E402
    EXPECTED_CANDIDATE_STAGE_COUNTS,
    EXPECTED_FINAL_STAGE_COUNTS,
    EXPECTED_FROZEN_COUNT,
    EXPECTED_GLOBAL_QUARANTINE_COUNT,
    EXPECTED_SHAPE512_FAMILY_EXCLUSION_COUNT,
    EXPECTED_TRAINING_EXCLUSION_COUNTS,
    SCALE_ATOL,
    SCALE_RTOL,
    TOKEN_LIMITS,
    canonical_count_contract,
    observed_count_contract,
    policy_evidence,
)
from data_toolkit.pipeline.training_materialization import (
    ProductionSourceSpec,
)


SOURCE = "ABO"
_VALIDATION_SOURCE: ContextVar[str] = ContextVar(
    "training_preflight_source", default=SOURCE
)
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
DEFAULT_ROOT = Path("/root/node17/data/pixal3d/train/production/abo")
DEFAULT_INDEX = Path("/root/data2/pixal3d/prepared/index/ABO/ABO-00000.json")
DEFAULT_REPORT = Path(
    "/root/data2/pixal3d/control/reports/gates/ABO/"
    "ABO-00000-valid-subset.json"
)
DEFAULT_HANDOFF = Path(
    "/root/data2/pixal3d/control/splits/ABO/"
    "ABO-00000-valid-subset-handoff.json"
)
DEFAULT_TRAINING_DATA = DEFAULT_ROOT / "training_data.json"
HANDOFF_CANDIDATE_STAGE_COUNTS = EXPECTED_CANDIDATE_STAGE_COUNTS
HANDOFF_TRAINING_EXCLUSION_COUNTS = EXPECTED_TRAINING_EXCLUSION_COUNTS
HANDOFF_STAGE_COUNTS = EXPECTED_FINAL_STAGE_COUNTS
HANDOFF_FROZEN_COUNT = EXPECTED_FROZEN_COUNT
HANDOFF_GLOBAL_QUARANTINE_COUNT = EXPECTED_GLOBAL_QUARANTINE_COUNT
HANDOFF_SHAPE512_FAMILY_EXCLUSION_COUNT = EXPECTED_SHAPE512_FAMILY_EXCLUSION_COUNT


def _validate_source_name(source: object) -> str:
    if not isinstance(source, str) or not source or source.strip() != source:
        raise ValueError("source must be a non-empty exact name")
    return source


def _source_context_entrypoint(function):
    """Add one exact source prefix to errors crossing a generic boundary."""

    @wraps(function)
    def wrapped(spec: ProductionSourceSpec, *args, **kwargs):
        source = getattr(spec, "source", "<unknown>")
        token = None
        if isinstance(source, str) and source:
            token = _VALIDATION_SOURCE.set(source)
        try:
            return function(spec, *args, **kwargs)
        except (FileExistsError, RuntimeError, TypeError, ValueError) as error:
            context = f"source={source}"
            if context in str(error):
                raise
            raise type(error)(f"{context}: {error}") from error
        finally:
            if token is not None:
                _VALIDATION_SOURCE.reset(token)

    return wrapped


def _validate_source_spec(spec: ProductionSourceSpec) -> tuple[str, ...]:
    if not isinstance(spec, ProductionSourceSpec):
        raise TypeError("spec must be a ProductionSourceSpec")
    _validate_source_name(spec.source)
    if (
        spec.acceptance_mode != "valid_subset_user_waiver"
        or spec.original_90_percent_gate_passed is not False
    ):
        raise ValueError("source acceptance policy is invalid")
    stages = tuple(COMPONENTS)
    if set(spec.expected_candidate_stages) != set(stages):
        raise ValueError("source candidate stages are invalid")
    if (
        isinstance(spec.expected_frozen, bool)
        or not isinstance(spec.expected_frozen, int)
        or spec.expected_frozen < 0
    ):
        raise ValueError("source frozen count is invalid")
    shard_ids = tuple(Path(path).stem for path in spec.indexes)
    if (
        not shard_ids
        or len(set(spec.indexes)) != len(spec.indexes)
        or len(set(shard_ids)) != len(shard_ids)
        or set(shard_ids) != set(spec.expected_batches)
    ):
        raise ValueError("source indexes do not match shard identities")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > spec.expected_frozen
        for value in spec.expected_candidate_stages.values()
    ):
        raise ValueError("source candidate counts are invalid")
    return shard_ids


def _handoff_counts() -> dict[str, object]:
    return canonical_count_contract(
        frozen=HANDOFF_FROZEN_COUNT,
        global_quarantine=HANDOFF_GLOBAL_QUARANTINE_COUNT,
        shape512_family_exclusions=HANDOFF_SHAPE512_FAMILY_EXCLUSION_COUNT,
        candidate_stages=HANDOFF_CANDIDATE_STAGE_COUNTS,
        training_exclusions=HANDOFF_TRAINING_EXCLUSION_COUNTS,
        stages=HANDOFF_STAGE_COUNTS,
    )


@dataclass(frozen=True)
class StagePreflight:
    stage: str
    root: Path
    asset_count: int
    asset_scope_sha256: str
    anchors_checked: int
    validation_counts: dict[str, int]
    materialization_bytes: bytes
    source: str = SOURCE

    @property
    def materialization_sha256(self) -> str:
        return sha256(self.materialization_bytes).hexdigest()


def _error(stage: str, asset: str | None, message: str, anchor: str | None = None) -> ValueError:
    parts = [f"source={_VALIDATION_SOURCE.get()}", f"stage={stage}"]
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
                maximum = TOKEN_LIMITS[stage]
                if len(integer_coords) > maximum:
                    raise _error(stage, asset, "sparse token count exceeds stage maximum", anchor)
                coords = integer_coords
    except ValueError:
        raise
    except Exception as error:
        raise _error(stage, asset, "invalid latent NPZ", anchor) from error
    scale = _scale(stage, asset, directory / asset / f"{anchor}_scale.json", anchor)
    return coords, scale


def validate_stage_structure(
    source: str,
    stage: str,
    root: Path,
    expected_assets: Sequence[str],
) -> dict[str, int]:
    """Validate all published files for one stage without invoking a dataset loader."""
    token = _VALIDATION_SOURCE.set(_validate_source_name(source))
    try:
        if stage not in COMPONENTS:
            raise _error(stage, None, "unknown stage")
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
                    values[f"{component}:{anchor}"] = _latent(
                        stage, asset, component, root / relative, anchor
                    )
                    latents += 1
                    scales += 1
            if stage == "pbr1024":
                for anchor in (0, 1):
                    shape_coords, shape_scale = values[f"shape:{anchor}"]
                    pbr_coords, pbr_scale = values[f"pbr:{anchor}"]
                    anchor_name = f"view{anchor:02d}"
                    if not np.array_equal(shape_coords, pbr_coords):
                        raise _error(
                            stage,
                            asset,
                            "PBR and Shape coordinates must match",
                            anchor_name,
                        )
                    if not np.isclose(
                        shape_scale, pbr_scale, rtol=SCALE_RTOL, atol=SCALE_ATOL
                    ):
                        raise _error(
                            stage,
                            asset,
                            "PBR and Shape float32 total_scale exceeds policy tolerance",
                            anchor_name,
                        )
        return {
            "assets": len(expected_assets),
            "renders": renders,
            "latents": latents,
            "scales": scales,
        }
    finally:
        _VALIDATION_SOURCE.reset(token)


def stage_data_dir(
    source: str, stage: str, root: Path
) -> dict[str, dict[str, str]]:
    source = _validate_source_name(source)
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
        raise ValueError(f"source={source} stage={stage}: unknown stage")
    return {source: values}


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


def validate_direct_loader(
    source: str,
    stage: str,
    root: Path,
    expected_assets: Sequence[str],
    config_path: Path,
) -> int:
    """Call the configured real dataset directly for both anchors of every asset."""
    source = _validate_source_name(source)
    token = _VALIDATION_SOURCE.set(source)
    try:
        return _validate_direct_loader(
            source, stage, root, expected_assets, config_path
        )
    finally:
        _VALIDATION_SOURCE.reset(token)


def _validate_direct_loader(
    source: str,
    stage: str,
    root: Path,
    expected_assets: Sequence[str],
    config_path: Path,
) -> int:
    if stage not in CONFIGS:
        raise _error(stage, None, "unknown stage")
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
        dataset = dataset_class(
            json.dumps(stage_data_dir(source, stage, Path(root))), **dataset_args
        )
    except Exception as error:
        raise _error(stage, None, f"failed to construct configured dataset {dataset_name}") from error
    expected = set(expected_assets)
    instances = list(dataset.instances)
    actual = {
        asset
        for _root_record, asset, instance_source in instances
        if instance_source == source
    }
    if (
        len(instances) != len(expected_assets)
        or actual != expected
        or any(
            instance_source != source
            for _root_record, _asset, instance_source in instances
        )
    ):
        raise _error(stage, None, "configured dataset instance set does not exactly match materialized scope")
    by_asset = {asset: root_record for root_record, asset, _source in instances}
    image_size = int(dataset_args["image_size"])
    checked = 0
    for asset in expected_assets:
        root_record = by_asset[asset]
        for anchor in (0, 1):
            dataset._current_dataset_name = source
            try:
                with patch.object(np.random, "randint", return_value=anchor):
                    pack = dataset.get_instance(root_record, asset)
            except Exception as error:
                raise RuntimeError(
                    f"source={source} stage={stage} asset={asset} "
                    f"anchor=view{anchor:02d}: direct dataset load failed"
                ) from error
            _validate_loader_pack(stage, asset, anchor, pack, image_size)
            checked += 1
    return checked


def _scope_digest(scope: Sequence[str]) -> str:
    return sha256("\n".join(scope).encode()).hexdigest()


def _allowed_exclusion_reasons(stage: str) -> set[str]:
    if stage == "ss64":
        return set()
    limit = TOKEN_LIMITS[stage]
    reasons = {f"shape_tokens_view{anchor:02d}_exceed_{limit}" for anchor in (0, 1)}
    if stage == "pbr1024":
        reasons.update(f"pbr_tokens_view{anchor:02d}_exceed_{limit}" for anchor in (0, 1))
        reasons.update(f"pbr_shape_coords_view{anchor:02d}_mismatch" for anchor in (0, 1))
        reasons.update(f"pbr_shape_scale_view{anchor:02d}_mismatch" for anchor in (0, 1))
    return reasons


def _eligibility_evidence_is_valid(
    stage: str,
    evidence: Mapping[str, object],
    *,
    candidate_counts: Mapping[str, int] = HANDOFF_CANDIDATE_STAGE_COUNTS,
    training_exclusion_counts: Mapping[
        str, int
    ] = HANDOFF_TRAINING_EXCLUSION_COUNTS,
    stage_counts: Mapping[str, int] = HANDOFF_STAGE_COUNTS,
    frozen_count: int = HANDOFF_FROZEN_COUNT,
    global_quarantine_count: int = HANDOFF_GLOBAL_QUARANTINE_COUNT,
    shape512_family_exclusion_count: int = (
        HANDOFF_SHAPE512_FAMILY_EXCLUSION_COUNT
    ),
) -> bool:
    """Check the exact Task 1 candidate/final eligibility contract."""
    candidate = evidence.get("candidate_stage_scope")
    final = evidence.get("stage_scope")
    exclusions = evidence.get("training_exclusions")
    expected_counts = canonical_count_contract(
        frozen=frozen_count,
        global_quarantine=global_quarantine_count,
        shape512_family_exclusions=shape512_family_exclusion_count,
        candidate_stages=candidate_counts,
        training_exclusions=training_exclusion_counts,
        stages=stage_counts,
    )
    if (
        evidence.get("counts") != expected_counts
        or evidence.get("eligibility_policy") != policy_evidence()
        or evidence.get("candidate_asset_count") != candidate_counts[stage]
        or evidence.get("asset_count") != stage_counts[stage]
        or evidence.get("training_exclusion_count")
        != training_exclusion_counts[stage]
        or not isinstance(candidate, list)
        or not all(isinstance(asset, str) and asset for asset in candidate)
        or candidate != sorted(candidate)
        or len(candidate) != len(set(candidate))
        or len(candidate) != evidence.get("candidate_asset_count")
        or evidence.get("candidate_stage_scope_sha256") != _scope_digest(candidate)
        or not isinstance(final, list)
        or not all(isinstance(asset, str) and asset for asset in final)
        or final != sorted(final)
        or len(final) != len(set(final))
        or len(final) != evidence.get("asset_count")
        or evidence.get("stage_scope_sha256") != _scope_digest(final)
        or not isinstance(exclusions, list)
        or len(exclusions) != training_exclusion_counts[stage]
    ):
        return False
    excluded_assets: list[str] = []
    reasons: list[str] = []
    for exclusion in exclusions:
        if not isinstance(exclusion, Mapping):
            return False
        asset = exclusion.get("asset")
        exclusion_reasons = exclusion.get("reasons")
        if (
            not isinstance(asset, str)
            or not asset
            or not isinstance(exclusion_reasons, list)
            or not exclusion_reasons
            or not all(isinstance(reason, str) and reason for reason in exclusion_reasons)
            or exclusion_reasons != sorted(exclusion_reasons)
            or len(exclusion_reasons) != len(set(exclusion_reasons))
            or not set(exclusion_reasons).issubset(_allowed_exclusion_reasons(stage))
        ):
            return False
        excluded_assets.append(asset)
        reasons.extend(exclusion_reasons)
    if (
        excluded_assets != sorted(excluded_assets)
        or len(excluded_assets) != len(set(excluded_assets))
        or not set(excluded_assets).issubset(candidate)
        or [asset for asset in candidate if asset not in set(excluded_assets)] != final
        or evidence.get("training_exclusion_reason_counts")
        != {reason: reasons.count(reason) for reason in sorted(set(reasons))}
    ):
        return False
    return True


def _materialization_scope_abo(
    stage: str,
    root: Path,
    *,
    candidate_counts: Mapping[str, int] = HANDOFF_CANDIDATE_STAGE_COUNTS,
    training_exclusion_counts: Mapping[
        str, int
    ] = HANDOFF_TRAINING_EXCLUSION_COUNTS,
    stage_counts: Mapping[str, int] = HANDOFF_STAGE_COUNTS,
    frozen_count: int = HANDOFF_FROZEN_COUNT,
    global_quarantine_count: int = HANDOFF_GLOBAL_QUARANTINE_COUNT,
    shape512_family_exclusion_count: int = (
        HANDOFF_SHAPE512_FAMILY_EXCLUSION_COUNT
    ),
) -> tuple[tuple[str, ...], str, bytes]:
    path = Path(root) / "materialization.json"
    _regular(path, stage, None, "materialization.json")
    try:
        raw = _existing_regular_bytes(path)
        evidence = json.loads(raw)
        assets = evidence["stage_scope"]
        digest = evidence["stage_scope_sha256"]
        count = evidence["asset_count"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise _error(stage, None, "invalid materialization evidence") from error
    if "stage_root" not in evidence:
        raise _error(stage, None, "materialization stage root identity is missing")
    source_index = evidence.get("source_index")
    if (
        evidence.get("schema_version") != 1
        or not isinstance(evidence.get("created_at"), str)
        or not evidence.get("created_at")
        or evidence.get("source") != SOURCE
        or evidence.get("shard_id") != "ABO-00000"
        or evidence.get("acceptance_mode") != "valid_subset_user_waiver"
        or evidence.get("original_90_percent_gate_passed") is not False
        or not isinstance(source_index, Mapping)
        or not isinstance(source_index.get("path"), str)
        or not source_index.get("path")
        or source_index.get("sha256") != evidence.get("index_sha256")
    ):
        raise _error(stage, None, "materialization provenance is invalid")
    stage_root = evidence["stage_root"]
    if evidence.get("stage") != stage or not isinstance(assets, list) or not assets or not all(isinstance(asset, str) for asset in assets):
        raise _error(stage, None, "materialization stage identity is invalid")
    canonical_root = str(Path(root).resolve())
    if not isinstance(stage_root, str) or stage_root != canonical_root:
        raise _error(stage, None, "materialization stage root identity mismatch")
    if not _eligibility_evidence_is_valid(
        stage,
        evidence,
        candidate_counts=candidate_counts,
        training_exclusion_counts=training_exclusion_counts,
        stage_counts=stage_counts,
        frozen_count=frozen_count,
        global_quarantine_count=global_quarantine_count,
        shape512_family_exclusion_count=shape512_family_exclusion_count,
    ):
        raise _error(stage, None, "materialization eligibility evidence is invalid")
    if count != len(assets) or digest != _scope_digest(assets):
        raise _error(stage, None, "materialization asset scope is not canonical")
    excluded = {exclusion["asset"] for exclusion in evidence["training_exclusions"]}
    for relative in (RENDER_ROOT, *(relative for relative, _component, _fields in COMPONENTS[stage])):
        for asset in excluded:
            if os.path.lexists(Path(root) / relative / asset):
                raise _error(stage, asset, f"training-excluded asset remains in final component: {relative}")
    return tuple(assets), digest, raw


def _preflight_stage_abo(
    stage: str,
    root: Path,
    config_path: Path,
    *,
    structure_validator=None,
    loader_validator=None,
) -> StagePreflight:
    assets, digest, evidence_bytes = _materialization_scope_abo(stage, root)
    structure_validator = structure_validator or validate_stage_structure
    loader_validator = loader_validator or validate_direct_loader
    counts = structure_validator(SOURCE, stage, root, assets)
    anchors = loader_validator(SOURCE, stage, root, assets, config_path)
    return StagePreflight(
        stage, Path(root), len(assets), digest, anchors, counts, evidence_bytes
    )


def _expected_source_indexes(
    spec: ProductionSourceSpec,
) -> list[dict[str, str]]:
    shard_ids = _validate_source_spec(spec)
    records = []
    for shard_id, index_path in zip(shard_ids, spec.indexes, strict=True):
        canonical = Path(index_path).resolve()
        records.append(
            {
                "shard_id": shard_id,
                "path": str(canonical),
                "sha256": sha256(_existing_regular_bytes(canonical)).hexdigest(),
            }
        )
    return records


def _source_count_contract(
    spec: ProductionSourceSpec,
    materializations: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if spec.fixed_count_contract is not None:
        raise ValueError("generic source reports require observed count evidence")
    exclusions: dict[str, int] = {}
    for stage in COMPONENTS:
        evidence = materializations[stage]
        value = evidence.get("training_exclusion_count")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"materialization training exclusion count is invalid for stage={stage}"
            )
        exclusions[stage] = value
    return observed_count_contract(
        frozen=spec.expected_frozen,
        candidate_stages=spec.expected_candidate_stages,
        training_exclusions=exclusions,
    )


def _source_stage_count_contract(
    spec: ProductionSourceSpec,
    stage: str,
    evidence: Mapping[str, object],
) -> dict[str, object] | None:
    """Rebuild the stage-local count view persisted by the materializer."""
    frozen = evidence.get("frozen_assets")
    quarantined = evidence.get("quarantined_assets")
    shape512_exclusions = evidence.get("shape512_exclusions")
    training_exclusions = evidence.get("training_exclusion_count")
    values = (frozen, quarantined, shape512_exclusions, training_exclusions)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        return None
    if (
        frozen != spec.expected_frozen
        or quarantined > frozen
        or shape512_exclusions > frozen
    ):
        return None
    candidate = spec.expected_candidate_stages[stage]
    if training_exclusions > candidate:
        return None
    return {
        "frozen": frozen,
        "global_quarantine": quarantined,
        "shape512_family_exclusions": shape512_exclusions,
        "candidate_stages": {stage: candidate},
        "pack_exclusions": {stage: frozen - candidate},
        "training_exclusions": {stage: training_exclusions},
        "stages": {stage: candidate - training_exclusions},
    }


def _source_eligibility_evidence_is_valid(
    spec: ProductionSourceSpec,
    stage: str,
    evidence: Mapping[str, object],
    counts: Mapping[str, object],
) -> bool:
    stage_counts = _source_stage_count_contract(spec, stage, evidence)
    if stage_counts is None:
        return False
    candidate = evidence.get("candidate_stage_scope")
    final = evidence.get("stage_scope")
    exclusions = evidence.get("training_exclusions")
    candidate_count = spec.expected_candidate_stages[stage]
    try:
        exclusion_count = counts["training_exclusions"][stage]
        final_count = counts["stages"][stage]
    except (KeyError, TypeError):
        return False
    if (
        evidence.get("counts") != stage_counts
        or stage_counts["training_exclusions"][stage] != exclusion_count
        or stage_counts["stages"][stage] != final_count
        or evidence.get("eligibility_policy") != policy_evidence()
        or evidence.get("candidate_asset_count") != candidate_count
        or evidence.get("asset_count") != final_count
        or evidence.get("training_exclusion_count") != exclusion_count
        or not isinstance(candidate, list)
        or not all(isinstance(asset, str) and asset for asset in candidate)
        or candidate != sorted(candidate)
        or len(candidate) != len(set(candidate))
        or len(candidate) != candidate_count
        or evidence.get("candidate_stage_scope_sha256")
        != _scope_digest(candidate)
        or not isinstance(final, list)
        or not all(isinstance(asset, str) and asset for asset in final)
        or final != sorted(final)
        or len(final) != len(set(final))
        or len(final) != final_count
        or evidence.get("stage_scope_sha256") != _scope_digest(final)
        or not isinstance(exclusions, list)
        or len(exclusions) != exclusion_count
    ):
        return False
    excluded_assets: list[str] = []
    reasons: list[str] = []
    allowed_reasons = _allowed_exclusion_reasons(stage)
    for exclusion in exclusions:
        if not isinstance(exclusion, Mapping):
            return False
        asset = exclusion.get("asset")
        exclusion_reasons = exclusion.get("reasons")
        if (
            not isinstance(asset, str)
            or not asset
            or not isinstance(exclusion_reasons, list)
            or not exclusion_reasons
            or not all(
                isinstance(reason, str) and reason
                for reason in exclusion_reasons
            )
            or exclusion_reasons != sorted(exclusion_reasons)
            or len(exclusion_reasons) != len(set(exclusion_reasons))
            or not set(exclusion_reasons).issubset(allowed_reasons)
        ):
            return False
        excluded_assets.append(asset)
        reasons.extend(exclusion_reasons)
    return (
        excluded_assets == sorted(excluded_assets)
        and len(excluded_assets) == len(set(excluded_assets))
        and set(excluded_assets).issubset(candidate)
        and [asset for asset in candidate if asset not in set(excluded_assets)]
        == final
        and evidence.get("training_exclusion_reason_counts")
        == {reason: reasons.count(reason) for reason in sorted(set(reasons))}
    )


def _validated_source_materialization(
    spec: ProductionSourceSpec,
    stage: str,
    root: Path,
    counts: Mapping[str, object],
    expected_indexes: Sequence[Mapping[str, str]],
) -> tuple[tuple[str, ...], str, bytes, dict[str, object]]:
    source = _validate_source_name(spec.source)
    token = _VALIDATION_SOURCE.set(source)
    try:
        path = Path(root) / "materialization.json"
        _regular(path, stage, None, "materialization.json")
        raw = _existing_regular_bytes(path)
        try:
            evidence = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise _error(stage, None, "invalid materialization evidence") from error
        if not isinstance(evidence, dict):
            raise _error(stage, None, "invalid materialization evidence")
        indexes = evidence.get("source_indexes")
        if (
            evidence.get("schema_version") != 1
            or not isinstance(evidence.get("created_at"), str)
            or not evidence.get("created_at")
            or evidence.get("source") != source
            or indexes != list(expected_indexes)
            or evidence.get("acceptance_mode") != spec.acceptance_mode
            or evidence.get("original_90_percent_gate_passed")
            is not spec.original_90_percent_gate_passed
        ):
            raise _error(stage, None, "materialization provenance is invalid")
        canonical_root = str(Path(root).resolve())
        assets = evidence.get("stage_scope")
        digest = evidence.get("stage_scope_sha256")
        if (
            evidence.get("stage") != stage
            or evidence.get("stage_root") != canonical_root
            or not _source_eligibility_evidence_is_valid(
                spec, stage, evidence, counts
            )
            or not isinstance(assets, list)
            or not assets
            or digest != _scope_digest(assets)
        ):
            raise _error(
                stage, None, "materialization scope or eligibility is invalid"
            )
        excluded = {
            exclusion["asset"] for exclusion in evidence["training_exclusions"]
        }
        for relative in (
            RENDER_ROOT,
            *(
                relative
                for relative, _component, _fields in COMPONENTS[stage]
            ),
        ):
            for asset in excluded:
                if os.path.lexists(Path(root) / relative / asset):
                    raise _error(
                        stage,
                        asset,
                        "training-excluded asset remains in final component: "
                        f"{relative}",
                    )
        return tuple(assets), digest, raw, evidence
    finally:
        _VALIDATION_SOURCE.reset(token)


@_source_context_entrypoint
def preflight_stage(
    spec: ProductionSourceSpec,
    stage: str,
    root: Path,
    config_path: Path,
) -> StagePreflight:
    """Strictly validate one source-aware materialized stage."""
    _validate_source_spec(spec)
    if spec.fixed_count_contract is not None:
        raise ValueError("generic preflight requires observed source counts")
    try:
        document = json.loads(
            _existing_regular_bytes(Path(root) / "materialization.json")
        )
        persisted_counts = document["counts"]
        persisted_exclusions = persisted_counts["training_exclusions"]
    except (
        FileExistsError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        KeyError,
        TypeError,
    ) as error:
        raise ValueError(
            f"invalid materialization count evidence for stage={stage}"
        ) from error
    if not isinstance(persisted_exclusions, Mapping):
        raise ValueError(
            f"invalid materialization count evidence for stage={stage}"
        )
    if set(persisted_exclusions) != {stage}:
        raise ValueError(
            f"invalid materialization count evidence for stage={stage}"
        )
    stage_exclusions = {name: 0 for name in COMPONENTS}
    stage_exclusions[stage] = persisted_exclusions[stage]
    counts = observed_count_contract(
        frozen=spec.expected_frozen,
        candidate_stages=spec.expected_candidate_stages,
        training_exclusions=stage_exclusions,
    )
    indexes = _expected_source_indexes(spec)
    assets, digest, evidence_bytes, _evidence = _validated_source_materialization(
        spec, stage, root, counts, indexes
    )
    counts_checked = validate_stage_structure(spec.source, stage, root, assets)
    anchors = validate_direct_loader(
        spec.source, stage, root, assets, config_path
    )
    return StagePreflight(
        stage,
        Path(root),
        len(assets),
        digest,
        anchors,
        counts_checked,
        evidence_bytes,
        spec.source,
    )


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _existing_regular_bytes(path: Path) -> bytes:
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise FileExistsError(f"existing path is not a regular non-symlink file: {path}") from error
    if not stat.S_ISREG(mode):
        raise FileExistsError(f"existing path is not a regular non-symlink file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise FileExistsError(f"existing path is not a regular non-symlink file: {path}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise FileExistsError(f"existing path is not a regular non-symlink file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_create_only_json(path: Path, value: Mapping[str, object]) -> str:
    """Publish canonical JSON once, accepting only byte-identical reruns."""
    path = Path(path)
    payload = _canonical_json_bytes(value)
    digest = sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        if _existing_regular_bytes(path) != payload:
            raise FileExistsError(f"existing create-only JSON has different content: {path}")
        return digest
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _existing_regular_bytes(path) != payload:
                raise FileExistsError(f"existing create-only JSON has different content: {path}")
        else:
            _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _write_atomic_json(path: Path, value: Mapping[str, object]) -> str:
    path = Path(path)
    payload = _canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256(payload).hexdigest()


def _load_existing_json(path: Path, label: str) -> tuple[dict[str, object], bytes] | None:
    path = Path(path)
    if not os.path.lexists(path):
        return None
    raw = _existing_regular_bytes(path)
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid existing {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid existing {label}: {path}")
    return value, raw


def _validated_handoff_inputs(
    results: Mapping[str, StagePreflight],
    materializations: Mapping[str, Mapping[str, object]],
    index_sha256: str | None = None,
    index_path: Path | None = None,
) -> None:
    stages = tuple(HANDOFF_STAGE_COUNTS)
    if set(results) != set(stages) or set(materializations) != set(stages):
        raise ValueError("all four strict preflight results and materializations are required")
    for stage, expected_count in HANDOFF_STAGE_COUNTS.items():
        result = results[stage]
        if (
            result.stage != stage
            or result.asset_count != expected_count
            or result.anchors_checked != expected_count * 2
            or result.validation_counts.get("assets") != expected_count
        ):
            raise ValueError(f"strict preflight did not complete successfully for stage={stage}")
        evidence = materializations[stage]
        scope = evidence.get("stage_scope")
        scope_digest = sha256("\n".join(scope).encode()).hexdigest() if isinstance(scope, list) and all(isinstance(asset, str) for asset in scope) else None
        source_index = evidence.get("source_index")
        try:
            validated_evidence = json.loads(result.materialization_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            validated_evidence = None
        if (
            evidence.get("schema_version") != 1
            or not isinstance(evidence.get("created_at"), str)
            or not evidence.get("created_at")
            or evidence.get("source") != SOURCE
            or evidence.get("shard_id") != "ABO-00000"
            or evidence.get("acceptance_mode") != "valid_subset_user_waiver"
            or evidence.get("original_90_percent_gate_passed") is not False
            or not _eligibility_evidence_is_valid(stage, evidence)
            or not isinstance(source_index, Mapping)
            or not isinstance(source_index.get("path"), str)
            or not source_index.get("path")
            or source_index.get("sha256") != evidence.get("index_sha256")
            or (index_path is not None and source_index.get("path") != str(index_path))
            or evidence.get("stage") != stage
            or evidence.get("stage_root") != str(result.root.resolve())
            or evidence.get("asset_count") != result.asset_count
            or evidence.get("stage_scope_sha256") != result.asset_scope_sha256
            or not isinstance(scope, list)
            or not scope
            or scope != sorted(scope)
            or len(set(scope)) != len(scope)
            or len(scope) != result.asset_count
            or scope_digest != result.asset_scope_sha256
            or not isinstance(result.materialization_sha256, str)
            or len(result.materialization_sha256) != 64
            or not isinstance(result.materialization_bytes, bytes)
            or validated_evidence != evidence
            or (index_sha256 is not None and evidence.get("index_sha256") != index_sha256)
        ):
            raise ValueError(f"materialization evidence does not match strict preflight for stage={stage}")


def _materialization_evidence(
    results: Mapping[str, StagePreflight],
    materializations: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, dict[str, object]], list[str]]:
    summaries: dict[str, dict[str, object]] = {}
    observed: set[str] = set()
    for stage in HANDOFF_STAGE_COUNTS:
        evidence = materializations[stage]
        commits = evidence.get("tool_commits")
        packs = evidence.get("packs")
        if not isinstance(commits, list) or not all(isinstance(commit, str) and commit for commit in commits):
            raise ValueError(f"materialization evidence lacks tool commits for stage={stage}")
        if not isinstance(packs, list) or not all(isinstance(pack, dict) for pack in packs):
            raise ValueError(f"materialization evidence lacks pack records for stage={stage}")
        pack_commits = [pack.get("tool_commit") for pack in packs]
        if not all(isinstance(commit, str) and commit for commit in pack_commits):
            raise ValueError(f"materialization evidence lacks observed pack tool commits for stage={stage}")
        observed.update(commits)
        observed.update(pack_commits)
        summaries[stage] = {
            "sha256": results[stage].materialization_sha256,
            "tool_commits": sorted(set(commits) | set(pack_commits)),
        }
    return summaries, sorted(observed)


def _stage_records(results: Mapping[str, StagePreflight]) -> dict[str, dict[str, object]]:
    return {
        stage: {
            "root": str(results[stage].root),
            "asset_count": results[stage].asset_count,
            "asset_scope_sha256": results[stage].asset_scope_sha256,
            "anchors_checked": results[stage].anchors_checked,
            "validation_counts": results[stage].validation_counts,
            "data_dir": stage_data_dir(SOURCE, stage, results[stage].root),
        }
        for stage in HANDOFF_STAGE_COUNTS
    }


def _validated_source_handoff_inputs(
    spec: ProductionSourceSpec,
    results: Mapping[str, StagePreflight],
    materializations: Mapping[str, Mapping[str, object]],
) -> tuple[
    dict[str, object],
    list[dict[str, str]],
    dict[str, dict[str, object]],
    list[str],
]:
    _validate_source_spec(spec)
    stages = tuple(COMPONENTS)
    if set(results) != set(stages) or set(materializations) != set(stages):
        raise ValueError("all four strict preflight stages are required")
    counts = _source_count_contract(spec, materializations)
    indexes = _expected_source_indexes(spec)
    summaries: dict[str, dict[str, object]] = {}
    observed_commits: set[str] = set()
    for stage in stages:
        result = results[stage]
        evidence = materializations[stage]
        try:
            validated_evidence = json.loads(result.materialization_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            validated_evidence = None
        expected_count = counts["stages"][stage]
        scope = evidence.get("stage_scope")
        if evidence.get("source_indexes") != indexes:
            raise ValueError(
                "materialization source index evidence does not match "
                f"current indexes for stage={stage}"
            )
        if (
            result.source != spec.source
            or result.stage != stage
            or result.asset_count != expected_count
            or result.anchors_checked != expected_count * 2
            or result.validation_counts.get("assets") != expected_count
            or evidence.get("source") != spec.source
            or evidence.get("stage") != stage
            or evidence.get("stage_root") != str(result.root.resolve())
            or evidence.get("asset_count") != result.asset_count
            or evidence.get("stage_scope_sha256")
            != result.asset_scope_sha256
            or not isinstance(scope, list)
            or not scope
            or scope != sorted(scope)
            or len(scope) != len(set(scope))
            or _scope_digest(scope) != result.asset_scope_sha256
            or validated_evidence != evidence
            or not _source_eligibility_evidence_is_valid(
                spec, stage, evidence, counts
            )
        ):
            raise ValueError(
                "materialization evidence does not match strict preflight "
                f"for stage={stage}"
            )
        current_path = Path(result.root) / "materialization.json"
        if _existing_regular_bytes(current_path) != result.materialization_bytes:
            raise ValueError(
                "materialization evidence bytes changed after strict preflight: "
                f"{current_path}"
            )
        commits = evidence.get("tool_commits")
        packs = evidence.get("packs")
        if (
            not isinstance(commits, list)
            or not all(isinstance(commit, str) and commit for commit in commits)
            or not isinstance(packs, list)
            or not all(isinstance(pack, Mapping) for pack in packs)
        ):
            raise ValueError(
                f"materialization evidence lacks tool provenance for stage={stage}"
            )
        pack_commits = [pack.get("tool_commit") for pack in packs]
        if not all(
            isinstance(commit, str) and commit for commit in pack_commits
        ):
            raise ValueError(
                f"materialization packs lack tool commits for stage={stage}"
            )
        stage_commits = sorted(set(commits) | set(pack_commits))
        observed_commits.update(stage_commits)
        summaries[stage] = {
            "sha256": result.materialization_sha256,
            "tool_commits": stage_commits,
        }
    return counts, indexes, summaries, sorted(observed_commits)


@_source_context_entrypoint
def build_source_report(
    spec: ProductionSourceSpec,
    results: Mapping[str, StagePreflight],
    materializations: Mapping[str, Mapping[str, object]],
    created_at: str,
) -> dict[str, object]:
    """Build schema-2 source evidence bound to every source index and stage."""
    if not isinstance(created_at, str) or not created_at:
        raise ValueError("created_at must be a non-empty timestamp")
    counts, indexes, evidence, observed = _validated_source_handoff_inputs(
        spec, results, materializations
    )
    stages = {
        stage: {
            "root": str(results[stage].root),
            "asset_count": results[stage].asset_count,
            "asset_scope_sha256": results[stage].asset_scope_sha256,
            "anchors_checked": results[stage].anchors_checked,
            "validation_counts": results[stage].validation_counts,
            "data_dir": stage_data_dir(
                spec.source, stage, results[stage].root
            ),
        }
        for stage in COMPONENTS
    }
    return {
        "schema_version": 2,
        "created_at": created_at,
        "source": spec.source,
        "source_indexes": indexes,
        "acceptance_mode": spec.acceptance_mode,
        "original_90_percent_gate_passed":
            spec.original_90_percent_gate_passed,
        "authorization": "training-input use only",
        "counts": counts,
        "eligibility_policy": policy_evidence(),
        "stages": stages,
        "materialization_evidence": evidence,
        "observed_tool_commits": observed,
    }


def _source_materializations_from_results(
    spec: ProductionSourceSpec,
    results: Mapping[str, StagePreflight],
) -> dict[str, dict[str, object]]:
    if set(results) != set(COMPONENTS):
        raise ValueError("all four strict preflight stages are required")
    materializations = {}
    for stage in COMPONENTS:
        result = results[stage]
        path = Path(result.root) / "materialization.json"
        current = _existing_regular_bytes(path)
        if current != result.materialization_bytes:
            raise ValueError(
                "materialization evidence bytes changed after strict preflight: "
                f"{path}"
            )
        try:
            value = json.loads(current)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(f"invalid materialization evidence: {path}") from error
        if not isinstance(value, dict) or value.get("source") != spec.source:
            raise ValueError(f"invalid materialization evidence: {path}")
        materializations[stage] = value
    return materializations


def _source_handoff_document(
    spec: ProductionSourceSpec,
    report_path: Path,
    report_sha256: str,
    report: Mapping[str, object],
) -> dict[str, object]:
    return {
        key: report[key]
        for key in (
            "schema_version",
            "created_at",
            "source",
            "source_indexes",
            "acceptance_mode",
            "original_90_percent_gate_passed",
            "authorization",
            "counts",
            "eligibility_policy",
            "stages",
            "materialization_evidence",
            "observed_tool_commits",
        )
    } | {
        "report": {
            "path": str(Path(report_path)),
            "sha256": report_sha256,
        }
    }


@_source_context_entrypoint
def publish_source_handoff(
    spec: ProductionSourceSpec,
    results: Mapping[str, StagePreflight],
    report_path: Path,
    handoff_path: Path,
    training_data_path: Path,
) -> tuple[Path, Path, Path]:
    """Publish schema-2 shared evidence before the local training manifest."""
    materializations = _source_materializations_from_results(spec, results)
    existing_report = _load_existing_json(report_path, "report")
    if existing_report is None:
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        report = build_source_report(
            spec, results, materializations, created_at
        )
        report_sha256 = write_create_only_json(report_path, report)
    else:
        report, raw_report = existing_report
        created_at = report.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            raise ValueError("invalid existing report creation time")
        expected_report = build_source_report(
            spec, results, materializations, created_at
        )
        if raw_report != _canonical_json_bytes(expected_report):
            raise ValueError(
                "existing report does not match current strict preflight evidence"
            )
        report_sha256 = sha256(raw_report).hexdigest()
    handoff = _source_handoff_document(
        spec, report_path, report_sha256, report
    )
    existing_handoff = _load_existing_json(handoff_path, "handoff")
    if existing_handoff is None:
        handoff_sha256 = write_create_only_json(handoff_path, handoff)
    else:
        existing_value, raw_handoff = existing_handoff
        if raw_handoff != _canonical_json_bytes(handoff):
            raise ValueError(
                "existing handoff does not match current report transaction"
            )
        handoff_sha256 = sha256(raw_handoff).hexdigest()
    training_data = {
        **handoff,
        "handoff": {
            "path": str(Path(handoff_path)),
            "sha256": handoff_sha256,
        },
    }
    write_create_only_json(training_data_path, training_data)
    return Path(report_path), Path(handoff_path), Path(training_data_path)


def build_report(
    index_path: Path,
    index_sha256: str,
    results: Mapping[str, StagePreflight],
    materializations: Mapping[str, Mapping[str, object]],
    created_at: str,
) -> dict[str, object]:
    """Build the immutable evidence report for the approved ABO valid subset."""
    index_path = Path(index_path).resolve()
    _validated_handoff_inputs(results, materializations, index_sha256, index_path)
    evidence, observed_tool_commits = _materialization_evidence(results, materializations)
    stages = _stage_records(results)
    return {
        "schema_version": 1,
        "created_at": created_at,
        "source": SOURCE,
        "shard_id": "ABO-00000",
        "source_index": {"path": str(index_path), "sha256": index_sha256},
        "acceptance_mode": "valid_subset_user_waiver",
        "original_90_percent_gate_passed": False,
        "authorization": "training-input use only",
        "counts": _handoff_counts(),
        "eligibility_policy": policy_evidence(),
        "stages": stages,
        "materialization_evidence": evidence,
        "observed_tool_commits": observed_tool_commits,
    }


def build_handoff(
    report_path: Path,
    report_sha256: str,
    report: Mapping[str, object],
    results: Mapping[str, StagePreflight],
    materializations: Mapping[str, Mapping[str, object]],
    created_at: str,
) -> dict[str, object]:
    """Build the training-input-only handoff that pins the report by digest."""
    source_index = report.get("source_index")
    if not isinstance(source_index, Mapping):
        raise ValueError("report source_index must be an object")
    index_path = source_index.get("path")
    index_sha256 = source_index.get("sha256")
    if (
        not isinstance(index_path, str)
        or not index_path
        or not isinstance(index_sha256, str)
        or len(index_sha256) != 64
    ):
        raise ValueError("report source_index path and digest are invalid")
    canonical_index_path = str(Path(index_path).resolve())
    if index_path != canonical_index_path:
        raise ValueError("report source_index path must be canonical")
    report_created_at = report.get("created_at")
    if not isinstance(report_created_at, str) or not report_created_at or report_created_at != created_at:
        raise ValueError("report creation time does not match handoff transaction")
    _validated_handoff_inputs(results, materializations, index_sha256, Path(canonical_index_path))
    expected_digest = sha256(_canonical_json_bytes(report)).hexdigest()
    if report_sha256 != expected_digest:
        raise ValueError("report digest does not match canonical report bytes")
    expected_report = build_report(
        Path(canonical_index_path), index_sha256, results, materializations, report_created_at
    )
    if report != expected_report:
        raise ValueError("report is not bound to supplied preflight evidence")
    evidence, observed_tool_commits = _materialization_evidence(results, materializations)
    return {
        "schema_version": 1,
        "created_at": created_at,
        "source": SOURCE,
        "shard_id": "ABO-00000",
        "acceptance_mode": "valid_subset_user_waiver",
        "original_90_percent_gate_passed": False,
        "authorization": "training-input use only",
        "counts": report["counts"],
        "eligibility_policy": policy_evidence(),
        "stages": report["stages"],
        "source_index": {"path": canonical_index_path, "sha256": index_sha256},
        "materialization_evidence": evidence,
        "observed_tool_commits": observed_tool_commits,
        "report": {"path": str(report_path), "sha256": report_sha256},
    }


def publish_handoff(
    index_path: Path,
    results: Mapping[str, StagePreflight],
    materializations: Mapping[str, Mapping[str, object]],
    report_path: Path,
    handoff_path: Path,
    training_data_path: Path,
    created_at: str,
) -> tuple[Path, Path, Path]:
    """Create shared immutable evidence before atomically writing local input data."""
    index_path = Path(index_path).resolve()
    index_sha256 = sha256(_existing_regular_bytes(index_path)).hexdigest()
    existing_report = _load_existing_json(report_path, "report")
    if existing_report is not None:
        report, raw_report = existing_report
        report_created_at = report.get("created_at")
        if not isinstance(report_created_at, str) or not report_created_at:
            raise ValueError("invalid existing report creation time")
        expected_report = build_report(index_path, index_sha256, results, materializations, report_created_at)
        if raw_report != _canonical_json_bytes(expected_report):
            raise ValueError("existing report does not match current strict preflight evidence")
        report_sha256 = sha256(raw_report).hexdigest()
        created_at = report_created_at
    else:
        report = build_report(index_path, index_sha256, results, materializations, created_at)
        report_sha256 = write_create_only_json(Path(report_path), report)
    handoff = build_handoff(
        Path(report_path), report_sha256, report, results, materializations, created_at
    )
    existing_handoff = _load_existing_json(handoff_path, "handoff")
    if existing_handoff is not None:
        existing_value, raw_handoff = existing_handoff
        if raw_handoff != _canonical_json_bytes(handoff):
            raise ValueError("existing handoff does not match current report transaction")
        handoff_sha256 = sha256(raw_handoff).hexdigest()
    else:
        handoff_sha256 = write_create_only_json(Path(handoff_path), handoff)
    training_data = {
        "schema_version": 1,
        "created_at": created_at,
        "source": SOURCE,
        "shard_id": "ABO-00000",
        "acceptance_mode": "valid_subset_user_waiver",
        "original_90_percent_gate_passed": False,
        "authorization": "training-input use only",
        "counts": report["counts"],
        "eligibility_policy": handoff["eligibility_policy"],
        "stages": report["stages"],
        "source_index": report["source_index"],
        "materialization_evidence": handoff["materialization_evidence"],
        "observed_tool_commits": handoff["observed_tool_commits"],
        "report": handoff["report"],
        "handoff": {"path": str(handoff_path), "sha256": handoff_sha256},
    }
    write_create_only_json(Path(training_data_path), training_data)
    return Path(report_path), Path(handoff_path), Path(training_data_path)


def _materialization_evidence_from_result(
    result: StagePreflight,
) -> dict[str, object]:
    path = Path(result.root) / "materialization.json"
    _regular(path, "handoff", None, "materialization.json")
    current = _existing_regular_bytes(path)
    if current != result.materialization_bytes:
        raise ValueError(
            f"materialization evidence bytes changed after strict preflight: {path}"
        )
    try:
        value = json.loads(result.materialization_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid materialization evidence: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid materialization evidence: {path}")
    return value
