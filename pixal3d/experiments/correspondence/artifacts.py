"""Atomic, resumable artifact bundles for controlled corruptions."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .corruptions import ControlledCorruption
from .inputs import CalibratedView, ForegroundMask


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EXPECTED_KINDS = {
    "c1": "c1_color",
    "c2": "c2_pattern",
    "c3": "c3_deletion",
}
_CONTACT_TILE = 192
_CONTACT_PADDING = 8
_CONTACT_LABEL_HEIGHT = 24
_CONTACT_COLUMNS = 4


@dataclass(frozen=True)
class BundleCorruption:
    """A corruption artifact and the calibrated view it modifies."""

    arm: str
    view_index: int
    corruption: ControlledCorruption


def _canonical_json(value: object) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest must contain finite JSON values") from exc
    return (rendered + "\n").encode("utf-8")


def _png_bytes(image: Image.Image) -> bytes:
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=9)
    return stream.getvalue()


def _mask_image(mask: torch.Tensor, *, expected_size: tuple[int, int]) -> Image.Image:
    if (
        not isinstance(mask, torch.Tensor)
        or mask.ndim != 2
        or mask.dtype != torch.bool
    ):
        raise ValueError("artifact mask must be a bool [H,W] tensor")
    if mask.device.type != "cpu":
        raise ValueError("artifact mask must be on CPU")
    if (mask.shape[1], mask.shape[0]) != expected_size:
        raise ValueError("artifact mask dimensions must match its calibrated view")
    pixels = mask.to(dtype=torch.uint8).mul(255).numpy()
    return Image.fromarray(pixels, mode="L")


def _tensor_image(
    image: torch.Tensor, *, expected_size: tuple[int, int]
) -> Image.Image:
    if (
        not isinstance(image, torch.Tensor)
        or image.ndim != 3
        or image.shape[0] != 3
        or image.dtype != torch.float32
    ):
        raise ValueError("corrupted image must be a float32 [3,H,W] tensor")
    if image.device.type != "cpu":
        raise ValueError("corrupted image must be on CPU")
    if (image.shape[2], image.shape[1]) != expected_size:
        raise ValueError("corrupted image dimensions must match its calibrated view")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("corrupted image values must be finite and in [0,1]")
    pixels = image.mul(255).round().to(dtype=torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(pixels, mode="RGB")


def _saved_png_oracle_mask(
    source: Image.Image,
    corrupted: Image.Image,
    reported_mask: torch.Tensor,
) -> torch.Tensor:
    """Keep only reported pixels that still exceed the saved PNG threshold."""
    expected_size = source.size
    _mask_image(reported_mask, expected_size=expected_size)
    source_pixels = torch.from_numpy(
        np.asarray(source.convert("RGB"), dtype=np.int16).copy()
    )
    corrupted_pixels = torch.from_numpy(
        np.asarray(corrupted.convert("RGB"), dtype=np.int16).copy()
    )
    saved_change = (corrupted_pixels - source_pixels).abs().amax(dim=2).gt(1)
    return reported_mask & saved_change


def _artifact(path: str, payload: bytes) -> dict[str, str]:
    return {"path": path, "sha256": sha256(payload).hexdigest()}


def _fit_contact_tile(image: Image.Image) -> Image.Image:
    tile = Image.new("RGB", (_CONTACT_TILE, _CONTACT_TILE), (238, 238, 238))
    content = image.convert("RGB")
    content.thumbnail(
        (_CONTACT_TILE, _CONTACT_TILE), resample=Image.Resampling.LANCZOS
    )
    offset = (
        (_CONTACT_TILE - content.width) // 2,
        (_CONTACT_TILE - content.height) // 2,
    )
    tile.paste(content, offset)
    return tile


def _mask_overlay(
    image: Image.Image, mask: torch.Tensor, *, color: tuple[int, int, int]
) -> Image.Image:
    base = image.convert("RGB")
    mask_image = _mask_image(mask, expected_size=base.size)
    tint = Image.new("RGB", base.size, color)
    tinted = Image.blend(base, tint, 0.45)
    return Image.composite(tinted, base, mask_image)


def _render_contact_sheet(
    views: Sequence[CalibratedView],
    foreground_masks: Sequence[ForegroundMask],
    corruptions: Sequence[BundleCorruption],
) -> Image.Image:
    ordered = sorted(corruptions, key=lambda item: item.arm)
    row_height = _CONTACT_LABEL_HEIGHT + _CONTACT_TILE + _CONTACT_PADDING
    width = _CONTACT_PADDING + _CONTACT_COLUMNS * (
        _CONTACT_TILE + _CONTACT_PADDING
    )
    height = _CONTACT_PADDING + len(ordered) * row_height
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    column_names = ("source", "foreground overlay", "corrupted", "oracle overlay")

    for row, entry in enumerate(ordered):
        view = views[entry.view_index]
        foreground = foreground_masks[entry.view_index]
        corrupted = _tensor_image(
            entry.corruption.image, expected_size=view.image.size
        )
        images = (
            view.image,
            _mask_overlay(view.image, foreground.mask, color=(255, 80, 40)),
            corrupted,
            _mask_overlay(
                corrupted, entry.corruption.oracle_mask, color=(255, 0, 180)
            ),
        )
        label_y = _CONTACT_PADDING + row * row_height
        image_y = label_y + _CONTACT_LABEL_HEIGHT
        for column, (name, image) in enumerate(zip(column_names, images)):
            x = _CONTACT_PADDING + column * (_CONTACT_TILE + _CONTACT_PADDING)
            draw.text(
                (x + 4, label_y + 5),
                f"{entry.arm.upper()} {name}",
                fill="black",
                font=font,
            )
            sheet.paste(_fit_contact_tile(image), (x, image_y))
    return sheet


def _render_failure_contact_sheet(failure_reason: str) -> Image.Image:
    width = _CONTACT_PADDING + _CONTACT_COLUMNS * (
        _CONTACT_TILE + _CONTACT_PADDING
    )
    sheet = Image.new("RGB", (width, 128), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.rectangle((8, 8, width - 8, 36), fill=(150, 20, 20))
    draw.text((16, 16), "CONTROLLED CORRUPTION RUN FAILED", fill="white", font=font)
    draw.text((16, 52), "Failure reason:", fill="black", font=font)
    draw.text((16, 72), failure_reason, fill=(120, 0, 0), font=font)
    return sheet


def _request_payload(
    run_id: str,
    views: Sequence[CalibratedView],
    foreground_masks: Sequence[ForegroundMask],
    corruptions: Sequence[BundleCorruption],
    *,
    seed: int,
    mesh_scale: float,
) -> tuple[dict, dict[str, bytes]]:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    try:
        scale = float(mesh_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("mesh_scale must be finite and positive") from exc
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("mesh_scale must be finite and positive")
    if not views:
        raise ValueError("artifact bundle requires at least one calibrated view")
    if len(views) != len(foreground_masks):
        raise ValueError("each calibrated view requires one foreground mask")

    files: dict[str, bytes] = {}
    view_records = []
    for view_index, (view, foreground) in enumerate(zip(views, foreground_masks)):
        if not isinstance(view, CalibratedView) or not isinstance(
            foreground, ForegroundMask
        ):
            raise ValueError("views and foreground_masks have invalid entries")
        if view.transform_matrix.device.type != "cpu":
            raise ValueError("calibrated transforms must be on CPU")
        source_path = f"views/view_{view_index:02d}/source.png"
        foreground_path = f"views/view_{view_index:02d}/foreground_mask.png"
        source_bytes = _png_bytes(view.image)
        foreground_bytes = _png_bytes(
            _mask_image(foreground.mask, expected_size=view.image.size)
        )
        files[source_path] = source_bytes
        files[foreground_path] = foreground_bytes
        source_record = _artifact(source_path, source_bytes)
        source_record["input_sha256"] = view.source_sha256
        foreground_record = _artifact(foreground_path, foreground_bytes)
        foreground_record.update(
            {
                "provenance": foreground.provenance,
                "source_sha256": foreground.source_sha256,
            }
        )
        view_records.append(
            {
                "view_index": view_index,
                "frame_index": view.frame_index,
                "camera_angle_x": view.camera_angle_x,
                "distance": view.distance,
                "transform_matrix": view.transform_matrix.tolist(),
                "source": source_record,
                "foreground_mask": foreground_record,
            }
        )

    if not all(isinstance(item, BundleCorruption) for item in corruptions):
        raise ValueError("corruptions have invalid entries")
    ordered = sorted(corruptions, key=lambda item: item.arm)
    if [item.arm for item in ordered] != ["c1", "c2", "c3"]:
        raise ValueError("completed bundles require exactly one c1, c2, and c3 arm")
    corruption_records = []
    serialized_corruptions = []
    for entry in ordered:
        if not 0 <= entry.view_index < len(views):
            raise ValueError("corruption view_index is outside the calibrated views")
        if entry.corruption.kind != _EXPECTED_KINDS[entry.arm]:
            raise ValueError(f"{entry.arm} has an incompatible corruption kind")
        view = views[entry.view_index]
        image_path = (
            f"corruptions/{entry.arm}/view_{entry.view_index:02d}/image.png"
        )
        oracle_path = (
            f"corruptions/{entry.arm}/view_{entry.view_index:02d}/oracle_mask.png"
        )
        image = _tensor_image(
            entry.corruption.image, expected_size=view.image.size
        )
        saved_oracle = _saved_png_oracle_mask(
            view.image, image, entry.corruption.oracle_mask
        )
        image_bytes = _png_bytes(image)
        oracle_bytes = _png_bytes(
            _mask_image(saved_oracle, expected_size=view.image.size)
        )
        files[image_path] = image_bytes
        files[oracle_path] = oracle_bytes
        corruption_records.append(
            {
                "arm": entry.arm,
                "view_index": entry.view_index,
                "kind": entry.corruption.kind,
                "parameters": entry.corruption.parameters,
                "image": _artifact(image_path, image_bytes),
                "oracle_mask": _artifact(oracle_path, oracle_bytes),
            }
        )
        serialized_corruptions.append(
            BundleCorruption(
                arm=entry.arm,
                view_index=entry.view_index,
                corruption=ControlledCorruption(
                    image=entry.corruption.image,
                    oracle_mask=saved_oracle,
                    kind=entry.corruption.kind,
                    parameters=entry.corruption.parameters,
                ),
            )
        )

    contact_bytes = _png_bytes(
        _render_contact_sheet(views, foreground_masks, serialized_corruptions)
    )
    files["contact_sheet.png"] = contact_bytes
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "completed",
        "failure_reason": None,
        "K": len(views),
        "seed": seed,
        "mesh_scale": scale,
        "views": view_records,
        "corruptions": corruption_records,
        "contact_sheet": _artifact("contact_sheet.png", contact_bytes),
    }
    _canonical_json(manifest)
    return manifest, files


def _failure_payload(
    run_id: str,
    *,
    seed: int,
    mesh_scale: float,
    num_views: int,
    failure_reason: str,
) -> tuple[dict, dict[str, bytes]]:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    try:
        scale = float(mesh_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("mesh_scale must be finite and positive") from exc
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("mesh_scale must be finite and positive")
    if (
        not isinstance(num_views, int)
        or isinstance(num_views, bool)
        or num_views <= 0
    ):
        raise ValueError("num_views must be a positive integer")
    if not isinstance(failure_reason, str) or not failure_reason.strip():
        raise ValueError("failure_reason must be nonempty")
    contact_bytes = _png_bytes(_render_failure_contact_sheet(failure_reason))
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "failed",
        "failure_reason": failure_reason,
        "K": num_views,
        "seed": seed,
        "mesh_scale": scale,
        "views": [],
        "corruptions": [],
        "contact_sheet": _artifact("contact_sheet.png", contact_bytes),
    }
    _canonical_json(manifest)
    return manifest, {"contact_sheet.png": contact_bytes}


def _artifact_records(manifest: dict) -> Iterable[dict]:
    for view in manifest.get("views", []):
        yield view["source"]
        yield view["foreground_mask"]
    for corruption in manifest.get("corruptions", []):
        yield corruption["image"]
        yield corruption["oracle_mask"]
    yield manifest["contact_sheet"]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _is_artifact_record(
    value: object, *, path: str, extra_keys: frozenset[str] = frozenset()
) -> bool:
    required = {"path", "sha256", *extra_keys}
    return (
        isinstance(value, dict)
        and set(value) == required
        and value.get("path") == path
        and _is_sha256(value.get("sha256"))
    )


def _is_transform(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(
            isinstance(row, list)
            and len(row) == 4
            and all(_is_finite_number(component) for component in row)
            for row in value
        )
    )


def _completed_contract_is_valid(manifest: dict) -> bool:
    views = manifest["views"]
    if (
        manifest["failure_reason"] is not None
        or not isinstance(views, list)
        or manifest["K"] != len(views)
        or not views
    ):
        return False
    view_keys = {
        "view_index",
        "frame_index",
        "camera_angle_x",
        "distance",
        "transform_matrix",
        "source",
        "foreground_mask",
    }
    for index, view in enumerate(views):
        source_path = f"views/view_{index:02d}/source.png"
        foreground_path = f"views/view_{index:02d}/foreground_mask.png"
        if not isinstance(view, dict) or set(view) != view_keys:
            return False
        if (
            not _is_int(view["view_index"])
            or view["view_index"] != index
            or not _is_int(view["frame_index"])
            or view["frame_index"] < 0
            or not _is_finite_number(view["camera_angle_x"])
            or not _is_finite_number(view["distance"])
            or view["distance"] < 0
            or not _is_transform(view["transform_matrix"])
            or not _is_artifact_record(
                view["source"],
                path=source_path,
                extra_keys=frozenset({"input_sha256"}),
            )
            or not _is_sha256(view["source"].get("input_sha256"))
            or not _is_artifact_record(
                view["foreground_mask"],
                path=foreground_path,
                extra_keys=frozenset({"provenance", "source_sha256"}),
            )
            or view["foreground_mask"].get("provenance")
            not in {"explicit", "alpha", "rembg"}
            or not _is_sha256(
                view["foreground_mask"].get("source_sha256")
            )
        ):
            return False

    corruptions = manifest["corruptions"]
    if not isinstance(corruptions, list) or len(corruptions) != 3:
        return False
    corruption_keys = {
        "arm",
        "view_index",
        "kind",
        "parameters",
        "image",
        "oracle_mask",
    }
    for arm, corruption in zip(("c1", "c2", "c3"), corruptions):
        if not isinstance(corruption, dict) or set(corruption) != corruption_keys:
            return False
        view_index = corruption["view_index"]
        if not _is_int(view_index) or not 0 <= view_index < len(views):
            return False
        root = f"corruptions/{arm}/view_{view_index:02d}"
        if (
            corruption["arm"] != arm
            or corruption["kind"] != _EXPECTED_KINDS[arm]
            or not isinstance(corruption["parameters"], dict)
            or not _is_artifact_record(
                corruption["image"], path=f"{root}/image.png"
            )
            or not _is_artifact_record(
                corruption["oracle_mask"], path=f"{root}/oracle_mask.png"
            )
        ):
            return False
    return True


def _failed_contract_is_valid(manifest: dict) -> bool:
    return (
        isinstance(manifest["failure_reason"], str)
        and bool(manifest["failure_reason"].strip())
        and manifest["views"] == []
        and manifest["corruptions"] == []
    )


def _validate_manifest_contract(manifest: dict, *, expected_run_id: str) -> None:
    required_keys = {
        "schema_version",
        "run_id",
        "status",
        "failure_reason",
        "K",
        "seed",
        "mesh_scale",
        "views",
        "corruptions",
        "contact_sheet",
    }
    status = manifest.get("status")
    common_valid = (
        set(manifest) == required_keys
        and _is_int(manifest.get("schema_version"))
        and manifest.get("schema_version") == 1
        and isinstance(manifest.get("run_id"), str)
        and _RUN_ID.fullmatch(manifest["run_id"]) is not None
        and manifest["run_id"] == expected_run_id
        and status in {"completed", "failed"}
        and _is_int(manifest.get("K"))
        and manifest["K"] > 0
        and _is_int(manifest.get("seed"))
        and _is_finite_number(manifest.get("mesh_scale"))
        and manifest["mesh_scale"] > 0
        and _is_artifact_record(
            manifest.get("contact_sheet"), path="contact_sheet.png"
        )
    )
    if not common_valid:
        label = status if status in {"completed", "failed"} else "bundle"
        raise ValueError(f"artifact bundle has invalid {label} bundle contract")
    if status == "completed" and not _completed_contract_is_valid(manifest):
        raise ValueError("artifact bundle has invalid completed bundle contract")
    if status == "failed" and not _failed_contract_is_valid(manifest):
        raise ValueError("artifact bundle has invalid failed bundle contract")


def _validate_artifact_bundle(run_dir: Path, *, expected_run_id: str) -> dict:
    unresolved_root = Path(run_dir).absolute()
    if unresolved_root.is_symlink() or not unresolved_root.is_dir():
        raise ValueError("artifact bundle must be a non-symlink directory")
    root = unresolved_root.resolve()
    manifest_path = unresolved_root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(
            "artifact bundle manifest must be a non-symlink regular file"
        )
    if not manifest_path.resolve().is_relative_to(root):
        raise ValueError(
            "artifact bundle manifest must be a non-symlink regular file"
        )

    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("artifact bundle has no readable manifest") from exc
    if not isinstance(manifest, dict):
        raise ValueError("artifact bundle manifest has invalid status")
    if raw != _canonical_json(manifest):
        raise ValueError("artifact bundle manifest is not canonical JSON")
    _validate_manifest_contract(manifest, expected_run_id=expected_run_id)
    records = list(_artifact_records(manifest))
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("artifact bundle manifest has invalid artifact records")
        relative_value = record.get("path")
        expected_hash = record.get("sha256")
        if not isinstance(relative_value, str) or not isinstance(expected_hash, str):
            raise ValueError("artifact bundle manifest has invalid artifact records")
        relative = PurePosixPath(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact path must be relative to its bundle")
        path = (root / Path(*relative.parts)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("artifact path must name a file inside its bundle")
        actual_hash = sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"artifact hash mismatch: {relative_value}")
    return manifest


def validate_artifact_bundle(run_dir: Path) -> dict:
    """Validate the bundle schema and every artifact it references."""

    path = Path(run_dir)
    return _validate_artifact_bundle(path, expected_run_id=path.name)


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id) or run_id in {
        ".",
        "..",
    }:
        raise ValueError("run_id must be a safe nonempty identifier")


def _publish_bundle(
    output_dir: Path, run_id: str, manifest: dict, files: dict[str, bytes]
) -> Path:
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or output.is_symlink()):
        raise ValueError("output_dir must be a real directory")
    output.mkdir(parents=True, exist_ok=True)
    target = output / run_id
    if target.is_symlink():
        raise ValueError("run directory must not be a symlink")
    if target.exists():
        existing = validate_artifact_bundle(target)
        if existing != manifest:
            raise ValueError("existing run hash mismatch with requested bundle")
        return target

    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=output))
    try:
        for relative, payload in files.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        (staging / "manifest.json").write_bytes(_canonical_json(manifest))
        _validate_artifact_bundle(staging, expected_run_id=run_id)
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


def write_artifact_bundle(
    output_dir: Path,
    run_id: str,
    views: Sequence[CalibratedView],
    foreground_masks: Sequence[ForegroundMask],
    corruptions: Sequence[BundleCorruption],
    *,
    seed: int,
    mesh_scale: float,
) -> Path:
    """Create or validate a complete controlled-corruption run bundle."""

    _validate_run_id(run_id)
    manifest, files = _request_payload(
        run_id,
        views,
        foreground_masks,
        corruptions,
        seed=seed,
        mesh_scale=mesh_scale,
    )
    return _publish_bundle(output_dir, run_id, manifest, files)


def write_failed_artifact_bundle(
    output_dir: Path,
    run_id: str,
    *,
    seed: int,
    mesh_scale: float,
    num_views: int,
    failure_reason: str,
) -> Path:
    """Create or validate a failed run manifest and its diagnostic sheet."""

    _validate_run_id(run_id)
    manifest, files = _failure_payload(
        run_id,
        seed=seed,
        mesh_scale=mesh_scale,
        num_views=num_views,
        failure_reason=failure_reason,
    )
    return _publish_bundle(output_dir, run_id, manifest, files)
