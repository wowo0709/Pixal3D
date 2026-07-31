"""Deterministic, CPU-only eligibility checks for extracted training stages."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


TOKEN_LIMITS = {"shape512": 8192, "shape1024": 32768, "pbr1024": 32768}
SCALE_RTOL = 0.0
SCALE_ATOL = 2e-7
EXPECTED_FROZEN_COUNT = 4485
EXPECTED_GLOBAL_QUARANTINE_COUNT = 825
EXPECTED_SHAPE512_FAMILY_EXCLUSION_COUNT = 29
EXPECTED_CANDIDATE_STAGE_COUNTS = {
    "ss64": 3660,
    "shape512": 3631,
    "shape1024": 3660,
    "pbr1024": 3660,
}
EXPECTED_TRAINING_EXCLUSION_COUNTS = {
    "ss64": 0,
    "shape512": 3,
    "shape1024": 26,
    "pbr1024": 62,
}
EXPECTED_FINAL_STAGE_COUNTS = {
    "ss64": 3660,
    "shape512": 3628,
    "shape1024": 3634,
    "pbr1024": 3598,
}
STAGES = ("ss64", "shape512", "shape1024", "pbr1024")

_SHAPE_ROOTS = {
    "shape512": "shape_latents/shape_enc_next_dc_f16c32_fp16_512_view",
    "shape1024": "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view",
    "pbr1024": "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view",
}
_PBR_ROOT = "pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix"


@dataclass(frozen=True)
class EligibilityExclusion:
    asset: str
    reasons: tuple[str, ...]


def canonical_count_contract(
    *,
    frozen: int = EXPECTED_FROZEN_COUNT,
    global_quarantine: int = EXPECTED_GLOBAL_QUARANTINE_COUNT,
    shape512_family_exclusions: int = EXPECTED_SHAPE512_FAMILY_EXCLUSION_COUNT,
    candidate_stages: Mapping[str, int] = EXPECTED_CANDIDATE_STAGE_COUNTS,
    training_exclusions: Mapping[str, int] = EXPECTED_TRAINING_EXCLUSION_COUNTS,
    stages: Mapping[str, int] = EXPECTED_FINAL_STAGE_COUNTS,
) -> dict[str, object]:
    """Return the one canonical count shape consumed across publication boundaries."""
    return {
        "frozen": frozen,
        "global_quarantine": global_quarantine,
        "shape512_family_exclusions": shape512_family_exclusions,
        "candidate_stages": dict(candidate_stages),
        "training_exclusions": dict(training_exclusions),
        "stages": dict(stages),
    }


def _stage_counts(values: Mapping[str, int], label: str) -> dict[str, int]:
    result = dict(values)
    if set(result) != set(STAGES):
        raise ValueError(f"{label} must contain exactly {STAGES}")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in result.values()
    ):
        raise ValueError(f"{label} must contain non-negative integers")
    return {stage: result[stage] for stage in STAGES}


def observed_count_contract(
    *,
    frozen: int,
    candidate_stages: Mapping[str, int],
    training_exclusions: Mapping[str, int],
) -> dict[str, object]:
    """Derive the source-independent count contract from observed stage populations."""
    candidates = _stage_counts(candidate_stages, "candidate_stages")
    exclusions = _stage_counts(training_exclusions, "training_exclusions")
    if isinstance(frozen, bool) or not isinstance(frozen, int) or frozen < 0:
        raise ValueError("frozen must be a non-negative integer")
    if any(candidates[stage] > frozen for stage in STAGES):
        raise ValueError("candidate count exceeds frozen count")
    if any(exclusions[stage] > candidates[stage] for stage in STAGES):
        raise ValueError("training exclusion exceeds candidate count")
    return {
        "frozen": frozen,
        "candidate_stages": candidates,
        "pack_exclusions": {
            stage: frozen - candidates[stage] for stage in STAGES
        },
        "training_exclusions": exclusions,
        "stages": {
            stage: candidates[stage] - exclusions[stage] for stage in STAGES
        },
    }


ABO_COUNT_CONTRACT = {
    "global_quarantine": EXPECTED_GLOBAL_QUARANTINE_COUNT,
    "shape512_family_exclusions": EXPECTED_SHAPE512_FAMILY_EXCLUSION_COUNT,
    **observed_count_contract(
        frozen=EXPECTED_FROZEN_COUNT,
        candidate_stages=EXPECTED_CANDIDATE_STAGE_COUNTS,
        training_exclusions=EXPECTED_TRAINING_EXCLUSION_COUNTS,
    ),
}


def policy_evidence() -> dict[str, object]:
    """Return the canonical policy values persisted with each materialization."""
    return {
        "schema_version": 1,
        "token_limits": dict(TOKEN_LIMITS),
        "pbr_shape_coordinates": "exact",
        "pbr_shape_scale": {
            "dtype": "float32",
            "rtol": SCALE_RTOL,
            "atol": SCALE_ATOL,
        },
    }


def _asset_path(root: Path, component: str, asset: str, filename: str) -> Path:
    path = root / component / asset / filename
    try:
        resolved = path.resolve(strict=True)
        component_root = (root / component).resolve(strict=True)
    except OSError as error:
        raise ValueError(f"missing eligibility input for {asset}: {path}") from error
    if not resolved.is_relative_to(component_root) or not resolved.is_file():
        raise ValueError(f"invalid eligibility input for {asset}: {path}")
    return resolved


def _load_latent(root: Path, component: str, asset: str, anchor: int) -> tuple[np.ndarray, np.ndarray]:
    path = _asset_path(root, component, asset, f"view{anchor:02d}.npz")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"coords", "feats"}:
                raise ValueError("NPZ keys must be exactly coords and feats")
            coords = archive["coords"]
            feats = archive["feats"]
    except (OSError, ValueError, EOFError, KeyError) as error:
        raise ValueError(f"invalid latent NPZ for {asset} view{anchor:02d}: {path}: {error}") from error
    if (
        not isinstance(coords, np.ndarray)
        or not isinstance(feats, np.ndarray)
        or coords.ndim != 2
        or feats.ndim != 2
        or coords.shape[1:] != (3,)
        or feats.shape[1:] != (32,)
        or coords.shape[0] != feats.shape[0]
    ):
        raise ValueError(f"invalid latent shapes for {asset} view{anchor:02d}: {path}")
    return coords, feats


def _load_scale(root: Path, component: str, asset: str, anchor: int) -> np.float32:
    path = _asset_path(root, component, asset, f"view{anchor:02d}_scale.json")
    try:
        document = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid scale for {asset} view{anchor:02d}: {path}: {error}") from error
    if not isinstance(document, dict) or "total_scale" not in document:
        raise ValueError(f"invalid scale for {asset} view{anchor:02d}: {path}")
    value = document["total_scale"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid scale for {asset} view{anchor:02d}: {path}")
    with np.errstate(over="ignore", invalid="ignore"):
        scale = np.float32(value)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid scale for {asset} view{anchor:02d}: {path}")
    return scale


def evaluate_stage_asset(stage: str, root: Path, asset: str) -> tuple[str, ...]:
    """Return deterministic exclusion reasons for one extracted stage asset."""
    if stage == "ss64":
        return ()
    if stage not in TOKEN_LIMITS:
        raise ValueError(f"unknown eligibility stage: {stage}")
    root = Path(root)
    shape_component = _SHAPE_ROOTS[stage]
    limit = TOKEN_LIMITS[stage]
    reasons: list[str] = []
    shape_latents = []
    for anchor in range(2):
        coords, feats = _load_latent(root, shape_component, asset, anchor)
        shape_latents.append((coords, feats))
        if coords.shape[0] > limit:
            reasons.append(f"shape_tokens_view{anchor:02d}_exceed_{limit}")
    if stage != "pbr1024":
        return tuple(sorted(set(reasons)))
    for anchor in range(2):
        shape_coords, _ = shape_latents[anchor]
        pbr_coords, _ = _load_latent(root, _PBR_ROOT, asset, anchor)
        if pbr_coords.shape[0] > limit:
            reasons.append(f"pbr_tokens_view{anchor:02d}_exceed_{limit}")
        if not np.array_equal(shape_coords, pbr_coords):
            reasons.append(f"pbr_shape_coords_view{anchor:02d}_mismatch")
        shape_scale = _load_scale(root, shape_component, asset, anchor)
        pbr_scale = _load_scale(root, _PBR_ROOT, asset, anchor)
        if not np.isclose(shape_scale, pbr_scale, rtol=SCALE_RTOL, atol=SCALE_ATOL):
            reasons.append(f"pbr_shape_scale_view{anchor:02d}_mismatch")
    return tuple(sorted(set(reasons)))


def filter_stage_scope(
    stage: str, root: Path, candidates: Sequence[str]
) -> tuple[tuple[str, ...], tuple[EligibilityExclusion, ...]]:
    """Split candidates into a canonical final scope and sorted exclusions."""
    final: list[str] = []
    exclusions: list[EligibilityExclusion] = []
    for asset in sorted(set(candidates)):
        reasons = evaluate_stage_asset(stage, root, asset)
        if reasons:
            exclusions.append(EligibilityExclusion(asset, reasons))
        else:
            final.append(asset)
    return tuple(final), tuple(exclusions)
