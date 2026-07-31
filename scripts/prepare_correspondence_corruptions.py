#!/usr/bin/env python3
"""Prepare deterministic CPU-only controlled-corruption artifacts."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixal3d.experiments.correspondence import (  # noqa: E402
    BundleCorruption,
    corrupt_local_color,
    corrupt_local_deletion,
    corrupt_procedural_pattern,
    load_calibrated_views,
    resolve_foreground_mask,
    sample_foreground_region,
    write_artifact_bundle,
    write_failed_artifact_bundle,
)


class _UsageError(ValueError):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _UsageError(message)


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _torch_seed(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not -(2**63) <= number <= 2**64 - 1:
        raise argparse.ArgumentTypeError(
            "must be in the inclusive PyTorch seed range "
            "[-9223372036854775808, 18446744073709551615]"
        )
    return number


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description="Prepare CPU-only C1-C3 controlled corruptions.",
        allow_abbrev=False,
    )
    parser.add_argument("--transforms", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh-scale", type=_positive_float, required=True)
    parser.add_argument("--num-views", type=_positive_int, default=4)
    parser.add_argument("--corrupt-view-index", type=int, default=1)
    parser.add_argument("--seed", type=_torch_seed, default=42)
    return parser


def _explicit_output_dir(arguments: list[str]) -> Path | None:
    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
        exit_on_error=False,
    )
    parser.add_argument("--output-dir", type=Path)
    try:
        known, _ = parser.parse_known_args(arguments)
    except argparse.ArgumentError:
        return None
    return known.output_dir


def _run_id(*, num_views: int, corrupt_view_index: int, seed: int) -> str:
    return f"controlled-corruption-k{num_views}-v{corrupt_view_index}-s{seed}"


def _failed_run_id(run_id: str, failure_reason: str) -> str:
    reason_digest = sha256(failure_reason.encode("utf-8")).hexdigest()[:12]
    return f"{run_id}-failed-{reason_digest}"


def _publish_usage_failure(arguments: list[str], failure_reason: str) -> None:
    output_dir = _explicit_output_dir(arguments)
    if output_dir is None:
        return
    run_id = _failed_run_id(
        "controlled-corruption-invalid-invocation", failure_reason
    )
    try:
        write_failed_artifact_bundle(
            output_dir,
            run_id,
            seed=42,
            mesh_scale=1.0,
            num_views=4,
            failure_reason=failure_reason,
        )
    except (OSError, TypeError, ValueError) as publication_error:
        print(
            f"error: failed to publish failure bundle: {publication_error}",
            file=sys.stderr,
        )


def _reject_model_fallback(_image):
    raise ValueError(
        "foreground mask requires an explicit path or meaningful RGBA alpha; "
        "model-based rembg is disabled in this CPU-only CLI"
    )


def _rgb_tensor(view_image) -> torch.Tensor:
    pixels = np.asarray(view_image.convert("RGB"), dtype=np.uint8).copy()
    return (
        torch.from_numpy(pixels)
        .permute(2, 0, 1)
        .to(dtype=torch.float32)
        .div_(255.0)
    )


def _prepare(arguments: argparse.Namespace) -> Path:
    views, mesh_scale = load_calibrated_views(
        arguments.transforms,
        mesh_scale=arguments.mesh_scale,
        num_views=arguments.num_views,
    )
    metadata = json.loads(arguments.transforms.read_text())
    frames = metadata["frames"][: arguments.num_views]
    foreground_masks = [
        resolve_foreground_mask(
            view,
            frame,
            rembg_provider=_reject_model_fallback,
        )
        for view, frame in zip(views, frames)
    ]

    foreground = foreground_masks[arguments.corrupt_view_index].mask
    image = _rgb_tensor(views[arguments.corrupt_view_index].image)
    region = sample_foreground_region(foreground, seed=arguments.seed)
    corruptions = [
        BundleCorruption(
            "c1",
            arguments.corrupt_view_index,
            corrupt_local_color(
                image,
                foreground,
                region,
                hue=0.18,
                saturation=1.3,
                brightness=0.82,
            ),
        ),
        BundleCorruption(
            "c2",
            arguments.corrupt_view_index,
            corrupt_procedural_pattern(
                image,
                foreground,
                region,
                seed=arguments.seed,
                pattern="sole",
            ),
        ),
        BundleCorruption(
            "c3",
            arguments.corrupt_view_index,
            corrupt_local_deletion(image, foreground, region),
        ),
    ]
    return write_artifact_bundle(
        arguments.output_dir,
        _run_id(
            num_views=arguments.num_views,
            corrupt_view_index=arguments.corrupt_view_index,
            seed=arguments.seed,
        ),
        views,
        foreground_masks,
        corruptions,
        seed=arguments.seed,
        mesh_scale=mesh_scale,
    )


def main() -> int:
    parser = _parser()
    raw_arguments = sys.argv[1:]
    try:
        arguments = parser.parse_args(raw_arguments)
        if not 0 <= arguments.corrupt_view_index < arguments.num_views:
            parser.error("--corrupt-view-index must be in [0, --num-views)")
    except _UsageError as error:
        reason = str(error)
        parser.print_usage(sys.stderr)
        print(f"{parser.prog}: error: {reason}", file=sys.stderr)
        _publish_usage_failure(raw_arguments, reason)
        return 2
    run_id = _run_id(
        num_views=arguments.num_views,
        corrupt_view_index=arguments.corrupt_view_index,
        seed=arguments.seed,
    )
    try:
        run_dir = _prepare(arguments)
    except (OSError, TypeError, ValueError) as error:
        reason = str(error).strip() or error.__class__.__name__
        print(f"error: {reason}", file=sys.stderr)
        try:
            write_failed_artifact_bundle(
                arguments.output_dir,
                _failed_run_id(run_id, reason),
                seed=arguments.seed,
                mesh_scale=arguments.mesh_scale,
                num_views=arguments.num_views,
                failure_reason=reason,
            )
        except (OSError, TypeError, ValueError) as publication_error:
            print(
                f"error: failed to publish failure bundle: {publication_error}",
                file=sys.stderr,
            )
        return 1
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
