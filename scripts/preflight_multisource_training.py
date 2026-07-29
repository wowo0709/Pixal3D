#!/usr/bin/env python3
"""CPU-only Dataset/DataLoader preflight for combined training inputs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Mapping
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_toolkit.pipeline.training_manifest import (  # noqa: E402
    KNOWN_SOURCES,
    STAGES,
    ResolvedTrainingData,
    resolve_training_data,
)


CONFIGS = {
    "ss64": Path(
        "configs/gen/"
        "ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json"
    ),
    "shape512": Path(
        "configs/gen/"
        "slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json"
    ),
    "shape1024": Path(
        "configs/gen/"
        "slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json"
    ),
    "pbr1024": Path(
        "configs/gen/"
        "slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json"
    ),
}


def _assert_cpu_only() -> None:
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError("CPU-only preflight initialized CUDA")


def _dataset_config(
    config_path: Path, expected_stage: str
) -> tuple[str, dict[str, object]]:
    try:
        config = json.loads(Path(config_path).read_text())
        dataset = config["dataset"]
        name = dataset["name"]
        args = dataset["args"]
        stage = config["trainer"]["args"]["multiview_stage"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"invalid training config: {config_path}") from error
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(args, dict)
        or stage != expected_stage
    ):
        raise ValueError(
            f"training config does not select stage={expected_stage}: "
            f"{config_path}"
        )
    return name, args


def _construct_configured_dataset(
    resolved: ResolvedTrainingData, config_path: Path
):
    name, args = _dataset_config(config_path, resolved.stage)
    _assert_cpu_only()
    try:
        # flex_gemm chooses import-time Triton kernels by querying a device
        # name. The patch is restricted to importing the dataset definition;
        # this script never initializes CUDA or executes those kernels.
        with patch.object(torch.cuda, "get_device_name", return_value="A100"):
            from pixal3d import datasets

            dataset_class = getattr(datasets, name)
    except Exception as error:
        raise RuntimeError(
            f"stage={resolved.stage}: failed to import configured "
            f"dataset {name}"
        ) from error
    _assert_cpu_only()
    try:
        dataset = dataset_class(json.dumps(resolved.data_dir), **args)
    except Exception as error:
        raise RuntimeError(
            f"stage={resolved.stage}: failed to construct configured "
            f"dataset {name}"
        ) from error
    _assert_cpu_only()
    return dataset


def _expected_instances(
    resolved: ResolvedTrainingData,
    source_order: tuple[str, ...],
) -> list[tuple[dict[str, str], str, str]]:
    return [
        (resolved.data_dir[source], asset, source)
        for source in source_order
        for asset in resolved.source_scopes[source]
    ]


def _validate_instances(
    dataset,
    resolved: ResolvedTrainingData,
    source_order: tuple[str, ...],
) -> None:
    expected = _expected_instances(resolved, source_order)
    actual = list(dataset.instances)
    if any(
        not isinstance(instance, tuple) or len(instance) != 3
        for instance in actual
    ):
        raise ValueError(
            f"stage={resolved.stage}: configured dataset instances are "
            "malformed"
        )
    unexpected_sources = sorted(
        {
            source
            for _root, _asset, source in actual
            if source not in source_order
        }
    )
    if unexpected_sources:
        raise ValueError(
            f"stage={resolved.stage}: unexpected source in configured "
            f"dataset: {unexpected_sources}"
        )
    actual_keys = [(source, asset) for _root, asset, source in actual]
    duplicates = sorted(
        key for key, count in Counter(actual_keys).items() if count != 1
    )
    if duplicates:
        raise ValueError(
            f"stage={resolved.stage}: duplicate source/SHA instances: "
            f"{duplicates}"
        )
    expected_keys = {(source, asset) for _root, asset, source in expected}
    actual_key_set = set(actual_keys)
    omitted = sorted(expected_keys - actual_key_set)
    if omitted:
        raise ValueError(
            f"stage={resolved.stage}: materialized instances omitted by "
            f"loader filtering: {omitted}"
        )
    unexpected = sorted(actual_key_set - expected_keys)
    if unexpected:
        raise ValueError(
            f"stage={resolved.stage}: unexpected configured dataset "
            f"instances: {unexpected}"
        )
    expected_roots = {
        (source, asset): root for root, asset, source in expected
    }
    wrong_roots = sorted(
        (source, asset)
        for root, asset, source in actual
        if root != expected_roots[(source, asset)]
    )
    if wrong_roots:
        raise ValueError(
            f"stage={resolved.stage}: configured dataset instance roots "
            f"do not match resolved data_dir: {wrong_roots}"
        )
    if actual != expected:
        raise ValueError(
            f"stage={resolved.stage}: configured dataset instances do not "
            "match canonical order"
        )
    observed_counts = Counter(source for _root, _asset, source in actual)
    if dict(observed_counts) != resolved.source_counts:
        raise ValueError(
            f"stage={resolved.stage}: configured dataset source counts "
            "do not match the manifest"
        )
    if len(actual) != resolved.total_count:
        raise ValueError(
            f"stage={resolved.stage}: configured dataset total does not "
            "match the manifest"
        )


def _boundary_samples(
    dataset,
    resolved: ResolvedTrainingData,
    source_order: tuple[str, ...],
) -> tuple[list[dict[str, object]], list[str], int]:
    by_key = {
        (source, asset): root
        for root, asset, source in dataset.instances
    }
    collate_samples = []
    collated_sources = []
    checked = 0
    for source in source_order:
        scope = resolved.source_scopes[source]
        boundary_assets = tuple(dict.fromkeys((scope[0], scope[-1])))
        for asset in boundary_assets:
            dataset._current_dataset_name = source
            _assert_cpu_only()
            try:
                with patch.object(
                    np.random, "randint", side_effect=lambda low, high: low
                ):
                    sample = dataset.get_instance(
                        by_key[(source, asset)], asset
                    )
            except Exception as error:
                raise RuntimeError(
                    f"source={source} stage={resolved.stage} asset={asset} "
                    "anchor=view00: direct dataset load failed"
                ) from error
            _assert_cpu_only()
            checked += 1
            if asset == scope[0]:
                collate_samples.append(sample)
                collated_sources.append(source)
    return collate_samples, collated_sources, checked


def _collate_cross_source(dataset, samples: list[dict[str, object]]) -> None:
    _assert_cpu_only()
    loader = DataLoader(
        samples,
        batch_size=len(samples),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=dataset.collate_fn,
    )
    try:
        with patch.object(
            np.random, "randint", side_effect=lambda low, high: low
        ):
            batch = next(iter(loader))
    except Exception as error:
        raise RuntimeError("cross-source collate failed") from error
    _assert_cpu_only()
    if not isinstance(batch, Mapping) or not batch:
        raise RuntimeError(
            "cross-source collate failed: collate_fn returned no batch"
        )


def preflight_multisource_stage(
    training_data: Path, stage: str, config: Path
) -> dict[str, object]:
    """Verify one combined stage through its real Dataset and collate_fn."""
    _assert_cpu_only()
    resolved = resolve_training_data(Path(training_data), stage)
    source_order = tuple(resolved.source_scopes)
    unknown_sources = sorted(
        source for source in source_order if source not in KNOWN_SOURCES
    )
    if unknown_sources:
        raise ValueError(
            f"stage={stage}: unknown resolved source: {unknown_sources}"
        )
    dataset = _construct_configured_dataset(resolved, Path(config))
    _assert_cpu_only()
    _validate_instances(dataset, resolved, source_order)
    _assert_cpu_only()
    samples, collated_sources, checked = _boundary_samples(
        dataset, resolved, source_order
    )
    _assert_cpu_only()
    _collate_cross_source(dataset, samples)
    _assert_cpu_only()
    return {
        "stage": stage,
        "source_counts": resolved.source_counts,
        "total_count": resolved.total_count,
        "sampling": resolved.sampling,
        "boundary_instances_checked": checked,
        "collated_sources": collated_sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only Dataset/DataLoader preflight for verified one-, "
            "two-, or three-source training data."
        )
    )
    parser.add_argument("--training-data", type=Path, required=True)
    args = parser.parse_args()
    results = {
        stage: preflight_multisource_stage(
            args.training_data, stage, CONFIGS[stage]
        )
        for stage in STAGES
    }
    print(json.dumps({"stages": results}, indent=2))


if __name__ == "__main__":
    main()
