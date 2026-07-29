"""Compatibility CLI for production multiview materialization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_toolkit.pipeline import training_materialization as _core
from data_toolkit.pipeline.training_materialization import (
    ABO_SOURCE_SPEC,
    DEFAULT_INDEX,
    DEFAULT_OUTPUT,
    DEFAULT_PREPARED,
    EXPECTED_BATCHES,
    EXPECTED_CANDIDATE_COUNTS,
    EXPECTED_STAGE_COUNTS,
    EXPECTED_WAIVER_COUNTS,
    FamilyPack,
    ProductionSourceSpec,
    SHARD_ID,
    SOURCE,
    STAGE_FAMILIES,
    THREED_FUTURE_OUTPUT,
    THREED_FUTURE_SOURCE_SPEC,
    compute_stage_scopes,
    load_production_catalog,
    load_source_catalog,
    build_source_spec,
    source_output_root,
)
from data_toolkit.pipeline.training_source_profiles import (
    SOURCE_PROFILE_NAMES,
)


# Preserve private helper imports used by the hardened legacy tests.
_expected_member_paths = _core._expected_member_paths
_publish_no_replace = _core._publish_no_replace


def __getattr__(name: str):
    """Proxy historical helpers and constants to the extracted core."""
    return getattr(_core, name)


def _legacy_spec(
    index_path: Path, catalog: Mapping[str, Sequence[FamilyPack]]
) -> ProductionSourceSpec:
    records = tuple(catalog.get("common", ()))
    shard_ids = tuple(dict.fromkeys(record.shard_id for record in records))
    source_ids = tuple(dict.fromkeys(record.source for record in records))
    if len(shard_ids) != 1 or len(source_ids) != 1:
        raise ValueError("legacy materialization requires one source shard")
    batches = tuple(
        dict.fromkeys(record.batch_id for record in records)
    )
    frozen = {
        asset for record in records for asset in record.frozen_assets
    }
    return ProductionSourceSpec(
        source=source_ids[0],
        indexes=(Path(index_path),),
        expected_batches={shard_ids[0]: batches},
        expected_frozen=len(frozen),
        expected_candidate_stages=EXPECTED_CANDIDATE_COUNTS,
        fixed_count_contract=ABO_SOURCE_SPEC.fixed_count_contract,
        acceptance_mode=ABO_SOURCE_SPEC.acceptance_mode,
        original_90_percent_gate_passed=(
            ABO_SOURCE_SPEC.original_90_percent_gate_passed
        ),
    )


def materialize_stage(
    stage: str,
    catalog: Mapping[str, Sequence[FamilyPack]],
    output_root: Path,
    *,
    index_path: Path,
    expected_counts: Mapping[str, int] = EXPECTED_CANDIDATE_COUNTS,
    expected_waiver: Mapping[str, int] = EXPECTED_WAIVER_COUNTS,
    expected_stage_counts: Mapping[str, int] | None = None,
    expected_training_exclusion_counts: Mapping[str, int] | None = None,
) -> Path:
    """Call the source-aware core through the historical ABO signature."""
    original_publish = _core._publish_no_replace
    _core._publish_no_replace = _publish_no_replace
    try:
        return _core._materialize_stage(
            stage,
            catalog,
            output_root,
            index_path=Path(index_path),
            spec=_legacy_spec(Path(index_path), catalog),
            expected_counts=expected_counts,
            expected_waiver=expected_waiver,
            expected_stage_counts=expected_stage_counts,
            expected_training_exclusion_counts=(
                expected_training_exclusion_counts
            ),
        )
    finally:
        _core._publish_no_replace = original_publish


def materialize_all(
    index_path: Path,
    prepared_root: Path,
    output_root: Path,
    *,
    expected_batches: Sequence[str] = EXPECTED_BATCHES,
    expected_counts: Mapping[str, int] = EXPECTED_CANDIDATE_COUNTS,
) -> dict[str, Path]:
    """Preserve the historical single-index ABO Python API."""
    catalog = load_production_catalog(
        index_path,
        prepared_root,
        SOURCE,
        SHARD_ID,
        expected_batches=expected_batches,
    )
    return {
        stage: materialize_stage(
            stage,
            catalog,
            output_root,
            index_path=index_path,
            expected_counts=expected_counts,
        )
        for stage in STAGE_FAMILIES
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=SOURCE_PROFILE_NAMES, default="abo"
    )
    parser.add_argument("--data2-root", type=Path, default=None)
    parser.add_argument("--local-root", type=Path, default=None)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--stage", choices=tuple(STAGE_FAMILIES), action="append"
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _argument_parser()
    return parser.parse_args(argv)


def resolve_profile_paths(
    args: argparse.Namespace,
) -> tuple[ProductionSourceSpec, Path, Path]:
    """Build source inputs and local output paths from optional node roots."""
    data2_root = args.data2_root or Path("/root/data2/pixal3d")
    local_root = args.local_root or Path("/root/node17/data/pixal3d")
    spec = build_source_spec(args.profile, data2_root)
    prepared = data2_root / "prepared"
    output = source_output_root(args.profile, local_root)
    return spec, prepared, output


def _legacy_paths(args: argparse.Namespace) -> tuple[ProductionSourceSpec, Path, Path]:
    """Preserve historical ABO and 3D-FUTURE path selections exactly."""
    if args.profile == "abo":
        spec = ABO_SOURCE_SPEC
        profile_output = DEFAULT_OUTPUT
    else:
        spec = THREED_FUTURE_SOURCE_SPEC
        profile_output = THREED_FUTURE_OUTPUT
    prepared_root = args.prepared_root or DEFAULT_PREPARED
    output_root = args.output_root or profile_output
    if prepared_root != DEFAULT_PREPARED:
        raise ValueError("--prepared-root must match the selected profile")
    if output_root != profile_output:
        raise ValueError("--output-root must match the selected profile")
    if args.index is not None and spec.indexes != (args.index,):
        raise ValueError("--index must match the selected profile")
    return spec, prepared_root, output_root


def main() -> None:
    parser = _argument_parser()
    args = parser.parse_args()
    root_aware = args.data2_root is not None or args.local_root is not None
    legacy_paths = (args.index, args.prepared_root, args.output_root)
    if root_aware:
        if any(path is not None for path in legacy_paths):
            parser.error(
                "root-aware profile selection cannot be combined with legacy paths"
            )
        spec, prepared_root, output_root = resolve_profile_paths(args)
    elif args.profile == "hssd":
        if args.index is not None:
            parser.error(
                "hssd profile binds both indexes from its source spec"
            )
        if args.prepared_root is not None or args.output_root is not None:
            parser.error(
                "hssd profile requires profile-derived prepared and output roots"
            )
        spec, prepared_root, output_root = resolve_profile_paths(args)
    else:
        try:
            spec, prepared_root, output_root = _legacy_paths(args)
        except ValueError as error:
            parser.error(str(error))

    catalog = load_source_catalog(spec, prepared_root)
    for stage in args.stage or tuple(STAGE_FAMILIES):
        print(_core.materialize_stage(spec, stage, catalog, output_root))


if __name__ == "__main__":
    main()
