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
    production_source_spec_from_indexes,
    validate_production_source_spec,
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


def _materialize_stage_for_fixture(
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
    """Exercise generic historical mechanics in isolated synthetic tests."""
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


def _materialize_all_for_fixture(
    index_path: Path,
    prepared_root: Path,
    output_root: Path,
    *,
    expected_batches: Sequence[str] = EXPECTED_BATCHES,
    expected_counts: Mapping[str, int] = EXPECTED_CANDIDATE_COUNTS,
) -> dict[str, Path]:
    """Exercise the historical multi-stage mechanics in fixture tests."""
    catalog = load_production_catalog(
        index_path,
        prepared_root,
        SOURCE,
        SHARD_ID,
        expected_batches=expected_batches,
    )
    return {
        stage: _materialize_stage_for_fixture(
            stage,
            catalog,
            output_root,
            index_path=index_path,
            expected_counts=expected_counts,
        )
        for stage in STAGE_FAMILIES
    }


def _same_typed_contract(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        return (
            set(actual) == set(expected)
            and all(
                _same_typed_contract(actual[key], expected[key])
                for key in expected
            )
        )
    if isinstance(expected, (tuple, list)):
        return len(actual) == len(expected) and all(
            _same_typed_contract(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _canonical_abo_output_root(output_root: Path) -> Path:
    selected = Path(output_root)
    expected_suffix = Path("train/production/abo").parts
    if (
        not selected.is_absolute()
        or selected != selected.resolve(strict=False)
        or selected.parts[-len(expected_suffix):] != expected_suffix
    ):
        raise ValueError(
            "output root does not identify the canonical production profile: "
            f"{selected}"
        )
    return selected


def _canonical_abo_spec(index_path: Path) -> tuple[ProductionSourceSpec, Path]:
    spec = production_source_spec_from_indexes(
        SOURCE, (Path(index_path),)
    )
    profile, data2_root = validate_production_source_spec(spec)
    if profile != "abo":
        raise ValueError("legacy API requires canonical production profile=abo")
    return spec, data2_root


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
    """Materialize only the canonical root-aware ABO production profile."""
    spec, _data2_root = _canonical_abo_spec(Path(index_path))
    fixed = spec.fixed_count_contract
    assert fixed is not None
    expected_optional = (
        (expected_stage_counts, fixed["stages"], "stage counts"),
        (
            expected_training_exclusion_counts,
            fixed["training_exclusions"],
            "training exclusion counts",
        ),
    )
    if not _same_typed_contract(
        expected_counts, spec.expected_candidate_stages
    ):
        raise ValueError(
            "expected counts do not match canonical production profile"
        )
    expected_fixed_waiver = {
        "frozen_assets": fixed["frozen"],
        "quarantined_assets": fixed["global_quarantine"],
        "shape512_exclusions": fixed["shape512_family_exclusions"],
    }
    if not _same_typed_contract(expected_waiver, expected_fixed_waiver):
        raise ValueError(
            "waiver counts do not match canonical production profile"
        )
    for actual, required, label in expected_optional:
        if actual is not None and not _same_typed_contract(actual, required):
            raise ValueError(
                f"{label} do not match canonical production profile"
            )
    output = _canonical_abo_output_root(output_root)
    return _core.materialize_stage(spec, stage, catalog, output)


def materialize_all(
    index_path: Path,
    prepared_root: Path,
    output_root: Path,
    *,
    expected_batches: Sequence[str] = EXPECTED_BATCHES,
    expected_counts: Mapping[str, int] = EXPECTED_CANDIDATE_COUNTS,
) -> dict[str, Path]:
    """Materialize all stages for the canonical root-aware ABO profile."""
    spec, data2_root = _canonical_abo_spec(Path(index_path))
    prepared = Path(prepared_root)
    if (
        prepared != data2_root / "prepared"
        or prepared != prepared.resolve(strict=False)
    ):
        raise ValueError(
            "prepared root does not match canonical production profile"
        )
    if not _same_typed_contract(
        tuple(expected_batches), spec.expected_batches[SHARD_ID]
    ):
        raise ValueError(
            "expected batches do not match canonical production profile"
        )
    if not _same_typed_contract(
        expected_counts, spec.expected_candidate_stages
    ):
        raise ValueError(
            "expected counts do not match canonical production profile"
        )
    return _core.materialize_all(
        spec, prepared, _canonical_abo_output_root(output_root)
    )


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
    for label, selected in (
        ("data2 root", data2_root),
        ("local root", local_root),
    ):
        if (
            not selected.is_absolute()
            or selected != selected.resolve(strict=False)
        ):
            raise ValueError(
                f"{label} must be an absolute canonical production root"
            )
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
