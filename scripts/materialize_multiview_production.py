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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=("abo", "3d-future"), default="abo"
    )
    parser.add_argument("--index", type=Path)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--stage", choices=tuple(STAGE_FAMILIES), action="append"
    )
    args = parser.parse_args()

    if args.profile == "abo":
        spec = ABO_SOURCE_SPEC
        profile_output = DEFAULT_OUTPUT
    else:
        spec = THREED_FUTURE_SOURCE_SPEC
        profile_output = THREED_FUTURE_OUTPUT
    prepared_root = args.prepared_root or DEFAULT_PREPARED
    output_root = args.output_root or profile_output
    if prepared_root != DEFAULT_PREPARED:
        parser.error("--prepared-root must match the selected profile")
    if output_root != profile_output:
        parser.error("--output-root must match the selected profile")
    if args.index is not None and spec.indexes != (args.index,):
        parser.error("--index must match the selected profile")

    catalog = load_source_catalog(spec, prepared_root)
    for stage in args.stage or tuple(STAGE_FAMILIES):
        print(_core.materialize_stage(spec, stage, catalog, output_root))


if __name__ == "__main__":
    main()
