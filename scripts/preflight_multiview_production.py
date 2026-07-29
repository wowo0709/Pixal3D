"""Compatibility CLI for strict multiview production preflight."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import sys
from typing import Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_toolkit.pipeline import training_preflight as _core  # noqa: E402
from data_toolkit.pipeline.training_materialization import (  # noqa: E402
    ABO_SOURCE_SPEC,
    THREED_FUTURE_SOURCE_SPEC,
)
from data_toolkit.pipeline.training_source_profiles import (  # noqa: E402
    SOURCE_PROFILE_NAMES,
    ProductionSourceSpec,
    build_source_spec,
    source_output_root,
)
from data_toolkit.pipeline.training_manifest import (  # noqa: E402
    validate_source_training_data,
)


SOURCE = _core.SOURCE
RENDER_ROOT = _core.RENDER_ROOT
COMPONENTS = _core.COMPONENTS
CONFIGS = _core.CONFIGS
DEFAULT_ROOT = _core.DEFAULT_ROOT
DEFAULT_INDEX = _core.DEFAULT_INDEX
DEFAULT_REPORT = _core.DEFAULT_REPORT
DEFAULT_HANDOFF = _core.DEFAULT_HANDOFF
DEFAULT_TRAINING_DATA = _core.DEFAULT_TRAINING_DATA
HANDOFF_CANDIDATE_STAGE_COUNTS = _core.HANDOFF_CANDIDATE_STAGE_COUNTS
HANDOFF_TRAINING_EXCLUSION_COUNTS = _core.HANDOFF_TRAINING_EXCLUSION_COUNTS
HANDOFF_STAGE_COUNTS = _core.HANDOFF_STAGE_COUNTS
HANDOFF_FROZEN_COUNT = _core.HANDOFF_FROZEN_COUNT
HANDOFF_GLOBAL_QUARANTINE_COUNT = _core.HANDOFF_GLOBAL_QUARANTINE_COUNT
HANDOFF_SHAPE512_FAMILY_EXCLUSION_COUNT = (
    _core.HANDOFF_SHAPE512_FAMILY_EXCLUSION_COUNT
)
TOKEN_LIMITS = _core.TOKEN_LIMITS
SCALE_RTOL = _core.SCALE_RTOL
SCALE_ATOL = _core.SCALE_ATOL
EXPECTED_CANDIDATE_STAGE_COUNTS = _core.EXPECTED_CANDIDATE_STAGE_COUNTS
EXPECTED_FINAL_STAGE_COUNTS = _core.EXPECTED_FINAL_STAGE_COUNTS
EXPECTED_FROZEN_COUNT = _core.EXPECTED_FROZEN_COUNT
EXPECTED_GLOBAL_QUARANTINE_COUNT = (
    _core.EXPECTED_GLOBAL_QUARANTINE_COUNT
)
EXPECTED_SHAPE512_FAMILY_EXCLUSION_COUNT = (
    _core.EXPECTED_SHAPE512_FAMILY_EXCLUSION_COUNT
)
EXPECTED_TRAINING_EXCLUSION_COUNTS = (
    _core.EXPECTED_TRAINING_EXCLUSION_COUNTS
)
canonical_count_contract = _core.canonical_count_contract
policy_evidence = _core.policy_evidence
StagePreflight = _core.StagePreflight

build_report = _core.build_report
build_handoff = _core.build_handoff
publish_handoff = _core.publish_handoff
publish_source_handoff = _core.publish_source_handoff
source_preflight_stage = _core.preflight_stage
write_create_only_json = _core.write_create_only_json
_existing_regular_bytes = _core._existing_regular_bytes
_error = _core._error
_regular = _core._regular
_metadata_assets = _core._metadata_assets
_validate_render = _core._validate_render
_scale = _core._scale
_numeric_finite = _core._numeric_finite
_latent = _core._latent
_require_tensor = _core._require_tensor
_validate_loader_pack = _core._validate_loader_pack
_scope_digest = _core._scope_digest
_allowed_exclusion_reasons = _core._allowed_exclusion_reasons
_canonical_json_bytes = _core._canonical_json_bytes
_fsync_directory = _core._fsync_directory
_write_atomic_json = _core._write_atomic_json
_load_existing_json = _core._load_existing_json
_validated_handoff_inputs = _core._validated_handoff_inputs
_materialization_evidence = _core._materialization_evidence
_stage_records = _core._stage_records
_materialization_evidence_from_result = (
    _core._materialization_evidence_from_result
)

THREED_FUTURE_ROOT = Path(
    "/root/node17/data/pixal3d/train/production/3d-future"
)
THREED_FUTURE_REPORT = Path(
    "/root/data2/pixal3d/control/reports/gates/3D-FUTURE/"
    "3D-FUTURE-production-training.json"
)
THREED_FUTURE_HANDOFF = Path(
    "/root/data2/pixal3d/control/splits/3D-FUTURE/"
    "3D-FUTURE-production-training-handoff.json"
)
THREED_FUTURE_TRAINING_DATA = THREED_FUTURE_ROOT / "training_data.json"


def _handoff_counts() -> dict[str, object]:
    return _core.canonical_count_contract(
        frozen=HANDOFF_FROZEN_COUNT,
        global_quarantine=HANDOFF_GLOBAL_QUARANTINE_COUNT,
        shape512_family_exclusions=(
            HANDOFF_SHAPE512_FAMILY_EXCLUSION_COUNT
        ),
        candidate_stages=HANDOFF_CANDIDATE_STAGE_COUNTS,
        training_exclusions=HANDOFF_TRAINING_EXCLUSION_COUNTS,
        stages=HANDOFF_STAGE_COUNTS,
    )


def validate_stage_structure(
    stage: str, root: Path, expected_assets: Sequence[str]
) -> dict[str, int]:
    """Legacy ABO signature for source-aware structural validation."""
    return _core.validate_stage_structure(
        SOURCE, stage, root, expected_assets
    )


def stage_data_dir(
    stage: str, root: Path
) -> dict[str, dict[str, str]]:
    """Legacy ABO signature for the configured Dataset data_dir."""
    return _core.stage_data_dir(SOURCE, stage, root)


def validate_direct_loader(
    stage: str,
    root: Path,
    expected_assets: Sequence[str],
    config_path: Path,
) -> int:
    """Legacy ABO signature for direct Dataset validation."""
    return _core.validate_direct_loader(
        SOURCE, stage, root, expected_assets, config_path
    )


def _materialization_scope(
    stage: str, root: Path
) -> tuple[tuple[str, ...], str, bytes]:
    """Legacy ABO materialization validator."""
    return _core._materialization_scope_abo(
        stage,
        root,
        candidate_counts=HANDOFF_CANDIDATE_STAGE_COUNTS,
        training_exclusion_counts=HANDOFF_TRAINING_EXCLUSION_COUNTS,
        stage_counts=HANDOFF_STAGE_COUNTS,
        frozen_count=HANDOFF_FROZEN_COUNT,
        global_quarantine_count=HANDOFF_GLOBAL_QUARANTINE_COUNT,
        shape512_family_exclusion_count=(
            HANDOFF_SHAPE512_FAMILY_EXCLUSION_COUNT
        ),
    )


def _eligibility_evidence_is_valid(
    stage: str, evidence: Mapping[str, object]
) -> bool:
    return _core._eligibility_evidence_is_valid(
        stage,
        evidence,
        candidate_counts=HANDOFF_CANDIDATE_STAGE_COUNTS,
        training_exclusion_counts=HANDOFF_TRAINING_EXCLUSION_COUNTS,
        stage_counts=HANDOFF_STAGE_COUNTS,
        frozen_count=HANDOFF_FROZEN_COUNT,
        global_quarantine_count=HANDOFF_GLOBAL_QUARANTINE_COUNT,
        shape512_family_exclusion_count=(
            HANDOFF_SHAPE512_FAMILY_EXCLUSION_COUNT
        ),
    )


def preflight_stage(
    stage: str, root: Path, config_path: Path
) -> StagePreflight:
    """Legacy ABO preflight preserving monkeypatchable wrapper validators."""
    assets, digest, evidence_bytes = _materialization_scope(stage, root)
    counts = validate_stage_structure(stage, root, assets)
    anchors = validate_direct_loader(stage, root, assets, config_path)
    return StagePreflight(
        stage,
        Path(root),
        len(assets),
        digest,
        anchors,
        counts,
        evidence_bytes,
    )


def _validate_existing_chain(
    source: str, paths: Mapping[str, Path]
) -> None:
    validated = validate_source_training_data(
        source, Path(paths["training-data"])
    )
    expected = {
        "report": validated.report_path,
        "handoff": validated.handoff_path,
        "training-data": validated.path,
    }
    for label, expected_path in expected.items():
        selected = Path(paths[label])
        canonical = selected.resolve()
        if str(selected) != str(canonical):
            raise ValueError(
                f"source={source} selected {label} path must be canonical: "
                f"{selected}"
            )
        if canonical != expected_path:
            raise ValueError(
                f"source={source} selected {label} path does not match "
                f"validated chain: {selected}"
            )


def _verify_existing(
    source: str, paths: Mapping[str, Path]
) -> None:
    """Validate the source chain, then print read-only byte digests."""
    _validate_existing_chain(source, paths)
    for label, path in paths.items():
        raw = _core._existing_regular_bytes(Path(path))
        print(f"{label} {path} sha256={sha256(raw).hexdigest()}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=SOURCE_PROFILE_NAMES, default="abo"
    )
    parser.add_argument("--data2-root", type=Path, default=None)
    parser.add_argument("--local-root", type=Path, default=None)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--training-data", type=Path)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args(argv)


def source_publication_paths(output_root: Path) -> dict[str, Path]:
    """Return every local publication path derived from one output root."""
    root = Path(output_root)
    return {
        "report": root / "publication/report.json",
        "handoff": root / "publication/handoff.json",
        "training-data": root / "training_data.json",
    }


def resolve_profile_paths(
    args: argparse.Namespace,
) -> tuple[ProductionSourceSpec, Path, Path]:
    """Build source inputs and a local output path from optional node roots."""
    data2_root = args.data2_root or Path("/root/data2/pixal3d")
    local_root = args.local_root or Path("/root/node17/data/pixal3d")
    spec = build_source_spec(args.profile, data2_root)
    prepared = data2_root / "prepared"
    output_root = source_output_root(args.profile, local_root)
    return spec, prepared, output_root


def _publish_legacy_abo(
    index: Path, root: Path, paths: Mapping[str, Path]
) -> tuple[Path, Path, Path]:
    results: dict[str, StagePreflight] = {}
    for stage in HANDOFF_STAGE_COUNTS:
        results[stage] = preflight_stage(
            stage, root / stage / "active", CONFIGS[stage]
        )
    materializations = {
        stage: _core._materialization_evidence_from_result(result)
        for stage, result in results.items()
    }
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return publish_handoff(
        index,
        results,
        materializations,
        paths["report"],
        paths["handoff"],
        paths["training-data"],
        created_at,
    )


def _publish_source_profile(
    spec: ProductionSourceSpec, root: Path, paths: Mapping[str, Path]
) -> tuple[Path, Path, Path]:
    results = {
        stage: source_preflight_stage(
            spec, stage, root / stage / "active", CONFIGS[stage]
        )
        for stage in COMPONENTS
    }
    return publish_source_handoff(
        spec,
        results,
        paths["report"],
        paths["handoff"],
        paths["training-data"],
    )


def main() -> None:
    args = _parse_args()
    root_aware = args.data2_root is not None or args.local_root is not None
    legacy_paths = (
        args.root,
        args.index,
        args.report,
        args.handoff,
        args.training_data,
    )
    if root_aware:
        if any(path is not None for path in legacy_paths):
            raise ValueError(
                "root-aware profile selection cannot be combined with legacy paths"
            )
        spec, _prepared, root = resolve_profile_paths(args)
        paths = source_publication_paths(root)
        if args.verify_existing:
            _verify_existing(spec.source, paths)
            return
        if spec.source == SOURCE:
            paths = _publish_legacy_abo(spec.indexes[0], root, paths)
        else:
            paths = _publish_source_profile(spec, root, paths)
    elif args.profile == "hssd":
        if args.index is not None:
            raise ValueError(
                "hssd profile binds both indexes from its source spec"
            )
        if any(path is not None for path in (
            args.root, args.report, args.handoff, args.training_data,
        )):
            raise ValueError("hssd profile requires profile-derived paths")
        spec, _prepared, root = resolve_profile_paths(args)
        paths = source_publication_paths(root)
        if args.verify_existing:
            _verify_existing(spec.source, paths)
            return
        paths = _publish_source_profile(spec, root, paths)
    elif args.profile == "abo":
        root = args.root or DEFAULT_ROOT
        index = args.index or DEFAULT_INDEX
        report_path = args.report or DEFAULT_REPORT
        handoff_path = args.handoff or DEFAULT_HANDOFF
        training_data_path = args.training_data or DEFAULT_TRAINING_DATA
        if args.verify_existing:
            _verify_existing(
                SOURCE,
                {
                    "report": report_path,
                    "handoff": handoff_path,
                    "training-data": training_data_path,
                }
            )
            return
        paths = _publish_legacy_abo(index, root, {
            "report": report_path,
            "handoff": handoff_path,
            "training-data": training_data_path,
        })
    else:
        if args.index is not None:
            raise ValueError(
                "3d-future profile binds both indexes from its source spec"
            )
        root = args.root or THREED_FUTURE_ROOT
        report_path = args.report or THREED_FUTURE_REPORT
        handoff_path = args.handoff or THREED_FUTURE_HANDOFF
        training_data_path = (
            args.training_data or THREED_FUTURE_TRAINING_DATA
        )
        if args.verify_existing:
            _verify_existing(
                THREED_FUTURE_SOURCE_SPEC.source,
                {
                    "report": report_path,
                    "handoff": handoff_path,
                    "training-data": training_data_path,
                }
            )
            return
        paths = _publish_source_profile(
            THREED_FUTURE_SOURCE_SPEC,
            root,
            {
                "report": report_path,
                "handoff": handoff_path,
                "training-data": training_data_path,
            },
        )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
