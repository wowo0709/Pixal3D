"""Immutable production-source contracts independent of machine roots."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping

from data_toolkit.pipeline.training_eligibility import ABO_COUNT_CONTRACT


STAGES = ("ss64", "shape512", "shape1024", "pbr1024")
SOURCE_PROFILE_NAMES = ("abo", "3d-future", "hssd")
SOURCE_ACCEPTANCE_CONTRACTS = {
    "ABO": ("valid_subset_user_waiver", False),
    "3D-FUTURE": ("valid_subset_user_waiver", False),
    "HSSD": ("production_gate", True),
}
_PROFILE_BY_SOURCE = {
    "ABO": "abo",
    "3D-FUTURE": "3d-future",
    "HSSD": "hssd",
}


@dataclass(frozen=True)
class ProductionSourceSpec:
    """Immutable source inputs and acceptance/count contracts."""

    source: str
    indexes: tuple[Path, ...]
    expected_batches: Mapping[str, tuple[str, ...]]
    expected_frozen: int
    expected_candidate_stages: Mapping[str, int]
    fixed_count_contract: Mapping[str, object] | None
    acceptance_mode: str
    original_90_percent_gate_passed: bool


def build_source_spec(
    profile: str, data2_root: Path
) -> ProductionSourceSpec:
    """Build one source contract under the supplied shared-data root."""
    root = Path(data2_root)
    definitions = {
        "abo": {
            "source": "ABO",
            "shards": {"ABO-00000": 18},
            "frozen": 4485,
            "candidates": {
                "ss64": 3660,
                "shape512": 3631,
                "shape1024": 3660,
                "pbr1024": 3660,
            },
            "fixed": ABO_COUNT_CONTRACT,
        },
        "3d-future": {
            "source": "3D-FUTURE",
            "shards": {
                "3D-FUTURE-00000": 20,
                "3D-FUTURE-00001": 18,
            },
            "frozen": 9472,
            "candidates": {
                "ss64": 8495,
                "shape512": 8513,
                "shape1024": 8495,
                "pbr1024": 8495,
            },
            "fixed": None,
        },
        "hssd": {
            "source": "HSSD",
            "shards": {"HSSD-00000": 20, "HSSD-00001": 7},
            "frozen": 6670,
            "candidates": dict.fromkeys(STAGES, 6078),
            "fixed": None,
        },
    }
    try:
        value = definitions[profile]
    except KeyError as error:
        raise ValueError(f"unknown source profile: {profile}") from error
    source = value["source"]
    shards = value["shards"]
    acceptance_mode, gate_passed = SOURCE_ACCEPTANCE_CONTRACTS[source]
    return ProductionSourceSpec(
        source=source,
        indexes=tuple(
            root / "prepared/index" / source / f"{shard}.json"
            for shard in shards
        ),
        expected_batches={
            shard: tuple(f"batch{i:03d}" for i in range(count))
            for shard, count in shards.items()
        },
        expected_frozen=value["frozen"],
        expected_candidate_stages=value["candidates"],
        fixed_count_contract=value["fixed"],
        acceptance_mode=acceptance_mode,
        original_90_percent_gate_passed=gate_passed,
    )


def source_output_root(profile: str, local_root: Path) -> Path:
    """Return a node-local production output root for one source profile."""
    names = {"abo": "abo", "3d-future": "3d-future", "hssd": "hssd"}
    try:
        name = names[profile]
    except KeyError as error:
        raise ValueError(f"unknown source profile: {profile}") from error
    return Path(local_root) / "train/production" / name


def validate_production_source_spec(
    spec: ProductionSourceSpec,
) -> tuple[str, Path]:
    """Validate one named production profile under its relocated data root."""
    if not isinstance(spec, ProductionSourceSpec):
        raise TypeError("spec must be a ProductionSourceSpec")
    try:
        profile = _PROFILE_BY_SOURCE[spec.source]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "source spec is not a canonical production profile: "
            f"unknown source={getattr(spec, 'source', None)!r}"
        ) from error
    expected = production_source_spec_from_indexes(
        spec.source, spec.indexes
    )
    data2_root = _source_spec_root(expected)
    if not _same_typed_value(spec, expected):
        raise ValueError(
            "source spec is not a canonical production profile: "
            f"source={spec.source}"
        )
    return profile, data2_root


def production_source_spec_from_indexes(
    source: str, indexes: tuple[Path, ...]
) -> ProductionSourceSpec:
    """Derive the only production profile authorized by index identities."""
    try:
        profile = _PROFILE_BY_SOURCE[source]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "source indexes do not identify a canonical production profile: "
            f"unknown source={source!r}"
        ) from error
    if not indexes:
        raise ValueError(
            "source indexes do not identify a canonical production profile: "
            "indexes are empty"
        )
    relative_indexes = build_source_spec(profile, Path()).indexes
    roots = []
    for actual, relative in zip(
        indexes, relative_indexes, strict=False
    ):
        actual = Path(actual)
        if (
            not actual.is_absolute()
            or actual != Path(os.path.normpath(actual))
            or actual != actual.resolve(strict=False)
            or len(actual.parts) <= len(relative.parts)
            or actual.parts[-len(relative.parts):] != relative.parts
        ):
            raise ValueError(
                "source indexes do not identify a canonical production "
                "profile: "
                f"invalid index suffix={actual}"
            )
        root = actual
        for _part in relative.parts:
            root = root.parent
        roots.append(root)
    if len(roots) != len(relative_indexes) or len(set(roots)) != 1:
        raise ValueError(
            "source indexes do not identify a canonical production profile: "
            "indexes do not share the exact profile root"
        )
    data2_root = roots[0]
    expected = build_source_spec(profile, data2_root)
    if tuple(Path(path) for path in indexes) != expected.indexes:
        raise ValueError(
            "source indexes do not identify a canonical production profile: "
            f"source={source}"
        )
    return expected


def _source_spec_root(spec: ProductionSourceSpec) -> Path:
    root = spec.indexes[0]
    relative = build_source_spec(
        _PROFILE_BY_SOURCE[spec.source], Path()
    ).indexes[0]
    for _part in relative.parts:
        root = root.parent
    return root


def _same_typed_value(actual: object, expected: object) -> bool:
    """Compare nested contracts without Python's bool/int equivalence."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, ProductionSourceSpec):
        return all(
            _same_typed_value(
                getattr(actual, field),
                getattr(expected, field),
            )
            for field in expected.__dataclass_fields__
        )
    if isinstance(expected, Mapping):
        return (
            set(actual) == set(expected)
            and all(
                _same_typed_value(actual[key], expected[key])
                for key in expected
            )
        )
    if isinstance(expected, (tuple, list)):
        return len(actual) == len(expected) and all(
            _same_typed_value(actual_value, expected_value)
            for actual_value, expected_value in zip(
                actual, expected, strict=True
            )
        )
    return actual == expected
