"""Immutable production-source contracts independent of machine roots."""

from dataclasses import dataclass
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
