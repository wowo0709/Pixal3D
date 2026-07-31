from pathlib import Path

import pytest

from scripts import materialize_multiview_production as materialize_cli
from data_toolkit.pipeline.training_materialization import (
    DEFAULT_INDEX,
    DEFAULT_OUTPUT,
    HSSD_SOURCE_SPEC,
    THREED_FUTURE_INDEXES,
    THREED_FUTURE_OUTPUT,
    THREED_FUTURE_SOURCE_SPEC,
)
from data_toolkit.pipeline.training_eligibility import (
    ABO_COUNT_CONTRACT,
    EXPECTED_FINAL_STAGE_COUNTS,
    observed_count_contract,
)


def test_3d_future_profile_has_exact_two_shards():
    """Dropping or renaming a completed shard must break the source contract."""
    assert THREED_FUTURE_SOURCE_SPEC.source == "3D-FUTURE"
    assert THREED_FUTURE_SOURCE_SPEC.indexes == (
        Path(
            "/root/data2/pixal3d/prepared/index/3D-FUTURE/"
            "3D-FUTURE-00000.json"
        ),
        Path(
            "/root/data2/pixal3d/prepared/index/3D-FUTURE/"
            "3D-FUTURE-00001.json"
        ),
    )
    assert THREED_FUTURE_SOURCE_SPEC.expected_batches == {
        "3D-FUTURE-00000": tuple(f"batch{i:03d}" for i in range(20)),
        "3D-FUTURE-00001": tuple(f"batch{i:03d}" for i in range(18)),
    }
    assert THREED_FUTURE_SOURCE_SPEC.expected_frozen == 9472


def test_legacy_materialization_aliases_keep_node17_paths_and_values():
    """Changing compatibility aliases must not redirect existing Node17 jobs."""
    assert DEFAULT_INDEX == Path(
        "/root/data2/pixal3d/prepared/index/ABO/ABO-00000.json"
    )
    assert DEFAULT_OUTPUT == Path(
        "/root/node17/data/pixal3d/train/production/abo"
    )
    assert THREED_FUTURE_INDEXES == (
        Path(
            "/root/data2/pixal3d/prepared/index/3D-FUTURE/"
            "3D-FUTURE-00000.json"
        ),
        Path(
            "/root/data2/pixal3d/prepared/index/3D-FUTURE/"
            "3D-FUTURE-00001.json"
        ),
    )
    assert THREED_FUTURE_OUTPUT == Path(
        "/root/node17/data/pixal3d/train/production/3d-future"
    )
    assert HSSD_SOURCE_SPEC.source == "HSSD"


def test_hssd_materialization_cli_derives_node16_paths():
    """A root-aware HSSD invocation must not inherit Node17 paths."""
    args = materialize_cli._parse_args([
        "--profile", "hssd",
        "--data2-root", "/file2/youngwoo/pixal3d",
        "--local-root", "/home/youngwoo/data/pixal3d",
    ])

    spec, prepared, output = materialize_cli.resolve_profile_paths(args)

    assert spec.source == "HSSD"
    assert prepared == Path("/file2/youngwoo/pixal3d/prepared")
    assert output == Path(
        "/home/youngwoo/data/pixal3d/train/production/hssd"
    )


def test_3d_future_profile_preserves_observed_candidate_counts():
    """Changing a pack-family intersection count must invalidate the profile."""
    assert THREED_FUTURE_SOURCE_SPEC.expected_candidate_stages == {
        "ss64": 8495,
        "shape512": 8513,
        "shape1024": 8495,
        "pbr1024": 8495,
    }
    assert THREED_FUTURE_SOURCE_SPEC.fixed_count_contract is None
    assert (
        THREED_FUTURE_SOURCE_SPEC.acceptance_mode
        == "valid_subset_user_waiver"
    )
    assert THREED_FUTURE_SOURCE_SPEC.original_90_percent_gate_passed is False


def test_abo_count_contract_preserves_published_values():
    """Changing published ABO final counts or top-level totals must break the contract."""
    assert ABO_COUNT_CONTRACT["stages"] == EXPECTED_FINAL_STAGE_COUNTS
    assert ABO_COUNT_CONTRACT["frozen"] == 4485
    assert ABO_COUNT_CONTRACT["global_quarantine"] == 825


def test_observed_count_contract_derives_final_counts():
    """A wrong subtraction or pack-exclusion calculation must alter the count contract."""
    value = observed_count_contract(
        frozen=10,
        candidate_stages={
            "ss64": 9, "shape512": 8, "shape1024": 9, "pbr1024": 7,
        },
        training_exclusions={
            "ss64": 0, "shape512": 2, "shape1024": 1, "pbr1024": 3,
        },
    )
    assert value == {
        "frozen": 10,
        "candidate_stages": {
            "ss64": 9, "shape512": 8, "shape1024": 9, "pbr1024": 7,
        },
        "pack_exclusions": {
            "ss64": 1, "shape512": 2, "shape1024": 1, "pbr1024": 3,
        },
        "training_exclusions": {
            "ss64": 0, "shape512": 2, "shape1024": 1, "pbr1024": 3,
        },
        "stages": {
            "ss64": 9, "shape512": 6, "shape1024": 8, "pbr1024": 4,
        },
    }


@pytest.mark.parametrize(
    ("frozen", "candidate_stages", "training_exclusions", "message"),
    [
        (
            10,
            {"ss64": 9, "shape512": 8, "shape1024": 9},
            {"ss64": 0, "shape512": 2, "shape1024": 1, "pbr1024": 3},
            "candidate_stages must contain exactly",
        ),
        (
            10,
            {"ss64": 9, "shape512": 8, "shape1024": 9, "pbr1024": 7},
            {"ss64": 0, "shape512": -2, "shape1024": 1, "pbr1024": 3},
            "training_exclusions must contain non-negative integers",
        ),
        (
            10,
            {"ss64": 9, "shape512": 8, "shape1024": 9, "pbr1024": 7},
            {"ss64": 0, "shape512": 9, "shape1024": 1, "pbr1024": 3},
            "training exclusion exceeds candidate count",
        ),
        (
            10,
            {"ss64": 11, "shape512": 8, "shape1024": 9, "pbr1024": 7},
            {"ss64": 0, "shape512": 2, "shape1024": 1, "pbr1024": 3},
            "candidate count exceeds frozen count",
        ),
    ],
)
def test_observed_count_contract_rejects_invalid_count_boundaries(
    frozen, candidate_stages, training_exclusions, message
):
    """Missing stages and invalid count relationships must not produce a contract."""
    with pytest.raises(ValueError, match=message):
        observed_count_contract(
            frozen=frozen,
            candidate_stages=candidate_stages,
            training_exclusions=training_exclusions,
        )
