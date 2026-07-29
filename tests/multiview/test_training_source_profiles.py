from pathlib import Path

import pytest

from data_toolkit.pipeline.training_source_profiles import (
    SOURCE_ACCEPTANCE_CONTRACTS,
    SOURCE_PROFILE_NAMES,
    build_source_spec,
    source_output_root,
)


def test_hssd_profile_pins_completed_two_shard_contract():
    spec = build_source_spec(
        "hssd", Path("/file2/youngwoo/pixal3d")
    )
    assert spec.source == "HSSD"
    assert spec.indexes == (
        Path(
            "/file2/youngwoo/pixal3d/prepared/index/HSSD/"
            "HSSD-00000.json"
        ),
        Path(
            "/file2/youngwoo/pixal3d/prepared/index/HSSD/"
            "HSSD-00001.json"
        ),
    )
    assert spec.expected_batches == {
        "HSSD-00000": tuple(f"batch{i:03d}" for i in range(20)),
        "HSSD-00001": tuple(f"batch{i:03d}" for i in range(7)),
    }
    assert spec.expected_frozen == 6670
    assert spec.expected_candidate_stages == {
        "ss64": 6078,
        "shape512": 6078,
        "shape1024": 6078,
        "pbr1024": 6078,
    }
    assert spec.fixed_count_contract is None
    assert spec.acceptance_mode == "production_gate"
    assert spec.original_90_percent_gate_passed is True
    assert SOURCE_ACCEPTANCE_CONTRACTS["HSSD"] == (
        "production_gate", True
    )


def test_profiles_relocate_indexes_without_changing_contracts():
    root = Path("/srv/data2/pixal3d")
    abo = build_source_spec("abo", root)
    future = build_source_spec("3d-future", root)
    assert abo.indexes == (
        root / "prepared/index/ABO/ABO-00000.json",
    )
    assert future.indexes[0] == (
        root
        / "prepared/index/3D-FUTURE/3D-FUTURE-00000.json"
    )
    assert future.indexes[1] == (
        root
        / "prepared/index/3D-FUTURE/3D-FUTURE-00001.json"
    )


def test_source_output_roots_are_node_local():
    local = Path("/home/youngwoo/data/pixal3d")
    assert source_output_root("abo", local) == (
        local / "train/production/abo"
    )
    assert source_output_root("3d-future", local) == (
        local / "train/production/3d-future"
    )
    assert source_output_root("hssd", local) == (
        local / "train/production/hssd"
    )


def test_unknown_profile_is_rejected():
    assert SOURCE_PROFILE_NAMES == ("abo", "3d-future", "hssd")
    with pytest.raises(ValueError, match="unknown source profile"):
        build_source_spec("toys4k", Path("/file2/youngwoo/pixal3d"))
