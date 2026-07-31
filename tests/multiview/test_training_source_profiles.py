from dataclasses import replace
from pathlib import Path

import pytest

from data_toolkit.pipeline.training_source_profiles import (
    SOURCE_ACCEPTANCE_CONTRACTS,
    SOURCE_PROFILE_NAMES,
    build_source_spec,
    source_output_root,
    validate_production_source_spec,
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


@pytest.mark.parametrize(
    ("profile", "mutation"),
    (
        ("hssd", lambda spec: replace(spec, source="3D-FUTURE")),
        (
            "hssd",
            lambda spec: replace(spec, indexes=tuple(reversed(spec.indexes))),
        ),
        (
            "hssd",
            lambda spec: replace(
                spec,
                indexes=(
                    spec.indexes[0].with_name("HSSD-99999.json"),
                    spec.indexes[1],
                ),
            ),
        ),
        (
            "hssd",
            lambda spec: replace(
                spec,
                expected_batches={
                    **spec.expected_batches,
                    "HSSD-00000": tuple(
                        reversed(spec.expected_batches["HSSD-00000"])
                    ),
                },
            ),
        ),
        ("hssd", lambda spec: replace(spec, expected_frozen=3)),
        (
            "hssd",
            lambda spec: replace(
                spec,
                expected_candidate_stages={
                    **spec.expected_candidate_stages,
                    "ss64": 3,
                },
            ),
        ),
        ("abo", lambda spec: replace(spec, fixed_count_contract=None)),
        (
            "3d-future",
            lambda spec: replace(
                spec, acceptance_mode="production_gate"
            ),
        ),
        (
            "hssd",
            lambda spec: replace(
                spec, original_90_percent_gate_passed=1
            ),
        ),
    ),
)
def test_production_profile_validation_rejects_noncanonical_contracts(
    profile, mutation
):
    """A named source must not substitute any part of its approved profile."""
    spec = build_source_spec(profile, Path("/srv/data2/pixal3d"))

    with pytest.raises(ValueError, match="canonical production profile"):
        validate_production_source_spec(mutation(spec))


@pytest.mark.parametrize("profile", SOURCE_PROFILE_NAMES)
def test_production_profile_validation_preserves_root_relocation(profile):
    root = Path("/srv/data2/pixal3d")
    spec = build_source_spec(profile, root)

    validated_profile, validated_root = validate_production_source_spec(spec)

    assert validated_profile == profile
    assert validated_root == root


def test_production_profile_validation_rejects_symlinked_data_root(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    spec = build_source_spec("hssd", linked)

    with pytest.raises(ValueError, match="canonical production profile"):
        validate_production_source_spec(spec)
