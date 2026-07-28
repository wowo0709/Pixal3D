import pytest

from data_toolkit.pipeline.training_eligibility import (
    ABO_COUNT_CONTRACT,
    EXPECTED_FINAL_STAGE_COUNTS,
    observed_count_contract,
)


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
