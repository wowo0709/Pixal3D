import json
import math

import pandas as pd
import pytest

from data_toolkit.pipeline.reporting import (
    ReportValidationError,
    build_gate_report,
    byte_quantiles,
    capacity_projection,
    checksum_summary,
    failure_categories,
    fp16_parity,
    resource_peaks,
    source_counts,
    split_overlap,
    throughput_quantiles,
    validate_gate_report,
    write_report,
)


CONFIG_HASH = "a" * 64


def valid_report_payload(gate="pilot"):
    return {
        "schema_version": 1,
        "report_type": "pipeline_gate",
        "gate": gate,
        "decision": "passed",
        "config_hash": CONFIG_HASH,
        "source_counts": {"ABO": 2},
        "failure_categories": {"none": 2},
        "resource_peaks": {
            "cpu_percent": 25.0,
            "available_ram_gib_min": 200.0,
            "gpu_memory_used_mib": 1024.0,
        },
        "throughput_quantiles": {
            "p50_assets_per_hour": 10.0,
            "p95_assets_per_hour": 12.0,
            "p99_assets_per_hour": 13.0,
        },
        "byte_quantiles": {
            "p50_final_bytes": 100.0,
            "p95_final_bytes": 120.0,
            "p99_final_bytes": 130.0,
        },
        "fp16_parity": {
            "samples": 32,
            "coordinates_exact": True,
            "non_finite_values": 0,
            "p99_abs_error": 0.001,
            "decode_degradation_percent": 0.05,
            "passed": True,
        },
        "checksums": {
            "algorithm": "sha256",
            "verified": 2,
            "failed": 0,
        },
        "split_overlap": {"count": 0, "passed": True},
        "capacity": {
            "assets": 2,
            "p95_final_bytes": 120.0,
            "headroom": 1.25,
            "projected_bytes": 300,
            "soft_limit_bytes": 1024,
            "within_soft_limit": True,
            "sources": {"ABO": {"p95_peak_local_bytes": 200}},
        },
        "handoff": {
            "ready": True,
            "pack_families": 8,
            "stage_extractable": True,
        },
        "audits": {
            "pack_verification": True,
            "raw_archive_verification": True,
            "split_overlap": True,
        },
        "quality": {
            "assets": 2,
            "end_to_end_failures": 0,
            "schema_failures": 0,
            "end_to_end_failure_rate": 0.0,
            "schema_failure_rate": 0.0,
        },
        "hardware": {
            "checked": True,
            "cycles_device": "OPTIX",
            "gpu_count": 1,
            "cpu_fallback_detected": False,
        },
    }


def test_capacity_projection_uses_p95_and_headroom():
    result = capacity_projection(
        pd.DataFrame({"final_bytes": [100, 120, 140, 160]}),
        500_777,
        1.25,
    )

    assert result["assets"] == 500_777
    assert result["projected_bytes"] >= 500_777 * 140 * 1.25


@pytest.mark.parametrize(
    "pilot",
    [
        pd.DataFrame(),
        pd.DataFrame({"other": [1]}),
        pd.DataFrame({"final_bytes": []}),
        pd.DataFrame({"final_bytes": [math.nan]}),
        pd.DataFrame({"final_bytes": [0]}),
    ],
)
def test_capacity_projection_rejects_misleading_input(pilot):
    with pytest.raises(ReportValidationError):
        capacity_projection(pilot, 10)


def test_report_builders_cover_required_metrics():
    registry = pd.DataFrame(
        {
            "sha256": ["a" * 64, "b" * 64],
            "owner_source": ["ABO", "HSSD"],
            "split": ["train", "validation"],
            "last_error_category": ["", "schema"],
        }
    )
    telemetry = [
        {
            "cpu_percent": 10.0,
            "available_ram_gib": 200.0,
            "gpu_metrics": [{"memory_used_mib": 100.0}],
        },
        {
            "cpu_percent": 20.0,
            "available_ram_gib": 180.0,
            "gpu_metrics": [{"memory_used_mib": 150.0}],
        },
    ]
    measurements = pd.DataFrame(
        {
            "assets_per_hour": [10.0, 20.0] * 16,
            "final_bytes": [100, 200] * 16,
            "fp16_abs_error": [0.001, 0.002] * 16,
            "coordinates_match": [True] * 32,
            "fp16_finite": [True] * 32,
            "decode_degradation_percent": [0.05] * 32,
        }
    )
    checksums = pd.DataFrame({"verified": [True, True]})

    assert source_counts(registry) == {"ABO": 1, "HSSD": 1}
    assert failure_categories(registry) == {"none": 1, "schema": 1}
    assert resource_peaks(telemetry)["cpu_percent"] == 20.0
    throughput = throughput_quantiles(measurements)
    bytes_result = byte_quantiles(measurements)
    assert throughput["p50_assets_per_hour"] == 15.0
    assert throughput["p99_assets_per_hour"] == 20.0
    assert bytes_result["p95_final_bytes"] > 100
    assert bytes_result["p99_final_bytes"] == 200.0
    assert fp16_parity(measurements, 0.01)["passed"] is True
    assert checksum_summary(checksums) == {
        "algorithm": "sha256",
        "verified": 2,
        "failed": 0,
    }
    assert split_overlap(registry)["passed"] is True


@pytest.mark.parametrize(
    ("builder", "value"),
    [
        (source_counts, pd.DataFrame()),
        (failure_categories, pd.DataFrame()),
        (resource_peaks, []),
        (throughput_quantiles, pd.DataFrame()),
        (byte_quantiles, pd.DataFrame()),
        (checksum_summary, pd.DataFrame()),
        (split_overlap, pd.DataFrame()),
    ],
)
def test_report_builders_reject_empty_input(builder, value):
    with pytest.raises(ReportValidationError):
        builder(value)


def test_build_gate_report_rejects_threshold_failure():
    payload = valid_report_payload()
    payload["quality"]["end_to_end_failures"] = 1
    payload["quality"]["end_to_end_failure_rate"] = 0.5

    with pytest.raises(ReportValidationError, match="failure rate"):
        build_gate_report(payload)


def test_failed_gate_report_retains_exact_failure_evidence():
    payload = valid_report_payload()
    payload["decision"] = "failed"
    payload["checksums"]["verified"] = 0
    payload["checksums"]["failed"] = 2
    payload["hardware"]["cycles_device"] = "CPU"
    payload["handoff"]["pack_families"] = 7

    assert validate_gate_report(payload)["decision"] == "failed"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("coordinates_match", [False] + [True] * 31),
        ("fp16_finite", [False] + [True] * 31),
        ("fp16_abs_error", [0.02] + [0.001] * 31),
        ("decode_degradation_percent", [0.2] + [0.05] * 31),
    ],
)
def test_fp16_parity_enforces_qualification_thresholds(field, value):
    frame = pd.DataFrame(
        {
            "fp16_abs_error": [0.001] * 32,
            "coordinates_match": [True] * 32,
            "fp16_finite": [True] * 32,
            "decode_degradation_percent": [0.05] * 32,
        }
    )
    frame[field] = value

    assert fp16_parity(frame, 0.01)["passed"] is False


def test_gate_report_rejects_forged_fp16_decision():
    payload = valid_report_payload()
    payload["fp16_parity"]["p99_abs_error"] = 0.02

    with pytest.raises(ReportValidationError, match="FP16"):
        validate_gate_report(payload, expected_config_hash=CONFIG_HASH)


def test_gate_report_rejects_forged_capacity_decision():
    payload = valid_report_payload()
    payload["capacity"]["projected_bytes"] = 2048

    with pytest.raises(ReportValidationError, match="capacity"):
        validate_gate_report(payload, expected_config_hash=CONFIG_HASH)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (("audits", "pack_verification"), "yes"),
        (("checksums", "verified"), True),
        (("quality", "end_to_end_failure_rate"), math.nan),
        (("hardware", "checked"), 1),
        (("hardware", "gpu_count"), 0),
        (("hardware", "cpu_fallback_detected"), "false"),
    ],
)
def test_gate_report_cannot_be_forged_with_truthy_fields(field, value):
    payload = valid_report_payload()
    payload[field[0]][field[1]] = value

    with pytest.raises(ReportValidationError):
        validate_gate_report(payload, expected_config_hash=CONFIG_HASH)


def test_gate_report_requires_exact_config_hash():
    with pytest.raises(ReportValidationError, match="config hash"):
        validate_gate_report(
            valid_report_payload(), expected_config_hash="b" * 64
        )


def test_write_report_writes_valid_json_and_markdown_atomically(tmp_path):
    payload = valid_report_payload()

    json_path, markdown_path = write_report(tmp_path, "pilot", payload)

    assert json.loads(json_path.read_text()) == payload
    assert "# pilot" in markdown_path.read_text()
    assert not tuple(tmp_path.glob("*.tmp"))


def test_write_report_rejects_symlinked_destination(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("unchanged")
    root = tmp_path / "reports"
    root.mkdir()
    (root / "pilot.json").symlink_to(outside)

    with pytest.raises(ReportValidationError, match="unsafe"):
        write_report(root, "pilot", valid_report_payload())

    assert outside.read_text() == "unchanged"


def test_write_report_preflights_markdown_before_publishing_json(tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("unchanged")
    root = tmp_path / "reports"
    root.mkdir()
    (root / "pilot.md").symlink_to(outside)

    with pytest.raises(ReportValidationError, match="unsafe"):
        write_report(root, "pilot", valid_report_payload())

    assert not (root / "pilot.json").exists()
    assert outside.read_text() == "unchanged"
