from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .orchestrator import (
    InfrastructureError,
    _atomic_write_bytes_nofollow,
    _regular_file_stat_nofollow,
)


REPORT_SCHEMA_VERSION = 1
GATE_REPORT_TYPE = "pipeline_gate"
GATES = frozenset(("smoke", "pilot", "production"))
FAILURE_RATE_LIMIT = 0.10
SCHEMA_FAILURE_RATE_LIMIT = 0.05


class ReportValidationError(ValueError):
    pass


def _frame(value: pd.DataFrame, columns: set[str], description: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame) or value.empty:
        raise ReportValidationError(f"{description} must not be empty")
    missing = columns - set(value.columns)
    if missing:
        raise ReportValidationError(
            f"{description} is missing columns: {sorted(missing)}"
        )
    return value


def _finite_number(
    value, description: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ReportValidationError(f"{description} must be finite")
    result = float(value)
    if positive and result <= 0:
        raise ReportValidationError(f"{description} must be positive")
    if nonnegative and result < 0:
        raise ReportValidationError(f"{description} must be non-negative")
    return result


def _count(value, description: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ReportValidationError(f"{description} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ReportValidationError(f"{description} must be {qualifier}")
    return value


def _bool(value, description: str) -> bool:
    if not isinstance(value, bool):
        raise ReportValidationError(f"{description} must be a boolean")
    return value


def _mapping(value, keys: set[str], description: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReportValidationError(f"invalid {description} schema")
    return value


def _sha256(value, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReportValidationError(f"invalid {description}")
    return value


def _numeric_series(
    frame: pd.DataFrame, column: str, description: str, *, positive: bool = False
) -> pd.Series:
    try:
        values = pd.to_numeric(frame[column], errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ReportValidationError(f"invalid {description}") from error
    if values.empty or not values.map(math.isfinite).all():
        raise ReportValidationError(f"invalid {description}")
    if positive and not (values > 0).all():
        raise ReportValidationError(f"{description} must be positive")
    return values


def source_counts(registry: pd.DataFrame) -> dict[str, int]:
    frame = _frame(registry, {"owner_source"}, "registry")
    values = frame["owner_source"]
    if not values.map(lambda item: isinstance(item, str) and bool(item)).all():
        raise ReportValidationError("invalid registry source")
    counts = values.value_counts(sort=False).sort_index()
    return {str(source): int(count) for source, count in counts.items()}


def failure_categories(registry: pd.DataFrame) -> dict[str, int]:
    frame = _frame(registry, {"last_error_category"}, "registry")
    values = frame["last_error_category"]
    if not values.map(lambda item: isinstance(item, str)).all():
        raise ReportValidationError("invalid failure category")
    normalized = values.map(lambda item: item or "none")
    counts = normalized.value_counts(sort=False).sort_index()
    return {str(category): int(count) for category, count in counts.items()}


def resource_peaks(telemetry: Sequence[Mapping]) -> dict[str, float]:
    if (
        not isinstance(telemetry, Sequence)
        or isinstance(telemetry, (str, bytes))
        or not telemetry
    ):
        raise ReportValidationError("telemetry must not be empty")
    cpu = []
    ram = []
    gpu_memory = []
    for index, record in enumerate(telemetry):
        if not isinstance(record, Mapping):
            raise ReportValidationError(f"invalid telemetry record {index}")
        cpu.append(
            _finite_number(
                record.get("cpu_percent"), f"telemetry {index} CPU", nonnegative=True
            )
        )
        ram.append(
            _finite_number(
                record.get("available_ram_gib"),
                f"telemetry {index} RAM",
                nonnegative=True,
            )
        )
        metrics = record.get("gpu_metrics")
        if not isinstance(metrics, (list, tuple)):
            raise ReportValidationError(f"invalid telemetry {index} GPU metrics")
        for metric in metrics:
            if not isinstance(metric, Mapping):
                raise ReportValidationError("invalid GPU metric")
            gpu_memory.append(
                _finite_number(
                    metric.get("memory_used_mib"),
                    "GPU memory",
                    nonnegative=True,
                )
            )
    return {
        "cpu_percent": max(cpu),
        "available_ram_gib_min": min(ram),
        "gpu_memory_used_mib": max(gpu_memory, default=0.0),
    }


def throughput_quantiles(measurements: pd.DataFrame) -> dict[str, float]:
    frame = _frame(measurements, {"assets_per_hour"}, "measurements")
    values = _numeric_series(
        frame, "assets_per_hour", "assets-per-hour measurement", positive=True
    )
    return {
        "p50_assets_per_hour": float(values.quantile(0.50)),
        "p95_assets_per_hour": float(values.quantile(0.95)),
        "p99_assets_per_hour": float(values.quantile(0.99)),
    }


def byte_quantiles(measurements: pd.DataFrame) -> dict[str, float]:
    frame = _frame(measurements, {"final_bytes"}, "measurements")
    values = _numeric_series(
        frame, "final_bytes", "final-byte measurement", positive=True
    )
    return {
        "p50_final_bytes": float(values.quantile(0.50)),
        "p95_final_bytes": float(values.quantile(0.95)),
        "p99_final_bytes": float(values.quantile(0.99)),
    }


def fp16_parity(
    measurements: pd.DataFrame,
    max_abs_error: float,
    max_decode_degradation_percent: float = 0.1,
) -> dict:
    frame = _frame(
        measurements,
        {
            "fp16_abs_error",
            "coordinates_match",
            "fp16_finite",
            "decode_degradation_percent",
        },
        "measurements",
    )
    threshold = _finite_number(max_abs_error, "FP16 threshold", positive=True)
    decode_threshold = _finite_number(
        max_decode_degradation_percent,
        "decode-degradation threshold",
        positive=True,
    )
    values = _numeric_series(frame, "fp16_abs_error", "FP16 error")
    if not (values >= 0).all():
        raise ReportValidationError("FP16 errors must be non-negative")
    for column in ("coordinates_match", "fp16_finite"):
        if not frame[column].map(lambda value: isinstance(value, bool)).all():
            raise ReportValidationError(f"{column} must contain booleans")
    degradation = _numeric_series(
        frame,
        "decode_degradation_percent",
        "decode degradation",
    )
    if not (degradation >= 0).all():
        raise ReportValidationError("decode degradation must be non-negative")
    coordinates_exact = bool(frame["coordinates_match"].all())
    non_finite_values = int((~frame["fp16_finite"]).sum())
    p99 = float(values.quantile(0.99))
    maximum_degradation = float(degradation.max())
    passed = all(
        (
            len(values) >= 32,
            coordinates_exact,
            non_finite_values == 0,
            p99 <= threshold,
            maximum_degradation <= decode_threshold,
        )
    )
    return {
        "samples": int(len(values)),
        "coordinates_exact": coordinates_exact,
        "non_finite_values": non_finite_values,
        "p99_abs_error": p99,
        "decode_degradation_percent": maximum_degradation,
        "passed": passed,
    }


def checksum_summary(checksums: pd.DataFrame) -> dict:
    frame = _frame(checksums, {"verified"}, "checksums")
    values = frame["verified"]
    if not values.map(lambda item: isinstance(item, bool)).all():
        raise ReportValidationError("checksum results must be booleans")
    verified = int(values.sum())
    return {
        "algorithm": "sha256",
        "verified": verified,
        "failed": int(len(values) - verified),
    }


def split_overlap(registry: pd.DataFrame) -> dict:
    frame = _frame(registry, {"sha256", "split"}, "registry")
    for asset_sha in frame["sha256"]:
        _sha256(asset_sha, "registry SHA-256")
    if not frame["split"].map(
        lambda value: value in {"train", "validation"}
    ).all():
        raise ReportValidationError("invalid registry split")
    overlap = int(
        (frame.groupby("sha256", sort=False)["split"].nunique() > 1).sum()
    )
    return {"count": overlap, "passed": overlap == 0}


def capacity_projection(
    pilot: pd.DataFrame, total_assets: int, headroom: float = 1.25
) -> dict:
    frame = _frame(pilot, {"final_bytes"}, "pilot")
    assets = _count(total_assets, "total assets", positive=True)
    multiplier = _finite_number(headroom, "capacity headroom", positive=True)
    if multiplier < 1:
        raise ReportValidationError("capacity headroom must be at least one")
    values = _numeric_series(frame, "final_bytes", "pilot final bytes", positive=True)
    p95 = float(values.quantile(0.95))
    return {
        "assets": assets,
        "p95_final_bytes": p95,
        "headroom": multiplier,
        "projected_bytes": math.ceil(p95 * assets * multiplier),
    }


def _validate_gate_sections(value: Mapping) -> None:
    counts = _mapping(
        value["source_counts"], set(value["source_counts"]), "source counts"
    )
    if not counts:
        raise ReportValidationError("source counts must not be empty")
    for source, count in counts.items():
        if not isinstance(source, str) or not source:
            raise ReportValidationError("invalid source count key")
        _count(count, f"source count {source}", positive=True)

    failures = value["failure_categories"]
    if not isinstance(failures, Mapping) or not failures:
        raise ReportValidationError("failure categories must not be empty")
    for category, count in failures.items():
        if not isinstance(category, str) or not category:
            raise ReportValidationError("invalid failure category")
        _count(count, f"failure count {category}")

    peaks = _mapping(
        value["resource_peaks"],
        {"cpu_percent", "available_ram_gib_min", "gpu_memory_used_mib"},
        "resource peaks",
    )
    for name, number in peaks.items():
        _finite_number(number, name, nonnegative=True)

    throughput = _mapping(
        value["throughput_quantiles"],
        {
            "p50_assets_per_hour",
            "p95_assets_per_hour",
            "p99_assets_per_hour",
        },
        "throughput quantiles",
    )
    for name, number in throughput.items():
        _finite_number(number, name, positive=True)

    byte_values = _mapping(
        value["byte_quantiles"],
        {"p50_final_bytes", "p95_final_bytes", "p99_final_bytes"},
        "byte quantiles",
    )
    for name, number in byte_values.items():
        _finite_number(number, name, positive=True)

    parity = _mapping(
        value["fp16_parity"],
        {
            "samples",
            "coordinates_exact",
            "non_finite_values",
            "p99_abs_error",
            "decode_degradation_percent",
            "passed",
        },
        "FP16 parity",
    )
    parity_samples = _count(parity["samples"], "FP16 samples", positive=True)
    coordinates_exact = _bool(
        parity["coordinates_exact"], "FP16 coordinate identity"
    )
    non_finite = _count(
        parity["non_finite_values"], "FP16 non-finite values"
    )
    p99_error = _finite_number(
        parity["p99_abs_error"], "FP16 p99 error", nonnegative=True
    )
    degradation = _finite_number(
        parity["decode_degradation_percent"],
        "FP16 decode degradation",
        nonnegative=True,
    )
    parity_passed = _bool(parity["passed"], "FP16 parity decision")
    expected_parity = all(
        (
            parity_samples >= 32,
            coordinates_exact,
            non_finite == 0,
            p99_error <= 0.01,
            degradation <= 0.1,
        )
    )
    if parity_passed != expected_parity:
        raise ReportValidationError("inconsistent FP16 parity decision")

    checksums = _mapping(
        value["checksums"],
        {"algorithm", "verified", "failed"},
        "checksums",
    )
    if checksums["algorithm"] != "sha256":
        raise ReportValidationError("checksums must use SHA-256")
    verified = _count(checksums["verified"], "verified checksums")
    failed_checksums = _count(checksums["failed"], "failed checksums")

    overlap = _mapping(
        value["split_overlap"], {"count", "passed"}, "split overlap"
    )
    overlap_count = _count(overlap["count"], "split overlap count")
    overlap_passed = _bool(overlap["passed"], "split overlap decision")
    if overlap_passed != (overlap_count == 0):
        raise ReportValidationError("inconsistent split overlap decision")

    capacity = _mapping(
        value["capacity"],
        {
            "assets",
            "p95_final_bytes",
            "headroom",
            "projected_bytes",
            "soft_limit_bytes",
            "within_soft_limit",
            "sources",
        },
        "capacity",
    )
    _count(capacity["assets"], "capacity assets", positive=True)
    _finite_number(capacity["p95_final_bytes"], "capacity p95", positive=True)
    if _finite_number(capacity["headroom"], "capacity headroom", positive=True) < 1:
        raise ReportValidationError("capacity headroom must be at least one")
    projected_bytes = _count(
        capacity["projected_bytes"], "projected bytes", positive=True
    )
    soft_limit_bytes = _count(
        capacity["soft_limit_bytes"], "capacity soft limit", positive=True
    )
    within_soft_limit = _bool(
        capacity["within_soft_limit"], "capacity decision"
    )
    if within_soft_limit != (projected_bytes <= soft_limit_bytes):
        raise ReportValidationError("inconsistent capacity decision")
    sources = capacity["sources"]
    if not isinstance(sources, Mapping) or not sources:
        raise ReportValidationError("capacity sources must not be empty")
    for source, source_value in sources.items():
        if not isinstance(source, str) or not source:
            raise ReportValidationError("invalid capacity source")
        source_value = _mapping(
            source_value, {"p95_peak_local_bytes"}, "source capacity"
        )
        _count(
            source_value["p95_peak_local_bytes"],
            f"{source} pilot p95",
            positive=True,
        )

    handoff = _mapping(
        value["handoff"],
        {"ready", "pack_families", "stage_extractable"},
        "handoff",
    )
    handoff_ready = _bool(handoff["ready"], "handoff decision")
    pack_families = _count(
        handoff["pack_families"], "pack families", positive=True
    )
    stage_extractable = _bool(
        handoff["stage_extractable"], "stage extraction decision"
    )

    audits = _mapping(
        value["audits"],
        {"pack_verification", "raw_archive_verification", "split_overlap"},
        "audits",
    )
    audit_values = [
        _bool(audits[name], f"{name} audit") for name in sorted(audits)
    ]

    quality = _mapping(
        value["quality"],
        {
            "assets",
            "end_to_end_failures",
            "schema_failures",
            "end_to_end_failure_rate",
            "schema_failure_rate",
        },
        "quality",
    )
    assets = _count(quality["assets"], "quality assets", positive=True)
    failures_count = _count(
        quality["end_to_end_failures"], "end-to-end failures"
    )
    schema_count = _count(quality["schema_failures"], "schema failures")
    failure_rate = _finite_number(
        quality["end_to_end_failure_rate"],
        "end-to-end failure rate",
        nonnegative=True,
    )
    schema_rate = _finite_number(
        quality["schema_failure_rate"],
        "schema failure rate",
        nonnegative=True,
    )
    if failures_count > assets or schema_count > assets:
        raise ReportValidationError("quality failure count exceeds assets")
    if not math.isclose(failure_rate, failures_count / assets, abs_tol=1e-12):
        raise ReportValidationError("inconsistent end-to-end failure rate")
    if not math.isclose(schema_rate, schema_count / assets, abs_tol=1e-12):
        raise ReportValidationError("inconsistent schema failure rate")

    hardware = _mapping(
        value["hardware"],
        {"checked", "cycles_device", "gpu_count", "cpu_fallback_detected"},
        "hardware",
    )
    checked = _bool(hardware["checked"], "hardware checked")
    if not isinstance(hardware["cycles_device"], str) or not hardware[
        "cycles_device"
    ]:
        raise ReportValidationError("hardware device must be non-empty")
    gpu_count = _count(hardware["gpu_count"], "hardware GPU count")
    cpu_fallback = _bool(
        hardware["cpu_fallback_detected"], "CPU fallback detection"
    )

    expected_pass = all(
        (
            parity_passed,
            within_soft_limit,
            failed_checksums == 0,
            overlap_passed,
            handoff_ready,
            stage_extractable,
            pack_families == 8,
            *audit_values,
            failure_rate <= FAILURE_RATE_LIMIT,
            schema_rate <= SCHEMA_FAILURE_RATE_LIMIT,
            checked,
            hardware["cycles_device"] == "OPTIX",
            gpu_count > 0,
            not cpu_fallback,
        )
    )
    if value["decision"] == "passed" and not expected_pass:
        if failure_rate > FAILURE_RATE_LIMIT:
            raise ReportValidationError("end-to-end failure rate exceeds 10%")
        if schema_rate > SCHEMA_FAILURE_RATE_LIMIT:
            raise ReportValidationError("schema failure rate exceeds 5%")
        raise ReportValidationError("gate decision is not supported by audits")
    if value["decision"] == "failed" and expected_pass:
        raise ReportValidationError("gate decision contradicts passing audits")
    if verified + failed_checksums <= 0:
        raise ReportValidationError("checksum audit must not be empty")
    if sum(counts.values()) != assets:
        raise ReportValidationError("source counts do not match quality assets")
    if sum(failures.values()) != assets:
        raise ReportValidationError(
            "failure categories do not match quality assets"
        )
    if verified + failed_checksums < assets:
        raise ReportValidationError("checksum coverage is incomplete")
    if schema_count > failures_count:
        raise ReportValidationError(
            "schema failures exceed end-to-end failures"
        )


def validate_gate_report(
    value: Mapping, *, expected_config_hash: str | None = None
) -> dict:
    required = {
        "schema_version",
        "report_type",
        "gate",
        "decision",
        "config_hash",
        "source_counts",
        "failure_categories",
        "resource_peaks",
        "throughput_quantiles",
        "byte_quantiles",
        "fp16_parity",
        "checksums",
        "split_overlap",
        "capacity",
        "handoff",
        "audits",
        "quality",
        "hardware",
    }
    value = _mapping(value, required, "gate report")
    if value["schema_version"] != REPORT_SCHEMA_VERSION:
        raise ReportValidationError("unsupported gate report schema")
    if value["report_type"] != GATE_REPORT_TYPE:
        raise ReportValidationError("invalid gate report type")
    if value["gate"] not in GATES:
        raise ReportValidationError("invalid gate report gate")
    if value["decision"] not in {"passed", "failed"}:
        raise ReportValidationError("invalid gate decision")
    config_hash = _sha256(value["config_hash"], "report config hash")
    if expected_config_hash is not None and config_hash != expected_config_hash:
        raise ReportValidationError("gate report config hash mismatch")
    _validate_gate_sections(value)
    return dict(value)


def build_gate_report(value: Mapping) -> dict:
    return validate_gate_report(value)


def _markdown(name: str, payload: Mapping) -> bytes:
    generated = datetime.now(timezone.utc).isoformat()
    lines = [f"# {name}", "", f"Generated: {generated}", ""]
    for key in sorted(payload):
        rendered = json.dumps(payload[key], sort_keys=True, allow_nan=False)
        lines.append(f"- {key}: `{rendered}`")
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_report(root: Path, name: str, payload: Mapping) -> tuple[Path, Path]:
    if (
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or "\\" in name
    ):
        raise ReportValidationError(f"unsafe report name: {name!r}")
    if not isinstance(payload, Mapping) or not payload:
        raise ReportValidationError("report payload must not be empty")
    if payload.get("report_type") == GATE_REPORT_TYPE:
        validate_gate_report(payload)
    try:
        json_payload = json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise ReportValidationError(f"report is not JSON-safe: {error}") from error
    root = Path(root)
    json_path = root / f"{name}.json"
    markdown_path = root / f"{name}.md"
    try:
        _regular_file_stat_nofollow(json_path, missing_ok=True)
        _regular_file_stat_nofollow(markdown_path, missing_ok=True)
        _atomic_write_bytes_nofollow(markdown_path, _markdown(name, payload))
        _atomic_write_bytes_nofollow(json_path, json_payload)
    except (InfrastructureError, OSError) as error:
        raise ReportValidationError(
            f"unsafe or unavailable report destination: {error}"
        ) from error
    return json_path, markdown_path
