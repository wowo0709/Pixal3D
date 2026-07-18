from __future__ import annotations

import csv
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .commands import ShardContext
from .config import PipelineConfig
from .orchestrator import (
    _atomic_write_bytes_nofollow,
    _read_regular_bytes_nofollow,
    _regular_file_size_nofollow,
)
from .runtime import (
    ArtifactValidationError,
    RuntimeReportBuilder,
    SafeRegistryStore,
    _safe_json,
)


GATES = frozenset(("smoke", "pilot", "production"))
REQUIRED_SEGMENT_COMMANDS = frozenset(
    ("stage_raw", "build_packs", "archive_raw", "cleanup_local")
)
MEASUREMENT_COLUMNS = (
    "sha256",
    "source",
    "shard_id",
    "outcome",
    "failure_category",
    "elapsed_seconds",
    "peak_local_bytes",
    "final_local_bytes",
    "final_data2_bytes",
    "final_data3_bytes",
)
FP16_COLUMNS = (
    "sha256",
    "family",
    "resolution",
    "fp16_abs_error",
    "coordinates_match",
    "fp16_finite",
    "decode_degradation_percent",
)


def allocate_bytes(size: int, included_assets: Sequence[str]) -> dict[str, int]:
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ArtifactValidationError("publication size must be non-negative")
    assets = tuple(sorted(included_assets))
    if len(assets) != len(set(assets)):
        raise ArtifactValidationError("publication membership is duplicated")
    if not assets:
        if size == 0:
            return {}
        raise ArtifactValidationError(
            "non-empty publication has no included asset for byte accounting"
        )
    quotient, remainder = divmod(size, len(assets))
    return {
        asset: quotient + (position < remainder)
        for position, asset in enumerate(assets)
    }


def select_complete_segments(
    records: Sequence[Mapping], expected_count: int
) -> tuple[tuple[Mapping, ...], ...]:
    if (
        not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count <= 0
    ):
        raise ArtifactValidationError("expected telemetry segment count must be positive")
    completed = []
    current = None
    for record in records:
        if not isinstance(record, Mapping):
            raise ArtifactValidationError("telemetry record must be an object")
        command = record.get("command")
        action = record.get("action")
        if not isinstance(command, str) or not isinstance(action, str):
            raise ArtifactValidationError("telemetry command identity is invalid")
        if command == "stage_raw" and action == "run":
            if current is None:
                current = []
            elif any(item.get("command") != "stage_raw" for item in current):
                current = []
            current.append(record)
            continue
        if current is None:
            continue
        current.append(record)
        if command == "cleanup_local" and action == "run":
            successful = {
                item["command"]
                for item in current
                if item.get("action") == "run"
            }
            if REQUIRED_SEGMENT_COMMANDS <= successful:
                completed.append(tuple(current))
            current = None
    if len(completed) < expected_count:
        raise ArtifactValidationError(
            "complete telemetry segments do not cover frozen batches"
        )
    return tuple(completed[-expected_count:])


def _csv_payload(columns: Sequence[str], rows: Sequence[Mapping]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=columns,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _timestamp(value, description: str) -> datetime:
    if not isinstance(value, str):
        raise ArtifactValidationError(f"{description} timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ArtifactValidationError(
            f"invalid {description} timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ArtifactValidationError(
            f"{description} timestamp must be timezone-aware"
        )
    return parsed.astimezone(timezone.utc)


class GateEvidenceCollector:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def _telemetry_records(self) -> list[dict]:
        path = (
            self.config.paths.data2_root
            / "control/telemetry/resources.jsonl"
        )
        try:
            payload = _read_regular_bytes_nofollow(path)
            lines = payload.decode("utf-8").splitlines()
            records = [json.loads(line) for line in lines if line]
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
        ) as error:
            raise ArtifactValidationError(
                f"invalid runtime telemetry: {path}: {error}"
            ) from error
        if not records or not all(isinstance(record, dict) for record in records):
            raise ArtifactValidationError("runtime telemetry is empty or invalid")
        return records

    def _ledger(self, gate: str, source: str, shard: str, batches) -> dict:
        control = self.config.paths.data2_root / "control"
        root = (
            control / "quality"
            if gate == "production"
            else control / "qualification" / gate / "quality"
        )
        path = root / source / f"{shard}.json"
        value = _safe_json(path, f"{gate} quality ledger")
        required = {
            "schema_version",
            "source",
            "shard_id",
            "gate",
            "batches",
            "entries",
            "quarantine",
            "family_exclusions",
        }
        expected_batches = {
            batch_id: assets
            for (batch_source, batch_shard, batch_id), assets in batches.items()
            if (batch_source, batch_shard) == (source, shard)
        }
        if (
            set(value) != required
            or value["schema_version"] != 3
            or value["source"] != source
            or value["shard_id"] != shard
            or value["gate"] != gate
            or not isinstance(value["batches"], Mapping)
            or set(value["batches"]) != set(expected_batches)
            or not isinstance(value["entries"], list)
            or not isinstance(value["quarantine"], Mapping)
            or not isinstance(value["family_exclusions"], Mapping)
        ):
            raise ArtifactValidationError(f"invalid quality ledger: {path}")
        for batch_id, assets in expected_batches.items():
            batch = value["batches"][batch_id]
            payload = "".join(f"{asset}\n" for asset in assets).encode("ascii")
            if (
                not isinstance(batch, Mapping)
                or set(batch) != {"instances_sha256", "admitted_prefix"}
                or batch["instances_sha256"] != sha256(payload).hexdigest()
                or batch["admitted_prefix"] != len(assets)
            ):
                raise ArtifactValidationError(
                    f"quality ledger batch is incomplete: {path}: {batch_id}"
                )
        all_assets = {
            asset for batch_assets in expected_batches.values() for asset in batch_assets
        }
        entries = {}
        identities = set()
        for entry in value["entries"]:
            if (
                not isinstance(entry, Mapping)
                or set(entry)
                != {"batch_id", "position", "asset_sha", "outcome"}
                or entry["batch_id"] not in expected_batches
                or entry["outcome"]
                not in {"completed", "failure", "schema_failure"}
                or not isinstance(entry["position"], int)
                or isinstance(entry["position"], bool)
                or entry["position"] < 0
                or entry["position"]
                >= len(expected_batches[entry["batch_id"]])
                or expected_batches[entry["batch_id"]][entry["position"]]
                != entry["asset_sha"]
                or entry["asset_sha"] in entries
                or (entry["batch_id"], entry["position"]) in identities
            ):
                raise ArtifactValidationError(f"invalid quality ledger entry: {path}")
            entries[entry["asset_sha"]] = entry["outcome"]
            identities.add((entry["batch_id"], entry["position"]))
        if not set(value["quarantine"]) <= all_assets:
            raise ArtifactValidationError(
                f"quality ledger quarantine is outside frozen scope: {path}"
            )
        return {"entries": entries, "quarantine": value["quarantine"]}

    @staticmethod
    def _included_assets(path: Path) -> tuple[str, ...]:
        manifest = _safe_json(
            path.with_suffix(".tar.manifest.json"),
            "validated publication manifest",
        )
        included = manifest.get("included_asset_sha256s")
        if not isinstance(included, list) or not all(
            isinstance(asset, str) for asset in included
        ):
            raise ArtifactValidationError(
                f"invalid publication membership: {path}"
            )
        return tuple(included)

    @staticmethod
    def _segment_measurements(segment, asset_count: int) -> tuple[float, int]:
        if asset_count <= 0:
            raise ArtifactValidationError("frozen batch is empty")
        timestamps = [
            _timestamp(record.get("timestamp"), "runtime telemetry")
            for record in segment
        ]
        if timestamps != sorted(timestamps):
            raise ArtifactValidationError("runtime telemetry is not chronological")
        elapsed = (timestamps[-1] - timestamps[0]).total_seconds()
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ArtifactValidationError("runtime telemetry elapsed time is invalid")
        local_values = []
        for record in segment:
            value = record.get("local_free_gib")
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ArtifactValidationError(
                    "runtime telemetry local capacity is invalid"
                )
            local_values.append(float(value))
        peak_local = math.ceil(
            (max(local_values) - min(local_values)) * 1024**3 / asset_count
        )
        return elapsed / asset_count, peak_local

    @staticmethod
    def _assert_cleaned(config, gate, source, shard, batch) -> None:
        context = ShardContext.from_config(
            config, source, shard, batch, gate=gate
        )
        for path in (
            context.download_root,
            context.work_root,
            context.output_root,
        ):
            if path.exists() or path.is_symlink():
                raise ArtifactValidationError(
                    f"local cleanup is incomplete: {path}"
                )

    def collect(self, gate: str) -> tuple[Path, Path, Path, Path]:
        if gate not in GATES:
            raise ArtifactValidationError(f"unknown gate: {gate}")
        if self.config.targets.latent_dtype != "float32":
            raise ArtifactValidationError(
                "float16 evidence requires real decoder parity measurements"
            )

        training = SafeRegistryStore(
            self.config.paths.data2_root / "control/assets.parquet",
            self.config,
        ).load()
        builder = RuntimeReportBuilder(self.config)
        scopes, frozen_assets, batches = builder._frozen_scopes(gate, training)
        packs, archives = builder._publications(gate, batches)

        data2_bytes = {asset: 0 for asset in frozen_assets}
        data3_bytes = {asset: 0 for asset in frozen_assets}
        common_membership = {}
        for pack in packs:
            key = (pack["source"], pack["shard_id"], pack["batch_id"])
            path = Path(pack["path"])
            included = self._included_assets(path)
            allocation = allocate_bytes(
                _regular_file_size_nofollow(path), included
            )
            for asset, size in allocation.items():
                data2_bytes[asset] += size
            if pack["family"] == "common":
                common_membership[key] = set(included)
        for archive in archives:
            key = (
                archive["source"],
                archive["shard_id"],
                archive["batch_id"],
            )
            path = Path(archive["path"])
            included = self._included_assets(path)
            if set(included) != common_membership.get(key):
                raise ArtifactValidationError(
                    f"raw/common publication membership mismatch: {key}"
                )
            for asset, size in allocate_bytes(
                _regular_file_size_nofollow(path), included
            ).items():
                data3_bytes[asset] += size

        telemetry_records = self._telemetry_records()
        measurements = []
        held_telemetry = []
        scope_pairs = {(scope["source"], scope["shard_id"]) for scope in scopes}
        for source, shard in sorted(scope_pairs):
            batch_keys = sorted(
                key for key in batches if key[:2] == (source, shard)
            )
            shard_records = []
            for record in telemetry_records:
                if record.get("shard_id") != shard:
                    continue
                if record.get("source", source) != source or record.get(
                    "gate", gate
                ) != gate:
                    raise ArtifactValidationError(
                        f"runtime telemetry scope conflict: {source}/{shard}"
                    )
                shard_records.append(record)
            segments = select_complete_segments(shard_records, len(batch_keys))
            ledger = self._ledger(gate, source, shard, batches)
            for key, segment in zip(batch_keys, segments, strict=True):
                _, _, batch = key
                batch_assets = batches[key]
                self._assert_cleaned(self.config, gate, source, shard, batch)
                elapsed, peak_local = self._segment_measurements(
                    segment, len(batch_assets)
                )
                common = common_membership.get(key)
                if common is None:
                    raise ArtifactValidationError(
                        f"common publication is missing: {key}"
                    )
                for asset in batch_assets:
                    quarantine = ledger["quarantine"].get(asset)
                    recorded_outcome = ledger["entries"].get(asset)
                    if quarantine is None:
                        outcome = recorded_outcome or "completed"
                        if outcome != "completed" or asset not in common:
                            raise ArtifactValidationError(
                                f"quality ledger conflicts with publication: {asset}"
                            )
                        failure_category = "none"
                    else:
                        category = quarantine.get("category")
                        if not isinstance(category, str) or not category:
                            raise ArtifactValidationError(
                                f"invalid quarantine category: {asset}"
                            )
                        outcome = (
                            "schema_failure"
                            if category == "schema_failure"
                            else "failure"
                        )
                        if (
                            recorded_outcome is not None
                            and recorded_outcome != outcome
                            or asset in common
                        ):
                            raise ArtifactValidationError(
                                f"quarantine conflicts with publication: {asset}"
                            )
                        failure_category = category
                    measurements.append(
                        {
                            "sha256": asset,
                            "source": source,
                            "shard_id": shard,
                            "outcome": outcome,
                            "failure_category": failure_category,
                            "elapsed_seconds": elapsed,
                            "peak_local_bytes": peak_local,
                            "final_local_bytes": 0,
                            "final_data2_bytes": data2_bytes[asset],
                            "final_data3_bytes": data3_bytes[asset],
                        }
                    )
                for record in segment:
                    held_telemetry.append(
                        dict(record, gate=gate, source=source, batch_id=batch)
                    )

        if {row["sha256"] for row in measurements} != set(frozen_assets):
            raise ArtifactValidationError(
                "derived measurements do not match the frozen scope"
            )
        measurements_payload = _csv_payload(MEASUREMENT_COLUMNS, measurements)
        fp16_payload = _csv_payload(FP16_COLUMNS, ())
        try:
            telemetry_payload = "".join(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
                for record in held_telemetry
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ArtifactValidationError(
                f"runtime telemetry is not serializable: {error}"
            ) from error

        evidence_root = (
            self.config.paths.data2_root / "control/report_evidence" / gate
        )
        measurements_path = evidence_root / "measurements.csv"
        fp16_path = evidence_root / "fp16.csv"
        telemetry_path = evidence_root / "telemetry.jsonl"
        for path, payload in (
            (measurements_path, measurements_payload),
            (fp16_path, fp16_payload),
            (telemetry_path, telemetry_payload),
        ):
            try:
                _atomic_write_bytes_nofollow(path, payload)
            except OSError as error:
                raise ArtifactValidationError(
                    f"cannot write gate evidence: {path}: {error}"
                ) from error

        created_at = datetime.now(timezone.utc).isoformat()
        manifest = {
            "schema_version": 2,
            "artifact_type": "gate_evidence_manifest",
            "config_hash": self.config.config_hash(),
            "gate": gate,
            "created_at": created_at,
            "artifacts": {
                "measurements_sha256": sha256(measurements_payload).hexdigest(),
                "fp16_sha256": sha256(fp16_payload).hexdigest(),
                "telemetry_sha256": sha256(telemetry_payload).hexdigest(),
            },
        }
        manifest_path = (
            self.config.paths.data2_root / "control/report_inputs" / f"{gate}.json"
        )
        try:
            manifest_payload = json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            _atomic_write_bytes_nofollow(manifest_path, manifest_payload)
        except (OSError, TypeError, ValueError) as error:
            raise ArtifactValidationError(
                f"cannot publish gate evidence manifest: {manifest_path}: {error}"
            ) from error
        return measurements_path, fp16_path, telemetry_path, manifest_path
