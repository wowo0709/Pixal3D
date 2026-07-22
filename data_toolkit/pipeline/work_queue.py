from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import uuid

from .orchestrator import _atomic_write_bytes_nofollow, _read_regular_bytes_nofollow


QUEUE_SCHEMA_VERSION = 1
LEASE_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class LeaseLostError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkUnit:
    source: str
    shard_id: str
    batch_id: str
    count: int

    def __post_init__(self) -> None:
        for name, value in (
            ("source", self.source),
            ("shard_id", self.shard_id),
            ("batch_id", self.batch_id),
        ):
            if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
                raise ValueError(f"invalid work unit {name}: {value!r}")
        if type(self.count) is not int or self.count <= 0:
            raise ValueError("work unit count must be positive")

    @property
    def unit_id(self) -> str:
        return f"{self.source}--{self.shard_id}--{self.batch_id}"


@dataclass(frozen=True)
class WorkLease:
    unit: WorkUnit
    node_id: str
    token: str
    attempt: int
    claimed_at: datetime
    heartbeat_at: datetime
    stage: str


class ProductionWorkQueue:
    def __init__(
        self,
        root: Path,
        *,
        lease_timeout: timedelta,
        max_attempts: int = 3,
    ):
        self.root = Path(root)
        if not isinstance(lease_timeout, timedelta) or lease_timeout <= timedelta(0):
            raise ValueError("lease timeout must be positive")
        if type(max_attempts) is not int or max_attempts <= 0:
            raise ValueError("maximum attempts must be positive")
        self.lease_timeout = lease_timeout
        self.max_attempts = max_attempts
        self.manifest_path = self.root / "units.json"
        self.leases_root = self.root / "leases"
        self.history_root = self.root / "history"
        self.completed_root = self.root / "completed"
        self.failed_root = self.root / "failed"

    def initialize(
        self,
        config_hash: str,
        units: tuple[WorkUnit, ...],
        *,
        now: datetime,
    ) -> None:
        _config_hash(config_hash)
        _aware(now)
        values = tuple(units)
        if not values or len({unit.unit_id for unit in values}) != len(values):
            raise ValueError("production work units must be non-empty and unique")
        manifest = {
            "schema_version": QUEUE_SCHEMA_VERSION,
            "config_hash": config_hash,
            "created_at": _timestamp(now),
            "units": [asdict(unit) for unit in values],
        }
        existing = _read_json(self.manifest_path, missing_ok=True)
        if existing is not None:
            comparable = dict(existing, created_at=manifest["created_at"])
            if comparable != manifest:
                raise ValueError("work queue already holds a different production scope")
            return
        for path in (
            self.root,
            self.leases_root,
            self.history_root,
            self.completed_root,
            self.failed_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        _write_json(self.manifest_path, manifest)

    def units(self) -> tuple[WorkUnit, ...]:
        manifest = _read_json(self.manifest_path)
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema_version",
            "config_hash",
            "created_at",
            "units",
        }:
            raise ValueError("invalid production work queue manifest")
        if manifest["schema_version"] != QUEUE_SCHEMA_VERSION:
            raise ValueError("unsupported production work queue manifest")
        _config_hash(manifest["config_hash"])
        datetime.fromisoformat(manifest["created_at"])
        result = tuple(WorkUnit(**value) for value in manifest["units"])
        if not result or len({unit.unit_id for unit in result}) != len(result):
            raise ValueError("invalid production work queue units")
        return result

    def claim(
        self,
        node_id: str,
        *,
        now: datetime,
        token: str | None = None,
    ) -> WorkLease | None:
        _identifier(node_id, "node id")
        _aware(now)
        requested_token = token or uuid.uuid4().hex
        _identifier(requested_token, "lease token")
        for unit in self.units():
            if self._terminal(unit):
                continue
            lease_dir = self._lease_dir(unit)
            attempt = self._next_attempt(unit)
            if attempt > self.max_attempts:
                continue
            try:
                lease_dir.mkdir()
            except FileExistsError:
                current = self._read_lease(unit)
                if current is None:
                    continue
                if now - current.heartbeat_at <= self.lease_timeout:
                    continue
                stale_path = self.history_root / (
                    f"{unit.unit_id}.{current.token}.stale"
                )
                try:
                    lease_dir.rename(stale_path)
                except FileNotFoundError:
                    continue
                attempt = max(current.attempt + 1, self._next_attempt(unit))
                if attempt > self.max_attempts:
                    self._mark_failed(
                        unit,
                        node_id=current.node_id,
                        token=current.token,
                        attempt=current.attempt,
                        reason="lease-timeout",
                        now=now,
                    )
                    continue
                try:
                    lease_dir.mkdir()
                except FileExistsError:
                    continue
            lease = WorkLease(
                unit=unit,
                node_id=node_id,
                token=requested_token,
                attempt=attempt,
                claimed_at=now,
                heartbeat_at=now,
                stage="claimed",
            )
            try:
                self._write_lease(lease)
            except BaseException:
                try:
                    lease_dir.rmdir()
                except OSError:
                    pass
                raise
            return lease
        return None

    def heartbeat(
        self,
        lease: WorkLease,
        *,
        stage: str,
        now: datetime,
    ) -> WorkLease:
        _identifier(stage, "stage")
        _aware(now)
        self.assert_owned(lease)
        renewed = WorkLease(
            unit=lease.unit,
            node_id=lease.node_id,
            token=lease.token,
            attempt=lease.attempt,
            claimed_at=lease.claimed_at,
            heartbeat_at=now,
            stage=stage,
        )
        self._write_lease(renewed)
        self.assert_owned(renewed)
        return renewed

    def assert_owned(self, lease: WorkLease) -> None:
        current = self._read_lease(lease.unit)
        if (
            current is None
            or current.node_id != lease.node_id
            or current.token != lease.token
        ):
            raise LeaseLostError(f"work lease was lost: {lease.unit.unit_id}")

    def complete(self, lease: WorkLease, *, now: datetime) -> None:
        _aware(now)
        self.assert_owned(lease)
        _write_json(
            self.completed_root / f"{lease.unit.unit_id}.json",
            {
                "schema_version": QUEUE_SCHEMA_VERSION,
                "unit_id": lease.unit.unit_id,
                "node_id": lease.node_id,
                "token": lease.token,
                "attempt": lease.attempt,
                "completed_at": _timestamp(now),
            },
        )
        self.assert_owned(lease)
        try:
            self._lease_dir(lease.unit).rename(
                self.history_root
                / f"{lease.unit.unit_id}.{lease.token}.completed"
            )
        except FileNotFoundError as error:
            raise LeaseLostError(
                f"work lease was lost: {lease.unit.unit_id}"
            ) from error

    def release(self, lease: WorkLease, *, reason: str, now: datetime) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("release reason must be non-empty")
        _aware(now)
        self.assert_owned(lease)
        if lease.attempt >= self.max_attempts:
            self._mark_failed(
                lease.unit,
                node_id=lease.node_id,
                token=lease.token,
                attempt=lease.attempt,
                reason=reason,
                now=now,
            )
        self.assert_owned(lease)
        try:
            self._lease_dir(lease.unit).rename(
                self.history_root / f"{lease.unit.unit_id}.{lease.token}.released"
            )
        except FileNotFoundError as error:
            raise LeaseLostError(
                f"work lease was lost: {lease.unit.unit_id}"
            ) from error

    def adopt_completed(
        self,
        unit: WorkUnit,
        *,
        now: datetime,
        node_id: str,
    ) -> None:
        _aware(now)
        _identifier(node_id, "node id")
        known_units = {candidate.unit_id: candidate for candidate in self.units()}
        if known_units.get(unit.unit_id) != unit:
            raise ValueError(f"unknown production work unit: {unit.unit_id}")
        if self._read_lease(unit) is not None:
            raise ValueError(f"cannot adopt leased work unit: {unit.unit_id}")
        if (self.failed_root / f"{unit.unit_id}.json").is_file():
            raise ValueError(f"cannot adopt failed work unit: {unit.unit_id}")
        completion_path = self._completion_path(unit)
        if completion_path.is_file():
            return
        _write_json(
            completion_path,
            {
                "schema_version": QUEUE_SCHEMA_VERSION,
                "unit_id": unit.unit_id,
                "node_id": node_id,
                "token": "adopted",
                "attempt": 0,
                "completed_at": _timestamp(now),
            },
        )

    def status(self, *, now: datetime) -> dict[str, int]:
        _aware(now)
        counts = {
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "stale": 0,
            "total": 0,
        }
        for unit in self.units():
            counts["total"] += 1
            if self._completion_path(unit).is_file():
                counts["completed"] += 1
            elif (self.failed_root / f"{unit.unit_id}.json").is_file():
                counts["failed"] += 1
            else:
                lease = self._read_lease(unit)
                if lease is None:
                    counts["pending"] += 1
                elif now - lease.heartbeat_at > self.lease_timeout:
                    counts["stale"] += 1
                else:
                    counts["running"] += 1
        return counts

    def _terminal(self, unit: WorkUnit) -> bool:
        return self._completion_path(unit).is_file() or (
            self.failed_root / f"{unit.unit_id}.json"
        ).is_file()

    def _completion_path(self, unit: WorkUnit) -> Path:
        return self.completed_root / f"{unit.unit_id}.json"

    def _lease_dir(self, unit: WorkUnit) -> Path:
        return self.leases_root / unit.unit_id

    def _next_attempt(self, unit: WorkUnit) -> int:
        highest = 0
        prefix = f"{unit.unit_id}."
        if self.history_root.is_dir():
            for path in self.history_root.iterdir():
                if not path.name.startswith(prefix) or not path.is_dir():
                    continue
                value = _read_json(path / "owner.json", missing_ok=True)
                if isinstance(value, dict):
                    attempt = value.get("attempt")
                    if type(attempt) is int and attempt > highest:
                        highest = attempt
        return highest + 1

    def _mark_failed(
        self,
        unit: WorkUnit,
        *,
        node_id: str,
        token: str,
        attempt: int,
        reason: str,
        now: datetime,
    ) -> None:
        _write_json(
            self.failed_root / f"{unit.unit_id}.json",
            {
                "schema_version": QUEUE_SCHEMA_VERSION,
                "unit_id": unit.unit_id,
                "node_id": node_id,
                "token": token,
                "attempt": attempt,
                "reason": reason,
                "failed_at": _timestamp(now),
            },
        )

    def _read_lease(self, unit: WorkUnit) -> WorkLease | None:
        value = _read_json(self._lease_dir(unit) / "owner.json", missing_ok=True)
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "unit",
            "node_id",
            "token",
            "attempt",
            "claimed_at",
            "heartbeat_at",
            "stage",
        }:
            raise ValueError("invalid production work lease")
        if value["schema_version"] != LEASE_SCHEMA_VERSION:
            raise ValueError("unsupported production work lease")
        return WorkLease(
            unit=WorkUnit(**value["unit"]),
            node_id=value["node_id"],
            token=value["token"],
            attempt=value["attempt"],
            claimed_at=datetime.fromisoformat(value["claimed_at"]),
            heartbeat_at=datetime.fromisoformat(value["heartbeat_at"]),
            stage=value["stage"],
        )

    def _write_lease(self, lease: WorkLease) -> None:
        value = {
            "schema_version": LEASE_SCHEMA_VERSION,
            "unit": asdict(lease.unit),
            "node_id": lease.node_id,
            "token": lease.token,
            "attempt": lease.attempt,
            "claimed_at": _timestamp(lease.claimed_at),
            "heartbeat_at": _timestamp(lease.heartbeat_at),
            "stage": lease.stage,
        }
        _write_json_existing_directory(
            self._lease_dir(lease.unit), "owner.json", value
        )


def _read_json(path: Path, *, missing_ok: bool = False):
    payload = _read_regular_bytes_nofollow(path, missing_ok=missing_ok)
    if payload is None:
        return None
    return json.loads(payload)


def _write_json(path: Path, value: dict) -> None:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    _atomic_write_bytes_nofollow(path, payload)


def _write_json_existing_directory(
    directory: Path, name: str, value: dict
) -> None:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    directory_fd = os.open(
        directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    temporary = f".{name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    file_fd = None
    try:
        file_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_fd,
        )
        remaining = memoryview(payload)
        while remaining:
            written = os.write(file_fd, remaining)
            if written <= 0:
                raise OSError("failed to write queue metadata")
            remaining = remaining[written:]
        os.fsync(file_fd)
        os.close(file_fd)
        file_fd = None
        os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        if file_fd is not None:
            os.close(file_fd)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)


def _aware(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("queue timestamps must be timezone-aware")
    return value


def _timestamp(value: datetime) -> str:
    return _aware(value).astimezone(timezone.utc).isoformat()


def _identifier(value: str, description: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"invalid {description}: {value!r}")
    return value


def _config_hash(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("invalid queue config hash")
    return value
