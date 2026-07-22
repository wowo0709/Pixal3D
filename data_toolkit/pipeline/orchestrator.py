from collections import deque
from concurrent.futures import ThreadPoolExecutor
import csv
import ctypes
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import errno
import fcntl
from hashlib import sha256
import io
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import pickle
import re
import select
import shutil
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable, Mapping, Protocol, Sequence
import zipfile

from .atomic_io import atomic_copy
from .commands import (
    CommandSpec,
    ShardContext,
    WorkerProfile,
    build_preprocessing_dag,
    choose_worker_profile,
    expand_ranked,
    select_render_workers,
)
from .config import PipelineConfig
from .packing import (
    PACK_FAMILIES,
    build_pack,
    publish_pack,
    verify_pack,
)
from .parallelism import NodeResourceBroker
from .registry import RegistryStore
from .resources import (
    ResourceAccountingError,
    ResourceAction,
    ResourceDecision,
    ResourceLimitExceeded,
)
from .scheduler import (
    ChunkContext,
    Lane,
    ParallelChunkScheduler,
    StageSpec,
    choose_chunk_assets,
    promote_chunk_outputs,
)
from .validation import (
    ValidationError,
    validate_render_dir,
    validate_scale,
    validate_sparse_latent,
    validate_ss_latent,
)


CHECKPOINT_SCHEMA_VERSION = 3
QUALITY_LEDGER_SCHEMA_VERSION = 3
MAX_COMMAND_ATTEMPTS = 3
QUALITY_WINDOW_SIZE = 500
QUALITY_OUTCOMES = {"completed", "failure", "schema_failure"}
PATH_VALIDATION_ERRNOS = {errno.ENOENT, errno.ENOTDIR, errno.ELOOP}


def family_dependencies(
    config: PipelineConfig,
) -> Mapping[str, frozenset[str]]:
    highest_resolution = max(config.targets.resolutions)
    dependencies = {
        "common": frozenset(),
        f"SS-{config.targets.ss_resolution}": frozenset(
            {f"shape-{highest_resolution}"}
        ),
    }
    for resolution in config.targets.resolutions:
        dependencies[f"shape-{resolution}"] = frozenset()
        dependencies[f"PBR-{resolution}"] = frozenset(
            {f"shape-{resolution}"}
        )
    return dependencies


def command_families(
    config: PipelineConfig, command_name: str
) -> tuple[str, ...]:
    if command_name == "dump_pbr":
        return tuple(
            f"PBR-{resolution}"
            for resolution in config.targets.resolutions
        )
    for prefix, family_prefix in (
        ("voxelize_pbr_", "PBR"),
        ("encode_pbr_", "PBR"),
        ("dual_grid_", "shape"),
        ("encode_shape_", "shape"),
    ):
        if command_name.startswith(prefix):
            return (f"{family_prefix}-{command_name.removeprefix(prefix)}",)
    if command_name == f"encode_ss_{config.targets.ss_resolution}":
        return (f"SS-{config.targets.ss_resolution}",)
    return ()


_SUPERVISOR_PROGRAM = r"""
import ctypes
import os
import signal
import subprocess
import sys
import time

libc = ctypes.CDLL(None, use_errno=True)
PR_SET_CHILD_SUBREAPER = 36
if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), "prctl(PR_SET_CHILD_SUBREAPER) failed")

leader = os.getpid()
worker_status = None
terminating = False

def pause_group(*_):
    os.killpg(leader, signal.SIGSTOP)

def resume_group(*_):
    signal.signal(signal.SIGCONT, signal.SIG_IGN)
    try:
        os.killpg(leader, signal.SIGCONT)
    finally:
        signal.signal(signal.SIGCONT, resume_group)

def terminate_group(*_):
    global terminating
    terminating = True
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.killpg(leader, signal.SIGTERM)

def kill_group(*_):
    os.killpg(leader, signal.SIGKILL)

signal.signal(signal.SIGUSR1, pause_group)
signal.signal(signal.SIGCONT, resume_group)
signal.signal(signal.SIGTERM, terminate_group)
signal.signal(signal.SIGUSR2, kill_group)

ready_fd = int(os.environ.pop("PIXAL3D_SUPERVISOR_READY_FD"))
release_fd = int(os.environ.pop("PIXAL3D_SUPERVISOR_RELEASE_FD"))
try:
    os.write(ready_fd, b"1")
finally:
    os.close(ready_fd)
try:
    release = os.read(release_fd, 1)
finally:
    os.close(release_fd)
if release != b"1":
    raise SystemExit(125)

worker = subprocess.Popen(sys.argv[1:])

while True:
    no_children = False
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            no_children = True
            break
        if pid == 0:
            break
        if pid == worker.pid:
            worker_status = os.waitstatus_to_exitcode(status)

    if not terminating and worker_status is not None and no_children:
        raise SystemExit(worker_status)
    time.sleep(0.05)
"""


def _linux_syscall(number: int, *arguments: int) -> int:
    result = ctypes.CDLL(None, use_errno=True).syscall(number, *arguments)
    if result < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return int(result)


def _wait_for_supervisor_ready(file_descriptor: int, timeout: float) -> None:
    poller = select.poll()
    poller.register(file_descriptor, select.POLLIN | select.POLLHUP)
    if not poller.poll(math.ceil(timeout * 1000)):
        raise TimeoutError("supervisor READY acknowledgement timed out")
    if os.read(file_descriptor, 1) != b"1":
        raise InfrastructureError("invalid supervisor READY acknowledgement")


class _LinuxProcessSupervisor:
    """Pidfd-controlled owner of one external rank and all descendants."""

    _PIDFD_SEND_SIGNAL = 424
    _PIDFD_OPEN = 434
    _STARTUP_TIMEOUT_SECONDS = 5.0
    _SIGNALS = {
        "pause": signal.SIGUSR1,
        "resume": signal.SIGCONT,
        "terminate": signal.SIGTERM,
        "kill": signal.SIGUSR2,
    }

    def __init__(self, process, pidfd: int):
        self.process = process
        self.pid = process.pid
        self.pidfd = pidfd

    @classmethod
    def launch(cls, process_factory, argv, environment):
        release_read_fd, release_write_fd = os.pipe2(os.O_CLOEXEC)
        ready_read_fd, ready_write_fd = os.pipe2(os.O_CLOEXEC)
        supervisor_environment = dict(environment)
        supervisor_environment["PIXAL3D_SUPERVISOR_READY_FD"] = str(
            ready_write_fd
        )
        supervisor_environment["PIXAL3D_SUPERVISOR_RELEASE_FD"] = str(
            release_read_fd
        )
        process = None
        pidfd = -1
        released = False
        try:
            process = process_factory(
                (sys.executable, "-c", _SUPERVISOR_PROGRAM, *argv),
                env=supervisor_environment,
                start_new_session=True,
                pass_fds=(release_read_fd, ready_write_fd),
            )
            os.close(release_read_fd)
            release_read_fd = -1
            os.close(ready_write_fd)
            ready_write_fd = -1
            pidfd = _linux_syscall(cls._PIDFD_OPEN, process.pid, 0)
            _wait_for_supervisor_ready(
                ready_read_fd, cls._STARTUP_TIMEOUT_SECONDS
            )
            released = True
            if os.write(release_write_fd, b"1") != 1:
                raise InfrastructureError("cannot release supervisor worker")
            supervisor = cls(process, pidfd)
            pidfd = -1
            return supervisor
        except BaseException as launch_error:
            if release_write_fd >= 0:
                os.close(release_write_fd)
                release_write_fd = -1
            if process is not None:
                exited = False
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                except BaseException as cleanup_error:
                    notes = list(getattr(launch_error, "__notes__", ()))
                    notes.append(
                        "supervisor launch initial reap failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                    launch_error.__notes__ = notes
                else:
                    exited = True
                if not exited and pidfd >= 0:
                    try:
                        _linux_syscall(
                            cls._PIDFD_SEND_SIGNAL,
                            pidfd,
                            int(signal.SIGUSR2 if released else signal.SIGKILL),
                            0,
                            0,
                        )
                    except BaseException as cleanup_error:
                        notes = list(getattr(launch_error, "__notes__", ()))
                        notes.append(
                            "supervisor launch cleanup failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                        launch_error.__notes__ = notes
                    try:
                        process.wait(timeout=1)
                    except BaseException as cleanup_error:
                        notes = list(getattr(launch_error, "__notes__", ()))
                        notes.append(
                            "supervisor launch final reap failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                        launch_error.__notes__ = notes
            raise launch_error
        finally:
            if pidfd >= 0:
                os.close(pidfd)
            for file_descriptor in (
                release_read_fd,
                release_write_fd,
                ready_read_fd,
                ready_write_fd,
            ):
                if file_descriptor >= 0:
                    os.close(file_descriptor)

    def poll(self):
        return self.process.poll()

    def wait(self, timeout: float):
        try:
            return self.process.wait(timeout=timeout)
        finally:
            if self.pidfd >= 0 and self.process.poll() is not None:
                os.close(self.pidfd)
                self.pidfd = -1

    def send_control(self, action: str) -> None:
        try:
            sent_signal = self._SIGNALS[action]
        except KeyError as error:
            raise ValueError(f"unknown supervisor control: {action}") from error
        _linux_syscall(
            self._PIDFD_SEND_SIGNAL,
            self.pidfd,
            int(sent_signal),
            0,
            0,
        )

    def close(self) -> None:
        if self.pidfd >= 0:
            os.close(self.pidfd)
            self.pidfd = -1


class CheckpointError(RuntimeError):
    pass


class InfrastructureError(RuntimeError):
    pass


class ProcessGroupSafetyError(InfrastructureError):
    pass


class OutputValidationError(ValidationError):
    pass


class IntegrationProviderRequired(InfrastructureError):
    pass


class EscalationCategory(str, Enum):
    INFRASTRUCTURE = "infrastructure"
    RESOURCE = "resource"
    DATA_QUALITY = "data_quality"
    COMMAND_FAILURE = "command_failure"


@dataclass
class PipelineCheckpoint:
    shard_id: str
    completed_commands: list[str] = field(default_factory=list)
    attempts: dict[str, int] = field(default_factory=dict)
    active_attempt: dict[str, object] | None = None
    quality_outcomes: dict[str, str] = field(default_factory=dict)
    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    gate: str = "production"

    def complete(self, command: str) -> None:
        if command not in self.completed_commands:
            self.completed_commands.append(command)


@dataclass(frozen=True)
class EscalationReport:
    source: str
    shard_id: str
    command: str
    category: EscalationCategory
    reason: str
    recent_telemetry: tuple[dict, ...]
    completed_counts: dict[str, int]
    safe_resume_command: str
    recovery_choices: tuple[str, ...]
    created_at: str
    persistence_errors: tuple[str, ...]
    gate: str = "production"


class PipelineStopped(RuntimeError):
    def __init__(self, report: EscalationReport, exit_code: int):
        super().__init__(report.reason)
        self.report = report
        self.exit_code = exit_code


class PilotReader(Protocol):
    def p95_peak_local_bytes(self, source: str) -> int:
        """Return a validated, positive pilot p95 for one asset."""


class GateSizingReader(Protocol):
    def p95_peak_local_bytes_for_gate(self, source: str, gate: str) -> int:
        """Return a validated source p95 for the requested admission gate."""


class RawReferenceCounter(Protocol):
    def pending_references(
        self,
        source: str,
        raw_relative_path: str,
        *,
        excluding_shard_id: str,
        excluding_batch_id: str,
        gate: str = "production",
    ) -> int:
        """Return unarchived references excluding the just-archived batch."""


class ProjectAccounting(Protocol):
    def record_registry_delta(self, path: Path, delta_bytes: int) -> None:
        """Record an atomic publication or deletion delta."""

    def reconcile_at_shard_boundary(self) -> tuple[int, int]:
        """Reconcile cached totals against both project roots."""


class _MissingPilotReader:
    def p95_peak_local_bytes(self, source: str) -> int:
        raise IntegrationProviderRequired(
            "pilot reader is required; wire a validated pilot p95 artifact"
        )


class _MissingReferenceCounter:
    def pending_references(
        self,
        source: str,
        raw_relative_path: str,
        *,
        excluding_shard_id: str,
        excluding_batch_id: str,
        gate: str = "production",
    ) -> int:
        raise IntegrationProviderRequired(
            "raw reference counter is required before data2 deletion"
        )


class _MissingProjectAccounting:
    def record_registry_delta(self, path: Path, delta_bytes: int) -> None:
        raise IntegrationProviderRequired(
            "Task 8 project accounting is required before publication"
        )

    def reconcile_at_shard_boundary(self) -> tuple[int, int]:
        raise IntegrationProviderRequired(
            "Task 8 project accounting is required before execution"
        )


class _MissingResourceGuard:
    def _fail(self):
        raise IntegrationProviderRequired(
            "resource guard is required for mutating pipeline operations"
        )

    def wait_for_admission(self, shard_id: str, command: str):
        self._fail()

    def check(self, shard_id: str, command: str):
        self._fail()

    def last_five_minutes(self) -> tuple[dict, ...]:
        return ()


class RollingQualityGate:
    def __init__(self, window_size: int = QUALITY_WINDOW_SIZE):
        if window_size != QUALITY_WINDOW_SIZE:
            raise ValueError(f"quality window must be {QUALITY_WINDOW_SIZE}")
        self._outcomes: deque[tuple[str, bool, bool]] = deque(
            maxlen=window_size
        )
        self._by_asset: dict[str, tuple[bool, bool]] = {}

    def record(
        self,
        *,
        asset_sha: str,
        succeeded: bool,
        schema_failure: bool,
    ) -> bool:
        asset_sha = _validated_asset_sha(asset_sha)
        if not isinstance(succeeded, bool) or not isinstance(
            schema_failure, bool
        ):
            raise TypeError("quality outcomes must be booleans")
        outcome = (succeeded, schema_failure)
        existing = self._by_asset.get(asset_sha)
        if existing is not None:
            if existing != outcome:
                raise InfrastructureError(
                    f"conflicting quality outcome for asset: {asset_sha}"
                )
            return False
        if len(self._outcomes) == self._outcomes.maxlen:
            expired_asset, _, _ = self._outcomes[0]
            self._by_asset.pop(expired_asset, None)
        self._by_asset[asset_sha] = outcome
        self._outcomes.append((asset_sha, succeeded, schema_failure))
        return True

    def restore_entries(
        self, entries: Sequence[tuple[str, str]]
    ) -> None:
        if len(entries) > QUALITY_WINDOW_SIZE:
            raise InfrastructureError("quality history exceeds rolling window")
        self._outcomes.clear()
        self._by_asset.clear()
        for asset_sha, outcome in entries:
            if outcome not in QUALITY_OUTCOMES:
                raise InfrastructureError(
                    f"invalid quality history outcome: {outcome!r}"
                )
            self.record(
                asset_sha=asset_sha,
                succeeded=outcome == "completed",
                schema_failure=outcome == "schema_failure",
            )

    def restore(
        self,
        outcomes: Mapping[str, str],
        frozen_assets: Sequence[str],
    ) -> int:
        self.restore_entries(())
        ordered_assets = tuple(
            _validated_asset_sha(asset_sha) for asset_sha in frozen_assets
        )
        if len(ordered_assets) != len(set(ordered_assets)):
            raise InfrastructureError("duplicate frozen quality asset")
        unknown = set(outcomes) - set(ordered_assets)
        if unknown:
            raise InfrastructureError(
                f"quality outcomes contain non-frozen assets: {sorted(unknown)}"
            )
        prefix_length = 0
        for asset_sha in ordered_assets:
            outcome = outcomes.get(asset_sha)
            if outcome is None:
                break
            self.record(
                asset_sha=asset_sha,
                succeeded=outcome == "completed",
                schema_failure=outcome == "schema_failure",
            )
            prefix_length += 1
        return prefix_length

    @property
    def count(self) -> int:
        return len(self._outcomes)

    def violation_reason(self) -> str | None:
        if len(self._outcomes) < QUALITY_WINDOW_SIZE:
            return None
        failures = sum(
            not succeeded for _, succeeded, _ in self._outcomes
        )
        schema_failures = sum(schema for _, _, schema in self._outcomes)
        reasons = []
        if failures * 100 > 10 * len(self._outcomes):
            reasons.append(
                "end-to-end failures exceed 10% "
                f"({failures}/{len(self._outcomes)})"
            )
        if schema_failures * 100 > 5 * len(self._outcomes):
            reasons.append(
                "schema failures exceed 5% "
                f"({schema_failures}/{len(self._outcomes)})"
            )
        return "; ".join(reasons) or None


def _validated_asset_sha(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"invalid asset SHA-256: {value!r}")
    return value


def plan_work_batches(
    asset_sha256s: tuple[str, ...],
    local_usable_bytes: int,
    p95_peak_bytes: int,
    shard_size: int,
    max_batch_assets: int | None = None,
) -> tuple[tuple[str, ...], ...]:
    for name, value in (
        ("local_usable_bytes", local_usable_bytes),
        ("p95_peak_bytes", p95_peak_bytes),
        ("shard_size", shard_size),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if local_usable_bytes < 0:
        raise ValueError("local_usable_bytes must be non-negative")
    if p95_peak_bytes <= 0:
        raise ValueError("p95_peak_bytes must be positive")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if max_batch_assets is not None and (
        not isinstance(max_batch_assets, int)
        or isinstance(max_batch_assets, bool)
        or max_batch_assets <= 0
    ):
        raise ValueError("max_batch_assets must be positive")

    ordered = tuple(sorted(_validated_asset_sha(item) for item in asset_sha256s))
    if len(ordered) != len(set(ordered)):
        raise ValueError("duplicate asset SHA-256")
    if not ordered:
        return ()

    per_asset = (p95_peak_bytes * 5 + 3) // 4
    budget = local_usable_bytes * 4 // 5
    batch_size = min(
        shard_size,
        budget // per_asset,
        max_batch_assets if max_batch_assets is not None else shard_size,
    )
    if batch_size < 1:
        raise ValueError("local budget cannot fit one p95 asset")
    return tuple(
        ordered[index : index + batch_size]
        for index in range(0, len(ordered), batch_size)
    )


INTERNAL_COMMANDS = {
    "stage_raw": ("internal:stage_raw",),
    "cleanup_voxels_256": ("internal:cleanup_voxels", "256"),
    "cleanup_voxels_512": ("internal:cleanup_voxels", "512"),
    "cleanup_voxels_1024": ("internal:cleanup_voxels", "1024"),
    "validate_outputs": ("internal:validate_outputs",),
    "build_packs": ("internal:build_packs",),
    "archive_raw": ("internal:archive_raw",),
    "cleanup_local": ("internal:cleanup_local",),
}


def _required_checkpoint_dict(value, path: Path) -> PipelineCheckpoint:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "shard_id",
        "completed_commands",
        "attempts",
        "active_attempt",
        "quality_outcomes",
        "gate",
    }:
        raise CheckpointError(f"invalid checkpoint schema: {path}")
    if value["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(f"unsupported checkpoint schema: {path}")
    shard_id = value["shard_id"]
    completed = value["completed_commands"]
    attempts = value["attempts"]
    active_attempt = value["active_attempt"]
    quality_outcomes = value["quality_outcomes"]
    gate = value["gate"]
    if not isinstance(shard_id, str) or not shard_id:
        raise CheckpointError(f"invalid checkpoint shard identity: {path}")
    if gate not in {"smoke", "pilot", "production"}:
        raise CheckpointError(f"invalid checkpoint gate identity: {path}")
    if (
        not isinstance(completed, list)
        or not all(isinstance(item, str) and item for item in completed)
        or len(completed) != len(set(completed))
    ):
        raise CheckpointError(f"invalid checkpoint completed commands: {path}")
    if not isinstance(attempts, dict):
        raise CheckpointError(f"invalid checkpoint attempts: {path}")
    for command, count in attempts.items():
        if (
            not isinstance(command, str)
            or not command
            or not isinstance(count, int)
            or isinstance(count, bool)
            or not 0 <= count <= MAX_COMMAND_ATTEMPTS
        ):
            raise CheckpointError(f"invalid checkpoint attempts: {path}")
    if active_attempt is not None:
        if (
            not isinstance(active_attempt, dict)
            or set(active_attempt) != {"command", "attempt"}
            or not isinstance(active_attempt["command"], str)
            or not active_attempt["command"]
            or not isinstance(active_attempt["attempt"], int)
            or isinstance(active_attempt["attempt"], bool)
            or attempts.get(active_attempt["command"])
            != active_attempt["attempt"]
        ):
            raise CheckpointError(f"invalid active checkpoint attempt: {path}")
    if not isinstance(quality_outcomes, dict):
        raise CheckpointError(f"invalid checkpoint quality outcomes: {path}")
    try:
        for asset_sha, outcome in quality_outcomes.items():
            _validated_asset_sha(asset_sha)
            if outcome not in QUALITY_OUTCOMES:
                raise ValueError("invalid terminal outcome")
    except (TypeError, ValueError) as error:
        raise CheckpointError(
            f"invalid checkpoint quality outcomes: {path}"
        ) from error
    return PipelineCheckpoint(
        shard_id=shard_id,
        completed_commands=list(completed),
        attempts=dict(attempts),
        active_attempt=(
            dict(active_attempt) if active_attempt is not None else None
        ),
        quality_outcomes=dict(quality_outcomes),
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        gate=gate,
    )


def _empty_quality_ledger(context: ShardContext) -> dict[str, object]:
    return {
        "schema_version": QUALITY_LEDGER_SCHEMA_VERSION,
        "source": context.source,
        "shard_id": context.shard_id,
        "gate": context.gate,
        "batches": {},
        "entries": [],
        "quarantine": {},
        "family_exclusions": {},
    }


def _load_quality_ledger(path: Path, context: ShardContext) -> dict[str, object]:
    payload = _read_regular_bytes_nofollow(path, missing_ok=True)
    if payload is None:
        return _empty_quality_ledger(context)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointError(f"invalid quality ledger JSON: {path}") from error
    if not isinstance(value, dict):
        raise CheckpointError(f"invalid quality ledger schema: {path}")
    schema_version = value.get("schema_version")
    common_fields = {
        "schema_version",
        "source",
        "shard_id",
        "gate",
        "batches",
        "entries",
    }
    legacy_fields = (common_fields, common_fields | {"quarantine"})
    current_fields = common_fields | {"quarantine", "family_exclusions"}
    if (
        schema_version == 2
        and set(value) not in legacy_fields
        or schema_version == QUALITY_LEDGER_SCHEMA_VERSION
        and set(value) != current_fields
    ):
        raise CheckpointError(f"invalid quality ledger schema: {path}")
    if schema_version not in {2, QUALITY_LEDGER_SCHEMA_VERSION}:
        raise CheckpointError(f"unsupported quality ledger schema: {path}")
    if schema_version == 2:
        value = dict(value)
        value["schema_version"] = QUALITY_LEDGER_SCHEMA_VERSION
        value.setdefault("quarantine", {})
        value["family_exclusions"] = {}
    if (
        value["source"] != context.source
        or value["shard_id"] != context.shard_id
        or value["gate"] != context.gate
    ):
        raise CheckpointError(f"quality ledger identity mismatch: {path}")

    batches = value["batches"]
    if not isinstance(batches, dict):
        raise CheckpointError(f"invalid quality ledger batches: {path}")
    for batch_id, batch in batches.items():
        if (
            not isinstance(batch_id, str)
            or not batch_id
            or not isinstance(batch, dict)
            or set(batch) != {"instances_sha256", "admitted_prefix"}
        ):
            raise CheckpointError(f"invalid quality ledger batch: {path}")
        try:
            _validated_asset_sha(batch["instances_sha256"])
        except (TypeError, ValueError) as error:
            raise CheckpointError(
                f"invalid quality ledger batch hash: {path}"
            ) from error
        prefix = batch["admitted_prefix"]
        if (
            not isinstance(prefix, int)
            or isinstance(prefix, bool)
            or prefix < 0
        ):
            raise CheckpointError(f"invalid quality ledger prefix: {path}")

    entries = value["entries"]
    if not isinstance(entries, list) or len(entries) > QUALITY_WINDOW_SIZE:
        raise CheckpointError(f"invalid quality ledger entries: {path}")
    identities = set()
    assets = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "batch_id",
            "position",
            "asset_sha",
            "outcome",
        }:
            raise CheckpointError(f"invalid quality ledger entry: {path}")
        batch_id = entry["batch_id"]
        position = entry["position"]
        try:
            asset_sha = _validated_asset_sha(entry["asset_sha"])
        except (TypeError, ValueError) as error:
            raise CheckpointError(
                f"invalid quality ledger entry asset: {path}"
            ) from error
        if (
            batch_id not in batches
            or not isinstance(position, int)
            or isinstance(position, bool)
            or position < 0
            or position >= batches[batch_id]["admitted_prefix"]
            or entry["outcome"] not in QUALITY_OUTCOMES
            or (batch_id, position) in identities
            or asset_sha in assets
        ):
            raise CheckpointError(f"invalid quality ledger entry: {path}")
        identities.add((batch_id, position))
        assets.add(asset_sha)
    quarantine = value.setdefault("quarantine", {})
    if not isinstance(quarantine, dict):
        raise CheckpointError(f"invalid quality ledger quarantine: {path}")
    for asset_sha, record in quarantine.items():
        try:
            _validated_asset_sha(asset_sha)
        except (TypeError, ValueError) as error:
            raise CheckpointError(f"invalid quarantine asset: {path}") from error
        if not isinstance(record, dict) or set(record) != {
            "category", "stage", "reason", "attempts"
        }:
            raise CheckpointError(f"invalid quarantine record: {path}")
        if not all(isinstance(record[key], str) and record[key] for key in ("category", "stage", "reason")):
            raise CheckpointError(f"invalid quarantine record: {path}")
        if not isinstance(record["attempts"], int) or isinstance(record["attempts"], bool) or record["attempts"] < 0:
            raise CheckpointError(f"invalid quarantine attempts: {path}")
    family_exclusions = value["family_exclusions"]
    if not isinstance(family_exclusions, dict):
        raise CheckpointError(
            f"invalid quality ledger family exclusions: {path}"
        )
    valid_families = set(PACK_FAMILIES) - {"common"}
    for asset_sha, exclusions in family_exclusions.items():
        try:
            _validated_asset_sha(asset_sha)
        except (TypeError, ValueError) as error:
            raise CheckpointError(
                f"invalid family exclusion asset: {path}"
            ) from error
        if not isinstance(exclusions, dict) or not exclusions:
            raise CheckpointError(
                f"invalid family exclusion mapping: {path}"
            )
        for family, record in exclusions.items():
            if family not in valid_families:
                raise CheckpointError(
                    f"invalid excluded family: {path}"
                )
            if not isinstance(record, dict) or set(record) != {
                "category",
                "stage",
                "reason",
                "attempts",
            }:
                raise CheckpointError(
                    f"invalid family exclusion record: {path}"
                )
            if not all(
                isinstance(record[key], str) and record[key]
                for key in ("category", "stage", "reason")
            ):
                raise CheckpointError(
                    f"invalid family exclusion record: {path}"
                )
            if (
                not isinstance(record["attempts"], int)
                or isinstance(record["attempts"], bool)
                or record["attempts"] < 0
            ):
                raise CheckpointError(
                    f"invalid family exclusion attempts: {path}"
                )
    return value


def _save_quality_ledger(path: Path, ledger: Mapping[str, object]) -> None:
    payload = json.dumps(
        ledger, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    _atomic_write_bytes_nofollow(path, payload)


class PipelineRunner:
    def __init__(
        self,
        config: PipelineConfig,
        resource_guard,
        validators: Mapping[str, Callable[[], bool]],
        internal_handlers: Mapping[str, Callable[[], None]],
        *,
        command_builder: Callable[
            [ShardContext, PipelineConfig], Sequence[CommandSpec]
        ]
        | None = None,
        report_writer: Callable[[EscalationReport], None] | None = None,
        fallback_report_path: Callable[[ShardContext, str], Path]
        | None = None,
        checkpoint_path: Callable[[ShardContext], Path] | None = None,
        quality_ledger_path: Callable[[ShardContext], Path] | None = None,
        process_factory=subprocess.Popen,
        supervisor_factory=None,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        killpg: Callable[[int, int], None] = os.killpg,
        getpgid: Callable[[int], int] = os.getpgid,
        termination_grace_seconds: float = 60.0,
        reap_timeout_seconds: float = 30.0,
        monitor_interval_seconds: float = 5.0,
        process_poll_interval_seconds: float = 0.1,
        environment: Mapping[str, str] | None = None,
        quality_gate: RollingQualityGate | None = None,
        utc_clock: Callable[[], datetime] | None = None,
    ):
        if termination_grace_seconds < 0:
            raise ValueError("termination grace must be non-negative")
        if reap_timeout_seconds <= 0:
            raise ValueError("reap timeout must be positive")
        if monitor_interval_seconds <= 0:
            raise ValueError("monitor interval must be positive")
        if (
            process_poll_interval_seconds <= 0
            or process_poll_interval_seconds > monitor_interval_seconds
        ):
            raise ValueError(
                "process poll interval must be positive and no greater than the monitor interval"
            )
        self.config = config
        self.resource_guard = resource_guard
        self.validators = validators
        self.internal_handlers = internal_handlers
        self.command_builder = command_builder or build_preprocessing_dag
        self.report_writer = report_writer
        self.fallback_report_path = fallback_report_path or (
            lambda context, command: self.config.paths.local_root
            / "control/escalations-fallback"
            / context.source
            / context.shard_id
            / f"{command}.json"
        )
        self.checkpoint_path = checkpoint_path or (
            lambda context: context.work_root / "checkpoint.json"
        )
        self.quality_ledger_path = quality_ledger_path or (
            lambda context: self.config.paths.data2_root
            / "control/quality"
            / context.source
            / f"{context.shard_id}.json"
        )
        self.process_factory = process_factory
        self.supervisor_factory = supervisor_factory or (
            lambda argv, environment: _LinuxProcessSupervisor.launch(
                self.process_factory, argv, environment
            )
        )
        self.monotonic_clock = monotonic_clock
        self.sleeper = sleeper
        self.killpg = killpg
        self.getpgid = getpgid
        self.termination_grace_seconds = termination_grace_seconds
        self.reap_timeout_seconds = reap_timeout_seconds
        self.monitor_interval_seconds = monitor_interval_seconds
        self.process_poll_interval_seconds = process_poll_interval_seconds
        self.environment = dict(os.environ if environment is None else environment)
        self.quality_gate = quality_gate or RollingQualityGate()
        self._quality_prefix_length = 0
        self.utc_clock = utc_clock or (lambda: datetime.now(timezone.utc))
        self.active_context: ShardContext | None = None
        self.active_checkpoint: PipelineCheckpoint | None = None
        self.active_checkpoint_path: Path | None = None
        self._active_quality_ledger: dict[str, object] | None = None
        self._active_quality_ledger_path: Path | None = None
        self._active_quality_assets: tuple[str, ...] | None = None
        self._active_instances_sha256: str | None = None
        self.worker_profile: WorkerProfile | None = None
        self.last_command_timings: dict[str, float] = {}

    def _build_commands(self, context: ShardContext) -> Sequence[CommandSpec]:
        try:
            recent = self.resource_guard.last_five_minutes()
        except (OSError, RuntimeError, ValueError):
            recent = ()
        self.worker_profile = choose_worker_profile(
            recent, self.config, self.worker_profile
        )
        parameters = inspect.signature(self.command_builder).parameters
        if len(parameters) >= 3:
            return self.command_builder(
                context, self.config, self.worker_profile
            )
        return self.command_builder(context, self.config)

    def _validator(self, command_name: str) -> Callable[[], bool]:
        try:
            validator = self.validators[command_name]
        except KeyError as error:
            raise InfrastructureError(
                f"missing validator for command: {command_name}"
            ) from error
        if not callable(validator):
            raise InfrastructureError(
                f"invalid validator for command: {command_name}"
            )
        return validator

    def _valid_output(self, command_name: str) -> bool:
        try:
            return self._validator(command_name)() is True
        except ValidationError:
            return False

    def _load_frozen_quality_assets(
        self, context: ShardContext
    ) -> tuple[str, ...]:
        if self._active_quality_assets is not None:
            return self._active_quality_assets
        payload = _read_regular_bytes_nofollow(context.instances)
        assets = tuple(
            _validated_asset_sha(item)
            for item in payload.decode("ascii").splitlines()
        )
        if len(assets) != len(set(assets)):
            raise InfrastructureError("duplicate frozen quality asset")
        self._active_quality_assets = assets
        self._active_instances_sha256 = sha256(payload).hexdigest()
        return assets

    @staticmethod
    def _contiguous_quality_prefix(
        checkpoint: PipelineCheckpoint, assets: Sequence[str]
    ) -> int:
        unknown = set(checkpoint.quality_outcomes) - set(assets)
        if unknown:
            raise InfrastructureError(
                f"quality outcomes contain non-frozen assets: {sorted(unknown)}"
            )
        prefix = 0
        for asset_sha in assets:
            if asset_sha not in checkpoint.quality_outcomes:
                break
            prefix += 1
        return prefix

    def _advance_quality_ledger(
        self,
        context: ShardContext,
        checkpoint: PipelineCheckpoint,
        assets: Sequence[str],
        *,
        update_gate: bool,
    ) -> int:
        ledger = self._active_quality_ledger
        ledger_path = self._active_quality_ledger_path
        instances_sha256 = self._active_instances_sha256
        if ledger is None or ledger_path is None or instances_sha256 is None:
            raise InfrastructureError("quality ledger is not active")

        prefix = self._contiguous_quality_prefix(checkpoint, assets)
        batches = ledger["batches"]
        batch = batches.get(context.batch_id)
        if batch is None:
            cursor = 0
        else:
            if batch["instances_sha256"] != instances_sha256:
                raise InfrastructureError(
                    f"quality ledger batch identity changed: {context.batch_id}"
                )
            cursor = batch["admitted_prefix"]
        if cursor > prefix:
            raise InfrastructureError(
                f"quality ledger is ahead of checkpoint: {context.batch_id}"
            )
        if cursor == prefix:
            return prefix

        next_ledger = {
            "schema_version": ledger["schema_version"],
            "source": ledger["source"],
            "shard_id": ledger["shard_id"],
            "gate": ledger["gate"],
            "batches": {
                batch_id: dict(batch_value)
                for batch_id, batch_value in batches.items()
            },
            "entries": [dict(entry) for entry in ledger["entries"]],
            "quarantine": {
                asset: dict(record)
                for asset, record in ledger.get("quarantine", {}).items()
            },
            "family_exclusions": {
                asset: {
                    family: dict(record)
                    for family, record in exclusions.items()
                }
                for asset, exclusions in ledger[
                    "family_exclusions"
                ].items()
            },
        }
        next_ledger["batches"][context.batch_id] = {
            "instances_sha256": instances_sha256,
            "admitted_prefix": prefix,
        }
        admitted = []
        for position in range(cursor, prefix):
            asset_sha = assets[position]
            outcome = checkpoint.quality_outcomes[asset_sha]
            entry = {
                "batch_id": context.batch_id,
                "position": position,
                "asset_sha": asset_sha,
                "outcome": outcome,
            }
            next_ledger["entries"].append(entry)
            admitted.append(entry)
        next_ledger["entries"] = next_ledger["entries"][-QUALITY_WINDOW_SIZE:]
        _save_quality_ledger(ledger_path, next_ledger)
        self._active_quality_ledger = next_ledger
        if update_gate:
            for entry in admitted:
                self.quality_gate.record(
                    asset_sha=entry["asset_sha"],
                    succeeded=entry["outcome"] == "completed",
                    schema_failure=entry["outcome"] == "schema_failure",
                )
        return prefix

    def _restore_quality_state(
        self, context: ShardContext, checkpoint: PipelineCheckpoint
    ) -> None:
        ledger_path = self.quality_ledger_path(context)
        ledger = _load_quality_ledger(ledger_path, context)
        self._active_quality_ledger = ledger
        self._active_quality_ledger_path = ledger_path
        batch = ledger["batches"].get(context.batch_id)
        if checkpoint.quality_outcomes or batch is not None:
            assets = self._load_frozen_quality_assets(context)
            self._quality_prefix_length = self._advance_quality_ledger(
                context, checkpoint, assets, update_gate=False
            )
            ledger = self._active_quality_ledger
        else:
            self._quality_prefix_length = 0
        self.quality_gate.restore_entries(
            tuple(
                (entry["asset_sha"], entry["outcome"])
                for entry in ledger["entries"]
            )
        )

    def run_shard(self, context: ShardContext) -> None:
        if self.active_context is not None:
            raise RuntimeError("pipeline runner is already active")
        self.last_command_timings = {}
        self.active_context = context
        checkpoint_path = self.checkpoint_path(context)
        try:
            try:
                checkpoint = self.load_checkpoint(
                    checkpoint_path, context.shard_id, context.gate
                )
            except CheckpointError as error:
                self.stop(
                    context,
                    "checkpoint",
                    str(error),
                    PipelineCheckpoint(context.shard_id, gate=context.gate),
                    category=EscalationCategory.INFRASTRUCTURE,
                    exit_code=2,
                    save_checkpoint=False,
                )
            self.active_checkpoint = checkpoint
            self.active_checkpoint_path = checkpoint_path
            try:
                self._restore_quality_state(context, checkpoint)
            except (
                CheckpointError,
                InfrastructureError,
                OSError,
                UnicodeError,
                ValueError,
            ) as error:
                self.stop(
                    context,
                    "quality_ledger",
                    str(error) or type(error).__name__,
                    checkpoint,
                    category=EscalationCategory.INFRASTRUCTURE,
                    exit_code=2,
                )
            if checkpoint.active_attempt is not None:
                abandoned = checkpoint.active_attempt["command"]
                checkpoint.active_attempt = None
                try:
                    self.save_checkpoint(checkpoint_path, checkpoint)
                except BaseException as error:
                    self.stop(
                        context,
                        str(abandoned),
                        f"cannot finalize abandoned attempt: {error}",
                        checkpoint,
                        category=EscalationCategory.INFRASTRUCTURE,
                        exit_code=2,
                        initial_persistence_errors=(
                            f"abandoned attempt persistence failed: {error}",
                        ),
                    )

            for command in self._build_commands(context):
                if command.name in checkpoint.completed_commands:
                    try:
                        if self._valid_output(command.name):
                            continue
                    except PipelineStopped:
                        raise
                    except (
                        CheckpointError,
                        InfrastructureError,
                        OSError,
                    ) as error:
                        self.stop(
                            context,
                            command.name,
                            str(error),
                            checkpoint,
                            category=EscalationCategory.INFRASTRUCTURE,
                            exit_code=2,
                        )

                while True:
                    quality_reason = self.quality_gate.violation_reason()
                    if quality_reason:
                        self.stop(
                            context,
                            command.name,
                            quality_reason,
                            checkpoint,
                            category=EscalationCategory.DATA_QUALITY,
                            exit_code=4,
                        )

                    prior_attempts = checkpoint.attempts.get(command.name, 0)
                    if prior_attempts >= MAX_COMMAND_ATTEMPTS:
                        try:
                            if self._valid_output(command.name):
                                checkpoint.complete(command.name)
                                try:
                                    self.save_checkpoint(
                                        checkpoint_path, checkpoint
                                    )
                                except BaseException as error:
                                    self.stop(
                                        context,
                                        command.name,
                                        f"completion checkpoint failed: {error}",
                                        checkpoint,
                                        category=EscalationCategory.INFRASTRUCTURE,
                                        exit_code=2,
                                        initial_persistence_errors=(
                                            "completion checkpoint persistence failed: "
                                            f"{error}",
                                        ),
                                    )
                                break
                        except PipelineStopped:
                            raise
                        except (
                            CheckpointError,
                            InfrastructureError,
                            OSError,
                        ) as error:
                            self.stop(
                                context,
                                command.name,
                                str(error),
                                checkpoint,
                                category=EscalationCategory.INFRASTRUCTURE,
                                exit_code=2,
                            )
                        self.stop(
                            context,
                            command.name,
                            "command attempt budget already exhausted",
                            checkpoint,
                            category=EscalationCategory.COMMAND_FAILURE,
                            exit_code=2,
                        )
                    try:
                        self.resource_guard.wait_for_admission(
                            context.shard_id, command.name
                        )
                    except ResourceLimitExceeded as error:
                        self.stop(
                            context,
                            command.name,
                            "; ".join(error.reasons),
                            checkpoint,
                            category=EscalationCategory.RESOURCE,
                            exit_code=3,
                        )
                    except (OSError, RuntimeError, ValueError) as error:
                        self.stop(
                            context,
                            command.name,
                            str(error) or type(error).__name__,
                            checkpoint,
                            category=EscalationCategory.INFRASTRUCTURE,
                            exit_code=2,
                        )

                    attempt = prior_attempts + 1
                    checkpoint.attempts[command.name] = attempt
                    checkpoint.active_attempt = {
                        "command": command.name,
                        "attempt": attempt,
                    }
                    try:
                        self.save_checkpoint(checkpoint_path, checkpoint)
                    except BaseException as error:
                        self.stop(
                            context,
                            command.name,
                            f"cannot persist attempt before launch: {error}",
                            checkpoint,
                            category=EscalationCategory.INFRASTRUCTURE,
                            exit_code=2,
                            initial_persistence_errors=(
                                f"prelaunch checkpoint persistence failed: {error}",
                            ),
                        )
                    attempt_started = self.monotonic_clock()
                    try:
                        launch_command = self._command_for_eligible_assets(
                            command, context, checkpoint
                        )
                        self.execute(launch_command, context.shard_id)
                        if not self._valid_output(command.name):
                            raise OutputValidationError(
                                f"validation failed: {command.name}"
                            )
                    except ResourceLimitExceeded as error:
                        checkpoint.active_attempt = None
                        self.stop(
                            context,
                            command.name,
                            "; ".join(error.reasons),
                            checkpoint,
                            category=EscalationCategory.RESOURCE,
                            exit_code=3,
                        )
                    except PipelineStopped:
                        raise
                    except (
                        OSError,
                        RuntimeError,
                        subprocess.SubprocessError,
                        ValidationError,
                    ) as error:
                        checkpoint.active_attempt = None
                        persistence_error = None
                        try:
                            self.save_checkpoint(checkpoint_path, checkpoint)
                        except BaseException as save_error:
                            persistence_error = save_error
                        if self.is_infrastructure_error(error):
                            self.stop(
                                context,
                                command.name,
                                str(error) or type(error).__name__,
                                checkpoint,
                                category=EscalationCategory.INFRASTRUCTURE,
                                exit_code=2,
                                initial_persistence_errors=(
                                    "attempt finalization failed: "
                                    f"{persistence_error}",
                                )
                                if persistence_error is not None
                                else (),
                            )
                        if (
                            checkpoint.attempts[command.name]
                            >= MAX_COMMAND_ATTEMPTS
                            or persistence_error is not None
                        ):
                            is_validation = isinstance(error, ValidationError)
                            self.stop(
                                context,
                                command.name,
                                str(error) or type(error).__name__,
                                checkpoint,
                                category=(
                                    EscalationCategory.DATA_QUALITY
                                    if is_validation
                                    else EscalationCategory.COMMAND_FAILURE
                                ),
                                exit_code=4 if is_validation else 2,
                                initial_persistence_errors=(
                                    f"attempt finalization failed: {persistence_error}",
                                )
                                if persistence_error is not None
                                else (),
                            )
                        if command.name == "render_cond":
                            stepped_workers = select_render_workers(
                                current=command.workers_per_gpu,
                                peak_percent=0.0,
                                temperature_celsius=0.0,
                                failed=True,
                                steps=(
                                    self.config.parallelism
                                    .render_workers_per_gpu_steps
                                ),
                            )
                            command = replace(
                                command,
                                workers_per_gpu=stepped_workers,
                            )
                            if self.worker_profile is not None:
                                self.worker_profile = replace(
                                    self.worker_profile,
                                    render_workers_per_gpu=stepped_workers,
                                )
                        continue
                    finally:
                        elapsed = max(
                            0.0, self.monotonic_clock() - attempt_started
                        )
                        self.last_command_timings[command.name] = (
                            self.last_command_timings.get(command.name, 0.0)
                            + elapsed
                        )

                    checkpoint.active_attempt = None
                    checkpoint.complete(command.name)
                    try:
                        self.save_checkpoint(checkpoint_path, checkpoint)
                    except BaseException as error:
                        self.stop(
                            context,
                            command.name,
                            f"completion checkpoint failed: {error}",
                            checkpoint,
                            category=EscalationCategory.INFRASTRUCTURE,
                            exit_code=2,
                            initial_persistence_errors=(
                                f"completion checkpoint persistence failed: {error}",
                            ),
                        )
                    break
        finally:
            self.active_context = None
            self.active_checkpoint = None
            self.active_checkpoint_path = None
            self._quality_prefix_length = 0
            self._active_quality_ledger = None
            self._active_quality_ledger_path = None
            self._active_quality_assets = None
            self._active_instances_sha256 = None

    def resume_shard(self, context: ShardContext) -> None:
        self.run_shard(context)

    def validate_completed_commands(
        self, context: ShardContext, command_names: Sequence[str]
    ) -> bool:
        """Validate a durable command frontier without executing leaf work."""

        if self.active_context is not None:
            raise RuntimeError("pipeline runner is already active")
        requested = tuple(command_names)
        if (
            not requested
            or any(not isinstance(name, str) or not name for name in requested)
            or len(requested) != len(set(requested))
        ):
            raise ValueError("completed command names must be unique")
        self.active_context = context
        checkpoint_path = self.checkpoint_path(context)
        try:
            checkpoint = self.load_checkpoint(
                checkpoint_path, context.shard_id, context.gate
            )
            self.active_checkpoint = checkpoint
            self.active_checkpoint_path = checkpoint_path
            self._restore_quality_state(context, checkpoint)
            commands = {
                command.name: command for command in self._build_commands(context)
            }
            if any(name not in commands for name in requested):
                raise InfrastructureError(
                    "completed command validation requested an unknown command"
                )
            for name in requested:
                if name not in checkpoint.completed_commands:
                    return False
                if not self._valid_output(name):
                    return False
            return True
        finally:
            self.active_context = None
            self.active_checkpoint = None
            self.active_checkpoint_path = None
            self._quality_prefix_length = 0
            self._active_quality_ledger = None
            self._active_quality_ledger_path = None
            self._active_quality_assets = None
            self._active_instances_sha256 = None

    def record_quality_outcome(self, asset_sha: str, outcome: str) -> None:
        self.record_asset_outcome(asset_sha, outcome)

    def family_exclusions(
        self, asset_sha: str
    ) -> Mapping[str, Mapping[str, object]]:
        asset_sha = _validated_asset_sha(asset_sha)
        ledger = self._active_quality_ledger
        if ledger is None:
            return {}
        exclusions = ledger["family_exclusions"].get(asset_sha, {})
        return {
            family: dict(record)
            for family, record in exclusions.items()
        }

    def family_is_eligible(self, asset_sha: str, family: str) -> bool:
        if family not in PACK_FAMILIES:
            raise ValueError(f"unknown pack family: {family}")
        excluded = set(self.family_exclusions(asset_sha))
        dependencies = family_dependencies(self.config)

        def eligible(candidate: str, visiting: frozenset[str]) -> bool:
            if candidate in excluded:
                return False
            if candidate in visiting:
                raise InfrastructureError(
                    f"cyclic family dependency: {candidate}"
                )
            return all(
                eligible(dependency, visiting | {candidate})
                for dependency in dependencies[candidate]
            )

        return eligible(family, frozenset())

    def record_family_exclusion(
        self,
        asset_sha: str,
        families: Sequence[str],
        *,
        category: str,
        stage: str,
        reason: str,
        attempts: int = 0,
    ) -> None:
        context = self.active_context
        ledger = self._active_quality_ledger
        ledger_path = self._active_quality_ledger_path
        if context is None or ledger is None or ledger_path is None:
            raise InfrastructureError(
                "family exclusion has no active quality ledger"
            )
        asset_sha = _validated_asset_sha(asset_sha)
        if asset_sha not in self._load_frozen_quality_assets(context):
            raise InfrastructureError(
                f"family exclusion asset is not frozen: {asset_sha}"
            )
        valid_families = set(PACK_FAMILIES) - {"common"}
        requested = tuple(sorted(set(families)))
        if not requested or not set(requested) <= valid_families:
            raise ValueError("invalid excluded family")
        if not all(
            isinstance(value, str) and value
            for value in (category, stage, reason)
        ):
            raise ValueError("invalid family exclusion record")
        if (
            not isinstance(attempts, int)
            or isinstance(attempts, bool)
            or attempts < 0
        ):
            raise ValueError("invalid family exclusion attempts")
        record = {
            "category": category,
            "stage": stage,
            "reason": reason,
            "attempts": attempts,
        }
        next_ledger = {
            **ledger,
            "family_exclusions": {
                asset: {
                    family: dict(existing_record)
                    for family, existing_record in exclusions.items()
                }
                for asset, exclusions in ledger[
                    "family_exclusions"
                ].items()
            },
        }
        by_family = next_ledger["family_exclusions"].setdefault(
            asset_sha, {}
        )
        for family in requested:
            existing = by_family.get(family)
            if existing is not None and existing != record:
                raise InfrastructureError(
                    f"conflicting family exclusion: {asset_sha}: {family}"
                )
            by_family[family] = dict(record)
        _save_quality_ledger(ledger_path, next_ledger)
        self._active_quality_ledger = next_ledger

    def record_asset_outcome(
        self,
        asset_sha: str,
        outcome: str,
        *,
        category: str = "asset_validation",
        stage: str = "validation",
        reason: str = "asset output failed validation",
        attempts: int = 0,
    ) -> None:
        checkpoint = self.active_checkpoint
        checkpoint_path = self.active_checkpoint_path
        context = self.active_context
        if checkpoint is None or checkpoint_path is None or context is None:
            raise InfrastructureError(
                "quality outcome has no active durable checkpoint"
            )
        if outcome not in QUALITY_OUTCOMES:
            raise ValueError(f"invalid terminal quality outcome: {outcome}")
        asset_sha = _validated_asset_sha(asset_sha)
        existing = checkpoint.quality_outcomes.get(asset_sha)
        if existing is not None:
            if existing != outcome:
                raise InfrastructureError(
                    f"conflicting durable quality outcome: {asset_sha}"
                )
        assets = self._load_frozen_quality_assets(context)
        if asset_sha not in assets:
            raise InfrastructureError(
                f"quality outcome asset is not frozen: {asset_sha}"
            )
        if existing is None:
            checkpoint.quality_outcomes[asset_sha] = outcome
        checkpoint.quality_outcomes = {
            frozen_sha: checkpoint.quality_outcomes[frozen_sha]
            for frozen_sha in assets
            if frozen_sha in checkpoint.quality_outcomes
        }
        self.save_checkpoint(checkpoint_path, checkpoint)
        if outcome in {"failure", "schema_failure"}:
            ledger = self._active_quality_ledger
            ledger_path = self._active_quality_ledger_path
            if ledger is None or ledger_path is None:
                raise InfrastructureError("quality ledger is not active")
            next_ledger = dict(ledger)
            next_ledger["quarantine"] = {
                asset: dict(record)
                for asset, record in ledger.get("quarantine", {}).items()
            }
            next_ledger["quarantine"][asset_sha] = {
                "category": category,
                "stage": stage,
                "reason": reason,
                "attempts": attempts,
            }
            _save_quality_ledger(ledger_path, next_ledger)
            self._active_quality_ledger = next_ledger
        self._quality_prefix_length = self._advance_quality_ledger(
            context, checkpoint, assets, update_gate=True
        )
        reason = self.quality_gate.violation_reason()
        if reason:
            checkpoint.active_attempt = None
            self.stop(
                context,
                "validate_outputs",
                reason,
                checkpoint,
                category=EscalationCategory.DATA_QUALITY,
                exit_code=4,
            )

    def _command_for_eligible_assets(
        self,
        command: CommandSpec,
        context: ShardContext,
        checkpoint: PipelineCheckpoint,
    ) -> CommandSpec:
        if "--instances" not in command.argv:
            return command
        families = command_families(self.config, command.name)
        ledger = self._active_quality_ledger
        has_family_exclusions = bool(
            ledger is not None and ledger.get("family_exclusions")
        )
        if not checkpoint.quality_outcomes and not (
            families and has_family_exclusions
        ):
            return command
        payload = _read_regular_bytes_nofollow(context.instances)
        assets = tuple(
            _validated_asset_sha(item)
            for item in payload.decode("ascii").splitlines()
        )
        eligible = tuple(
            asset
            for asset in assets
            if asset not in checkpoint.quality_outcomes
            and (
                not families
                or any(
                    self.family_is_eligible(asset, family)
                    for family in families
                )
            )
        )
        eligible_path = (
            context.work_root
            / "control/eligible"
            / f"{_safe_component(command.name, 'command name')}.txt"
        )
        _atomic_write_bytes_nofollow(
            eligible_path,
            "".join(f"{asset}\n" for asset in eligible).encode("ascii"),
        )
        argv = list(command.argv)
        argv[argv.index("--instances") + 1] = str(eligible_path)
        return replace(command, argv=tuple(argv))

    def execute(self, command: CommandSpec, shard_id: str) -> None:
        if not command.argv:
            raise InfrastructureError(f"empty command argv: {command.name}")
        try:
            expanded_commands = expand_ranked(command)
        except (TypeError, ValueError) as error:
            raise InfrastructureError(
                f"invalid rank count or worker count for command "
                f"{command.name}: {error}"
            ) from error
        if (
            command.name in INTERNAL_COMMANDS
            or command.argv[0].startswith("internal:")
        ):
            expected = INTERNAL_COMMANDS.get(command.name)
            if expected is None or command.argv != expected:
                raise InfrastructureError(
                    f"unknown internal command: {command.name}: {command.argv!r}"
                )
            handler = self.internal_handlers.get(command.name)
            if handler is None or not callable(handler):
                raise InfrastructureError(
                    f"missing internal handler: {command.name}"
                )
            handler()
            return

        processes = []
        paused_groups: set[int] = set()
        try:
            for argv, additions in expanded_commands:
                environment = dict(self.environment)
                environment.update(dict(additions))
                process = self.supervisor_factory(argv, environment)
                processes.append(process)
            self._monitor_processes(
                processes, paused_groups, shard_id, command
            )
            self._reap(processes)
            self._close_processes(processes)
        except BaseException as error:
            try:
                self._terminate_and_reap(processes, paused_groups)
            except BaseException as cleanup_error:
                if isinstance(
                    error, (ResourceLimitExceeded, ProcessGroupSafetyError)
                ):
                    notes = list(getattr(error, "__notes__", ()))
                    notes.append(
                        "supervisor cleanup failure: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                    error.__notes__ = notes
                else:
                    raise cleanup_error from error
            raise

    def _monitor_processes(
        self,
        processes,
        paused_groups: set[int],
        shard_id: str,
        command: CommandSpec,
    ) -> None:
        next_resource_check = self.monotonic_clock()
        while True:
            statuses = [process.poll() for process in processes]
            failed_index = next(
                (
                    index
                    for index, status in enumerate(statuses)
                    if status not in (None, 0)
                ),
                None,
            )
            if failed_index is not None:
                raise subprocess.CalledProcessError(
                    statuses[failed_index], command.argv
                )
            if all(status is not None for status in statuses):
                return

            now = self.monotonic_clock()
            if now >= next_resource_check:
                try:
                    decision = self.resource_guard.check(
                        shard_id, command.name
                    )
                    if not isinstance(decision, ResourceDecision):
                        raise TypeError(
                            f"invalid resource decision: {decision!r}"
                        )
                    action = decision.action
                    if not isinstance(action, ResourceAction):
                        raise TypeError(
                            f"invalid resource action: {action!r}"
                        )
                except ResourceLimitExceeded:
                    raise
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    raise InfrastructureError(
                        f"resource monitor failed: {error}"
                    ) from error
                if action == ResourceAction.STOP:
                    raise ResourceLimitExceeded(decision.reasons)
                if action == ResourceAction.PAUSE:
                    for process, status in zip(processes, statuses):
                        if (
                            status is None
                            and process.pid not in paused_groups
                        ):
                            self._signal_process(process, "pause")
                            paused_groups.add(process.pid)
                elif action == ResourceAction.RUN and paused_groups:
                    for process, status in zip(processes, statuses):
                        if (
                            status is None
                            and process.pid in paused_groups
                        ):
                            self._signal_process(process, "resume")
                            paused_groups.discard(process.pid)
                next_resource_check = now + self.monitor_interval_seconds
            delay = min(
                self.process_poll_interval_seconds,
                max(0.0, next_resource_check - self.monotonic_clock()),
            )
            self.sleeper(delay or self.process_poll_interval_seconds)

    @staticmethod
    def _signal_process(process, action: str) -> None:
        try:
            process.send_control(action)
        except ProcessLookupError:
            if process.poll() is not None:
                return
            raise ProcessGroupSafetyError(
                f"stable supervisor disappeared during {action}: {process.pid}"
            )
        except ProcessGroupSafetyError:
            raise
        except OSError as error:
            if process.poll() is not None:
                return
            raise ProcessGroupSafetyError(
                f"cannot control stable supervisor {process.pid}: {error}"
            ) from error

    @staticmethod
    def _alive(processes):
        return [process for process in processes if process.poll() is None]

    def _reap(self, processes) -> None:
        failure = None
        for process in processes:
            try:
                process.wait(timeout=self.reap_timeout_seconds)
            except subprocess.TimeoutExpired as error:
                error = ProcessGroupSafetyError(
                    f"supervisor did not exit within "
                    f"{self.reap_timeout_seconds}s: {process.pid}"
                )
                if failure is None:
                    failure = error
            except BaseException as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure

    @staticmethod
    def _close_processes(processes) -> None:
        failure = None
        for process in processes:
            close = getattr(process, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except BaseException as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure

    def _terminate_and_reap(self, processes, paused_groups: set[int]) -> None:
        failure = None

        def remember(error):
            nonlocal failure
            if failure is None:
                failure = error

        def alive_or_owned():
            try:
                return self._alive(processes)
            except BaseException as error:
                remember(error)
                return list(processes)

        def signal_all(candidates, action):
            for process in candidates:
                try:
                    self._signal_process(process, action)
                except BaseException as error:
                    remember(error)

        try:
            alive = alive_or_owned()
            signal_all(
                [process for process in alive if process.pid in paused_groups],
                "resume",
            )
            paused_groups.clear()
            signal_all(alive, "terminate")

            try:
                deadline = self.monotonic_clock() + self.termination_grace_seconds
                alive = alive_or_owned()
                while alive and self.monotonic_clock() < deadline:
                    self.sleeper(
                        min(
                            1.0,
                            max(0.0, deadline - self.monotonic_clock()),
                        )
                    )
                    alive = alive_or_owned()
            except BaseException as error:
                remember(error)
                alive = list(processes)
            signal_all(alive, "kill")
            try:
                self._reap(processes)
            except BaseException as error:
                remember(error)
            alive = alive_or_owned()
            if alive:
                signal_all(alive, "kill")
                try:
                    self._reap(alive)
                except BaseException as error:
                    remember(error)
        finally:
            try:
                self._close_processes(processes)
            except BaseException as error:
                remember(error)
        if failure is not None:
            raise failure

    def load_checkpoint(
        self, path: Path, shard_id: str, gate: str = "production"
    ) -> PipelineCheckpoint:
        path = Path(path)
        try:
            payload = _read_regular_bytes_nofollow(path, missing_ok=True)
            if payload is None:
                return PipelineCheckpoint(shard_id, gate=gate)
            value = json.loads(payload)
        except OSError as error:
            raise CheckpointError(
                f"checkpoint is not a regular file or has an unsafe path: {path}: {error}"
            ) from error
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
        ) as error:
            raise CheckpointError(f"invalid checkpoint: {path}: {error}") from error
        checkpoint = _required_checkpoint_dict(value, path)
        if checkpoint.shard_id != shard_id:
            raise CheckpointError(
                f"checkpoint shard identity mismatch: "
                f"expected {shard_id}, found {checkpoint.shard_id}"
            )
        if checkpoint.gate != gate:
            raise CheckpointError(
                f"checkpoint gate identity mismatch: expected {gate}, "
                f"found {checkpoint.gate}"
            )
        return checkpoint

    def save_checkpoint(
        self, path: Path, checkpoint: PipelineCheckpoint
    ) -> None:
        validated = _required_checkpoint_dict(asdict(checkpoint), Path(path))
        try:
            payload = json.dumps(
                asdict(validated), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            _atomic_write_bytes_nofollow(path, payload)
        except OSError as error:
            raise CheckpointError(
                f"unsafe checkpoint destination: {path}: {error}"
            ) from error

    def is_infrastructure_error(self, error: Exception) -> bool:
        if isinstance(
            error,
            (
                CheckpointError,
                InfrastructureError,
                ProcessGroupSafetyError,
                OSError,
            ),
        ):
            return True
        text = str(error).lower()
        return any(
            term in text
            for term in (
                "authentication",
                "checkpoint",
                "checksum",
                "no optix device",
                "manifest corruption",
                "resource hard",
            )
        )

    @staticmethod
    def _recovery_choices(category: EscalationCategory) -> tuple[str, ...]:
        if category == EscalationCategory.RESOURCE:
            return (
                "Restore resource headroom without changing pipeline policy.",
                "Resume the same frozen shard and batch.",
            )
        if category == EscalationCategory.DATA_QUALITY:
            return (
                "Audit recent failures and repair the unchanged data contract.",
                "Resume only after operator approval; do not change scope or dtype silently.",
            )
        return (
            "Repair the reported dependency or command failure.",
            "Resume the same frozen shard and batch without changing retention policy.",
        )

    def stop(
        self,
        context: ShardContext,
        command: str,
        reason: str,
        checkpoint: PipelineCheckpoint,
        *,
        category: EscalationCategory,
        exit_code: int,
        save_checkpoint: bool = True,
        initial_persistence_errors: Sequence[str] = (),
    ) -> None:
        persistence_errors = list(initial_persistence_errors)
        if save_checkpoint:
            try:
                self.save_checkpoint(self.checkpoint_path(context), checkpoint)
            except BaseException as error:
                persistence_errors.append(
                    f"checkpoint persistence failed: "
                    f"{type(error).__name__}: {error}"
                )
        try:
            recent_telemetry = tuple(
                self.resource_guard.last_five_minutes()
            )
        except BaseException as error:
            recent_telemetry = ()
            persistence_errors.append(
                f"telemetry retrieval failed: {type(error).__name__}: {error}"
            )
        try:
            created_at = self.utc_clock()
            if created_at.tzinfo is None or created_at.utcoffset() is None:
                raise ValueError("escalation timestamp must be timezone-aware")
        except BaseException as error:
            persistence_errors.append(
                f"escalation clock failed: {type(error).__name__}: {error}"
            )
            created_at = datetime.now(timezone.utc)
        outcome_counts = {
            outcome: tuple(checkpoint.quality_outcomes.values()).count(outcome)
            for outcome in QUALITY_OUTCOMES
        }
        report = EscalationReport(
            source=context.source,
            shard_id=context.shard_id,
            command=command,
            category=category,
            reason=reason,
            recent_telemetry=recent_telemetry,
            completed_counts={
                "commands": len(checkpoint.completed_commands),
                "outcomes": self.quality_gate.count,
                "completed_assets": outcome_counts["completed"],
                "quarantined_assets": (
                    outcome_counts["failure"]
                    + outcome_counts["schema_failure"]
                ),
                "failure_assets": outcome_counts["failure"],
                "schema_failure_assets": outcome_counts[
                    "schema_failure"
                ],
            },
            safe_resume_command=(
                "python -m data_toolkit.pipeline.cli resume "
                f"--gate {shlex.quote(context.gate)} "
                f"--source {shlex.quote(context.source)} "
                f"--shard {shlex.quote(context.shard_id)}"
            ),
            recovery_choices=self._recovery_choices(category),
            created_at=created_at.astimezone(timezone.utc).isoformat(),
            persistence_errors=tuple(persistence_errors),
            gate=context.gate,
        )
        primary_failed = self.report_writer is None
        if self.report_writer is None:
            persistence_errors.append("primary escalation report writer unavailable")
        else:
            try:
                self.report_writer(report)
            except BaseException as error:
                primary_failed = True
                persistence_errors.append(
                    f"primary report persistence failed: "
                    f"{type(error).__name__}: {error}"
                )
        if primary_failed:
            report = replace(
                report, persistence_errors=tuple(persistence_errors)
            )
            try:
                payload = json.dumps(
                    asdict(report), sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                _atomic_write_bytes_nofollow(
                    self.fallback_report_path(context, command), payload
                )
            except BaseException as error:
                persistence_errors.append(
                    f"fallback report persistence failed: "
                    f"{type(error).__name__}: {error}"
                )
                report = replace(
                    report, persistence_errors=tuple(persistence_errors)
                )
        raise PipelineStopped(report, exit_code)


def _atomic_write_text(path: Path, value: str) -> None:
    _atomic_write_bytes_nofollow(Path(path), value.encode("utf-8"))


def _safe_component(value: str, description: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
        or "\\" in value
        or "\0" in value
    ):
        raise ValueError(f"unsafe {description}: {value!r}")
    return value


def _safe_raw_relative(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValidationError(f"unsafe raw path: {value!r}")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or value == "."
        or ".." in pure.parts
    ):
        raise ValidationError(f"unsafe raw path: {value!r}")
    return Path(*pure.parts)


def _sha_stream(stream) -> str:
    digest = sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _open_directory_nofollow(path: Path, *, create: bool = False) -> int:
    absolute = Path(os.path.abspath(path))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    directory_fd = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            try:
                next_fd = os.open(component, flags, dir_fd=directory_fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, 0o755, dir_fd=directory_fd)
                next_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except BaseException:
        os.close(directory_fd)
        raise


def _same_inode(first, second) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _rename_noreplace(
    source_name: str,
    destination_name: str,
    source_directory_fd: int,
    destination_directory_fd: int | None = None,
) -> None:
    if destination_directory_fd is None:
        destination_directory_fd = source_directory_fd
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = libc.renameat2
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            source_directory_fd,
            os.fsencode(source_name),
            destination_directory_fd,
            os.fsencode(destination_name),
            1,
        )
        != 0
    ):
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _read_regular_bytes_nofollow(
    path: Path,
    *,
    missing_ok: bool = False,
) -> bytes | None:
    path = Path(path)
    try:
        directory_fd = _open_directory_nofollow(path.parent)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise
    try:
        try:
            file_fd = os.open(
                path.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            if missing_ok:
                return None
            raise
        try:
            opened = os.fstat(file_fd)
            if not stat.S_ISREG(opened.st_mode):
                raise OSError(errno.ELOOP, f"not a regular file: {path}")
            with os.fdopen(file_fd, "rb", closefd=False) as stream:
                value = stream.read()
            current = os.stat(
                path.name, dir_fd=directory_fd, follow_symlinks=False
            )
            if not stat.S_ISREG(current.st_mode) or not _same_inode(
                opened, current
            ):
                raise OSError(
                    errno.ELOOP, f"file identity changed while reading: {path}"
                )
            return value
        finally:
            os.close(file_fd)
    finally:
        os.close(directory_fd)


def _regular_file_stat_nofollow(
    path: Path, *, missing_ok: bool = False
):
    path = Path(path)
    try:
        directory_fd = _open_directory_nofollow(path.parent)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise
    try:
        try:
            file_fd = os.open(
                path.name,
                os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            if missing_ok:
                return None
            raise
        try:
            opened = os.fstat(file_fd)
            current = os.stat(
                path.name, dir_fd=directory_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(current.st_mode)
                or not _same_inode(opened, current)
            ):
                raise OSError(errno.ELOOP, f"unsafe accounted path: {path}")
            return opened
        finally:
            os.close(file_fd)
    finally:
        os.close(directory_fd)


def _regular_file_size_nofollow(path: Path, *, missing_ok: bool = False) -> int:
    value = _regular_file_stat_nofollow(path, missing_ok=missing_ok)
    return 0 if value is None else value.st_size


def _require_regular_artifact(path: Path) -> None:
    try:
        value = _regular_file_stat_nofollow(path, missing_ok=True)
    except OSError as error:
        if error.errno in {errno.ENOENT, errno.ENOTDIR}:
            raise OutputValidationError(f"missing artifact: {path}") from error
        if error.errno == errno.ELOOP:
            raise ValidationError(f"unsafe artifact: {path}") from error
        raise
    if value is None:
        raise OutputValidationError(f"missing artifact: {path}")


def _atomic_write_bytes_nofollow(path: Path, value: bytes) -> None:
    path = Path(path)
    directory_fd = _open_directory_nofollow(path.parent, create=True)
    temporary_name = (
        f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    )
    temporary_fd = None
    existing = None
    try:
        try:
            existing = os.stat(
                path.name, dir_fd=directory_fd, follow_symlinks=False
            )
            if not stat.S_ISREG(existing.st_mode):
                raise OSError(errno.ELOOP, f"unsafe destination: {path}")
        except FileNotFoundError:
            existing = None
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_fd,
        )
        view = memoryview(value)
        while view:
            written = os.write(temporary_fd, view)
            view = view[written:]
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = None
        try:
            current = os.stat(
                path.name, dir_fd=directory_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            current = None
        if (existing is None) != (current is None) or (
            existing is not None
            and current is not None
            and not _same_inode(existing, current)
        ):
            raise OSError(
                errno.ELOOP, f"destination identity changed: {path}"
            )
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)


def _unlink_regular_nofollow(path: Path, *, missing_ok: bool = False) -> bool:
    path = Path(path)
    try:
        directory_fd = _open_directory_nofollow(path.parent)
    except FileNotFoundError:
        if missing_ok:
            return False
        raise
    try:
        try:
            value = os.stat(
                path.name, dir_fd=directory_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            if missing_ok:
                return False
            raise
        if not stat.S_ISREG(value.st_mode):
            raise OSError(errno.ELOOP, f"unsafe removal target: {path}")
        os.unlink(path.name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        return True
    finally:
        os.close(directory_fd)


def _open_regular_beneath(root: Path, relative: Path):
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        root_fd = _open_directory_nofollow(root)
    except OSError as error:
        if error.errno in PATH_VALIDATION_ERRNOS:
            raise ValidationError(f"unsafe raw source root: {root}") from error
        raise
    directory_fd = os.dup(root_fd)
    os.close(root_fd)
    try:
        for component in relative.parts[:-1]:
            try:
                next_fd = os.open(
                    component,
                    flags | os.O_DIRECTORY,
                    dir_fd=directory_fd,
                )
            except OSError as error:
                if error.errno in PATH_VALIDATION_ERRNOS:
                    raise ValidationError(
                        f"symlink or unsafe raw source: {relative.as_posix()}"
                    ) from error
                raise
            os.close(directory_fd)
            directory_fd = next_fd
        try:
            file_fd = os.open(
                relative.parts[-1],
                flags | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
        except OSError as error:
            if error.errno in PATH_VALIDATION_ERRNOS:
                raise ValidationError(
                    f"symlink or missing raw source: {relative.as_posix()}"
                ) from error
            raise
        try:
            if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                raise ValidationError(
                    f"raw source is not a regular file: {relative.as_posix()}"
                )
            return os.fdopen(file_fd, "rb", closefd=True)
        except BaseException:
            os.close(file_fd)
            raise
    finally:
        os.close(directory_fd)


def _unlink_regular_beneath(
    root: Path,
    relative: Path,
    *,
    rename_noreplace: Callable[..., None] = _rename_noreplace,
) -> int:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        root_fd = _open_directory_nofollow(root)
    except FileNotFoundError:
        return False
    except OSError as error:
        if error.errno in PATH_VALIDATION_ERRNOS:
            raise ValidationError(f"unsafe raw source root: {root}") from error
        raise
    try:
        directory_fd = os.dup(root_fd)
    except BaseException:
        os.close(root_fd)
        raise
    try:
        for component in relative.parts[:-1]:
            try:
                next_fd = os.open(
                    component,
                    flags | os.O_DIRECTORY,
                    dir_fd=directory_fd,
                )
            except FileNotFoundError:
                return False
            except OSError as error:
                if error.errno in PATH_VALIDATION_ERRNOS:
                    raise ValidationError(
                        f"symlink or unsafe raw source: {relative.as_posix()}"
                    ) from error
                raise
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = None
        try:
            file_fd = os.open(
                relative.parts[-1],
                flags | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            return False
        except OSError as error:
            if error.errno in PATH_VALIDATION_ERRNOS:
                raise ValidationError(
                    f"refusing to delete unsafe raw source: {relative.as_posix()}"
                ) from error
            raise
        try:
            opened = os.fstat(file_fd)
            if not stat.S_ISREG(opened.st_mode):
                raise ValidationError(
                    f"refusing to delete unsafe raw source: {relative.as_posix()}"
                )
            source_name = relative.parts[-1]
            quarantine_name = (
                f".pixal3d-quarantine.{os.getpid()}."
                f"{time.monotonic_ns()}"
            )
            quarantine_fd = None
            quarantine_created = False
            try:
                os.mkdir(quarantine_name, 0o700, dir_fd=root_fd)
                quarantine_created = True
                quarantine_fd = os.open(
                    quarantine_name,
                    flags | os.O_DIRECTORY,
                    dir_fd=root_fd,
                )
                quarantined_name = "payload"
                rename_noreplace(
                    source_name,
                    quarantined_name,
                    directory_fd,
                    quarantine_fd,
                )
                quarantined_fd = os.open(
                    quarantined_name,
                    flags | os.O_NONBLOCK,
                    dir_fd=quarantine_fd,
                )
                try:
                    quarantined = os.fstat(quarantined_fd)
                    if not stat.S_ISREG(
                        quarantined.st_mode
                    ) or not _same_inode(opened, quarantined):
                        try:
                            rename_noreplace(
                                quarantined_name,
                                source_name,
                                quarantine_fd,
                                directory_fd,
                            )
                        except FileExistsError:
                            pass
                        raise ValidationError(
                            f"raw source identity changed before delete: "
                            f"{relative.as_posix()}"
                        )
                    os.unlink(quarantined_name, dir_fd=quarantine_fd)
                    os.fsync(quarantine_fd)
                finally:
                    os.close(quarantined_fd)
            finally:
                if quarantine_fd is not None:
                    os.close(quarantine_fd)
                if quarantine_created:
                    try:
                        os.rmdir(quarantine_name, dir_fd=root_fd)
                    except OSError as error:
                        if error.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
                            raise
        finally:
            os.close(file_fd)
        os.fsync(directory_fd)
        return opened.st_size
    finally:
        os.close(directory_fd)
        os.close(root_fd)


def _safe_destination_parent(root: Path, relative: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValidationError(f"symlink staging root: {root}")
    current = root
    for component in relative.parent.parts:
        current = current / component
        try:
            current.mkdir()
        except FileExistsError:
            pass
        try:
            mode = current.lstat().st_mode
        except OSError as error:
            raise ValidationError(
                f"unsafe staging path: {relative.as_posix()}"
            ) from error
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ValidationError(
                f"symlink staging path: {relative.as_posix()}"
            )
    return current


def _atomic_stage_stream(
    stream, root: Path, relative: Path, expected_sha: str
) -> None:
    destination = Path(root) / relative
    try:
        directory_fd = _open_directory_nofollow(
            destination.parent, create=True
        )
    except OSError as error:
        if error.errno in PATH_VALIDATION_ERRNOS:
            raise ValidationError(
                f"unsafe staging destination: {relative.as_posix()}: {error}"
            ) from error
        raise
    temporary_name = (
        f".{destination.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    )
    temporary_fd = None
    digest = sha256()
    try:
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_fd,
        )
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(temporary_fd, view)
                view = view[written:]
        if digest.hexdigest() != expected_sha:
            raise ValidationError(
                f"raw checksum mismatch: {relative.as_posix()}"
            )
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = None
        try:
            _rename_noreplace(
                temporary_name, destination.name, directory_fd
            )
        except FileExistsError:
            existing_fd = os.open(
                destination.name,
                os.O_RDONLY
                | os.O_NOFOLLOW
                | os.O_CLOEXEC
                | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
            try:
                opened = os.fstat(existing_fd)
                if not stat.S_ISREG(opened.st_mode):
                    raise ValidationError(
                        f"unsafe staging destination: {relative.as_posix()}"
                    )
                with os.fdopen(existing_fd, "rb", closefd=False) as existing:
                    existing_sha = _sha_stream(existing)
                current = os.stat(
                    destination.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(current.st_mode)
                    or not _same_inode(opened, current)
                    or existing_sha != expected_sha
                ):
                    raise ValidationError(
                        f"staging destination already differs: "
                        f"{relative.as_posix()}"
                    )
            finally:
                os.close(existing_fd)
        os.fsync(directory_fd)
    except OSError as error:
        if error.errno in PATH_VALIDATION_ERRNOS:
            raise ValidationError(
                f"unsafe staging destination: {relative.as_posix()}: {error}"
            ) from error
        raise
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)


def _remove_tree_nofollow(path: Path) -> bool:
    path = Path(path)
    try:
        parent_fd = _open_directory_nofollow(path.parent)
    except FileNotFoundError:
        return False
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC

    def remove_contents(directory_fd: int) -> None:
        for name in os.listdir(directory_fd):
            value = os.stat(
                name, dir_fd=directory_fd, follow_symlinks=False
            )
            if stat.S_ISLNK(value.st_mode):
                raise ValidationError(f"refusing symlink during cleanup: {name}")
            if stat.S_ISDIR(value.st_mode):
                child_fd = os.open(name, flags, dir_fd=directory_fd)
                try:
                    remove_contents(child_fd)
                finally:
                    os.close(child_fd)
                os.rmdir(name, dir_fd=directory_fd)
            elif stat.S_ISREG(value.st_mode):
                os.unlink(name, dir_fd=directory_fd)
            else:
                raise ValidationError(
                    f"refusing non-file during cleanup: {name}"
                )
        os.fsync(directory_fd)

    try:
        try:
            directory_fd = os.open(path.name, flags, dir_fd=parent_fd)
        except FileNotFoundError:
            return False
        try:
            remove_contents(directory_fd)
        finally:
            os.close(directory_fd)
        os.rmdir(path.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        return True
    except OSError as error:
        if error.errno in PATH_VALIDATION_ERRNOS:
            raise ValidationError(
                f"unsafe cleanup path: {path}: {error}"
            ) from error
        raise
    finally:
        os.close(parent_fd)


def _zip_parts(relative: Path) -> tuple[Path, str] | None:
    for index, component in enumerate(relative.parts):
        if component.lower().endswith(".zip"):
            member_parts = relative.parts[index + 1 :]
            if not member_parts:
                return None
            return Path(*relative.parts[: index + 1]), PurePosixPath(
                *member_parts
            ).as_posix()
    return None


def _safe_zip_infos(bundle: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    result = {}
    for info in bundle.infolist():
        name = info.filename
        pure = PurePosixPath(name)
        mode = info.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        if (
            not name
            or "\\" in name
            or pure.is_absolute()
            or pure.as_posix() != name.rstrip("/")
            or ".." in pure.parts
            or stat.S_ISLNK(mode)
            or file_type not in {0, stat.S_IFREG, stat.S_IFDIR}
        ):
            raise ValidationError(f"unsafe ZIP member: {name}")
        if name in result:
            raise ValidationError(f"duplicate ZIP member: {name}")
        result[name] = info
    return result


class _ParallelChunkExecutor:
    """Own one isolated PipelineServices/runner pair per active chunk."""

    def __init__(
        self,
        parent_services: "PipelineServices",
        checkpoint_root: Path,
        stage_commands: Mapping[str, tuple[str, ...]],
    ) -> None:
        self.parent_services = parent_services
        self.checkpoint_root = Path(checkpoint_root)
        self.stage_commands = dict(stage_commands)
        self._services: dict[str, PipelineServices] = {}
        self._contexts: dict[str, ShardContext] = {}
        self._lock = threading.Lock()

    def _service(
        self, chunk: ChunkContext
    ) -> tuple["PipelineServices", ShardContext]:
        with self._lock:
            existing = self._services.get(chunk.chunk_id)
            if existing is not None:
                return existing, self._contexts[chunk.chunk_id]
            parent = self.parent_services
            context = chunk.as_shard_context()
            service = PipelineServices(
                parent.config,
                resource_guard=parent.resource_guard,
                pilot_reader=parent.pilot_reader,
                reference_counter=parent.reference_counter,
                project_accounting=parent.project_accounting,
                registry_store=parent.registry,
                disk_usage=parent.disk_usage,
                tool_commit=parent._tool_commit,
            )
            control = self.checkpoint_root / chunk.chunk_id
            service.runner.checkpoint_path = (
                lambda _context, path=control / "pipeline.json": path
            )
            service.runner.quality_ledger_path = (
                lambda _context, path=control / "quality.json": path
            )
            parent_runner = parent.runner
            for attribute in (
                "process_factory",
                "supervisor_factory",
                "monotonic_clock",
                "sleeper",
                "killpg",
                "getpgid",
                "termination_grace_seconds",
                "reap_timeout_seconds",
                "monitor_interval_seconds",
                "process_poll_interval_seconds",
                "environment",
                "utc_clock",
            ):
                if hasattr(parent_runner, attribute):
                    value = getattr(parent_runner, attribute)
                    if attribute == "environment":
                        value = dict(value)
                    setattr(service.runner, attribute, value)
            self._seed_parent_quality(service, context, control, chunk.parent)
            self._services[chunk.chunk_id] = service
            self._contexts[chunk.chunk_id] = context
            return service, context

    def _seed_parent_quality(
        self,
        service: "PipelineServices",
        context: ShardContext,
        control: Path,
        parent_context: ShardContext,
    ) -> None:
        checkpoint_path = control / "pipeline.json"
        quality_path = control / "quality.json"
        if checkpoint_path.exists():
            if not quality_path.exists():
                raise InfrastructureError(
                    f"parallel chunk checkpoint lacks quality ledger: {control}"
                )
            return
        parent = self.parent_services
        parent_checkpoint = parent.runner.load_checkpoint(
            parent._checkpoint_path(parent_context),
            parent_context.shard_id,
            parent_context.gate,
        )
        assets = tuple(
            _read_regular_bytes_nofollow(context.instances)
            .decode("ascii")
            .splitlines()
        )
        seeded = {
            asset: parent_checkpoint.quality_outcomes[asset]
            for asset in assets
            if asset in parent_checkpoint.quality_outcomes
        }
        parent_ledger = _load_quality_ledger(
            parent._quality_ledger_path(parent_context), parent_context
        )
        child_ledger = _empty_quality_ledger(context)
        child_ledger["quarantine"] = {
            asset: dict(parent_ledger["quarantine"][asset])
            for asset in assets
            if asset in parent_ledger["quarantine"]
        }
        child_ledger["family_exclusions"] = {
            asset: {
                family: dict(record)
                for family, record in parent_ledger["family_exclusions"][asset].items()
            }
            for asset in assets
            if asset in parent_ledger["family_exclusions"]
        }
        _save_quality_ledger(quality_path, child_ledger)
        child_checkpoint = PipelineCheckpoint(
            context.shard_id,
            quality_outcomes=seeded,
            gate=context.gate,
        )
        service.runner.save_checkpoint(checkpoint_path, child_checkpoint)

    def _select_builder(self, service: "PipelineServices", stage: StageSpec) -> None:
        try:
            names = frozenset(self.stage_commands[stage.name])
        except KeyError as error:
            raise InfrastructureError(
                f"parallel stage has no command mapping: {stage.name}"
            ) from error

        def builder(context, config, profile):
            return tuple(
                command
                for command in build_preprocessing_dag(context, config, profile)
                if command.name in names
            )

        service.runner.command_builder = builder

    def execute(
        self, chunk: ChunkContext, stage: StageSpec
    ) -> Mapping[str, float]:
        service, context = self._service(chunk)
        self._select_builder(service, stage)
        started = time.monotonic()
        service.runner.run_shard(context)
        return {"elapsed_seconds": time.monotonic() - started}

    def validate(self, chunk: ChunkContext, stage: StageSpec) -> bool:
        service, context = self._service(chunk)
        self._select_builder(service, stage)
        return service.runner.validate_completed_commands(
            context, self.stage_commands[stage.name]
        )

    def worker_profile(
        self, chunk: ChunkContext, _stage: StageSpec
    ) -> Mapping[str, int]:
        service, _context = self._service(chunk)
        profile = service.runner.worker_profile
        return asdict(profile) if profile is not None else {}

    def service_context(
        self, chunk: ChunkContext
    ) -> tuple["PipelineServices", ShardContext]:
        return self._service(chunk)


class PipelineServices:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        resource_guard=None,
        pilot_reader: PilotReader | None = None,
        reference_counter: RawReferenceCounter | None = None,
        project_accounting: ProjectAccounting | None = None,
        registry_store=None,
        disk_usage: Callable[[Path], object] = shutil.disk_usage,
        runner=None,
        output_validator: Callable[[ShardContext], None] | None = None,
        asset_output_validator: Callable[[ShardContext, str], None]
        | None = None,
        family_output_validator: Callable[[ShardContext, str, str], None]
        | None = None,
        shape_resolution_validator: Callable[[ShardContext, int], None]
        | None = None,
        pbr_resolution_validator: Callable[[ShardContext, int], None]
        | None = None,
        resolution_validator: Callable[[ShardContext, int], None]
        | None = None,
        pack_publisher=publish_pack,
        pack_member_builder: Callable[
            [ShardContext], Mapping[str, Sequence[Path]]
        ]
        | None = None,
        published_batch_verifier: Callable[[ShardContext], None]
        | None = None,
        raw_archive_verifier: Callable[[ShardContext], None] | None = None,
        batch_auditor: Callable[[ShardContext], None] | None = None,
        registry_builder: Callable[[], object] | None = None,
        report_builder: Callable[..., object] | None = None,
        tool_commit: str | None = None,
        parallel_scheduler_factory: Callable[[ShardContext], object]
        | None = None,
        telemetry_flush: Callable[[], None] | None = None,
    ):
        self.config = config
        self.registry = registry_store or RegistryStore(
            config.paths.data2_root / "control" / "assets.parquet"
        )
        self._registry_frame_cache = None
        self._registry_frame_lock = threading.Lock()
        self.resource_guard = (
            resource_guard
            if resource_guard is not None
            else _MissingResourceGuard()
        )
        self.pilot_reader = pilot_reader or _MissingPilotReader()
        self.reference_counter = (
            reference_counter
            if reference_counter is not None
            else _MissingReferenceCounter()
        )
        self.project_accounting = (
            project_accounting
            if project_accounting is not None
            else _MissingProjectAccounting()
        )
        self.disk_usage = disk_usage
        self.asset_output_validator = (
            asset_output_validator or self._validate_asset_outputs
        )
        self.family_output_validator = (
            family_output_validator or self._validate_family_output
        )
        self._legacy_asset_output_validation = (
            asset_output_validator is not None
            and family_output_validator is None
        )
        self.output_validator = (
            output_validator or self._validate_terminal_outputs
        )
        self.shape_resolution_validator = (
            shape_resolution_validator
            or resolution_validator
            or self._validate_shape_resolution
        )
        self.pbr_resolution_validator = (
            pbr_resolution_validator
            or resolution_validator
            or self._validate_pbr_resolution
        )
        self.pack_publisher = pack_publisher
        self.pack_member_builder = pack_member_builder or self._pack_members
        self.published_batch_verifier = (
            published_batch_verifier or self._verify_published_batch
        )
        self.raw_archive_verifier = (
            raw_archive_verifier or self._verify_raw_archive
        )
        self.batch_auditor = batch_auditor or self._audit_batch
        self.registry_builder = registry_builder
        self.report_builder = report_builder
        self.parallel_scheduler_factory = (
            parallel_scheduler_factory or self._build_parallel_scheduler
        )
        if telemetry_flush is not None and not callable(telemetry_flush):
            raise ValueError("telemetry flush provider must be callable")
        self.telemetry_flush = telemetry_flush
        if tool_commit is not None and (
            not isinstance(tool_commit, str) or not tool_commit
        ):
            raise ValueError("tool_commit must be non-empty")
        self._tool_commit = tool_commit

        self.validators: dict[str, Callable[[], bool]] = {}
        for command in build_preprocessing_dag(
            ShardContext.for_test(Path("/nonexistent"), "ABO", "ABO-00000"),
            config,
        ):
            self.validators[command.name] = (
                lambda name=command.name: self._validate_command(name)
            )
        self.internal_handlers: dict[str, Callable[[], None]] = {
            "stage_raw": lambda: self.stage_raw(self._active_context()),
            "cleanup_voxels_256": lambda: self.cleanup_voxels(
                self._active_context(), 256
            ),
            "cleanup_voxels_512": lambda: self.cleanup_voxels(
                self._active_context(), 512
            ),
            "cleanup_voxels_1024": lambda: self.cleanup_voxels(
                self._active_context(), 1024
            ),
            "validate_outputs": lambda: self.validate_outputs(
                self._active_context()
            ),
            "build_packs": lambda: self.build_packs(self._active_context()),
            "archive_raw": lambda: self.archive_raw(self._active_context()),
            "cleanup_local": lambda: self.cleanup_local(
                self._active_context()
            ),
        }
        self.runner = runner or PipelineRunner(
            config,
            self.resource_guard,
            self.validators,
            self.internal_handlers,
            report_writer=self._write_escalation,
            checkpoint_path=self._checkpoint_path,
            quality_ledger_path=self._quality_ledger_path,
        )

    def _build_parallel_scheduler(self, context: ShardContext):
        commands: dict[str, tuple[str, ...]] = {
            "prepare": (
                "stage_raw",
                "dump_mesh",
                "dump_pbr",
                "asset_stats",
            ),
            "render": ("render_cond",),
        }
        stages = [
            StageSpec(
                "prepare",
                Lane.PREPARE,
                cpu_cores=self.config.parallelism.cpu_physical_cores,
            ),
            StageSpec(
                "render",
                Lane.RENDER,
                dependencies=("prepare",),
                gpu_indices=tuple(range(self.config.parallelism.gpu_count)),
                gpu_memory_percent=20.0,
            ),
        ]
        dependency = "render"
        for resolution in self.config.targets.resolutions:
            geometry = f"geometry_{resolution}"
            encode = f"encode_{resolution}"
            commands[geometry] = (
                f"dual_grid_{resolution}",
                f"voxelize_pbr_{resolution}",
            )
            commands[encode] = (
                f"encode_shape_{resolution}",
                f"encode_pbr_{resolution}",
                f"cleanup_voxels_{resolution}",
            )
            stages.extend(
                (
                    StageSpec(
                        geometry,
                        Lane.GEOMETRY,
                        dependencies=(dependency,),
                        cpu_cores=(
                            self.config.parallelism.cpu_physical_cores
                        ),
                    ),
                    StageSpec(
                        encode,
                        Lane.ENCODE,
                        dependencies=(geometry,),
                        gpu_indices=tuple(
                            range(self.config.parallelism.gpu_count)
                        ),
                        gpu_memory_percent=float(
                            self.config.parallelism
                            .gpu_memory_target_percent
                        ),
                    ),
                )
            )
            dependency = encode
        final = "finalize"
        commands[final] = (
            f"encode_ss_{self.config.targets.ss_resolution}",
            "validate_outputs",
        )

        stages.append(
            StageSpec(
                final,
                Lane.ENCODE,
                dependencies=(dependency,),
                gpu_indices=tuple(range(self.config.parallelism.gpu_count)),
                gpu_memory_percent=float(
                    self.config.parallelism.gpu_memory_target_percent
                ),
            )
        )

        gate_reader = getattr(
            self.pilot_reader, "p95_peak_local_bytes_for_gate", None
        )
        p95 = (
            gate_reader(context.source, context.gate)
            if gate_reader is not None
            else self.pilot_reader.p95_peak_local_bytes(context.source)
        )
        if type(p95) is not int or p95 <= 0:
            raise IntegrationProviderRequired(
                "pilot reader must return a validated positive integer p95"
            )
        usage = self.disk_usage(self.config.paths.local_root)
        total = getattr(usage, "total", None)
        free = getattr(usage, "free", None)
        if (
            type(total) is not int
            or total <= 0
            or type(free) is not int
            or free < 0
            or free > total
        ):
            raise InfrastructureError("invalid local filesystem usage")
        reserve = max((total * 15 + 99) // 100, 120 * 1024**3)
        usable = max(0, free - reserve)
        per_chunk_budget = usable // self.config.parallelism.max_chunks_in_flight
        chunk_assets = choose_chunk_assets(
            configured=self.config.parallelism.chunk_assets,
            p95_scratch_bytes=p95,
            usable_bytes=per_chunk_budget,
        )
        minimum = (p95 * chunk_assets * 5 + 3) // 4
        if minimum > per_chunk_budget:
            raise InfrastructureError(
                "local scratch budget cannot admit a 32-asset parallel chunk"
            )

        checkpoint_root = (
            self._checkpoint_path(context).parent
            / "chunks"
            / context.batch_id
        )
        executor = _ParallelChunkExecutor(self, checkpoint_root, commands)
        return ParallelChunkScheduler(
            config_hash=self.config.config_hash(),
            broker=NodeResourceBroker(
                cpu_limit=self.config.parallelism.cpu_physical_cores,
                gpu_count=self.config.parallelism.gpu_count,
                gpu_hard_percent=(
                    self.config.parallelism.gpu_memory_hard_percent
                ),
            ),
            executor=executor,
            stages=tuple(stages),
            checkpoint_root=checkpoint_root,
            chunk_assets=chunk_assets,
            max_chunks_in_flight=(
                self.config.parallelism.max_chunks_in_flight
            ),
            promoter=lambda parent, chunk: self._promote_parallel_chunk(
                parent, chunk, executor
            ),
            publisher=lambda parent, chunks: self._publish_parallel_batch(
                parent, chunks, executor
            ),
        )

    def _run_parallel_parent_download(self, context: ShardContext) -> None:
        original_builder = self.runner.command_builder

        def download_builder(candidate, config, profile):
            return tuple(
                command
                for command in build_preprocessing_dag(
                    candidate, config, profile
                )
                if command.name == "download"
            )

        self.runner.command_builder = download_builder
        try:
            self.runner.run_shard(context)
        finally:
            self.runner.command_builder = original_builder

    @staticmethod
    def _regular_digest(path: Path) -> str:
        try:
            details = path.lstat()
        except OSError as error:
            raise InfrastructureError(
                f"cannot inspect parallel artifact: {path}: {error}"
            ) from error
        if not stat.S_ISREG(details.st_mode):
            raise InfrastructureError(
                f"parallel artifact is not a regular file: {path}"
            )
        with path.open("rb") as stream:
            return _sha_stream(stream)

    def _link_parallel_tree(
        self,
        source_root: Path,
        destination_root: Path,
        *,
        skip: frozenset[Path] = frozenset(),
        record_prefix: str | None = None,
        records_only: bool = False,
    ) -> None:
        if not source_root.exists():
            return
        if source_root.is_symlink() or not source_root.is_dir():
            raise InfrastructureError(
                f"unsafe parallel source tree: {source_root}"
            )
        for source in sorted(source_root.rglob("*")):
            relative = source.relative_to(source_root)
            if relative in skip:
                continue
            if records_only and "new_records" not in relative.parts:
                continue
            details = source.lstat()
            if stat.S_ISLNK(details.st_mode):
                raise InfrastructureError(
                    f"unsafe symlink in parallel source tree: {source}"
                )
            if stat.S_ISDIR(details.st_mode):
                continue
            if not stat.S_ISREG(details.st_mode):
                raise InfrastructureError(
                    f"unsafe file type in parallel source tree: {source}"
                )
            target_relative = relative
            if record_prefix is not None and source.parent.name == "new_records":
                target_relative = (
                    relative.parent / f"{record_prefix}{source.name}"
                )
            destination = destination_root / target_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination, follow_symlinks=False)
            except FileExistsError:
                if self._regular_digest(source) != self._regular_digest(
                    destination
                ):
                    raise InfrastructureError(
                        f"conflicting parallel artifact: {destination}"
                    )
            except OSError as error:
                raise InfrastructureError(
                    f"cannot hard-link parallel artifact {source}: {error}"
                ) from error

    def _promote_parallel_chunk(
        self,
        parent: ShardContext,
        chunk: ChunkContext,
        executor: _ParallelChunkExecutor,
    ) -> None:
        promote_chunk_outputs(parent, chunk)
        self._link_parallel_tree(
            chunk.download_root,
            parent.download_root,
            skip=frozenset({Path("raw/metadata.csv")}),
        )
        self._link_parallel_tree(
            chunk.work_root,
            parent.work_root,
            record_prefix=f"{chunk.chunk_id}_",
            records_only=True,
        )
        service, context = executor.service_context(chunk)
        checkpoint = service.runner.load_checkpoint(
            service.runner.checkpoint_path(context),
            context.shard_id,
            context.gate,
        )
        ledger = _load_quality_ledger(
            service.runner.quality_ledger_path(context), context
        )
        dependencies = family_dependencies(self.config)
        for asset in chunk.assets():
            if checkpoint.quality_outcomes.get(asset) != "completed":
                continue
            self.family_output_validator(parent, asset, "common")
            excluded = set(ledger["family_exclusions"].get(asset, {}))

            def eligible(family: str, visiting: frozenset[str]) -> bool:
                if family in excluded:
                    return False
                if family in visiting:
                    raise InfrastructureError(
                        f"cyclic promoted family dependency: {family}"
                    )
                return all(
                    eligible(dependency, visiting | {family})
                    for dependency in dependencies[family]
                )

            for family in PACK_FAMILIES:
                if family != "common" and eligible(family, frozenset()):
                    self.family_output_validator(parent, asset, family)

    def _merge_parallel_quality(
        self,
        parent: ShardContext,
        chunks: Sequence[ChunkContext],
        executor: _ParallelChunkExecutor,
    ) -> None:
        assets = self._instances(parent)
        outcomes: dict[str, str] = {}
        quarantine: dict[str, dict[str, object]] = {}
        family_exclusions: dict[
            str, dict[str, dict[str, object]]
        ] = {}
        for chunk in chunks:
            service, context = executor.service_context(chunk)
            checkpoint = service.runner.load_checkpoint(
                service.runner.checkpoint_path(context),
                context.shard_id,
                context.gate,
            )
            chunk_assets = chunk.assets()
            if set(checkpoint.quality_outcomes) != set(chunk_assets):
                raise InfrastructureError(
                    f"parallel chunk lacks terminal quality outcomes: {chunk.chunk_id}"
                )
            for asset in chunk_assets:
                if asset in outcomes:
                    raise InfrastructureError(
                        f"duplicate parallel quality asset: {asset}"
                    )
                outcomes[asset] = checkpoint.quality_outcomes[asset]
            ledger = _load_quality_ledger(
                service.runner.quality_ledger_path(context), context
            )
            for asset, record in ledger["quarantine"].items():
                existing = quarantine.get(asset)
                if existing is not None and existing != record:
                    raise InfrastructureError(
                        f"conflicting parallel quarantine: {asset}"
                    )
                quarantine[asset] = dict(record)
            for asset, exclusions in ledger["family_exclusions"].items():
                target = family_exclusions.setdefault(asset, {})
                for family, record in exclusions.items():
                    existing = target.get(family)
                    if existing is not None and existing != record:
                        raise InfrastructureError(
                            f"conflicting parallel family exclusion: {asset}/{family}"
                        )
                    target[family] = dict(record)
        if set(outcomes) != set(assets):
            raise InfrastructureError(
                "parallel quality outcomes do not cover the frozen batch"
            )
        ordered_outcomes = {asset: outcomes[asset] for asset in assets}
        parent_checkpoint_path = self._checkpoint_path(parent)
        parent_checkpoint = self.runner.load_checkpoint(
            parent_checkpoint_path, parent.shard_id, parent.gate
        )
        for asset, outcome in parent_checkpoint.quality_outcomes.items():
            if ordered_outcomes.get(asset) != outcome:
                raise InfrastructureError(
                    f"conflicting parent quality outcome: {asset}"
                )
        parent_checkpoint.quality_outcomes = ordered_outcomes
        parent_checkpoint.active_attempt = None
        self.runner.save_checkpoint(parent_checkpoint_path, parent_checkpoint)

        ledger_path = self._quality_ledger_path(parent)
        ledger = _load_quality_ledger(ledger_path, parent)
        prior_entries = [
            entry
            for entry in ledger["entries"]
            if entry["batch_id"] != parent.batch_id
        ]
        current_entries = [
            {
                "batch_id": parent.batch_id,
                "position": position,
                "asset_sha": asset,
                "outcome": ordered_outcomes[asset],
            }
            for position, asset in enumerate(assets)
        ]
        next_quarantine = {
            asset: dict(record)
            for asset, record in ledger["quarantine"].items()
            if asset not in set(assets)
        }
        next_quarantine.update(quarantine)
        next_exclusions = {
            asset: {
                family: dict(record)
                for family, record in exclusions.items()
            }
            for asset, exclusions in ledger["family_exclusions"].items()
            if asset not in set(assets)
        }
        next_exclusions.update(family_exclusions)
        next_batches = dict(ledger["batches"])
        next_batches[parent.batch_id] = {
            "instances_sha256": self._asset_scope_sha256(assets),
            "admitted_prefix": len(assets),
        }
        next_ledger = {
            **ledger,
            "batches": next_batches,
            "entries": (prior_entries + current_entries)[
                -QUALITY_WINDOW_SIZE:
            ],
            "quarantine": next_quarantine,
            "family_exclusions": next_exclusions,
        }
        _save_quality_ledger(ledger_path, next_ledger)

    def _write_parallel_raw_metadata(
        self,
        parent: ShardContext,
        chunks: Sequence[ChunkContext],
        executor: _ParallelChunkExecutor,
    ) -> None:
        assets = self._instances(parent)
        records = {}
        for chunk in chunks:
            service, context = executor.service_context(chunk)
            for record in service._staged_records(context):
                asset = _validated_asset_sha(record.get("sha256"))
                if asset in records and records[asset] != record:
                    raise InfrastructureError(
                        f"conflicting parallel raw record: {asset}"
                    )
                records[asset] = record
        if set(records) != set(assets):
            raise InfrastructureError(
                "parallel raw metadata does not cover the frozen batch"
            )
        self._write_raw_records(
            parent.download_root / "raw/metadata.csv",
            tuple(records[asset] for asset in assets),
        )

    def _publish_parallel_batch(
        self,
        parent: ShardContext,
        chunks: tuple[ChunkContext, ...],
        executor: _ParallelChunkExecutor,
    ) -> None:
        self._merge_parallel_quality(parent, chunks, executor)
        already_durable = self._published_is_valid(
            parent
        ) and self._archive_is_valid(parent)
        if not already_durable:
            self._write_parallel_raw_metadata(parent, chunks, executor)
        publication_names = frozenset(
            {"validate_outputs", "build_packs", "archive_raw", "cleanup_local"}
        )
        original_builder = self.runner.command_builder

        def publication_builder(context, config, profile):
            return tuple(
                command
                for command in build_preprocessing_dag(context, config, profile)
                if command.name in publication_names
            )

        self.runner.command_builder = publication_builder
        try:
            self.runner.run_shard(parent)
        finally:
            self.runner.command_builder = original_builder
        for chunk in chunks:
            root = chunk.work_root.parent
            if root.exists():
                shutil.rmtree(root)
        if chunks:
            chunks_root = chunks[0].work_root.parent.parent
            try:
                chunks_root.rmdir()
            except FileNotFoundError:
                pass
            except OSError as error:
                if error.errno != errno.ENOTEMPTY:
                    raise

    def _execute_batch(
        self,
        context: ShardContext,
        assets: Sequence[str],
        *,
        resume: bool,
    ) -> None:
        if (
            context.gate == "production"
            and len(assets) > self.config.parallelism.chunk_assets
        ):
            self._run_parallel_parent_download(context)
            scheduler = self.parallel_scheduler_factory(context)
            scheduler.run_batch(context, tuple(assets))
            return
        if resume:
            self.runner.resume_shard(context)
        else:
            self.runner.run_shard(context)

    def _active_context(self) -> ShardContext:
        context = getattr(self.runner, "active_context", None)
        if context is None:
            raise InfrastructureError("internal handler has no active shard")
        return context

    def _checkpoint_path(self, context: ShardContext) -> Path:
        root = self.config.paths.data2_root / "control"
        if context.gate == "production":
            root = root / "checkpoints"
        else:
            root = root / "qualification" / context.gate / "checkpoints"
        return root / context.source / context.shard_id / f"{context.batch_id}.json"

    def _quality_ledger_path(self, context: ShardContext) -> Path:
        root = self.config.paths.data2_root / "control"
        if context.gate == "production":
            root = root / "quality"
        else:
            root = root / "qualification" / context.gate / "quality"
        return root / context.source / f"{context.shard_id}.json"

    def _write_escalation(self, report: EscalationReport) -> None:
        root = self.config.paths.data2_root / "control/reports/escalations"
        if report.gate != "production":
            root = root / "qualification" / report.gate
        path = root / report.source / report.shard_id / f"{report.command}.json"
        payload = json.dumps(
            asdict(report), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        _atomic_write_bytes_nofollow(path, payload)

    def _resolved_tool_commit(self) -> str:
        if self._tool_commit is not None:
            return self._tool_commit
        deployed_commit = os.environ.get("PIXAL3D_TOOL_COMMIT")
        if deployed_commit is not None:
            value = deployed_commit.strip().lower()
            if (
                len(value) != 40
                or any(
                    character not in "0123456789abcdef"
                    for character in value
                )
            ):
                raise InfrastructureError(
                    "invalid PIXAL3D_TOOL_COMMIT deployment identity"
                )
            self._tool_commit = value
            return value
        try:
            completed = subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError, UnicodeError) as error:
            raise InfrastructureError(
                f"cannot resolve tool Git commit: {error}"
            ) from error
        value = completed.stdout.strip().lower()
        if (
            len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise InfrastructureError(
                f"invalid tool Git commit from repository: {value!r}"
            )
        self._tool_commit = value
        return value

    def _registry_shas(
        self, source: str, shard_id: str, count: int | None = None
    ) -> tuple[str, ...]:
        _safe_component(source, "source")
        _safe_component(shard_id, "shard id")
        if count is not None and (
            not isinstance(count, int) or isinstance(count, bool) or count <= 0
        ):
            raise ValueError("count must be a positive integer")
        with self._registry_frame_lock:
            if self._registry_frame_cache is None:
                self._registry_frame_cache = self.registry.load()
            frame = self._registry_frame_cache
        required = {"sha256", "owner_source", "shard_id"}
        if not required.issubset(frame.columns):
            raise InfrastructureError(
                f"registry is missing required columns: {sorted(required - set(frame.columns))}"
            )
        selected = frame.loc[
            (frame["owner_source"] == source)
            & (frame["shard_id"] == shard_id),
            "sha256",
        ]
        shas = tuple(sorted(_validated_asset_sha(item) for item in selected))
        if len(shas) != len(set(shas)):
            raise InfrastructureError("registry shard contains duplicate SHA-256")
        if not shas:
            raise InfrastructureError(
                f"registry shard has no assets: {source}/{shard_id}"
            )
        return shas[:count] if count is not None else shas

    def _batch_root(self, gate: str, source: str, shard_id: str) -> Path:
        if gate not in {"smoke", "pilot", "production"}:
            raise ValueError(f"unknown gate: {gate}")
        root = self.config.paths.data2_root / "control"
        if gate == "production":
            root = root / "shards"
        else:
            root = root / "qualification" / gate / "shards"
        return root / source / shard_id

    @staticmethod
    def _batch_file_payload(batch: tuple[str, ...]) -> str:
        return "".join(f"{item}\n" for item in batch)

    @classmethod
    def _asset_scope_sha256(cls, assets: tuple[str, ...]) -> str:
        return sha256(cls._batch_file_payload(assets).encode("ascii")).hexdigest()

    def _read_frozen_batches(
        self,
        gate: str,
        source: str,
        shard_id: str,
        canonical_shas: tuple[str, ...],
        *,
        expected_scope: tuple[str, ...] | None = None,
    ) -> tuple[tuple[str, ...], ...] | None:
        root = self._batch_root(gate, source, shard_id)
        try:
            root_fd = _open_directory_nofollow(root)
        except FileNotFoundError:
            return None
        except OSError as error:
            raise InfrastructureError(
                f"unsafe frozen batch root: {root}: {error}"
            ) from error
        try:
            actual_names = sorted(
                name
                for name in os.listdir(root_fd)
                if name.startswith("batch") and name.endswith(".txt")
            )
        finally:
            os.close(root_fd)
        marker_path = root / "batches.json"
        try:
            marker = json.loads(_read_regular_bytes_nofollow(marker_path))
        except InfrastructureError:
            raise
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
        ) as error:
            raise InfrastructureError(
                f"invalid frozen batch manifest: {marker_path}: {error}"
            ) from error
        if (
            not isinstance(marker, dict)
            or set(marker)
            != {
                "schema_version",
                "gate",
                "source",
                "shard_id",
                "config_hash",
                "canonical_shard_sha256",
                "scope_sha256",
                "batches",
            }
            or marker["schema_version"] != 2
            or marker["gate"] != gate
            or marker["source"] != source
            or marker["shard_id"] != shard_id
            or marker["config_hash"] != self.config.config_hash()
            or marker["canonical_shard_sha256"]
            != self._asset_scope_sha256(canonical_shas)
            or not isinstance(marker["batches"], list)
            or not marker["batches"]
        ):
            raise InfrastructureError(
                f"invalid frozen batch manifest: {marker_path}"
            )

        batches = []
        expected_names = []
        for index, entry in enumerate(marker["batches"]):
            name = f"batch{index:03d}.txt"
            expected_names.append(name)
            path = root / name
            if (
                not isinstance(entry, dict)
                or set(entry) != {"name", "count", "sha256"}
                or entry["name"] != name
                or not isinstance(entry["count"], int)
                or isinstance(entry["count"], bool)
                or entry["count"] <= 0
                or not isinstance(entry["sha256"], str)
            ):
                raise InfrastructureError(
                    f"invalid frozen batch entry: {marker_path}: {name}"
                )
            try:
                payload = _read_regular_bytes_nofollow(path)
                if sha256(payload).hexdigest() != entry["sha256"]:
                    raise InfrastructureError(
                        f"frozen batch checksum mismatch: {path}"
                    )
                text = payload.decode("ascii")
                if not text.endswith("\n"):
                    raise ValueError("missing final newline")
                batch = tuple(
                    _validated_asset_sha(item) for item in text.splitlines()
                )
            except InfrastructureError:
                raise
            except (OSError, UnicodeDecodeError, ValueError) as error:
                raise InfrastructureError(
                    f"invalid frozen batch: {path}: {error}"
                ) from error
            if (
                len(batch) != entry["count"]
                or len(batch) > self.config.shard_size
                or tuple(sorted(batch)) != batch
                or len(batch) != len(set(batch))
            ):
                raise InfrastructureError(f"invalid frozen batch: {path}")
            batches.append(batch)
        if actual_names != expected_names:
            raise InfrastructureError(
                f"frozen batch file set mismatch: {root}"
            )
        flattened = tuple(item for batch in batches for item in batch)
        if marker["scope_sha256"] != self._asset_scope_sha256(flattened):
            raise InfrastructureError(
                f"frozen batch scope checksum mismatch: {root}"
            )
        if gate == "production":
            identity_valid = flattened == canonical_shas
        else:
            expected_set = set(canonical_shas)
            flattened_set = set(flattened)
            identity_valid = (
                bool(flattened)
                and flattened_set.issubset(expected_set)
                and flattened
                == tuple(
                    item for item in canonical_shas if item in flattened_set
                )
            )
        if expected_scope is not None:
            identity_valid = identity_valid and flattened == expected_scope
        if not identity_valid:
            raise InfrastructureError(
                f"frozen batch asset identity mismatch: {root}"
            )
        return tuple(batches)

    def _freeze_batches(
        self,
        gate: str,
        source: str,
        shard_id: str,
        batches: tuple[tuple[str, ...], ...],
        canonical_shas: tuple[str, ...],
    ) -> tuple[tuple[str, ...], ...]:
        destination = self._batch_root(gate, source, shard_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        lock_fd = None
        try:
            temporary = Path(
                tempfile.mkdtemp(
                    prefix=f".{shard_id}.freeze-", dir=destination.parent
                )
            )
            lock_fd = os.open(
                destination.parent / f".{shard_id}.freeze.lock",
                os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
            )
            entries = []
            batch_files = []
            for index, batch in enumerate(batches):
                name = f"batch{index:03d}.txt"
                payload = self._batch_file_payload(batch)
                batch_files.append((temporary / name, payload))
                entries.append(
                    {
                        "name": name,
                        "count": len(batch),
                        "sha256": sha256(payload.encode("ascii")).hexdigest(),
                    }
                )
            with ThreadPoolExecutor(
                max_workers=max(1, min(16, len(batch_files))),
                thread_name_prefix="pixal3d-freeze",
            ) as executor:
                tuple(
                    executor.map(
                        lambda item: _atomic_write_text(*item),
                        batch_files,
                    )
                )
            marker = {
                "schema_version": 2,
                "gate": gate,
                "source": source,
                "shard_id": shard_id,
                "config_hash": self.config.config_hash(),
                "canonical_shard_sha256": self._asset_scope_sha256(
                    canonical_shas
                ),
                "scope_sha256": self._asset_scope_sha256(
                    tuple(item for batch in batches for item in batch)
                ),
                "batches": entries,
            }
            _atomic_write_bytes_nofollow(
                temporary / "batches.json",
                json.dumps(
                    marker, sort_keys=True, separators=(",", ":")
                ).encode("utf-8"),
            )
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if destination.exists() or destination.is_symlink():
                raise InfrastructureError(
                    f"frozen batch destination already exists: {destination}"
                )
            os.rename(temporary, destination)
            temporary = None
            directory = os.open(
                destination.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return batches
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)

    def _planned_batches(
        self,
        gate: str,
        source: str,
        shard_id: str,
        count: int | None,
        *,
        freeze: bool,
    ) -> tuple[tuple[str, ...], ...]:
        canonical_shas = self._registry_shas(source, shard_id)
        shas = canonical_shas[:count] if count is not None else canonical_shas
        if gate == "production" and shas != canonical_shas:
            raise ValueError("production requires the full canonical shard")
        existing = self._read_frozen_batches(
            gate,
            source,
            shard_id,
            canonical_shas,
            expected_scope=shas,
        )
        if existing is not None:
            return existing
        if not freeze and self._batch_root(gate, source, shard_id).exists():
            raise InfrastructureError("incomplete frozen batch publication")
        gate_reader = getattr(
            self.pilot_reader, "p95_peak_local_bytes_for_gate", None
        )
        if gate_reader is None:
            p95 = self.pilot_reader.p95_peak_local_bytes(source)
        else:
            p95 = gate_reader(source, gate)
        if not isinstance(p95, int) or isinstance(p95, bool) or p95 <= 0:
            raise IntegrationProviderRequired(
                "pilot reader must return a validated positive integer p95"
            )
        usage = self.disk_usage(self.config.paths.local_root)
        total = getattr(usage, "total", None)
        free = getattr(usage, "free", None)
        if (
            not isinstance(total, int)
            or isinstance(total, bool)
            or total <= 0
            or not isinstance(free, int)
            or isinstance(free, bool)
            or free < 0
            or free > total
        ):
            raise InfrastructureError("invalid local filesystem usage")
        reserve = max((total * 15 + 99) // 100, 120 * 1024**3)
        usable = max(0, free - reserve)
        cap = {
            "smoke": self.config.batching.smoke_max_assets,
            "pilot": self.config.batching.pilot_max_assets,
            "production": self.config.batching.production_max_assets,
        }[gate]
        batches = plan_work_batches(
            shas, usable, p95, self.config.shard_size, cap
        )
        return (
            self._freeze_batches(gate, source, shard_id, batches, canonical_shas)
            if freeze
            else batches
        )

    def plan(
        self,
        gate: str,
        source: str | None = None,
        shard: str | None = None,
        count: int | None = None,
        *,
        freeze: bool = False,
    ) -> tuple[str, ...]:
        if gate not in {"smoke", "pilot", "production"}:
            raise ValueError(f"unknown gate: {gate}")
        if source is None and shard is None:
            context = ShardContext.for_test(
                Path("/nonexistent"), self.config.sources[0], "plan-00000"
            )
            return tuple(
                command.name
                for command in build_preprocessing_dag(context, self.config)
            )
        if source is None or shard is None:
            raise ValueError("source and shard must be provided together")
        batches = self._planned_batches(
            gate, source, shard, count, freeze=freeze
        )
        return tuple(
            f"batch{index:03d}: {len(batch)} assets"
            for index, batch in enumerate(batches)
        )

    @staticmethod
    def _instances(context: ShardContext) -> tuple[str, ...]:
        try:
            payload = _read_regular_bytes_nofollow(context.instances)
            values = tuple(
                _validated_asset_sha(item)
                for item in payload.decode("ascii").splitlines()
            )
        except OSError as error:
            if error.errno in {errno.ENOENT, errno.ENOTDIR, errno.ELOOP}:
                raise ValidationError(
                    f"invalid instances manifest: {context.instances}: {error}"
                ) from error
            raise
        except (UnicodeDecodeError, TypeError, ValueError) as error:
            raise ValidationError(
                f"invalid instances manifest: {context.instances}: {error}"
            ) from error
        if not values or len(values) != len(set(values)) or tuple(sorted(values)) != values:
            raise ValidationError(
                f"invalid instances manifest: {context.instances}"
            )
        return values

    @staticmethod
    def _read_raw_record_map(
        path: Path, selected: tuple[str, ...]
    ) -> dict[str, dict]:
        try:
            payload = _read_regular_bytes_nofollow(path)
            with io.StringIO(payload.decode("utf-8"), newline="") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames is None or not {
                    "sha256",
                    "local_path",
                }.issubset(reader.fieldnames):
                    raise ValidationError(
                        f"raw metadata missing required columns: {path}"
                    )
                by_sha = {}
                for row in reader:
                    sha = row.get("sha256")
                    if sha not in selected:
                        continue
                    if sha in by_sha:
                        raise ValidationError(
                            f"duplicate raw metadata SHA-256: {sha}"
                        )
                    relative = _safe_raw_relative(row.get("local_path", ""))
                    content_sha = _validated_asset_sha(
                        row.get("content_sha256") or sha
                    )
                    companion_value = row.get("companion_files") or "{}"

                    def unique_companions(pairs):
                        value = {}
                        for companion_path, companion_sha in pairs:
                            if companion_path in value:
                                raise ValidationError(
                                    "duplicate raw companion path: "
                                    f"{companion_path}"
                                )
                            value[companion_path] = companion_sha
                        return value

                    companions = json.loads(
                        companion_value, object_pairs_hook=unique_companions
                    )
                    if not isinstance(companions, dict):
                        raise ValidationError(
                            "invalid raw companion mapping"
                        )
                    normalized_companions = {}
                    for companion_path, companion_sha in companions.items():
                        companion_relative = _safe_raw_relative(
                            companion_path
                        ).as_posix()
                        if companion_relative == relative.as_posix():
                            raise ValidationError(
                                "primary raw path repeated as companion"
                            )
                        normalized_companions[companion_relative] = (
                            _validated_asset_sha(companion_sha)
                        )
                    by_sha[sha] = {
                        "sha256": sha,
                        "local_path": relative.as_posix(),
                        "content_sha256": content_sha,
                        "companion_files": normalized_companions,
                    }
        except ValidationError:
            raise
        except OSError as error:
            if error.errno in {errno.ENOENT, errno.ENOTDIR, errno.ELOOP}:
                raise ValidationError(
                    f"invalid raw metadata: {path}: {error}"
                ) from error
            raise
        except (UnicodeDecodeError, csv.Error, TypeError, ValueError) as error:
            raise ValidationError(f"invalid raw metadata: {path}: {error}") from error
        PipelineServices._raw_file_map(tuple(by_sha.values()))
        return by_sha

    @staticmethod
    def _raw_file_map(
        records: Sequence[Mapping[str, object]],
    ) -> dict[str, str]:
        files = {}
        for record in records:
            candidates = {
                record["local_path"]: record["content_sha256"],
                **record["companion_files"],
            }
            for path, digest in candidates.items():
                if path in files:
                    raise ValidationError(
                        f"duplicate selected raw path: {path}"
                    )
                files[path] = digest
        return files

    @classmethod
    def _read_raw_records(
        cls, path: Path, selected: tuple[str, ...]
    ) -> tuple[dict, ...]:
        by_sha = cls._read_raw_record_map(path, selected)
        missing = set(selected) - set(by_sha)
        if missing:
            raise ValidationError(
                f"raw metadata missing selected assets: {sorted(missing)}"
            )
        return tuple(by_sha[item] for item in selected)

    @staticmethod
    def _write_raw_records(path: Path, records: tuple[dict, ...]) -> None:
        path = Path(path)
        stream = io.StringIO(newline="")
        try:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "sha256",
                    "local_path",
                    "content_sha256",
                    "companion_files",
                ),
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        **record,
                        "content_sha256": record.get("content_sha256")
                        or record["sha256"],
                        "companion_files": json.dumps(
                            record.get("companion_files", {}),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                )
            _atomic_write_bytes_nofollow(path, stream.getvalue().encode("utf-8"))
        except OSError as error:
            if error.errno in PATH_VALIDATION_ERRNOS:
                raise ValidationError(
                    f"unsafe raw metadata output: {path}"
                ) from error
            raise
        finally:
            stream.close()

    @staticmethod
    def _source_raw_relative(relative: Path) -> Path:
        zip_value = _zip_parts(relative)
        return zip_value[0] if zip_value is not None else relative

    def stage_raw(self, context: ShardContext) -> None:
        selected = self._eligible_assets(context)
        records = self._read_raw_records(
            context.source_root / "raw/metadata.csv", selected
        )
        for relative_value, expected_sha in self._raw_file_map(records).items():
            relative = _safe_raw_relative(relative_value)
            zip_value = _zip_parts(relative)
            if zip_value is None:
                with _open_regular_beneath(
                    context.source_root, relative
                ) as stream:
                    _atomic_stage_stream(
                        stream,
                        context.download_root,
                        relative,
                        expected_sha,
                    )
                continue

            archive_relative, member_name = zip_value
            try:
                with _open_regular_beneath(
                    context.source_root, archive_relative
                ) as archive_stream:
                    with zipfile.ZipFile(archive_stream) as bundle:
                        infos = _safe_zip_infos(bundle)
                        info = infos.get(member_name)
                        if info is None or info.is_dir():
                            raise ValidationError(
                                f"missing selected ZIP member: {member_name}"
                            )
                        with bundle.open(info) as member:
                            _atomic_stage_stream(
                                member,
                                context.download_root,
                                relative,
                                expected_sha,
                            )
            except ValidationError:
                raise
            except OSError:
                raise
            except zipfile.BadZipFile as error:
                raise ValidationError(
                    f"invalid raw ZIP archive: {archive_relative.as_posix()}: {error}"
                ) from error
        self._write_raw_records(
            context.download_root / "raw/metadata.csv", records
        )

    def _staged_records(self, context: ShardContext) -> tuple[dict, ...]:
        return self._read_raw_records(
            context.download_root / "raw/metadata.csv",
            self._eligible_assets(context),
        )

    def _validate_staged_raw(self, context: ShardContext) -> None:
        records = self._staged_records(context)
        for relative_value, expected_sha in self._raw_file_map(records).items():
            relative = _safe_raw_relative(relative_value)
            with _open_regular_beneath(
                context.download_root, relative
            ) as stream:
                if _sha_stream(stream) != expected_sha:
                    raise ValidationError(
                        f"staged raw checksum mismatch: {relative.as_posix()}"
                    )

    def _validate_dump_output(
        self, context: ShardContext, directory: str, asset_sha: str
    ) -> None:
        relative = Path(directory) / f"{asset_sha}.pickle"
        _require_regular_artifact(context.work_root / relative)
        try:
            with _open_regular_beneath(context.work_root, relative) as stream:
                value = pickle.load(stream)
        except ValidationError:
            raise
        except (pickle.PickleError, EOFError, AttributeError, ValueError) as error:
            raise ValidationError(
                f"invalid dump pickle: {relative.as_posix()}: {error}"
            ) from error
        if not isinstance(value, dict) or not isinstance(
            value.get("objects"), list
        ):
            raise ValidationError(
                f"invalid dump structure: {relative.as_posix()}"
            )
        if directory == "pbr_dumps" and not isinstance(
            value.get("materials"), list
        ):
            raise ValidationError(
                f"invalid PBR dump structure: {relative.as_posix()}"
            )

    def _pbr_dump_records(
        self, context: ShardContext
    ) -> Mapping[str, Mapping[str, object]]:
        root = context.work_root / "pbr_dumps/new_records"
        try:
            directory_fd = _open_directory_nofollow(root)
        except OSError as error:
            if error.errno in {errno.ENOENT, errno.ENOTDIR}:
                return {}
            if error.errno == errno.ELOOP:
                raise ValidationError(
                    f"unsafe PBR dump records: {root}"
                ) from error
            raise
        try:
            names = sorted(
                name
                for name in os.listdir(directory_fd)
                if name.startswith("part_") and name.endswith(".csv")
            )
        finally:
            os.close(directory_fd)

        records = {}
        required = {"sha256", "pbr_dumped"}
        for name in names:
            path = root / name
            try:
                reader = csv.DictReader(
                    io.StringIO(
                        _read_regular_bytes_nofollow(path).decode("utf-8"),
                        newline="",
                    )
                )
                if reader.fieldnames is None or not required.issubset(
                    reader.fieldnames
                ):
                    raise ValidationError(
                        f"PBR dump CSV missing required columns: {path}"
                    )
                rows = tuple(reader)
            except (UnicodeDecodeError, csv.Error) as error:
                raise ValidationError(
                    f"invalid PBR dump CSV: {path}: {error}"
                ) from error
            for row in rows:
                try:
                    asset_sha = _validated_asset_sha(row.get("sha256"))
                except (TypeError, ValueError) as error:
                    raise ValidationError(
                        f"invalid PBR dump SHA: {path}"
                    ) from error
                if asset_sha in records:
                    raise ValidationError(
                        f"duplicate PBR dump SHA: {asset_sha}"
                    )
                dumped_value = str(row.get("pbr_dumped", "")).lower()
                if dumped_value not in {"true", "false"}:
                    raise ValidationError(
                        f"invalid PBR dump outcome: {asset_sha}"
                    )
                records[asset_sha] = {
                    "pbr_dumped": dumped_value == "true",
                    "error_category": str(
                        row.get("error_category", "")
                    ).strip(),
                    "error_reason": str(
                        row.get("error_reason", "")
                    ).strip(),
                }
        return records

    def _validate_pbr_dump_stage(self, context: ShardContext) -> None:
        pbr_families = tuple(
            f"PBR-{resolution}"
            for resolution in self.config.targets.resolutions
        )
        assets = tuple(
            asset
            for asset in self._eligible_assets(context)
            if any(
                self.runner.family_is_eligible(asset, family)
                for family in pbr_families
            )
        )
        records = self._pbr_dump_records(context)
        checkpoint = getattr(self.runner, "active_checkpoint", None)
        attempts = (
            checkpoint.attempts.get("dump_pbr", 0)
            if checkpoint is not None
            else 0
        )
        for asset_sha in assets:
            try:
                self._validate_dump_output(
                    context, "pbr_dumps", asset_sha
                )
                continue
            except OutputValidationError as error:
                missing_error = error
            record = records.get(asset_sha, {})
            if record.get("pbr_dumped") is True:
                raise ValidationError(
                    f"PBR record claims missing output is complete: {asset_sha}"
                )
            category = str(
                record.get("error_category", "pbr_dump_failure")
            )
            reason = str(record.get("error_reason", "")).strip()
            if not reason:
                reason = str(missing_error)
            if category == "unsupported_shader" or attempts >= MAX_COMMAND_ATTEMPTS:
                self._exclude_stage_failure(
                    context,
                    asset_sha,
                    "dump_pbr",
                    category=category,
                    reason=reason,
                )
                continue
            raise OutputValidationError(reason)

    @staticmethod
    def _validate_asset_stats_record(asset_sha: str, row: dict | None) -> None:
        if row is None:
            raise OutputValidationError(
                f"missing asset stats record: {asset_sha}"
            )
        try:
            counts = (int(row["num_faces"]), int(row["num_vertices"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValidationError("invalid asset stats counts") from error
        if any(value < 0 for value in counts):
            raise ValidationError("negative asset stats counts")

    def _asset_stats_records(self, context: ShardContext) -> dict[str, dict]:
        root = context.metadata_root / "asset_stats/new_records"
        try:
            directory_fd = _open_directory_nofollow(root)
        except OSError as error:
            if error.errno in {errno.ENOENT, errno.ENOTDIR}:
                raise OutputValidationError(
                    f"missing asset stats parts: {root}"
                ) from error
            if error.errno == errno.ELOOP:
                raise ValidationError(
                    f"unsafe asset stats parts: {root}"
                ) from error
            raise
        try:
            names = sorted(
                name
                for name in os.listdir(directory_fd)
                if name.startswith("part_") and name.endswith(".csv")
            )
        finally:
            os.close(directory_fd)
        if not names:
            raise OutputValidationError(f"missing asset stats parts: {root}")

        records = {}
        required = {"sha256", "num_faces", "num_vertices"}
        for name in names:
            payload = _read_regular_bytes_nofollow(root / name)
            if not payload.strip():
                continue
            try:
                reader = csv.DictReader(
                    io.StringIO(payload.decode("utf-8"), newline="")
                )
                if reader.fieldnames is None or not required.issubset(
                    reader.fieldnames
                ):
                    raise ValidationError(
                        "asset stats CSV missing required columns"
                    )
                rows = tuple(reader)
            except (UnicodeDecodeError, csv.Error) as error:
                raise ValidationError(
                    f"invalid asset stats CSV: {root / name}: {error}"
                ) from error
            for row in rows:
                try:
                    asset_sha = _validated_asset_sha(row.get("sha256"))
                except (TypeError, ValueError) as error:
                    raise ValidationError(
                        f"invalid asset stats SHA: {root / name}"
                    ) from error
                if asset_sha in records:
                    raise ValidationError(
                        f"duplicate asset stats SHA: {asset_sha}"
                    )
                records[asset_sha] = row
        return records

    def _validate_asset_stats(self, context: ShardContext) -> None:
        records = self._asset_stats_records(context)
        assets = self._instances(context)
        if set(records) != set(assets):
            raise ValidationError("asset stats SHA set mismatch")
        for asset_sha in assets:
            self._validate_asset_stats_record(asset_sha, records.get(asset_sha))

    @staticmethod
    def _validate_voxel_output(
        context: ShardContext, directory: str, asset_sha: str, view: int
    ) -> None:
        relative = Path(directory) / asset_sha / f"view{view:02d}.vxz"
        _require_regular_artifact(context.work_root / relative)
        try:
            import o_voxel
        except ImportError as error:
            raise InfrastructureError("o_voxel validator is unavailable") from error
        try:
            with _open_regular_beneath(context.work_root, relative) as stream:
                info = o_voxel.io.read_vxz_info(
                    f"/proc/self/fd/{stream.fileno()}"
                )
        except ValidationError:
            raise
        except OSError:
            raise
        except (
            AssertionError,
            EOFError,
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            raise ValidationError(
                f"invalid voxel output: {relative.as_posix()}: {error}"
            ) from error
        if not isinstance(info, Mapping) or not info:
            raise ValidationError(
                f"invalid voxel metadata: {relative.as_posix()}"
            )
        scale_relative = relative.with_name(f"view{view:02d}_scale.json")
        _require_regular_artifact(context.work_root / scale_relative)
        with _open_regular_beneath(
            context.work_root, scale_relative
        ) as stream:
            try:
                scale = json.loads(stream.read())
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValidationError("invalid voxel scale metadata") from error
        if (
            not isinstance(scale, dict)
            or not scale
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in scale.values()
            )
        ):
            raise ValidationError("invalid voxel scale metadata")

    def _shape_directory(self, resolution: int) -> Path:
        return Path(
            "shape_latents",
            f"shape_enc_next_dc_f16c32_fp16_{resolution}_view",
        )

    def _pbr_directory(self, resolution: int) -> Path:
        return Path(
            "pbr_latents",
            f"tex_enc_next_dc_f16c32_fp16_{resolution}_view_fix",
        )

    def _ss_directory(self) -> Path:
        return Path(
            "ss_latents",
            f"ss_enc_conv3d_16l8_fp16_{self.config.targets.ss_resolution}_view",
        )

    def _validate_render_output(
        self, context: ShardContext, asset_sha: str
    ) -> None:
        root = context.output_root / "renders_cond" / asset_sha
        _require_regular_artifact(root / "transforms.json")
        for view in range(self.config.render.num_views):
            _require_regular_artifact(root / f"{view:03d}.png")
        validate_render_dir(
            root,
            self.config.render.num_views,
            self.config.render.resolution,
        )

    @staticmethod
    def _validate_sparse_output(
        output: Path, resolution: int, *, ss: bool = False
    ) -> None:
        _require_regular_artifact(output)
        scale = output.with_name(f"{output.stem}_scale.json")
        _require_regular_artifact(scale)
        if ss:
            validate_ss_latent(output)
        else:
            validate_sparse_latent(output, resolution, resolution**3)
        validate_scale(scale)

    def _validate_resolution_family(
        self, context: ShardContext, resolution: int, relative: Path
    ) -> None:
        if resolution not in self.config.targets.resolutions:
            raise ValidationError(f"unexpected resolution: {resolution}")
        for asset_sha in self._instances(context):
            self._validate_resolution_asset(
                context, resolution, relative, asset_sha
            )

    def _validate_resolution_asset(
        self,
        context: ShardContext,
        resolution: int,
        relative: Path,
        asset_sha: str,
    ) -> None:
        if resolution not in self.config.targets.resolutions:
            raise ValidationError(f"unexpected resolution: {resolution}")
        for view in self.config.targets.views:
            output = (
                context.output_root
                / relative
                / asset_sha
                / f"view{view:02d}.npz"
            )
            self._validate_sparse_output(output, resolution)

    def _validate_shape_resolution(
        self, context: ShardContext, resolution: int
    ) -> None:
        self._validate_resolution_family(
            context, resolution, self._shape_directory(resolution)
        )

    def _validate_pbr_resolution(
        self, context: ShardContext, resolution: int
    ) -> None:
        self._validate_resolution_family(
            context, resolution, self._pbr_directory(resolution)
        )

    def _validate_asset_outputs(
        self, context: ShardContext, asset_sha: str
    ) -> None:
        self._validate_render_output(context, asset_sha)
        for resolution in self.config.targets.resolutions:
            for relative in (
                self._shape_directory(resolution),
                self._pbr_directory(resolution),
            ):
                for view in self.config.targets.views:
                    output = (
                        context.output_root
                        / relative
                        / asset_sha
                        / f"view{view:02d}.npz"
                    )
                    self._validate_sparse_output(output, resolution)
        for view in self.config.targets.views:
            output = (
                context.output_root
                / self._ss_directory()
                / asset_sha
                / f"view{view:02d}.npz"
            )
            self._validate_sparse_output(
                output, self.config.targets.ss_resolution, ss=True
            )

    def _validate_family_output(
        self, context: ShardContext, asset_sha: str, family: str
    ) -> None:
        if family == "common":
            self._validate_render_output(context, asset_sha)
            return
        if family == f"SS-{self.config.targets.ss_resolution}":
            for view in self.config.targets.views:
                output = (
                    context.output_root
                    / self._ss_directory()
                    / asset_sha
                    / f"view{view:02d}.npz"
                )
                self._validate_sparse_output(
                    output,
                    self.config.targets.ss_resolution,
                    ss=True,
                )
            return
        prefix, resolution_value = family.split("-", 1)
        resolution = int(resolution_value)
        if resolution not in self.config.targets.resolutions:
            raise ValidationError(f"unexpected family: {family}")
        if prefix == "shape":
            relative = self._shape_directory(resolution)
        elif prefix == "PBR":
            relative = self._pbr_directory(resolution)
        else:
            raise ValidationError(f"unexpected family: {family}")
        self._validate_resolution_asset(
            context, resolution, relative, asset_sha
        )

    def _exclude_family_output(
        self,
        context: ShardContext,
        asset_sha: str,
        family: str,
        reason: str,
    ) -> None:
        if family.startswith("shape-"):
            command_name = f"encode_shape_{family.removeprefix('shape-')}"
        elif family.startswith("PBR-"):
            command_name = f"encode_pbr_{family.removeprefix('PBR-')}"
        elif family == f"SS-{self.config.targets.ss_resolution}":
            command_name = f"encode_ss_{self.config.targets.ss_resolution}"
        else:
            raise InfrastructureError(
                f"cannot exclude family output: {family}"
            )
        self._exclude_stage_failure(
            context,
            asset_sha,
            command_name,
            category="missing_output",
            reason=reason,
        )

    def _eligible_assets(self, context: ShardContext) -> tuple[str, ...]:
        assets = self._instances(context)
        checkpoint = getattr(self.runner, "active_checkpoint", None)
        if checkpoint is None:
            return assets
        return tuple(
            asset
            for asset in assets
            if asset not in checkpoint.quality_outcomes
        )

    def _candidate_assets(
        self, context: ShardContext, family: str | None = None
    ) -> tuple[str, ...]:
        assets = self._eligible_assets(context)
        if family is None:
            return assets
        return tuple(
            asset
            for asset in assets
            if self.runner.family_is_eligible(asset, family)
        )

    def _exclude_stage_failure(
        self,
        context: ShardContext,
        asset_sha: str,
        command_name: str,
        *,
        category: str,
        reason: str,
    ) -> None:
        families = set(command_families(self.config, command_name))
        for family in tuple(families):
            if family.startswith("shape-"):
                resolution = family.removeprefix("shape-")
                families.add(f"PBR-{resolution}")
                if int(resolution) == max(self.config.targets.resolutions):
                    families.add(f"SS-{self.config.targets.ss_resolution}")
        if not families:
            raise InfrastructureError(
                f"command has no family-scoped failure: {command_name}"
            )
        checkpoint = getattr(self.runner, "active_checkpoint", None)
        attempts = (
            checkpoint.attempts.get(command_name, 0)
            if checkpoint is not None
            else 0
        )
        self.runner.record_family_exclusion(
            asset_sha,
            tuple(sorted(families)),
            category=category,
            stage=command_name,
            reason=reason,
            attempts=attempts,
        )

    def _validate_stage_assets(
        self,
        context: ShardContext,
        validator: Callable[[str], None],
        *,
        command_name: str | None = None,
    ) -> None:
        checkpoint = getattr(self.runner, "active_checkpoint", None)
        families = (
            command_families(self.config, command_name)
            if command_name is not None
            else ()
        )
        assets = tuple(
            asset
            for asset in self._eligible_assets(context)
            if not families
            or any(
                self.runner.family_is_eligible(asset, family)
                for family in families
            )
        )
        for asset_sha in assets:
            try:
                validator(asset_sha)
            except OutputValidationError as error:
                if checkpoint is None:
                    raise
                if command_name is None:
                    self.runner.record_quality_outcome(
                        asset_sha, "failure"
                    )
                else:
                    self._exclude_stage_failure(
                        context,
                        asset_sha,
                        command_name,
                        category="missing_output",
                        reason=str(error),
                    )
            except ValidationError:
                if checkpoint is None:
                    raise
                self.runner.record_quality_outcome(
                    asset_sha, "schema_failure"
                )

    def _validate_asset_stats_stage(self, context: ShardContext) -> None:
        checkpoint = getattr(self.runner, "active_checkpoint", None)
        if checkpoint is None:
            self._validate_asset_stats(context)
            return
        assets = self._eligible_assets(context)
        try:
            records = self._asset_stats_records(context)
        except OutputValidationError:
            for asset_sha in assets:
                self.runner.record_quality_outcome(asset_sha, "failure")
            return
        except ValidationError:
            for asset_sha in assets:
                self.runner.record_quality_outcome(
                    asset_sha, "schema_failure"
                )
            return
        for asset_sha in assets:
            try:
                self._validate_asset_stats_record(
                    asset_sha, records.get(asset_sha)
                )
            except OutputValidationError:
                self.runner.record_quality_outcome(asset_sha, "failure")
            except ValidationError:
                self.runner.record_quality_outcome(
                    asset_sha, "schema_failure"
                )

    def _validate_all_outputs(self, context: ShardContext) -> None:
        for asset_sha in self._instances(context):
            self.asset_output_validator(context, asset_sha)

    def _quality_state(
        self, context: ShardContext
    ) -> tuple[tuple[str, ...], tuple[str, ...], int]:
        checkpoint = self.runner.active_checkpoint
        if checkpoint is None:
            checkpoint = self.runner.load_checkpoint(
                self._checkpoint_path(context), context.shard_id, context.gate
            )
        assets = self._instances(context)
        if tuple(checkpoint.quality_outcomes) != assets:
            raise ValidationError(
                "terminal quality outcomes do not match frozen SHA order"
            )
        completed = tuple(
            asset_sha
            for asset_sha in assets
            if checkpoint.quality_outcomes[asset_sha] == "completed"
        )
        quarantined = len(assets) - len(completed)
        return assets, completed, quarantined

    def _validate_terminal_outputs(self, context: ShardContext) -> None:
        _, completed, _ = self._quality_state(context)
        if not self._legacy_asset_output_validation:
            for asset_sha in completed:
                self.family_output_validator(
                    context, asset_sha, "common"
                )
                for family in PACK_FAMILIES:
                    if family == "common" or not self.runner.family_is_eligible(
                        asset_sha, family
                    ):
                        continue
                    self.family_output_validator(
                        context, asset_sha, family
                    )
            return
        for asset_sha in completed:
            self.asset_output_validator(context, asset_sha)

    def validate_outputs(self, context: ShardContext) -> None:
        checkpoint = self.runner.active_checkpoint
        if checkpoint is None:
            raise InfrastructureError(
                "output validation has no active durable checkpoint"
            )
        if self._legacy_asset_output_validation:
            for asset_sha in self._instances(context):
                if asset_sha in checkpoint.quality_outcomes:
                    continue
                try:
                    self.asset_output_validator(context, asset_sha)
                except OutputValidationError:
                    outcome = "failure"
                except ValidationError:
                    outcome = "schema_failure"
                else:
                    outcome = "completed"
                self.runner.record_quality_outcome(asset_sha, outcome)
            return

        for asset_sha in self._instances(context):
            if asset_sha in checkpoint.quality_outcomes:
                continue
            try:
                self.family_output_validator(
                    context, asset_sha, "common"
                )
            except OutputValidationError as error:
                self.runner.record_asset_outcome(
                    asset_sha,
                    "failure",
                    category="missing_render_output",
                    stage="validate_outputs",
                    reason=str(error),
                )
                continue
            except ValidationError as error:
                self.runner.record_asset_outcome(
                    asset_sha,
                    "schema_failure",
                    category="schema_failure",
                    stage="validate_outputs",
                    reason=str(error),
                )
                continue

            schema_failure = None
            for family in PACK_FAMILIES:
                if family == "common" or not self.runner.family_is_eligible(
                    asset_sha, family
                ):
                    continue
                try:
                    self.family_output_validator(
                        context, asset_sha, family
                    )
                except OutputValidationError as error:
                    self._exclude_family_output(
                        context, asset_sha, family, str(error)
                    )
                except ValidationError as error:
                    schema_failure = error
                    break
            if schema_failure is not None:
                self.runner.record_asset_outcome(
                    asset_sha,
                    "schema_failure",
                    category="schema_failure",
                    stage="validate_outputs",
                    reason=str(schema_failure),
                )
                continue
            included = tuple(
                family
                for family in PACK_FAMILIES
                if family != "common"
                and self.runner.family_is_eligible(asset_sha, family)
            )
            if included:
                self.runner.record_asset_outcome(
                    asset_sha, "completed"
                )
            else:
                self.runner.record_asset_outcome(
                    asset_sha,
                    "failure",
                    category="no_eligible_training_family",
                    stage="validate_outputs",
                    reason="asset has no validated training family",
                )

    def cleanup_voxels(
        self, context: ShardContext, resolution: int
    ) -> None:
        checkpoint = getattr(self.runner, "active_checkpoint", None)
        if checkpoint is None:
            self.shape_resolution_validator(context, resolution)
            self.pbr_resolution_validator(context, resolution)
        else:
            self._validate_stage_assets(
                context,
                lambda asset_sha: self._validate_resolution_asset(
                    context,
                    resolution,
                    self._shape_directory(resolution),
                    asset_sha,
                ),
                command_name=f"encode_shape_{resolution}",
            )
            self._validate_stage_assets(
                context,
                lambda asset_sha: self._validate_resolution_asset(
                    context,
                    resolution,
                    self._pbr_directory(resolution),
                    asset_sha,
                ),
                command_name=f"encode_pbr_{resolution}",
            )
        for path in (
            context.work_root / f"dual_grid_view_{resolution}",
            context.work_root / f"pbr_voxels_view_fix_{resolution}",
        ):
            try:
                _remove_tree_nofollow(path)
            except OSError as error:
                if error.errno in PATH_VALIDATION_ERRNOS:
                    raise ValidationError(
                        f"unsafe voxel cleanup path: {path}: {error}"
                    ) from error
                raise

    def _family_included_assets(
        self,
        completed: Sequence[str],
        *,
        context: ShardContext | None = None,
    ) -> Mapping[str, tuple[str, ...]]:
        ordered = tuple(_validated_asset_sha(asset) for asset in completed)
        if (
            context is None
            or getattr(self.runner, "active_context", None) == context
            and getattr(self.runner, "_active_quality_ledger", None)
            is not None
        ):
            family_is_eligible = self.runner.family_is_eligible
        else:
            try:
                ledger = _load_quality_ledger(
                    self._quality_ledger_path(context), context
                )
            except CheckpointError as error:
                raise InfrastructureError(
                    f"cannot restore family eligibility: {error}"
                ) from error
            exclusions = ledger["family_exclusions"]
            dependencies = family_dependencies(self.config)

            def family_is_eligible(asset: str, family: str) -> bool:
                excluded = set(exclusions.get(asset, ()))

                def eligible(
                    candidate: str, visiting: frozenset[str]
                ) -> bool:
                    if candidate in excluded:
                        return False
                    if candidate in visiting:
                        raise InfrastructureError(
                            f"cyclic family dependency: {candidate}"
                        )
                    return all(
                        eligible(dependency, visiting | {candidate})
                        for dependency in dependencies[candidate]
                    )

                return eligible(family, frozenset())

        included = {
            family: tuple(
                asset
                for asset in ordered
                if family_is_eligible(asset, family)
            )
            for family in PACK_FAMILIES
            if family != "common"
        }
        common_assets = set().union(
            *(set(assets) for assets in included.values())
        )
        included["common"] = tuple(
            asset for asset in ordered if asset in common_assets
        )
        dependencies = family_dependencies(self.config)
        for family, required in dependencies.items():
            for dependency in required:
                if not set(included[family]) <= set(included[dependency]):
                    raise ValidationError(
                        f"{family} membership is not a {dependency} subset"
                    )
        return {
            family: included[family] for family in PACK_FAMILIES
        }

    def _pack_members_for_assets(
        self,
        context: ShardContext,
        included_by_family: Mapping[str, Sequence[str]],
    ) -> Mapping[str, Sequence[Path]]:
        if set(included_by_family) != set(PACK_FAMILIES):
            raise ValidationError(
                "included asset mapping must contain exactly eight families"
            )
        members: dict[str, list[Path]] = {
            family: [] for family in PACK_FAMILIES
        }
        for asset_sha in included_by_family["common"]:
            render_root = Path("renders_cond", asset_sha)
            members["common"].extend(
                [
                    render_root / f"{view:03d}.png"
                    for view in range(self.config.render.num_views)
                ]
                + [render_root / "transforms.json"]
            )
        for family, relative in (
            (f"SS-{self.config.targets.ss_resolution}", self._ss_directory()),
            *(
                (f"shape-{resolution}", self._shape_directory(resolution))
                for resolution in self.config.targets.resolutions
            ),
            *(
                (f"PBR-{resolution}", self._pbr_directory(resolution))
                for resolution in self.config.targets.resolutions
            ),
        ):
            for asset_sha in included_by_family[family]:
                for view in self.config.targets.views:
                    members[family].extend(
                        (
                            relative / asset_sha / f"view{view:02d}.npz",
                            relative
                            / asset_sha
                            / f"view{view:02d}_scale.json",
                        )
                    )
        return members

    def _pack_members(
        self, context: ShardContext
    ) -> Mapping[str, Sequence[Path]]:
        _, completed, _ = self._quality_state(context)
        included = self._family_included_assets(completed, context=context)
        return self._pack_members_for_assets(context, included)

    @staticmethod
    def _path_size(path: Path) -> int:
        try:
            return _regular_file_size_nofollow(path, missing_ok=True)
        except OSError as error:
            if error.errno in PATH_VALIDATION_ERRNOS:
                raise ValidationError(
                    f"unsafe accounted path: {path}"
                ) from error
            raise

    def _record_delta(self, path: Path, delta: int) -> None:
        if delta == 0:
            return
        try:
            self.project_accounting.record_registry_delta(path, delta)
        except (
            InfrastructureError,
            IntegrationProviderRequired,
            OSError,
            ResourceAccountingError,
        ) as error:
            try:
                self.project_accounting.reconcile_at_shard_boundary()
            except (
                InfrastructureError,
                IntegrationProviderRequired,
                OSError,
                ResourceAccountingError,
            ) as reconcile_error:
                raise InfrastructureError(
                    f"project accounting delta failed for {path}: {error}; "
                    f"reconciliation failed: {reconcile_error}"
                ) from error
            raise InfrastructureError(
                f"project accounting delta failed for {path}: {error}"
            ) from error

    def _published_paths(self, context: ShardContext) -> tuple[Path, ...]:
        prepared = self.config.paths.data2_root / "prepared"
        prefix = (
            Path()
            if context.gate == "production"
            else Path("qualification", context.gate)
        )
        paths = []
        for family in PACK_FAMILIES:
            if family == "common":
                root = Path("common")
            elif family.startswith("SS-"):
                root = Path("ss", family.removeprefix("SS-"))
            elif family.startswith("shape-"):
                root = Path("shape", family.removeprefix("shape-"))
            else:
                root = Path("pbr", family.removeprefix("PBR-"))
            pack = (
                prepared
                / prefix
                / root
                / context.source
                / context.shard_id
                / f"{context.batch_id}.tar"
            )
            paths.extend((pack, pack.with_suffix(".tar.manifest.json")))
        paths.append(
            prepared
            / prefix
            / "index"
            / context.source
            / f"{context.shard_id}.json"
        )
        return tuple(paths)

    def build_packs(self, context: ShardContext) -> None:
        self.output_validator(context)
        shas, completed, _quarantined = self._quality_state(context)
        included = self._family_included_assets(completed, context=context)
        members = self.pack_member_builder(context)
        if set(members) != set(PACK_FAMILIES):
            raise ValidationError(
                "pack member mapping must contain exactly eight families"
            )
        expected_members = self._pack_members_for_assets(context, included)
        if any(
            tuple(sorted(Path(item).as_posix() for item in members[family]))
            != tuple(
                sorted(
                    Path(item).as_posix()
                    for item in expected_members[family]
                )
            )
            for family in PACK_FAMILIES
        ):
            raise ValidationError("pack member mapping does not match admission")
        accounted_paths = self._published_paths(context)
        before = {path: self._path_size(path) for path in accounted_paths}
        manifests = self.pack_publisher(
            self.config.paths.data2_root,
            context.output_root,
            members,
            context.shard_id,
            source=context.source,
            batch_id=context.batch_id,
            config_hash=self.config.config_hash(),
            tool_commit=self._resolved_tool_commit(),
            asset_sha256s=shas,
            included_asset_sha256s_by_family=included,
            gate=context.gate,
        )
        if (
            len(manifests) != len(PACK_FAMILIES)
            or any(not getattr(item, "validated_at", "") for item in manifests)
        ):
            raise ValidationError("pack publisher returned unvalidated packs")
        self.published_batch_verifier(context)
        for path in accounted_paths:
            self._record_delta(path, self._path_size(path) - before[path])

    def _verify_published_batch(self, context: ShardContext) -> None:
        shas, completed, _quarantined = self._quality_state(context)
        included = self._family_included_assets(completed, context=context)
        expected_members = self._pack_members_for_assets(context, included)
        tool_commit = self._resolved_tool_commit()
        prepared = self.config.paths.data2_root / "prepared"
        prefix = (
            Path()
            if context.gate == "production"
            else Path("qualification", context.gate)
        )
        index_path = (
            prepared
            / prefix
            / "index"
            / context.source
            / f"{context.shard_id}.json"
        )
        try:
            index = json.loads(_read_regular_bytes_nofollow(index_path))
            entries = index["batches"][context.batch_id]
        except OSError as error:
            if error.errno not in {errno.ENOENT, errno.ENOTDIR, errno.ELOOP}:
                raise
            raise ValidationError(
                f"invalid published shard index: {index_path}: {error}"
            ) from error
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValidationError(
                f"invalid published shard index: {index_path}: {error}"
            ) from error
        if (
            index.get("source") != context.source
            or index.get("shard_id") != context.shard_id
            or index.get("gate") != context.gate
            or set(entries) != set(PACK_FAMILIES)
        ):
            raise ValidationError(f"invalid published shard index: {index_path}")
        for family, entry in entries.items():
            try:
                if family == "common":
                    family_root = Path("common")
                elif family.startswith("SS-"):
                    family_root = Path("ss", family.removeprefix("SS-"))
                elif family.startswith("shape-"):
                    family_root = Path(
                        "shape", family.removeprefix("shape-")
                    )
                elif family.startswith("PBR-"):
                    family_root = Path("pbr", family.removeprefix("PBR-"))
                else:
                    raise ValidationError(f"unknown pack family: {family}")
                expected_pack = (
                    prefix
                    / family_root
                    / context.source
                    / context.shard_id
                    / f"{context.batch_id}.tar"
                )
                expected_manifest = expected_pack.with_suffix(
                    ".tar.manifest.json"
                )
                if (
                    entry.get("pack") != expected_pack.as_posix()
                    or entry.get("manifest")
                    != expected_manifest.as_posix()
                ):
                    raise ValidationError(
                        f"non-canonical pack path: {family}"
                    )
                pack_relative = _safe_raw_relative(entry["pack"])
                manifest_relative = _safe_raw_relative(entry["manifest"])
                pack_path = prepared / pack_relative
                manifest_path = prepared / manifest_relative
                if pack_path.is_symlink() or manifest_path.is_symlink():
                    raise ValidationError(
                        f"symlinked published pack: {family}"
                    )
                try:
                    verify_pack(pack_path, manifest_path)
                except ValidationError as error:
                    raise ValidationError(
                        f"published pack identity mismatch: {family}: {error}"
                    ) from error
                manifest_payload = _read_regular_bytes_nofollow(manifest_path)
                try:
                    manifest = json.loads(manifest_payload)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValidationError(
                        f"invalid published pack manifest: {family}"
                    ) from error
                if (
                    manifest["shard_id"] != context.shard_id
                    or manifest["gate"] != context.gate
                    or manifest["batch_id"] != context.batch_id
                    or manifest["family"] != family
                    or manifest["config_hash"] != self.config.config_hash()
                    or manifest["tool_commit"] != tool_commit
                    or tuple(manifest["asset_sha256s"]) != shas
                    or manifest.get("schema_version") != 2
                    or tuple(manifest["included_asset_sha256s"])
                    != included[family]
                    or manifest["completed_count"]
                    != len(included[family])
                    or manifest["quarantined_count"]
                    != len(shas) - len(included[family])
                    or {
                        item["path"] for item in manifest["members"]
                    }
                    != {
                        Path(item).as_posix()
                        for item in expected_members[family]
                    }
                    or not manifest["validated_at"]
                    or entry["pack_sha256"] != manifest["pack_sha256"]
                    or entry["manifest_sha256"]
                    != sha256(manifest_payload).hexdigest()
                ):
                    raise ValidationError(
                        f"published pack frozen SHA or member identity mismatch: {family}"
                    )
            except (KeyError, TypeError) as error:
                raise ValidationError(
                    f"invalid published pack entry: {family}"
                ) from error

    def _raw_archive_paths(self, context: ShardContext) -> tuple[Path, Path]:
        root = self.config.paths.data3_root / "archive"
        if context.gate == "production":
            root = root / "raw"
        else:
            root = root / "qualification" / context.gate / "raw"
        archive = root / context.source / context.shard_id / f"{context.batch_id}.tar"
        return archive, archive.with_suffix(".tar.manifest.json")

    def _verify_raw_archive(self, context: ShardContext) -> None:
        shas, completed, quarantined = self._quality_state(context)
        archive, manifest_path = self._raw_archive_paths(context)
        if archive.is_symlink() or manifest_path.is_symlink():
            raise ValidationError(
                f"refusing symlinked raw archive output: {archive}"
        )
        try:
            verify_pack(archive, manifest_path)
        except ValidationError as error:
            raise ValidationError(
                f"raw archive identity mismatch: {manifest_path}: {error}"
            ) from error
        try:
            manifest = json.loads(
                _read_regular_bytes_nofollow(manifest_path)
            )
        except OSError as error:
            if error.errno not in {errno.ENOENT, errno.ENOTDIR, errno.ELOOP}:
                raise
            raise ValidationError(
                f"invalid raw archive manifest: {manifest_path}: {error}"
            ) from error
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
            raise ValidationError(
                f"invalid raw archive manifest: {manifest_path}: {error}"
            ) from error
        expected_records = self._read_raw_records(
            context.source_root / "raw/metadata.csv", completed
        )
        expected_members = self._raw_file_map(expected_records)
        try:
            actual_members = {
                item["path"]: item["sha256"]
                for item in manifest.get("members", ())
            }
        except (KeyError, TypeError) as error:
            raise ValidationError(
                f"invalid raw archive member mapping: {manifest_path}"
            ) from error
        if (
            manifest.get("shard_id") != context.shard_id
            or manifest.get("gate") != context.gate
            or manifest.get("batch_id") != context.batch_id
            or manifest.get("family") != "raw"
            or manifest.get("config_hash") != self.config.config_hash()
            or manifest.get("tool_commit") != self._resolved_tool_commit()
            or manifest.get("completed_count") != len(completed)
            or manifest.get("quarantined_count") != quarantined
            or manifest.get("schema_version") != 2
            or tuple(manifest.get("included_asset_sha256s", ()))
            != completed
            or not manifest.get("validated_at")
            or tuple(manifest.get("asset_sha256s", ()))
            != shas
            or len(manifest.get("members", ())) != len(expected_members)
            or len(actual_members) != len(manifest.get("members", ()))
        ):
            raise ValidationError(
                f"raw archive identity mismatch: {manifest_path}"
            )
        if actual_members != expected_members:
            raise ValidationError(
                f"raw archive member mapping mismatch: {manifest_path}"
            )

    def archive_raw(self, context: ShardContext) -> None:
        self.published_batch_verifier(context)
        _shas, completed, _quarantined = self._quality_state(context)
        records = self._read_raw_records(
            context.download_root / "raw/metadata.csv", completed
        )
        file_map = self._raw_file_map(records)
        members = [
            _safe_raw_relative(relative) for relative in sorted(file_map)
        ]
        local_archive = (
            context.work_root / "raw_archive" / f"{context.batch_id}.tar"
        )
        shas, completed, quarantined = self._quality_state(context)
        manifest = build_pack(
            context.download_root,
            members,
            local_archive,
            context.shard_id,
            batch_id=context.batch_id,
            family="raw",
            config_hash=self.config.config_hash(),
            tool_commit=self._resolved_tool_commit(),
            asset_sha256s=shas,
            included_asset_sha256s=completed,
            completed_count=len(completed),
            quarantined_count=quarantined,
            gate=context.gate,
        )
        local_manifest = local_archive.with_suffix(".tar.manifest.json")
        verify_pack(local_archive, local_manifest)
        manifest = replace(
            manifest,
            validated_at=datetime.now(timezone.utc).isoformat(),
        )
        _atomic_write_bytes_nofollow(
            local_manifest,
            json.dumps(
                asdict(manifest), sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
        )
        verify_pack(local_archive, local_manifest)
        if (
            len(manifest.members) != len(file_map)
            or sum(item.size for item in manifest.members)
            != sum((context.download_root / member).stat().st_size for member in members)
            or {item.path: item.sha256 for item in manifest.members}
            != file_map
        ):
            raise ValidationError("raw archive count, byte, or member hash mismatch")

        archive, archive_manifest = self._raw_archive_paths(context)
        before_archive = self._path_size(archive)
        before_manifest = self._path_size(archive_manifest)
        if archive.is_symlink() or archive_manifest.is_symlink():
            raise ValidationError(
                f"refusing symlinked raw archive output: {archive}"
            )
        if archive.exists() and archive_manifest.exists():
            verify_pack(archive, archive_manifest)
            existing = json.loads(
                _read_regular_bytes_nofollow(archive_manifest)
            )
            if (
                existing.get("pack_sha256") != manifest.pack_sha256
                or existing.get("members")
                != [asdict(item) for item in manifest.members]
                or tuple(existing.get("asset_sha256s", ())) != shas
            ):
                raise ValidationError(
                    f"different valid raw archive already exists: {archive}"
                )
        else:
            atomic_copy(local_archive, archive)
            atomic_copy(local_manifest, archive_manifest)
        self._verify_raw_archive(context)
        self._record_delta(
            archive, self._path_size(archive) - before_archive
        )
        self._record_delta(
            archive_manifest,
            self._path_size(archive_manifest) - before_manifest,
        )

        for record in records:
            primary_value = self._source_raw_relative(
                _safe_raw_relative(record["local_path"])
            ).as_posix()
            pending = self.reference_counter.pending_references(
                context.source,
                primary_value,
                excluding_shard_id=context.shard_id,
                excluding_batch_id=context.batch_id,
                gate=context.gate,
            )
            if (
                not isinstance(pending, int)
                or isinstance(pending, bool)
                or pending < 0
            ):
                raise IntegrationProviderRequired(
                    "raw reference counter must return a non-negative integer"
            )
            if pending:
                continue
            source_paths = {
                self._source_raw_relative(_safe_raw_relative(relative)).as_posix()
                for relative in (
                    record["local_path"],
                    *record["companion_files"],
                )
            }
            for relative_value in sorted(source_paths):
                relative = _safe_raw_relative(relative_value)
                removed_bytes = _unlink_regular_beneath(
                    context.source_root, relative
                )
                if removed_bytes:
                    self._record_delta(
                        context.source_root / relative, -removed_bytes
                    )

    def cleanup_local(self, context: ShardContext) -> None:
        self.published_batch_verifier(context)
        self.raw_archive_verifier(context)
        for root in (
            context.output_root,
            context.work_root,
            context.download_root,
        ):
            try:
                _remove_tree_nofollow(root)
            except OSError as error:
                if error.errno in PATH_VALIDATION_ERRNOS:
                    raise ValidationError(
                        f"unsafe local cleanup path: {root}: {error}"
                    ) from error
                raise

    def _archive_is_valid(self, context: ShardContext) -> bool:
        try:
            self.raw_archive_verifier(context)
            return True
        except ValidationError:
            return False

    def _published_is_valid(self, context: ShardContext) -> bool:
        try:
            self.published_batch_verifier(context)
            return True
        except ValidationError:
            return False

    def _validate_download_output(self, context: ShardContext) -> bool:
        selected = self._eligible_assets(context)
        by_sha = self._read_raw_record_map(
            context.source_root / "raw/metadata.csv", selected
        )
        missing = tuple(asset for asset in selected if asset not in by_sha)
        if not missing:
            return True
        if not by_sha:
            raise OutputValidationError(
                "download produced no verified selected assets"
            )
        checkpoint = getattr(self.runner, "active_checkpoint", None)
        attempts = 0
        if checkpoint is not None:
            attempts = checkpoint.attempts.get("download", 0)
        if attempts < MAX_COMMAND_ATTEMPTS:
            raise OutputValidationError(
                f"download missing selected assets: {list(missing)}"
            )
        for asset_sha in missing:
            recorder = getattr(self.runner, "record_asset_outcome", None)
            if (
                callable(recorder)
                and getattr(self.runner, "active_context", None) is not None
                and getattr(self.runner, "active_checkpoint", None) is not None
                and getattr(self.runner, "active_checkpoint_path", None) is not None
            ):
                recorder(
                    asset_sha,
                    "failure",
                    category="provider_asset_unavailable",
                    stage="download",
                    reason="source metadata has no verified downloaded record after maximum attempts",
                    attempts=attempts,
                )
            else:
                self.runner.record_quality_outcome(asset_sha, "failure")
        return True

    def _validate_command(self, name: str) -> bool:
        context = self._active_context()
        if name == "archive_raw":
            return self._archive_is_valid(context)
        if name == "cleanup_local":
            return self._published_is_valid(context) and self._archive_is_valid(
                context
            ) and not any(
                path.exists()
                for path in (
                    context.download_root,
                    context.work_root,
                    context.output_root,
                )
            )
        if self._published_is_valid(context):
            return True
        try:
            if name == "download":
                return self._validate_download_output(context)
            if name == "stage_raw":
                self._validate_staged_raw(context)
                return True
            if name == "dump_mesh":
                self._validate_stage_assets(
                    context,
                    lambda asset: self._validate_dump_output(
                        context, "mesh_dumps", asset
                    ),
                )
                return True
            if name == "dump_pbr":
                self._validate_pbr_dump_stage(context)
                return True
            if name == "asset_stats":
                self._validate_asset_stats_stage(context)
                return True
            if name == "render_cond":
                self._validate_stage_assets(
                    context,
                    lambda asset: self._validate_render_output(
                        context, asset
                    ),
                )
                return True
            for resolution in self.config.targets.resolutions:
                if name in {
                    f"encode_shape_{resolution}",
                    f"encode_pbr_{resolution}",
                    f"cleanup_voxels_{resolution}",
                }:
                    if name == f"encode_shape_{resolution}":
                        if getattr(self.runner, "active_checkpoint", None) is None:
                            self.shape_resolution_validator(context, resolution)
                        else:
                            self._validate_stage_assets(
                                context,
                                lambda asset: self._validate_resolution_asset(
                                    context,
                                    resolution,
                                    self._shape_directory(resolution),
                                    asset,
                                ),
                                command_name=name,
                            )
                    elif name == f"encode_pbr_{resolution}":
                        if getattr(self.runner, "active_checkpoint", None) is None:
                            self.pbr_resolution_validator(context, resolution)
                        else:
                            self._validate_stage_assets(
                                context,
                                lambda asset: self._validate_resolution_asset(
                                    context,
                                    resolution,
                                    self._pbr_directory(resolution),
                                    asset,
                                ),
                                command_name=name,
                            )
                    else:
                        if getattr(self.runner, "active_checkpoint", None) is None:
                            self.shape_resolution_validator(context, resolution)
                            self.pbr_resolution_validator(context, resolution)
                    if name.startswith("cleanup_"):
                        return not any(
                            path.exists()
                            for path in (
                                context.work_root
                                / f"dual_grid_view_{resolution}",
                                context.work_root
                                / f"pbr_voxels_view_fix_{resolution}",
                            )
                        )
                    return True
                if name in {
                    f"dual_grid_{resolution}",
                    f"voxelize_pbr_{resolution}",
                }:
                    directory = (
                        f"dual_grid_view_{resolution}"
                        if name.startswith("dual_grid")
                        else f"pbr_voxels_view_fix_{resolution}"
                    )
                    self._validate_stage_assets(
                        context,
                        lambda asset: [
                            self._validate_voxel_output(
                                context, directory, asset, view
                            )
                            for view in self.config.targets.views
                        ],
                        command_name=name,
                    )
                    return True
            if name == f"encode_ss_{self.config.targets.ss_resolution}":
                def validate_ss_asset(asset):
                    for view in self.config.targets.views:
                        output = (
                            context.output_root
                            / self._ss_directory()
                            / asset
                            / f"view{view:02d}.npz"
                        )
                        self._validate_sparse_output(
                            output,
                            self.config.targets.ss_resolution,
                            ss=True,
                        )

                self._validate_stage_assets(
                    context, validate_ss_asset, command_name=name
                )
                return True
            if name == "validate_outputs":
                self._validate_terminal_outputs(context)
                return True
            if name == "build_packs":
                return self._published_is_valid(context)
        except ValidationError:
            return False
        raise InfrastructureError(f"missing production validator: {name}")

    def _audit_batch(self, context: ShardContext) -> None:
        self.published_batch_verifier(context)
        self.raw_archive_verifier(context)

    def _reconcile_accounting(
        self, context: ShardContext, boundary: str
    ) -> None:
        try:
            self.project_accounting.reconcile_at_shard_boundary()
        except (
            InfrastructureError,
            IntegrationProviderRequired,
            OSError,
            ResourceAccountingError,
        ) as error:
            infrastructure = InfrastructureError(
                f"project accounting reconciliation failed at {boundary}: {error}"
            )
            if all(
                hasattr(self.runner, attribute)
                for attribute in ("load_checkpoint", "stop")
            ):
                try:
                    checkpoint = self.runner.load_checkpoint(
                        self._checkpoint_path(context),
                        context.shard_id,
                        context.gate,
                    )
                except (CheckpointError, InfrastructureError, OSError):
                    checkpoint = PipelineCheckpoint(
                        context.shard_id, gate=context.gate
                    )
                self.runner.stop(
                    context,
                    f"accounting_{boundary}",
                    str(infrastructure),
                    checkpoint,
                    category=EscalationCategory.INFRASTRUCTURE,
                    exit_code=2,
                )
            raise infrastructure from error

    def _frozen_for_execution(
        self, gate: str, source: str, shard: str
    ) -> tuple[tuple[str, ...], ...]:
        shas = self._registry_shas(source, shard)
        batches = self._read_frozen_batches(
            gate, source, shard, shas
        )
        if batches is None:
            raise InfrastructureError(
                f"no frozen {gate} batch manifest for resume: {source}/{shard}"
            )
        return batches

    def _verify_logical_index(
        self,
        source: str,
        shard: str,
        batches: tuple[tuple[str, ...], ...],
        *,
        gate: str = "production",
    ) -> None:
        if self.published_batch_verifier == self._verify_published_batch:
            prepared = self.config.paths.data2_root / "prepared"
            if gate != "production":
                prepared = prepared / "qualification" / gate
            index_path = prepared / "index" / source / f"{shard}.json"
            try:
                index = json.loads(
                    _read_regular_bytes_nofollow(index_path)
                )
                indexed_batches = set(index["batches"])
            except OSError as error:
                if error.errno not in {
                    errno.ENOENT,
                    errno.ENOTDIR,
                    errno.ELOOP,
                }:
                    raise
                raise ValidationError(
                    f"invalid logical shard index: {index_path}: {error}"
                ) from error
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
            ) as error:
                raise ValidationError(
                    f"invalid logical shard index: {index_path}: {error}"
                ) from error
            expected_batches = {
                f"batch{index:03d}" for index in range(len(batches))
            }
            if indexed_batches != expected_batches:
                raise ValidationError(
                    f"logical shard index batch set mismatch: {index_path}"
                )
        for index in range(len(batches)):
            context = ShardContext.from_config(
                self.config,
                source,
                shard,
                f"batch{index:03d}",
                gate=gate,
            )
            self.published_batch_verifier(context)

    def build_registry(self):
        if self.registry_builder is None:
            raise IntegrationProviderRequired(
                "registry builder provider is required"
            )
        return self.registry_builder()

    def run(
        self,
        gate: str,
        source: str | None,
        shard: str | None,
        count: int | None = None,
    ) -> None:
        if source is None or shard is None:
            raise ValueError("run requires source and shard")
        if gate not in {"smoke", "pilot", "production"}:
            raise ValueError(f"unknown gate: {gate}")
        if gate == "production" and count is not None:
            raise ValueError("production requires the full canonical shard")
        canonical_shas = self._registry_shas(source, shard)
        shas = canonical_shas[:count] if count is not None else canonical_shas
        batches = self._read_frozen_batches(
            gate,
            source,
            shard,
            canonical_shas,
            expected_scope=shas,
        )
        if batches is None:
            batches = self._planned_batches(
                gate, source, shard, count, freeze=True
            )
        for index in range(len(batches)):
            context = ShardContext.from_config(
                self.config,
                source,
                shard,
                f"batch{index:03d}",
                gate=gate,
            )
            self._execute_batch(context, batches[index], resume=False)
            self.batch_auditor(context)
            self._reconcile_accounting(context, "batch")
        self._verify_logical_index(source, shard, batches, gate=gate)
        self._reconcile_accounting(context, "shard")

    def run_batch(
        self,
        gate: str,
        source: str,
        shard: str,
        batch_id: str,
    ) -> ShardContext:
        if gate != "production":
            raise ValueError("work queue batch execution requires production gate")
        match = re.fullmatch(r"batch([0-9]{3})", batch_id)
        if match is None:
            raise ValueError(f"invalid production batch id: {batch_id}")
        batches = self._frozen_for_execution(gate, source, shard)
        index = int(match.group(1))
        if index >= len(batches):
            raise ValueError(
                f"unknown frozen batch: {source}/{shard}/{batch_id}"
            )
        context = ShardContext.from_config(
            self.config, source, shard, batch_id, gate=gate
        )
        self._execute_batch(context, batches[index], resume=False)
        self.batch_auditor(context)
        self._reconcile_accounting(context, "batch")
        return context

    def resume(
        self, gate: str, source: str | None, shard: str | None
    ) -> None:
        if source is None or shard is None:
            raise ValueError("resume requires source and shard")
        batches = self._frozen_for_execution(gate, source, shard)
        for index in range(len(batches)):
            context = ShardContext.from_config(
                self.config,
                source,
                shard,
                f"batch{index:03d}",
                gate=gate,
            )
            self._execute_batch(context, batches[index], resume=True)
            self.batch_auditor(context)
            self._reconcile_accounting(context, "batch")
        self._verify_logical_index(source, shard, batches, gate=gate)
        self._reconcile_accounting(context, "shard")

    def audit(
        self, gate: str, source: str | None, shard: str | None
    ) -> None:
        if source is None or shard is None:
            raise ValueError("audit requires source and shard")
        batches = self._frozen_for_execution(gate, source, shard)
        for index in range(len(batches)):
            self.batch_auditor(
                context := ShardContext.from_config(
                    self.config,
                    source,
                    shard,
                    f"batch{index:03d}",
                    gate=gate,
                )
            )
            self._reconcile_accounting(context, "batch")
        self._verify_logical_index(source, shard, batches, gate=gate)
        self._reconcile_accounting(context, "shard")

    def _benchmark_scope(
        self, source: str, shard: str, count: int
    ) -> tuple[str, ...]:
        if type(count) is not int or count <= 0:
            raise ValueError("parallelism benchmark count must be positive")
        if count != self.config.parallelism.chunk_assets:
            raise ValueError(
                "parallelism benchmark count must equal configured chunk assets"
            )
        assets = self._registry_shas(source, shard, count)
        if len(assets) != count:
            raise InfrastructureError(
                f"benchmark shard has fewer than {count} assets: {source}/{shard}"
            )
        return assets

    def _benchmark_context(
        self,
        source: str,
        shard: str,
        count: int,
        instances: Path,
    ) -> ShardContext:
        local = (
            self.config.paths.local_root
            / "preprocess/benchmark"
            / source
            / shard
            / f"count{count:03d}"
        )
        return ShardContext(
            source=source,
            shard_id=f"{shard}-benchmark{count:03d}",
            instances=instances,
            metadata_root=(
                self.config.paths.data2_root / "control/metadata" / source
            ),
            source_root=self.config.paths.data2_root / "raw" / source,
            download_root=local / "source",
            work_root=local / "work",
            output_root=local / "output",
            batch_id=f"benchmark{count:03d}",
            gate="pilot",
        )

    @staticmethod
    def _benchmark_telemetry(
        path: Path,
        shard_id: str,
        started_at: datetime,
        finished_at: datetime,
    ) -> tuple[dict, ...]:
        try:
            descriptor = os.open(
                path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
            )
        except FileNotFoundError:
            return ()
        except OSError as error:
            raise InfrastructureError(
                f"cannot read benchmark telemetry: {path}: {error}"
            ) from error
        records = []
        try:
            with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                descriptor = -1
                for line_number, line in enumerate(stream, start=1):
                    try:
                        record = json.loads(line)
                        timestamp = datetime.fromisoformat(
                            record["timestamp"]
                        )
                    except (
                        KeyError,
                        TypeError,
                        ValueError,
                        json.JSONDecodeError,
                    ) as error:
                        raise InfrastructureError(
                            f"invalid benchmark telemetry line {line_number}: {error}"
                        ) from error
                    if (
                        record.get("shard_id") == shard_id
                        and timestamp.tzinfo is not None
                        and timestamp.utcoffset() is not None
                        and started_at <= timestamp <= finished_at
                    ):
                        records.append(record)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        return tuple(records)

    @staticmethod
    def _benchmark_resource_summary(
        telemetry: Sequence[Mapping[str, object]], expected_gpus: int
    ) -> dict[str, object]:
        cpu = []
        gpu_memory = []
        gpu_utilization = []
        gpu_temperature = []
        gpu_indices = set()
        pauses = 0
        for record in telemetry:
            try:
                cpu.append(float(record["cpu_percent"]))
                if record.get("action") == "pause":
                    pauses += 1
                metrics = record["gpu_metrics"]
                if not isinstance(metrics, list):
                    raise TypeError("GPU metrics are not a list")
                sample_memory = []
                sample_utilization = []
                sample_temperature = []
                for metric in metrics:
                    if not isinstance(metric, Mapping):
                        raise TypeError("GPU metric is not an object")
                    used = float(metric["memory_used_mib"])
                    total = float(metric["memory_total_mib"])
                    if total <= 0 or used < 0:
                        raise ValueError("invalid GPU memory")
                    sample_memory.append(100.0 * used / total)
                    sample_utilization.append(
                        float(metric["utilization_percent"])
                    )
                    sample_temperature.append(
                        float(metric["temperature_celsius"])
                    )
                    gpu_indices.add(int(metric["index"]))
                if sample_memory:
                    gpu_memory.append(max(sample_memory))
                    gpu_utilization.append(
                        sum(sample_utilization) / len(sample_utilization)
                    )
                    gpu_temperature.append(max(sample_temperature))
            except (KeyError, TypeError, ValueError) as error:
                raise InfrastructureError(
                    f"invalid parallelism telemetry record: {error}"
                ) from error
        finite = (
            *cpu,
            *gpu_memory,
            *gpu_utilization,
            *gpu_temperature,
        )
        if any(not math.isfinite(value) or value < 0 for value in finite):
            raise InfrastructureError("non-finite parallelism telemetry")
        return {
            "samples": len(telemetry),
            "gpu_count_observed": len(gpu_indices),
            "telemetry_valid": (
                bool(telemetry)
                and len(gpu_indices) == expected_gpus
                and bool(gpu_memory)
            ),
            "gpu_memory_peak_percent": max(gpu_memory, default=0.0),
            "gpu_memory_mean_percent": (
                sum(gpu_memory) / len(gpu_memory) if gpu_memory else 0.0
            ),
            "gpu_utilization_mean_percent": (
                sum(gpu_utilization) / len(gpu_utilization)
                if gpu_utilization
                else 0.0
            ),
            "gpu_temperature_peak_celsius": max(
                gpu_temperature, default=0.0
            ),
            "cpu_observed_peak_percent": max(cpu, default=0.0),
            "cpu_observed_mean_percent": (
                sum(cpu) / len(cpu) if cpu else 0.0
            ),
            "pauses": pauses,
        }

    def benchmark_parallelism(
        self,
        source: str,
        shard: str,
        count: int,
        *,
        dry_run: bool,
    ):
        if type(dry_run) is not bool:
            raise ValueError("parallelism benchmark dry-run flag must be boolean")
        assets = self._benchmark_scope(source, shard, count)
        scope_payload = self._batch_file_payload(assets).encode("ascii")
        scope_sha = sha256(scope_payload).hexdigest()
        control = (
            self.config.paths.data2_root
            / "control/benchmarks/parallelism"
            / source
            / shard
            / f"count{count:03d}"
        )
        report_root = self.config.paths.data2_root / "control/reports"
        report_paths = (
            report_root / "parallelism.json",
            report_root / "parallelism.md",
        )
        if dry_run:
            return {
                "decision": "dry_run",
                "source": source,
                "shard_id": shard,
                "assets": count,
                "scope_sha256": scope_sha,
                "render_resolution": self.config.render.resolution,
                "condition_views": self.config.render.num_views,
                "aligned_views": list(self.config.targets.views),
                "resolutions": list(self.config.targets.resolutions),
                "chunk_assets": self.config.parallelism.chunk_assets,
                "max_chunks_in_flight": (
                    self.config.parallelism.max_chunks_in_flight
                ),
                "evidence_paths": [str(path) for path in report_paths],
            }

        existing_json = _read_regular_bytes_nofollow(
            report_paths[0], missing_ok=True
        )
        existing_markdown = _read_regular_bytes_nofollow(
            report_paths[1], missing_ok=True
        )
        if existing_json is not None or existing_markdown is not None:
            if existing_json is None or existing_markdown is None:
                raise InfrastructureError(
                    "incomplete parallelism benchmark report publication"
                )
            try:
                existing = json.loads(existing_json)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise InfrastructureError(
                    f"invalid parallelism benchmark report: {error}"
                ) from error
            if (
                not isinstance(existing, dict)
                or existing.get("report_type")
                != "parallelism_benchmark"
                or existing.get("config_hash")
                != self.config.config_hash()
                or existing.get("scope_sha256") != scope_sha
            ):
                raise InfrastructureError(
                    "existing parallelism report belongs to another benchmark scope"
                )
            return report_paths

        instances = control / "instances.txt"
        manifest_path = control / "manifest.json"
        manifest = {
            "schema_version": 1,
            "artifact_type": "parallelism_benchmark_scope",
            "config_hash": self.config.config_hash(),
            "source": source,
            "shard_id": shard,
            "count": count,
            "scope_sha256": scope_sha,
        }
        existing_manifest = _read_regular_bytes_nofollow(
            manifest_path, missing_ok=True
        )
        if existing_manifest is not None:
            try:
                decoded_manifest = json.loads(existing_manifest)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise InfrastructureError(
                    f"invalid benchmark scope manifest: {error}"
                ) from error
            if decoded_manifest != manifest:
                raise InfrastructureError(
                    "benchmark scope manifest identity changed"
                )
            if _read_regular_bytes_nofollow(instances) != scope_payload:
                raise InfrastructureError(
                    "benchmark frozen instance identity changed"
                )
        else:
            existing_instances = _read_regular_bytes_nofollow(
                instances, missing_ok=True
            )
            if existing_instances is not None and existing_instances != scope_payload:
                raise InfrastructureError(
                    "orphan benchmark instance identity changed"
                )
            _atomic_write_bytes_nofollow(instances, scope_payload)
            _atomic_write_bytes_nofollow(
                manifest_path,
                json.dumps(
                    manifest, sort_keys=True, separators=(",", ":")
                ).encode("utf-8"),
            )

        context = self._benchmark_context(
            source, shard, count, instances
        )
        self._read_raw_records(
            context.source_root / "raw/metadata.csv", assets
        )
        benchmark = PipelineServices(
            self.config,
            resource_guard=self.resource_guard,
            pilot_reader=self.pilot_reader,
            reference_counter=self.reference_counter,
            project_accounting=self.project_accounting,
            registry_store=self.registry,
            disk_usage=self.disk_usage,
            tool_commit=self._tool_commit,
        )
        benchmark.runner.checkpoint_path = (
            lambda _context: control / "checkpoint.json"
        )
        benchmark.runner.quality_ledger_path = (
            lambda _context: control / "quality.json"
        )
        for attribute in (
            "process_factory",
            "supervisor_factory",
            "monotonic_clock",
            "sleeper",
            "killpg",
            "getpgid",
            "termination_grace_seconds",
            "reap_timeout_seconds",
            "monitor_interval_seconds",
            "process_poll_interval_seconds",
            "environment",
            "utc_clock",
        ):
            if hasattr(self.runner, attribute):
                value = getattr(self.runner, attribute)
                if attribute == "environment":
                    value = dict(value)
                setattr(benchmark.runner, attribute, value)
        benchmark_commands = frozenset(
            command.name
            for command in build_preprocessing_dag(
                context, self.config
            )
            if command.name
            not in {"download", "build_packs", "archive_raw", "cleanup_local"}
        )

        def command_builder(candidate, config, profile):
            return tuple(
                command
                for command in build_preprocessing_dag(
                    candidate, config, profile
                )
                if command.name in benchmark_commands
            )

        benchmark.runner.command_builder = command_builder
        before = benchmark.runner.load_checkpoint(
            control / "checkpoint.json", context.shard_id, context.gate
        )
        resumed = bool(before.completed_commands or before.attempts)
        started_at = datetime.now(timezone.utc)
        started = time.monotonic()
        benchmark.runner.run_shard(context)
        elapsed = time.monotonic() - started
        finished_at = datetime.now(timezone.utc)
        audit_passed = benchmark.runner.validate_completed_commands(
            context, ("validate_outputs",)
        )
        checkpoint = benchmark.runner.load_checkpoint(
            control / "checkpoint.json", context.shard_id, context.gate
        )
        if self.telemetry_flush is not None:
            self.telemetry_flush()
        telemetry = self._benchmark_telemetry(
            self.config.paths.data2_root
            / "control/telemetry/resources.jsonl",
            context.shard_id,
            started_at,
            finished_at,
        )
        resources = self._benchmark_resource_summary(
            telemetry, self.config.parallelism.gpu_count
        )
        from .reporting import parallelism_summary, write_report

        summary = parallelism_summary(
            completed_assets=len(checkpoint.quality_outcomes),
            elapsed_seconds=elapsed,
            gpu_peak_percent=resources["gpu_memory_peak_percent"],
            gpu_steady_state_percent=(
                resources["gpu_memory_mean_percent"]
            ),
            cpu_assigned_cores=(
                self.config.parallelism.cpu_physical_cores
            ),
            audit_passed=audit_passed,
        )
        measurement_valid = not resumed and resources["telemetry_valid"]
        decision_passed = summary["passed"] and measurement_valid
        report = {
            "schema_version": 1,
            "report_type": "parallelism_benchmark",
            "decision": "passed" if decision_passed else "held",
            "config_hash": self.config.config_hash(),
            "tool_commit": self._resolved_tool_commit(),
            "source": source,
            "shard_id": shard,
            "count": count,
            "scope_sha256": scope_sha,
            "created_at": finished_at.isoformat(),
            "measurement_valid": measurement_valid,
            "resumed": resumed,
            "summary": summary,
            "stage_seconds": dict(
                sorted(benchmark.runner.last_command_timings.items())
            ),
            "resources": resources,
            "retries": sum(
                max(0, attempts - 1)
                for attempts in checkpoint.attempts.values()
            ),
            "quality": {
                "terminal_assets": len(checkpoint.quality_outcomes),
                "completed_assets": sum(
                    outcome == "completed"
                    for outcome in checkpoint.quality_outcomes.values()
                ),
                "quarantined_assets": sum(
                    outcome != "completed"
                    for outcome in checkpoint.quality_outcomes.values()
                ),
            },
            "audit": {
                "passed": audit_passed,
                "families_validated": len(PACK_FAMILIES),
                "raw_staging_validated": True,
            },
        }
        return write_report(report_root, "parallelism", report)

    def report(self, gate: str | None, hardware_check: bool = False):
        if self.report_builder is None:
            raise IntegrationProviderRequired("report builder provider is required")
        return self.report_builder(gate, hardware_check)


def build_services(config: PipelineConfig) -> PipelineServices:
    return PipelineServices(config)
