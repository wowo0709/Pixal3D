from collections import deque
import csv
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import shlex
import signal
import stat
import subprocess
import tempfile
import time
from typing import Callable, Mapping, Protocol, Sequence
import zipfile

from .atomic_io import atomic_copy, atomic_write_json
from .commands import (
    CommandSpec,
    ShardContext,
    build_preprocessing_dag,
    expand_ranked,
)
from .config import PipelineConfig
from .packing import (
    PACK_FAMILIES,
    build_pack,
    file_sha,
    publish_pack,
    verify_pack,
)
from .registry import RegistryStore
from .resources import ResourceAction, ResourceLimitExceeded
from .validation import (
    ValidationError,
    validate_render_dir,
    validate_scale,
    validate_sparse_latent,
    validate_ss_latent,
)


CHECKPOINT_SCHEMA_VERSION = 1
MAX_COMMAND_ATTEMPTS = 3
QUALITY_WINDOW_SIZE = 500


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
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

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


class PipelineStopped(RuntimeError):
    def __init__(self, report: EscalationReport, exit_code: int):
        super().__init__(report.reason)
        self.report = report
        self.exit_code = exit_code


class PilotReader(Protocol):
    def p95_peak_local_bytes(self, source: str) -> int:
        """Return a validated, positive pilot p95 for one asset."""


class RawReferenceCounter(Protocol):
    def pending_references(
        self,
        source: str,
        raw_relative_path: str,
        *,
        excluding_shard_id: str,
        excluding_batch_id: str,
    ) -> int:
        """Return unarchived references excluding the just-archived batch."""


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
    ) -> int:
        raise IntegrationProviderRequired(
            "raw reference counter is required before data2 deletion"
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
        self._outcomes: deque[tuple[bool, bool]] = deque(maxlen=window_size)

    def record(self, *, succeeded: bool, schema_failure: bool) -> None:
        if not isinstance(succeeded, bool) or not isinstance(
            schema_failure, bool
        ):
            raise TypeError("quality outcomes must be booleans")
        self._outcomes.append((succeeded, schema_failure))

    @property
    def count(self) -> int:
        return len(self._outcomes)

    def violation_reason(self) -> str | None:
        if len(self._outcomes) < QUALITY_WINDOW_SIZE:
            return None
        failures = sum(not succeeded for succeeded, _ in self._outcomes)
        schema_failures = sum(schema for _, schema in self._outcomes)
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

    ordered = tuple(sorted(_validated_asset_sha(item) for item in asset_sha256s))
    if len(ordered) != len(set(ordered)):
        raise ValueError("duplicate asset SHA-256")
    if not ordered:
        return ()

    per_asset = (p95_peak_bytes * 5 + 3) // 4
    budget = local_usable_bytes * 4 // 5
    batch_size = min(shard_size, budget // per_asset)
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
    }:
        raise CheckpointError(f"invalid checkpoint schema: {path}")
    if value["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(f"unsupported checkpoint schema: {path}")
    shard_id = value["shard_id"]
    completed = value["completed_commands"]
    attempts = value["attempts"]
    if not isinstance(shard_id, str) or not shard_id:
        raise CheckpointError(f"invalid checkpoint shard identity: {path}")
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
    return PipelineCheckpoint(
        shard_id=shard_id,
        completed_commands=list(completed),
        attempts=dict(attempts),
        schema_version=CHECKPOINT_SCHEMA_VERSION,
    )


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
        checkpoint_path: Callable[[ShardContext], Path] | None = None,
        process_factory=subprocess.Popen,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        killpg: Callable[[int, int], None] = os.killpg,
        getpgid: Callable[[int], int] = os.getpgid,
        termination_grace_seconds: float = 60.0,
        monitor_interval_seconds: float = 5.0,
        environment: Mapping[str, str] | None = None,
        quality_gate: RollingQualityGate | None = None,
        utc_clock: Callable[[], datetime] | None = None,
    ):
        if termination_grace_seconds < 0:
            raise ValueError("termination grace must be non-negative")
        if monitor_interval_seconds <= 0:
            raise ValueError("monitor interval must be positive")
        self.config = config
        self.resource_guard = resource_guard
        self.validators = validators
        self.internal_handlers = internal_handlers
        self.command_builder = command_builder or build_preprocessing_dag
        self.report_writer = report_writer
        self.checkpoint_path = checkpoint_path or (
            lambda context: context.work_root / "checkpoint.json"
        )
        self.process_factory = process_factory
        self.monotonic_clock = monotonic_clock
        self.sleeper = sleeper
        self.killpg = killpg
        self.getpgid = getpgid
        self.termination_grace_seconds = termination_grace_seconds
        self.monitor_interval_seconds = monitor_interval_seconds
        self.environment = dict(os.environ if environment is None else environment)
        self.quality_gate = quality_gate or RollingQualityGate()
        self.utc_clock = utc_clock or (lambda: datetime.now(timezone.utc))
        self.active_context: ShardContext | None = None

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
        except InfrastructureError:
            raise
        except Exception:
            return False

    def run_shard(self, context: ShardContext) -> None:
        if self.active_context is not None:
            raise RuntimeError("pipeline runner is already active")
        self.active_context = context
        checkpoint_path = self.checkpoint_path(context)
        try:
            try:
                checkpoint = self.load_checkpoint(
                    checkpoint_path, context.shard_id
                )
            except CheckpointError as error:
                self.stop(
                    context,
                    "checkpoint",
                    str(error),
                    PipelineCheckpoint(context.shard_id),
                    category=EscalationCategory.INFRASTRUCTURE,
                    exit_code=2,
                    save_checkpoint=False,
                )

            for command in self.command_builder(context, self.config):
                if command.name in checkpoint.completed_commands:
                    try:
                        if self._valid_output(command.name):
                            continue
                    except InfrastructureError as error:
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

                    try:
                        self.execute(command, context.shard_id)
                        if not self._valid_output(command.name):
                            raise OutputValidationError(
                                f"validation failed: {command.name}"
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
                    except Exception as error:
                        checkpoint.attempts[command.name] = prior_attempts + 1
                        self.save_checkpoint(checkpoint_path, checkpoint)
                        if self.is_infrastructure_error(error):
                            self.stop(
                                context,
                                command.name,
                                str(error) or type(error).__name__,
                                checkpoint,
                                category=EscalationCategory.INFRASTRUCTURE,
                                exit_code=2,
                            )
                        if checkpoint.attempts[command.name] >= MAX_COMMAND_ATTEMPTS:
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
                            )
                        continue

                    checkpoint.complete(command.name)
                    self.save_checkpoint(checkpoint_path, checkpoint)
                    break
        finally:
            self.active_context = None

    def resume_shard(self, context: ShardContext) -> None:
        self.run_shard(context)

    def execute(self, command: CommandSpec, shard_id: str) -> None:
        if not command.argv:
            raise InfrastructureError(f"empty command argv: {command.name}")
        if command.gpu_ranks < 0:
            raise InfrastructureError(
                f"invalid rank count for command {command.name}: "
                f"{command.gpu_ranks}"
            )
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
            for argv, additions in expand_ranked(command):
                environment = dict(self.environment)
                environment.update(dict(additions))
                process = self.process_factory(
                    argv, env=environment, start_new_session=True
                )
                processes.append(process)
            self._monitor_processes(
                processes, paused_groups, shard_id, command
            )
        except BaseException as error:
            try:
                self._terminate_and_reap(processes, paused_groups)
            except BaseException as cleanup_error:
                raise cleanup_error from error
            raise
        else:
            self._reap(processes)

    def _monitor_processes(
        self,
        processes,
        paused_groups: set[int],
        shard_id: str,
        command: CommandSpec,
    ) -> None:
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

            decision = self.resource_guard.check(shard_id, command.name)
            if decision.action == ResourceAction.STOP:
                raise ResourceLimitExceeded(decision.reasons)
            if decision.action == ResourceAction.PAUSE:
                for process, status in zip(processes, statuses):
                    if status is None and process.pid not in paused_groups:
                        self._signal_process(process, signal.SIGSTOP)
                        paused_groups.add(process.pid)
            elif decision.action == ResourceAction.RUN and paused_groups:
                for process, status in zip(processes, statuses):
                    if status is None and process.pid in paused_groups:
                        self._signal_process(process, signal.SIGCONT)
                        paused_groups.discard(process.pid)
            self.sleeper(self.monitor_interval_seconds)

    def _signal_process(self, process, sent_signal: int) -> None:
        group_id = self.getpgid(process.pid)
        if group_id != process.pid:
            raise ProcessGroupSafetyError(
                f"refusing to signal unexpected process group "
                f"{group_id} for pid {process.pid}"
            )
        self.killpg(group_id, sent_signal)

    @staticmethod
    def _alive(processes):
        return [process for process in processes if process.poll() is None]

    @staticmethod
    def _reap(processes) -> None:
        failure = None
        for process in processes:
            try:
                process.wait()
            except BaseException as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure

    def _terminate_and_reap(self, processes, paused_groups: set[int]) -> None:
        failure = None

        def signal_all(candidates, sent_signal):
            nonlocal failure
            for process in candidates:
                try:
                    self._signal_process(process, sent_signal)
                except BaseException as error:
                    if failure is None:
                        failure = error

        alive = self._alive(processes)
        signal_all(
            [process for process in alive if process.pid in paused_groups],
            signal.SIGCONT,
        )
        paused_groups.clear()
        signal_all(alive, signal.SIGTERM)

        deadline = self.monotonic_clock() + self.termination_grace_seconds
        alive = self._alive(processes)
        while alive and self.monotonic_clock() < deadline:
            self.sleeper(
                min(
                    1.0,
                    max(0.0, deadline - self.monotonic_clock()),
                )
            )
            alive = self._alive(processes)
        signal_all(alive, signal.SIGKILL)
        try:
            self._reap(processes)
        except BaseException as error:
            if failure is None:
                failure = error
        if failure is not None:
            raise failure

    def load_checkpoint(
        self, path: Path, shard_id: str
    ) -> PipelineCheckpoint:
        path = Path(path)
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            return PipelineCheckpoint(shard_id)
        except OSError as error:
            raise CheckpointError(
                f"cannot inspect checkpoint: {path}: {error}"
            ) from error
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise CheckpointError(
                f"checkpoint is not a regular file: {path}"
            )
        try:
            value = json.loads(path.read_text())
        except Exception as error:
            raise CheckpointError(f"invalid checkpoint: {path}: {error}") from error
        checkpoint = _required_checkpoint_dict(value, path)
        if checkpoint.shard_id != shard_id:
            raise CheckpointError(
                f"checkpoint shard identity mismatch: "
                f"expected {shard_id}, found {checkpoint.shard_id}"
            )
        return checkpoint

    def save_checkpoint(
        self, path: Path, checkpoint: PipelineCheckpoint
    ) -> None:
        validated = _required_checkpoint_dict(asdict(checkpoint), Path(path))
        atomic_write_json(path, asdict(validated))

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
    ) -> None:
        if save_checkpoint:
            self.save_checkpoint(self.checkpoint_path(context), checkpoint)
        created_at = self.utc_clock()
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("escalation timestamp must be timezone-aware")
        report = EscalationReport(
            source=context.source,
            shard_id=context.shard_id,
            command=command,
            category=category,
            reason=reason,
            recent_telemetry=tuple(
                self.resource_guard.last_five_minutes()
            ),
            completed_counts={
                "commands": len(checkpoint.completed_commands),
                "outcomes": self.quality_gate.count,
            },
            safe_resume_command=(
                "python -m data_toolkit.pipeline.cli resume "
                f"--source {shlex.quote(context.source)} "
                f"--shard {shlex.quote(context.shard_id)}"
            ),
            recovery_choices=self._recovery_choices(category),
            created_at=created_at.astimezone(timezone.utc).isoformat(),
        )
        if self.report_writer is not None:
            self.report_writer(report)
        raise PipelineStopped(report, exit_code)


def _atomic_write_text(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        directory = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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


def _open_regular_beneath(root: Path, relative: Path):
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        root_fd = os.open(root, flags | os.O_DIRECTORY)
    except OSError as error:
        raise ValidationError(f"unsafe raw source root: {root}") from error
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
                raise ValidationError(
                    f"symlink or unsafe raw source: {relative.as_posix()}"
                ) from error
            os.close(directory_fd)
            directory_fd = next_fd
        try:
            file_fd = os.open(
                relative.parts[-1],
                flags | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
        except OSError as error:
            raise ValidationError(
                f"symlink or missing raw source: {relative.as_posix()}"
            ) from error
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


def _unlink_regular_beneath(root: Path, relative: Path) -> bool:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        directory_fd = os.open(root, flags | os.O_DIRECTORY)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ValidationError(f"unsafe raw source root: {root}") from error
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
                raise ValidationError(
                    f"symlink or unsafe raw source: {relative.as_posix()}"
                ) from error
            os.close(directory_fd)
            directory_fd = next_fd
        try:
            mode = os.stat(
                relative.parts[-1],
                dir_fd=directory_fd,
                follow_symlinks=False,
            ).st_mode
        except FileNotFoundError:
            return False
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ValidationError(
                f"refusing to delete unsafe raw source: {relative.as_posix()}"
            )
        os.unlink(relative.parts[-1], dir_fd=directory_fd)
        os.fsync(directory_fd)
        return True
    finally:
        os.close(directory_fd)


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
    parent = _safe_destination_parent(root, relative)
    destination = root / relative
    temporary = None
    digest = sha256()
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != expected_sha:
            raise ValidationError(
                f"raw checksum mismatch: {relative.as_posix()}"
            )
        os.replace(temporary, destination)
        temporary = None
        directory = os.open(
            parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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


class PipelineServices:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        resource_guard=None,
        pilot_reader: PilotReader | None = None,
        reference_counter: RawReferenceCounter | None = None,
        registry_store=None,
        disk_usage: Callable[[Path], object] = shutil.disk_usage,
        runner=None,
        output_validator: Callable[[ShardContext], None] | None = None,
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
    ):
        self.config = config
        self.registry = registry_store or RegistryStore(
            config.paths.data2_root / "control" / "assets.parquet"
        )
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
        self.disk_usage = disk_usage
        self.output_validator = output_validator or self._validate_all_outputs
        self.resolution_validator = (
            resolution_validator or self._validate_resolution
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
        )

    def _active_context(self) -> ShardContext:
        context = getattr(self.runner, "active_context", None)
        if context is None:
            raise InfrastructureError("internal handler has no active shard")
        return context

    def _checkpoint_path(self, context: ShardContext) -> Path:
        return (
            self.config.paths.data2_root
            / "control/checkpoints"
            / context.source
            / context.shard_id
            / f"{context.batch_id}.json"
        )

    def _write_escalation(self, report: EscalationReport) -> None:
        path = (
            self.config.paths.data2_root
            / "control/reports/escalations"
            / report.source
            / report.shard_id
            / f"{report.command}.json"
        )
        atomic_write_json(path, asdict(report))

    def _resolved_tool_commit(self) -> str:
        if self._tool_commit is not None:
            return self._tool_commit
        try:
            completed = subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
        except Exception as error:
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
        frame = self.registry.load()
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

    def _batch_root(self, source: str, shard_id: str) -> Path:
        return (
            self.config.paths.data2_root
            / "control/shards"
            / source
            / shard_id
        )

    @staticmethod
    def _batch_file_payload(batch: tuple[str, ...]) -> str:
        return "".join(f"{item}\n" for item in batch)

    def _read_frozen_batches(
        self,
        source: str,
        shard_id: str,
        expected_shas: tuple[str, ...],
        *,
        allow_subset: bool = False,
    ) -> tuple[tuple[str, ...], ...] | None:
        root = self._batch_root(source, shard_id)
        if root.is_symlink():
            raise InfrastructureError(f"frozen batch root is a symlink: {root}")
        if not root.exists():
            return None
        if not root.is_dir():
            raise InfrastructureError(
                f"frozen batch root is not a directory: {root}"
            )
        marker_path = root / "batches.json"
        try:
            marker_mode = marker_path.lstat().st_mode
            if stat.S_ISLNK(marker_mode) or not stat.S_ISREG(marker_mode):
                raise InfrastructureError(
                    f"frozen batch marker is not a regular file: {marker_path}"
                )
            marker = json.loads(marker_path.read_text())
        except InfrastructureError:
            raise
        except Exception as error:
            raise InfrastructureError(
                f"invalid frozen batch manifest: {marker_path}: {error}"
            ) from error
        if (
            not isinstance(marker, dict)
            or set(marker)
            != {"schema_version", "source", "shard_id", "config_hash", "batches"}
            or marker["schema_version"] != 1
            or marker["source"] != source
            or marker["shard_id"] != shard_id
            or marker["config_hash"] != self.config.config_hash()
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
                path_mode = path.lstat().st_mode
                if stat.S_ISLNK(path_mode) or not stat.S_ISREG(path_mode):
                    raise InfrastructureError(
                        f"frozen batch is not a regular file: {path}"
                    )
                payload = path.read_bytes()
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
            except Exception as error:
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
        actual_names = sorted(path.name for path in root.glob("batch*.txt"))
        if actual_names != expected_names:
            raise InfrastructureError(
                f"frozen batch file set mismatch: {root}"
            )
        flattened = tuple(item for batch in batches for item in batch)
        if allow_subset:
            expected_set = set(expected_shas)
            flattened_set = set(flattened)
            identity_valid = (
                bool(flattened)
                and flattened_set.issubset(expected_set)
                and flattened
                == tuple(
                    item for item in expected_shas if item in flattened_set
                )
            )
        else:
            identity_valid = flattened == expected_shas
        if not identity_valid:
            raise InfrastructureError(
                f"frozen batch asset identity mismatch: {root}"
            )
        return tuple(batches)

    def _freeze_batches(
        self,
        source: str,
        shard_id: str,
        batches: tuple[tuple[str, ...], ...],
    ) -> tuple[tuple[str, ...], ...]:
        destination = self._batch_root(source, shard_id)
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
            for index, batch in enumerate(batches):
                name = f"batch{index:03d}.txt"
                payload = self._batch_file_payload(batch)
                _atomic_write_text(temporary / name, payload)
                entries.append(
                    {
                        "name": name,
                        "count": len(batch),
                        "sha256": sha256(payload.encode("ascii")).hexdigest(),
                    }
                )
            atomic_write_json(
                temporary / "batches.json",
                {
                    "schema_version": 1,
                    "source": source,
                    "shard_id": shard_id,
                    "config_hash": self.config.config_hash(),
                    "batches": entries,
                },
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
        source: str,
        shard_id: str,
        count: int | None,
        *,
        freeze: bool,
    ) -> tuple[tuple[str, ...], ...]:
        shas = self._registry_shas(source, shard_id, count)
        existing = self._read_frozen_batches(source, shard_id, shas)
        if existing is not None:
            return existing
        if not freeze and self._batch_root(source, shard_id).exists():
            raise InfrastructureError("incomplete frozen batch publication")
        p95 = self.pilot_reader.p95_peak_local_bytes(source)
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
        batches = plan_work_batches(shas, usable, p95, self.config.shard_size)
        return self._freeze_batches(source, shard_id, batches) if freeze else batches

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
            source, shard, count, freeze=freeze
        )
        return tuple(
            f"batch{index:03d}: {len(batch)} assets"
            for index, batch in enumerate(batches)
        )

    @staticmethod
    def _instances(context: ShardContext) -> tuple[str, ...]:
        try:
            mode = context.instances.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ValidationError(
                    f"instances manifest is not a regular file: {context.instances}"
                )
            values = tuple(
                _validated_asset_sha(item)
                for item in context.instances.read_text().splitlines()
            )
        except Exception as error:
            raise ValidationError(
                f"invalid instances manifest: {context.instances}: {error}"
            ) from error
        if not values or len(values) != len(set(values)) or tuple(sorted(values)) != values:
            raise ValidationError(
                f"invalid instances manifest: {context.instances}"
            )
        return values

    @staticmethod
    def _read_raw_records(path: Path, selected: tuple[str, ...]) -> tuple[dict, ...]:
        try:
            mode = Path(path).lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ValidationError(
                    f"raw metadata is not a regular file: {path}"
                )
            with Path(path).open(newline="", encoding="utf-8") as stream:
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
                    by_sha[sha] = {
                        "sha256": sha,
                        "local_path": relative.as_posix(),
                    }
        except ValidationError:
            raise
        except Exception as error:
            raise ValidationError(f"invalid raw metadata: {path}: {error}") from error
        missing = set(selected) - set(by_sha)
        if missing:
            raise ValidationError(
                f"raw metadata missing selected assets: {sorted(missing)}"
            )
        records = tuple(by_sha[item] for item in selected)
        paths = [item["local_path"] for item in records]
        if len(paths) != len(set(paths)):
            raise ValidationError("duplicate selected raw path")
        return records

    @staticmethod
    def _write_raw_records(path: Path, records: tuple[dict, ...]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                writer = csv.DictWriter(
                    stream, fieldnames=("sha256", "local_path")
                )
                writer.writeheader()
                writer.writerows(records)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            temporary = None
            directory = os.open(
                path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _source_raw_relative(relative: Path) -> Path:
        zip_value = _zip_parts(relative)
        return zip_value[0] if zip_value is not None else relative

    def stage_raw(self, context: ShardContext) -> None:
        selected = self._instances(context)
        records = self._read_raw_records(
            context.source_root / "raw/metadata.csv", selected
        )
        for record in records:
            relative = _safe_raw_relative(record["local_path"])
            zip_value = _zip_parts(relative)
            if zip_value is None:
                with _open_regular_beneath(
                    context.source_root, relative
                ) as stream:
                    _atomic_stage_stream(
                        stream,
                        context.download_root,
                        relative,
                        record["sha256"],
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
                                record["sha256"],
                            )
            except ValidationError:
                raise
            except (OSError, zipfile.BadZipFile) as error:
                raise ValidationError(
                    f"invalid raw ZIP archive: {archive_relative.as_posix()}: {error}"
                ) from error
        self._write_raw_records(
            context.download_root / "raw/metadata.csv", records
        )

    def _staged_records(self, context: ShardContext) -> tuple[dict, ...]:
        return self._read_raw_records(
            context.download_root / "raw/metadata.csv",
            self._instances(context),
        )

    def _validate_staged_raw(self, context: ShardContext) -> None:
        for record in self._staged_records(context):
            relative = _safe_raw_relative(record["local_path"])
            with _open_regular_beneath(
                context.download_root, relative
            ) as stream:
                if _sha_stream(stream) != record["sha256"]:
                    raise ValidationError(
                        f"staged raw checksum mismatch: {relative.as_posix()}"
                    )

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

    def _validate_resolution(
        self, context: ShardContext, resolution: int
    ) -> None:
        if resolution not in self.config.targets.resolutions:
            raise ValidationError(f"unexpected resolution: {resolution}")
        for asset_sha in self._instances(context):
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
                    validate_sparse_latent(
                        output, resolution, resolution**3
                    )
                    validate_scale(
                        output.with_name(f"view{view:02d}_scale.json")
                    )

    def _validate_asset_outputs(
        self, context: ShardContext, asset_sha: str
    ) -> None:
        validate_render_dir(
            context.output_root / "renders_cond" / asset_sha,
            self.config.render.num_views,
            self.config.render.resolution,
        )
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
                    validate_sparse_latent(
                        output, resolution, resolution**3
                    )
                    validate_scale(
                        output.with_name(f"view{view:02d}_scale.json")
                    )
        for view in self.config.targets.views:
            output = (
                context.output_root
                / self._ss_directory()
                / asset_sha
                / f"view{view:02d}.npz"
            )
            validate_ss_latent(output)
            validate_scale(output.with_name(f"view{view:02d}_scale.json"))

    def _validate_all_outputs(self, context: ShardContext) -> None:
        for asset_sha in self._instances(context):
            self._validate_asset_outputs(context, asset_sha)

    def validate_outputs(self, context: ShardContext) -> None:
        failures = []
        for asset_sha in self._instances(context):
            try:
                self._validate_asset_outputs(context, asset_sha)
            except ValidationError as error:
                self.runner.quality_gate.record(
                    succeeded=False, schema_failure=True
                )
                failures.append((asset_sha, error))
            else:
                self.runner.quality_gate.record(
                    succeeded=True, schema_failure=False
                )
        if failures:
            asset_sha, error = failures[0]
            raise ValidationError(
                f"output validation failed for {asset_sha}: {error}"
            ) from error

    def cleanup_voxels(
        self, context: ShardContext, resolution: int
    ) -> None:
        self.resolution_validator(context, resolution)
        for path in (
            context.work_root / f"dual_grid_view_{resolution}",
            context.work_root / f"pbr_voxels_view_fix_{resolution}",
        ):
            if path.is_symlink():
                raise ValidationError(f"refusing to clean symlink: {path}")
            shutil.rmtree(path, ignore_errors=False) if path.exists() else None

    def _pack_members(
        self, context: ShardContext
    ) -> Mapping[str, Sequence[Path]]:
        members: dict[str, list[Path]] = {
            family: [] for family in PACK_FAMILIES
        }
        for asset_sha in self._instances(context):
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

    def build_packs(self, context: ShardContext) -> None:
        self.output_validator(context)
        shas = self._instances(context)
        members = self.pack_member_builder(context)
        if set(members) != set(PACK_FAMILIES):
            raise ValidationError(
                "pack member mapping must contain exactly eight families"
            )
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
            completed_count=len(shas),
            quarantined_count=0,
        )
        if (
            len(manifests) != len(PACK_FAMILIES)
            or any(not getattr(item, "validated_at", "") for item in manifests)
        ):
            raise ValidationError("pack publisher returned unvalidated packs")
        self.published_batch_verifier(context)

    def _verify_published_batch(self, context: ShardContext) -> None:
        prepared = self.config.paths.data2_root / "prepared"
        index_path = (
            prepared / "index" / context.source / f"{context.shard_id}.json"
        )
        try:
            index_mode = index_path.lstat().st_mode
            if stat.S_ISLNK(index_mode) or not stat.S_ISREG(index_mode):
                raise ValidationError(
                    f"published shard index is not a regular file: {index_path}"
                )
            index = json.loads(index_path.read_text())
            entries = index["batches"][context.batch_id]
        except Exception as error:
            raise ValidationError(
                f"invalid published shard index: {index_path}: {error}"
            ) from error
        if (
            index.get("source") != context.source
            or index.get("shard_id") != context.shard_id
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
                    family_root
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
                verify_pack(pack_path, manifest_path)
                manifest = json.loads(manifest_path.read_text())
                if (
                    manifest["shard_id"] != context.shard_id
                    or manifest["batch_id"] != context.batch_id
                    or manifest["family"] != family
                    or manifest["config_hash"] != self.config.config_hash()
                    or not manifest["validated_at"]
                    or entry["pack_sha256"] != manifest["pack_sha256"]
                    or entry["manifest_sha256"] != file_sha(manifest_path)
                ):
                    raise ValidationError(
                        f"published pack identity mismatch: {family}"
                    )
            except (KeyError, TypeError) as error:
                raise ValidationError(
                    f"invalid published pack entry: {family}"
                ) from error

    def _raw_archive_paths(self, context: ShardContext) -> tuple[Path, Path]:
        archive = (
            self.config.paths.data3_root
            / "archive/raw"
            / context.source
            / context.shard_id
            / f"{context.batch_id}.tar"
        )
        return archive, archive.with_suffix(".tar.manifest.json")

    def _verify_raw_archive(self, context: ShardContext) -> None:
        archive, manifest_path = self._raw_archive_paths(context)
        if archive.is_symlink() or manifest_path.is_symlink():
            raise ValidationError(
                f"refusing symlinked raw archive output: {archive}"
            )
        verify_pack(archive, manifest_path)
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as error:
            raise ValidationError(
                f"invalid raw archive manifest: {manifest_path}: {error}"
            ) from error
        if (
            manifest.get("shard_id") != context.shard_id
            or manifest.get("batch_id") != context.batch_id
            or manifest.get("family") != "raw"
            or manifest.get("config_hash") != self.config.config_hash()
            or not manifest.get("validated_at")
            or tuple(manifest.get("asset_sha256s", ()))
            != self._instances(context)
            or len(manifest.get("members", ())) != len(self._instances(context))
        ):
            raise ValidationError(
                f"raw archive identity mismatch: {manifest_path}"
            )

    def archive_raw(self, context: ShardContext) -> None:
        self.published_batch_verifier(context)
        records = self._staged_records(context)
        members = [
            _safe_raw_relative(record["local_path"]) for record in records
        ]
        local_archive = (
            context.work_root / "raw_archive" / f"{context.batch_id}.tar"
        )
        shas = self._instances(context)
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
            completed_count=len(shas),
            quarantined_count=0,
        )
        local_manifest = local_archive.with_suffix(".tar.manifest.json")
        verify_pack(local_archive, local_manifest)
        manifest = replace(
            manifest,
            validated_at=datetime.now(timezone.utc).isoformat(),
        )
        atomic_write_json(local_manifest, asdict(manifest))
        verify_pack(local_archive, local_manifest)
        if (
            len(manifest.members) != len(records)
            or sum(item.size for item in manifest.members)
            != sum((context.download_root / member).stat().st_size for member in members)
            or {item.path: item.sha256 for item in manifest.members}
            != {
                record["local_path"]: record["sha256"] for record in records
            }
        ):
            raise ValidationError("raw archive count, byte, or member hash mismatch")

        archive, archive_manifest = self._raw_archive_paths(context)
        if archive.is_symlink() or archive_manifest.is_symlink():
            raise ValidationError(
                f"refusing symlinked raw archive output: {archive}"
            )
        if archive.exists() and archive_manifest.exists():
            verify_pack(archive, archive_manifest)
            existing = json.loads(archive_manifest.read_text())
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

        source_paths = {
            self._source_raw_relative(member).as_posix() for member in members
        }
        for relative_value in sorted(source_paths):
            pending = self.reference_counter.pending_references(
                context.source,
                relative_value,
                excluding_shard_id=context.shard_id,
                excluding_batch_id=context.batch_id,
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
            relative = _safe_raw_relative(relative_value)
            _unlink_regular_beneath(context.source_root, relative)

    def cleanup_local(self, context: ShardContext) -> None:
        self.published_batch_verifier(context)
        self.raw_archive_verifier(context)
        for root in (
            context.download_root,
            context.work_root,
            context.output_root,
        ):
            if root.is_symlink():
                raise ValidationError(f"refusing to clean symlink: {root}")
            if root.exists():
                shutil.rmtree(root)

    def _archive_is_valid(self, context: ShardContext) -> bool:
        try:
            self.raw_archive_verifier(context)
            return True
        except Exception:
            return False

    def _published_is_valid(self, context: ShardContext) -> bool:
        try:
            self.published_batch_verifier(context)
            return True
        except Exception:
            return False

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
                records = self._read_raw_records(
                    context.source_root / "raw/metadata.csv",
                    self._instances(context),
                )
                return len(records) == len(self._instances(context))
            if name == "stage_raw":
                self._validate_staged_raw(context)
                return True
            if name in {"dump_mesh", "dump_pbr"}:
                directory = "mesh_dumps" if name == "dump_mesh" else "pbr_dumps"
                return all(
                    (context.work_root / directory / f"{asset}.pickle").is_file()
                    for asset in self._instances(context)
                )
            if name == "asset_stats":
                return (
                    context.metadata_root / "asset_stats/metadata.csv"
                ).is_file()
            if name == "render_cond":
                for asset in self._instances(context):
                    validate_render_dir(
                        context.output_root / "renders_cond" / asset,
                        self.config.render.num_views,
                        self.config.render.resolution,
                    )
                return True
            for resolution in self.config.targets.resolutions:
                if name in {
                    f"encode_shape_{resolution}",
                    f"encode_pbr_{resolution}",
                    f"cleanup_voxels_{resolution}",
                }:
                    self.resolution_validator(context, resolution)
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
                    return all(
                        (
                            context.work_root
                            / directory
                            / asset
                            / f"view{view:02d}.vxz"
                        ).is_file()
                        for asset in self._instances(context)
                        for view in self.config.targets.views
                    )
            if name == f"encode_ss_{self.config.targets.ss_resolution}":
                for asset in self._instances(context):
                    for view in self.config.targets.views:
                        output = (
                            context.output_root
                            / self._ss_directory()
                            / asset
                            / f"view{view:02d}.npz"
                        )
                        validate_ss_latent(output)
                        validate_scale(
                            output.with_name(f"view{view:02d}_scale.json")
                        )
                return True
            if name == "validate_outputs":
                self.output_validator(context)
                return True
            if name == "build_packs":
                return self._published_is_valid(context)
        except Exception:
            return False
        raise InfrastructureError(f"missing production validator: {name}")

    def _audit_batch(self, context: ShardContext) -> None:
        self.published_batch_verifier(context)
        self.raw_archive_verifier(context)

    def _frozen_for_execution(
        self, source: str, shard: str
    ) -> tuple[tuple[str, ...], ...]:
        shas = self._registry_shas(source, shard)
        batches = self._read_frozen_batches(
            source, shard, shas, allow_subset=True
        )
        if batches is None:
            raise InfrastructureError(
                f"no frozen batch manifest for resume: {source}/{shard}"
            )
        return batches

    def _verify_logical_index(
        self,
        source: str,
        shard: str,
        batches: tuple[tuple[str, ...], ...],
    ) -> None:
        if self.published_batch_verifier == self._verify_published_batch:
            index_path = (
                self.config.paths.data2_root
                / "prepared/index"
                / source
                / f"{shard}.json"
            )
            try:
                index = json.loads(index_path.read_text())
                indexed_batches = set(index["batches"])
            except Exception as error:
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
                self.config, source, shard, f"batch{index:03d}"
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
        shas = self._registry_shas(source, shard, count)
        batches = self._read_frozen_batches(
            source, shard, shas, allow_subset=True
        )
        if batches is None:
            batches = self._planned_batches(
                source, shard, count, freeze=True
            )
        for index in range(len(batches)):
            context = ShardContext.from_config(
                self.config, source, shard, f"batch{index:03d}"
            )
            self.runner.run_shard(context)
            self.batch_auditor(context)
        self._verify_logical_index(source, shard, batches)

    def resume(self, source: str | None, shard: str | None) -> None:
        if source is None or shard is None:
            raise ValueError("resume requires source and shard")
        batches = self._frozen_for_execution(source, shard)
        for index in range(len(batches)):
            context = ShardContext.from_config(
                self.config, source, shard, f"batch{index:03d}"
            )
            self.runner.resume_shard(context)
            self.batch_auditor(context)
        self._verify_logical_index(source, shard, batches)

    def audit(self, source: str | None, shard: str | None) -> None:
        if source is None or shard is None:
            raise ValueError("audit requires source and shard")
        batches = self._frozen_for_execution(source, shard)
        for index in range(len(batches)):
            self.batch_auditor(
                ShardContext.from_config(
                    self.config, source, shard, f"batch{index:03d}"
                )
            )
        self._verify_logical_index(source, shard, batches)

    def report(self, gate: str | None, hardware_check: bool = False):
        if self.report_builder is None:
            raise IntegrationProviderRequired("report builder provider is required")
        return self.report_builder(gate, hardware_check)


def build_services(config: PipelineConfig) -> PipelineServices:
    return PipelineServices(config)
