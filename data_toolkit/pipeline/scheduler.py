from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
from typing import Callable, Mapping, Protocol, Sequence

from .commands import ShardContext
from .parallelism import NodeResourceBroker


CHUNK_CHECKPOINT_SCHEMA_VERSION = 1
CHUNK_MANIFEST_SCHEMA_VERSION = 1


class Lane(str, Enum):
    PREPARE = "prepare"
    RENDER = "render"
    GEOMETRY = "geometry"
    ENCODE = "encode"
    PUBLISH = "publish"


@dataclass(frozen=True)
class StageSpec:
    name: str
    lane: Lane
    dependencies: tuple[str, ...] = ()
    cpu_cores: int = 0
    gpu_indices: tuple[int, ...] = ()
    gpu_memory_percent: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("stage name must be non-empty")
        if not isinstance(self.lane, Lane):
            raise ValueError("stage lane must be a Lane")
        if (
            not isinstance(self.dependencies, tuple)
            or any(not isinstance(item, str) or not item for item in self.dependencies)
            or len(self.dependencies) != len(set(self.dependencies))
        ):
            raise ValueError("stage dependencies must be unique names")
        if type(self.cpu_cores) is not int or self.cpu_cores < 0:
            raise ValueError("stage CPU cores must be a nonnegative integer")
        if (
            not isinstance(self.gpu_indices, tuple)
            or any(type(index) is not int or index < 0 for index in self.gpu_indices)
            or len(self.gpu_indices) != len(set(self.gpu_indices))
        ):
            raise ValueError("stage GPU indices must be unique nonnegative integers")
        if not isinstance(self.gpu_memory_percent, (int, float)) or isinstance(
            self.gpu_memory_percent, bool
        ):
            raise ValueError("stage GPU memory percent must be numeric")
        memory = float(self.gpu_memory_percent)
        if not math.isfinite(memory) or memory < 0:
            raise ValueError("stage GPU memory percent must be finite and nonnegative")
        if not self.gpu_indices and memory:
            raise ValueError("stage GPU memory requires GPU indices")


@dataclass(frozen=True)
class ChunkContext:
    parent: ShardContext
    chunk_id: str
    instances: Path
    source_root: Path
    download_root: Path
    work_root: Path
    output_root: Path

    def assets(self) -> tuple[str, ...]:
        try:
            payload = self.instances.read_bytes()
            text = payload.decode("ascii")
        except (OSError, UnicodeDecodeError) as error:
            raise ChunkIdentityError(
                f"chunk identity cannot be read: {self.chunk_id}: {error}"
            ) from error
        if text and not text.endswith("\n"):
            raise ChunkIdentityError(
                f"chunk identity is missing final newline: {self.chunk_id}"
            )
        assets = tuple(text.splitlines())
        if (
            not assets
            or tuple(sorted(assets)) != assets
            or len(assets) != len(set(assets))
            or any(not _is_sha(item) for item in assets)
        ):
            raise ChunkIdentityError(f"invalid chunk identity: {self.chunk_id}")
        return assets

    def as_shard_context(self) -> ShardContext:
        return ShardContext(
            source=self.parent.source,
            shard_id=f"{self.parent.shard_id}-{self.chunk_id}",
            instances=self.instances,
            metadata_root=self.parent.metadata_root,
            source_root=self.source_root,
            download_root=self.download_root,
            work_root=self.work_root,
            output_root=self.output_root,
            batch_id=f"{self.parent.batch_id}_{self.chunk_id}",
            gate=self.parent.gate,
            record_prefix=f"{self.chunk_id}_",
        )


@dataclass
class ChunkCheckpoint:
    chunk_id: str
    instances_sha256: str
    config_hash: str
    completed_stages: list[str] = field(default_factory=list)
    worker_profiles: dict[str, dict[str, int]] = field(default_factory=dict)
    resource_peaks: dict[str, float] = field(default_factory=dict)
    promotion_started: bool = False
    promoted: bool = False
    schema_version: int = CHUNK_CHECKPOINT_SCHEMA_VERSION

    def complete(self, stage: str) -> None:
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)


@dataclass(frozen=True)
class SchedulerResult:
    chunks: tuple[ChunkContext, ...]
    max_chunks_in_flight: int
    publication_order: tuple[str, ...]
    intervals: Mapping[tuple[str, str], tuple[float, float]]

    def interval(self, chunk_id: str, stage: str) -> tuple[float, float]:
        return self.intervals[(chunk_id, stage)]


class ChunkStageExecutor(Protocol):
    def execute(
        self, chunk: ChunkContext, stage: StageSpec
    ) -> Mapping[str, float] | None:
        """Run one admitted stage and return optional numeric observations."""

    def validate(self, chunk: ChunkContext, stage: StageSpec) -> bool:
        """Validate a completed stage before it is skipped on restart."""


class ChunkExecutionError(RuntimeError):
    pass


class ChunkIdentityError(ChunkExecutionError):
    pass


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _instances_payload(assets: Sequence[str]) -> bytes:
    values = tuple(assets)
    if (
        not values
        or tuple(sorted(values)) != values
        or len(values) != len(set(values))
        or any(not _is_sha(item) for item in values)
    ):
        raise ChunkIdentityError("invalid frozen batch identity")
    return "".join(f"{item}\n" for item in values).encode("ascii")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    descriptor: int | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(name)
        os.write(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        temporary = None
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def choose_chunk_assets(
    *, configured: int, p95_scratch_bytes: int, usable_bytes: int
) -> int:
    for name, value in (
        ("configured", configured),
        ("p95 scratch bytes", p95_scratch_bytes),
        ("usable bytes", usable_bytes),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if configured not in {32, 64}:
        raise ValueError("configured chunk assets must be 32 or 64")
    required = (p95_scratch_bytes * configured * 5 + 3) // 4
    return configured if required <= usable_bytes else 32


def _tree_digest(root: Path) -> str:
    digest = sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        details = path.lstat()
        if stat.S_ISLNK(details.st_mode):
            raise ChunkExecutionError(f"unsafe symlink in chunk output: {path}")
        if stat.S_ISDIR(details.st_mode):
            digest.update(f"d:{relative}\0".encode())
            continue
        if not stat.S_ISREG(details.st_mode):
            raise ChunkExecutionError(f"unsafe chunk output type: {path}")
        digest.update(f"f:{relative}:{details.st_size}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _merge_record_parts(parent: ShardContext, chunk: ChunkContext) -> None:
    if not chunk.output_root.exists():
        return
    for source in sorted(chunk.output_root.rglob("new_records")):
        if not source.is_dir() or source.is_symlink():
            raise ChunkExecutionError(f"unsafe chunk record directory: {source}")
        relative = source.relative_to(chunk.output_root)
        destination = parent.output_root / relative
        destination.mkdir(parents=True, exist_ok=True)
        for part in sorted(source.iterdir()):
            if not part.is_file() or part.is_symlink():
                raise ChunkExecutionError(f"unsafe chunk record part: {part}")
            target = destination / f"{chunk.chunk_id}_{part.name}"
            if target.exists():
                if sha256(target.read_bytes()).digest() != sha256(part.read_bytes()).digest():
                    raise ChunkExecutionError(f"conflicting chunk record part: {target}")
                part.unlink()
            else:
                os.replace(part, target)


def promote_chunk_outputs(parent: ShardContext, chunk: ChunkContext) -> None:
    """Atomically move validated per-asset output directories into a batch root."""

    assets = chunk.assets()
    if not chunk.output_root.exists():
        return
    if chunk.output_root.is_symlink():
        raise ChunkExecutionError(f"unsafe chunk output root: {chunk.output_root}")
    _merge_record_parts(parent, chunk)
    for asset in assets:
        candidates = sorted(
            path
            for path in chunk.output_root.rglob(asset)
            if path.name == asset
        )
        for source in candidates:
            if not source.is_dir() or source.is_symlink():
                raise ChunkExecutionError(f"unsafe chunk asset output: {source}")
            relative = source.relative_to(chunk.output_root)
            destination = parent.output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if (
                    not destination.is_dir()
                    or destination.is_symlink()
                    or _tree_digest(source) != _tree_digest(destination)
                ):
                    raise ChunkExecutionError(
                        f"conflicting promoted chunk output: {destination}"
                    )
                shutil.rmtree(source)
            else:
                os.replace(source, destination)


class ParallelChunkScheduler:
    def __init__(
        self,
        *,
        config_hash: str,
        broker: NodeResourceBroker,
        executor: ChunkStageExecutor,
        stages: Sequence[StageSpec],
        checkpoint_root: Path,
        chunk_assets: int,
        max_chunks_in_flight: int,
        promoter: Callable[[ShardContext, ChunkContext], None],
        publisher: Callable[[ShardContext, tuple[ChunkContext, ...]], None],
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not _is_sha(config_hash):
            raise ValueError("scheduler config hash must be a SHA-256")
        if type(chunk_assets) is not int or chunk_assets <= 0:
            raise ValueError("chunk assets must be a positive integer")
        if type(max_chunks_in_flight) is not int or max_chunks_in_flight <= 0:
            raise ValueError("max chunks in flight must be a positive integer")
        values = tuple(stages)
        names = tuple(stage.name for stage in values)
        if not values or len(names) != len(set(names)):
            raise ValueError("scheduler stages must have unique names")
        known = set(names)
        for index, stage in enumerate(values):
            if not set(stage.dependencies) <= known:
                raise ValueError(f"unknown dependency for stage: {stage.name}")
            earlier = set(names[:index])
            if not set(stage.dependencies) <= earlier:
                raise ValueError("scheduler stages must be topologically ordered")
        self.config_hash = config_hash
        self.broker = broker
        self.executor = executor
        self.stages = values
        self.checkpoint_root = Path(checkpoint_root)
        self.chunk_assets = chunk_assets
        self.max_chunks_in_flight = max_chunks_in_flight
        self.promoter = promoter
        self.publisher = publisher
        self.monotonic_clock = monotonic_clock

    def _manifest(self, parent: ShardContext, chunks: Sequence[tuple[str, bytes]]) -> dict:
        parent_payload = parent.instances.read_bytes()
        return {
            "schema_version": CHUNK_MANIFEST_SCHEMA_VERSION,
            "config_hash": self.config_hash,
            "source": parent.source,
            "shard_id": parent.shard_id,
            "batch_id": parent.batch_id,
            "parent_instances_sha256": sha256(parent_payload).hexdigest(),
            "chunk_assets": self.chunk_assets,
            "chunks": [
                {
                    "chunk_id": chunk_id,
                    "count": len(payload.decode("ascii").splitlines()),
                    "instances_sha256": sha256(payload).hexdigest(),
                }
                for chunk_id, payload in chunks
            ],
        }

    def freeze_chunks(
        self, parent: ShardContext, assets: Sequence[str]
    ) -> tuple[ChunkContext, ...]:
        assets = tuple(assets)
        parent_payload = _instances_payload(assets)
        try:
            actual_parent = parent.instances.read_bytes()
        except OSError as error:
            raise ChunkIdentityError(f"parent batch identity cannot be read: {error}") from error
        if actual_parent != parent_payload:
            raise ChunkIdentityError("parent batch identity does not match frozen assets")
        parts = []
        for index, start in enumerate(range(0, len(assets), self.chunk_assets)):
            chunk_id = f"chunk{index:03d}"
            payload = _instances_payload(assets[start : start + self.chunk_assets])
            parts.append((chunk_id, payload))
        expected_manifest = self._manifest(parent, parts)
        manifest_path = self.checkpoint_root / "manifest.json"
        if manifest_path.exists():
            try:
                actual_manifest = json.loads(manifest_path.read_bytes())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ChunkIdentityError(f"invalid chunk identity manifest: {error}") from error
            if actual_manifest != expected_manifest:
                raise ChunkIdentityError("chunk identity manifest does not match frozen batch")
        else:
            self.checkpoint_root.mkdir(parents=True, exist_ok=True)
            _atomic_write(
                manifest_path,
                json.dumps(expected_manifest, sort_keys=True, separators=(",", ":")).encode(),
            )
        result = []
        local_chunks = parent.work_root.parent / "chunks"
        for chunk_id, payload in parts:
            control = self.checkpoint_root / chunk_id
            instances = control / "instances.txt"
            if instances.exists():
                if instances.read_bytes() != payload:
                    raise ChunkIdentityError(f"chunk identity mismatch: {chunk_id}")
            else:
                _atomic_write(instances, payload)
            root = local_chunks / chunk_id
            result.append(
                ChunkContext(
                    parent=parent,
                    chunk_id=chunk_id,
                    instances=instances,
                    source_root=parent.source_root,
                    download_root=root / "source",
                    work_root=root / "work",
                    output_root=root / "output",
                )
            )
        return tuple(result)

    def _checkpoint_path(self, chunk: ChunkContext) -> Path:
        return self.checkpoint_root / chunk.chunk_id / "checkpoint.json"

    def _load_checkpoint(self, chunk: ChunkContext) -> ChunkCheckpoint:
        digest = sha256(chunk.instances.read_bytes()).hexdigest()
        path = self._checkpoint_path(chunk)
        if not path.exists():
            return ChunkCheckpoint(chunk.chunk_id, digest, self.config_hash)
        try:
            value = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ChunkIdentityError(f"invalid chunk checkpoint: {chunk.chunk_id}: {error}") from error
        expected_fields = {
            "schema_version",
            "chunk_id",
            "instances_sha256",
            "config_hash",
            "completed_stages",
            "worker_profiles",
            "resource_peaks",
            "promotion_started",
            "promoted",
        }
        if not isinstance(value, dict) or set(value) != expected_fields:
            raise ChunkIdentityError(f"invalid chunk checkpoint schema: {chunk.chunk_id}")
        completed = value["completed_stages"]
        known = tuple(stage.name for stage in self.stages)
        if (
            value["schema_version"] != CHUNK_CHECKPOINT_SCHEMA_VERSION
            or value["chunk_id"] != chunk.chunk_id
            or value["instances_sha256"] != digest
            or value["config_hash"] != self.config_hash
            or not isinstance(completed, list)
            or len(completed) != len(set(completed))
            or any(item not in known for item in completed)
            or [item for item in known if item in completed] != completed
            or not isinstance(value["worker_profiles"], dict)
            or not isinstance(value["resource_peaks"], dict)
            or type(value["promotion_started"]) is not bool
            or type(value["promoted"]) is not bool
            or value["promoted"] and not value["promotion_started"]
        ):
            raise ChunkIdentityError(f"chunk checkpoint identity mismatch: {chunk.chunk_id}")
        try:
            peaks = {str(name): float(peak) for name, peak in value["resource_peaks"].items()}
        except (TypeError, ValueError) as error:
            raise ChunkIdentityError(f"invalid chunk resource peaks: {chunk.chunk_id}") from error
        if any(not math.isfinite(peak) or peak < 0 for peak in peaks.values()):
            raise ChunkIdentityError(f"invalid chunk resource peaks: {chunk.chunk_id}")
        return ChunkCheckpoint(
            chunk_id=chunk.chunk_id,
            instances_sha256=digest,
            config_hash=self.config_hash,
            completed_stages=list(completed),
            worker_profiles={
                str(stage): {str(name): int(item) for name, item in profile.items()}
                for stage, profile in value["worker_profiles"].items()
            },
            resource_peaks=peaks,
            promotion_started=value["promotion_started"],
            promoted=value["promoted"],
            schema_version=value["schema_version"],
        )

    def _save_checkpoint(self, chunk: ChunkContext, checkpoint: ChunkCheckpoint) -> None:
        value = {
            "schema_version": checkpoint.schema_version,
            "chunk_id": checkpoint.chunk_id,
            "instances_sha256": checkpoint.instances_sha256,
            "config_hash": checkpoint.config_hash,
            "completed_stages": checkpoint.completed_stages,
            "worker_profiles": checkpoint.worker_profiles,
            "resource_peaks": checkpoint.resource_peaks,
            "promotion_started": checkpoint.promotion_started,
            "promoted": checkpoint.promoted,
        }
        _atomic_write(
            self._checkpoint_path(chunk),
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(),
        )

    def _validated_checkpoints(
        self, chunks: Sequence[ChunkContext]
    ) -> dict[str, ChunkCheckpoint]:
        checkpoints = {}
        stage_by_name = {stage.name: stage for stage in self.stages}
        for chunk in chunks:
            checkpoint = self._load_checkpoint(chunk)
            if checkpoint.promoted or checkpoint.promotion_started:
                if set(checkpoint.completed_stages) != set(stage_by_name):
                    raise ChunkIdentityError(
                        f"promotion began before chunk completion: {chunk.chunk_id}"
                    )
                checkpoints[chunk.chunk_id] = checkpoint
                continue
            completed_set = set(checkpoint.completed_stages)
            covered = {
                dependency
                for stage in self.stages
                if stage.name in completed_set
                for dependency in stage.dependencies
            }
            frontiers = completed_set - covered
            retained = []
            for name in checkpoint.completed_stages:
                if name not in frontiers:
                    retained.append(name)
                    continue
                stage = stage_by_name[name]
                try:
                    valid = self.executor.validate(chunk, stage) is True
                except BaseException as error:
                    raise ChunkExecutionError(
                        f"completed stage validation failed: {chunk.chunk_id}/{name}: {error}"
                    ) from error
                if valid:
                    retained.append(name)
            if retained != checkpoint.completed_stages:
                checkpoint.completed_stages = retained
                self._save_checkpoint(chunk, checkpoint)
            checkpoints[chunk.chunk_id] = checkpoint
        return checkpoints

    def run_batch(
        self, parent: ShardContext, assets: Sequence[str]
    ) -> SchedulerResult:
        chunks = self.freeze_chunks(parent, assets)
        checkpoints = self._validated_checkpoints(chunks)
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        complete_names = {stage.name for stage in self.stages}
        active: set[str] = set()
        failed: set[str] = set()
        queued_index = 0
        running: dict[Future, tuple[ChunkContext, StageSpec, object, float]] = {}
        intervals: dict[tuple[str, str], tuple[float, float]] = {}
        errors: list[tuple[str, str, BaseException]] = []
        observed_max = 0

        def terminal(chunk_id: str) -> bool:
            checkpoint = checkpoints[chunk_id]
            return checkpoint.promoted or set(checkpoint.completed_stages) == complete_names

        with ThreadPoolExecutor(max_workers=self.max_chunks_in_flight) as pool:
            while True:
                active -= {chunk_id for chunk_id in active if terminal(chunk_id) or chunk_id in failed}
                while len(active) < self.max_chunks_in_flight and queued_index < len(chunks):
                    candidate = chunks[queued_index]
                    queued_index += 1
                    if not terminal(candidate.chunk_id):
                        active.add(candidate.chunk_id)
                observed_max = max(observed_max, len(active) + sum(
                    1 for chunk in chunks[:queued_index]
                    if terminal(chunk.chunk_id) and not checkpoints[chunk.chunk_id].promoted
                ))
                observed_max = min(observed_max, self.max_chunks_in_flight)

                running_chunks = {item[0].chunk_id for item in running.values()}
                admitted = False
                for chunk_id in sorted(active):
                    if chunk_id in running_chunks:
                        continue
                    checkpoint = checkpoints[chunk_id]
                    completed = set(checkpoint.completed_stages)
                    stage = next(
                        (
                            candidate
                            for candidate in self.stages
                            if candidate.name not in completed
                            and set(candidate.dependencies) <= completed
                        ),
                        None,
                    )
                    if stage is None:
                        continue
                    lease = self.broker.try_acquire(
                        cpu_cores=stage.cpu_cores,
                        gpu_indices=stage.gpu_indices,
                        gpu_memory_percent=stage.gpu_memory_percent,
                    )
                    if lease is None:
                        continue
                    chunk = by_id[chunk_id]
                    start = self.monotonic_clock()
                    future = pool.submit(self.executor.execute, chunk, stage)
                    running[future] = (chunk, stage, lease, start)
                    admitted = True

                unfinished = any(not terminal(chunk.chunk_id) and chunk.chunk_id not in failed for chunk in chunks)
                if not running:
                    if not unfinished:
                        break
                    if not admitted:
                        pending = next(
                            chunk.chunk_id
                            for chunk in chunks
                            if not terminal(chunk.chunk_id) and chunk.chunk_id not in failed
                        )
                        raise ChunkExecutionError(
                            f"resource admission deadlock for chunk: {pending}"
                        )

                completed_futures, _ = wait(tuple(running), return_when=FIRST_COMPLETED)
                for future in completed_futures:
                    chunk, stage, lease, start = running.pop(future)
                    finish = self.monotonic_clock()
                    lease.release()
                    intervals[(chunk.chunk_id, stage.name)] = (start, finish)
                    try:
                        observations = future.result()
                        if observations is not None:
                            for name, value in observations.items():
                                peak = float(value)
                                if not math.isfinite(peak) or peak < 0:
                                    raise ValueError("stage observation must be finite and nonnegative")
                                key = f"{stage.name}.{name}"
                                checkpoints[chunk.chunk_id].resource_peaks[key] = max(
                                    peak,
                                    checkpoints[chunk.chunk_id].resource_peaks.get(key, 0.0),
                                )
                        profile_reader = getattr(
                            self.executor, "worker_profile", None
                        )
                        if callable(profile_reader):
                            profile = profile_reader(chunk, stage)
                            if not isinstance(profile, Mapping) or any(
                                not isinstance(name, str)
                                or not name
                                or type(value) is not int
                                or value < 0
                                for name, value in profile.items()
                            ):
                                raise ValueError(
                                    "worker profile must map names to nonnegative integers"
                                )
                            checkpoints[chunk.chunk_id].worker_profiles[
                                stage.name
                            ] = dict(profile)
                        checkpoints[chunk.chunk_id].complete(stage.name)
                        self._save_checkpoint(chunk, checkpoints[chunk.chunk_id])
                    except BaseException as error:
                        failed.add(chunk.chunk_id)
                        errors.append((chunk.chunk_id, stage.name, error))

        if errors:
            identities = ", ".join(f"{chunk}/{stage}: {error}" for chunk, stage, error in errors)
            raise ChunkExecutionError(f"chunk stage failure: {identities}") from errors[0][2]

        publication_order = []
        for chunk in chunks:
            checkpoint = checkpoints[chunk.chunk_id]
            if not checkpoint.promoted:
                if not checkpoint.promotion_started:
                    checkpoint.promotion_started = True
                    self._save_checkpoint(chunk, checkpoint)
                self.promoter(parent, chunk)
                checkpoint.promoted = True
                self._save_checkpoint(chunk, checkpoint)
            publication_order.append(chunk.chunk_id)
        self.publisher(parent, chunks)
        publication_order.append(parent.batch_id)
        return SchedulerResult(
            chunks=chunks,
            max_chunks_in_flight=observed_max,
            publication_order=tuple(publication_order),
            intervals=dict(intervals),
        )
