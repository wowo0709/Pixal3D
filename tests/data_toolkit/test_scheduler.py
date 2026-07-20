from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import threading
import time

import pytest

from data_toolkit.pipeline.commands import ShardContext, build_preprocessing_dag
from data_toolkit.pipeline.parallelism import (
    DynamicResourceBroker,
    NodeResourceBroker,
    WorkerSpec,
)
from data_toolkit.pipeline.scheduler import (
    ChunkExecutionError,
    Lane,
    ParallelChunkScheduler,
    StageSpec,
    choose_chunk_assets,
    promote_chunk_outputs,
)


def _assets(count: int) -> tuple[str, ...]:
    return tuple(
        sorted(
            sha256(f"asset-{index}".encode()).hexdigest()
            for index in range(count)
        )
    )


def _stages() -> tuple[StageSpec, ...]:
    return (
        StageSpec("prepare", Lane.PREPARE, cpu_cores=1),
        StageSpec(
            "render",
            Lane.RENDER,
            dependencies=("prepare",),
            gpu_indices=(0,),
            gpu_memory_percent=40,
        ),
        StageSpec(
            "geometry",
            Lane.GEOMETRY,
            dependencies=("render",),
            cpu_cores=1,
        ),
        StageSpec(
            "encode",
            Lane.ENCODE,
            dependencies=("geometry",),
            gpu_indices=(0,),
            gpu_memory_percent=80,
        ),
    )


class _TimelineExecutor:
    def __init__(self, *, fail: tuple[str, str] | None = None):
        self.fail = fail
        self.calls: list[tuple[str, str]] = []
        self.validations: list[tuple[str, str]] = []
        self.intervals: dict[tuple[str, str], tuple[float, float]] = {}
        self._lock = threading.Lock()

    def execute(self, chunk, stage):
        start = time.monotonic()
        with self._lock:
            self.calls.append((chunk.chunk_id, stage.name))
        time.sleep(0.025)
        if self.fail == (chunk.chunk_id, stage.name):
            raise RuntimeError("synthetic stage failure")
        finish = time.monotonic()
        with self._lock:
            self.intervals[(chunk.chunk_id, stage.name)] = (start, finish)
        return {"elapsed_seconds": finish - start}

    def validate(self, chunk, stage):
        with self._lock:
            self.validations.append((chunk.chunk_id, stage.name))
        return True

    def worker_profile(self, _chunk, stage):
        return {"workers": 1, "stage_index": tuple(item.name for item in _stages()).index(stage.name)}


def _overlaps(left, right):
    return left[0] < right[1] and right[0] < left[1]


def _scheduler(tmp_path: Path, executor, *, promoter=None):
    parent = ShardContext.for_test(tmp_path / "batch", "ABO", "ABO-00000")
    parent.instances.parent.mkdir(parents=True, exist_ok=True)
    parent.instances.write_text("".join(f"{asset}\n" for asset in _assets(4)))
    scheduler = ParallelChunkScheduler(
        config_hash="c" * 64,
        broker=NodeResourceBroker(cpu_limit=1, gpu_count=1),
        executor=executor,
        stages=_stages(),
        checkpoint_root=tmp_path / "checkpoints",
        chunk_assets=2,
        max_chunks_in_flight=2,
        promoter=promoter or (lambda _parent, _chunk: None),
        publisher=lambda _parent, _chunks: None,
    )
    return parent, scheduler


def test_scheduler_overlaps_independent_resource_lanes(tmp_path):
    executor = _TimelineExecutor()
    parent, scheduler = _scheduler(tmp_path, executor)

    result = scheduler.run_batch(parent, _assets(4))

    assert result.max_chunks_in_flight == 2
    assert _overlaps(
        executor.intervals[("chunk001", "render")],
        executor.intervals[("chunk000", "geometry")],
    )
    assert _overlaps(
        executor.intervals[("chunk001", "geometry")],
        executor.intervals[("chunk000", "encode")],
    )
    assert result.publication_order == ("chunk000", "chunk001", "batch000")


def test_scheduler_passes_dynamic_worker_lease_to_lease_aware_executor(tmp_path):
    class Executor(_TimelineExecutor):
        def __init__(self):
            super().__init__()
            self.nodes = []

        def execute_with_lease(self, chunk, stage, lease):
            self.nodes.append((chunk.chunk_id, stage.name, lease.node_id))
            return self.execute(chunk, stage)

    parent = ShardContext.for_test(tmp_path / "batch", "ABO", "ABO-00000")
    parent.instances.parent.mkdir(parents=True, exist_ok=True)
    parent.instances.write_text("".join(f"{asset}\n" for asset in _assets(2)))
    broker = DynamicResourceBroker()
    broker.register(WorkerSpec("node16", cpu_limit=2, gpu_indices=(2,)))
    executor = Executor()
    scheduler = ParallelChunkScheduler(
        config_hash="c" * 64,
        broker=broker,
        executor=executor,
        stages=_stages(),
        checkpoint_root=tmp_path / "checkpoints",
        chunk_assets=2,
        max_chunks_in_flight=1,
        promoter=lambda _parent, _chunk: None,
        publisher=lambda _parent, _chunks: None,
    )

    scheduler.run_batch(parent, _assets(2))

    assert {node for _, _, node in executor.nodes} == {"node16"}


def test_scheduler_preserves_same_chunk_dependencies(tmp_path):
    executor = _TimelineExecutor()
    parent, scheduler = _scheduler(tmp_path, executor)

    scheduler.run_batch(parent, _assets(4))

    by_chunk = defaultdict(list)
    for chunk_id, stage in executor.calls:
        by_chunk[chunk_id].append(stage)
    assert by_chunk == {
        "chunk000": ["prepare", "render", "geometry", "encode"],
        "chunk001": ["prepare", "render", "geometry", "encode"],
    }
    checkpoint = json.loads(
        (tmp_path / "checkpoints/chunk000/checkpoint.json").read_text()
    )
    assert checkpoint["worker_profiles"]["geometry"] == {
        "stage_index": 2,
        "workers": 1,
    }


def test_restart_validates_and_skips_completed_stages(tmp_path):
    failing = _TimelineExecutor(fail=("chunk000", "render"))
    parent, scheduler = _scheduler(tmp_path, failing)
    with pytest.raises(ChunkExecutionError, match="chunk000/render"):
        scheduler.run_batch(parent, _assets(4))

    resumed = _TimelineExecutor()
    parent, scheduler = _scheduler(tmp_path, resumed)
    scheduler.run_batch(parent, _assets(4))

    assert ("chunk000", "prepare") in resumed.validations
    assert ("chunk000", "prepare") not in resumed.calls
    assert ("chunk000", "render") in resumed.calls


def test_checkpoint_is_bound_to_immutable_chunk_instances(tmp_path):
    executor = _TimelineExecutor(fail=("chunk000", "render"))
    parent, scheduler = _scheduler(tmp_path, executor)
    with pytest.raises(ChunkExecutionError):
        scheduler.run_batch(parent, _assets(4))


def test_restart_resumes_idempotent_promotion_without_rerunning_leaf_stages(
    tmp_path,
):
    first_executor = _TimelineExecutor()
    promoted = []

    def interrupted_promotion(_parent, _chunk):
        promoted.append(_chunk.chunk_id)
        if len(promoted) == 1:
            return
        raise RuntimeError("synthetic promotion interruption")

    parent, scheduler = _scheduler(
        tmp_path, first_executor, promoter=interrupted_promotion
    )
    with pytest.raises(RuntimeError, match="promotion interruption"):
        scheduler.run_batch(parent, _assets(4))

    resumed = _TimelineExecutor()
    resumed.validate = lambda _chunk, _stage: False
    parent, scheduler = _scheduler(tmp_path, resumed)
    scheduler.run_batch(parent, _assets(4))

    assert resumed.calls == []
    instances = tmp_path / "checkpoints/chunk000/instances.txt"
    instances.write_text(f"{'f' * 64}\n")

    parent, scheduler = _scheduler(tmp_path, _TimelineExecutor())
    with pytest.raises(ChunkExecutionError, match="identity"):
        scheduler.run_batch(parent, _assets(4))


def test_choose_chunk_assets_uses_25_percent_scratch_headroom():
    gib = 1024**3
    assert choose_chunk_assets(
        configured=64, p95_scratch_bytes=gib, usable_bytes=80 * gib
    ) == 64
    assert choose_chunk_assets(
        configured=64, p95_scratch_bytes=gib, usable_bytes=79 * gib
    ) == 32


def test_single_chunk_reports_one_chunk_in_flight(tmp_path):
    parent = ShardContext.for_test(tmp_path / "parent", "ABO", "ABO-00000")
    assets = _assets(1)
    parent.instances.parent.mkdir(parents=True, exist_ok=True)
    parent.instances.write_text(f"{assets[0]}\n")
    scheduler = ParallelChunkScheduler(
        config_hash="c" * 64,
        broker=NodeResourceBroker(cpu_limit=1, gpu_count=1),
        executor=_TimelineExecutor(),
        stages=_stages(),
        checkpoint_root=tmp_path / "checkpoints",
        chunk_assets=1,
        max_chunks_in_flight=3,
        promoter=lambda _parent, _chunk: None,
        publisher=lambda _parent, _chunks: None,
    )

    assert scheduler.run_batch(parent, assets).max_chunks_in_flight == 1


def test_promote_chunk_outputs_renames_asset_directories_and_is_idempotent(tmp_path):
    parent = ShardContext.for_test(tmp_path / "parent", "ABO", "ABO-00000")
    assets = _assets(1)
    parent.instances.parent.mkdir(parents=True, exist_ok=True)
    parent.instances.write_text(f"{assets[0]}\n")
    executor = _TimelineExecutor()
    scheduler = ParallelChunkScheduler(
        config_hash="c" * 64,
        broker=NodeResourceBroker(cpu_limit=1, gpu_count=1),
        executor=executor,
        stages=_stages(),
        checkpoint_root=tmp_path / "checkpoints",
        chunk_assets=1,
        max_chunks_in_flight=1,
        promoter=promote_chunk_outputs,
        publisher=lambda _parent, _chunks: None,
    )
    chunks = scheduler.freeze_chunks(parent, assets)
    source = chunks[0].output_root / "renders_cond" / assets[0]
    source.mkdir(parents=True)
    (source / "000.png").write_bytes(b"image")

    promote_chunk_outputs(parent, chunks[0])
    destination = parent.output_root / "renders_cond" / assets[0] / "000.png"
    assert destination.read_bytes() == b"image"

    source.mkdir(parents=True)
    (source / "000.png").write_bytes(b"image")
    promote_chunk_outputs(parent, chunks[0])
    assert not source.exists()


def test_chunk_context_scopes_all_shared_record_parts(tmp_path, config):
    parent = ShardContext.for_test(tmp_path / "parent", "ABO", "ABO-00000")
    assets = _assets(1)
    parent.instances.parent.mkdir(parents=True, exist_ok=True)
    parent.instances.write_text(f"{assets[0]}\n")
    scheduler = ParallelChunkScheduler(
        config_hash=config.config_hash(),
        broker=NodeResourceBroker(cpu_limit=1, gpu_count=1),
        executor=_TimelineExecutor(),
        stages=_stages(),
        checkpoint_root=tmp_path / "checkpoints",
        chunk_assets=1,
        max_chunks_in_flight=1,
        promoter=lambda _parent, _chunk: None,
        publisher=lambda _parent, _chunks: None,
    )
    context = scheduler.freeze_chunks(parent, assets)[0].as_shard_context()
    commands = {command.name: command for command in build_preprocessing_dag(context, config)}

    assert context.record_prefix == "chunk000_"
    for name in ("asset_stats", "render_cond", "dual_grid_256", "voxelize_pbr_256"):
        argv = commands[name].argv
        assert argv[argv.index("--record_prefix") + 1] == "chunk000_"
