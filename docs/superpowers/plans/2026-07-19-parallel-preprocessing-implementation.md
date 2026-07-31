# Parallel Preprocessing Saturation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Saturate seven 97.9 GiB GPUs and 44 physical CPU cores with batched encoders, concurrent Blender rendering, and overlapped chunk lanes while preserving every Pixal3D preprocessing output contract.

**Architecture:** Keep frozen publication batches at 256 assets, but execute them as isolated 32- or 64-asset chunks. A node-level broker admits Blender, CPU geometry, encoder, and publication work under fixed resource budgets; each chunk advances independently through the DAG, and publication remains serialized after all chunks are promoted into the canonical batch root.

**Tech Stack:** Python 3.11, PyTorch >=2.8, CUDA 12.8, Pixal3D sparse tensors, Blender 4.5.1 LTS with OptiX, psutil, pytest, pandas, YAML.

## Global Constraints

- Implement in a new git worktree; do not modify files used by the currently running qualification process.
- GPU memory target is 80%; no new GPU work is admitted at or above 90%.
- Use at most 44 physical CPU cores; reserve four physical cores for the OS, monitoring, and publication.
- Render exactly eight 512 x 512 condition images per asset with the existing camera policy.
- Preserve aligned views 0 and 1, resolutions 256/512/1024, SS-64, and FP32 latents.
- Preserve output paths, schemas, eight pack families, quarantine semantics, raw archives, and atomic publication.
- Provider-unavailable assets are quarantined; schema failures stop the affected source.
- A failed rollout returns to the last audited profile without changing frozen SHA order.
- Acceptance requires at least 215.03 assets/hour, 1.8 times the measured ABO baseline of 119.46 assets/hour.

---

## File Structure

- `data_toolkit/pipeline/config.py`: strict parallel resource configuration.
- `data_toolkit/pipeline/resources.py`: total GPU memory sampling and admission.
- `data_toolkit/pipeline/parallelism.py`: pure profile selection and node resource leases.
- `data_toolkit/pipeline/sparse_batching.py`: sparse batch/split and bounded encoder execution.
- `data_toolkit/pipeline/commands.py`: round-robin GPU placement and leaf arguments.
- `data_toolkit/pipeline/scheduler.py`: isolated chunk DAG, checkpoints, and promotion.
- `data_toolkit/pipeline/orchestrator.py`: production integration and serialized publication.
- `data_toolkit/encode_{shape,pbr,ss}_latent_view.py`: adaptive encoder micro-batches.
- `data_toolkit/{render_cond,dual_grid_view,voxelize_pbr_view}.py`: concurrent worker-safe records.
- `tests/data_toolkit/test_{parallelism,sparse_batching,scheduler}.py`: new unit tests.
- Existing config, command, resource, orchestrator, reporting, CLI, and integration tests.
- `docs/data_preprocessing_runbook_ko.md`: benchmark, rollout, monitoring, and rollback.

### Task 1: Add strict resource profiles and GPU memory telemetry

**Files:**
- Modify: `data_toolkit/pipeline/config.py`
- Modify: `data_toolkit/configs/multiview_preprocess.yaml`
- Modify: `data_toolkit/pipeline/resources.py`
- Create: `data_toolkit/pipeline/parallelism.py`
- Test: `tests/data_toolkit/test_config.py`
- Test: `tests/data_toolkit/test_resources.py`
- Create: `tests/data_toolkit/test_parallelism.py`

**Interfaces:**
- Produces: `ParallelismConfig`, `GpuMemoryState`, `select_micro_batch(...)`, and `NodeResourceBroker`.
- Consumes: existing `ResourceSnapshot`, `GpuMetric`, and strict YAML helpers.

- [ ] **Step 1: Write failing strict-schema and selection tests**

```python
def test_checked_in_parallelism_profile(config):
    value = config.parallelism
    assert value.gpu_memory_target_percent == 80
    assert value.gpu_memory_hard_percent == 90
    assert value.cpu_physical_cores == 44
    assert value.chunk_assets == 64
    assert value.max_chunks_in_flight == 3
    assert value.render_workers_per_gpu_steps == (2, 3, 4)
    assert value.encoder_micro_batches == ((256, 16), (512, 8), (1024, 4), (64, 16))


def test_micro_batch_steps_down_above_target(config):
    assert select_micro_batch(resolution=1024, configured=4,
        peak_percent=84.0, oom=False, config=config.parallelism) == 2


def test_micro_batch_halves_after_oom(config):
    assert select_micro_batch(resolution=512, configured=8,
        peak_percent=50.0, oom=True, config=config.parallelism) == 4
```

- [ ] **Step 2: Run tests and verify failure**

```bash
PYTHONPATH=. conda run -n pixal3d pytest \
  tests/data_toolkit/test_config.py tests/data_toolkit/test_resources.py \
  tests/data_toolkit/test_parallelism.py -k 'parallelism or gpu_memory or micro_batch' -v
```

Expected: import or attribute failures because the new schema and module do not exist.

- [ ] **Step 3: Add the complete strict configuration contract**

```python
@dataclass(frozen=True)
class ParallelismConfig:
    gpu_count: int
    gpu_memory_target_percent: int
    gpu_memory_hard_percent: int
    cpu_physical_cores: int
    chunk_assets: int
    max_chunks_in_flight: int
    render_workers_per_gpu_steps: tuple[int, ...]
    encoder_micro_batches: tuple[tuple[int, int], ...]

    def micro_batch(self, resolution: int) -> int:
        try:
            return dict(self.encoder_micro_batches)[resolution]
        except KeyError as error:
            raise ValueError(f"missing encoder micro-batch for {resolution}") from error
```

```yaml
parallelism:
  gpu_count: 7
  gpu_memory_target_percent: 80
  gpu_memory_hard_percent: 90
  cpu_physical_cores: 44
  chunk_assets: 64
  max_chunks_in_flight: 3
  render_workers_per_gpu_steps: [2, 3, 4]
  encoder_micro_batches: [[256, 16], [512, 8], [1024, 4], [64, 16]]
```

Reject booleans, duplicate resolutions, unordered steps, CPU cores above 44,
target greater than or equal to hard, and hard above 95.

- [ ] **Step 4: Sample total GPU memory and implement pure selection**

Extend `GpuMetric` with `memory_total_mib: float` and query
`memory.used,memory.total`. Add:

```python
@dataclass(frozen=True)
class GpuMemoryState:
    index: int
    used_mib: float
    total_mib: float

    @property
    def percent(self) -> float:
        return self.used_mib * 100.0 / self.total_mib


def select_micro_batch(*, resolution, configured, peak_percent, oom, config):
    if oom or peak_percent > config.gpu_memory_target_percent:
        return max(1, configured // 2)
    if peak_percent < 70.0:
        return min(config.micro_batch(resolution), configured * 2)
    return configured
```

`NodeResourceBroker.try_acquire(cpu_cores, gpu_indices,
gpu_memory_percent)` must be lock-protected and reject requests above 44 cores
or at 90% GPU memory. Its lease is an idempotent context manager.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONPATH=. conda run -n pixal3d pytest \
  tests/data_toolkit/test_config.py tests/data_toolkit/test_resources.py \
  tests/data_toolkit/test_parallelism.py -v
git add data_toolkit/pipeline/config.py data_toolkit/configs/multiview_preprocess.yaml \
  data_toolkit/pipeline/resources.py data_toolkit/pipeline/parallelism.py \
  tests/data_toolkit/test_config.py tests/data_toolkit/test_resources.py \
  tests/data_toolkit/test_parallelism.py
git commit -m "feat: configure bounded preprocessing saturation"
```

### Task 2: Map multiple Blender workers safely onto each GPU

**Files:**
- Modify: `data_toolkit/pipeline/commands.py`
- Modify: `data_toolkit/pipeline/orchestrator.py`
- Modify: `data_toolkit/render_cond.py`
- Test: `tests/data_toolkit/test_commands.py`
- Test: `tests/data_toolkit/test_orchestrator.py`

**Interfaces:**
- Consumes: `ParallelismConfig.render_workers_per_gpu_steps` and `CommandSpec`.
- Produces: `CommandSpec.workers_per_gpu` and round-robin `expand_ranked(...)`.

- [ ] **Step 1: Write failing GPU mapping tests**

```python
def test_render_workers_map_round_robin_to_seven_gpus(config, tmp_path):
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")
    command = next(c for c in build_preprocessing_dag(context, config)
                   if c.name == "render_cond")
    expanded = expand_ranked(command)
    assert len(expanded) == 14
    assert [dict(env)["CUDA_VISIBLE_DEVICES"] for _, env in expanded] == [
        "0", "1", "2", "3", "4", "5", "6",
        "0", "1", "2", "3", "4", "5", "6",
    ]
    assert [argv[argv.index("--rank") + 1] for argv, _ in expanded] == [
        str(index) for index in range(14)
    ]
    assert all(argv[argv.index("--world_size") + 1] == "14"
               for argv, _ in expanded)
```

Add a supervisor test proving one failed worker terminates and reaps all 14
process groups.

- [ ] **Step 2: Run tests and verify failure**

```bash
PYTHONPATH=. conda run -n pixal3d pytest \
  tests/data_toolkit/test_commands.py tests/data_toolkit/test_orchestrator.py \
  -k 'round_robin or render_workers_per_gpu' -v
```

Expected: only seven processes are produced.

- [ ] **Step 3: Extend `CommandSpec` and `expand_ranked`**

```python
@dataclass(frozen=True)
class CommandSpec:
    name: str
    argv: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()
    gpu_ranks: int = 0
    workers_per_gpu: int = 1


def expand_ranked(command):
    if not command.gpu_ranks:
        return ((command.argv, command.env),)
    total = command.gpu_ranks * command.workers_per_gpu
    return tuple(
        ((*command.argv, "--rank", str(rank), "--world_size", str(total)),
         (*command.env,
          ("CUDA_VISIBLE_DEVICES", str(rank % command.gpu_ranks))))
        for rank in range(total)
    )
```

Validate positive counts and cap total child processes at 28.

- [ ] **Step 4: Configure two Blender workers per GPU and unique records**

Set `render_cond.workers_per_gpu=2`. Add `--record_prefix` and write records as
`{prefix}part_{rank}.csv`; reject prefixes containing path separators. An empty
prefix retains legacy paths.

Add and test the boundary selector:

```python
def select_render_workers(*, current, peak_percent, temperature_celsius,
                          failed, steps=(2, 3, 4)):
    index = steps.index(current)
    if failed or peak_percent > 80 or temperature_celsius >= 80:
        return steps[max(0, index - 1)]
    if peak_percent < 70 and temperature_celsius < 75:
        return steps[min(len(steps) - 1, index + 1)]
    return current
```

Apply a changed value only at the next render command boundary.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONPATH=. conda run -n pixal3d pytest \
  tests/data_toolkit/test_commands.py tests/data_toolkit/test_orchestrator.py -v
git add data_toolkit/pipeline/commands.py data_toolkit/pipeline/orchestrator.py \
  data_toolkit/render_cond.py tests/data_toolkit/test_commands.py \
  tests/data_toolkit/test_orchestrator.py
git commit -m "perf: run multiple blender workers per gpu"
```

### Task 3: Add backend-native sparse batching and splitting

**Files:**
- Create: `data_toolkit/pipeline/sparse_batching.py`
- Create: `tests/data_toolkit/test_sparse_batching.py`

**Interfaces:**
- Produces: `micro_batches(...)`, `batch_sparse_tensors(...)`, and `split_sparse_tensor(...)`.
- Consumes: `SparseTensor.from_tensor_list(...)` and `.layout`.

- [ ] **Step 1: Write failing batch/split identity tests**

```python
def test_sparse_batch_round_trip_preserves_order_and_values(sparse_factory):
    first = sparse_factory([[1.0], [2.0]],
        [[0, 1, 2, 3], [0, 4, 5, 6]])
    second = sparse_factory([[3.0]], [[0, 7, 8, 9]])
    combined = batch_sparse_tensors([first, second])
    restored = split_sparse_tensor(combined)
    assert len(combined) == 2
    assert torch.equal(restored[0].feats, first.feats)
    assert torch.equal(restored[0].coords[:, 1:], first.coords[:, 1:])
    assert torch.equal(restored[1].feats, second.feats)
    assert torch.equal(restored[1].coords[:, 1:], second.coords[:, 1:])


def test_micro_batches_are_bounded_and_ordered():
    assert list(micro_batches(list(range(10)), 4)) == [
        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9]
    ]
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=. conda run -n pixal3d pytest tests/data_toolkit/test_sparse_batching.py -v
```

Expected: import failure for `sparse_batching`.

- [ ] **Step 3: Implement the complete pure helpers**

```python
def micro_batches(tasks, size):
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("micro-batch size must be a positive integer")
    values = list(tasks)
    return [values[index:index + size]
            for index in range(0, len(values), size)]


def batch_sparse_tensors(tensors):
    values = list(tensors)
    if not values:
        raise ValueError("cannot batch zero sparse tensors")
    return type(values[0]).from_tensor_list(
        [value.feats for value in values],
        [value.coords for value in values],
    )


def split_sparse_tensor(tensor):
    outputs = []
    for layout in tensor.layout:
        coords = tensor.coords[layout].clone()
        coords[:, 0] = 0
        outputs.append(type(tensor)(tensor.feats[layout], coords))
    return outputs
```

Validate common tensor type, finite features, integral coordinates, contiguous
layout, and output count.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=. conda run -n pixal3d pytest tests/data_toolkit/test_sparse_batching.py -v
git add data_toolkit/pipeline/sparse_batching.py \
  tests/data_toolkit/test_sparse_batching.py
git commit -m "feat: batch and split sparse encoder tasks"
```

### Task 4: Execute view encoders in adaptive micro-batches

**Files:**
- Modify: `data_toolkit/pipeline/sparse_batching.py`
- Modify: `data_toolkit/encode_shape_latent_view.py`
- Modify: `data_toolkit/encode_pbr_latent_view.py`
- Modify: `data_toolkit/encode_ss_latent_view.py`
- Modify: `data_toolkit/pipeline/commands.py`
- Create: `tests/data_toolkit/test_encoder_micro_batch.py`
- Test: `tests/data_toolkit/test_commands.py`

**Interfaces:**
- Consumes: sparse helpers and `ParallelismConfig.micro_batch(resolution)`.
- Produces: `run_encoder_tasks(...)`, `--micro_batch_size`, batched model calls, and peak-memory records.

- [ ] **Step 1: Write failing batching, OOM fallback, and parity tests**

```python
def test_eight_tasks_at_four_use_two_model_calls(fake_encoder):
    records = run_encoder_tasks(tasks=list(range(8)), micro_batch_size=4,
        load=fake_encoder.load, process_batch=fake_encoder.process,
        save=fake_encoder.save)
    assert fake_encoder.call_sizes == [4, 4]
    assert [record.task for record in records] == list(range(8))


def test_oom_halves_pending_batch_without_losing_tasks(fake_encoder):
    fake_encoder.fail_once_at_size(4)
    records = run_encoder_tasks(tasks=list(range(6)), micro_batch_size=4,
        load=fake_encoder.load, process_batch=fake_encoder.process,
        save=fake_encoder.save)
    assert fake_encoder.call_sizes == [4, 2, 2, 2]
    assert [record.task for record in records] == list(range(6))
```

Add a backend test comparing batch-one and batch-four coordinates exactly and
features with `torch.testing.assert_close(rtol=1e-6, atol=1e-6)`.

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=. conda run -n pixal3d pytest \
  tests/data_toolkit/test_encoder_micro_batch.py tests/data_toolkit/test_commands.py \
  -k 'micro_batch or encoder' -v
```

- [ ] **Step 3: Implement `run_encoder_tasks(...)`**

Keep bounded loader and saver queues. Collect up to the current micro-batch,
batch inputs, invoke the model once, synchronize, split outputs, and enqueue
per-task atomic saves in frozen order. Record
`torch.cuda.max_memory_reserved()`. On CUDA OOM, empty the cache, halve only
the pending micro-batch, and retry; re-raise every non-OOM error.

The public signature is:

```python
def run_encoder_tasks(*, tasks, micro_batch_size, load, process_batch, save,
                      loader_workers=2, saver_workers=1,
                      timeout_seconds=300):
    """Return save records in input task order or raise without silent loss."""
```

- [ ] **Step 4: Integrate all three encoders**

Add these parser arguments to each view encoder:

```python
parser.add_argument("--micro_batch_size", type=int, required=True)
parser.add_argument("--gpu_memory_target_percent", type=float, default=80.0)
parser.add_argument("--record_prefix", default="")
```

Shape batches `vertices` and `intersected` in identical order. PBR batches
`voxels`. SS batches its shape-latent sparse tensors. Split model outputs
before the existing atomic save and validation functions.

- [ ] **Step 5: Materialize resolution-specific command arguments**

Pass micro-batch 16 for 256, 8 for 512, 4 for 1024, and 16 for SS-64. Encoder
commands retain seven ranks and one model process per GPU.

- [ ] **Step 6: Run CPU and one-GPU parity tests**

```bash
PYTHONPATH=. conda run -n pixal3d pytest \
  tests/data_toolkit/test_encoder_micro_batch.py \
  tests/data_toolkit/test_sparse_batching.py tests/data_toolkit/test_commands.py -v
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. conda run -n pixal3d pytest \
  tests/data_toolkit/test_encoder_micro_batch.py -m gpu -v
```

Expected: batch-one and configured batches pass coordinate and FP32 parity.
If the backend fails isolation or numerical parity, set micro-batch to one and
use `workers_per_gpu=2`, then 3 and 4, through the round-robin launcher from
Task 2. This fallback must pass the identical output tests before benchmarking.

- [ ] **Step 7: Commit Task 4**

```bash
git add data_toolkit/pipeline/sparse_batching.py \
  data_toolkit/encode_shape_latent_view.py \
  data_toolkit/encode_pbr_latent_view.py \
  data_toolkit/encode_ss_latent_view.py data_toolkit/pipeline/commands.py \
  tests/data_toolkit/test_encoder_micro_batch.py tests/data_toolkit/test_commands.py
git commit -m "perf: encode sparse views in adaptive gpu batches"
```

### Task 5: Use a bounded 44-core geometry pool

**Files:**
- Modify: `data_toolkit/pipeline/parallelism.py`
- Modify: `data_toolkit/pipeline/commands.py`
- Modify: `data_toolkit/dual_grid_view.py`
- Modify: `data_toolkit/voxelize_pbr_view.py`
- Test: `tests/data_toolkit/test_parallelism.py`
- Test: `tests/data_toolkit/test_commands.py`

**Interfaces:**
- Consumes: `NodeResourceBroker` and `ParallelismConfig.cpu_physical_cores`.
- Produces: `GeometryProfile(processes=11, native_threads=4)` and CPU leases.

- [ ] **Step 1: Write failing CPU-budget tests**

```python
def test_geometry_profile_uses_44_physical_cores(config):
    profile = geometry_profile(config.parallelism)
    assert profile.processes == 11
    assert profile.native_threads == 4
    assert profile.processes * profile.native_threads == 44


def test_broker_never_oversubscribes_cpu():
    broker = NodeResourceBroker(cpu_limit=44, gpu_count=7)
    first = broker.try_acquire(cpu_cores=24, gpu_indices=())
    second = broker.try_acquire(cpu_cores=20, gpu_indices=())
    assert first is not None and second is not None
    assert broker.try_acquire(cpu_cores=1, gpu_indices=()) is None
    second.release()
    assert broker.try_acquire(cpu_cores=1, gpu_indices=()) is not None
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=. conda run -n pixal3d pytest \
  tests/data_toolkit/test_parallelism.py tests/data_toolkit/test_commands.py \
  -k 'geometry or cpu_budget' -v
```

- [ ] **Step 3: Apply the 11 x 4 profile without nested oversubscription**

Set geometry commands to `--max_workers 11 --native_threads 4` when they own
the full geometry lane. Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and
`MKL_NUM_THREADS=1` outside the explicit native worker. Add unique
`--record_prefix` output names for chunk execution.

Pin five four-core workers to NUMA node 0 cores `0-19` and six four-core
workers to NUMA node 1 cores `24-47`. Cores `20-23` remain reserved. Add an
affinity test that asserts the 11 sets are disjoint, contain four physical
cores each, and their union contains exactly 44 cores.

- [ ] **Step 4: Implement pressure step-down**

Reduce 11 x 4 to 10 x 4 and then 8 x 4 when snapshots report CPU >=80%, I/O
wait >=10%, available RAM <96 GiB, or a soft thermal reason. Step up only after
three stable command boundaries.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONPATH=. conda run -n pixal3d pytest \
  tests/data_toolkit/test_parallelism.py tests/data_toolkit/test_commands.py \
  tests/data_toolkit/test_resources.py -v
git add data_toolkit/pipeline/parallelism.py data_toolkit/pipeline/commands.py \
  data_toolkit/dual_grid_view.py data_toolkit/voxelize_pbr_view.py \
  tests/data_toolkit/test_parallelism.py tests/data_toolkit/test_commands.py
git commit -m "perf: saturate bounded cpu geometry workers"
```

### Task 6: Pipeline isolated chunks across resource lanes

**Files:**
- Create: `data_toolkit/pipeline/scheduler.py`
- Modify: `data_toolkit/pipeline/orchestrator.py`
- Modify: `data_toolkit/pipeline/runtime.py`
- Create: `tests/data_toolkit/test_scheduler.py`
- Test: `tests/data_toolkit/test_orchestrator.py`
- Test: `tests/data_toolkit/test_integration_run.py`

**Interfaces:**
- Consumes: `NodeResourceBroker`, `CommandExecutor`, frozen batches, and validators.
- Produces: `Lane`, `ChunkContext`, `ChunkCheckpoint`, `ParallelChunkScheduler.run_batch(...)`, and `promote_chunk_outputs(...)`.

- [ ] **Step 1: Write failing dependency, overlap, and restart tests**

```python
def test_scheduler_overlaps_independent_lanes(fake_clock, scheduler):
    result = scheduler.run_batch(batch_of(128, chunk_assets=64))
    assert result.max_chunks_in_flight == 2
    assert overlaps(result.interval("chunk001", "render"),
                    result.interval("chunk000", "geometry"))
    assert overlaps(result.interval("chunk001", "geometry"),
                    result.interval("chunk000", "encode"))
    assert result.publication_order == ["chunk000", "chunk001", "batch000"]


def test_restart_validates_and_skips_completed_chunk(scheduler, checkpoints):
    checkpoints.complete("chunk000", "encode")
    scheduler.run_batch(batch_of(128, chunk_assets=64))
    assert not scheduler.executor.was_called("chunk000", "encode")
    assert scheduler.validator.was_called("chunk000", "encode")
```

Also prove no geometry starts before render for the same chunk, no encode starts
before its resolution's geometry, and publication waits for every chunk.

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=. conda run -n pixal3d pytest tests/data_toolkit/test_scheduler.py -v
```

Expected: scheduler module does not exist.

- [ ] **Step 3: Implement immutable chunk identity and checkpoint schema**

```python
class Lane(str, Enum):
    PREPARE = "prepare"
    RENDER = "render"
    GEOMETRY = "geometry"
    ENCODE = "encode"
    PUBLISH = "publish"


@dataclass(frozen=True)
class ChunkContext:
    parent: ShardContext
    chunk_id: str
    instances: Path
    source_root: Path
    work_root: Path
    output_root: Path


@dataclass
class ChunkCheckpoint:
    schema_version: int
    chunk_id: str
    instances_sha256: str
    config_hash: str
    completed_stages: list[str]
    worker_profiles: dict[str, dict[str, int]]
    resource_peaks: dict[str, float]
```

Split frozen SHA files deterministically into `chunk000.txt`, `chunk001.txt`,
and so on. Bind checkpoints to chunk-file SHA-256 and config hash.

Choose 64 assets when source p95 scratch bytes times 64 plus 25% headroom fits
the current local admission budget; otherwise choose 32. Freeze the chosen
chunk size in the parent checkpoint so resume cannot silently repartition it.

- [ ] **Step 4: Implement bounded lane scheduling**

Use one scheduling thread, a bounded ready queue, and futures only for admitted
leaf stages. Acquire a broker lease before submitting. Allow at most three
chunks in flight. Store exceptions with chunk and stage identity, stop
downstream admission for that chunk, and let independent running tasks finish
unless the resource guard signals a hard stop.

- [ ] **Step 5: Promote chunk outputs without payload copies**

Keep chunk roots on `/root/node17/data/pixal3d`. After validation, atomically
rename each asset directory into the canonical batch output root. Reject
duplicate destinations unless the existing output validates with the same
checksum. Merge `{chunk_id}_part_{rank}.csv` records in frozen SHA order. Only
the parent batch calls `build_packs`, `archive_raw`, and `cleanup_local`.

- [ ] **Step 6: Integrate production behind strict configuration**

Use `ParallelChunkScheduler` only for production batches larger than 64.
Smoke and existing pilot execution remain sequential reference paths. Add
parallel checkpoints under `control/checkpoints/.../chunks/` without removing
legacy checkpoint fields.

- [ ] **Step 7: Run scheduler and integration tests**

```bash
PYTHONPATH=. conda run -n pixal3d pytest \
  tests/data_toolkit/test_scheduler.py tests/data_toolkit/test_orchestrator.py \
  tests/data_toolkit/test_integration_run.py -k 'parallel or chunk or restart' -v
```

- [ ] **Step 8: Commit Task 6**

```bash
git add data_toolkit/pipeline/scheduler.py data_toolkit/pipeline/orchestrator.py \
  data_toolkit/pipeline/runtime.py tests/data_toolkit/test_scheduler.py \
  tests/data_toolkit/test_orchestrator.py tests/data_toolkit/test_integration_run.py
git commit -m "feat: pipeline isolated preprocessing chunks"
```

### Task 7: Gate rollout on performance and audit evidence

**Files:**
- Modify: `data_toolkit/pipeline/reporting.py`
- Modify: `data_toolkit/pipeline/runtime.py`
- Modify: `data_toolkit/pipeline/cli.py`
- Modify: `docs/data_preprocessing_runbook_ko.md`
- Test: `tests/data_toolkit/test_reporting.py`
- Test: `tests/data_toolkit/test_cli.py`
- Test: `tests/data_toolkit/test_integration_run.py`

**Interfaces:**
- Consumes: chunk checkpoints and stage resource peaks.
- Produces: `parallelism_summary(...)`, `benchmark-parallelism`, and held rollout evidence.

- [ ] **Step 1: Write failing report and CLI tests**

```python
def test_parallelism_summary_enforces_acceptance_target():
    value = parallelism_summary(completed_assets=64, elapsed_seconds=900.0,
        baseline_assets_per_hour=119.46, gpu_peak_percent=79.0,
        cpu_assigned_cores=44, audit_passed=True)
    assert value["assets_per_hour"] == 256.0
    assert value["speedup"] > 1.8
    assert value["passed"] is True
```

CLI tests require explicit source, frozen shard, output evidence path, and
`--dry-run` support.

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=. conda run -n pixal3d pytest \
  tests/data_toolkit/test_reporting.py tests/data_toolkit/test_cli.py \
  -k 'parallelism or benchmark' -v
```

- [ ] **Step 3: Implement held benchmark reporting**

Report per-stage seconds, assets/hour, speedup, GPU peak/mean memory and
utilization, CPU assigned/observed utilization, temperatures, pauses, retries,
quarantines, and audit status. Pass only at >=215.03 assets/hour, successful
audit, GPU peak <=90%, steady-state target <=80%, and CPU assignment <=44.

- [ ] **Step 4: Add exact operator commands**

```bash
PYTHONPATH=. conda run -n pixal3d python -m data_toolkit.pipeline.cli \
  benchmark-parallelism --config data_toolkit/configs/multiview_preprocess.yaml \
  --source ABO --shard ABO-00000 --count 64

PYTHONPATH=. conda run -n pixal3d python -m data_toolkit.pipeline.cli \
  report --config data_toolkit/configs/multiview_preprocess.yaml \
  --parallelism-check

PYTHONPATH=. conda run -n pixal3d python -m data_toolkit.pipeline.cli \
  full-run --config data_toolkit/configs/multiview_preprocess.yaml --dry-run
```

Document recovery-only rollback; never delete checkpoints or published data.

- [ ] **Step 5: Run all tests and a held GPU benchmark**

```bash
PYTHONPATH=. conda run -n pixal3d pytest tests/data_toolkit -q
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. conda run -n pixal3d pytest \
  tests/data_toolkit/test_encoder_micro_batch.py -m gpu -v
```

Then run the documented 64-asset ABO benchmark. Compare eight output families
and raw archives with the sequential reference. Do not enable production
unless the report decision is `passed`.

- [ ] **Step 6: Commit Task 7**

```bash
git add data_toolkit/pipeline/reporting.py data_toolkit/pipeline/runtime.py \
  data_toolkit/pipeline/cli.py docs/data_preprocessing_runbook_ko.md \
  tests/data_toolkit/test_reporting.py tests/data_toolkit/test_cli.py \
  tests/data_toolkit/test_integration_run.py
git commit -m "feat: gate parallel rollout on performance evidence"
```

### Task 8: Activate production source by source

**Files:**
- Modify only if held evidence exposes a defect.
- Evidence: `/root/data2/pixal3d/control/reports/parallelism.json`
- Evidence: `/root/data2/pixal3d/control/reports/parallelism.md`

**Interfaces:**
- Consumes: passed parallelism report and established source order.
- Produces: audited production launch state.

- [ ] **Step 1: Verify the implementation worktree has no active qualification process**

```bash
ps -eo pid,stat,cmd | rg 'data_toolkit.pipeline.cli (run|resume|full-run)'
```

Expected: no process points at the implementation worktree before its code is
used for production.

- [ ] **Step 2: Inspect the production dry run**

```bash
PYTHONPATH=. conda run -n pixal3d python -m data_toolkit.pipeline.cli \
  full-run --config data_toolkit/configs/multiview_preprocess.yaml --dry-run
```

Expected: ABO, HSSD, 3D-FUTURE, ObjaverseXL_sketchfab,
ObjaverseXL_github; publication batches <=256 and chunks <=64.

- [ ] **Step 3: Run one 256-asset ABO batch and hold**

Audit packs, archives, quality ledger, accounting, throughput, GPU memory, CPU
allocation, and temperature. Require Task 7 thresholds.

- [ ] **Step 4: Continue only after a passed hold**

Resume ABO, then HSSD, 3D-FUTURE, ObjaverseXL_sketchfab, and
ObjaverseXL_github. On failure, stop new admission, preserve the frozen batch
and completed outputs, and return to the last passed profile.
