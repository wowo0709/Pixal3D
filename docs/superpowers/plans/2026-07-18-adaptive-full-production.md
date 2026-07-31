# Adaptive Full Production Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the over-conservative three-asset preprocessing schedule with a source-sized, resource-safe pipeline that qualifies 64 assets per source and then preprocesses all five training sources with bounded 256-asset production batches.

**Architecture:** Preserve the Pixal3D data contract and every leaf preprocessing command, but separate sizing and admission decisions by gate. Hardware evidence sizes smoke, smoke evidence sizes pilot, and pilot evidence sizes production; batch caps prevent a small p95 from creating an unbounded batch. Resource admission uses sustained pressure and temperatures instead of pausing on any non-zero swap-in, while command-boundary tuning and a two-lane CPU/GPU scheduler keep the 48 physical CPU cores and seven GPUs occupied without changing generated artifacts.

**Tech Stack:** Python 3.11, PyTorch >=2.8 with CUDA 12.8, Blender 4.5.1/OptiX, pytest, psutil, pandas, YAML, existing Pixal3D pipeline CLI.

## Global Constraints

- CUDA 12.8 and PyTorch 2.8 or newer are mandatory.
- Local scratch is `/root/node17/data/pixal3d`; control/prepared storage remains `/root/data2/pixal3d`; archive storage remains `/root/data3/pixal3d`.
- Keep the paper-faithful preprocessing contract: eight 512px condition renders, view-aligned views `0,1`, resolutions `256,512,1024`, SS resolution `64`, and FP32 latents.
- Do not add learned preprocessing modules, change camera selection, or change output schemas.
- K remains variable in the later model implementation; this plan only prepares the existing calibrated views.
- Data/provider failures are quarantined and excluded from training; schema failures stop the affected source.
- Never discard completed packs, archives, quality ledgers, or frozen scopes without first moving them into a timestamped recovery root.
- Source order for preprocessing is `ABO`, `HSSD`, `3D-FUTURE`, `ObjaverseXL_sketchfab`, `ObjaverseXL_github`; already-authorized ObjaverseXL downloads may continue in parallel with the three small sources.

## Execution note (2026-07-18)

Gate-aware sizing, sustained resource guarding, command-boundary worker tuning,
source ordering, and throughput reporting are implemented and tested. The
two-lane CPU/GPU scheduler is intentionally deferred: the current runner owns
a mutable active context and is not safe for concurrent batches without
changing command semantics. Production therefore uses sequential audited
batches, with seven GPU ranks within each encoder stage.

---

## File Structure

- `data_toolkit/pipeline/config.py`: strict schemas for gate batch caps, sustained resource thresholds, and command-boundary worker bounds.
- `data_toolkit/configs/multiview_preprocess.yaml`: production values for this 48-core/7-GPU node.
- `data_toolkit/pipeline/runtime.py`: gate-aware source sizing artifact reader.
- `data_toolkit/pipeline/orchestrator.py`: capped batch planning, command-boundary tuning, and bounded two-lane scheduling.
- `data_toolkit/pipeline/resources.py`: sustained swap/temperature policy and short hysteresis recovery.
- `data_toolkit/pipeline/commands.py`: materialize tuned worker counts without altering leaf command semantics.
- `data_toolkit/pipeline/reporting.py`: source throughput, utilization, and ETA evidence.
- `data_toolkit/pipeline/full_run.py`: small-source-first audited production order and resumable launch.
- `tests/data_toolkit/test_config.py`: strict configuration contract tests.
- `tests/data_toolkit/test_resources.py`: sustained-pressure and temperature policy tests.
- `tests/data_toolkit/test_orchestrator.py`: source sizing, caps, tuning, and scheduler tests.
- `tests/data_toolkit/test_commands.py`: tuned argv/environment tests.
- `tests/data_toolkit/test_reporting.py`: throughput and ETA tests.
- `tests/data_toolkit/test_full_run.py`: fixed source order and resume tests.
- `docs/data_preprocessing_runbook_ko.md`: exact operator commands, thresholds, expected artifacts, and recovery procedure.

### Task 1: Add gate-specific caps and gate-aware source sizing

**Files:**
- Modify: `data_toolkit/pipeline/config.py`
- Modify: `data_toolkit/configs/multiview_preprocess.yaml`
- Modify: `data_toolkit/pipeline/runtime.py`
- Modify: `data_toolkit/pipeline/orchestrator.py`
- Test: `tests/data_toolkit/test_config.py`
- Test: `tests/data_toolkit/test_cli.py`
- Test: `tests/data_toolkit/test_orchestrator.py`

**Interfaces:**
- Produces: `BatchConfig(smoke_max_assets=3, pilot_max_assets=64, production_max_assets=256)` and `SizingArtifactReader.p95_peak_local_bytes(source: str, gate: str) -> int`.
- Consumes: hardware `pilot_sizing.sources`, smoke gate `capacity.sources`, and pilot gate `capacity.sources`.

- [ ] **Step 1: Write failing strict-schema tests**

Add assertions that the checked-in config loads caps `3/64/256`, rejects zero or boolean caps, and rejects unknown batching keys. Add sizing-reader fixtures with distinct values `hardware=350 GiB`, `smoke=115 MiB`, and `pilot=180 MiB`; assert smoke uses hardware, pilot uses smoke, and production uses pilot.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
conda run -n pixal3d pytest \
  tests/data_toolkit/test_config.py \
  tests/data_toolkit/test_cli.py -k 'sizing or batch' \
  tests/data_toolkit/test_orchestrator.py -k 'work_batches or cap' -v
```

Expected: failures because `BatchConfig`, gate-aware reads, and batch caps do not exist.

- [ ] **Step 3: Add the configuration contract**

Add a frozen `BatchConfig` dataclass, require a top-level `batching` section, validate all three values as positive integers, and require `smoke <= pilot <= production <= shard_size`. Configure:

```yaml
batching: {smoke_max_assets: 3, pilot_max_assets: 64, production_max_assets: 256}
```

Increment `pipeline_version` to `pixal3d-mv-v2` because frozen qualification markers bind the configuration hash.

- [ ] **Step 4: Replace the one-argument pilot reader with gate-aware sizing**

Implement the exact precedence:

```python
def p95_peak_local_bytes(self, source: str, gate: str) -> int:
    if gate == "smoke":
        sources = read_hardware_report(self.config)["pilot_sizing"]["sources"]
    elif gate == "pilot":
        sources = read_gate_report(self.config, "smoke")["capacity"]["sources"]
    elif gate == "production":
        sources = read_gate_report(self.config, "pilot")["capacity"]["sources"]
    else:
        raise ValueError(f"unknown gate: {gate}")
    return _nonnegative_count(
        sources[source]["p95_peak_local_bytes"],
        f"{gate} sizing p95",
        positive=True,
    )
```

Rename the concrete provider to `SizingArtifactReader` and update its protocol and service wiring consistently.

- [ ] **Step 5: Cap frozen work batches**

Extend `plan_work_batches(..., max_batch_assets: int)` and calculate:

```python
per_asset = (p95_peak_bytes * 5 + 3) // 4
budget = local_usable_bytes * 4 // 5
batch_size = min(shard_size, max_batch_assets, budget // per_asset)
```

Select the cap from the gate and pass `gate` to the sizing reader. Assert a 115MiB p95 with ample disk yields one 64-asset pilot batch and 256-asset production batches.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
conda run -n pixal3d pytest \
  tests/data_toolkit/test_config.py \
  tests/data_toolkit/test_cli.py \
  tests/data_toolkit/test_orchestrator.py -v
git add data_toolkit/pipeline/config.py data_toolkit/pipeline/runtime.py \
  data_toolkit/pipeline/orchestrator.py data_toolkit/configs/multiview_preprocess.yaml \
  tests/data_toolkit/test_config.py tests/data_toolkit/test_cli.py \
  tests/data_toolkit/test_orchestrator.py
git commit -m "perf: size preprocessing batches from held gate evidence"
```

Expected: all focused tests pass.

### Task 2: Replace instantaneous swap pauses with sustained safety controls

**Files:**
- Modify: `data_toolkit/pipeline/config.py`
- Modify: `data_toolkit/configs/multiview_preprocess.yaml`
- Modify: `data_toolkit/pipeline/resources.py`
- Test: `tests/data_toolkit/test_resources.py`

**Interfaces:**
- Produces: temperature fields on `ResourceSnapshot`, sustained swap-rate decisions, and configurable recovery hysteresis.
- Consumes: psutil CPU temperature data and existing GPU metrics.

- [ ] **Step 1: Write failing policy tests**

Cover these exact cases:

- A single 4KiB or 24KiB swap-in sample with more than 64GiB available RAM remains `RUN`.
- Swap-in totaling at least 256MiB over 60 seconds for three consecutive samples becomes `PAUSE`.
- Available RAM below 64GiB remains `STOP`.
- CPU temperature at or above 85C for 30 seconds becomes `PAUSE`; at or above 92C becomes `STOP`.
- GPU temperature at or above 80C for 30 seconds becomes `PAUSE`; at or above 88C becomes `STOP`.
- A paused command resumes after 30 stable seconds, not five minutes.

- [ ] **Step 2: Run the focused tests and verify failure**

```bash
conda run -n pixal3d pytest tests/data_toolkit/test_resources.py \
  -k 'swap or temperature or recovery' -v
```

Expected: existing policy pauses immediately on 4KiB and lacks temperature/recovery configuration.

- [ ] **Step 3: Extend the strict limits schema**

Add these checked-in values:

```yaml
swap_soft_mib_per_minute: 256
swap_soft_samples: 3
cpu_temp_soft_celsius: 85
cpu_temp_hard_celsius: 92
gpu_temp_soft_celsius: 80
gpu_temp_hard_celsius: 88
temperature_soft_seconds: 30
recovery_stable_seconds: 30
```

Retain CPU soft 80%, CPU hard 90%, load soft 72, I/O wait soft 10%, RAM soft 96GiB, RAM hard 64GiB, and all storage floors.

- [ ] **Step 4: Sample temperatures and evaluate sustained pressure**

Add `cpu_max_temperature_celsius: float | None` to `ResourceSnapshot`. Select the maximum finite value from `psutil.sensors_temperatures()` when available; record `None` when the platform exposes no sensor. Maintain a timestamped 60-second swap deque and a consecutive-over-threshold count. Never infer thermal safety from a missing sensor; retain CPU percentage/load protection in that case.

- [ ] **Step 5: Shorten recovery without flapping**

Replace the hard-coded `5 * 60` in `ResourceGuard` with `limits.recovery_stable_seconds`. Any new soft/hard violation resets the stable clock. Preserve supervisor pause/resume and telemetry writes.

- [ ] **Step 6: Run tests and commit**

```bash
conda run -n pixal3d pytest tests/data_toolkit/test_resources.py \
  tests/data_toolkit/test_config.py -v
git add data_toolkit/pipeline/config.py data_toolkit/pipeline/resources.py \
  data_toolkit/configs/multiview_preprocess.yaml \
  tests/data_toolkit/test_resources.py tests/data_toolkit/test_config.py
git commit -m "perf: gate preprocessing on sustained resource pressure"
```

Expected: one-off swap traffic remains runnable and sustained/thermal hazards still pause or stop.

### Task 3: Tune CPU workers at command boundaries and fully feed seven GPUs

**Files:**
- Modify: `data_toolkit/pipeline/config.py`
- Modify: `data_toolkit/configs/multiview_preprocess.yaml`
- Modify: `data_toolkit/pipeline/commands.py`
- Modify: `data_toolkit/pipeline/orchestrator.py`
- Test: `tests/data_toolkit/test_commands.py`
- Test: `tests/data_toolkit/test_orchestrator.py`

**Interfaces:**
- Produces: `WorkerProfile` and `choose_worker_profile(recent_snapshots) -> WorkerProfile` applied only before a leaf command starts.
- Consumes: recent telemetry; never changes a running process's argv.

- [ ] **Step 1: Write failing profile tests**

Require a cold-start profile of dump 32, voxel 8x4 native threads, render 7, encoder 7 ranks. After a stable completed command with CPU below 70%, I/O wait below 5%, RAM above 128GiB, and no thermal/swap violation, require dump workers to advance `32 -> 36 -> 40 -> 44`. Require pressure to reduce one step. Require all profiles to satisfy `processes * native_threads <= 44` for CPU voxel work.

- [ ] **Step 2: Run focused tests and verify failure**

```bash
conda run -n pixal3d pytest \
  tests/data_toolkit/test_commands.py \
  tests/data_toolkit/test_orchestrator.py -k 'worker_profile or tuned_command' -v
```

- [ ] **Step 3: Add bounded profile configuration**

Configure physical-core bounds rather than using all 96 logical CPUs:

```yaml
worker_tuning:
  dump_steps: [32, 36, 40, 44]
  voxel_profiles: [[8, 4], [10, 4], [11, 4]]
  render_workers: 7
  encoder_ranks: 7
```

Keep `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1` for CPU leaf commands. Preserve render's two OpenMP threads and one OptiX process per GPU.

- [ ] **Step 4: Materialize command argv from the selected profile**

Change `build_preprocessing_dag` to accept a `WorkerProfile`. Use its dump, voxel, and native-thread values in argv. Continue emitting exactly seven render/encoder ranks with `CUDA_VISIBLE_DEVICES=0..6`.

- [ ] **Step 5: Apply tuning only between commands**

After a command exits and outputs validate, summarize its telemetry window and choose the next profile. Persist the selected profile in checkpoint evidence so resume uses the same profile for an active attempt and may retune only for the next command.

- [ ] **Step 6: Run tests and commit**

```bash
conda run -n pixal3d pytest \
  tests/data_toolkit/test_commands.py \
  tests/data_toolkit/test_orchestrator.py \
  tests/data_toolkit/test_pipeline_integration.py -v
git add data_toolkit/pipeline/config.py data_toolkit/pipeline/commands.py \
  data_toolkit/pipeline/orchestrator.py data_toolkit/configs/multiview_preprocess.yaml \
  tests/data_toolkit/test_commands.py tests/data_toolkit/test_orchestrator.py
git commit -m "perf: tune preprocessing workers at command boundaries"
```

### Task 4: Add a bounded CPU/GPU two-lane scheduler

**Files:**
- Modify: `data_toolkit/pipeline/commands.py`
- Modify: `data_toolkit/pipeline/orchestrator.py`
- Test: `tests/data_toolkit/test_orchestrator.py`
- Test: `tests/data_toolkit/test_pipeline_integration.py`

**Interfaces:**
- Produces: command resource class `cpu`, `gpu`, or `internal`, and at most two active batches: one CPU-lane batch and one GPU-lane batch.
- Consumes: existing command dependencies, checkpoints, resource guard, validators, quality ledger, and pack publisher.

- [ ] **Step 1: Write failing deterministic scheduler tests**

Use fake leaf workers and two frozen batches. Assert that CPU preprocessing for batch 1 may overlap batch 0 latent encoding, but two render/encoder GPU commands never overlap. Assert packing/archive/cleanup remains batch-local and begins only after that batch's dependencies validate. Assert interruption leaves independently resumable checkpoints for both batches.

- [ ] **Step 2: Run focused integration tests and verify failure**

```bash
conda run -n pixal3d pytest tests/data_toolkit/test_orchestrator.py \
  tests/data_toolkit/test_pipeline_integration.py -k 'lane or overlap or resume' -v
```

- [ ] **Step 3: Classify existing commands without changing them**

Classify download, stage, dump, stats, dual-grid, and voxelization as CPU; render and all latent encoders as GPU; validation, packing, archive, and cleanup as internal. The GPU lane owns all seven GPUs, so render and latent encoders are mutually exclusive.

- [ ] **Step 4: Implement a two-batch upper bound**

Permit only:

```text
CPU lane: batch N+1 download/dump/voxel
GPU lane: batch N render/encode
```

Do not start batch N+2 until one lane releases a batch. Acquire resource admission before every command, preserve per-batch supervisors, and stop both lanes cleanly on schema/infrastructure/resource escalation.

- [ ] **Step 5: Run integration tests and commit**

```bash
conda run -n pixal3d pytest tests/data_toolkit/test_orchestrator.py \
  tests/data_toolkit/test_pipeline_integration.py -v
git add data_toolkit/pipeline/commands.py data_toolkit/pipeline/orchestrator.py \
  tests/data_toolkit/test_orchestrator.py \
  tests/data_toolkit/test_pipeline_integration.py
git commit -m "perf: overlap bounded CPU and GPU preprocessing lanes"
```

### Task 5: Report source throughput and trustworthy ETA

**Files:**
- Modify: `data_toolkit/pipeline/reporting.py`
- Modify: `data_toolkit/pipeline/cli.py`
- Test: `tests/data_toolkit/test_reporting.py`
- Test: `tests/data_toolkit/test_cli.py`

**Interfaces:**
- Produces: `control/reports/performance/<gate>/<source>.json` with held counts, elapsed time, assets/hour, per-stage p50/p95, CPU/GPU utilization, pause fraction, peak scratch, and ETA.

- [ ] **Step 1: Write failing report tests**

Build telemetry containing active and paused periods. Require throughput to use wall time, expose pause fraction separately, reject incomplete segments, and calculate `eta_hours = remaining_assets / assets_per_hour`. Require source identity, config hash, frozen-scope hash, and telemetry checksum.

- [ ] **Step 2: Run tests and verify failure**

```bash
conda run -n pixal3d pytest tests/data_toolkit/test_reporting.py \
  tests/data_toolkit/test_cli.py -k 'performance or eta' -v
```

- [ ] **Step 3: Implement and expose the report**

Add:

```bash
python -m data_toolkit.pipeline.cli performance \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --gate pilot --source ABO --shard ABO-00000
```

Return non-zero for missing/incomplete held evidence. Do not extrapolate a source from another source's performance.

- [ ] **Step 4: Run tests and commit**

```bash
conda run -n pixal3d pytest tests/data_toolkit/test_reporting.py \
  tests/data_toolkit/test_cli.py -v
git add data_toolkit/pipeline/reporting.py data_toolkit/pipeline/cli.py \
  tests/data_toolkit/test_reporting.py tests/data_toolkit/test_cli.py
git commit -m "feat: report held preprocessing throughput and ETA"
```

### Task 6: Recover the current v1 pilot and run v2 smoke/pilot qualification

**Files:**
- Modify: `docs/data_preprocessing_runbook_ko.md`
- Runtime: `/root/data2/pixal3d/control/recovery/**`
- Runtime: `/root/data3/pixal3d/recovery/**`
- Runtime: `/root/node17/data/pixal3d/recovery/**`

**Interfaces:**
- Produces: preserved v1 state, five passed v2 smoke reports, five audited 64-asset pilot scopes, and source-specific performance reports.

- [ ] **Step 1: Verify the complete unit suite before touching runtime state**

```bash
conda run -n pixal3d pytest tests/data_toolkit -q
git status --short
```

Expected: tests pass; only intentional changes are present.

- [ ] **Step 2: Disable automatic continuation and terminate the active v1 command gracefully**

Resolve exact PIDs using full command lines, send `TERM` first, wait for the orchestrator to persist its active attempt, and use the supervisor's existing escalation only if it does not exit. Do not use a broad `pkill` pattern. Record PIDs, commands, exit status, and timestamp in the runbook.

- [ ] **Step 3: Move v1 qualification state into timestamped recovery roots**

Preserve frozen scopes, checkpoints, ledgers, pilot reports/evidence, local qualification scratch, and continuation logs under a single recovery ID. Verify file counts and SHA-256 manifests before removing the active names. Leave prepared packs and raw archives untouched.

- [ ] **Step 4: Rebuild v2 held hardware/smoke evidence**

Run hardware preflight with a bootstrap reservation used only for the capped three-asset smoke. Then run and audit the frozen smoke scope for each source and collect/report smoke. Exact CLI pattern:

```bash
python -m data_toolkit.pipeline.cli run --config data_toolkit/configs/multiview_preprocess.yaml \
  --gate smoke --source SOURCE --shard SHARD --count 9
python -m data_toolkit.pipeline.cli audit --config data_toolkit/configs/multiview_preprocess.yaml \
  --gate smoke --source SOURCE --shard SHARD
python -m data_toolkit.pipeline.cli evidence --config data_toolkit/configs/multiview_preprocess.yaml --gate smoke
python -m data_toolkit.pipeline.cli report --config data_toolkit/configs/multiview_preprocess.yaml --gate smoke
```

Require every source p95 to come from its own smoke telemetry.

- [ ] **Step 5: Run one 64-asset pilot per source**

Use source/shard pairs:

```text
ABO / ABO-00000
HSSD / HSSD-00000
3D-FUTURE / 3D-FUTURE-00000
ObjaverseXL_sketchfab / ObjaverseXL_sketchfab-00000
ObjaverseXL_github / ObjaverseXL_github-00000
```

For each pair run `run --gate pilot --count 64`, `audit --gate pilot`, and `performance --gate pilot`. After all five, collect pilot evidence and build the pilot gate report.

- [ ] **Step 6: Apply the production admission criteria**

Admit production only when every source satisfies all of:

- audit exit code 0;
- zero schema failures;
- end-to-end failure rate at most 10%, with provider/data failures quarantined;
- no resource hard stop;
- pause fraction below 10%;
- local peak projection within storage limits;
- no CPU/GPU hard temperature event;
- all seven GPUs receive work during GPU stages;
- performance report contains a source-local ETA.

If throughput is below 400 assets/hour, run a second 64-asset performance canary after tuning one bounded parameter; never proceed by assuming a speedup.

- [ ] **Step 7: Commit the runbook update**

```bash
git add docs/data_preprocessing_runbook_ko.md
git commit -m "docs: record adaptive qualification and recovery evidence"
```

### Task 7: Launch resumable full production and audit every shard

**Files:**
- Modify: `data_toolkit/pipeline/full_run.py`
- Modify: `docs/data_preprocessing_runbook_ko.md`
- Test: `tests/data_toolkit/test_full_run.py`

**Interfaces:**
- Produces: audited prepared packs for all usable assets and a checksum-bound training handoff.

- [ ] **Step 1: Write failing fixed-order/resume tests**

Require exact order `ABO`, `HSSD`, `3D-FUTURE`, `ObjaverseXL_sketchfab`, `ObjaverseXL_github`; require an audit after every canonical shard; require resume to skip only validated published batches; require no production command to carry `--count`.

- [ ] **Step 2: Run the focused tests and verify failure**

```bash
conda run -n pixal3d pytest tests/data_toolkit/test_full_run.py -v
```

- [ ] **Step 3: Implement the fixed audited order**

Update `FullProductionRunner` to use the exact order above, while preserving canonical shard order within each source. Continue downloading missing ObjaverseXL raw assets on demand; reuse already downloaded ABO/HSSD/3D-FUTURE material.

- [ ] **Step 4: Run tests and commit**

```bash
conda run -n pixal3d pytest tests/data_toolkit/test_full_run.py \
  tests/data_toolkit/test_pipeline_integration.py -v
git add data_toolkit/pipeline/full_run.py tests/data_toolkit/test_full_run.py
git commit -m "feat: run small-source-first audited full production"
```

- [ ] **Step 5: Start full production only after pilot admission**

```bash
conda run -n pixal3d python -m data_toolkit.pipeline.cli full-run \
  --config data_toolkit/configs/multiview_preprocess.yaml
```

Run it in the existing durable background launcher, write PID/commit/config hash to the launch log, and ensure restart invokes the identical command.

- [ ] **Step 6: Monitor and report at deterministic checkpoints**

Recompute held performance and ETA after the first 256 assets of each source, every completed shard, and every 24 hours. Reduce one worker step on sustained pressure; increase one step only after a complete stable command. Never change batch boundaries after they are frozen.

- [ ] **Step 7: Final audit and training handoff**

After all source shards audit, collect/report production, verify every pack checksum, require quarantine ledgers for excluded assets, and publish `control/splits/training_handoff.json`. Only then mark data preprocessing complete and begin the already-approved paper-faithful multi-view model implementation plan.

## Expected Timeline Gates

- Implementation and tests: approximately 1–2 working days.
- Recovery plus five-source smoke: approximately 4–8 hours.
- Five 64-asset pilots and one retune if needed: approximately 6–18 hours.
- Production ETA: calculated from source-local pilot evidence; the current planning envelope is 30–45 days, not an admission promise.
- First useful completed sources: ABO/HSSD/3D-FUTURE should finish before ObjaverseXL and provide an early end-to-end training handoff validation.

## Self-Review

- The plan preserves CUDA/Torch, paths, FP32, views, resolutions, camera policy, quarantine, and paper-faithful output requirements.
- The 115MiB observation is used only as source-local scratch evidence, never as RAM per asset or a universal source value.
- Batch size and concurrent workers are separate controls.
- The scheduler never overlaps two all-GPU commands and never exceeds two active batches.
- Full production cannot begin without passed smoke/pilot evidence and exact audits.
- Existing v1 state is recoverable and prepared/raw publications are not deleted.
