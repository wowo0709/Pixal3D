# Parallel Preprocessing Saturation Design

## Goal

Increase Pixal3D preprocessing throughput by keeping the seven GPUs and 48
physical CPU cores busy at the same time, while preserving the existing
paper-faithful data contract and recoverable audit trail.

The operating target is 80% GPU memory use and 44 physical CPU cores. The
resource guard keeps the existing temperature, RAM, swap, and storage limits;
GPU memory above 90% is a hard admission stop for additional work.

## Frozen Data Contract

- Eight Blender condition images per asset at 512 x 512.
- Calibrated views 0 and 1 for downstream aligned features.
- Shape and PBR resolutions 256, 512, and 1024.
- SS resolution 64.
- FP32 latent outputs.
- Existing camera policy, OptiX rendering, validation schemas, pack families,
  quarantine ledger, and raw archives.
- No learned module, camera selection rule, or training-data schema changes.

## Current Bottleneck

The current command DAG places a full-stage barrier between render,
dual-grid, PBR voxelization, encoding, validation, and publication. Each GPU
encoder process handles one asset-view at a time. CPU geometry uses 8 workers
with 4 native threads each, leaving 12 physical cores unused. Consequently,
CPU-heavy and GPU-heavy stages alternate instead of overlapping.

The completed ABO pilot processed 64 assets in 1,928.717 seconds. The largest
stages were condition rendering (475.492 seconds), dual-grid 1024 (372.492
seconds), and PBR voxelization 1024 (372.778 seconds).

## Architecture

Production retains a frozen output batch of at most 256 assets. Each output
batch is divided into independently resumable internal chunks of 32 or 64
assets. An isolated runtime context owns each chunk so concurrent work never
shares the existing mutable active context.

Chunks move through five bounded lanes:

1. Download, stage, mesh dump, and PBR dump.
2. Blender condition rendering.
3. Dual-grid and PBR voxel generation.
4. Shape, PBR, and SS encoding.
5. Validation, pack creation, raw archive publication, and cleanup.

Dependencies are enforced per chunk, but different chunks may occupy
different lanes. Publication remains serialized per frozen output batch so
indexes, accounting, quality ledgers, and archives retain their current
atomicity.

## GPU Encoder Batching

The shape, PBR, and SS leaf encoders will accept an explicit micro-batch size.
Loader threads collect variable-size sparse tasks, assign distinct sparse
batch coordinates, run one encoder call, then split the output by batch
coordinate before the existing per-view atomic save.

Initial micro-batch profiles are:

- 256: 16 asset-views per GPU.
- 512: 8 asset-views per GPU.
- 1024: 4 asset-views per GPU.
- SS-64: 16 asset-views per GPU.

Profiles are calibrated against high-token pilot assets. After each command,
the controller reads peak reserved GPU memory. It increases one step while
peak use is below 70%, holds between 70% and 80%, and decreases one step above
80%. An out-of-memory error discards no completed outputs, reduces the
micro-batch by half, and resumes the same frozen task list. No new tasks are
admitted above 90% memory use.

If a sparse backend cannot preserve output isolation for a true batched call,
the compatible fallback is two to four independent ranks per GPU, mapped
round-robin. This fallback still uses the same single-task encoder computation
and must pass the same parity checks.

## Blender Rendering

The render scheduler starts with two Blender workers per GPU (14 total) and
may advance to three or four workers per GPU when peak memory remains below
80%, temperature remains below the configured soft threshold, and no OptiX
failure occurs. Each worker renders one asset independently with the existing
eight 512 x 512 cameras.

GPU assignment is round-robin rather than assuming rank equals GPU index.
Failure of one asset is retried on the same frozen asset before quarantine;
other workers continue.

## CPU Geometry

Dual-grid and PBR voxel tasks share one global pool limited to 44 physical
cores. The initial profile is 11 processes with four native threads each.
The pool schedules asset-view tasks from ready chunks rather than running two
independent 44-thread commands that would oversubscribe the node.

Worker placement is NUMA-aware across the two 24-core sockets. Admission
backs down to 10 x 4 and then 8 x 4 when CPU temperature, load, RAM, swap, or
I/O wait crosses an existing soft threshold.

## Resource Coordination

One node-level broker owns concurrency decisions for all lanes. Separate CLI
processes may not independently saturate the same resource. The broker uses:

- GPU memory target: 80%; hard admission ceiling: 90%.
- CPU allocation: 44 physical cores; four cores reserved for the OS,
  monitoring, and publication.
- Existing CPU/GPU temperature limits and 30-second recovery hysteresis.
- Existing RAM, swap, local scratch, data2, and data3 floors.
- Bounded lane queues so upstream work cannot exhaust scratch storage.

When pressure occurs, the broker stops admitting new tasks and reduces the
relevant lane by one profile step. Running leaf tasks are allowed to complete
unless a hard thermal, memory, or storage limit is crossed.

## Recovery and Audit

Every internal chunk records its frozen SHA order, completed stage, worker
profile, peak resource use, retry count, and output checksums. Restarting a
production batch reconstructs lane queues from these checkpoints. Completed
outputs are validated and skipped; stale generated artifacts are moved to a
timestamped recovery directory before replacement.

Provider-unavailable assets remain quarantined and excluded from the training
denominator. Schema failures stop the affected source. Pack publication,
quality-ledger advancement, accounting, and raw archive publication use the
existing serialized validators.

## Verification

Before production, benchmark micro-batches and renderer concurrency on pilot
assets including the largest sparse-token examples. Required checks are:

- Exact output family, view, resolution, dtype, and coordinate contracts.
- Coordinate identity and FP32 numerical parity against batch size one.
- Successful pack, archive, quality-ledger, and end-to-end audits.
- No missing or duplicate asset-view output under restart.
- Peak GPU memory at or below 80% in steady state and never above 90%.
- At most 44 physical CPU cores assigned to preprocessing workers.
- No thermal, RAM, swap, or storage hard-limit violation.

The throughput acceptance target is at least 1.8 times the ABO pilot baseline
of 119.46 assets/hour, measured end to end including validation and
publication. If true sparse batching fails parity, the multi-rank fallback may
be accepted only if it meets the same output and throughput requirements.

## Rollout

1. Add telemetry-only peak memory and stage timing measurements.
2. Benchmark encoder micro-batches 2, 4, 8, and 16 by resolution.
3. Benchmark two, three, and four Blender workers per GPU.
4. Enable the 44-core shared geometry pool.
5. Enable two chunks in flight, then three after a clean audit.
6. Run a fresh 64-asset qualification batch and compare it with the existing
   baseline.
7. Use the selected profiles for 256-asset production batches.

Any failed rollout step returns to the last audited profile without changing
the frozen batch or data contract.
