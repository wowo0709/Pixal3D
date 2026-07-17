# Pixal3D Multi-View Data Preprocessing Design

**Date:** 2026-07-16

**Status:** Approved

**Scope:** Data download, preprocessing, validation, packing, and archival before multi-view fine-tuning

## Objective

Build a resumable preprocessing pipeline that prepares the complete public Pixal3D/TRELLIS-500K training pool before fine-tuning begins. The output must support independently training and validating Pixal3D Stage 1, Stage 2, and Stage 3 with known cameras and two view-aligned target anchors.

The pipeline renders eight condition views per asset, generates target latents for `view00` and `view01`, retains only validated training-ready data on `/root/data2/pixal3d`, archives raw assets on `/root/data3/pixal3d`, and uses the local NVMe-backed root `/root/node17/data/pixal3d` for bounded scratch work.

## References

- [Pixal3D data toolkit](../../../data_toolkit/README.md)
- [Pixal3D training overview](../../../README.md)
- [TRELLIS-500K dataset](https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K/tree/main)
- [TRELLIS.2 data toolkit](https://github.com/microsoft/TRELLIS.2/blob/main/data_toolkit/README.md)
- [Blender 4.5 LTS](https://www.blender.org/download/lts/4-5/)
- [Blender Cycles GPU configuration](https://docs.blender.org/manual/en/3.3/editors/preferences/system.html)

## Scope

### Included

- Public training metadata and assets from ObjaverseXL Sketchfab, ObjaverseXL GitHub, ABO, HSSD, and 3D-FUTURE.
- Toys4K as a separate evaluation-only registry.
- Toolkit hardening required to run the published pipeline reliably at production scale.
- Deterministic eight-view 512-pixel RGBA condition rendering with known camera parameters.
- Two view-aligned anchor targets at `view00` and `view01`.
- Stage 1 SS latents at source resolution 64.
- Stage 2 shape latents at 256, 512, and 1024.
- Stage 3 PBR latents at 256, 512, and 1024.
- Canonical manifests, deterministic splits, checksums, validation reports, and training packs.
- Continuous CPU, RAM, I/O, and storage headroom monitoring.

### Excluded

- TexVerse until its raw GLB files and metadata are supplied. Its current adapter intentionally does not download data.
- Camera pose estimation. All inputs use renderer-provided intrinsics and extrinsics.
- Multi-view model, dataset loader, trainer, and inference implementation. Those belong to the subsequent fine-tuning project.
- Fine-tuning while preprocessing is active. The two workloads must not overlap.

## Dataset Inventory

| Source | Training assets | Acquisition |
|---|---:|---|
| ObjaverseXL Sketchfab | 168,307 | Existing ObjaverseXL adapter |
| ObjaverseXL GitHub | 311,843 | Existing ObjaverseXL adapter |
| ABO | 4,485 | Existing adapter after API fix |
| HSSD | 6,670 | Add adapter from public metadata/source |
| 3D-FUTURE | 9,472 | Add adapter from public metadata/source |
| **Total metadata rows** | **500,777** | Canonical count is reported after global SHA deduplication |

Toys4K contains 3,229 evaluation assets. It is stored in a separate registry and is never included in a training shard.

## Fixed Data Contract

Each successful training asset has the following immutable contract:

- Eight condition images named `000.png` through `007.png`.
- Images are 512 by 512 RGBA PNGs.
- `transforms.json` contains exactly eight frames with finite `camera_angle_x`, camera-to-world `transform_matrix`, and actual render radius.
- Camera samples use NumPy `PCG64` seeded from the first eight bytes of `SHA256("pixal3d-mv-camera-v1:" + asset_sha256)`.
- `view00` and `view01` are the only view-aligned target anchors.
- SS packs contain both anchor NPZ files and their scale metadata.
- Shape packs contain both anchor NPZ files and scale metadata for each of 256, 512, and 1024.
- PBR packs contain both anchor NPZ files and scale metadata for each of 256, 512, and 1024.
- Views `02` through `07` are condition-only views. They do not receive target voxel or latent files.

During multi-view training, the target anchor is selected from `view00` or `view01`; the same view is placed first in the condition sequence, followed by a sampled subset of the other rendered views.

## Storage Architecture

### Warm Tier: `/root/data2/pixal3d`

`/root/data2` has a project budget of 20 TiB. It receives downloads and stores all validated training-ready packs. Normal operation targets at most 16 TiB and hard-stops at 18 TiB.

```text
/root/data2/pixal3d/
├── control/
│   ├── assets.parquet
│   ├── eval_assets.parquet
│   ├── splits/
│   ├── shards/
│   └── reports/
├── raw/<source>/
├── staging/<shard_id>/
└── prepared/
    ├── common/<source>/<shard_id>.tar
    ├── ss/64/<source>/<shard_id>.tar
    ├── shape/{256,512,1024}/<source>/<shard_id>.tar
    └── pbr/{256,512,1024}/<source>/<shard_id>.tar
```

### Archive Tier: `/root/data3/pixal3d`

```text
/root/data3/pixal3d/archive/
├── raw/<source>/<shard_id>.tar
├── failed/<source>/
└── retired/
```

Raw data moves to the archive only after every final pack for its shard passes validation. Because `/root/data2` and `/root/data3` are different NFS mounts, archival is copy, checksum verification, and source deletion rather than an unchecked cross-filesystem move.

### Local Scratch: `/root/node17/data/pixal3d`

```text
/root/node17/data/pixal3d/
├── preprocess/
│   ├── active/
│   ├── output/
│   └── tmp/
└── train/
    ├── active/
    ├── next/
    └── runtime/
```

The preprocessing scratch budget is at most 350 GiB while the local filesystem has approximately 607 GiB free. After local capacity is expanded to approximately 2.5 TiB, a work batch may grow to at most 800 GiB. The resource controller also enforces a dynamic free-space floor, so these are upper bounds rather than allocation targets.

## Registry and Sharding

`control/assets.parquet` is the canonical source of truth. Toolkit `metadata.csv` files are compatibility projections generated from it, not independent state stores.

Each asset row includes:

- `sha256`, source, source identifier, and raw relative path.
- Global deduplication owner and duplicate source references.
- Split, logical shard, and work batch.
- Camera policy version and deterministic camera seed.
- Download, dump, render, voxel, latent, validation, pack, and archive states.
- Per-stage output size and checksum.
- Attempt count, last error category, error message, and timestamps.
- Tool Git commit, preprocessing config hash, encoder identifiers, and Blender version.

States are `pending`, `running`, `complete`, or `quarantined`. A stale `running` state returns to `pending` on resume. A state becomes `complete` only after the corresponding output is opened and validated.

Global SHA deduplication occurs before split assignment. Training assets use `int(sha256[:8], 16) % 100`: bucket 0 is validation and buckets 1 through 99 are training. Toys4K is the external test set. No SHA may occur in more than one split.

A logical source shard contains 5,000 assets in stable SHA order. The orchestrator divides it into smaller work batches when projected scratch use would exceed the current local limit. Logical shard identity remains stable even if work-batch sizing changes.

## Pack Format

Final packs are uncompressed tar archives. PNG, NPZ, and many raw source files are already compressed; another compression layer adds CPU cost with little space reduction. Tar provides sequential NFS reads and avoids transferring hundreds of thousands of small files.

Pack paths reproduce the directory layout expected by Pixal3D dataset classes after extraction. Stage 3 assembles common, shape, and PBR packs; shape data is not duplicated inside PBR packs.

Every pack has a sibling manifest containing:

- Ordered asset SHA list.
- Expected member paths and uncompressed sizes.
- Per-member SHA-256 and whole-pack SHA-256.
- Pipeline config hash and tool commit.
- Creation and validation timestamps.
- Counts for completed and quarantined assets.

Publishing uses `/root/data2/pixal3d/staging/<shard_id>` followed by content validation and an atomic rename into `prepared`. A partial pack is never visible as ready.

## Toolkit Architecture

Existing toolkit scripts remain leaf workers. A new preprocessing orchestrator owns planning, state, resource limits, retries, validation, and packing. It provides `plan`, `run`, `resume`, `audit`, and `report` operations against one versioned configuration.

The hardening work includes:

- Pass the download root, not the render output root, to `render_cond.py` dataset iteration.
- Expose render resolution and set the production default to 512.
- Replace process-global random camera generation with SHA-derived deterministic generation.
- Generate only the requested count of radius samples rather than one million samples per asset.
- Scale boundary-distance thresholds with render resolution.
- Normalize dataset adapter download signatures; the current ABO adapter does not accept the keyword used by `download.py`.
- Add HSSD and 3D-FUTURE adapters.
- Make loader and saver thread counts configurable in all latent encoders.
- Add timeout, atomic temporary output, content validation, and bounded retry behavior to every leaf stage.
- Add an explicit latent storage dtype while retaining FP32 as the safe fallback.
- Avoid full metadata scans when an `--instances` shard manifest is supplied.
- Pin Blender 4.5.1 LTS and select one OptiX device explicitly per renderer process. Blender 3.0.1 is not used for production.

## End-to-End Preprocessing DAG

### Phase 0: Toolkit Qualification

1. Implement and test toolkit hardening.
2. Build source registries and the global deduplicated registry.
3. Freeze split assignment, camera policy version, and configuration hash.
4. Run source smoke tests and the stratified pilot.
5. Calculate measured storage and time projections before production starts.

### Phase 1: Source Acquisition

Sources run sequentially in this order: ObjaverseXL Sketchfab, ObjaverseXL GitHub, ABO, HSSD, and 3D-FUTURE. Downloads land under `/root/data2/pixal3d/raw/<source>`. Each downloaded file is hash-checked against metadata before it is eligible for staging.

### Phase 2: Per-Shard Processing

For each logical shard:

1. Stage a bounded raw work batch under `/root/node17/data/pixal3d/preprocess/active`.
2. Dump normalized mesh and PBR data and collect asset statistics.
3. Render eight deterministic 512-pixel RGBA conditions and known cameras.
4. Generate `view00` and `view01` dual-grid and PBR voxels at 256.
5. Encode and validate 256 shape and PBR latents, then delete 256 voxel intermediates.
6. Repeat voxelization, encoding, validation, and deletion at 512.
7. Repeat voxelization, encoding, validation, and deletion at 1024.
8. Generate the two SS latents at source resolution 64 from validated 1024 shape coordinates.
9. Decode sampled latents for visual and metric validation.
10. Build and validate common, SS, shape, and PBR packs.
11. Publish packs atomically to `/root/data2/pixal3d/prepared`.
12. Remove local mesh, PBR dump, output, and temporary files.

Mesh and PBR dumps survive until all resolutions and SS encoding finish. Resolution-specific voxel intermediates are deleted immediately after their final latent passes validation.

### Phase 3: Raw Archival

After all work batches in a logical shard are published, pack the shard raw inputs, copy them to `/root/data3/pixal3d/archive/raw`, verify file count, byte count, and SHA-256, and delete the corresponding `/root/data2` raw files. Failed or quarantined source files remain addressable through the registry.

### Phase 4: Dataset Audit

After a source completes, audit every pack manifest, aggregate terminal states, sample decoded outputs, and produce source-level success, failure, throughput, and storage reports. After all sources complete, generate the immutable training handoff manifests used to populate local training shards.

## Resource Controller

Preprocessing must maintain system headroom instead of maximizing utilization. The orchestrator samples host and project metrics every five seconds and writes JSONL telemetry plus 30-second rollups under `control/reports/resource/`.

Tracked metrics include:

- Host and pipeline CPU utilization, one- and five-minute load average, process/thread counts, and context switches.
- CPU I/O wait and, when exposed by the host, temperature and throttling state.
- Available RAM, pipeline RSS, page cache, swap use, and swap-in/swap-out rate.
- Local, `/root/data2`, and `/root/data3` free bytes and Pixal3D project directory sizes.
- Read/write throughput and latency for local storage and both NFS mounts.
- Per-GPU utilization, memory, temperature, power, and active preprocessing rank.
- Per-stage queue depth, completion rate, retry rate, and estimated time to completion.

### CPU and Memory Guardrails

- CPU-heavy phases start with a maximum aggregate budget of 32 runnable CPU threads.
- Mesh/PBR dump and statistics use no more than 24 workers and constrain nested OpenMP-style libraries to one thread unless a lower worker count is selected.
- O-Voxel work uses at most eight workers with four native threads each.
- Rendering uses at most one OptiX renderer process per GPU and at most two CPU support threads per renderer.
- Latent encoding uses one rank per GPU with four loader threads and two saver threads per rank.
- Different CPU-heavy or GPU-heavy pipeline phases do not overlap.
- New tasks pause when host CPU utilization exceeds 80% or load average exceeds 72 for two minutes.
- Worker concurrency decreases when I/O wait exceeds 10% for two minutes.
- Available RAM must remain at least 96 GiB. New tasks pause below that threshold.
- Any sustained swap-in activity pauses new tasks and triggers a concurrency reduction.
- CPU utilization above 90% for five minutes, available RAM below 64 GiB, or repeated OOM causes a graceful checkpoint and orchestrator stop.

Paused scheduling resumes only after all soft metrics remain below their thresholds for five consecutive minutes. Adaptive control may reduce concurrency automatically. It never raises concurrency above the configured phase maximum, and it raises a reduced limit by only one worker after ten stable minutes.

### Storage Guardrails

- Local free space must remain at least the greater of 15% of the filesystem or 120 GiB.
- `/root/data2/pixal3d` has a 16 TiB soft limit and 18 TiB hard limit.
- `/root/data3/pixal3d` has a 26 TiB soft limit under the currently available archive capacity.
- The orchestrator also requires at least 2 TiB filesystem free on `/root/data2` and 4 TiB free on `/root/data3` before starting a new logical shard.
- Crossing a soft limit stops new work and runs cleanup/audit only.
- Crossing a hard local floor or data2 limit checkpoints state and exits before another output write begins.
- Temporary and intermediate paths have per-shard byte budgets; a worker cannot reserve output space that would violate a floor.

These limits are checked before task admission and continuously while tasks execute. An operator report is emitted immediately when a threshold remains violated after automatic cleanup or concurrency reduction.

## Execution Gates

### Smoke Gate

Run 20 assets from each of the five sources, covering all available source formats. All 100 assets traverse the complete DAG. Infrastructure, adapter, checkpoint, Blender device-selection, and schema errors must be zero before proceeding.

### Pilot Gate

Run a stratified set of 1,000 assets selected across source, file extension, raw byte size, mesh complexity, material count, and alpha usage. The pilot produces:

- Per-stage wall time and throughput distributions.
- Peak CPU, RAM, local storage, NFS throughput, and GPU memory.
- Raw, render, intermediate, and final output byte distributions.
- End-to-end success and categorized failure rates.
- Full-dataset storage and completion-time projections.
- FP16 versus FP32 latent parity results.

Production does not start until the pilot report is reviewed.

### Production Gate

Production starts with one 5,000-asset logical shard. Source-wide scheduling is enabled only after that shard passes the same audit as the pilot.

## Validation Rules

### Render Validation

- Exactly eight decodable 512 by 512 RGBA PNG files.
- Non-empty and bounded alpha mask with no invalid pixels.
- Exactly eight transform frames with finite values.
- Camera matrices are invertible and camera FOV lies in the configured range.
- Re-rendering the same SHA with the same policy produces identical camera metadata.

### Latent Validation

- Both anchor files exist and open for every required output family and resolution.
- Features and coordinates contain no NaN or infinity.
- Coordinates are integral, unique where required, and within the expected encoder grid.
- Token counts fit the corresponding training configuration limit.
- Scale metadata exists, is finite, and matches the target anchor.
- The official decoder can load sampled outputs without schema or device errors.

FP16 storage is enabled only when coordinates match exactly, no values become non-finite, the latent absolute-error p99 is at most 0.01, and sampled decode metrics degrade by at most 0.1% relative to FP32. Otherwise production uses FP32 and storage projections are recalculated before continuing.

### Pack Validation

- Every manifest member exists exactly once and has the expected size and SHA-256.
- No absolute path, parent traversal, device file, or symlink is allowed.
- Whole-pack checksum matches the published manifest.
- Extraction into a clean directory reconstructs the expected Pixal3D dataset layout.
- A dataset-loader smoke test reads at least one sample per output family from the extracted pack.

## Failure and Recovery Policy

An individual asset retries at most three times with captured stderr and a categorized error. It then becomes `quarantined`, allowing the shard to continue. Infrastructure errors do not consume all assets' retry budgets.

The current shard stops and reports immediately for:

- Authentication or source-wide download failure.
- Missing or incompatible encoder checkpoint.
- Blender failure to select the assigned OptiX GPU.
- Manifest or checksum corruption.
- Repeated OOM after one automatic concurrency reduction.
- Camera or latent schema errors affecting more than 5% of the recent 500 attempts.
- End-to-end success below 90% over the recent 500 terminal assets.
- Any resource hard limit or unrecovered soft-limit violation.

An immediate report includes the affected source and shard, failed command, categorized error, last five minutes of resource telemetry, completed output counts, safe resume point, and available recovery choices. The orchestrator does not silently change dataset scope, camera policy, output dtype, encoder checkpoint, or retention policy. A decision that changes any of those items waits for user confirmation.

All outputs are written to temporary sibling paths, validated, and atomically renamed. Resume logic checks actual content before accepting existing files, merges valid records idempotently, and regenerates only missing or corrupt outputs.

## Capacity Model

The current optimized estimate for all final training-ready data is 8 to 12 TiB, with a planning ceiling of approximately 15 TiB including variance and operational headroom. This assumes eight 512-pixel PNG renders, two anchors, uint8 coordinates, and FP16 shape/PBR features after parity qualification.

The estimate is not used as an admission guarantee. Pilot byte distributions replace it before production. If FP16 qualification fails or observed PNG/latent sizes project beyond the 16 TiB soft limit, production pauses for an explicit storage decision rather than silently consuming reserve space.

## Acceptance Criteria

The preprocessing project is complete when:

- Toolkit unit and integration tests pass in the `pixal3d` environment.
- Smoke and pilot gates pass and their reports are retained.
- Every source metadata row resolves to one canonical asset whose terminal state is `complete` or `quarantined`, or is explicitly marked as a duplicate of that canonical asset.
- Every published pack and archive has verified manifests and checksums.
- The source-level and global end-to-end success rates are at least 90%; exact retained counts and failure categories are reported.
- The global split audit reports no SHA overlap.
- Random decode and dataset-loader checks pass for all output families and resolutions.
- Final data2, data3, and local storage usage remains within the defined limits.
- An immutable training handoff manifest can materialize Stage 1, Stage 2, or Stage 3 shards independently under `/root/node17/data/pixal3d/train`.

## Operational Handoff

Fine-tuning starts only after preprocessing is stopped and the final audit passes. During training, data movement is one-way prefetch from `/root/data2/pixal3d/prepared` to `/root/node17/data/pixal3d/train/next`. Training reads only `/root/node17/data/pixal3d/train/active`; raw archival, Blender rendering, voxelization, and latent encoding remain disabled.
