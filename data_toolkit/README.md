# Pixal3D Multi-View Preprocessing Runbook

This toolkit builds the canonical Pixal3D training and evaluation registry,
renders eight deterministic 512x512 RGBA condition views per asset, and creates
view-aligned sparse, shape, and PBR outputs. Shape, PBR, and SS training packs
contain the two configured anchor views, `view00` and `view01`, at shape/PBR
resolutions 256, 512, and 1024 plus SS resolution 64.

The legacy leaf scripts remain available for development, but production work
must use `python -m data_toolkit.pipeline.cli`. The orchestrator freezes exact
SHA lists, validates every output, publishes checksummed packs and raw archives,
and resumes only from validated checkpoints.

## Fixed Paths

- Config: `data_toolkit/configs/multiview_preprocess.yaml`
- Project data and control: `/root/data2/pixal3d`
- Verified raw archive: `/root/data3/pixal3d`
- Local scratch and training materialization: `/root/pixal3d-data`
- Canonical registry: `/root/data2/pixal3d/control/assets.parquet`
- Frozen batches: `/root/data2/pixal3d/control/shards/<source>/<shard>/`
- Prepared packs: `/root/data2/pixal3d/prepared/`
- Raw archives: `/root/data3/pixal3d/archive/raw/`
- Resource telemetry: `/root/data2/pixal3d/control/telemetry/resources.jsonl`
- Gate reports: `/root/data2/pixal3d/control/reports/gates/`
- Escalations: `/root/data2/pixal3d/control/reports/escalations/`

Do not redirect one root beneath another. Production accounting assumes three
distinct filesystems and reconciles their project totals only at mutating
command boundaries, never in the five-second sampler.

## Environment And Tests

Run every command from the repository root in the `pixal3d` environment.

```bash
conda run -n pixal3d python -m compileall -q data_toolkit
conda run -n pixal3d python -m pytest tests/data_toolkit -v
```

For a focused CLI/report check:

```bash
conda run -n pixal3d python -m pytest \
  tests/data_toolkit/test_cli.py tests/data_toolkit/test_reporting.py -v
```

Set a shell helper for the operator commands:

```bash
CLI="conda run -n pixal3d python -m data_toolkit.pipeline.cli"
CONFIG="data_toolkit/configs/multiview_preprocess.yaml"
```

## Source Access

The configured training source order is ObjaverseXL Sketchfab, ObjaverseXL
GitHub, ABO, HSSD, and 3D-FUTURE. Toys4K is evaluation-only and must never
appear in training packs.

Authenticate for the gated HSSD dataset before downloading:

```bash
conda run -n pixal3d huggingface-cli login
```

Place the two manual archives at these exact paths:

```text
/root/data2/pixal3d/raw/3D-FUTURE/3D-FUTURE-model.zip
/root/data2/pixal3d/raw/Toys4k/toys4k_blend_files.zip
```

Run access preflight and stop on exit `2`:

```bash
$CLI preflight --config "$CONFIG"
```

The command checks HSSD access and the manual archive paths. It does not begin
downloads when a source is blocked.

Build each canonical source metadata file before `registry`. These adapter calls
may access the configured public metadata endpoints; registry construction
itself reads only the resulting local CSV files.

```bash
conda run -n pixal3d python data_toolkit/build_metadata.py ObjaverseXL \
  --source sketchfab \
  --root /root/data2/pixal3d/control/metadata/ObjaverseXL_sketchfab
conda run -n pixal3d python data_toolkit/build_metadata.py ObjaverseXL \
  --source github \
  --root /root/data2/pixal3d/control/metadata/ObjaverseXL_github
conda run -n pixal3d python data_toolkit/build_metadata.py ABO \
  --root /root/data2/pixal3d/control/metadata/ABO
conda run -n pixal3d python data_toolkit/build_metadata.py HSSD \
  --root /root/data2/pixal3d/control/metadata/HSSD
conda run -n pixal3d python data_toolkit/build_metadata.py 3D-FUTURE \
  --root /root/data2/pixal3d/control/metadata/3D-FUTURE
conda run -n pixal3d python data_toolkit/build_metadata.py Toys4k \
  --root /root/data2/pixal3d/control/metadata/Toys4k
$CLI registry --config "$CONFIG"
```

Registry publication is source-order deterministic, globally deduplicated,
checksummed, bound to the config hash, and accompanied by compatible per-source
metadata. Treat a changed count or checksum on an unchanged input as a stop.

## Hardware And Local-Space Preflight

Hardware and local-space preflight must finish before the first actual smoke
asset is downloaded. Test each of the seven GPUs with one distinct visible
OptiX device, a non-empty cube render, and no CPU fallback. Sequentially write,
fsync, read, and delete one bounded 10 GiB fixture on local, data2, and data3;
record throughput, free bytes before/after, and successful fixture removal.

The machine-produced candidate belongs at:

```text
/root/data2/pixal3d/control/report_inputs/hardware.json
```

It must include the active config hash, exact typed CUDA/torch/Blender/OptiX
evidence, seven isolated GPU cube-render checks, the three 10 GiB storage
measurements, and per-asset local-byte samples for every training source. The
report command derives p95 sizing, throughput, free-space floors, and the pass
decision; the candidate cannot supply those fields. Publish it only through the
validator:

```bash
$CLI report --config "$CONFIG" --hardware-check
```

The validated report is
`/root/data2/pixal3d/control/reports/hardware.json`. Smoke and pilot capacity
planning use its conservative p95 sizing until a passed pilot report supplies
measured values. Do not begin smoke if local free space is below the greater of
15% of the filesystem or 120 GiB.

## CLI Contract

A plan with no source and shard is a read-only static DAG inspection. It creates
no configured directory and never freezes a batch.

```bash
$CLI plan --config "$CONFIG" --gate smoke
```

A capacity plan requires both source and shard. `--count` is optional and must
be positive. This remains read-only and prints planned batch sizes.

```bash
$CLI plan --config "$CONFIG" --gate smoke \
  --source ABO --shard ABO-00000 --count 20
$CLI plan --config "$CONFIG" --gate production \
  --source ObjaverseXL_sketchfab \
  --shard ObjaverseXL_sketchfab-00000
```

Only `run` may freeze missing batches. `resume` and `audit` require an existing,
checksummed frozen batch manifest and never replan from current free space.

```bash
$CLI run --config "$CONFIG" --gate smoke \
  --source ABO --shard ABO-00000 --count 20
$CLI resume --config "$CONFIG" \
  --gate smoke \
  --source ABO --shard ABO-00000
$CLI audit --config "$CONFIG" \
  --gate smoke \
  --source ABO --shard ABO-00000
```

Production runs cannot use `--count`. A shard must match its configured source.
Invalid source/shard/count/gate combinations exit `2` before runtime providers
are initialized.

## Gate Order

Run gates in this order: code tests, access preflight, hardware/local-space
preflight, registry, smoke, pilot, first production shard, then source-wide
production. Never skip an audit or overlap raw archive publication with another
CPU/GPU-heavy phase.

### Smoke

Select 20 assets from each training source, using a separate canonical shard
scope per source. Capacity-plan first, then run with the same count. The complete
smoke totals 100 assets and must exercise all eight views and all eight pack
families.

```bash
$CLI plan --config "$CONFIG" --gate smoke \
  --source ObjaverseXL_sketchfab \
  --shard ObjaverseXL_sketchfab-00000 --count 20
$CLI run --config "$CONFIG" --gate smoke \
  --source ObjaverseXL_sketchfab \
  --shard ObjaverseXL_sketchfab-00000 --count 20
```

Repeat for ObjaverseXL GitHub, ABO, HSSD, and 3D-FUTURE. Audit every frozen
scope. Require zero infrastructure, adapter, checkpoint, OptiX, checksum, and
schema failures. Write held per-asset measurements and FP16 results under
`control/report_evidence/smoke/`, plus scope-bound telemetry in
`telemetry.jsonl`. Record those three artifact SHA-256 values in the fresh,
config-bound manifest at `control/report_inputs/smoke.json`; the manifest must
not contain aggregate pass fields. The report command reopens the frozen
scopes, registry, packs, archives, hardware report, telemetry, and measurements
and derives both JSON and Markdown:

```bash
$CLI report --config "$CONFIG" --gate smoke
```

Do not start pilot unless the smoke decision is `passed` with the active config
hash.

### Pilot

Run 1,000 assets stratified across source, extension, raw size, mesh complexity,
material count, and alpha usage. Use bounded per-source counts that sum to 1,000;
plan and run each source scope explicitly. Pilot admission automatically checks
the passed smoke report.

Measure source counts, failure categories, stage throughput, output bytes,
resource peaks, checksums, split overlap, capacity, and training handoff. FP16
qualification requires at least 32 decoded assets for shape and PBR at every
resolution, exact coordinates, zero non-finite values, absolute-error p99 at
most `0.01`, and decode degradation at most `0.1%`. Otherwise retain FP32 and
recalculate capacity.

Publish the same checksum-bound evidence set under
`control/report_evidence/pilot/` only after all pilot audits complete:

```bash
$CLI report --config "$CONFIG" --gate pilot
```

Stop for explicit operator review if projected data2 use exceeds 16 TiB,
projected data3 archive use exceeds 26 TiB, success is below 90%, schema failure
exceeds 5%, a source has abnormal failures, FP16 qualification fails, or any
checksum/split/hardware audit fails.

### Production

Production admission requires both passed smoke and pilot reports with the
active config hash. Begin with one full 5,000-asset logical shard:

```bash
$CLI plan --config "$CONFIG" --gate production \
  --source ObjaverseXL_sketchfab \
  --shard ObjaverseXL_sketchfab-00000
$CLI run --config "$CONFIG" --gate production \
  --source ObjaverseXL_sketchfab \
  --shard ObjaverseXL_sketchfab-00000
$CLI resume --config "$CONFIG" \
  --gate production \
  --source ObjaverseXL_sketchfab \
  --shard ObjaverseXL_sketchfab-00000
$CLI audit --config "$CONFIG" \
  --gate production \
  --source ObjaverseXL_sketchfab \
  --shard ObjaverseXL_sketchfab-00000
```

Audit before scheduling the next shard. Finish sources in fixed order:
ObjaverseXL Sketchfab, ObjaverseXL GitHub, ABO, HSSD, then 3D-FUTURE. Prepare
Toys4K separately as evaluation-only and audit that none of its SHAs occur in a
training split or pack.

## Thresholds And Exit Codes

The five-second Task 8 sampler reads cached project totals and uses a bounded
five-second `nvidia-smi` timeout. It never walks large roots. Telemetry fsyncs at
30-second intervals; accounting reconciles by walking roots only at mutating
batch/shard boundaries.

- Exit `0`: success.
- Exit `2`: operator/provider/infrastructure block.
- Exit `3`: resource stop.
- Exit `4`: data-quality stop.
- Pause after CPU above 80%, load above 72, or I/O wait above 10% for two minutes.
- Pause immediately below 96 GiB available RAM, on swap-in, or at data2/data3 soft limits.
- Stop after CPU above 90% for five minutes or immediately below 64 GiB RAM.
- Stop below local `max(15%, 120 GiB)`, below 2 TiB free on data2, or below 4 TiB free on data3.
- Pause at 16 TiB data2 project use; stop at 18 TiB. Pause at 26 TiB data3 project use.
- Resume only after five uninterrupted stable minutes.
- Stop when recent-500 end-to-end failure is strictly above 10% or schema failure is strictly above 5%.

An asset receives at most three total command launches. Ordinary sub-threshold
asset failures quarantine the asset; authentication, provider, checksum,
checkpoint, process-control, telemetry, accounting, and resource failures stop
the shard immediately.

## Pause, Stop, And Escalation

On pause, do not start another rank or shard. The orchestrator pauses the stable
supervised process group, samples every five seconds, and resumes only after the
full recovery interval. On a hard stop it resumes a paused group before TERM,
uses bounded reap/KILL handling, checkpoints, and exits.

Read the escalation JSON under:

```text
/root/data2/pixal3d/control/reports/escalations/<source>/<shard>/<command>.json
```

It records category, primary reason, last five telemetry minutes, terminal asset
counts, safe resume command, recovery choices, and any report persistence
errors. Escalate rather than changing source scope, camera policy, dtype,
checkpoint, or raw-retention rules. Use the exact `safe_resume_command` only
after the failed dependency or capacity condition is corrected.

## Archive Verification And Cleanup

Every work batch publishes exactly eight prepared families: `common`, `SS-64`,
`shape-256`, `shape-512`, `shape-1024`, `PBR-256`, `PBR-512`, and `PBR-1024`.
The logical shard index binds all pack and manifest checksums. The raw tar under
data3 is separately bound to frozen SHAs, member sizes and SHA-256 values, tool
commit, config hash, completed count, and quarantined count.

Run `audit` before manual extraction or cleanup. Cleanup order is fixed:

1. Validate terminal outputs.
2. Publish and verify all eight prepared packs.
3. Publish and verify the raw archive.
4. Delete a data2 raw file only when no other frozen unarchived batch references it.
5. Remove local output, work, and staged-source scratch.

ObjaverseXL GitHub repository ZIPs are shared. Never delete one manually: the
validated complete-source reference index preserves it until every canonical
reference, including future unfrozen shards, has a verified raw archive. A
missing/corrupt checksum-bound training registry, source-input digest, reference
index, frozen manifest, pack, or archive blocks deletion. Qualification gates
never delete production raw data.

Do not use broad `rm -rf` against `raw`, `prepared`, `archive`, `control`, or
`preprocess/active`. After a successful audit, stale training extraction staging
directories may be removed individually.

## Independent Stage Extraction

Audit the source/shard first. Materialize each training stage into its own empty
directory so no stage silently inherits an unrelated family. For a verified
`<source>/<shard>/<batch>.tar`, use:

```bash
SOURCE=ObjaverseXL_sketchfab
SHARD=ObjaverseXL_sketchfab-00000
BATCH=batch000
PREPARED=/root/data2/pixal3d/prepared
TRAIN=/root/pixal3d-data/train

mkdir -p "$TRAIN/stage1/active"
tar -xf "$PREPARED/common/$SOURCE/$SHARD/$BATCH.tar" \
  -C "$TRAIN/stage1/active"
tar -xf "$PREPARED/ss/64/$SOURCE/$SHARD/$BATCH.tar" \
  -C "$TRAIN/stage1/active"

mkdir -p "$TRAIN/stage2/active"
tar -xf "$PREPARED/common/$SOURCE/$SHARD/$BATCH.tar" \
  -C "$TRAIN/stage2/active"
for RES in 256 512 1024; do
  tar -xf "$PREPARED/shape/$RES/$SOURCE/$SHARD/$BATCH.tar" \
    -C "$TRAIN/stage2/active"
done

mkdir -p "$TRAIN/stage3/active"
tar -xf "$PREPARED/common/$SOURCE/$SHARD/$BATCH.tar" \
  -C "$TRAIN/stage3/active"
for RES in 256 512 1024; do
  tar -xf "$PREPARED/shape/$RES/$SOURCE/$SHARD/$BATCH.tar" \
    -C "$TRAIN/stage3/active"
  tar -xf "$PREPARED/pbr/$RES/$SOURCE/$SHARD/$BATCH.tar" \
    -C "$TRAIN/stage3/active"
done
```

Stage 1 is `common + SS-64`; Stage 2 is `common + shape` at all three
resolutions; Stage 3 is `common + shape + PBR` at all three resolutions. Load at
least one sample per resolution and anchor before promoting a materialized
directory. The final immutable handoff at
`/root/data2/pixal3d/control/splits/training_handoff.json` records config hash,
training/evaluation registry checksums, frozen scopes, pack/archive checksums,
train/validation/evaluation identities, resolutions, anchors, and these path
mappings. Stop all preprocessing processes before fine-tuning reads
`train/active`.
