# ABO Production Valid-Subset Materialization Design

## Goal

Materialize the validated portion of the completed ABO production packs into
four stage-isolated training roots, run a strict CPU-only preflight over every
admitted asset and both target anchors, and publish an immutable ABO-specific
training handoff. This work prepares training inputs only; it does not start
fine-tuning.

## Accepted Quality-Gate Exception

The user explicitly accepts the currently valid ABO subset even though the
source-level 90% production gate is not met.

- Frozen ABO scope: 4,485 assets.
- Global valid/common scope: 3,660 assets (81.605%).
- Global quarantine: 825 assets.
- Shape-512 has 29 additional family exclusions, leaving 3,631 assets
  (80.959%).
- The 90% threshold would require at least 4,037 valid assets.

The materializer must never silently describe this subset as a normal
90%-passing production release. Every evidence and handoff document must record
the exception, the frozen count, quarantine count, family exclusions, and exact
stage counts.

## Inputs

Only immutable production packs and their index may be consumed:

```text
/root/data2/pixal3d/prepared/index/ABO/ABO-00000.json
/root/data2/pixal3d/prepared/common/ABO/ABO-00000/
/root/data2/pixal3d/prepared/ss/64/ABO/ABO-00000/
/root/data2/pixal3d/prepared/shape/512/ABO/ABO-00000/
/root/data2/pixal3d/prepared/shape/1024/ABO/ABO-00000/
/root/data2/pixal3d/prepared/pbr/1024/ABO/ABO-00000/
```

Active preprocessing scratch trees under either node must not be read.

The production index must contain 18 batches (`batch000` through `batch017`)
and all eight published families per batch. The materializer uses only the five
families required by the multi-view baseline:

- `common`
- `SS-64`
- `shape-512`
- `shape-1024`
- `PBR-1024`

Before extraction, every referenced pack and manifest must be a non-symlink
regular non-empty file. Manifest identity, production gate, batch, family,
counts, recorded manifest digest, and recorded pack digest must match the
index. Full pack verification uses the existing
`data_toolkit.pipeline.packing.verify_pack` implementation.

## Output Layout

The fixed local output root is:

```text
/root/node17/data/pixal3d/train/production/abo/
├── ss64/active/
├── shape512/active/
├── shape1024/active/
├── pbr1024/active/
└── training_data.json
```

Each stage is isolated and contains only assets admitted to that stage:

| Stage | Families | Exact asset count |
|---|---|---:|
| `ss64` | `common` + `SS-64` | 3,660 |
| `shape512` | `common` + `shape-512` | 3,631 |
| `shape1024` | `common` + `shape-1024` | 3,660 |
| `pbr1024` | `common` + `shape-1024` + `PBR-1024` | 3,660 |

The component directories use the existing pilot-compatible names:

```text
renders_cond/
ss_latents/ss_enc_conv3d_16l8_fp16_64_view/
shape_latents/shape_enc_next_dc_f16c32_fp16_512_view/
shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view/
pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix/
```

Each component root receives a deterministic `metadata.csv` whose rows are
sorted by SHA-256. It contains only the exact admitted stage scope and the
explicit boolean columns required by the corresponding loader:

- render: `cond_rendered=True`
- SS: `ss_latent_view_scale00_encoded=True` and
  `ss_latent_view_scale01_encoded=True`
- Shape: `shape_latent_view00_encoded=True` and
  `shape_latent_view01_encoded=True`
- PBR: `pbr_latent_view00_encoded=True` and
  `pbr_latent_view01_encoded=True`

Each `active` directory also contains `materialization.json` with the source
index digest, verified pack/manifest digests, accepted exception, exact asset
scope digest, family tool commits, stage count, creation time, and output
paths.

## Atomic and Non-Destructive Publication

For each stage, extraction occurs in a hidden temporary sibling directory on
the same local filesystem. Only manifest-listed regular members belonging to
the stage asset intersection are extracted. Paths must be resolved beneath the
temporary root; links and non-regular members are rejected.

The materializer verifies every selected member's size and SHA-256 while
copying. It writes metadata and evidence only after extraction succeeds, then
publishes with `os.replace`.

An existing `stage/active` is never overwritten. A rerun must fail before
extracting that stage and report the exact existing path. On any failure, the
temporary stage directory is removed while previously published stages remain
unchanged and auditable.

## Strict Full Preflight

Preflight reads only the published stage roots and bypasses
`StandardDatasetBase.__getitem__` retry behavior. Failure of any admitted
asset fails the stage; no fallback sample is selected.

### Metadata and Scope

- Required component roots and `metadata.csv` files exist.
- Metadata columns exist exactly as required and every admitted flag is the
  boolean value `True`, not merely non-null.
- Component SHA sets equal the materialization stage scope.
- No unlisted asset directory is admitted.
- Dataset construction with the corresponding fine-tuning config yields the
  exact stage count.

### Render and Camera

For every asset:

- exactly `000.png` through `007.png` and one `transforms.json`;
- every image is a non-empty RGBA PNG with source resolution 512×512;
- exactly eight frames in matching order;
- safe relative `file_path` values;
- finite positive `camera_angle_x`;
- finite 4×4 camera-to-world matrices with finite positive derived camera
  distance and an invertible rotation block.

### Target Latents and Scale

For both `view00` and `view01` of every admitted asset:

- the NPZ and matching `_scale.json` exist and are regular non-empty files;
- every required NPZ key is present and all numeric values are finite;
- `total_scale` exists and remains finite and strictly positive after float32
  conversion;
- SS `z` is float-compatible with exact shape `[8, 16, 16, 16]`;
- Shape/PBR `coords` has shape `[N, 3]`, is integral, unique, and lies in the
  stage grid;
- Shape/PBR `feats` has shape `[N, 32]`, row count `N`, and finite values;
- Shape-512 has `N <= 8,192`;
- Shape-1024 and PBR-1024 have `N <= 32,768`;
- PBR and Shape coordinates are exactly equal for the same asset and anchor.
- PBR and Shape `total_scale` values are exactly equal after float32
  conversion for the same asset and anchor.

### Direct Loader Contract

For every admitted asset in every stage, preflight forces anchor 0 and anchor
1 separately and calls the real dataset `get_instance` method directly.

It requires:

- selected `view_idx` equals the forced anchor;
- `view_indices` is an eight-element permutation with the anchor first;
- `cond` is finite float32 with shape `[8, 3, H, W]`, where H/W are the
  config image size;
- `camera_angle_x` and camera distance are finite with shape `[8]`;
- `transform_matrix` is finite with shape `[8, 4, 4]`;
- `mesh_scale` is a finite positive float32 scalar;
- stage target tensors satisfy their loader contract;
- PBR loader coordinate equality completes without assertion.

The command is CPU-only and defaults to one worker. Production execution uses
`nice` and `ionice` so it does not claim a GPU or increase preprocessing
parallelism.

## Handoff Publication

Handoff is published only after all four stage preflights pass.

The immutable shared artifacts are:

```text
/root/data2/pixal3d/control/splits/ABO/ABO-00000-valid-subset-handoff.json
/root/data2/pixal3d/control/reports/gates/ABO/ABO-00000-valid-subset.json
```

The local convenience manifest is:

```text
/root/node17/data/pixal3d/train/production/abo/training_data.json
```

All three documents include:

- schema version and creation time;
- `source=ABO`, `shard_id=ABO-00000`;
- source index path and digest;
- `acceptance_mode=valid_subset_user_waiver`;
- frozen, quarantine, family-exclusion, and stage counts;
- exact stage roots and loader `data_dir` objects;
- materialization evidence digests;
- preflight result and report digest;
- all observed pack `tool_commit` values;
- a statement that this handoff authorizes training-input use only and does
  not claim the original 90% production gate passed.

The shared handoff/report use create-only semantics. If the path already
exists, an identical file is accepted; different content is rejected. The
local `training_data.json` is written atomically after the shared artifacts
succeed.

## Code Structure

Create two focused commands:

1. `scripts/materialize_multiview_production.py`
   - load and validate the production index;
   - compute stage intersections;
   - verify packs/manifests;
   - safely extract selected members;
   - write deterministic metadata and stage evidence;
   - atomically publish stage roots.

2. `scripts/preflight_multiview_production.py`
   - validate every materialized artifact and metadata row;
   - execute the direct loader contract for both anchors;
   - write the ABO-specific report, handoff, and `training_data.json` only
     after all checks pass.

Generic pilot helpers may be reused, but pilot constants and the behavior of
`scripts/materialize_multiview_pilot.py` must not change.

## Tests

Tests use small synthetic packs and temporary roots. No test reads production
packs, writes production paths, starts training, initializes CUDA, or performs
network access.

Required behavior coverage:

- exact family intersections and Shape-512 exclusion handling;
- index/manifest identity and digest rejection;
- non-regular, symlink, empty, unsafe-path, and member-hash rejection;
- selected extraction excludes non-admitted assets;
- deterministic exact metadata columns and sorted rows;
- non-overwrite and failed-extraction cleanup;
- full structural validation for render, cameras, both latent anchors, scales,
  token limits, and PBR/Shape coordinate equality;
- direct loader validation bypasses retry and checks both anchors;
- invalid `total_scale`, missing view, non-finite values, coordinate mismatch,
  and wrong metadata booleans fail with source/stage/asset/anchor context;
- handoff is withheld on any stage failure;
- successful handoff records the valid-subset waiver and exact counts;
- create-only shared artifacts accept identical content and reject different
  content.

Every new production behavior is implemented through a failing test followed
by the minimum implementation needed to pass it.

## Operational Sequence

1. Implement and review the materializer.
2. Implement and review strict preflight and handoff publication.
3. Run focused and full multi-view regression tests.
4. Run production materialization at low CPU/I/O priority without GPU use.
5. Run strict full preflight at low CPU/I/O priority.
6. Independently inspect output counts, reports, handoff digests, GPU process
   state, and worktree state.
7. Stop before fine-tuning and report whether the ABO valid subset is released
   for the next training step.
