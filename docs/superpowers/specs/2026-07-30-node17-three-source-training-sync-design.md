# Node17 Three-Source Training Sync Design

## Goal

Make Node17 match the current, training-authoritative
`/home/youngwoo/Pixal3D-multiview` checkout on Node16 and prepare ABO,
3D-FUTURE, and HSSD as one immediately usable training input on Node17.

The completed result must support all four multi-view fine-tuning stages:

- `ss64`;
- `shape512`;
- `shape1024`;
- `pbr1024`.

This work synchronizes code, configurations, and already materialized training
data. It does not re-run raw HSSD preprocessing, start a GPU training process,
create a W&B run, or change the multi-view model architecture.

## Source of truth

Node16's `/home/youngwoo/Pixal3D-multiview` checkout is authoritative for
current model, dataset, trainer, rendering, test, and fine-tuning
configuration behavior. Node17's clean
`feature/multiview-model-extension` worktree is the integration target.
Node17's dirty `/root/dev/Pixal3D` checkout must not be modified.

Only project source is synchronized. Cache directories, logs, generated
artifacts, inaccessible root-owned files, local scratch files, and unrelated
documents are excluded.

The observed Node16 delta includes:

- the four multi-view fine-tuning configurations;
- six multi-view dataset implementations;
- `BasicTrainer`;
- the shape and PBR VAE trainers;
- render utilities;
- four focused multi-view regression test files.

The synchronized dataset and rendering code uses the installed `utils3d`
interface:

```python
utils3d.torch.intrinsics_from_fov(fov_x=fov, fov_y=fov)
```

The obsolete `intrinsics_from_fov_xy` interface must not remain in these
paths.

## Authoritative fine-tuning policy

Node16's current configuration values are intentional and override older
design documents and stale test expectations:

| Stage | Batch/GPU | Split | Six-GPU global batch | Steps |
|---|---:|---:|---:|---:|
| `ss64` | 8 | 4 | 48 | 20,000 |
| `shape512` | 8 | 4 | 48 | 20,000 |
| `shape1024` | 2 | 1 | 12 | 20,000 |
| `pbr1024` | 2 | 1 | 12 | 20,000 |

All four configurations use:

```text
trainer.args.num_workers = 2
trainer.args.i_print = 10
trainer.args.i_log = 10
trainer.args.i_sample = 1000
trainer.args.i_save = 1000
trainer.args.max_checkpoints = 3
```

`ss64` additionally uses:

```text
trainer.args.snapshot_dataset_on_start = false
```

Other stage behavior is preserved from Node16. In particular, this sync does
not independently tune batch size, split, learning rate, attention backend,
optimizer, or conditioning-view policy.

The focused configuration regression test is updated to assert
`i_sample=1000`, `i_save=1000`, and `max_checkpoints=3`. The current stale
expectations of `2000`, `2000`, and `5` are not retained.

## Code synchronization architecture

### Integration boundary

The Node16 checkout is copied into a temporary comparison area. A
checksum-based diff determines the exact source changes relative to the clean
Node17 worktree. Changes are then applied to the worktree as a reviewable Git
patch.

The default synchronization scope is limited to the observed changed files.
If a changed file depends on another Node16 file, that dependency is included
only after a direct import, test, or runtime reference proves it is required.
Unrelated Node16 changes do not enter the patch.

### Runtime path portability

Committed fine-tuning configs retain Node16's authoritative training
semantics. Node17-specific runtime copies are generated outside the Git
checkout under:

```text
/root/node17/data/pixal3d/train/runtime-configs
```

The runtime copies change only machine-local absolute storage prefixes:

```text
/file2/youngwoo/pixal3d  -> /root/data2/pixal3d
/file3/youngwoo/pixal3d  -> /root/data3/pixal3d
```

Every non-path field must be byte-for-byte equivalent after canonical JSON
normalization. The generated runtime configs are validated before publication
and are the configs used by the Node17 loader preflight and handoff commands.

## Data architecture

### Existing sources

Node17's already published ABO and 3D-FUTURE source directories are reused
in place:

```text
/root/node17/data/pixal3d/train/production/abo
/root/node17/data/pixal3d/train/production/3d-future
```

They are not copied, rematerialized, rewritten, deleted, or subjected to
another full population scan. Reuse validation checks the already published
immutable report, handoff, source manifest, and stage materialization
digests, plus the existence of their bound roots. Their current source
manifests and stage scopes remain the inputs to the new combined publication.

### HSSD transfer

The completed Node16 HSSD materialization is the transfer source:

```text
/home/youngwoo/data/pixal3d/train/production/hssd
```

Node17 receives it into a hidden, uniquely named staging directory beneath:

```text
/root/node17/data/pixal3d/train/production
```

Before transfer, the workflow records source file counts and bytes, confirms
at least the measured HSSD size plus a safety margin is free on Node17 local
storage, and confirms that no canonical Node17 HSSD publication already
conflicts with the operation.

The transfer is resumable. It preserves a partial staging directory after a
failure for inspection or continuation. It does not write directly into the
canonical `hssd` path.

After transfer, the workflow verifies:

- all four stage directories and evidence files are present;
- source and target file counts and aggregate bytes agree;
- a checksum verification pass reports no changed or missing regular file;
- no symlink, empty unexpected file, or unsafe path was introduced.

### Evidence rebasing and strict validation

The copied HSSD evidence contains Node16 absolute paths. A deterministic
rebasing step creates Node17-native evidence in staging:

```text
/home/youngwoo/data/pixal3d/train/production/hssd
  -> /root/node17/data/pixal3d/train/production/hssd

/file2/youngwoo/pixal3d
  -> /root/data2/pixal3d
```

Rebasing changes path fields only. Counts, ordered asset scopes, exclusions,
source-index identities, manifest and pack digests, configuration hashes, and
tool revision evidence must remain unchanged. Both the original and rebased
evidence digests are recorded in the transfer report.

The entire staged HSSD population then passes the existing strict source
preflight on Node17. A boundary-only or sampled check is insufficient for
publication. The preflight covers every selected pack, every materialized
asset, all four stage structures, coordinate/scale contracts, and the
standalone HSSD `training_data.json`.

Only after all checks pass is staging atomically renamed to:

```text
/root/node17/data/pixal3d/train/production/hssd
```

Publication is create-only. A pre-existing canonical HSSD directory is reused
only if its full evidence and content already validate; otherwise the
workflow aborts without replacing it.

### Three-source publication

The canonical source order is:

```text
ABO
3D-FUTURE
HSSD
```

The combined, proportional, unweighted training input is published at:

```text
/root/node17/data/pixal3d/train/production/
  abo-3d-future-hssd/training_data.json
```

The existing ABO + 3D-FUTURE bundle remains unchanged. The new publisher
requires pairwise-disjoint source/asset identities and pins each source
publication by path and SHA-256. It performs no oversampling and adds no
source-specific loss weighting.

## Verification

### Code verification

Run the synchronized focused tests first:

- fine-tuning configuration policy;
- projection geometry and installed `utils3d` API;
- fresh-start/snapshot trainer behavior;
- W&B multi-view image logging;
- HSSD materialization and source-profile behavior;
- one-source and three-source training-data resolution;
- Node17 runtime-path config generation.

Then run the complete `tests/multiview` suite in Node17's
`/opt/conda/envs/pixal3d` environment. Test collection and execution must use
the clean worktree and must not initialize CUDA for data-readiness checks.

### Production-data verification

For HSSD standalone and the three-source bundle, every stage must prove:

- configured Dataset construction succeeds with CUDA hidden;
- the first and last final asset of each source direct-load;
- manifest order, source, root, count, and digest match;
- a zero-worker DataLoader executes the production `collate_fn`;
- the combined collate includes an actual ABO, 3D-FUTURE, and HSSD sample;
- CUDA remains unavailable and uninitialized throughout preflight.

The final report includes per-source/per-stage counts, exclusions, manifest
digests, materialization digests, runtime-config digests, publication paths,
elapsed validation time, and copy-paste Node17 launch commands for all four
models.

### Training-command validation

Each of the four generated launch commands uses:

```text
--training_data \
  /root/node17/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json
```

Existing resolver and entrypoint regression tests exercise the equivalent
arguments and prove that they resolve the correct stage roots and six-GPU
batch policy. Command validation does not load model weights, initialize
distributed GPU workers, contact W&B, or start training. No new generic
`train.py` dry-run interface is added solely for this handoff.

## Failure handling and safety

- HSSD transfer and validation failures leave canonical Node17 publications
  untouched.
- Partial transfers remain in an explicitly reported staging path.
- Insufficient disk space aborts before copying.
- A changed source index, digest mismatch, path-rebase mismatch, duplicate
  source/asset identity, loader failure, or CUDA initialization aborts
  publication.
- Existing ABO, 3D-FUTURE, checkpoints, logs, active preprocess workers, and
  training processes are never modified.
- No operation targets a broad workspace, home directory, or unresolved
  recursive path.
- The Node16 source checkout is read-only for synchronization.

## Acceptance criteria

The task is complete when:

1. the clean Node17 worktree contains a reviewed Git commit representing the
   required Node16 code and test delta;
2. focused and full multi-view tests pass with the authoritative
   `1000/1000/3` checkpoint and snapshot policy;
3. Node17 runtime configs preserve every non-path Node16 setting;
4. canonical standalone HSSD publication passes full strict preflight;
5. the existing ABO and 3D-FUTURE publication digests and bound roots pass
   immutable-evidence reuse validation without another full population scan;
6. the canonical three-source `training_data.json` is published create-only;
7. all four configured Dataset/DataLoader preflights pass across all three
   sources without CUDA;
8. all four training commands resolve the combined input and correct config
   without starting training; and
9. a final evidence report makes the code revision, data digests, paths,
   counts, and commands reproducible.
