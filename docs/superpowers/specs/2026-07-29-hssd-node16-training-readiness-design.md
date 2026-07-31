# HSSD Node16 Training Readiness

## Goal

Turn the completed HSSD multiview preprocessing output into the same strict,
training-ready source publication used for ABO and 3D-FUTURE. Publish both an
HSSD-only input and an unweighted ABO + 3D-FUTURE + HSSD input that can be
loaded and trained from Node16 local storage without Node17 path dependencies.

This work prepares and verifies training inputs. It does not start a GPU
training process or a W&B run.

## Observed HSSD production boundary

The source contract is pinned to the completed production state observed on
2026-07-29 UTC:

- source metadata contains 6,670 assets;
- `HSSD-00000` contains `batch000` through `batch019`;
- `HSSD-00001` contains `batch000` through `batch006`;
- all 27 work units are complete;
- the shared queue contains zero pending, leased, failed, or quarantined HSSD
  work units;
- the frozen population is identical across `common`, `SS-64`, `shape-512`,
  `shape-1024`, and `PBR-1024`;
- the frozen population is 6,670 assets;
- the pack-family intersection contains 6,078 candidates for each of `ss64`,
  `shape512`, `shape1024`, and `pbr1024`.

The 6,078 counts are pre-eligibility candidate counts. Final stage counts are
observed during materialization, after token, coordinate, and scale checks, and
are then pinned in immutable evidence.

## Source acceptance policy

HSSD's production pack population retains 6,078 of 6,670 frozen assets, an
observed source success rate of 91.124 percent. The original source-level
90-percent production threshold therefore passes; HSSD does not inherit the
ABO or 3D-FUTURE waiver claim. Failed preprocessing assets are not silently
restored. Only assets present in every family required by a stage enter that
stage's candidate scope.

The training eligibility policy remains unchanged:

- `shape512` permits at most 8,192 shape tokens per anchor;
- `shape1024` permits at most 32,768 shape tokens per anchor;
- `pbr1024` permits at most 32,768 shape and PBR tokens per anchor;
- PBR and shape coordinates must be exactly equal for each anchor;
- PBR and shape float32 scales must match with `rtol=0` and `atol=2e-7`;
- exclusions are recorded by source, stage, asset, and deterministic reason.

No model, optimizer, attention backend, batch, split, iteration, checkpoint,
snapshot, or sampling-objective setting changes as part of source
publication.

## Architecture

### Source profile

Add HSSD as a source-aware production profile alongside ABO and 3D-FUTURE.
The profile binds exactly two production indexes and their exact batch sets:

```text
/file2/youngwoo/pixal3d/prepared/index/HSSD/HSSD-00000.json
/file2/youngwoo/pixal3d/prepared/index/HSSD/HSSD-00001.json
```

The profile has:

```text
source = HSSD
expected_frozen = 6670
expected_candidate_stages =
  ss64: 6078
  shape512: 6078
  shape1024: 6078
  pbr1024: 6078
acceptance_mode = production_gate
original_90_percent_gate_passed = true
```

Source specs must be constructed from explicit storage roots rather than
assuming Node17's `/root/data2` or `/root/node17` paths. Existing Node17
defaults remain backward compatible.

### Node16-native materialization

Node16 reads verified production indexes, manifests, and packs from shared
data2 and materializes all three training sources onto its local SSD:

```text
/home/youngwoo/data/pixal3d/train/production/abo
/home/youngwoo/data/pixal3d/train/production/3d-future
/home/youngwoo/data/pixal3d/train/production/hssd
```

Each source has four create-only stage roots:

```text
ss64/active
shape512/active
shape1024/active
pbr1024/active
```

Running all three source materializations on Node16 creates a single
canonical-path evidence chain. Copying a Node17 `training_data.json` or
rewriting its absolute paths is not allowed.

### Source publication

After materialization, strict source preflight publishes an immutable report,
handoff, and `training_data.json` for each source. HSSD uses the multi-index
source schema because it contains two pinned source indexes.

The standalone HSSD training input is:

```text
/home/youngwoo/data/pixal3d/train/production/hssd/training_data.json
```

It is independently loadable for all four model stages.

The training resolver must recognize a validated source `training_data.json`
as a one-source input. This is distinct from a combined manifest but uses the
same `--training_data` CLI. Exact-key schema dispatch prevents a source
publication from being confused with a combined publication.

### Combined bundles

Preserve the existing two-source ABO + 3D-FUTURE bundle and add a separate
three-source bundle. Source membership must be explicit per bundle rather than
changing the meaning of every existing combined manifest.

The new canonical source order is:

```text
ABO
3D-FUTURE
HSSD
```

The new combined input is:

```text
/home/youngwoo/data/pixal3d/train/production/
  abo-3d-future-hssd/training_data.json
```

The bundle performs proportional, unweighted concatenation. It does not
oversample a smaller source and does not add source-specific loss weights.
Asset SHA scopes must be pairwise disjoint across all three sources.

The manifest resolver, publisher, Dataset preflight, and training entrypoint
must accept both the existing two-source bundle and the new three-source
bundle.

### Node16 preparation workflow

Provide one CPU-only Node16 workflow that:

1. validates the selected code revision and storage roots;
2. confirms the exact HSSD source indexes and batch contract;
3. calculates required input pack bytes and verifies local free space before
   creating stage roots;
4. materializes or strictly verifies ABO, 3D-FUTURE, and HSSD;
5. runs source preflight and publishes each immutable source chain;
6. verifies HSSD as a standalone training input;
7. publishes the three-source combined manifest;
8. performs configured Dataset boundary loads and a real CPU-only DataLoader
   collate across all three sources for every stage;
9. emits final source/stage counts, digests, artifact paths, and copy-paste
   training commands.

The workflow runs with `CUDA_VISIBLE_DEVICES=""`. It must prove that CUDA was
unavailable and uninitialized before and after configured Dataset validation.
It does not initialize a model, trainer, W&B client, or GPU context.

## Node16 training configuration

Create Node16 runtime copies of the four reviewed production configs. The only
training-field override is:

```text
trainer.args.num_workers = 1
```

With six DDP ranks this creates six DataLoader workers. It does not change
batch size, batch split, global batch, optimizer steps, learning rate, or data
sampling.

The retained production policies are:

| Stage | Batch/GPU | Split | Six-GPU global batch | Steps | Save interval | Keep |
|---|---:|---:|---:|---:|---:|---:|
| `ss64` | 8 | 4 | 48 | 20,000 | 2,000 | 5 |
| `shape512` | 8 | 4 | 48 | 20,000 | 2,000 | 5 |
| `shape1024` | 2 | 1 | 12 | 20,000 | 2,000 | 5 |
| `pbr1024` | 2 | 1 | 12 | 20,000 | 2,000 | 5 |

Snapshots remain disabled with `i_sample=-1`.

Training uses:

```text
--training_data /home/youngwoo/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json
```

It does not require hand-written `--data_dir` JSON and does not reference a
Node17-local path.

## Validation contract

### Production evidence

For all selected HSSD packs:

- the index identity, source, shard, batch, family, and gate must match;
- manifest bytes must match the index-pinned SHA-256;
- pack bytes must match the manifest-pinned SHA-256;
- tar member paths must be safe and complete;
- tool commit and config hash evidence must remain pinned;
- source indexes must not change between materialization and publication.

### Materialized structure

For every final asset and stage:

- render conditioning contains exactly eight ordered RGBA 512×512 PNGs and a
  valid calibrated `transforms.json`;
- stage metadata contains exactly the final asset scope;
- latent NPZ files contain only `coords` and `feats` with valid shapes;
- scale files contain finite positive float32-compatible `total_scale`;
- no symlink, empty file, unexpected component, or out-of-scope asset is
  accepted.

### Loader validation

For HSSD standalone and the three-source bundle:

- every configured Dataset constructs without CUDA;
- the first and last final asset for every source direct-load;
- Dataset instances match manifest order, source, root, and count;
- no source/asset pair is duplicated or omitted;
- a zero-worker DataLoader uses the production `collate_fn`;
- the three-source check includes at least one actual sample from ABO,
  3D-FUTURE, and HSSD for every stage.

### Failure behavior

Publication is create-only:

- a complete existing artifact is reused only after full byte and digest
  validation;
- an incomplete stage root, mismatched digest, changed source index,
  insufficient disk space, cross-source overlap, or loader failure aborts;
- the workflow never deletes, truncates, replaces, or silently repairs an
  existing production artifact;
- rejected staging directories remain identified for operator inspection;
- active preprocessing checkouts, workers, queues, source packs, and
  checkpoints are not modified.

## Deliverables

- HSSD source profile and regression tests;
- acceptance validation for both a passed production gate and the existing
  valid-subset waiver;
- storage-root-aware materialization and preflight CLIs;
- standalone HSSD source publication;
- backward-compatible two-source manifest handling;
- explicit three-source publication and resolver support;
- three-source configured Dataset/DataLoader preflight;
- Node16 CPU-only preparation workflow;
- Node16 `num_workers=1` runtime config generation;
- Korean runbook updates with exact preparation, verification, launch,
  monitoring, and resume commands;
- final evidence containing HSSD eligibility exclusions, source/stage counts,
  SHA-256 digests, local artifact paths, and training commands.

## Success criteria

- HSSD is independently usable by all four training configs.
- ABO + 3D-FUTURE + HSSD is usable through one strict `--training_data`
  manifest with unweighted proportional sampling.
- All source data and evidence paths resolve locally on Node16.
- The preparation workflow initializes no CUDA context and starts no training
  or W&B run.
- Every source, combined-manifest, configured Dataset, and DataLoader check
  passes.
- Existing two-source publications and launch behavior remain valid.
- No active preprocessing or unrelated user data is changed.
