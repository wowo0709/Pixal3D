# ABO + 3D-FUTURE Training Readiness Design

**Date:** 2026-07-28

**Status:** Approved design awaiting written-spec review

## Goal

Make all four multi-view fine-tuning models immediately launchable with the
completed ABO and 3D-FUTURE production data:

- SS-64;
- Shape-512;
- Shape-1024;
- PBR-1024.

This work prepares and validates the data and launch interface. It does not
start a CUDA context, a W&B run, a smoke-training run, or a 100,000-step
fine-tuning run.

## Fixed Decisions

- Use every stage-eligible asset from both sources for training.
- Do not create train, validation, or test splits inside either source.
- Keep Toys4K separate as evaluation-only data.
- Concatenate the two source instance sets through the existing dataset
  behavior. Sampling is therefore proportional to the number of eligible
  instances, with no balancing sampler or source weights.
- Preserve the four existing model configs, their 100,000-step schedules,
  checkpoint directories, checkpoint interval, retention, and batch split.
- Reuse the already published ABO materialization and evidence byte-for-byte.
- Do not touch GPU 0 or any preprocessing worker.

## Current Inputs

### ABO

The existing strict production handoff is authoritative:

```text
/root/node17/data/pixal3d/train/production/abo/training_data.json
```

Its final stage scopes are:

| Stage | Eligible assets |
|---|---:|
| SS-64 | 3,660 |
| Shape-512 | 3,628 |
| Shape-1024 | 3,634 |
| PBR-1024 | 3,598 |

The implementation must validate the report and handoff digests referenced by
this document but must not republish, overwrite, or rematerialize ABO.

### 3D-FUTURE

The completed production source consists of two immutable shard indexes:

```text
/root/data2/pixal3d/prepared/index/3D-FUTURE/3D-FUTURE-00000.json
/root/data2/pixal3d/prepared/index/3D-FUTURE/3D-FUTURE-00001.json
```

Together they contain 9,472 frozen assets. Before model-specific training
eligibility is applied, their pack-family intersections contain:

| Stage | Candidate assets |
|---|---:|
| SS-64 | 8,495 |
| Shape-512 | 8,513 |
| Shape-1024 | 8,495 |
| PBR-1024 | 8,495 |

The final counts are observed results of the shared eligibility policy and
must not be guessed or hard-coded before evaluation.

## Selected Architecture

Use source-specific materialization and evidence, followed by a small combined
training manifest.

```text
ABO immutable handoff ─────────────┐
                                  ├─ combined training manifest ─ train.py
3D-FUTURE indexes                 │
  └─ materialize ─ strict handoff ┘
```

Physical data remains separated by source. The combined manifest contains
only evidence bindings and stage-specific `data_dir` maps; it never copies or
merges the two sources into a shared data tree.

This preserves source identity in dataset samples and W&B metadata, avoids
duplicating ABO, and uses the existing `StandardDatasetBase` object-root
behavior without adding a new sampler.

## Source-Agnostic Materialization

Refactor the existing ABO production materializer into a reusable core with a
strict source description:

- source name;
- one or more canonical shard index paths;
- expected shard identities and batch sets;
- prepared-data root;
- output root;
- optional fixed count contract;
- acceptance metadata.

Each `FamilyPack` record is identified by `(source, shard_id, batch_id,
family)`. Duplicate batch names across different shards are therefore valid,
while duplicate asset identities across shards are rejected.

The loader must verify every index, manifest digest, pack digest, manifest
identity, family set, and included scope before extracting data. Stage scopes
are the intersections of the same pack families already used for ABO:

| Stage | Required families |
|---|---|
| SS-64 | `common`, `SS-64` |
| Shape-512 | `common`, `shape-512` |
| Shape-1024 | `common`, `shape-1024` |
| PBR-1024 | `common`, `shape-1024`, `PBR-1024` |

For ABO, the existing fixed waiver/count contract remains mandatory. For
3D-FUTURE, the frozen and candidate counts are derived from the two verified
indexes, while final counts are derived by applying the shared eligibility
policy. No source can relax token, coordinate, finite-value, or scale checks.

3D-FUTURE is materialized under:

```text
/root/node17/data/pixal3d/train/production/3d-future/{ss64,shape512,shape1024,pbr1024}/active
```

Publication is create-only. A partially completed or failed attempt is moved
to a unique child of the existing `production/rejected` directory on the same
filesystem. Existing `active` data is never overwritten or deleted.

The materialization command runs CPU-only with low CPU and I/O priority while
the unrelated preprocessing pipeline is active.

## Eligibility and Strict Preflight

3D-FUTURE uses the exact policy already applied to ABO:

- Shape-512 token limit: 8,192;
- Shape-1024 token limit: 32,768;
- PBR-1024 token limit: 32,768;
- both target anchors are checked;
- PBR and Shape coordinates must match exactly for each anchor;
- PBR and Shape `float32 total_scale` values use `rtol=0`, `atol=2e-7`;
- scale values must be finite and positive.

The current policy module must be separated into:

1. source-independent eligibility rules and evidence;
2. the existing ABO-specific fixed count contract.

This refactor must leave ABO validation behavior unchanged.

Strict preflight for every final 3D-FUTURE stage performs:

- exact materialized scope and metadata validation;
- render, transform, latent, token, coordinate, and scale validation;
- direct construction of the real configured Dataset;
- direct `get_instance` loading of both anchors for every eligible asset;
- exact source name and instance-set comparison.

Errors are fail-closed and include source, shard where available, stage,
asset, and anchor context. The dataset retry fallback is not used during
preflight.

## Source Handoff and Combined Manifest

After all four 3D-FUTURE stages pass, publish create-only evidence:

```text
/root/data2/pixal3d/control/reports/gates/3D-FUTURE/3D-FUTURE-production-training.json
/root/data2/pixal3d/control/splits/3D-FUTURE/3D-FUTURE-production-training-handoff.json
/root/node17/data/pixal3d/train/production/3d-future/training_data.json
```

These documents bind:

- both source index paths and byte digests;
- every materialization evidence digest and observed tool commit;
- frozen, candidate, excluded, and final counts;
- the eligibility policy;
- stage roots, scope digests, and validation counts.

Then publish the combined local launch manifest:

```text
/root/node17/data/pixal3d/train/production/abo-3d-future/training_data.json
```

The combined manifest binds the exact bytes and SHA-256 digest of the ABO and
3D-FUTURE handoffs. For each stage it records:

- source counts;
- total count;
- the union scope digest;
- a `data_dir` object with exactly the `ABO` and `3D-FUTURE` keys;
- the component paths required by that stage.

Creation fails if a source handoff changes, a stage is missing, roots are
unsafe, component keys differ from the stage contract, a source scope does
not match its handoff, or an asset identity occurs in both sources.

The source-specific shared evidence is immutable. The combined local manifest
is written atomically and may be reproduced only when its canonical content
is identical.

## Training Entry Point

Add an optional launch argument:

```text
--training_data /root/node17/data/pixal3d/train/production/abo-3d-future/training_data.json
```

`train.py` determines the stage from the selected config's existing
`trainer.args.multiview_stage`, validates the combined manifest, and resolves
the corresponding `data_dir` JSON before constructing the Dataset.

Rules:

- `--training_data` and an explicitly supplied `--data_dir` are mutually
  exclusive;
- existing `--data_dir` callers retain their current behavior;
- an unknown or missing `multiview_stage` fails before model or CUDA
  initialization;
- manifest, handoff, digest, stage, and path errors fail before model or CUDA
  initialization;
- resolved manifest identity and stage counts are printed and stored in the
  resolved run config.

No source balancing, split filtering, dataset changes, or model changes are
introduced.

The four ready-to-run commands use the existing configs and checkpoint
defaults:

```text
configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json
configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json
configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json
configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json
```

The runbook will provide exact `conda run -n pixal3d python train.py`
commands. GPU visibility remains an operator-supplied environment setting so
the preparation work cannot claim or select GPU 0.

## Combined Loader Verification

After source preflight and combined-manifest publication, CPU-only verification
must prove for each stage:

- the Dataset constructs from the combined manifest;
- its instance set equals the disjoint union of the two handoff scopes;
- source counts and total count match the combined manifest;
- no train/validation/test split or balancing sampler is active;
- the existing DataLoader can collate deterministic boundary samples from
  both sources.

Source-specific full direct loading is not repeated during combined
verification because both source handoffs already bind it. Combined
verification checks the integration boundary and representative cross-source
collation.

## Testing Strategy

Implementation follows test-driven development.

Unit tests cover:

- multi-index catalog loading and duplicate detection;
- source-independent eligibility with the ABO fixed contract preserved;
- 3D-FUTURE observed-count evidence;
- create-only and atomic publication behavior;
- combined handoff digest validation and source overlap rejection;
- training-manifest resolution before CUDA initialization;
- backward compatibility of `--data_dir`.

Integration tests use synthetic two-source, two-shard fixtures to cover all
four stages. Existing ABO production tests remain unchanged and must pass.

Final verification consists of:

1. the complete CPU test suite;
2. strict 3D-FUTURE production preflight;
3. combined four-stage Dataset/DataLoader verification;
4. read-only inspection of the four rendered launch commands.

No GPU validation, W&B logging, or training step is part of this task.

## Operational Safety

- Do not signal, pause, stop, or reconfigure preprocessing workers.
- Do not inspect or initialize CUDA as part of preparation.
- Do not use GPU 0.
- Run large materialization and preflight jobs with low CPU/I/O priority.
- Verify free space and active publishers before publication.
- Never overwrite ABO data, source indexes, pack files, reports, or handoffs.
- Preserve failed attempts under `production/rejected`.
- Checkpoint output remains under `/root/data3/pixal3d/ckpts`.

## Acceptance Criteria

The task is complete only when:

- 3D-FUTURE has four strict, published training stage roots and a verified
  source handoff;
- the combined manifest cryptographically binds the unchanged ABO handoff and
  the new 3D-FUTURE handoff;
- all four real Datasets construct with both sources and exact expected
  instance sets;
- proportional source sampling follows from the unweighted concatenated
  instance list;
- all CPU tests and production preflights pass;
- the four documented commands can be invoked without manually constructing
  `data_dir`;
- no CUDA context, W&B run, or training process was started.
