# ABO Training-Eligibility Subset Design

## Context

The immutable ABO production packs passed their pack/index integrity checks, but
the first full training preflight exposed constraints that were not represented
in the pack-family intersections:

- three Shape-512 assets exceed the configured `max_tokens=8192`;
- twenty-six Shape-1024 assets exceed the configured `max_tokens=32768`;
- the same twenty-six assets exceed the PBR-1024 stage token limit;
- thirty-seven PBR/Shape asset-anchor pairs have non-identical coordinates;
- PBR/Shape scale values differ for 1,994 assets, but the observed float32
  difference is at most three ULP and `1.7881393432617188e-7`.

The user approved preserving the existing fine-tuning configurations, excluding
token/coordinate-invalid assets, and treating the observed scale drift as
numerically equivalent. This replaces the earlier assumption that every
pack-intersection asset is directly training-eligible.

## Goal

Publish stage-isolated ABO roots that satisfy the existing fine-tuning configs
and strict real-loader contracts without increasing token limits, changing the
model, re-encoding assets, or weakening coordinate alignment.

The exact final stage counts are:

| Stage | Candidate count | Training exclusions | Final count |
|---|---:|---:|---:|
| `ss64` | 3,660 | 0 | 3,660 |
| `shape512` | 3,631 | 3 | 3,628 |
| `shape1024` | 3,660 | 26 | 3,634 |
| `pbr1024` | 3,660 | 62 | 3,598 |

The PBR exclusion total is the union of twenty-six over-limit assets and
thirty-seven coordinate-mismatch assets, with one asset present in both sets.

## Architecture

### Shared eligibility policy

Create one CPU-only policy module used by the materializer and strict
preflight. It owns:

- Shape-512 token maximum: `8192`;
- Shape/PBR-1024 token maximum: `32768`;
- PBR/Shape scale tolerance: `rtol=0`, `atol=2e-7`;
- exact PBR/Shape coordinate equality;
- deterministic reason identifiers and policy evidence.

The policy reads only extracted NPZ and scale JSON files. It does not import a
model, trainer, CUDA, W&B, multiprocessing, or a DataLoader.

### Candidate extraction and filtering

The production materializer continues to:

1. verify the immutable production index and every pack;
2. derive the raw family-intersection candidate scope;
3. stream selected members into a hidden same-filesystem temporary stage;
4. verify every selected member's manifest size and SHA-256.

Before metadata or `materialization.json` is written, the materializer evaluates
each candidate asset:

- `ss64`: no additional training exclusion;
- `shape512`: reject if either anchor exceeds 8,192 coordinates;
- `shape1024`: reject if either anchor exceeds 32,768 coordinates;
- `pbr1024`: reject if either Shape or PBR anchor exceeds 32,768 coordinates,
  if Shape/PBR coordinates are not exactly equal, or if finite positive
  float32 scales differ by more than `2e-7`.

Every rejected asset is removed only from the hidden temporary stage and from
all of that stage's components, including renders. Published `active` roots
therefore contain exactly the final training scope and no unlisted asset
directories. Existing active roots remain non-overwritable.

### Evidence

Each stage `materialization.json` keeps the existing pack/index/waiver
provenance and adds:

- candidate scope count and SHA-256;
- final scope count and SHA-256;
- eligibility policy with token limits, coordinate mode, and scale tolerance;
- sorted excluded assets with sorted deterministic reason identifiers;
- reason counts and total training-exclusion count.

The materializer rejects any run whose candidate or final counts differ from
the table above. The strict preflight recomputes eligibility for every final
asset, validates the policy evidence, and rejects any excluded asset that
remains physically present.

The report, handoff, and local `training_data.json` record both candidate and
final stage counts plus the training-exclusion counts. They continue to record:

- `acceptance_mode=valid_subset_user_waiver`;
- `original_90_percent_gate_passed=false`;
- frozen count `4485`;
- global quarantine count `825`;
- original Shape-512 pack-family exclusions `29`.

The new training exclusions are separate from those original pack-level
counts.

## Error Handling and Recovery

- Eligibility inspection errors are fatal; the materializer cleans only its
  known hidden temporary stage.
- A final-count mismatch is fatal before atomic publication.
- The strict preflight publishes no report, handoff, or local training manifest
  unless all four stages pass structural and direct-loader validation.
- Existing failed materializations are never edited in place. The operator
  moves each failed attempt to a unique `rejected` path on the same filesystem
  before a corrected materializer run.
- No command in this design deletes rejected attempts or starts fine-tuning.

## Testing

Synthetic TDD coverage must prove:

- both anchors participate in token eligibility;
- the exact 8,192 and 32,768 boundaries pass and one token above fails;
- PBR/Shape coordinate mismatch is excluded even when shapes match;
- scale drift at `2e-7` passes and a larger drift fails;
- exclusions remove every component directory and metadata row;
- candidate/final scopes, reason lists, reason counts, and policy evidence are
  deterministic;
- exact production counts are enforced;
- preflight and handoff reject altered eligibility policy or counts;
- the configured real loaders still validate both anchors directly.

After focused tests and review, run the complete guarded CPU-only multiview
suite once at the final commit.

## Production Procedure

1. Preserve the current failed materialization by same-filesystem rename into
   `/root/node17/data/pixal3d/train/production/rejected/`.
2. Confirm the four target `active` roots and all shared handoff artifacts are
   absent.
3. Run the corrected materializer with `CUDA_VISIBLE_DEVICES=""`, `nice -n 15`,
   and `ionice -c 2 -n 7`.
4. Confirm counts `3660/3628/3634/3598`, evidence, and no hidden residue.
5. Run the corrected strict preflight under the same resource constraints.
6. Independently verify report/handoff/training-data cross-digests and dataset
   lengths.
7. Stop before fine-tuning.

## Out of Scope

- increasing `max_tokens`;
- changing batch size, batch split, model architecture, optimizer, or trainer;
- re-encoding the thirty-seven coordinate-mismatch pairs;
- performance or quality improvements beyond making the approved subset
  trainable;
- starting W&B logging or fine-tuning.
