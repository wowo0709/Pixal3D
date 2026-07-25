# ABO Training-Eligibility Subset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Filter immutable ABO pack-intersection candidates into four exact, config-compatible training scopes and publish evidence that distinguishes pack validity from training eligibility.

**Architecture:** A shared CPU-only policy module evaluates extracted Shape/PBR anchors using the existing fine-tuning token limits, exact coordinate equality, and the user-approved scale tolerance. The materializer applies the policy only inside its hidden temporary stage, removes excluded assets from every component before atomic publication, and records candidate/final scopes plus deterministic reasons. Strict preflight and handoff publication consume the same policy contract and enforce the new exact counts.

**Tech Stack:** Python 3.11, dataclasses, pathlib, NumPy, JSON/CSV, pytest, existing production pack materializer, existing strict real-loader preflight.

## Global Constraints

- Preserve fine-tuning configs: Shape-512 `max_tokens=8192`; Shape/PBR-1024 `max_tokens=32768`.
- PBR/Shape coordinates remain exactly equal; no tolerance is permitted.
- PBR/Shape finite positive float32 scales use `rtol=0`, `atol=2e-7`.
- Candidate counts remain `ss64=3660`, `shape512=3631`, `shape1024=3660`, `pbr1024=3660`.
- Final counts are exactly `ss64=3660`, `shape512=3628`, `shape1024=3634`, `pbr1024=3598`.
- Training-exclusion counts are exactly `ss64=0`, `shape512=3`, `shape1024=26`, `pbr1024=62`.
- The original frozen count `4485`, global quarantine `825`, and Shape-512 pack-family exclusions `29` remain unchanged and distinct from training exclusions.
- Continue to verify immutable packs and every selected member's size/SHA-256 before eligibility filtering.
- Remove excluded assets only from the known hidden temporary stage before publication.
- Never edit or overwrite an existing `active` root.
- Run policy, materialization, and preflight CPU-only with one application-level worker.
- Publish no report, handoff, or `training_data.json` unless all four final scopes pass strict structural and direct-loader validation.
- Do not change model architecture, trainer, optimizer, batch policy, checkpoints, W&B, or start fine-tuning.

---

## File Structure

- Create `data_toolkit/pipeline/training_eligibility.py`: shared deterministic token/coordinate/scale policy and stage evaluation.
- Modify `scripts/materialize_multiview_production.py`: candidate/final count separation, temporary-stage filtering, exclusion evidence.
- Modify `tests/multiview/test_production_materialization.py`: real NPZ/scale policy fixtures, removal/evidence/count behavior.
- Modify `scripts/preflight_multiview_production.py`: shared limits/tolerance, eligibility evidence validation, final handoff counts.
- Modify `tests/multiview/test_production_preflight.py`: tolerance boundary, policy/count mutations, new report/handoff contracts.
- Modify `docs/data_preprocessing_runbook_ko.md`: final counts, eligibility policy, failed-attempt preservation, exact commands.

---

### Task 1: Shared Eligibility Policy and Atomic Candidate Filtering

**Files:**
- Create: `data_toolkit/pipeline/training_eligibility.py`
- Modify: `scripts/materialize_multiview_production.py`
- Modify: `tests/multiview/test_production_materialization.py`

**Interfaces:**
- Produces:
  - `TOKEN_LIMITS = {"shape512": 8192, "shape1024": 32768, "pbr1024": 32768}`.
  - `SCALE_RTOL = 0.0` and `SCALE_ATOL = 2e-7`.
  - `EligibilityExclusion` frozen dataclass with `asset: str` and `reasons: tuple[str, ...]`.
  - `policy_evidence() -> dict[str, object]`.
  - `evaluate_stage_asset(stage: str, root: Path, asset: str) -> tuple[str, ...]`.
  - `filter_stage_scope(stage: str, root: Path, candidates: Sequence[str]) -> tuple[tuple[str, ...], tuple[EligibilityExclusion, ...]]`.
- Consumed by Task 2:
  - exact policy evidence;
  - deterministic reason identifiers;
  - candidate/final scopes and exclusions stored in `materialization.json`.

- [ ] **Step 1: Add failing policy boundary tests with real files**

Create real NPZ/scale fixtures under the exact component paths. Add tests that
fail because `training_eligibility` does not exist:

```python
def test_training_eligibility_uses_both_anchor_token_limits(tmp_path):
    root, asset = write_eligibility_stage(
        tmp_path, "shape512", shape_counts=(8192, 8193)
    )
    assert evaluate_stage_asset("shape512", root, asset) == (
        "shape_tokens_view01_exceed_8192",
    )
```

Add independent tests for:

- Shape-512 counts `(8192, 8192)` pass and either anchor at `8193` fails;
- Shape-1024 counts at `32768` pass and `32769` fails;
- PBR-1024 applies the limit to both Shape and PBR arrays;
- equal coordinates pass and same-shape/different-value coordinates fail;
- PBR/Shape scale absolute difference `2e-7` passes;
- scale difference greater than `2e-7` fails;
- non-numeric, non-scalar, non-finite, or non-positive scales raise contextual `ValueError`;
- reasons are sorted, unique, and include the anchor.

- [ ] **Step 2: Run policy tests and verify RED**

Run:

```bash
CUDA_VISIBLE_DEVICES="" conda run --no-capture-output -n pixal3d \
  python -m pytest -q \
  tests/multiview/test_production_materialization.py -k training_eligibility
```

Expected: collection fails with
`ModuleNotFoundError: data_toolkit.pipeline.training_eligibility`.

- [ ] **Step 3: Implement the minimal shared policy**

Implement strict helpers that:

- load NPZ with `allow_pickle=False`;
- require exact `coords`/`feats` keys and `[N,3]`/`[N,32]` compatible rows;
- convert scales to a JSON numeric scalar and then float32;
- compare coordinates with `numpy.array_equal`;
- compare scales with
  `numpy.isclose(shape, pbr, rtol=SCALE_RTOL, atol=SCALE_ATOL)`;
- return deterministic reason tuples instead of silently skipping malformed
  data.

`policy_evidence()` returns canonical JSON-compatible values:

```python
{
    "schema_version": 1,
    "token_limits": {
        "shape512": 8192,
        "shape1024": 32768,
        "pbr1024": 32768,
    },
    "pbr_shape_coordinates": "exact",
    "pbr_shape_scale": {
        "dtype": "float32",
        "rtol": 0.0,
        "atol": 2e-7,
    },
}
```

The exact reason identifiers are:

```text
shape_tokens_view00_exceed_<limit>
shape_tokens_view01_exceed_<limit>
pbr_tokens_view00_exceed_<limit>
pbr_tokens_view01_exceed_<limit>
pbr_shape_coords_view00_mismatch
pbr_shape_coords_view01_mismatch
pbr_shape_scale_view00_mismatch
pbr_shape_scale_view01_mismatch
```

- [ ] **Step 4: Add failing temporary-stage filtering tests**

Extend the synthetic production packs to contain valid NPZ and scale JSON
members. Add tests proving:

```python
def test_materialize_stage_removes_training_ineligible_assets_from_every_component(tmp_path):
    # one valid and one Shape-512 asset whose view01 has 8193 tokens
    final = materialize_stage(...)
    evidence = json.loads((final / "materialization.json").read_text())
    assert evidence["candidate_asset_count"] == 2
    assert evidence["asset_count"] == 1
    assert evidence["training_exclusions"] == [{
        "asset": rejected,
        "reasons": ["shape_tokens_view01_exceed_8192"],
    }]
```

Also assert:

- the rejected render and every rejected latent directory are absent;
- every component metadata file contains only the final scope;
- candidate/final scope digests are independently correct;
- exclusion order and reason-count maps are deterministic;
- scale-within-tolerance remains in PBR scope;
- coordinate mismatch is removed from PBR render, Shape, and PBR components;
- a final-count mismatch cleans the hidden temporary directory and publishes
  nothing;
- original pack/member verification and active non-overwrite tests remain
  green.

- [ ] **Step 5: Run filtering tests and verify RED**

Run Task 1 tests. Expected: policy tests pass while filtering/evidence tests
fail because the materializer still publishes the raw candidate scope.

- [ ] **Step 6: Implement candidate/final filtering**

In the materializer:

1. rename the old exact counts to `EXPECTED_CANDIDATE_COUNTS`;
2. define the approved `EXPECTED_STAGE_COUNTS` and
   `EXPECTED_TRAINING_EXCLUSION_COUNTS`;
3. verify and extract all candidate members into the hidden stage;
4. call `filter_stage_scope`;
5. remove each excluded asset from `renders_cond` and every stage component
   using exact, validated paths beneath the hidden root;
6. write metadata from the final scope;
7. write candidate/final scope counts/digests, policy evidence, sorted
   exclusions, and reason counts;
8. require exact candidate/final/exclusion counts before
   `_publish_no_replace`.

No option may disable eligibility inspection or exact count enforcement for the
default production run.

The exact added evidence keys are:

```text
candidate_asset_count
candidate_stage_scope
candidate_stage_scope_sha256
training_exclusion_count
training_exclusions
training_exclusion_reason_counts
eligibility_policy
```

The existing `asset_count`, `stage_scope`, and `stage_scope_sha256` keys remain
the final published scope.

- [ ] **Step 7: Verify Task 1 GREEN and regressions**

Run:

```bash
CUDA_VISIBLE_DEVICES="" conda run --no-capture-output -n pixal3d \
  python -m pytest -q \
  tests/multiview/test_production_materialization.py \
  tests/multiview/test_pilot_materialization.py \
  tests/data_toolkit/test_packing.py
CUDA_VISIBLE_DEVICES="" conda run --no-capture-output -n pixal3d \
  python -m compileall -q \
  data_toolkit/pipeline/training_eligibility.py \
  scripts/materialize_multiview_production.py \
  tests/multiview/test_production_materialization.py
git diff --check
```

- [ ] **Step 8: Self-review and commit Task 1**

Confirm filtering happens only in the hidden temporary stage and exact member
verification still precedes filtering. Commit:

```bash
git add data_toolkit/pipeline/training_eligibility.py \
  scripts/materialize_multiview_production.py \
  tests/multiview/test_production_materialization.py
git commit -m "feat: filter ABO training-eligible stage scopes"
```

---

### Task 2: Strict Preflight, Handoff Evidence, and Operator Contract

**Files:**
- Modify: `scripts/preflight_multiview_production.py`
- Modify: `tests/multiview/test_production_preflight.py`
- Modify: `docs/data_preprocessing_runbook_ko.md`

**Interfaces:**
- Consumes:
  - `policy_evidence`, `TOKEN_LIMITS`, `SCALE_RTOL`, and `SCALE_ATOL`;
  - candidate/final scopes and exclusion evidence from Task 1.
- Produces:
  - reports/handoffs with exact candidate/final/exclusion counts;
  - strict validation of the stored eligibility policy;
  - updated operator commands and recovery steps.

- [ ] **Step 1: Add failing preflight policy/count tests**

Update fixtures to the approved final counts and valid materialization
eligibility evidence. Add failures for:

- missing or altered policy evidence;
- missing/reordered/duplicated exclusions;
- candidate minus excluded assets not equal to final scope;
- wrong candidate/final/exclusion counts;
- physically present excluded directories;
- PBR/Shape scale difference exactly `2e-7` accepted;
- scale difference above `2e-7` rejected;
- coordinate mismatch still rejected exactly;
- token limits sourced from the shared policy rather than duplicated literals.

- [ ] **Step 2: Run targeted preflight tests and verify RED**

Run:

```bash
CUDA_VISIBLE_DEVICES="" conda run --no-capture-output -n pixal3d \
  python -m pytest -q \
  tests/multiview/test_production_preflight.py
```

Expected: failures show the old counts and exact scale comparison are still in
use and eligibility evidence is not validated.

- [ ] **Step 3: Implement shared-policy preflight validation**

Replace local token/tolerance literals with shared policy values. Validate:

- canonical candidate and final scopes and both digests;
- sorted unique exclusion objects and reason tuples;
- `candidate_scope - excluded_assets == final_scope`;
- exact approved stage counts and exclusion totals;
- exact `policy_evidence()` equality;
- no excluded asset remains in any final component.

Keep all existing structural, direct-loader, evidence-byte, canonical-index,
partial-publication recovery, and create-only checks.

- [ ] **Step 4: Add failing report/handoff tests**

Require report, handoff, and local training data to contain:

```python
"counts": {
    "frozen": 4485,
    "global_quarantine": 825,
    "shape512_family_exclusions": 29,
    "candidate_stages": {
        "ss64": 3660,
        "shape512": 3631,
        "shape1024": 3660,
        "pbr1024": 3660,
    },
    "training_exclusions": {
        "ss64": 0,
        "shape512": 3,
        "shape1024": 26,
        "pbr1024": 62,
    },
    "stages": {
        "ss64": 3660,
        "shape512": 3628,
        "shape1024": 3634,
        "pbr1024": 3598,
    },
}
```

Require exact eligibility policy and materialization evidence digests in every
shared document. Mutating only a policy value, candidate count, or exclusion
reason must fail before publication.

- [ ] **Step 5: Implement report/handoff contract**

Update `HANDOFF_STAGE_COUNTS`, input validation, report/handoff builders, and
recovery equality checks. Preserve:

- valid-subset waiver and failed original 90% gate;
- canonical index path/digest;
- exact validated materialization-byte digest;
- report -> handoff -> local publication order.

- [ ] **Step 6: Update the Korean runbook**

Record:

- candidate/final counts and exclusion reasons;
- the unchanged model token limits;
- exact-coordinate and `atol=2e-7` scale policy;
- same-filesystem preservation of the failed current attempt;
- no-active-ABO and at-least-70-GiB pre-run checks;
- low-priority CPU-only materializer/preflight commands;
- stop-before-fine-tuning boundary.

- [ ] **Step 7: Verify Task 2 GREEN and full regressions**

Run focused tests:

```bash
CUDA_VISIBLE_DEVICES="" conda run --no-capture-output -n pixal3d \
  python -m pytest -q \
  tests/multiview/test_production_materialization.py \
  tests/multiview/test_production_preflight.py
```

Run the complete suite exactly once at the final Task 2 state with the existing
process-local `flex_gemm` import guard, `nice -n 15`, `ionice -c 2 -n 7`, and a
JUnit output path. Also run:

```bash
CUDA_VISIBLE_DEVICES="" conda run --no-capture-output -n pixal3d \
  python -m compileall -q \
  data_toolkit/pipeline/training_eligibility.py \
  scripts/materialize_multiview_production.py \
  scripts/preflight_multiview_production.py \
  tests/multiview/test_production_materialization.py \
  tests/multiview/test_production_preflight.py
git diff --check
```

- [ ] **Step 8: Self-review and commit Task 2**

Confirm the final handoff authorizes only the training-eligible scopes and no
training command was added. Commit:

```bash
git add scripts/preflight_multiview_production.py \
  tests/multiview/test_production_preflight.py \
  docs/data_preprocessing_runbook_ko.md
git commit -m "feat: publish ABO training-eligibility evidence"
```

---

## Post-Implementation Production Gate

Run only after both Task reviews and a cumulative review have no open Critical
or Important findings.

1. Confirm the existing second attempt has no shared report/handoff/local
   training manifest.
2. Move exactly
   `/root/node17/data/pixal3d/train/production/abo` to a unique child of
   `/root/node17/data/pixal3d/train/production/rejected/`; do not delete it.
3. Confirm at least 70 GiB free and no active ABO publisher.
4. Run the low-priority CPU-only materializer.
5. Verify final counts `3660/3628/3634/3598`, exclusion evidence, exact
   component counts, and no hidden residue.
6. Run the low-priority CPU-only strict preflight.
7. Verify report/handoff/training-data cross-digests and independently
   construct all four configured datasets with exact final lengths.
8. Confirm no CUDA context or training process was created.
9. Stop before fine-tuning and report the release state.
