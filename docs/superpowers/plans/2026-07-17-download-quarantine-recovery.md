# Download Quarantine Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Resume a frozen smoke batch when an asset-specific source disappears by quarantining only the missing SHA after three download attempts.

**Architecture:** Keep strict raw metadata validation for complete outputs, add a retry-aware partial validator for the download command, and make exhausted checkpoints validate before refusing further launches. Internal staging consumes the same eligible SHA set used by leaf workers.

**Tech Stack:** Python 3.11, pandas, pytest, existing `data_toolkit.pipeline.orchestrator` checkpoint and quality-ledger code.

## Global Constraints

- Preserve the frozen SHA manifest, source order, camera policy, output schema, dtype, and config hash.
- Keep the existing three total command-launch limit.
- Quarantine only missing rows after a non-empty partial download survives the third attempt.
- Keep zero-result, malformed, unsafe-path, duplicate-path, checksum, provider, authentication, checkpoint, process-control, and resource failures fail-closed.
- Run all Python and pytest commands through `conda run -n pixal3d`.
- Do not manually edit or reset the live checkpoint.

---

### Task 1: Add failing recovery tests

**Files:**
- Modify: `tests/data_toolkit/test_orchestrator.py`

**Interfaces:**
- Exercise `PipelineServices` and its existing `PipelineRunner` with synthetic frozen batches.

- [ ] **Step 1: Add tests for retry-aware partial download validation, exhausted-checkpoint recovery, staging eligibility, and fail-closed empty output.**
- [ ] **Step 2: Run only the new tests and confirm they fail because the current validator requires every selected SHA and the runner stops on an exhausted attempt budget.**

### Task 2: Implement minimal orchestrator recovery

**Files:**
- Modify: `data_toolkit/pipeline/orchestrator.py`

**Interfaces:**
- Add internal partial-record reading/validation helpers.
- Reuse `record_quality_outcome(asset_sha, "failure")` for durable quarantine.

- [ ] **Step 1: Validate partial download rows and distinguish missing SHA rows from malformed or unsafe rows.**
- [ ] **Step 2: Quarantine missing rows only at the third attempt and only when at least one selected row is valid.**
- [ ] **Step 3: Validate an exhausted command before raising the attempt-budget stop.**
- [ ] **Step 4: Make `stage_raw` select only eligible assets while retaining strict complete-record checks.**
- [ ] **Step 5: Run the focused recovery tests and confirm they pass.**

### Task 3: Regression verification and commit

**Files:**
- Modify: `data_toolkit/pipeline/orchestrator.py`
- Modify: `tests/data_toolkit/test_orchestrator.py`

- [ ] **Step 1: Run the full orchestrator test module.**
- [ ] **Step 2: Run the complete `tests/data_toolkit` suite, compileall, and `git diff --check`.**
- [ ] **Step 3: Commit the implementation and tests.**
- [ ] **Step 4: Resume the exact frozen GitHub smoke batch and run its audit before starting any other source.**
