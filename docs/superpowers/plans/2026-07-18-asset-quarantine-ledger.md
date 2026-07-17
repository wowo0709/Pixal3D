# Asset Quarantine Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Persist asset-level quarantine reasons and exclude quarantined assets while preserving infrastructure retry/stop behavior.

**Architecture:** Extend the durable quality ledger with validated per-asset quarantine records. Keep `quality_outcomes` backward compatible, classify only asset-level validation failures, and route all later eligibility/pack logic through the quarantine-aware state.

**Tech Stack:** Python 3.11, dataclasses, JSON checkpoints/ledgers, pytest.

## Global Constraints

- Preserve frozen batch scope and existing checkpoint compatibility.
- Quarantine only asset-intrinsic failures; never quarantine infrastructure/provider-wide/resource/process failures.
- Only completed assets may enter downstream packs and training handoff.

### Task 1: Add quarantine record schema and persistence

**Files:**
- Modify: `data_toolkit/pipeline/orchestrator.py`
- Test: `tests/data_toolkit/test_orchestrator.py`

- [ ] Add validated category/stage/reason fields and a `quarantine` mapping to the quality ledger schema, accepting legacy ledgers without it.
- [ ] Add tests for round-trip persistence, malformed quarantine records, and legacy compatibility.
- [ ] Run the focused tests and commit.

### Task 2: Classify asset-level failures

**Files:**
- Modify: `data_toolkit/pipeline/orchestrator.py`
- Modify: `data_toolkit/pipeline/runtime.py`
- Test: `tests/data_toolkit/test_orchestrator.py`

- [ ] Add an explicit asset-failure recording API carrying category, stage, reason, and attempt count.
- [ ] Update asset-level validators/download quarantine to use it; leave command/infrastructure exception paths unchanged.
- [ ] Test unsupported shader, missing transform, and unavailable source classification plus infrastructure non-quarantine.
- [ ] Run focused tests and commit.

### Task 3: Enforce quarantine in handoff and audit

**Files:**
- Modify: `data_toolkit/pipeline/orchestrator.py`
- Modify: `data_toolkit/pipeline/reporting.py`
- Test: `tests/data_toolkit/test_orchestrator.py`

- [ ] Ensure eligible assets and pack/archive member validation use completed state while quarantine records remain auditable.
- [ ] Add quarantine category counts to the report output without changing frozen asset identity.
- [ ] Test resume behavior and pack exclusion.
- [ ] Run all `tests/data_toolkit` and commit.

### Task 4: Verify and document

- [ ] Run compileall, `git diff --check`, focused tests, and full data-toolkit tests.
- [ ] Verify current GitHub quarantine records and manifest member counts.
- [ ] Commit documentation/status updates.
