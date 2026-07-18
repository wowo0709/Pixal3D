# ObjaverseXL Historical Smoke Recovery Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify Sketchfab and finish GitHub's already-frozen smoke work without mixing tool commits or losing schema-v2 quarantine evidence.

**Architecture:** Run each frozen shard with the exact Git commit stored in its existing manifests. Keep live artifacts immutable during diagnosis, use timestamped backups and a temporary commit-compatible quality ledger for GitHub continuation, then merge only newly durable outcomes into the current schema-v2 ledger.

**Tech Stack:** Git detached clones, Python 3.11, Pixal3D pipeline CLI, JSON quality ledgers, SHA-256 pack and raw archive audit.

## Global Constraints

- Sketchfab historical commit is `db605ec3c9bf1d1b79d93d785e18518459a0f472`.
- GitHub historical commit is `480999ac1b2f77e751bf46b596283f300a7eaac7`.
- Never relabel an existing pack with a different tool commit.
- Never run current preprocessing code while recording a historical tool commit.
- Preserve all current checkpoints, ledgers, packs, archives, and escalation evidence before mutation.
- Quarantine only terminally unusable individual assets; preserve all valid assets.
- Future production work uses one post-fix current commit from its first batch.

---

### Task 1: Freeze and verify the recovery inputs

**Files:**
- Read: `/root/data2/pixal3d/prepared/qualification/smoke/**/ObjaverseXL_*`
- Read: `/root/data2/pixal3d/control/qualification/smoke/checkpoints/ObjaverseXL_*`
- Read: `/root/data2/pixal3d/control/qualification/smoke/quality/ObjaverseXL_*`
- Create: `/root/data2/pixal3d/control/recovery/objaversexl-<timestamp>/`

**Interfaces:**
- Consumes: existing immutable pack manifests, raw manifests, checkpoints, and quality ledgers.
- Produces: a recovery inventory and byte-for-byte backup of mutable control state.

- [x] **Step 1: Verify every existing batch uses one expected commit**

Run a Python read-only inventory that loads every prepared and raw manifest and
asserts all Sketchfab commits equal `db605ec...` and all GitHub commits equal
`480999a...`. Print batch IDs, completed counts, quarantined counts, and manifest
SHA-256 values.

- [x] **Step 2: Verify both historical commits exist**

```bash
git cat-file -e db605ec3c9bf1d1b79d93d785e18518459a0f472^{commit}
git cat-file -e 480999ac1b2f77e751bf46b596283f300a7eaac7^{commit}
```

Expected: both commands exit 0.

- [x] **Step 3: Create a timestamped control-state backup**

Create a bounded recovery directory and copy only ObjaverseXL checkpoint,
quality-ledger, frozen-shard, prepared-index, and escalation JSON files into it.
Record `sha256sum` output for every copied file. Do not copy the large tar packs.

- [x] **Step 4: Confirm the backup matches live control files**

Run `sha256sum -c` against the recorded inventory. Expected: every entry reports `OK`.

### Task 2: Re-audit completed Sketchfab outputs at the producing commit

**Files:**
- Create: a temporary detached shared clone under `/tmp/pixal3d-sketchfab-audit.*`
- Read: all Sketchfab prepared packs, raw archives, checkpoints, and indexes.

**Interfaces:**
- Consumes: exact source tree at `db605ec...` and existing frozen smoke artifacts.
- Produces: successful CLI audit exit status and an immutable audit log under the recovery directory.

- [x] **Step 1: Create and verify the detached historical clone**

```bash
audit_root=$(mktemp -d /tmp/pixal3d-sketchfab-audit.XXXXXX)
git clone --shared --quiet --no-checkout /root/dev/Pixal3D "$audit_root/repo"
git -C "$audit_root/repo" checkout --quiet --detach db605ec3c9bf1d1b79d93d785e18518459a0f472
git -C "$audit_root/repo" rev-parse HEAD
```

Expected: printed commit equals `db605ec...`.

- [x] **Step 2: Run the exact-commit audit**

From the detached clone run:

```bash
conda run --no-capture-output -n pixal3d python -m data_toolkit.pipeline.cli audit \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --gate smoke \
  --source ObjaverseXL_sketchfab \
  --shard ObjaverseXL_sketchfab-00000
```

Expected: exit 0.

- [x] **Step 3: Save audit evidence**

Record the command, exact commit, UTC timestamp, and exit code in the recovery
directory. Keep the detached clone until the GitHub recovery is complete.

### Task 3: Continue GitHub batches with commit-compatible ledger state

**Files:**
- Create: a temporary detached shared clone at commit `480999a...`
- Create: a temporary historical schema-v2 quality ledger, without the later
  `quarantine` field, initialized from the live schema-v2 ledger.
- Modify: GitHub batch 3-6 checkpoints and new prepared/raw artifacts through the historical pipeline only.

**Interfaces:**
- Consumes: frozen GitHub batches, existing commit-`480999a` outputs, and live ledger entries.
- Produces: completed batch 3-6 checkpoints and historical-commit packs/archives.

- [x] **Step 1: Create and verify the GitHub historical clone**

Use the Task 2 clone procedure but check out
`480999ac1b2f77e751bf46b596283f300a7eaac7`. Confirm `git rev-parse HEAD` matches.

- [x] **Step 2: Build a historical schema-v2 temporary ledger**

Read the live schema-v2 ledger and copy only `schema_version`, `source`,
`shard_id`, `gate`, `batches`, and `entries`. Keep `schema_version` set to `2`
but omit the later `quarantine` field, then atomically write it under the
recovery directory. Preserve the live ledger unchanged.

- [x] **Step 3: Run each incomplete context with the temporary ledger**

In a one-off Python driver imported from the historical clone:

```python
with build_mutating_services(config) as runtime:
    services = runtime.services
    services.runner.quality_ledger_path = lambda _context: temporary_ledger
    for index in range(3, 7):
        context = ShardContext.from_config(
            config,
            "ObjaverseXL_github",
            "ObjaverseXL_github-00000",
            f"batch{index:03d}",
            gate="smoke",
        )
        services.runner.resume_shard(context)
        services.batch_auditor(context)
```

The stale batch003 active download attempt is finalized by the existing runner
before retry. If a provider asset remains unavailable after three attempts, the
historical missing-download logic records an asset failure and continues.

- [x] **Step 4: Monitor each terminal batch checkpoint**

For every batch require 26 completed commands, `active_attempt: null`, and one
terminal outcome per frozen asset. Stop only for an infrastructure error or a
new non-asset-scoped data error.

### Task 4: Merge GitHub quality evidence and run audit

**Files:**
- Modify: `/root/data2/pixal3d/control/qualification/smoke/quality/ObjaverseXL_github/ObjaverseXL_github-00000.json`
- Read: all GitHub checkpoints, prepared manifests, and raw manifests.

**Interfaces:**
- Consumes: completed historical schema-v2 ledger without `quarantine`, and
  the unchanged live schema-v2 ledger.
- Produces: one schema-v2 ledger with all batches/entries and quarantine details, plus a passing exact-commit audit.

- [x] **Step 1: Validate the temporary ledger before merging**

Require ordered terminal entries to match every completed batch checkpoint and
frozen instance order. Reject conflicting outcomes for an existing asset.

- [x] **Step 2: Merge only new durable batches and entries**

Atomically update the live schema-v2 ledger with new `batches` and `entries`.
Retain every existing quarantine record. For each newly failed asset without a
detailed historical reason, add:

```json
{
  "category": "historical_asset_failure",
  "stage": "historical_smoke_recovery",
  "reason": "asset failed terminal validation under producing commit 480999a",
  "attempts": 3
}
```

- [x] **Step 3: Run the exact-commit GitHub audit**

```bash
conda run --no-capture-output -n pixal3d python -m data_toolkit.pipeline.cli audit \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --gate smoke \
  --source ObjaverseXL_github \
  --shard ObjaverseXL_github-00000
```

Run from the detached `480999a` clone. Expected: exit 0.

- [x] **Step 4: Verify final source counts and provenance**

Print total completed/quarantined assets and confirm every GitHub pack/raw
manifest uses `480999a`. Save audit command, commit, UTC timestamp, and exit code
beside the recovery inventory.

### Task 5: Record future execution policy

**Files:**
- Modify: `docs/data_preprocessing_runbook_ko.md`

**Interfaces:**
- Consumes: exact-commit recovery results.
- Produces: an operator rule preventing mixed-commit frozen shards.

- [x] **Step 1: Add the immutable-shard provenance rule**

Document that an interrupted frozen shard must be resumed with its producing
commit or restarted from backed-up control state; passing a historical commit
override to current code is forbidden.

- [x] **Step 2: Document the completed source status**

Record Sketchfab and GitHub smoke audit results, exact commits, completed and
quarantined counts, and recovery evidence location.

- [x] **Step 3: Verify documentation and commit**

```bash
git diff --check
git add docs/data_preprocessing_runbook_ko.md
git commit -m "docs: record ObjaverseXL smoke recovery provenance"
```

## Execution Record (2026-07-18)

- Sketchfab exact-commit audit exited 0 under
  `db605ec3c9bf1d1b79d93d785e18518459a0f472`; its frozen scope contains
  18 completed assets and 2 terminal failures.
- GitHub batches 003 through 006 were resumed under
  `480999ac1b2f77e751bf46b596283f300a7eaac7`. All seven batches have 26
  completed commands and no active attempt.
- The atomically merged GitHub ledger contains 20 entries: 12 completed and 8
  quarantined. The exact-commit audit exited 0.
- Individual clone failures were limited to `kaktu5/Nascar` and
  `RetroJohn86/Pogo-APK`, for which GitHub returned `Repository not found`;
  this was not a global GitHub authentication or connectivity failure. Other
  quarantines retain their asset-specific preprocessing outcomes.
- Recovery evidence is stored at
  `/root/data2/pixal3d/control/recovery/objaversexl-20260718-xI6nPr`.
