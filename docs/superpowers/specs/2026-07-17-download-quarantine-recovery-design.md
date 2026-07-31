# Asset-Scoped Download Quarantine and Recovery Design

## Context

The frozen `ObjaverseXL_github` smoke scope contains an asset whose original
GitHub repository is now deleted or private. The ObjaverseXL adapter returns
the other downloaded rows but no row for that asset. The current download
validator requires every selected SHA, so it retries the whole command three
times and then stops the batch even though two verified assets are available.

The recovery must preserve the frozen SHA manifest, existing artifacts, and
attempt evidence. It must not replace the unavailable asset with a more
convenient sample or silently weaken provider, checksum, schema, resource, or
checkpoint failures.

## Approved Behavior

- Preserve the frozen manifest and its SHA order.
- Retry an incomplete download command up to the existing limit of three total
  launches.
- After the third launch, record only still-missing selected SHA rows as durable
  per-asset `failure` outcomes.
- Continue later commands with eligible assets only. Quarantined SHA values stay
  in the frozen scope, quality ledger, pack/archive counts, reports, and audits.
- Reuse already verified repository ZIPs and metadata rows. Do not redownload
  successful repositories merely to recover the missing asset.
- For an already exhausted checkpoint, run the same terminal download
  validation before raising `command attempt budget already exhausted`. If the
  partial result is valid, complete the command without clearing or editing its
  recorded attempt count.
- Do not quarantine on the first or second incomplete attempt.
- Do not convert a zero-result batch, malformed metadata, unsafe path,
  duplicate path, checksum failure, authentication failure, provider-wide
  error, process-control error, checkpoint error, or resource error into an
  asset failure. Those conditions retain the existing fail-closed behavior.

## Data Flow

1. The download leaf worker atomically merges successful `sha256,local_path`
   rows exactly as it does now.
2. Download validation reads available rows for the selected frozen assets and
   validates their identities and paths.
3. Before the third attempt, any missing selected row keeps the command
   incomplete so the normal retry path runs.
4. At the third attempt, a non-empty valid subset makes each remaining missing
   SHA a durable `failure`; validation then succeeds for the eligible subset.
5. `stage_raw` reads and stages only eligible assets. All later leaf commands
   receive the runner's eligible SHA file and therefore never require the
   quarantined source path.
6. Terminal validation, pack publication, raw archive publication, audit, and
   reporting retain the original frozen count and record the unavailable asset
   in `quarantined_count`.

## Recovery of the Current Batch

`ObjaverseXL_github-00000/batch002` has `download: 3` attempts, two valid
download rows, and no completed command. On resume, the runner first performs
terminal validation because the attempt budget is exhausted. That validation
records the missing SHA as `failure`, marks `download` complete, and proceeds
to `stage_raw` for the two eligible SHA values. No checkpoint field or frozen
manifest is reset manually.

If another asset-specific repository is unavailable in later batches, the same
three-attempt rule quarantines it and continues. Source admission thresholds
are not changed: smoke may finish for diagnosis but its report must still fail
if the measured source success rate is below the configured 90 percent gate.

## Code Boundaries

- `data_toolkit/pipeline/orchestrator.py`
  - Add partial raw-record validation without relaxing existing strict readers.
  - Make download validation retry-aware and asset-scoped only at terminal
    attempt exhaustion.
  - Allow exhausted commands to accept valid terminal partial results before
    stopping.
  - Make raw staging consume eligible assets.
- `tests/data_toolkit/test_orchestrator.py`
  - Prove attempts one and two do not quarantine a missing download.
  - Prove attempt three quarantines only the missing SHA and completes.
  - Prove an already exhausted checkpoint resumes without another launch.
  - Prove raw staging excludes the quarantined SHA.
  - Prove empty and structurally invalid partial results still fail closed.

No dataset sampling, camera policy, output schema, dtype, model code, or source
ordering changes are in scope.

## Verification

Use a strict red-green cycle for each recovery behavior. Then run the focused
orchestrator tests, the full `tests/data_toolkit` suite, `compileall`, and a
clean-diff check. After committing the fix, preserve the failed escalation and
resume the exact frozen GitHub smoke scope. Audit that scope before any next
source starts.
