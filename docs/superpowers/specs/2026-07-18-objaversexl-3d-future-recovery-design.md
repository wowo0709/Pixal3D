# ObjaverseXL and 3D-FUTURE Recovery Design

## Goal

Make ObjaverseXL Sketchfab, ObjaverseXL GitHub, and 3D-FUTURE usable by the
existing Pixal3D preprocessing pipeline without weakening raw-data integrity,
mixing preprocessing provenance, or adding model-side behavior.

Success means:

- ObjaverseXL Sketchfab retains its verified historical smoke outputs and
  passes audit under the commit that produced them.
- ObjaverseXL GitHub resumes and finishes its frozen smoke shard under the
  commit that produced its existing batches, with unusable individual assets
  recorded in the quarantine ledger.
- 3D-FUTURE distinguishes the canonical asset identity from raw file content
  hashes, stages every file required by the OBJ, and archives those files with
  verified checksums.
- 3D-FUTURE false checksum quarantines are recovered, the frozen smoke shard is
  rerun, and its audit result is recorded.
- Future production execution uses one fixed post-fix commit for every batch.

## Confirmed Root Causes

### ObjaverseXL

The existing prepared packs are not corrupt. Sketchfab packs were produced at
`db605ec3c9bf1d1b79d93d785e18518459a0f472`; GitHub packs were produced at
`480999ac1b2f77e751bf46b596283f300a7eaac7`. Audit intentionally rejects them
when run from a different Git commit. Sketchfab passes audit when checked out
at its producing commit.

The GitHub frozen shard is incomplete. Its first three batches were produced
at `480999a`, while batch 3 was interrupted during download. Continuing that
shard with the current code would create mixed provenance. Provider-level or
format-level failures remain asset-scoped and must be quarantined rather than
used to reject every valid ObjaverseXL asset.

### 3D-FUTURE

The canonical TRELLIS-500K `sha256` for 3D-FUTURE is the SHA-256 of
`image.jpg`, which identifies the asset. The pipeline incorrectly used that
identity as the expected checksum of `raw_model.obj`. All inspected assets
match their canonical `image.jpg` SHA-256 and have different, internally
consistent OBJ content hashes.

The selected OBJ also depends on sibling files such as `model.mtl` and
`texture.png`. Copying and archiving only `raw_model.obj` is insufficient for
paper-faithful PBR preprocessing and later reproduction.

## Data Contract

Raw download metadata keeps `sha256` as the canonical asset identity and
`local_path` as the primary model path. It gains:

- `content_sha256`: SHA-256 of the primary raw model file. If absent for an
  older adapter, it defaults to `sha256` for backward compatibility.
- `companion_files`: a deterministic JSON object mapping each required sibling
  relative path to its SHA-256. It is optional and defaults to an empty object.

The 3D-FUTURE adapter computes `content_sha256` for `raw_model.obj` and lists
the other regular files from the selected archive directory as companion
files. It continues to verify `image.jpg` against the canonical asset identity
before publishing any download record.

All paths remain relative, normalized, non-symlinked, and confined beneath the
source root. Duplicate paths across selected assets are rejected.

## Pipeline Behavior

`stage_raw` verifies and copies the primary file and every declared companion
file. The staged metadata preserves the new fields. The staged-output validator
recomputes each declared content hash.

`archive_raw` includes all declared primary and companion files. Raw archive
verification compares the complete path-to-content-hash mapping, while
`asset_sha256s`, completed counts, and quarantine counts continue to use
canonical asset identities. Existing sources without companion files retain
their current one-primary-file behavior.

No checksum check is disabled and no source-specific validation bypass is
introduced.

## Recovery Procedure

### ObjaverseXL

1. Preserve the current checkpoints, ledgers, packs, archives, and escalation
   evidence.
2. Audit Sketchfab with the exact producing commit; retain the passing result.
3. Execute GitHub's remaining frozen batches using the exact `480999a` source
   tree. Use a temporary schema-v1 quality ledger compatible with that commit,
   initialized from the durable live ledger.
4. Merge new terminal outcomes back into the schema-v2 live ledger. Add
   quarantine details for newly failed assets without altering existing
   records.
5. Audit all GitHub batches under `480999a` and preserve the result.

Historical commits are used only to finish and verify already-frozen smoke
work. They are not used for new production execution.

### 3D-FUTURE

1. Back up the false-quarantine smoke checkpoint, ledger, empty prepared packs,
   raw archives, and escalation reports to a timestamped recovery directory.
2. Rebuild the selected raw metadata from the canonical archive using the new
   data contract.
3. Restore the frozen smoke scope with fresh checkpoints and a fresh quality
   ledger; do not change its selected asset identities.
4. Run all frozen smoke batches and quarantine only failures reproduced after
   the checksum fix.
5. Audit the completed shard with the same fixed commit.

## Error Handling

- A canonical `image.jpg` identity mismatch remains a download failure.
- A primary or companion content mismatch stops staging and reports the exact
  relative path.
- Missing OBJ dependencies are asset-scoped failures once retries are
  exhausted.
- An unavailable ObjaverseXL provider asset is quarantined after the existing
  retry limit; valid assets continue.
- A historical shard is never resumed if its existing pack commits are mixed
  or unavailable in the repository.

## Tests and Verification

Tests are added before implementation for:

- 3D-FUTURE identity hash differing from `raw_model.obj` content hash.
- deterministic companion-file metadata and preservation of OBJ dependencies.
- rejection of unsafe, missing, duplicate, or checksum-mismatched companion
  paths.
- staged raw and raw archive verification using content hashes rather than
  canonical asset identities.
- backward compatibility for existing two-column raw metadata.

Verification consists of focused adapter and orchestrator tests, the complete
`tests/data_toolkit` suite, exact-commit ObjaverseXL audits, a recovered
3D-FUTURE smoke run, and a final 3D-FUTURE audit.

## Out of Scope

- Model architecture changes.
- Changes to variable-K multi-view fusion.
- Relaxing data-quality thresholds or silently accepting invalid assets.
- Reprocessing completed ObjaverseXL historical batches solely to relabel their
  tool commit.
