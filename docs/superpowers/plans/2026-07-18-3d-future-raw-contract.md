# 3D-FUTURE Raw Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve 3D-FUTURE's canonical image-based asset identity while checksum-verifying and carrying the OBJ, MTL, texture, and other required raw files through staging and archival.

**Architecture:** Extend adapter raw metadata with a primary content hash and a deterministic companion-file hash map. Normalize this contract in `PipelineServices`, default missing fields for existing sources, and use the normalized file map for staging, validation, raw archival, and source cleanup.

**Tech Stack:** Python 3.11, pandas, CSV/JSON, `hashlib.sha256`, pytest, the existing Pixal3D pipeline and pack verifier.

## Global Constraints

- Keep `sha256` as the canonical asset identity; never replace it with the OBJ hash.
- Do not disable checksum verification or introduce an unchecked 3D-FUTURE exception.
- Existing two-column `sha256,local_path` raw metadata must remain valid.
- Preserve relative-path, no-symlink, no-traversal, duplicate-path, and archive integrity checks.
- Stage and archive every declared companion file required to reproduce OBJ/PBR processing.
- CUDA 12.8 and PyTorch 2.8 or newer remain mandatory for end-to-end smoke verification.
- Use `/root/node17/data/pixal3d`, `/root/data2/pixal3d`, and `/root/data3/pixal3d` as configured roots.

---

### Task 1: Publish the 3D-FUTURE content contract

**Files:**
- Modify: `tests/data_toolkit/test_dataset_adapters.py`
- Modify: `data_toolkit/datasets/3D-FUTURE.py`

**Interfaces:**
- Consumes: canonical rows with `sha256` equal to the selected `image.jpg` SHA-256 and `file_identifier` equal to the archive directory.
- Produces: download rows with `sha256: str`, `local_path: str`, `content_sha256: str`, and `companion_files: str` containing deterministic compact JSON `{relative_path: sha256}`.

- [ ] **Step 1: Extend the adapter test with differing identity/content hashes and companion files**

Update `test_3d_future_extracts_only_selected_verified_directory` so the fixture contains `raw_model.obj`, `model.mtl`, `texture.png`, and `image.jpg`, and assert:

```python
mesh = b"mtllib model.mtl\nmesh"
material = b"map_Kd texture.png\n"
texture = b"texture"
expected_companions = json.dumps(
    {
        "raw/3D-FUTURE-model/selected/image.jpg": sha256(selected_image).hexdigest(),
        "raw/3D-FUTURE-model/selected/model.mtl": sha256(material).hexdigest(),
        "raw/3D-FUTURE-model/selected/texture.png": sha256(texture).hexdigest(),
    },
    sort_keys=True,
    separators=(",", ":"),
)
assert result.to_dict("records") == [{
    "sha256": sha256(selected_image).hexdigest(),
    "local_path": "raw/3D-FUTURE-model/selected/raw_model.obj",
    "content_sha256": sha256(mesh).hexdigest(),
    "companion_files": expected_companions,
}]
```

- [ ] **Step 2: Run the focused adapter test and confirm RED**

Run:

```bash
conda run --no-capture-output -n pixal3d python -m pytest \
  tests/data_toolkit/test_dataset_adapters.py::test_3d_future_extracts_only_selected_verified_directory -v
```

Expected: FAIL because the adapter does not publish `content_sha256` or `companion_files`.

- [ ] **Step 3: Implement deterministic adapter metadata**

In `data_toolkit/datasets/3D-FUTURE.py`, import `json`, hash the primary OBJ independently, and serialize every other selected regular archive member:

```python
primary_relative = f"raw/{identifier}/raw_model.obj"
companions = {}
for member in selected:
    destination = _safe_destination(raw_dir, member.filename)
    relative = f"raw/{member.filename}"
    if relative == primary_relative:
        continue
    companions[relative] = get_file_hash(str(destination))
return {
    "sha256": actual_sha256,
    "local_path": primary_relative,
    "content_sha256": get_file_hash(str(model_path)),
    "companion_files": json.dumps(
        companions, sort_keys=True, separators=(",", ":")
    ),
}
```

Return a DataFrame with all four columns even when no record succeeds.

- [ ] **Step 4: Run adapter tests and confirm GREEN**

Run:

```bash
conda run --no-capture-output -n pixal3d python -m pytest \
  tests/data_toolkit/test_dataset_adapters.py -v
```

Expected: all adapter tests PASS.

- [ ] **Step 5: Commit the adapter contract**

```bash
git add data_toolkit/datasets/3D-FUTURE.py tests/data_toolkit/test_dataset_adapters.py
git commit -m "fix: separate 3D-FUTURE identity and content hashes"
```

### Task 2: Normalize primary and companion raw records

**Files:**
- Modify: `tests/data_toolkit/test_orchestrator.py`
- Modify: `data_toolkit/pipeline/orchestrator.py`

**Interfaces:**
- Consumes: CSV rows with required `sha256`, `local_path` and optional `content_sha256`, `companion_files`.
- Produces: normalized records `{"sha256": str, "local_path": str, "content_sha256": str, "companion_files": dict[str, str]}` and `_raw_file_map(records) -> dict[str, str]`.

- [ ] **Step 1: Add normalization and backward-compatibility tests**

Add tests that call `_read_raw_records` through `stage_raw`:

```python
def test_stage_raw_defaults_primary_content_hash_to_asset_identity(...):
    # Existing two-column metadata stages exactly as before.

def test_stage_raw_rejects_invalid_companion_metadata(...):
    # Parametrize traversal, primary-path duplication, non-hex hashes,
    # non-object JSON, and duplicate paths across selected assets.
```

For valid extended metadata, write `companion_files` using compact JSON and assert the staged metadata contains all four columns.

- [ ] **Step 2: Run focused orchestrator tests and confirm RED**

Run:

```bash
conda run --no-capture-output -n pixal3d python -m pytest \
  tests/data_toolkit/test_orchestrator.py \
  -k 'stage_raw and (content_hash or companion or defaults_primary)' -v
```

Expected: new tests FAIL because optional raw fields are not parsed.

- [ ] **Step 3: Implement normalized records and a flattened file map**

In `PipelineServices._read_raw_record_map`:

```python
content_sha = row.get("content_sha256") or sha
content_sha = _validated_asset_sha(content_sha)
raw_companions = row.get("companion_files") or "{}"
companions = json.loads(raw_companions)
if not isinstance(companions, dict):
    raise ValidationError("invalid raw companion mapping")
normalized_companions = {
    _safe_raw_relative(path).as_posix(): _validated_asset_sha(digest)
    for path, digest in companions.items()
}
```

Reject the primary path in companions and reject every path used by more than
one selected record. Add:

```python
@staticmethod
def _raw_file_map(records: Sequence[Mapping[str, object]]) -> dict[str, str]:
    files = {}
    for record in records:
        candidates = {
            record["local_path"]: record["content_sha256"],
            **record["companion_files"],
        }
        for path, digest in candidates.items():
            if path in files:
                raise ValidationError(f"duplicate selected raw path: {path}")
            files[path] = digest
    return files
```

Make `_write_raw_records` serialize `companion_files` with sorted compact JSON
and always write the four-column schema.

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Run the command from Step 2. Expected: all selected tests PASS.

- [ ] **Step 5: Commit metadata normalization**

```bash
git add data_toolkit/pipeline/orchestrator.py tests/data_toolkit/test_orchestrator.py
git commit -m "feat: normalize raw content and companion hashes"
```

### Task 3: Stage and verify the complete declared raw set

**Files:**
- Modify: `tests/data_toolkit/test_orchestrator.py`
- Modify: `data_toolkit/pipeline/orchestrator.py`

**Interfaces:**
- Consumes: normalized raw records and `_raw_file_map` from Task 2.
- Produces: checksum-verified primary and companion files under `context.download_root`.

- [ ] **Step 1: Add a failing full-package staging test**

Create primary OBJ, MTL, texture, and image fixtures under `context.source_root`.
Use an asset identity different from the OBJ hash, publish the extended raw
metadata, run `stage_raw`, and assert every file is copied byte-for-byte. Add a
second test that changes one companion after metadata creation and expects:

```python
with pytest.raises(ValidationError, match="raw checksum mismatch: .*texture.png"):
    services.stage_raw(context)
```

- [ ] **Step 2: Run the new tests and confirm RED**

```bash
conda run --no-capture-output -n pixal3d python -m pytest \
  tests/data_toolkit/test_orchestrator.py \
  -k 'stage_raw and (full_declared or companion_checksum)' -v
```

Expected: primary identity/content mismatch or missing companion output causes FAIL.

- [ ] **Step 3: Stage all normalized files with their content hashes**

Replace the single-record copy loop in `stage_raw` with iteration over
`_raw_file_map(records).items()`. Reuse the existing safe regular-file and ZIP
member branches for each `(relative_path, expected_content_sha256)`. Preserve
the original normalized records when writing staged metadata.

Update `_validate_staged_raw` to iterate over the same flattened file map and
compare each staged stream to its content checksum.

- [ ] **Step 4: Run all stage_raw tests and confirm GREEN**

```bash
conda run --no-capture-output -n pixal3d python -m pytest \
  tests/data_toolkit/test_orchestrator.py -k stage_raw -v
```

Expected: all stage_raw tests PASS.

- [ ] **Step 5: Commit complete staging**

```bash
git add data_toolkit/pipeline/orchestrator.py tests/data_toolkit/test_orchestrator.py
git commit -m "fix: stage complete checksum-bound raw packages"
```

### Task 4: Archive and audit every declared raw file

**Files:**
- Modify: `tests/data_toolkit/test_orchestrator.py`
- Modify: `data_toolkit/pipeline/orchestrator.py`

**Interfaces:**
- Consumes: normalized `_read_raw_records` and `_raw_file_map`.
- Produces: raw pack manifests whose member map contains every primary and companion path with its content hash.

- [ ] **Step 1: Add failing archive manifest and cleanup tests**

Extend the raw archive test fixture with a primary and two companions. Assert:

```python
assert {item["path"]: item["sha256"] for item in manifest["members"]} == {
    primary_path: primary_content_sha,
    mtl_path: mtl_sha,
    texture_path: texture_sha,
}
```

Assert `pending_references` is called only with the primary canonical raw path,
and when it returns zero all three source files are removed. When it returns a
positive count, all three remain.

- [ ] **Step 2: Run raw archive tests and confirm RED**

```bash
conda run --no-capture-output -n pixal3d python -m pytest \
  tests/data_toolkit/test_orchestrator.py \
  -k 'raw_archive and (companion or content_hash or cleanup)' -v
```

Expected: FAIL because only the primary path is archived and its identity hash is expected.

- [ ] **Step 3: Flatten archive members and group cleanup by asset**

In `archive_raw`, build `file_map = self._raw_file_map(records)`, pass all
sorted paths to `build_pack`, and compare manifest members to `file_map`.

In `_verify_raw_archive`, derive the expected complete file map from the
completed records. Replace the member-count-equals-completed-count condition
with equality of the complete manifest member map.

For source cleanup, call `pending_references` once per record using its primary
`local_path`. If zero, remove the primary and its declared companions; if
positive, remove none of the group.

- [ ] **Step 4: Run archive and orchestrator tests and confirm GREEN**

```bash
conda run --no-capture-output -n pixal3d python -m pytest \
  tests/data_toolkit/test_orchestrator.py -v
```

Expected: all orchestrator tests PASS.

- [ ] **Step 5: Commit raw package archival**

```bash
git add data_toolkit/pipeline/orchestrator.py tests/data_toolkit/test_orchestrator.py
git commit -m "fix: archive complete raw asset packages"
```

### Task 5: Verify compatibility and document the contract

**Files:**
- Modify: `data_toolkit/README.md`
- Modify: `docs/data_preprocessing_runbook_ko.md`

**Interfaces:**
- Consumes: the completed raw metadata contract.
- Produces: operator documentation and regression evidence.

- [ ] **Step 1: Document canonical identity versus raw content hashes**

Add a concise 3D-FUTURE section stating that `sha256` verifies `image.jpg`,
`content_sha256` verifies `raw_model.obj`, and `companion_files` binds sibling
dependencies. Document that no operator should rewrite the canonical registry
SHA to the OBJ hash.

- [ ] **Step 2: Run focused tests**

```bash
conda run --no-capture-output -n pixal3d python -m pytest \
  tests/data_toolkit/test_dataset_adapters.py \
  tests/data_toolkit/test_orchestrator.py -v
```

Expected: PASS.

- [ ] **Step 3: Run the complete data_toolkit suite**

```bash
conda run --no-capture-output -n pixal3d python -m pytest tests/data_toolkit -v
```

Expected: all tests PASS.

- [ ] **Step 4: Run static verification**

```bash
conda run --no-capture-output -n pixal3d python -m compileall -q data_toolkit
git diff --check
git status --short
```

Expected: compile and diff checks succeed; only intended documentation changes remain.

- [ ] **Step 5: Commit documentation**

```bash
git add data_toolkit/README.md docs/data_preprocessing_runbook_ko.md
git commit -m "docs: explain 3D-FUTURE raw hash contract"
```

### Task 6: Recover and rerun the frozen 3D-FUTURE smoke shard

**Files:**
- Create: `/root/data2/pixal3d/control/recovery/3d-future-<timestamp>/`
- Create: `/root/data3/pixal3d/archive/recovery/3d-future-<timestamp>/`
- Replace after backup: 3D-FUTURE smoke checkpoints, quality ledger, prepared indexes/packs, raw archives, and escalation reports.
- Preserve: `/root/data2/pixal3d/control/qualification/smoke/shards/3D-FUTURE/3D-FUTURE-00000/`

**Interfaces:**
- Consumes: the unchanged frozen nine-asset scope, canonical 3D-FUTURE ZIP, and the fixed current commit.
- Produces: rebuilt raw metadata, terminal checkpoints, packs, raw archives, quarantine evidence, and a smoke audit result.

- [ ] **Step 1: Verify the fixed code and frozen scope before mutation**

Run the focused/full tests from Task 5, record `git rev-parse HEAD`, and verify
the three frozen batch files still contain the original nine sorted asset IDs.

- [ ] **Step 2: Back up every mutable 3D-FUTURE smoke artifact**

Create a timestamped recovery directory. Copy the 3D-FUTURE checkpoint tree,
schema-v2 quality ledger, prepared index, prepared tar manifests, raw archive
manifests, and escalation reports. Record and verify SHA-256 inventory files.
Move the existing empty prepared tar files into the data2 recovery tree and the
empty raw archives into the data3 recovery tree. Keep each move on its original
filesystem so canonical publication paths become available without copying or
deleting evidence.

- [ ] **Step 3: Reset only false-quarantine execution state**

Keep the frozen `batch000.txt` through `batch002.txt` and `batches.json`
unchanged. Remove the backed-up live batch checkpoints and source quality
ledger so the runner creates new durable state. Do not modify the canonical
training registry or its SHA-256 identities.

- [ ] **Step 4: Rebuild selected raw metadata with the fixed adapter**

Run:

```bash
conda run --no-capture-output -n pixal3d python data_toolkit/download.py 3D-FUTURE \
  --root /root/data2/pixal3d/control/metadata/3D-FUTURE \
  --download_root /root/data2/pixal3d/raw/3D-FUTURE \
  --instances /root/data2/pixal3d/control/qualification/smoke/shards/3D-FUTURE/3D-FUTURE-00000/batch000.txt \
  --max_workers 8
```

Repeat for `batch001.txt` and `batch002.txt`. Verify each selected row has a
canonical `sha256`, distinct `content_sha256`, and non-empty
`companion_files`.

- [ ] **Step 5: Run the complete frozen smoke shard**

```bash
conda run --no-capture-output -n pixal3d python -m data_toolkit.pipeline.cli resume \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --gate smoke \
  --source 3D-FUTURE \
  --shard 3D-FUTURE-00000
```

Monitor each batch until it has 26 completed commands, no active attempt, and
one terminal outcome per frozen asset. Newly reproducible asset failures are
quarantined with their actual stage/reason; the previous checksum-mismatch
records are not restored.

- [ ] **Step 6: Audit 3D-FUTURE under the producing commit**

```bash
conda run --no-capture-output -n pixal3d python -m data_toolkit.pipeline.cli audit \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --gate smoke \
  --source 3D-FUTURE \
  --shard 3D-FUTURE-00000
```

Expected: exit 0. Record the command, current commit, UTC timestamp, completed
count, quarantined count, and audit exit status in the recovery directory.
