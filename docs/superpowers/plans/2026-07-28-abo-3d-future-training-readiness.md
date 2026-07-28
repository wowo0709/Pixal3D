# ABO + 3D-FUTURE Training Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materialize and strictly validate 3D-FUTURE, bind it with the unchanged ABO handoff, and make all four multi-view configs launchable through one verified training manifest.

**Architecture:** Extract the existing ABO-only materialization and preflight behavior into source-aware reusable modules while retaining the ABO wrappers and fixed count contract. Publish source-specific immutable evidence, then publish a lightweight combined manifest whose stage `data_dir` maps contain `ABO` and `3D-FUTURE`; `train.py --training_data` resolves that manifest before any CUDA initialization.

**Tech Stack:** Python 3.10, PyTorch Dataset/DataLoader, NumPy, pandas, pytest, JSON/SHA-256 evidence, atomic filesystem publication, conda environment `pixal3d`.

## Global Constraints

- Use all stage-eligible ABO and 3D-FUTURE assets for training; create no internal train/validation/test split.
- Keep Toys4K evaluation-only.
- Use the existing unweighted concatenated instance list, so source sampling is proportional to eligible instance count.
- Do not change model architecture, model config hyperparameters, 100,000 steps, `batch_split=1`, checkpoint interval 5,000, retention 5, or `/root/data3/pixal3d/ckpts/*`.
- Reuse `/root/node17/data/pixal3d/train/production/abo` and its report/handoff byte-for-byte; never republish, overwrite, or rematerialize ABO.
- Do not initialize CUDA, start W&B, start fine-tuning, signal workers, or use GPU 0.
- Run production materialization and preflight with `nice -n 15 ionice -c2 -n7`.
- All new shared report/handoff files are create-only; local manifests are atomic and reproducible only from identical canonical content.
- Failed materializations move to a unique child of `/root/node17/data/pixal3d/train/production/rejected` on the same filesystem; do not delete them.
- Preserve the current `--data_dir` interface and fail on simultaneous explicit `--data_dir` and `--training_data`.
- The exact source names are `ABO` and `3D-FUTURE`.

---

## File Structure

- `data_toolkit/pipeline/training_eligibility.py`: source-independent eligibility policy plus backward-compatible ABO count constants.
- `data_toolkit/pipeline/training_materialization.py`: source profiles, multi-index catalog verification, scope computation, safe stage publication.
- `scripts/materialize_multiview_production.py`: thin backward-compatible CLI/wrapper over the materialization module.
- `data_toolkit/pipeline/training_preflight.py`: source-aware structural/direct-loader validation and immutable source handoff publication.
- `scripts/preflight_multiview_production.py`: thin backward-compatible CLI/wrapper over the preflight module.
- `data_toolkit/pipeline/training_manifest.py`: combined-manifest creation, digest validation, overlap checks, and launch-time resolution.
- `scripts/publish_multisource_training.py`: create the ABO + 3D-FUTURE local combined manifest.
- `scripts/preflight_multisource_training.py`: CPU-only combined Dataset/DataLoader integration validation.
- `train.py`: optional `--training_data` resolution before CUDA/model construction.
- `tests/multiview/test_training_materialization.py`: multi-index/source materialization tests.
- `tests/multiview/test_training_preflight.py`: generic source handoff tests.
- `tests/multiview/test_training_manifest.py`: combined manifest and `train.py` resolution tests.
- `tests/multiview/test_multisource_preflight.py`: synthetic two-source Dataset/DataLoader tests.
- `README.md`: exact four launch commands and safety notes.
- `docs/data_preprocessing_runbook_ko.md`: 3D-FUTURE and combined publication/recovery procedure.

---

### Task 1: Separate Eligibility Policy from the ABO Count Contract

**Files:**
- Modify: `data_toolkit/pipeline/training_eligibility.py`
- Modify: `tests/multiview/test_production_materialization.py`
- Test: `tests/multiview/test_training_materialization.py`

**Interfaces:**
- Consumes: existing `canonical_count_contract`, `filter_stage_scope`, and ABO `EXPECTED_*` constants.
- Produces:
  - `STAGES: tuple[str, ...]`
  - `ABO_COUNT_CONTRACT: dict[str, object]`
  - `observed_count_contract(*, frozen: int, candidate_stages: Mapping[str, int], training_exclusions: Mapping[str, int]) -> dict[str, object]`
  - all current `EXPECTED_*` imports remain valid aliases.

- [ ] **Step 1: Write failing policy/count tests**

Create `tests/multiview/test_training_materialization.py` with:

```python
from data_toolkit.pipeline.training_eligibility import (
    ABO_COUNT_CONTRACT,
    EXPECTED_FINAL_STAGE_COUNTS,
    observed_count_contract,
)


def test_abo_count_contract_preserves_published_values():
    assert ABO_COUNT_CONTRACT["stages"] == EXPECTED_FINAL_STAGE_COUNTS
    assert ABO_COUNT_CONTRACT["frozen"] == 4485
    assert ABO_COUNT_CONTRACT["global_quarantine"] == 825


def test_observed_count_contract_derives_final_counts():
    value = observed_count_contract(
        frozen=10,
        candidate_stages={
            "ss64": 9, "shape512": 8, "shape1024": 9, "pbr1024": 7,
        },
        training_exclusions={
            "ss64": 0, "shape512": 2, "shape1024": 1, "pbr1024": 3,
        },
    )
    assert value == {
        "frozen": 10,
        "candidate_stages": {
            "ss64": 9, "shape512": 8, "shape1024": 9, "pbr1024": 7,
        },
        "pack_exclusions": {
            "ss64": 1, "shape512": 2, "shape1024": 1, "pbr1024": 3,
        },
        "training_exclusions": {
            "ss64": 0, "shape512": 2, "shape1024": 1, "pbr1024": 3,
        },
        "stages": {
            "ss64": 9, "shape512": 6, "shape1024": 8, "pbr1024": 4,
        },
    }
```

Also add rejection tests for a missing stage, negative count, exclusions greater
than candidates, and candidate count greater than frozen.

- [ ] **Step 2: Run the new tests and confirm RED**

Run:

```bash
conda run --no-capture-output -n pixal3d \
  python -m pytest tests/multiview/test_training_materialization.py -q
```

Expected: collection failure because `ABO_COUNT_CONTRACT` and
`observed_count_contract` do not exist.

- [ ] **Step 3: Implement the minimal source-independent count builder**

Add:

```python
STAGES = ("ss64", "shape512", "shape1024", "pbr1024")


def _stage_counts(values: Mapping[str, int], label: str) -> dict[str, int]:
    result = dict(values)
    if set(result) != set(STAGES):
        raise ValueError(f"{label} must contain exactly {STAGES}")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in result.values()):
        raise ValueError(f"{label} must contain non-negative integers")
    return {stage: result[stage] for stage in STAGES}


def observed_count_contract(*, frozen, candidate_stages, training_exclusions):
    candidates = _stage_counts(candidate_stages, "candidate_stages")
    exclusions = _stage_counts(training_exclusions, "training_exclusions")
    if isinstance(frozen, bool) or not isinstance(frozen, int) or frozen < 0:
        raise ValueError("frozen must be a non-negative integer")
    if any(candidates[stage] > frozen for stage in STAGES):
        raise ValueError("candidate count exceeds frozen count")
    if any(exclusions[stage] > candidates[stage] for stage in STAGES):
        raise ValueError("training exclusion exceeds candidate count")
    return {
        "frozen": frozen,
        "candidate_stages": candidates,
        "pack_exclusions": {
            stage: frozen - candidates[stage] for stage in STAGES
        },
        "training_exclusions": exclusions,
        "stages": {
            stage: candidates[stage] - exclusions[stage] for stage in STAGES
        },
    }
```

Construct `ABO_COUNT_CONTRACT` through the existing constants. Do not alter
`policy_evidence()` or eligibility rules.

- [ ] **Step 4: Run focused and existing ABO tests**

Run:

```bash
conda run --no-capture-output -n pixal3d \
  python -m pytest \
  tests/multiview/test_training_materialization.py \
  tests/multiview/test_production_materialization.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data_toolkit/pipeline/training_eligibility.py \
  tests/multiview/test_training_materialization.py \
  tests/multiview/test_production_materialization.py
git commit -m "refactor: separate training eligibility count contracts"
```

---

### Task 2: Generalize Production Materialization to Multiple Source Indexes

**Files:**
- Create: `data_toolkit/pipeline/training_materialization.py`
- Modify: `scripts/materialize_multiview_production.py`
- Modify: `tests/multiview/test_training_materialization.py`
- Modify: `tests/multiview/test_production_materialization.py`

**Interfaces:**
- Consumes: Task 1 `ABO_COUNT_CONTRACT`, `observed_count_contract`,
  `filter_stage_scope`, and existing pack verification functions.
- Produces:
  - `ProductionSourceSpec`
  - `ABO_SOURCE_SPEC`
  - `THREED_FUTURE_SOURCE_SPEC`
  - `FamilyPack` including `source` and `shard_id`
  - `load_source_catalog(spec, prepared_root) -> dict[str, tuple[FamilyPack, ...]]`
  - `compute_stage_scopes(catalog) -> dict[str, tuple[str, ...]]`
  - `materialize_stage(spec, stage, catalog, output_root) -> Path`
  - `materialize_all(spec, prepared_root, output_root) -> dict[str, Path]`
  - existing imports from `scripts.materialize_multiview_production` remain valid.

- [ ] **Step 1: Add failing multi-index catalog tests**

Use the existing pack fixture helpers and construct:

```python
def test_3d_future_profile_has_exact_two_shards():
    assert THREED_FUTURE_SOURCE_SPEC.source == "3D-FUTURE"
    assert THREED_FUTURE_SOURCE_SPEC.expected_batches == {
        "3D-FUTURE-00000": tuple(f"batch{i:03d}" for i in range(20)),
        "3D-FUTURE-00001": tuple(f"batch{i:03d}" for i in range(18)),
    }
    assert THREED_FUTURE_SOURCE_SPEC.expected_frozen == 9472


def test_multi_index_catalog_accepts_duplicate_batch_names_across_shards(tmp_path):
    spec, prepared = make_two_shard_source(tmp_path)
    catalog = load_source_catalog(spec, prepared)
    assert {(pack.shard_id, pack.batch_id)
            for pack in catalog["common"]} == {
        ("Fixture-00000", "batch000"),
        ("Fixture-00001", "batch000"),
    }


def test_multi_index_catalog_rejects_asset_overlap_across_shards(tmp_path):
    spec, prepared = make_two_shard_source(tmp_path, overlapping_assets=True)
    with pytest.raises(ValueError, match="asset overlap across shards"):
        load_source_catalog(spec, prepared)
```

Add tests for wrong source, wrong shard identity, missing expected batch, manifest
digest mismatch, and candidate counts `8495/8513/8495/8495` in the real
3D-FUTURE profile without reading pack payloads.

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
conda run --no-capture-output -n pixal3d \
  python -m pytest tests/multiview/test_training_materialization.py -q
```

Expected: failure because `training_materialization` does not exist.

- [ ] **Step 3: Extract the reusable materialization core**

Define:

```python
@dataclass(frozen=True)
class ProductionSourceSpec:
    source: str
    indexes: tuple[Path, ...]
    expected_batches: Mapping[str, tuple[str, ...]]
    expected_frozen: int
    expected_candidate_stages: Mapping[str, int]
    fixed_count_contract: Mapping[str, object] | None
    acceptance_mode: str
    original_90_percent_gate_passed: bool


@dataclass(frozen=True)
class FamilyPack:
    source: str
    shard_id: str
    batch_id: str
    family: str
    pack: Path
    manifest: Path
    frozen_assets: tuple[str, ...]
    included_assets: tuple[str, ...]
    members: tuple[PackMember, ...]
    config_hash: str
    tool_commit: str
    pack_sha256: str
    manifest_sha256: str
```

Move safe path, regular-file, catalog, extraction, metadata, locking, eligibility,
and no-replace publication logic without weakening it. Aggregate source scopes
only after rejecting overlap between shard frozen scopes.

`THREED_FUTURE_SOURCE_SPEC` uses the two exact index paths, expected batch sets,
frozen count 9,472, and candidate stage counts from the design. Its fixed count
contract is `None`; final counts are observed.

- [ ] **Step 4: Convert the old script to a compatible wrapper and CLI**

The default command remains ABO. Add:

```text
--profile {abo,3d-future}
--stage {ss64,shape512,shape1024,pbr1024}
```

`--stage` remains repeatable. Reject custom index/output combinations that do
not match the selected profile unless tests explicitly construct a
`ProductionSourceSpec` through the Python API.

- [ ] **Step 5: Run materialization tests**

Run:

```bash
conda run --no-capture-output -n pixal3d \
  python -m pytest \
  tests/multiview/test_training_materialization.py \
  tests/multiview/test_production_materialization.py -q
```

Expected: PASS, including all old ABO tests.

- [ ] **Step 6: Run the broader CPU regression subset**

Run:

```bash
conda run --no-capture-output -n pixal3d \
  python -m pytest tests/data_toolkit tests/multiview \
  -q -m "not gpu and not integration"
```

Expected: PASS with only existing skips/warnings.

- [ ] **Step 7: Commit**

```bash
git add data_toolkit/pipeline/training_materialization.py \
  scripts/materialize_multiview_production.py \
  tests/multiview/test_training_materialization.py \
  tests/multiview/test_production_materialization.py
git commit -m "feat: support multi-shard training materialization"
```

---

### Task 3: Generalize Strict Preflight and Source Handoff Publication

**Files:**
- Create: `data_toolkit/pipeline/training_preflight.py`
- Modify: `scripts/preflight_multiview_production.py`
- Create: `tests/multiview/test_training_preflight.py`
- Modify: `tests/multiview/test_production_preflight.py`

**Interfaces:**
- Consumes: Task 2 `ProductionSourceSpec`, source profiles, stage roots, and
  `materialization.json`.
- Produces:
  - `StagePreflight`
  - `stage_data_dir(source, stage, root) -> dict[str, dict[str, str]]`
  - `preflight_stage(spec, stage, root, config_path) -> StagePreflight`
  - `build_source_report(spec, results, materializations, created_at) -> dict[str, object]`
  - `publish_source_handoff(spec, results, report_path, handoff_path, training_data_path) -> tuple[Path, Path, Path]`
  - existing ABO script imports and default CLI behavior remain valid.

- [ ] **Step 1: Add failing generic preflight tests**

Create tests including:

```python
def test_stage_data_dir_preserves_exact_source_name(tmp_path):
    assert stage_data_dir("3D-FUTURE", "ss64", tmp_path) == {
        "3D-FUTURE": {
            "base": str(tmp_path),
            "render_cond": str(tmp_path / "renders_cond"),
            "ss_latent": str(
                tmp_path / "ss_latents/ss_enc_conv3d_16l8_fp16_64_view"
            ),
        }
    }


def test_source_handoff_binds_both_index_digests(two_index_preflight):
    report = build_source_report(*two_index_preflight)
    assert [entry["shard_id"] for entry in report["source_indexes"]] == [
        "Fixture-00000", "Fixture-00001",
    ]
    assert all(len(entry["sha256"]) == 64 for entry in report["source_indexes"])


def test_observed_source_handoff_uses_materialization_counts(two_index_preflight):
    report = build_source_report(*two_index_preflight)
    assert report["counts"]["candidate_stages"]["shape512"] == 2
    assert report["counts"]["training_exclusions"]["shape512"] == 1
    assert report["counts"]["stages"]["shape512"] == 1
```

Add fail-closed tests for source mismatch in materialization evidence, changed
index bytes, changed materialization bytes, missing stage, invalid policy,
noncanonical scopes, create-only mismatch, and direct-loader instances with the
wrong source name.

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
conda run --no-capture-output -n pixal3d \
  python -m pytest tests/multiview/test_training_preflight.py -q
```

Expected: import failure because `training_preflight` does not exist.

- [ ] **Step 3: Extract source-aware validation**

Move structural, render, latent, scale, Dataset, canonical JSON, create-only,
atomic write, and evidence-binding logic into the new module. Every public
validator accepts `source` or `ProductionSourceSpec`; no generic helper reads a
module-global source.

The report stores:

```python
{
    "schema_version": 2,
    "source": spec.source,
    "source_indexes": [
        {"shard_id": shard, "path": canonical_path, "sha256": digest},
        ...
    ],
    "acceptance_mode": spec.acceptance_mode,
    "original_90_percent_gate_passed":
        spec.original_90_percent_gate_passed,
    "authorization": "training-input use only",
    "counts": count_contract,
    "eligibility_policy": policy_evidence(),
    "stages": stage_records,
    "materialization_evidence": evidence,
    "observed_tool_commits": observed_commits,
}
```

ABO wrapper output must remain schema version 1 and byte-compatible with its
already published evidence. The generic schema version 2 is used for
3D-FUTURE.

- [ ] **Step 4: Add the 3D-FUTURE CLI profile**

The command:

```bash
python scripts/preflight_multiview_production.py --profile 3d-future
```

uses:

```text
root=/root/node17/data/pixal3d/train/production/3d-future
report=/root/data2/pixal3d/control/reports/gates/3D-FUTURE/3D-FUTURE-production-training.json
handoff=/root/data2/pixal3d/control/splits/3D-FUTURE/3D-FUTURE-production-training-handoff.json
training-data=/root/node17/data/pixal3d/train/production/3d-future/training_data.json
```

- [ ] **Step 5: Run generic and legacy preflight tests**

Run:

```bash
conda run --no-capture-output -n pixal3d \
  python -m pytest \
  tests/multiview/test_training_preflight.py \
  tests/multiview/test_production_preflight.py -q
```

Expected: PASS.

- [ ] **Step 6: Verify the existing ABO evidence read-only**

Run a new `--verify-existing --profile abo` mode that performs no writes:

```bash
conda run --no-capture-output -n pixal3d \
  python scripts/preflight_multiview_production.py \
  --profile abo --verify-existing
```

Expected: exit 0 and print the existing report/handoff/training-data digests.

- [ ] **Step 7: Commit**

```bash
git add data_toolkit/pipeline/training_preflight.py \
  scripts/preflight_multiview_production.py \
  tests/multiview/test_training_preflight.py \
  tests/multiview/test_production_preflight.py
git commit -m "feat: publish source-aware training handoffs"
```

---

### Task 4: Add the Combined Manifest and Pre-CUDA Training Resolution

**Files:**
- Create: `data_toolkit/pipeline/training_manifest.py`
- Create: `scripts/publish_multisource_training.py`
- Modify: `train.py`
- Create: `tests/multiview/test_training_manifest.py`
- Modify: `tests/multiview/test_training_entrypoint.py`

**Interfaces:**
- Consumes: existing ABO schema-1 training data and Task 3 schema-2
  3D-FUTURE training data.
- Produces:
  - `SourceTrainingData`
  - `CombinedStage`
  - `build_combined_training_data(source_paths: Mapping[str, Path]) -> dict[str, object]`
  - `publish_combined_training_data(source_paths, output_path) -> Path`
  - `resolve_training_data(path: Path, stage: str) -> ResolvedTrainingData`
  - `resolve_training_input(config, cli_data_dir, cli_training_data) -> tuple[str, dict[str, object] | None]`

- [ ] **Step 1: Write failing manifest tests**

Create synthetic source handoffs and test:

```python
def test_combined_manifest_has_exact_sources_and_proportional_counts(source_inputs):
    value = build_combined_training_data(source_inputs)
    stage = value["stages"]["ss64"]
    assert list(stage["data_dir"]) == ["ABO", "3D-FUTURE"]
    assert stage["source_counts"] == {"ABO": 2, "3D-FUTURE": 5}
    assert stage["total_count"] == 7
    assert "source_weights" not in stage
    assert "split" not in stage


def test_combined_manifest_rejects_cross_source_asset_overlap(source_inputs):
    make_source_scopes_overlap(source_inputs, stage="shape1024")
    with pytest.raises(ValueError, match="cross-source asset overlap"):
        build_combined_training_data(source_inputs)


def test_resolve_training_input_rejects_both_interfaces(tmp_path, config):
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_training_input(config, "{}", tmp_path / "training_data.json")


def test_resolve_training_input_fails_before_cuda(monkeypatch, manifest, config):
    monkeypatch.setattr(
        torch.cuda, "device_count",
        lambda: pytest.fail("CUDA queried before manifest resolution"),
    )
    data_dir, evidence = resolve_training_input(config, None, manifest)
    assert json.loads(data_dir).keys() == {"ABO", "3D-FUTURE"}
    assert evidence["stage"] == "ss64"
```

Add changed-handoff-digest, symlink, missing stage, wrong component key,
noncanonical path, changed scope, and unknown `multiview_stage` tests.

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
conda run --no-capture-output -n pixal3d \
  python -m pytest tests/multiview/test_training_manifest.py -q
```

Expected: import failure because `training_manifest` does not exist.

- [ ] **Step 3: Implement strict combined-manifest creation**

Canonical source order is `("ABO", "3D-FUTURE")`. Validate source training-data
files, their referenced handoff bytes/digests, all stage records, materialized
scope lists/digests, disjoint source scopes, and exact component keys.

Write schema:

```python
{
    "schema_version": 1,
    "authorization": "training-input use only",
    "sampling": "proportional-unweighted-concatenation",
    "sources": {
        source: {
            "training_data": {"path": path, "sha256": digest},
            "handoff": {"path": handoff_path, "sha256": handoff_digest},
        }
    },
    "stages": {
        stage: {
            "source_counts": {"ABO": abo_count, "3D-FUTURE": future_count},
            "total_count": abo_count + future_count,
            "union_scope_sha256": digest,
            "data_dir": {"ABO": {...}, "3D-FUTURE": {...}},
        }
    },
}
```

Publish atomically to:

```text
/root/node17/data/pixal3d/train/production/abo-3d-future/training_data.json
```

- [ ] **Step 4: Integrate `--training_data` before CUDA**

Add parser argument:

```python
parser.add_argument(
    "--training_data",
    type=str,
    default=None,
    help="Verified combined training manifest; mutually exclusive with --data_dir",
)
parser.add_argument("--data_dir", type=str, default=None, help="Data directory")
```

Immediately after loading the config and before `torch.cuda.device_count()`:

```python
resolved_data_dir, training_evidence = resolve_training_input(
    config,
    cli_data_dir=opt.data_dir,
    cli_training_data=opt.training_data,
)
opt.data_dir = resolved_data_dir
```

If neither interface is supplied, preserve `./data/`. Store
`training_evidence` in the resolved config when present.

- [ ] **Step 5: Run manifest and entrypoint tests**

Run:

```bash
conda run --no-capture-output -n pixal3d \
  python -m pytest \
  tests/multiview/test_training_manifest.py \
  tests/multiview/test_training_entrypoint.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add data_toolkit/pipeline/training_manifest.py \
  scripts/publish_multisource_training.py train.py \
  tests/multiview/test_training_manifest.py \
  tests/multiview/test_training_entrypoint.py
git commit -m "feat: resolve verified multisource training manifests"
```

---

### Task 5: Add Combined Dataset/DataLoader Verification and Launch Documentation

**Files:**
- Create: `scripts/preflight_multisource_training.py`
- Create: `tests/multiview/test_multisource_preflight.py`
- Modify: `README.md`
- Modify: `docs/data_preprocessing_runbook_ko.md`

**Interfaces:**
- Consumes: Task 4 `resolve_training_data` and the four existing config paths.
- Produces:
  - `preflight_multisource_stage(training_data: Path, stage: str, config: Path) -> dict[str, object]`
  - CLI exit 0 only when all four combined Dataset/DataLoader checks pass.

- [ ] **Step 1: Write failing synthetic combined-loader tests**

Create two synthetic source roots per stage and assert:

```python
@pytest.mark.parametrize("stage", ("ss64", "shape512", "shape1024", "pbr1024"))
def test_combined_preflight_matches_disjoint_union(two_source_fixture, stage):
    result = preflight_multisource_stage(
        two_source_fixture.training_data,
        stage,
        CONFIGS[stage],
    )
    assert result["source_counts"] == {"ABO": 1, "3D-FUTURE": 2}
    assert result["total_count"] == 3
    assert result["sampling"] == "proportional-unweighted-concatenation"
    assert result["collated_sources"] == ["ABO", "3D-FUTURE"]
```

Also prove rejection of an omitted instance, unexpected source, duplicate SHA,
loader filtering, and a collate failure.

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
conda run --no-capture-output -n pixal3d \
  python -m pytest tests/multiview/test_multisource_preflight.py -q
```

Expected: import failure because the script does not exist.

- [ ] **Step 3: Implement CPU-only integration validation**

Patch `torch.cuda.get_device_name` only around dataset module import, as the
existing preflight does. Never call `torch.cuda.device_count`, construct a
model, or invoke a trainer.

Construct the configured Dataset from the resolved stage `data_dir`; compare
`dataset.instances` to source materialization scopes exactly. Directly call
`get_instance` for deterministic first and last assets from each source and
run the dataset's existing `collate_fn` over one sample from each source.

- [ ] **Step 4: Document publication, recovery, and exact launch commands**

Document:

```bash
TRAINING_DATA=/root/node17/data/pixal3d/train/production/abo-3d-future/training_data.json

conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json \
  --training_data "$TRAINING_DATA" --num_gpus 6 --use_wandb

conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json \
  --training_data "$TRAINING_DATA" --num_gpus 6 --use_wandb

conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  --training_data "$TRAINING_DATA" --num_gpus 6 --use_wandb

conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  --training_data "$TRAINING_DATA" --num_gpus 6 --use_wandb
```

State that the operator must set `CUDA_VISIBLE_DEVICES` only after checking
ownership and available memory; no command assumes GPU 0.

Document low-priority 3D-FUTURE materialization, preflight, combined publication,
combined verification, rejected-attempt preservation, and digest inspection.

- [ ] **Step 5: Run focused and complete CPU tests**

Run:

```bash
conda run --no-capture-output -n pixal3d \
  python -m pytest tests/multiview/test_multisource_preflight.py -q

conda run --no-capture-output -n pixal3d \
  python -m pytest tests/data_toolkit tests/multiview \
  -q -m "not gpu"
```

Expected: all selected tests pass; integration tests may skip only for
documented missing external checkpoints.

- [ ] **Step 6: Commit**

```bash
git add scripts/preflight_multisource_training.py \
  tests/multiview/test_multisource_preflight.py \
  README.md docs/data_preprocessing_runbook_ko.md
git commit -m "docs: add verified multisource training launch flow"
```

---

### Task 6: Materialize, Publish, and Verify the Real 3D-FUTURE + ABO Input

**Files:**
- Runtime create-only data:
  `/root/node17/data/pixal3d/train/production/3d-future`
- Runtime create-only shared evidence:
  `/root/data2/pixal3d/control/reports/gates/3D-FUTURE`
- Runtime create-only shared handoff:
  `/root/data2/pixal3d/control/splits/3D-FUTURE`
- Runtime atomic combined manifest:
  `/root/node17/data/pixal3d/train/production/abo-3d-future/training_data.json`
- Modify only if measured commands/counts differ:
  `docs/data_preprocessing_runbook_ko.md`

**Interfaces:**
- Consumes: Tasks 1-5 and completed production packs.
- Produces: real four-stage 3D-FUTURE handoff, combined manifest, and verified
  ready-to-run commands.

- [ ] **Step 1: Perform read-only safety checks**

Run:

```bash
pgrep -af 'materialize_multiview_production|preflight_multiview_production|preflight_multisource_training' || true
df -h /root/node17/data/pixal3d/train/production /root/data2 /root/data3
test ! -e /root/node17/data/pixal3d/train/production/3d-future
test ! -e /root/data2/pixal3d/control/reports/gates/3D-FUTURE/3D-FUTURE-production-training.json
test ! -e /root/data2/pixal3d/control/splits/3D-FUTURE/3D-FUTURE-production-training-handoff.json
```

Expected: no publisher is active, at least 150 GiB local space is available,
and all three create-only targets are absent. If a target exists, verify it;
never delete or overwrite it.

- [ ] **Step 2: Materialize all four 3D-FUTURE stages**

Run:

```bash
cd /root/dev/Pixal3D/.worktrees/multiview-model-extension
nice -n 15 ionice -c2 -n7 \
  conda run --no-capture-output -n pixal3d \
  python scripts/materialize_multiview_production.py --profile 3d-future
```

Expected: four `active` roots and four `materialization.json` files. On failure,
move only the exact failed attempt directory to a timestamped child of
`production/rejected`, then diagnose before retrying.

- [ ] **Step 3: Run strict full source preflight and publish**

Run:

```bash
nice -n 15 ionice -c2 -n7 \
  conda run --no-capture-output -n pixal3d \
  python scripts/preflight_multiview_production.py --profile 3d-future
```

Expected: all assets and both anchors pass direct real-loader validation, then
the report, handoff, and local source training data are published.

- [ ] **Step 4: Verify unchanged ABO and publish the combined manifest**

Run:

```bash
conda run --no-capture-output -n pixal3d \
  python scripts/preflight_multiview_production.py \
  --profile abo --verify-existing

conda run --no-capture-output -n pixal3d \
  python scripts/publish_multisource_training.py
```

Expected: ABO digests match existing evidence and combined publication reports
exact per-source and total stage counts.

- [ ] **Step 5: Verify all four real combined loaders**

Run:

```bash
nice -n 15 ionice -c2 -n7 \
  conda run --no-capture-output -n pixal3d \
  python scripts/preflight_multisource_training.py \
  --training-data \
  /root/node17/data/pixal3d/train/production/abo-3d-future/training_data.json
```

Expected: all four stages pass exact union and cross-source collate checks with
no CUDA initialization.

- [ ] **Step 6: Render the four commands without launching them**

Run:

```bash
rg -n -A8 'ABO \\+ 3D-FUTURE' README.md
ps -eo pid,cmd | rg 'python train.py|wandb' | rg -v 'rg python' || true
```

Expected: four documented commands are present and no new training/W&B process
exists.

- [ ] **Step 7: Run final verification**

Run:

```bash
git diff --check
conda run --no-capture-output -n pixal3d \
  python -m pytest tests/data_toolkit tests/multiview -q -m "not gpu"
git status --short --branch
```

Expected: no whitespace errors, all CPU tests pass, and only intentional
evidence-count documentation changes remain.

- [ ] **Step 8: Commit measured documentation only if needed**

If the observed final 3D-FUTURE counts were not already written by Task 5,
record the exact counts and evidence paths, then:

```bash
git add docs/data_preprocessing_runbook_ko.md
git commit -m "docs: record 3D-FUTURE training handoff counts"
```

Do not commit runtime data, reports, handoffs, or local manifests.

---

## Final Review and Publication

- [ ] Generate a whole-branch review package from commit `132689b` to `HEAD`.
- [ ] Dispatch a final reviewer for spec compliance, code quality, evidence
  safety, backward compatibility, and absence of CUDA/W&B/training side effects.
- [ ] Address any load-bearing findings and rerun the affected tests.
- [ ] Run `superpowers:verification-before-completion`.
- [ ] Push `feature/multiview-model-extension` and update existing draft PR #2;
  do not create a duplicate PR.
