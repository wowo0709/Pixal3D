# ABO Production Valid-Subset Materialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate, and operate a non-destructive pipeline that turns the user-approved valid ABO production subset into four stage-isolated Pixal3D training roots and publishes an ABO-specific immutable handoff.

**Architecture:** A production materializer reads the canonical production index, verifies each required pack once, computes exact per-stage family intersections, and atomically extracts only admitted assets. A separate strict preflight validates every published artifact and calls the real dataset loader directly for both anchors of every asset before a create-only report, handoff, and stage `data_dir` manifest may be published.

**Tech Stack:** Python 3.11, pytest, pathlib, dataclasses, tarfile, hashlib, csv, NumPy, Pillow, PyTorch CPU, existing Pixal3D dataset classes, existing `data_toolkit.pipeline.packing.verify_pack`.

## Global Constraints

- The accepted input is the valid ABO subset even though the original 90% source gate failed.
- Frozen scope is exactly 4,485; global quarantine is exactly 825.
- Stage counts are exactly `ss64=3660`, `shape512=3631`, `shape1024=3660`, and `pbr1024=3660`.
- Shape-512 has exactly 29 additional family exclusions.
- Read only immutable packs rooted at `/root/data2/pixal3d/prepared`; never read active preprocessing scratch.
- Publish only under `/root/node17/data/pixal3d/train/production/abo/{ss64,shape512,shape1024,pbr1024}/active`.
- Never overwrite an existing `active` stage root.
- Extract through a hidden temporary sibling and publish with `os.replace`.
- Every selected member must be a safe regular tar member whose size and SHA-256 match its manifest.
- Every component `metadata.csv` must contain the exact admitted stage scope with explicit `True` flags.
- Strict preflight must inspect every admitted asset and directly load both anchor 0 and anchor 1 without `Dataset.__getitem__` retry.
- Every `total_scale` must be finite and strictly positive after float32 conversion.
- PBR and Shape coordinates and float32 `total_scale` must match exactly for the same asset and anchor.
- Preflight and production operation are CPU-only and default to one worker.
- Shared report and handoff paths use create-only semantics; different existing content is an error.
- Handoff records `acceptance_mode=valid_subset_user_waiver` and must not claim the original 90% gate passed.
- Do not start fine-tuning in this plan.
- Do not change pilot materializer behavior, model architecture, dataset training behavior, trainer behavior, checkpoints, batch policy, snapshot policy, or W&B behavior.

---

## File Structure

- Create `scripts/materialize_multiview_production.py`: production index catalog, pack verification, selected safe extraction, deterministic metadata, atomic stage publication.
- Create `scripts/preflight_multiview_production.py`: full structural validation, direct real-loader validation, report/handoff/training-data publication.
- Create `tests/multiview/test_production_materialization.py`: synthetic production pack and materialization behavior.
- Create `tests/multiview/test_production_preflight.py`: strict validation, real-loader, and handoff behavior.
- Modify `docs/data_preprocessing_runbook_ko.md`: operator commands, user-approved waiver, output paths, stop-before-training boundary.

---

### Task 1: Production Pack Catalog and Atomic Stage Materializer

**Files:**
- Create: `scripts/materialize_multiview_production.py`
- Create: `tests/multiview/test_production_materialization.py`

**Interfaces:**
- Produces:
  - `FamilyPack` frozen dataclass with fields `batch_id`, `family`, `pack`, `manifest`, `frozen_assets`, `included_assets`, `members`, `config_hash`, `tool_commit`, `pack_sha256`, and `manifest_sha256`.
  - `load_production_catalog(index_path: Path, prepared_root: Path, source: str, shard_id: str, *, expected_batches: Sequence[str] = EXPECTED_BATCHES) -> dict[str, tuple[FamilyPack, ...]]`.
  - `compute_stage_scopes(catalog: Mapping[str, Sequence[FamilyPack]], expected_counts: Mapping[str, int] | None = None) -> dict[str, tuple[str, ...]]`.
  - `materialize_stage(stage: str, catalog: Mapping[str, Sequence[FamilyPack]], output_root: Path, *, index_path: Path, expected_counts: Mapping[str, int] = EXPECTED_STAGE_COUNTS) -> Path`.
  - `materialize_all(index_path: Path, prepared_root: Path, output_root: Path, *, expected_batches: Sequence[str] = EXPECTED_BATCHES, expected_counts: Mapping[str, int] = EXPECTED_STAGE_COUNTS) -> dict[str, Path]`.
- Consumed by Task 2:
  - exact output layout and `materialization.json`.
- Required stage-family mapping:

```python
STAGE_FAMILIES = {
    "ss64": ("common", "SS-64"),
    "shape512": ("common", "shape-512"),
    "shape1024": ("common", "shape-1024"),
    "pbr1024": ("common", "shape-1024", "PBR-1024"),
}
EXPECTED_STAGE_COUNTS = {
    "ss64": 3660,
    "shape512": 3631,
    "shape1024": 3660,
    "pbr1024": 3660,
}
```

- Required defaults:

```python
DEFAULT_INDEX = Path("/root/data2/pixal3d/prepared/index/ABO/ABO-00000.json")
DEFAULT_PREPARED = Path("/root/data2/pixal3d/prepared")
DEFAULT_OUTPUT = Path("/root/node17/data/pixal3d/train/production/abo")
SOURCE = "ABO"
SHARD_ID = "ABO-00000"
EXPECTED_BATCHES = tuple(f"batch{index:03d}" for index in range(18))
```

- [ ] **Step 1: Add synthetic pack fixtures and failing catalog tests**

Create a test-local `write_pack` that writes real tar bytes and a schema-2
manifest with hand-derived literal fields. Create a two-batch index containing
all eight production family keys, while only the five required families need
non-empty included scopes.

Add tests that fail because `FamilyPack`, `load_production_catalog`, and
`compute_stage_scopes` do not exist:

```python
def test_catalog_computes_exact_family_intersections(tmp_path):
    # common/SS/shape1024/PBR1024 contain asset_a and asset_b.
    # shape512 contains only asset_a.
    catalog = load_production_catalog(
        index,
        prepared,
        "ABO",
        "ABO-00000",
        expected_batches=("batch000", "batch001"),
    )
    assert compute_stage_scopes(catalog) == {
        "ss64": (asset_a, asset_b),
        "shape512": (asset_a,),
        "shape1024": (asset_a, asset_b),
        "pbr1024": (asset_a, asset_b),
    }
```

Add independent rejection tests for:

- wrong index source/shard/gate;
- missing or extra batch keys;
- missing family key;
- symlink, non-regular, missing, or zero-byte pack/manifest;
- index manifest digest mismatch;
- index pack digest disagreement with manifest;
- wrong manifest batch/family/gate;
- non-canonical or duplicate included scope;
- missing `validated_at`.

Each test names the production mutation it catches and asserts a contextual
`ValueError` or existing `ValidationError`.

- [ ] **Step 2: Run catalog tests and verify RED**

Run:

```bash
conda run --no-capture-output -n pixal3d python -m pytest -q \
  tests/multiview/test_production_materialization.py
```

Expected: collection fails because
`scripts.materialize_multiview_production` does not exist.

- [ ] **Step 3: Implement the minimal validated catalog**

Implement strict JSON/type/path checks. Resolve every index-relative pack and
manifest beneath `prepared_root`; reject absolute or escaping paths. Require
all eight family records in every expected batch, but return only the five
families in `STAGE_FAMILIES`.

For each required record:

1. require regular, non-symlink, non-empty pack and manifest;
2. compare SHA-256 of manifest bytes to the index;
3. parse schema-2 manifest;
4. require matching production identity, batch, family, counts, and
   non-empty `validated_at`;
5. require index `pack_sha256 == manifest["pack_sha256"]`;
6. call `verify_pack(pack, manifest)` once;
7. record canonical frozen/included tuples and manifest members.

Compute each stage scope as the sorted set intersection of all required family
included assets. Reject duplicate assets across batches and require the
production CLI path to match `EXPECTED_STAGE_COUNTS`. Unit tests may pass an
explicit expected-count mapping to exercise small fixtures:

```python
def compute_stage_scopes(catalog, expected_counts=None):
    scopes = {}
    for stage, families in STAGE_FAMILIES.items():
        family_sets = [
            {
                asset
                for record in catalog[family]
                for asset in record.included_assets
            }
            for family in families
        ]
        scopes[stage] = tuple(sorted(set.intersection(*family_sets)))
        if expected_counts is not None:
            assert len(scopes[stage]) == expected_counts[stage]
    return scopes
```

- [ ] **Step 4: Add failing selected-extraction and publication tests**

Add real tar tests for the following observable behavior:

```python
def test_materialize_stage_extracts_only_intersection_and_exact_metadata(tmp_path):
    index, prepared, asset_a, _ = write_catalog_fixture(tmp_path)
    catalog = load_production_catalog(
        index,
        prepared,
        "ABO",
        "ABO-00000",
        expected_batches=("batch000", "batch001"),
    )
    output = tmp_path / "output"
    final = materialize_stage(
        "shape512", catalog, output, index_path=index,
        expected_counts={"shape512": 1},
    )
    assert sorted(path.name for path in (final / "renders_cond").iterdir()) == [
        asset_a,
        "metadata.csv",
    ]
    assert read_csv(final / "renders_cond/metadata.csv") == [
        {"sha256": asset_a, "cond_rendered": "True"},
    ]
```

Also require:

- rows and evidence asset scope are sorted;
- non-admitted common members are not extracted;
- unsafe member path, tar link, unexpected member, wrong member size, and
  wrong selected-member digest fail;
- existing `active` fails before a temporary directory is created;
- mid-extraction failure removes the temporary stage directory;
- successful publication leaves no temporary sibling;
- `materialization.json` records the waiver, index digest, exact count,
  stage scope digest, pack/manifest digests, and all tool commits.

- [ ] **Step 5: Run extraction tests and verify RED**

Run the same Task 1 test file. Expected: catalog tests pass and extraction /
publication tests fail because the functions are absent.

- [ ] **Step 6: Implement selected safe extraction and atomic publication**

Index manifest members by admitted asset path component. Require every selected
asset to have exactly:

- common: `renders_cond/<sha>/000.png` through `007.png` and
  `transforms.json`;
- latent family: `<family-root>/<sha>/view00.npz`,
  `view00_scale.json`, `view01.npz`, and `view01_scale.json`.

Open each already verified tar and stream only selected regular members. Resolve
the destination beneath the temporary root, copy while hashing, and compare
size/digest to the manifest before closing the file.

Write exact deterministic metadata using `csv.DictWriter`, then write
`materialization.json` with sorted keys and a terminal newline. Publish with
`os.replace(temporary, final)` and clean only the known temporary sibling on
failure.

The CLI accepts:

```text
--index
--prepared-root
--output-root
--stage (repeatable)
```

The default invocation validates exact production counts. No flag may disable
pack verification, selected-member verification, non-overwrite, or the waiver
record.

- [ ] **Step 7: Verify Task 1 GREEN and regressions**

Run:

```bash
conda run --no-capture-output -n pixal3d python -m pytest -q \
  tests/multiview/test_production_materialization.py \
  tests/multiview/test_pilot_materialization.py \
  tests/data_toolkit/test_packing.py
conda run --no-capture-output -n pixal3d python -m compileall -q \
  scripts/materialize_multiview_production.py \
  tests/multiview/test_production_materialization.py
git diff --check
```

Expected: all tests pass, compilation succeeds, diff check is silent.

- [ ] **Step 8: Self-review and commit Task 1**

Confirm no production path was created by tests and pilot code is unchanged.
Commit:

```bash
git add scripts/materialize_multiview_production.py \
  tests/multiview/test_production_materialization.py
git commit -m "feat: materialize ABO production subset"
```

---

### Task 2: Strict Structural and Direct-Loader Preflight

**Files:**
- Create: `scripts/preflight_multiview_production.py`
- Create: `tests/multiview/test_production_preflight.py`

**Interfaces:**
- Consumes:
  - Task 1 stage layout and `materialization.json`.
  - Four existing fine-tuning configs.
- Produces:
  - `StagePreflight` frozen dataclass containing `stage`, `root`,
    `asset_count`, `asset_scope_sha256`, `anchors_checked`, and
    `validation_counts`.
  - `validate_stage_structure(stage: str, root: Path, expected_assets: Sequence[str]) -> dict[str, int]`.
  - `validate_direct_loader(stage: str, root: Path, expected_assets: Sequence[str], config_path: Path) -> int`.
  - `preflight_stage(stage: str, root: Path, config_path: Path) -> StagePreflight`.
- Exact config mapping:

```python
CONFIGS = {
    "ss64": Path("configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json"),
    "shape512": Path("configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json"),
    "shape1024": Path("configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json"),
    "pbr1024": Path("configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json"),
}
```

- [ ] **Step 1: Add failing metadata, render, latent, and scale tests**

Build one-asset stage fixtures with actual RGBA PNG, transform JSON, NPZ, scale
JSON, and metadata files. Expected values are literal and independent of the
validators.

Add passing-fixture tests plus one failure test per mutation:

- metadata missing a required view column;
- `cond_rendered=False`;
- metadata/component asset-set mismatch;
- missing or extra render view;
- non-RGBA or non-512×512 PNG;
- unsafe frame path;
- non-finite or non-positive camera angle/distance;
- wrong transform shape, non-finite transform, or singular rotation;
- missing anchor NPZ/scale;
- missing/non-finite/non-positive/float32-underflow `total_scale`;
- SS wrong `z` key, dtype compatibility, finiteness, or exact shape;
- Shape/PBR wrong keys, coordinate/feature rank, row count, feature width,
  finiteness, uniqueness, integral coordinates, grid bounds, or token maximum;
- PBR/Shape coordinate mismatch;
- PBR/Shape float32 scale mismatch.

Require errors to include `source=ABO`, stage, asset SHA, and anchor where
available.

- [ ] **Step 2: Run structural tests and verify RED**

Run:

```bash
conda run --no-capture-output -n pixal3d python -m pytest -q \
  tests/multiview/test_production_preflight.py
```

Expected: collection fails because `scripts.preflight_multiview_production`
does not exist.

- [ ] **Step 3: Implement minimal full structural validators**

Use:

- `csv.DictReader` with exact required boolean string `"True"`;
- Pillow `Image.open`, `verify`, and reopened RGBA/size inspection;
- `numpy.asarray` and `numpy.linalg.det`;
- `numpy.load(path, allow_pickle=False)`;
- explicit float32 conversion before scale finite/positive/equality checks.

Validate every expected asset sequentially. Do not catch and skip validation
errors. Return counters only after the full stage passes.

Read expected assets from `materialization.json`, require its stage/count/root
identity, and compare its asset scope digest before walking files.

- [ ] **Step 4: Add failing real-loader tests**

For an SS fixture with `image_size=4` in a test config, use the real
`MultiViewImageConditionedSparseStructureLatentView`. Require the preflight to:

- construct a dataset whose instance set exactly equals expected assets;
- call `get_instance` directly for anchor 0 and 1;
- never call `dataset[index]`;
- force anchor selection while leaving the remaining condition order a valid
  permutation;
- assert output dtype, shape, finite camera/condition tensors, positive scale,
  and anchor-first indices;
- report exactly two checked anchors per asset.

Add a damaged-asset test proving direct-loader failure is surfaced instead of
being replaced by another dataset sample.

Add targeted Shape and PBR fixtures proving sparse target contracts and PBR
coordinate equality execute through the real loader.

- [ ] **Step 5: Run loader tests and verify RED**

Run Task 2 tests. Expected: structural tests pass while direct-loader tests fail
because `validate_direct_loader` / `preflight_stage` are absent.

- [ ] **Step 6: Implement direct real-loader validation**

Build the stage `data_dir` object from the fixed stage root:

```python
def stage_data_dir(stage: str, root: Path) -> dict[str, dict[str, str]]:
    values = {
        "base": str(root),
        "render_cond": str(root / "renders_cond"),
    }
    if stage == "ss64":
        values["ss_latent"] = str(
            root / "ss_latents/ss_enc_conv3d_16l8_fp16_64_view"
        )
    elif stage == "shape512":
        values["shape_latent"] = str(
            root
            / "shape_latents/shape_enc_next_dc_f16c32_fp16_512_view"
        )
    elif stage == "shape1024":
        values["shape_latent"] = str(
            root
            / "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view"
        )
    elif stage == "pbr1024":
        values["shape_latent"] = str(
            root
            / "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view"
        )
        values["pbr_latent"] = str(
            root
            / "pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix"
        )
    else:
        raise ValueError(f"unknown stage: {stage}")
    return {"ABO": values}
```

Load config JSON, construct the real configured dataset class, require exact
instance count and SHA set, and call `dataset.get_instance(root_record, sha)`
under a narrow anchor-selection patch for each anchor. Set
`dataset._current_dataset_name = "ABO"` so failures retain source context.

Require exact output contract from the design. Do not initialize a trainer,
model, CUDA, W&B, multiprocessing, or a DataLoader.

- [ ] **Step 7: Verify Task 2 GREEN and regressions**

Run:

```bash
conda run --no-capture-output -n pixal3d python -m pytest -q \
  tests/multiview/test_production_preflight.py \
  tests/multiview/test_dataset_conditions.py \
  tests/multiview/test_dataset_collation.py \
  tests/multiview/test_pilot_dataset.py
conda run --no-capture-output -n pixal3d python -m compileall -q \
  scripts/preflight_multiview_production.py \
  tests/multiview/test_production_preflight.py
git diff --check
```

Expected: all available tests pass; existing integration skips remain explicit.

- [ ] **Step 8: Self-review and commit Task 2**

Confirm every admitted asset and both anchors are iterated and no retry path is
used. Commit:

```bash
git add scripts/preflight_multiview_production.py \
  tests/multiview/test_production_preflight.py
git commit -m "feat: preflight ABO production training data"
```

---

### Task 3: Immutable Valid-Subset Handoff and Operator Runbook

**Files:**
- Modify: `scripts/preflight_multiview_production.py`
- Modify: `tests/multiview/test_production_preflight.py`
- Modify: `docs/data_preprocessing_runbook_ko.md`

**Interfaces:**
- Consumes:
  - `StagePreflight` values from Task 2.
  - Task 1 `materialization.json` documents.
- Produces:
  - `build_report(index_path: Path, index_sha256: str, results: Mapping[str, StagePreflight], materializations: Mapping[str, Mapping[str, object]], created_at: str) -> dict[str, object]`.
  - `build_handoff(report_path: Path, report_sha256: str, report: Mapping[str, object], results: Mapping[str, StagePreflight], materializations: Mapping[str, Mapping[str, object]], created_at: str) -> dict[str, object]`.
  - `write_create_only_json(path: Path, value: Mapping[str, object]) -> str`
    returning the SHA-256 of canonical JSON bytes.
  - `publish_handoff(index_path: Path, results: Mapping[str, StagePreflight], materializations: Mapping[str, Mapping[str, object]], report_path: Path, handoff_path: Path, training_data_path: Path, created_at: str) -> tuple[Path, Path, Path]` returning report, handoff, and local training-data paths.
- Fixed paths:

```python
DEFAULT_REPORT = Path(
    "/root/data2/pixal3d/control/reports/gates/ABO/"
    "ABO-00000-valid-subset.json"
)
DEFAULT_HANDOFF = Path(
    "/root/data2/pixal3d/control/splits/ABO/"
    "ABO-00000-valid-subset-handoff.json"
)
DEFAULT_TRAINING_DATA = (
    Path("/root/node17/data/pixal3d/train/production/abo")
    / "training_data.json"
)
```

- [ ] **Step 1: Add failing report and create-only tests**

Use literal synthetic `StagePreflight` values and materialization evidence.
Require:

```python
assert report["acceptance_mode"] == "valid_subset_user_waiver"
assert report["original_90_percent_gate_passed"] is False
assert report["counts"] == {
    "frozen": 4485,
    "global_quarantine": 825,
    "shape512_family_exclusions": 29,
    "stages": {
        "ss64": 3660,
        "shape512": 3631,
        "shape1024": 3660,
        "pbr1024": 3660,
    },
}
```

Add tests proving:

- handoff is not written when any stage result is absent or failed;
- report digest recorded in handoff matches canonical report bytes;
- all materialization evidence digests and all observed tool commits are
  present;
- exact stage `data_dir` objects point into their isolated roots;
- output explicitly says it authorizes training-input use only;
- create-only writer accepts identical bytes and rejects different bytes;
- local `training_data.json` is withheld until both shared files succeed.

- [ ] **Step 2: Run handoff tests and verify RED**

Run Task 2 test file. Expected: new tests fail because handoff builders and
publication do not exist.

- [ ] **Step 3: Implement canonical immutable publication**

Canonical JSON is UTF-8, `indent=2`, `sort_keys=True`, with one terminal
newline. `write_create_only_json`:

1. creates parent directories;
2. if target exists, requires a regular non-symlink file with byte-identical
   content and returns its digest;
3. otherwise writes a temporary sibling, fsyncs it, publishes with a
   create-only operation, and fsyncs the parent directory;
4. rejects a race that published different bytes.

Build the report first, then the handoff referencing its digest, then write the
local training-data manifest atomically only after both shared artifacts are
accepted.

The preflight CLI accepts:

```text
--root
--index
--report
--handoff
--training-data
```

It runs all four strict preflights before calling `publish_handoff`.

- [ ] **Step 4: Document the exact production commands**

Add a Korean runbook section recording:

- user-approved 90% waiver and exact counts;
- immutable input and isolated output paths;
- commands:

```bash
CUDA_VISIBLE_DEVICES="" nice -n 15 ionice -c 2 -n 7 \
  conda run --no-capture-output -n pixal3d \
  python scripts/materialize_multiview_production.py

CUDA_VISIBLE_DEVICES="" nice -n 15 ionice -c 2 -n 7 \
  conda run --no-capture-output -n pixal3d \
  python scripts/preflight_multiview_production.py
```

- expected report/handoff/training-data paths;
- explicit statement that these commands do not start training;
- existing `active` roots are not overwritten;
- recovery requires operator inspection rather than deletion by the command.

Human documentation receives no source-text test.

- [ ] **Step 5: Verify Task 3 GREEN and full regressions**

Run:

```bash
conda run --no-capture-output -n pixal3d python -m pytest -q \
  tests/multiview/test_production_materialization.py \
  tests/multiview/test_production_preflight.py
conda run --no-capture-output -n pixal3d python -m pytest -q tests/multiview
conda run --no-capture-output -n pixal3d python -m compileall -q \
  scripts/materialize_multiview_production.py \
  scripts/preflight_multiview_production.py \
  tests/multiview/test_production_materialization.py \
  tests/multiview/test_production_preflight.py
git diff --check
```

Expected: production tests and full multi-view suite pass, compilation
succeeds, diff check is silent.

- [ ] **Step 6: Self-review and commit Task 3**

Confirm handoff publication occurs only after all strict checks pass and no
training command appears in either script. Commit:

```bash
git add scripts/preflight_multiview_production.py \
  tests/multiview/test_production_preflight.py \
  docs/data_preprocessing_runbook_ko.md
git commit -m "feat: publish ABO valid-subset handoff"
```

---

## Post-Implementation Production Gate

This section runs only after all three tasks pass task review and the cumulative
branch passes final review.

1. Confirm no ABO preprocessing command is active and record current non-ABO
   preprocessing/GPU state.
2. Confirm at least 70 GiB free under `/root/node17/data`.
3. Run the documented materializer under `CUDA_VISIBLE_DEVICES=""`, `nice`,
   and `ionice`.
4. Inspect exact stage counts, absence of hidden temporary siblings, evidence
   digests, permissions, and output size.
5. Run strict preflight under the same CPU/I/O constraints.
6. Confirm report, handoff, and `training_data.json` exist and cross-digests
   match.
7. Independently reconstruct each configured dataset from `training_data.json`
   and verify exact lengths.
8. Confirm no CUDA context or training process was created.
9. Stop and report the release state. Fine-tuning remains a separate,
   user-directed action.
