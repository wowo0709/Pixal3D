# HSSD Node16 Training Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish HSSD as a strict standalone training source and publish an unweighted ABO + 3D-FUTURE + HSSD bundle that is fully materialized, verified, and launchable from Node16 local storage.

**Architecture:** Introduce root-aware immutable source profiles, extend the existing source materializer and preflight without changing their create-only semantics, and make combined manifests support the existing two-source bundle plus an explicit three-source bundle. A CPU-only Node16 preparation driver materializes all sources from shared production packs, publishes trust chains, validates real configured loaders, creates `num_workers=1` runtime configs, and emits launch evidence without starting training.

**Tech Stack:** Python 3.11, pathlib/dataclasses, JSON and SHA-256 evidence, tar production packs, NumPy/PyTorch CPU DataLoader validation, pytest, Bash/SSH deployment.

## Global Constraints

- HSSD production scope is exactly two indexes, 27 batches, 6,670 frozen assets, and 6,078 pre-eligibility candidates for every stage.
- HSSD source order follows `ABO`, `3D-FUTURE`, `HSSD` in the three-source bundle.
- Combined sampling is proportional unweighted concatenation; no source weights or oversampling are added.
- Existing ABO + 3D-FUTURE manifests remain byte-schema compatible and resolvable.
- Node16 source packs are read from `/file2/youngwoo/pixal3d/prepared`.
- Node16 materialized data is written create-only below `/home/youngwoo/data/pixal3d/train/production`.
- Preparation runs with `CUDA_VISIBLE_DEVICES=""` and starts no model, trainer, W&B run, or GPU context.
- Existing artifacts are reused only after strict validation; no production artifact is deleted, truncated, replaced, or silently repaired.
- Node16 runtime configs differ from reviewed production configs only at `trainer.args.num_workers=1`.
- SS64 and Shape512 retain batch `8/4`, global batch `48`; Shape1024 and PBR1024 retain batch `2/1`, global batch `12`.
- All four stages retain `max_steps=20000`, `i_save=2000`, `i_sample=-1`, and `max_checkpoints=5`.

---

### Task 1: Add Root-Aware Source Profiles and the HSSD Contract

**Files:**
- Create: `data_toolkit/pipeline/training_source_profiles.py`
- Modify: `data_toolkit/pipeline/training_materialization.py`
- Create: `tests/multiview/test_training_source_profiles.py`
- Modify: `tests/multiview/test_training_materialization.py`

**Interfaces:**
- Produces: `ProductionSourceSpec`, `SOURCE_PROFILE_NAMES`, `SOURCE_ACCEPTANCE_CONTRACTS`, `build_source_spec(profile: str, data2_root: Path) -> ProductionSourceSpec`, and `source_output_root(profile: str, local_root: Path) -> Path`.
- Consumes: the existing `STAGE_FAMILIES` and eligibility count contract.

- [ ] **Step 1: Write failing HSSD and root-relocation tests**

Create `tests/multiview/test_training_source_profiles.py` with these contracts:

```python
from pathlib import Path

import pytest

from data_toolkit.pipeline.training_source_profiles import (
    SOURCE_ACCEPTANCE_CONTRACTS,
    SOURCE_PROFILE_NAMES,
    build_source_spec,
    source_output_root,
)


def test_hssd_profile_pins_completed_two_shard_contract():
    spec = build_source_spec(
        "hssd", Path("/file2/youngwoo/pixal3d")
    )
    assert spec.source == "HSSD"
    assert spec.indexes == (
        Path(
            "/file2/youngwoo/pixal3d/prepared/index/HSSD/"
            "HSSD-00000.json"
        ),
        Path(
            "/file2/youngwoo/pixal3d/prepared/index/HSSD/"
            "HSSD-00001.json"
        ),
    )
    assert spec.expected_batches == {
        "HSSD-00000": tuple(f"batch{i:03d}" for i in range(20)),
        "HSSD-00001": tuple(f"batch{i:03d}" for i in range(7)),
    }
    assert spec.expected_frozen == 6670
    assert spec.expected_candidate_stages == {
        "ss64": 6078,
        "shape512": 6078,
        "shape1024": 6078,
        "pbr1024": 6078,
    }
    assert spec.fixed_count_contract is None
    assert spec.acceptance_mode == "production_gate"
    assert spec.original_90_percent_gate_passed is True
    assert SOURCE_ACCEPTANCE_CONTRACTS["HSSD"] == (
        "production_gate", True
    )


def test_profiles_relocate_indexes_without_changing_contracts():
    root = Path("/srv/data2/pixal3d")
    abo = build_source_spec("abo", root)
    future = build_source_spec("3d-future", root)
    assert abo.indexes == (
        root / "prepared/index/ABO/ABO-00000.json",
    )
    assert future.indexes[0] == (
        root
        / "prepared/index/3D-FUTURE/3D-FUTURE-00000.json"
    )
    assert future.indexes[1] == (
        root
        / "prepared/index/3D-FUTURE/3D-FUTURE-00001.json"
    )


def test_source_output_roots_are_node_local():
    local = Path("/home/youngwoo/data/pixal3d")
    assert source_output_root("abo", local) == (
        local / "train/production/abo"
    )
    assert source_output_root("3d-future", local) == (
        local / "train/production/3d-future"
    )
    assert source_output_root("hssd", local) == (
        local / "train/production/hssd"
    )


def test_unknown_profile_is_rejected():
    assert SOURCE_PROFILE_NAMES == ("abo", "3d-future", "hssd")
    with pytest.raises(ValueError, match="unknown source profile"):
        build_source_spec("toys4k", Path("/file2/youngwoo/pixal3d"))
```

Add a regression to `tests/multiview/test_training_materialization.py` proving
the old exported constants retain their Node17 paths and values.

- [ ] **Step 2: Run the new tests and observe RED**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/multiview/test_training_source_profiles.py \
  tests/multiview/test_training_materialization.py
```

Expected: collection fails because `training_source_profiles` does not exist.

- [ ] **Step 3: Implement the profile module**

Create `data_toolkit/pipeline/training_source_profiles.py` with:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from data_toolkit.pipeline.training_eligibility import ABO_COUNT_CONTRACT

STAGES = ("ss64", "shape512", "shape1024", "pbr1024")
SOURCE_PROFILE_NAMES = ("abo", "3d-future", "hssd")
SOURCE_ACCEPTANCE_CONTRACTS = {
    "ABO": ("valid_subset_user_waiver", False),
    "3D-FUTURE": ("valid_subset_user_waiver", False),
    "HSSD": ("production_gate", True),
}


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


def build_source_spec(
    profile: str, data2_root: Path
) -> ProductionSourceSpec:
    root = Path(data2_root)
    definitions = {
        "abo": {
            "source": "ABO",
            "shards": {"ABO-00000": 18},
            "frozen": 4485,
            "candidates": {
                "ss64": 3660,
                "shape512": 3631,
                "shape1024": 3660,
                "pbr1024": 3660,
            },
            "fixed": ABO_COUNT_CONTRACT,
        },
        "3d-future": {
            "source": "3D-FUTURE",
            "shards": {
                "3D-FUTURE-00000": 20,
                "3D-FUTURE-00001": 18,
            },
            "frozen": 9472,
            "candidates": {
                "ss64": 8495,
                "shape512": 8513,
                "shape1024": 8495,
                "pbr1024": 8495,
            },
            "fixed": None,
        },
        "hssd": {
            "source": "HSSD",
            "shards": {"HSSD-00000": 20, "HSSD-00001": 7},
            "frozen": 6670,
            "candidates": dict.fromkeys(STAGES, 6078),
            "fixed": None,
        },
    }
    try:
        value = definitions[profile]
    except KeyError as error:
        raise ValueError(f"unknown source profile: {profile}") from error
    source = value["source"]
    shards = value["shards"]
    acceptance_mode, gate_passed = SOURCE_ACCEPTANCE_CONTRACTS[source]
    return ProductionSourceSpec(
        source=source,
        indexes=tuple(
            root / "prepared/index" / source / f"{shard}.json"
            for shard in shards
        ),
        expected_batches={
            shard: tuple(f"batch{i:03d}" for i in range(count))
            for shard, count in shards.items()
        },
        expected_frozen=value["frozen"],
        expected_candidate_stages=value["candidates"],
        fixed_count_contract=value["fixed"],
        acceptance_mode=acceptance_mode,
        original_90_percent_gate_passed=gate_passed,
    )


def source_output_root(profile: str, local_root: Path) -> Path:
    names = {"abo": "abo", "3d-future": "3d-future", "hssd": "hssd"}
    try:
        name = names[profile]
    except KeyError as error:
        raise ValueError(f"unknown source profile: {profile}") from error
    return Path(local_root) / "train/production" / name
```

Move the dataclass definition out of `training_materialization.py`, import it
from the new module, and re-export the existing constants as:

```python
DEFAULT_DATA2_ROOT = Path("/root/data2/pixal3d")
DEFAULT_LOCAL_ROOT = Path("/root/node17/data/pixal3d")
ABO_SOURCE_SPEC = build_source_spec("abo", DEFAULT_DATA2_ROOT)
THREED_FUTURE_SOURCE_SPEC = build_source_spec(
    "3d-future", DEFAULT_DATA2_ROOT
)
HSSD_SOURCE_SPEC = build_source_spec("hssd", DEFAULT_DATA2_ROOT)
```

Keep `DEFAULT_INDEX`, `DEFAULT_OUTPUT`, `THREED_FUTURE_INDEXES`, and
`THREED_FUTURE_OUTPUT` as compatibility aliases derived from those values.

- [ ] **Step 4: Run profile and materialization unit tests**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/multiview/test_training_source_profiles.py \
  tests/multiview/test_training_materialization.py \
  tests/multiview/test_production_materialization.py
git diff --check
```

Expected: all tests pass and no production files are read or written.

- [ ] **Step 5: Commit Task 1**

```bash
git add \
  data_toolkit/pipeline/training_source_profiles.py \
  data_toolkit/pipeline/training_materialization.py \
  tests/multiview/test_training_source_profiles.py \
  tests/multiview/test_training_materialization.py
git commit -m "feat: add HSSD training source profile"
```

---

### Task 2: Make Source Materialization and Preflight Root-Aware

**Files:**
- Modify: `scripts/materialize_multiview_production.py`
- Modify: `scripts/preflight_multiview_production.py`
- Modify: `data_toolkit/pipeline/training_preflight.py`
- Modify: `tests/multiview/test_training_materialization.py`
- Modify: `tests/multiview/test_training_preflight.py`
- Modify: `tests/multiview/test_training_manifest.py`

**Interfaces:**
- Consumes: `build_source_spec` and `source_output_root` from Task 1.
- Produces: `source_publication_paths(output_root: Path) -> dict[str, Path]` and root-aware `--profile`, `--data2-root`, `--local-root` CLIs.

- [ ] **Step 1: Add failing CLI relocation and acceptance tests**

Add tests that call both scripts through their argument parsers and assert:

```python
def test_hssd_materialization_cli_derives_node16_paths():
    args = materialize_cli._parse_args([
        "--profile", "hssd",
        "--data2-root", "/file2/youngwoo/pixal3d",
        "--local-root", "/home/youngwoo/data/pixal3d",
    ])
    spec, prepared, output = materialize_cli.resolve_profile_paths(args)
    assert spec.source == "HSSD"
    assert prepared == Path("/file2/youngwoo/pixal3d/prepared")
    assert output == Path(
        "/home/youngwoo/data/pixal3d/train/production/hssd"
    )


def test_hssd_preflight_accepts_passed_production_gate():
    spec = build_source_spec(
        "hssd", Path("/file2/youngwoo/pixal3d")
    )
    validate_acceptance_evidence(spec, {
        "acceptance_mode": "production_gate",
        "original_90_percent_gate_passed": True,
    })


def test_waiver_source_cannot_claim_gate_passed():
    spec = build_source_spec(
        "3d-future", Path("/file2/youngwoo/pixal3d")
    )
    with pytest.raises(ValueError, match="acceptance evidence"):
        validate_acceptance_evidence(spec, {
            "acceptance_mode": "valid_subset_user_waiver",
            "original_90_percent_gate_passed": True,
        })
```

Also test that a passed HSSD gate with waiver mode and a production-gate mode
with `False` are rejected.

- [ ] **Step 2: Run focused tests and observe RED**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/multiview/test_training_materialization.py \
  tests/multiview/test_training_preflight.py
```

Expected: failures because `hssd`, root arguments, and generalized acceptance
validation are absent.

- [ ] **Step 3: Generalize acceptance evidence**

Add to `training_preflight.py`:

```python
def validate_acceptance_evidence(
    spec: ProductionSourceSpec, evidence: Mapping[str, object]
) -> None:
    expected = (
        spec.acceptance_mode,
        spec.original_90_percent_gate_passed,
    )
    actual = (
        evidence.get("acceptance_mode"),
        evidence.get("original_90_percent_gate_passed"),
    )
    if actual != expected:
        raise ValueError(
            f"source={spec.source}: acceptance evidence does not "
            f"match source spec: expected={expected} actual={actual}"
        )
```

Replace hard-coded checks requiring waiver/`False` in source-aware report,
handoff, materialization, and training-data validation paths with this helper.
Retain legacy ABO wrapper checks where their schema explicitly pins the ABO
waiver. Generalize `_validate_source_spec` to require
`(spec.acceptance_mode, spec.original_90_percent_gate_passed)` to equal
`SOURCE_ACCEPTANCE_CONTRACTS[spec.source]`; reject unknown source contracts.

- [ ] **Step 4: Implement root-aware profile selection**

Both CLIs accept:

```python
parser.add_argument(
    "--profile", choices=SOURCE_PROFILE_NAMES, default="abo"
)
parser.add_argument(
    "--data2-root",
    type=Path,
    default=None,
)
parser.add_argument(
    "--local-root",
    type=Path,
    default=None,
)
```

Add pure resolvers:

```python
def resolve_profile_paths(args):
    data2_root = args.data2_root or Path("/root/data2/pixal3d")
    local_root = (
        args.local_root or Path("/root/node17/data/pixal3d")
    )
    spec = build_source_spec(args.profile, data2_root)
    prepared = data2_root / "prepared"
    output = source_output_root(args.profile, local_root)
    return spec, prepared, output


def source_publication_paths(output_root: Path) -> dict[str, Path]:
    root = Path(output_root)
    return {
        "report": root / "publication/report.json",
        "handoff": root / "publication/handoff.json",
        "training-data": root / "training_data.json",
    }
```

The materialization CLI calls:

```python
catalog = load_source_catalog(spec, prepared_root)
for stage in args.stage or tuple(STAGE_FAMILIES):
    print(materialize_stage(spec, stage, catalog, output_root))
```

The preflight CLI calls `preflight_stage(spec, ...)` and
`publish_source_handoff(spec, ...)` for all three profiles. `--verify-existing`
uses `validate_source_training_data(spec.source, training_data_path)`.

Keep the materializer's legacy `--index`, `--prepared-root`, and
`--output-root` arguments and the preflight's legacy `--root`, `--index`,
`--report`, `--handoff`, and `--training-data` arguments. When no new root
argument is supplied, preserve the current ABO and 3D-FUTURE resolution and
validation byte-for-byte. If either new root argument is explicitly supplied,
reject every legacy path argument and derive all paths from the selected
profile. HSSD always uses the profile-derived multi-index path and rejects a
legacy `--index`.

- [ ] **Step 5: Verify all source-aware contracts**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/multiview/test_training_source_profiles.py \
  tests/multiview/test_training_materialization.py \
  tests/multiview/test_training_preflight.py \
  tests/multiview/test_production_preflight.py \
  tests/multiview/test_training_manifest.py
git diff --check
```

Expected: all unit/synthetic tests pass; real external artifact tests may skip
only under their existing skip conditions.

- [ ] **Step 6: Commit Task 2**

```bash
git add \
  scripts/materialize_multiview_production.py \
  scripts/preflight_multiview_production.py \
  data_toolkit/pipeline/training_preflight.py \
  tests/multiview/test_training_materialization.py \
  tests/multiview/test_training_preflight.py \
  tests/multiview/test_training_manifest.py
git commit -m "feat: support root-aware source publication"
```

---

### Task 3: Support HSSD Standalone and Explicit Three-Source Manifests

**Files:**
- Modify: `data_toolkit/pipeline/training_manifest.py`
- Modify: `train.py`
- Modify: `tests/multiview/test_training_manifest.py`
- Modify: `tests/multiview/test_training_entrypoint.py`

**Interfaces:**
- Produces: `KNOWN_SOURCES`, `TWO_SOURCE_BUNDLE`, `THREE_SOURCE_BUNDLE`, `resolve_source_training_data(path: Path, stage: str) -> ResolvedTrainingData`, and bundle-aware `resolve_training_data`.
- Preserves: `CANONICAL_SOURCES` as the two-source compatibility alias.

- [ ] **Step 1: Add failing one-source and three-source tests**

Extend the synthetic source fixture to create schema-2 HSSD evidence and add:

```python
def test_hssd_source_training_data_resolves_as_one_source(
    hssd_training_data,
):
    resolved = resolve_training_data(hssd_training_data, "ss64")
    assert tuple(resolved.data_dir) == ("HSSD",)
    assert resolved.source_counts == {"HSSD": 3}
    assert resolved.total_count == 3
    assert resolved.sampling == "proportional-unweighted-concatenation"


def test_three_source_bundle_preserves_canonical_order(source_inputs):
    source_inputs["HSSD"] = make_hssd_source(count=3)
    value = build_combined_training_data(source_inputs)
    assert list(value["sources"]) == [
        "ABO", "3D-FUTURE", "HSSD"
    ]
    assert value["stages"]["ss64"]["source_counts"] == {
        "ABO": 2, "3D-FUTURE": 5, "HSSD": 3,
    }
    assert value["stages"]["ss64"]["total_count"] == 10


def test_three_source_pairwise_overlap_is_rejected(source_inputs):
    source_inputs["HSSD"] = make_hssd_source(
        scope=("shared-with-abo",)
    )
    with pytest.raises(
        ValueError, match="cross-source asset overlap"
    ):
        build_combined_training_data(source_inputs)


def test_existing_two_source_bundle_remains_valid(source_inputs):
    value = build_combined_training_data({
        "ABO": source_inputs["ABO"],
        "3D-FUTURE": source_inputs["3D-FUTURE"],
    })
    assert list(value["sources"]) == ["ABO", "3D-FUTURE"]
```

Add entrypoint coverage proving `resolve_training_input` accepts the standalone
HSSD source file and records one-source evidence before CUDA initialization.
Also mutate HSSD's report/handoff/training-data chain to a self-consistent
waiver/`False` chain and prove validation rejects it, while existing ABO and
3D-FUTURE waiver chains remain valid.

- [ ] **Step 2: Run focused manifest tests and observe RED**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/multiview/test_training_manifest.py \
  tests/multiview/test_training_entrypoint.py
```

Expected: HSSD schema and bundle membership are rejected by the two-source
constants.

- [ ] **Step 3: Generalize source and bundle identity**

Use:

```python
KNOWN_SOURCES = ("ABO", "3D-FUTURE", "HSSD")
CANONICAL_SOURCES = ("ABO", "3D-FUTURE")
TWO_SOURCE_BUNDLE = CANONICAL_SOURCES
THREE_SOURCE_BUNDLE = ("ABO", "3D-FUTURE", "HSSD")
SUPPORTED_BUNDLES = (TWO_SOURCE_BUNDLE, THREE_SOURCE_BUNDLE)
_SOURCE_SCHEMAS = {"ABO": 1, "3D-FUTURE": 2, "HSSD": 2}
```

During `_validate_source_training_data_value`, require the acceptance fields
in the fully pinned report/handoff/training-data chain to equal
`SOURCE_ACCEPTANCE_CONTRACTS[source]`. This independently protects launch-time
validation, where no storage-root-specific `ProductionSourceSpec` is passed.

Add:

```python
def _source_order(source_paths: Mapping[str, Path]) -> tuple[str, ...]:
    order = tuple(source_paths)
    if order not in SUPPORTED_BUNDLES:
        raise ValueError(
            f"source order must be one of {SUPPORTED_BUNDLES}: {order}"
        )
    return order
```

Pass the selected source order through `_validate_source_paths`,
`_document_from_sources`, `_combined_stage`, `_validate_combined_shape`, and
`resolve_training_data`. Pairwise overlap uses:

```python
owners: dict[str, str] = {}
for source in source_order:
    for asset in scopes[source]:
        previous = owners.setdefault(asset, source)
        if previous != source:
            raise ValueError(
                "cross-source asset overlap: "
                f"{previous}/{source}: {asset}"
            )
```

- [ ] **Step 4: Add exact-key source-manifest dispatch**

Implement:

```python
def _resolve_source_value(
    path: Path,
    value: Mapping[str, object],
    raw: bytes,
    stage: str,
) -> ResolvedTrainingData:
    source = value.get("source")
    if source not in KNOWN_SOURCES:
        raise ValueError(f"unknown source training data: {source}")
    validated = _validate_source_training_data_value(
        source, path, value, raw
    )
    selected = validated.stages[stage]
    scope = selected.source_scopes[source]
    return ResolvedTrainingData(
        path=validated.path,
        manifest_sha256=validated.sha256,
        stage=stage,
        data_dir=selected.data_dir,
        source_counts={source: len(scope)},
        total_count=len(scope),
        source_scopes={source: scope},
        union_scope_sha256=selected.union_scope_sha256,
        sampling=SAMPLING,
    )


def resolve_source_training_data(
    path: Path, stage: str
) -> ResolvedTrainingData:
    value, raw = _load_json(path, "source training data")
    return _resolve_source_value(path, value, raw, stage)
```

Refactor `_validate_source_training_data` so its file-loading wrapper delegates
to `_validate_source_training_data_value(source, path, value, raw)`. At the top
of `resolve_training_data`, load the JSON once and dispatch only when the
complete top-level key set matches a supported schema family:

```python
combined_keys = {
    "schema_version", "authorization", "sampling", "sources", "stages"
}
source = value.get("source")
source_keys = (
    set(_REPORT_FIELDS[_SOURCE_SCHEMAS[source]]) | {"report", "handoff"}
    if source in KNOWN_SOURCES
    else set()
)
if set(value) == combined_keys:
    return _resolve_combined_value(path, value, raw, stage)
if source_keys and set(value) == source_keys:
    return _resolve_source_value(path, value, raw, stage)
raise ValueError("unrecognized training_data schema")
```

Do not fall back from a failed combined validation to source validation.

- [ ] **Step 5: Verify manifest, entrypoint, and config tests**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/multiview/test_training_manifest.py \
  tests/multiview/test_training_entrypoint.py \
  tests/multiview/test_configs.py
git diff --check
```

Expected: two-source, three-source, standalone source, and pre-CUDA entrypoint
tests all pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add \
  data_toolkit/pipeline/training_manifest.py \
  train.py \
  tests/multiview/test_training_manifest.py \
  tests/multiview/test_training_entrypoint.py
git commit -m "feat: resolve HSSD and three-source training data"
```

---

### Task 4: Generalize Publication and Real Loader Preflight CLIs

**Files:**
- Modify: `scripts/publish_multisource_training.py`
- Modify: `scripts/preflight_multisource_training.py`
- Modify: `tests/multiview/test_training_manifest.py`
- Modify: `tests/multiview/test_multisource_preflight.py`

**Interfaces:**
- Consumes: bundle-aware manifest functions from Task 3.
- Produces: optional `--hssd` publication and source-order-driven configured loader validation.

- [ ] **Step 1: Add failing CLI and three-source collate tests**

Add a CLI test that runs:

```python
completed = subprocess.run(
    [
        sys.executable,
        "scripts/publish_multisource_training.py",
        "--abo", str(source_inputs["ABO"]),
        "--3d-future", str(source_inputs["3D-FUTURE"]),
        "--hssd", str(source_inputs["HSSD"]),
        "--output", str(output),
    ],
    check=True,
    capture_output=True,
    text=True,
)
payload = json.loads(completed.stdout)
assert payload["sources"] == ["ABO", "3D-FUTURE", "HSSD"]
```

Extend the synthetic configured Dataset test so
`preflight_multisource_stage` returns:

```python
assert result["collated_sources"] == [
    "ABO", "3D-FUTURE", "HSSD"
]
assert result["boundary_instances_checked"] == 6
```

- [ ] **Step 2: Run focused tests and observe RED**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/multiview/test_training_manifest.py \
  tests/multiview/test_multisource_preflight.py
```

Expected: `--hssd` is unknown and preflight still iterates the two-source
compatibility constant.

- [ ] **Step 3: Implement optional HSSD publication**

Add:

```python
DEFAULT_HSSD = Path(
    "/root/node17/data/pixal3d/train/production/hssd/"
    "training_data.json"
)
parser.add_argument("--hssd", type=Path)
```

Build sources in canonical order:

```python
sources = {
    "ABO": args.abo,
    "3D-FUTURE": args.future,
}
if args.hssd is not None:
    sources["HSSD"] = args.hssd
output = publish_combined_training_data(sources, args.output)
```

The JSON result includes `sources=list(manifest["sources"])`.
Without `--hssd`, output must remain the existing two-source schema.

- [ ] **Step 4: Drive preflight from resolved source order**

Replace every loop over `CANONICAL_SOURCES` with:

```python
source_order = tuple(resolved.source_scopes)
```

Pass `source_order` explicitly to `_expected_instances`,
`_validate_instances`, and `_boundary_samples`. Reject any resolved source not
in `KNOWN_SOURCES`, but do not require HSSD in a valid two-source manifest.
Update CLI help to describe verified one-, two-, or three-source training data.

Before and after all stages, assert:

```python
if torch.cuda.is_available() or torch.cuda.is_initialized():
    raise RuntimeError("CPU-only preflight initialized CUDA")
```

- [ ] **Step 5: Run publication and loader suites**

Run:

```bash
CUDA_VISIBLE_DEVICES="" \
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/multiview/test_training_manifest.py \
  tests/multiview/test_multisource_preflight.py \
  tests/multiview/test_training_entrypoint.py
git diff --check
```

Expected: one-source HSSD, legacy two-source, and three-source tests pass with
CUDA unavailable and uninitialized.

- [ ] **Step 6: Commit Task 4**

```bash
git add \
  scripts/publish_multisource_training.py \
  scripts/preflight_multisource_training.py \
  tests/multiview/test_training_manifest.py \
  tests/multiview/test_multisource_preflight.py
git commit -m "feat: publish and preflight three-source training"
```

---

### Task 5: Build the CPU-Only Node16 Preparation Driver and Runtime Configs

**Files:**
- Create: `data_toolkit/pipeline/node16_training_prepare.py`
- Create: `scripts/prepare_node16_training.py`
- Create: `tests/multiview/test_node16_training_prepare.py`

**Interfaces:**
- Produces: `PreparationPaths`, `estimate_required_bytes`, `create_runtime_configs`, `prepare_node16_training`, and a JSON final report.
- Consumes: Tasks 1–4 materialization, preflight, publication, and loader interfaces.

- [ ] **Step 1: Write failing path, disk, config, and orchestration tests**

Create tests for:

```python
def test_node16_paths_are_local_and_three_source():
    paths = PreparationPaths.from_roots(
        data2_root=Path("/file2/youngwoo/pixal3d"),
        local_root=Path("/home/youngwoo/data/pixal3d"),
        repo_root=Path("/home/youngwoo/Pixal3D-training-hssd"),
    )
    assert paths.combined_training_data == Path(
        "/home/youngwoo/data/pixal3d/train/production/"
        "abo-3d-future-hssd/training_data.json"
    )
    assert paths.hssd_training_data == Path(
        "/home/youngwoo/data/pixal3d/train/production/"
        "hssd/training_data.json"
    )


def test_insufficient_space_aborts_before_materialization(monkeypatch):
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(100, 99, 1),
    )
    with pytest.raises(ValueError, match="insufficient local free space"):
        assert_free_space(Path("/local"), required_bytes=2)


def test_runtime_configs_change_only_num_workers(tmp_path):
    outputs = create_runtime_configs(CONFIGS, tmp_path)
    assert tuple(outputs) == tuple(CONFIGS)
    for stage, source in CONFIGS.items():
        output = outputs[stage]
        original = json.loads(source.read_text())
        runtime = json.loads(output.read_text())
        assert runtime["trainer"]["args"]["num_workers"] == 1
        runtime["trainer"]["args"]["num_workers"] = (
            original["trainer"]["args"]["num_workers"]
        )
        assert runtime == original


def test_prepare_orders_source_then_combined_validation(monkeypatch):
    calls = []
    monkeypatch.setattr(core, "materialize_source",
                        lambda name, paths: calls.append(("materialize", name)))
    monkeypatch.setattr(core, "publish_source",
                        lambda name, paths: calls.append(("source", name)))
    monkeypatch.setattr(core, "publish_combined",
                        lambda paths: calls.append(("combined", None)))
    monkeypatch.setattr(core, "preflight_training_data",
                        lambda path: calls.append(("preflight", path.name)))
    prepare_node16_training(paths)
    assert calls[:6] == [
        ("materialize", "abo"), ("source", "abo"),
        ("materialize", "3d-future"), ("source", "3d-future"),
        ("materialize", "hssd"), ("source", "hssd"),
    ]
    assert calls[-2][0] == "combined"
    assert calls[-1][0] == "preflight"
```

Add create-only reuse tests: valid existing source calls verification, partial
stage roots abort, and no `unlink`, `rmtree`, or replacement call occurs.

- [ ] **Step 2: Run the new tests and observe RED**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/multiview/test_node16_training_prepare.py
```

Expected: collection fails because the preparation module is absent.

- [ ] **Step 3: Implement paths and conservative disk admission**

Use:

```python
@dataclass(frozen=True)
class PreparationPaths:
    data2_root: Path
    local_root: Path
    repo_root: Path
    production_root: Path
    runtime_config_root: Path
    evidence_root: Path

    @classmethod
    def from_roots(cls, data2_root, local_root, repo_root):
        local = Path(local_root)
        production = local / "train/production"
        return cls(
            Path(data2_root),
            local,
            Path(repo_root),
            production,
            local / "runtime-configs",
            production / "node16-preparation-evidence",
        )

    @property
    def hssd_training_data(self):
        return self.production_root / "hssd/training_data.json"

    @property
    def combined_training_data(self):
        return (
            self.production_root
            / "abo-3d-future-hssd/training_data.json"
        )
```

`estimate_required_bytes` validates every selected index/manifest reference
and sums pack sizes per stage-family consumption. Repeated families are
intentional because the materialized layout duplicates `common` into every
stage and duplicates `shape-1024` into Shape1024 and PBR1024:

```python
verified_stage_pack_bytes = sum(
    pack.pack.stat().st_size
    for profile in SOURCE_PROFILE_NAMES
    for stage in STAGES
    for family in STAGE_FAMILIES[stage]
    for pack in catalogs[profile][family]
)
```

Disk admission requires:

```python
required_bytes = verified_stage_pack_bytes * 2 + 10 * 1024**3
```

The factor of two conservatively covers final extracted bytes plus create-only
staging/copy overhead; the 10-GiB reserve covers metadata, evidence, and
runtime files. Persist both the stage-expanded pack sum and admission
requirement in the report.

- [ ] **Step 4: Implement create-only orchestration**

For each profile in `SOURCE_PROFILE_NAMES`:

```python
spec = build_source_spec(profile, paths.data2_root)
output = source_output_root(profile, paths.local_root)
publication = source_publication_paths(output)
if publication["training-data"].exists():
    verify_existing_source(spec, publication)
else:
    refuse_partial_source(output)
    catalog = load_source_catalog(spec, paths.data2_root / "prepared")
    for stage in STAGES:
        materialize_stage(spec, stage, catalog, output)
    results = preflight_all_source_stages(spec, output, config_paths)
    publish_source_handoff(
        spec,
        results,
        publication["report"],
        publication["handoff"],
        publication["training-data"],
    )
```

`refuse_partial_source` returns only when the source output root does not
exist. If the root or any stage/publication child exists without a fully
validated `training_data.json`, it raises and reports every discovered path;
it never deletes, renames, truncates, or repairs anything. A present
`training_data.json` is reusable only when the entire source trust chain and
all materialized stage evidence validate.

Then:

```python
preflight_all_training_stages(paths.hssd_training_data)
publish_combined_training_data(
    {
        "ABO": paths.production_root / "abo/training_data.json",
        "3D-FUTURE": (
            paths.production_root / "3d-future/training_data.json"
        ),
        "HSSD": paths.hssd_training_data,
    },
    paths.combined_training_data,
)
preflight_all_training_stages(paths.combined_training_data)
```

Check `torch.cuda.is_available()` and `torch.cuda.is_initialized()` before and
after any configured Dataset import. The final report contains source/stage
counts, final scope digests, eligibility exclusion counts, artifact SHA-256,
free-space evidence, runtime config paths, and launch commands.

- [ ] **Step 5: Implement runtime config generation**

Accept the existing stage-to-config `CONFIGS` mapping, read exactly those four
production configs, and set only:

```python
runtime["trainer"]["args"]["num_workers"] = 1
```

Write create-only canonical JSON below:

```text
/home/youngwoo/data/pixal3d/runtime-configs
```

Name each output `{original-stem}.node16-workers1.json` and return a
stage-to-output mapping so launch commands cannot swap stages. If an output
exists, its bytes may differ in JSON formatting, but its complete parsed JSON
must equal the expected runtime JSON.

The report asserts each config's batch, split, global batch, steps, save
interval, retention, and disabled snapshots.

- [ ] **Step 6: Add the CLI**

`scripts/prepare_node16_training.py` accepts:

```text
--data2-root /file2/youngwoo/pixal3d
--local-root /home/youngwoo/data/pixal3d
--repo-root /home/youngwoo/Pixal3D-training-hssd
--execute
```

Without `--execute`, it performs source contract and disk admission checks and
prints the planned paths only. With `--execute`, it runs the create-only
workflow. It requires `CUDA_VISIBLE_DEVICES` to be explicitly set to the empty
string; both an unset variable and a non-empty value are rejected before
PyTorch or any Dataset module is imported.

- [ ] **Step 7: Verify the preparation module**

Run:

```bash
CUDA_VISIBLE_DEVICES="" \
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/multiview/test_node16_training_prepare.py \
  tests/multiview/test_training_source_profiles.py \
  tests/multiview/test_training_materialization.py \
  tests/multiview/test_training_preflight.py \
  tests/multiview/test_training_manifest.py \
  tests/multiview/test_multisource_preflight.py
git diff --check
```

Expected: all tests pass with no GPU initialization.

- [ ] **Step 8: Commit Task 5**

```bash
git add \
  data_toolkit/pipeline/node16_training_prepare.py \
  scripts/prepare_node16_training.py \
  tests/multiview/test_node16_training_prepare.py
git commit -m "feat: prepare Node16 multisource training data"
```

---

### Task 6: Document Exact Node16 Preparation and Launch Commands

**Files:**
- Modify: `README.md`
- Modify: `docs/data_preprocessing_runbook_ko.md`
- Create: `docs/hssd_node16_training_runbook_ko.md`

**Interfaces:**
- Consumes: paths and CLI from Task 5.
- Produces: operator-ready local Node16 commands and recovery rules.

- [ ] **Step 1: Add a documentation contract test**

Add to `tests/multiview/test_node16_training_prepare.py`:

```python
def test_node16_runbook_contains_exact_artifacts_and_safety_rules():
    text = Path(
        "docs/hssd_node16_training_runbook_ko.md"
    ).read_text()
    for required in (
        "CUDA_VISIBLE_DEVICES=\"\"",
        "scripts/prepare_node16_training.py",
        "--execute",
        "hssd/training_data.json",
        "abo-3d-future-hssd/training_data.json",
        "trainer.args.num_workers = 1",
        "--training_data",
        "nvidia-smi",
        "wandb status",
        "재실행",
    ):
        assert required in text
```

- [ ] **Step 2: Run the focused test and observe RED**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/multiview/test_node16_training_prepare.py::test_node16_runbook_contains_exact_artifacts_and_safety_rules
```

Expected: failure because the runbook is absent.

- [ ] **Step 3: Write the Korean runbook**

The runbook contains these exact Node16-local preparation commands:

```bash
source /home/youngwoo/miniconda3/etc/profile.d/conda.sh
conda activate pixal3d
cd /home/youngwoo/Pixal3D-training-hssd
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=""

python scripts/prepare_node16_training.py \
  --data2-root /file2/youngwoo/pixal3d \
  --local-root /home/youngwoo/data/pixal3d \
  --repo-root /home/youngwoo/Pixal3D-training-hssd

python scripts/prepare_node16_training.py \
  --data2-root /file2/youngwoo/pixal3d \
  --local-root /home/youngwoo/data/pixal3d \
  --repo-root /home/youngwoo/Pixal3D-training-hssd \
  --execute
```

Document HSSD-only verification, three-source verification, final count/digest
inspection, runtime config inspection, and one-at-a-time six-GPU launch
commands for all four models. Launches use:

```text
--training_data /home/youngwoo/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json
```

Document `tmux`, local log directory creation, `wandb status`, `nvidia-smi`,
checkpoint inspection, and same-command resume. State that partial artifacts
are never automatically removed; the operator must preserve and report the
path before any recovery decision.

- [ ] **Step 4: Update existing documentation**

Update README's verified launch section to retain the two-source commands and
link the new three-source Node16 runbook. Update the preprocessing runbook's
completed-source section with the observed HSSD boundary and new preparation
workflow. Do not relabel the older ABO + 3D-FUTURE evidence as three-source.

- [ ] **Step 5: Verify documentation**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/multiview/test_node16_training_prepare.py
rg -n \
  'HSSD|abo-3d-future-hssd|num_workers|--training_data|CUDA_VISIBLE_DEVICES' \
  README.md docs/data_preprocessing_runbook_ko.md \
  docs/hssd_node16_training_runbook_ko.md
git diff --check
```

Expected: contract tests pass and exact local paths are present.

- [ ] **Step 6: Commit Task 6**

```bash
git add \
  README.md \
  docs/data_preprocessing_runbook_ko.md \
  docs/hssd_node16_training_runbook_ko.md \
  tests/multiview/test_node16_training_prepare.py
git commit -m "docs: add HSSD Node16 training runbook"
```

---

### Task 7: Full Regression, Independent Review, and Node16 Execution

**Files:**
- Create on Node16: `/home/youngwoo/Pixal3D-training-hssd`
- Create on Node16: `/home/youngwoo/data/pixal3d/train/production/*`
- Create on Node16: `/home/youngwoo/data/pixal3d/runtime-configs/*`
- Create on Node16: `/home/youngwoo/data/pixal3d/train/production/node16-preparation-evidence/*`
- Modify after execution: `docs/hssd_node16_training_runbook_ko.md` only if observed final counts or commands require factual updates.

**Interfaces:**
- Consumes: reviewed repository HEAD and Task 5 preparation CLI.
- Produces: real HSSD standalone and three-source manifests, verified runtime configs, and final Node16 evidence.

- [ ] **Step 1: Run the complete local regression suite**

Run:

```bash
CUDA_VISIBLE_DEVICES="" \
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/multiview/test_training_source_profiles.py \
  tests/multiview/test_training_materialization.py \
  tests/multiview/test_production_materialization.py \
  tests/multiview/test_training_preflight.py \
  tests/multiview/test_production_preflight.py \
  tests/multiview/test_training_manifest.py \
  tests/multiview/test_training_entrypoint.py \
  tests/multiview/test_multisource_preflight.py \
  tests/multiview/test_node16_training_prepare.py \
  tests/multiview/test_configs.py \
  tests/multiview/test_checkpoint_retention.py \
  tests/multiview/test_train_smoke_override.py
git diff --check
git status --short --branch
```

Expected: all runnable tests pass; only existing external-file integration
conditions may skip.

- [ ] **Step 2: Obtain an independent implementation review**

Review every task commit against:

```text
docs/superpowers/specs/2026-07-29-hssd-node16-training-readiness-design.md
```

The reviewer must verify backward-compatible two-source behavior, exact HSSD
counts, pairwise overlap checks, source-manifest dispatch, create-only failure
semantics, no CUDA initialization, and runtime-config semantic diff.

- [ ] **Step 3: Audit Node16 before deployment**

From a read-only Node16 shell, record:

```bash
hostname
df -h /home/youngwoo/data /file2/youngwoo /file3/youngwoo
nvidia-smi
ps -eo pid,ppid,etime,stat,cmd |
  grep -E 'data_toolkit.pipeline.cli (supervisor|worker)|train.py' |
  grep -v grep || true
```

Do not stop preprocessing, training, or another user's process.

- [ ] **Step 4: Deploy the reviewed source to an isolated Node16 directory**

From the current repository host:

```bash
git archive --format=tar HEAD |
  ssh -p 55555 youngwoo@n16.unist.info \
    'set -e
     test ! -e /home/youngwoo/Pixal3D-training-hssd
     mkdir -p /home/youngwoo/Pixal3D-training-hssd
     tar -xf - -C /home/youngwoo/Pixal3D-training-hssd'
```

If the destination already exists, stop and inspect it; do not overwrite it.
Record local HEAD and deployed file hashes in the evidence directory.

- [ ] **Step 5: Run Node16 CPU-only tests and dry-run admission**

On Node16:

```bash
source /home/youngwoo/miniconda3/etc/profile.d/conda.sh
conda activate pixal3d
cd /home/youngwoo/Pixal3D-training-hssd
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=""

python -m pytest -q \
  tests/multiview/test_training_source_profiles.py \
  tests/multiview/test_training_manifest.py \
  tests/multiview/test_multisource_preflight.py \
  tests/multiview/test_node16_training_prepare.py

python scripts/prepare_node16_training.py \
  --data2-root /file2/youngwoo/pixal3d \
  --local-root /home/youngwoo/data/pixal3d \
  --repo-root /home/youngwoo/Pixal3D-training-hssd
```

Expected: tests pass, CUDA remains unavailable/uninitialized, all 135 HSSD
selected-family manifest references validate, and disk admission passes before
any stage root is created.

- [ ] **Step 6: Execute Node16 preparation**

On Node16:

```bash
export CUDA_VISIBLE_DEVICES=""
python scripts/prepare_node16_training.py \
  --data2-root /file2/youngwoo/pixal3d \
  --local-root /home/youngwoo/data/pixal3d \
  --repo-root /home/youngwoo/Pixal3D-training-hssd \
  --execute
```

Allow the command to finish. It may run for hours because it verifies pack
bytes, extracts three sources, evaluates every eligible latent, and direct
loads real samples. Poll with non-blocking waits and communicate progress at
least every 60 seconds while the command is active, without interrupting it.

- [ ] **Step 7: Verify real standalone and three-source artifacts**

On Node16:

```bash
export CUDA_VISIBLE_DEVICES=""
python scripts/preflight_multisource_training.py \
  --training-data \
  /home/youngwoo/data/pixal3d/train/production/hssd/training_data.json

python scripts/preflight_multisource_training.py \
  --training-data \
  /home/youngwoo/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json

sha256sum \
  /home/youngwoo/data/pixal3d/train/production/hssd/training_data.json \
  /home/youngwoo/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json
```

Expected: all four stages pass for both manifests, HSSD-only collates HSSD,
the combined manifest collates ABO/3D-FUTURE/HSSD, and CUDA remains
uninitialized.

- [ ] **Step 8: Verify runtime configs and launch prerequisites**

On Node16:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/home/youngwoo/data/pixal3d/runtime-configs")
for path in sorted(root.glob("*.json")):
    args = json.loads(path.read_text())["trainer"]["args"]
    print(
        path.name,
        args["batch_size_per_gpu"],
        args["batch_split"],
        args["num_workers"],
        args["max_steps"],
        args["i_save"],
        args["max_checkpoints"],
        args["i_sample"],
    )
PY

test -f /file3/youngwoo/pixal3d/train/checkpoints/single_view/ss_flow_img_dit_1_3B_64_bf16.pt
test -f /file3/youngwoo/pixal3d/train/checkpoints/single_view/slat_flow_img2shape_dit_1_3B_512_bf16.pt
test -f /file3/youngwoo/pixal3d/train/checkpoints/single_view/slat_flow_img2shape_dit_1_3B_1024_bf16.pt
test -f /file3/youngwoo/pixal3d/train/checkpoints/single_view/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.pt
wandb status
nvidia-smi
```

Expected: four configs show workers `1` and the approved batch policies; four
single-view checkpoints exist; W&B is authenticated; six GPUs are selected
only after ownership and memory are checked.

- [ ] **Step 9: Record observed results and commit factual documentation**

Update the HSSD runbook with the actual final eligibility counts, exclusion
reason counts, manifest SHA-256 values, elapsed preparation time, and final
evidence path. Do not alter measured values or claim that GPU training ran.

Run:

```bash
git diff --check
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/multiview/test_node16_training_prepare.py
git add docs/hssd_node16_training_runbook_ko.md
git commit -m "docs: record HSSD Node16 training handoff"
```

- [ ] **Step 10: Final verification**

Run the complete Task 7 Step 1 suite again, verify the local worktree is clean,
and report:

```text
HSSD standalone manifest path and SHA-256
three-source manifest path and SHA-256
source/stage final counts
runtime config values
Node16 free space after materialization
W&B status
exact four local training commands
```

Do not start fine-tuning until the user explicitly chooses the first model.
