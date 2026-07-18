# Paper-Faithful Family Eligibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the global all-or-nothing preprocessing admission rule with audited per-family eligibility so geometry-valid assets remain usable when their materials are outside TRELLIS.2's standard PBR subset.

**Architecture:** Keep the frozen SHA scope and global terminal quality outcomes, then add durable family exclusions to the existing quality ledger. Derive command-specific instance lists and per-family pack memberships from those exclusions; version pack manifests so every publication records both its frozen scope and its exact included scope.

**Tech Stack:** Python 3.11, pytest, pandas, Blender 4.5.1, deterministic tar/JSON manifests, existing Pixal3D pipeline CLI.

## Global Constraints

- Run every command in conda environment `pixal3d` with CUDA 12.8 and torch 2.8 or newer.
- Use `/root/data2/pixal3d`, `/root/data3/pixal3d`, and `/root/node17/data/pixal3d`; do not restore `/root/pixel3d-data`.
- Preserve frozen SHA order, source identity, views, cameras, anchors, latent models, resolutions, and dtype.
- Keep Blender pinned to 4.5.1 and retain `bpy.ops.wm.obj_import` for Blender 4.
- Do not bake, rewrite, or silently accept unsupported material graphs.
- Do not mix artifacts produced by different tool commits.
- Write production behavior only after the corresponding test has failed for the expected reason.

## Execution Status (2026-07-18)

- Tasks 1-5 are implemented in commits `b300e30`, `64a484a`, `5b85e3c`,
  `e07702b`, and `21fa97f`.
- The complete automated preprocessing suite passes: `581 passed`.
- Task 6 documentation is updated. Old-contract state preservation and the
  clean 20-asset ObjaverseXL GitHub smoke rerun remain in progress.

---

### Task 1: Define family dependencies and durable exclusions

**Files:**
- Modify: `data_toolkit/pipeline/orchestrator.py:55-65,606-806,947-1020,1327-1433`
- Test: `tests/data_toolkit/test_orchestrator.py`

**Interfaces:**
- Produces: `family_dependencies(config) -> Mapping[str, frozenset[str]]`
- Produces: `PipelineRunner.record_family_exclusion(asset_sha, families, *, category, stage, reason, attempts) -> None`
- Produces: `PipelineRunner.family_exclusions(asset_sha) -> Mapping[str, Mapping[str, object]]`
- Persists: quality-ledger schema 3 field `family_exclusions: {sha: {family: record}}`

- [ ] **Step 1: Write failing family dependency tests**

```python
def test_family_dependencies_make_pbr_a_subset_of_matching_shape(config):
    dependencies = family_dependencies(config)
    assert dependencies["PBR-256"] == frozenset({"shape-256"})
    assert dependencies["PBR-512"] == frozenset({"shape-512"})
    assert dependencies["PBR-1024"] == frozenset({"shape-1024"})
    assert dependencies["SS-64"] == frozenset({"shape-1024"})


def test_record_family_exclusion_round_trips_through_quality_ledger(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "family-ledger", "ABO", "ABO-00000"
    )
    asset_sha = "a" * 64
    write_instances(context, (asset_sha,))
    runner = PipelineRunner(
        isolated_config,
        FakeResourceGuard(),
        {},
        {},
        command_builder=lambda _context, _config: (),
    )
    runner.active_context = context
    runner.active_checkpoint = PipelineCheckpoint(context.shard_id)
    runner.active_checkpoint_path = tmp_path / "checkpoint.json"
    runner._active_quality_assets = (asset_sha,)
    runner._active_instances_sha256 = sha256(
        context.instances.read_bytes()
    ).hexdigest()
    runner._active_quality_ledger = orchestrator_module._empty_quality_ledger(
        context
    )
    ledger_path = tmp_path / "quality.json"
    runner._active_quality_ledger_path = ledger_path
    orchestrator_module._save_quality_ledger(
        ledger_path, runner._active_quality_ledger
    )
    runner.record_family_exclusion(
        asset_sha,
        ("PBR-256", "PBR-512", "PBR-1024"),
        category="unsupported_shader",
        stage="dump_pbr",
        reason="Material is not supported",
        attempts=1,
    )
    ledger = json.loads(ledger_path.read_text())
    assert set(ledger["family_exclusions"][asset_sha]) == {
        "PBR-256", "PBR-512", "PBR-1024"
    }
    assert runner.family_exclusions(asset_sha)["PBR-256"]["category"] == (
        "unsupported_shader"
    )
    assert asset_sha not in runner.active_checkpoint.quality_outcomes
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
conda run --no-capture-output -n pixal3d python -m pytest \
  tests/data_toolkit/test_orchestrator.py \
  -k 'family_dependencies or record_family_exclusion' -v
```

Expected: collection/import failure because the family interfaces and ledger field do not exist.

- [ ] **Step 3: Implement the family model and schema-3 ledger**

Add these constants and dependency function next to `QUALITY_OUTCOMES`:

```python
QUALITY_LEDGER_SCHEMA_VERSION = 3


def family_dependencies(config: PipelineConfig) -> Mapping[str, frozenset[str]]:
    highest = max(config.targets.resolutions)
    dependencies = {
        "common": frozenset(),
        f"SS-{config.targets.ss_resolution}": frozenset({f"shape-{highest}"}),
    }
    for resolution in config.targets.resolutions:
        dependencies[f"shape-{resolution}"] = frozenset()
        dependencies[f"PBR-{resolution}"] = frozenset(
            {f"shape-{resolution}"}
        )
    return dependencies
```

Extend `_empty_quality_ledger`, `_load_quality_ledger`, and
`_advance_quality_ledger` with a validated `family_exclusions` mapping. Add an
idempotent runner method whose write order is copy ledger → validate record →
atomic ledger save → replace active ledger:

```python
def record_family_exclusion(
    self,
    asset_sha: str,
    families: Sequence[str],
    *,
    category: str,
    stage: str,
    reason: str,
    attempts: int = 0,
) -> None:
    context = self.active_context
    if context is None or self._active_quality_ledger is None:
        raise InfrastructureError("family exclusion has no active ledger")
    asset_sha = _validated_asset_sha(asset_sha)
    frozen = self._load_frozen_quality_assets(context)
    if asset_sha not in frozen:
        raise InfrastructureError("family exclusion asset is not frozen")
    valid_families = set(PACK_FAMILIES) - {"common"}
    requested = tuple(sorted(set(families)))
    if not requested or not set(requested) <= valid_families:
        raise ValueError("invalid excluded family")
    record = {
        "category": category,
        "stage": stage,
        "reason": reason,
        "attempts": attempts,
    }
    next_ledger = copy.deepcopy(self._active_quality_ledger)
    by_family = next_ledger["family_exclusions"].setdefault(asset_sha, {})
    for family in requested:
        previous = by_family.get(family)
        if previous is not None and previous != record:
            raise InfrastructureError("conflicting family exclusion")
        by_family[family] = dict(record)
    _save_quality_ledger(self._active_quality_ledger_path, next_ledger)
    self._active_quality_ledger = next_ledger
```

Schema-2 ledgers are loaded as a backward-compatible empty
`family_exclusions` mapping and are rewritten as schema 3 only by the new
commit. Reject malformed family names, records, SHAs, and counts.

- [ ] **Step 4: Run focused and legacy ledger tests**

```bash
conda run --no-capture-output -n pixal3d python -m pytest \
  tests/data_toolkit/test_orchestrator.py \
  -k 'family or quality_ledger or quarantine' -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data_toolkit/pipeline/orchestrator.py tests/data_toolkit/test_orchestrator.py
git commit -m "feat: persist family-scoped eligibility"
```

### Task 2: Preserve PBR leaf-worker failure evidence

**Files:**
- Modify: `data_toolkit/dump_pbr.py:65-127,200-240`
- Modify: `data_toolkit/datasets/ObjaverseXL.py:50-136`
- Test: `tests/data_toolkit/test_leaf_worker_contracts.py`
- Test: `tests/data_toolkit/test_dataset_adapters.py`

**Interfaces:**
- Produces: a PBR `new_records/part_<rank>.csv` row for every selected SHA.
- Success row: `sha256,pbr_dumped,error_category,error_reason` with `pbr_dumped=True` and empty error fields.
- Failure row: `pbr_dumped=False` plus `unsupported_shader`, `timeout`, or `pbr_dump_failure` evidence.

- [ ] **Step 1: Write failing evidence tests**

```python
def test_pbr_failure_record_classifies_official_unsupported_marker():
    asset = "a" * 64
    assert dump_pbr._pbr_failure_record(
        asset, "Material is not supported"
    ) == {
        "sha256": asset,
        "pbr_dumped": False,
        "error_category": "unsupported_shader",
        "error_reason": "Material is not supported",
    }
```

Add an adapter test proving `_process_instance` returns a structured failure
record instead of `None` when the callback raises.

- [ ] **Step 2: Run the tests and verify RED**

```bash
conda run --no-capture-output -n pixal3d python -m pytest \
  tests/data_toolkit/test_leaf_worker_contracts.py \
  tests/data_toolkit/test_dataset_adapters.py \
  -k 'pbr and (error or unsupported)' -v
```

Expected: FAIL because `_dump_pbr` raises and ObjaverseXL drops callback exceptions.

- [ ] **Step 3: Return structured PBR records**

Catch only the asset-scoped parse/output exceptions inside `_dump_pbr`; preserve
`OSError`, subprocess launch errors, and process-wide failures as exceptions.
Classify an error-file payload containing the official marker exactly as:

```python
def _pbr_failure_record(sha256: str, reason: str) -> dict[str, object]:
    category = (
        "unsupported_shader"
        if "Material is not supported" in reason
        else "pbr_dump_failure"
    )
    return {
        "sha256": sha256,
        "pbr_dumped": False,
        "error_category": category,
        "error_reason": reason,
    }
```

Make `ObjaverseXL._process_instance` preserve callback exceptions as
`pbr_dump_failure` records only when the callback identifies itself as the PBR
dumper; ordinary adapter/download exceptions keep their current retry behavior.
Write every returned row atomically through the existing `_atomic_write_csv`.

- [ ] **Step 4: Run focused tests**

```bash
conda run --no-capture-output -n pixal3d python -m pytest \
  tests/data_toolkit/test_leaf_worker_contracts.py \
  tests/data_toolkit/test_dataset_adapters.py \
  -k 'pbr or ObjaverseXL' -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data_toolkit/dump_pbr.py data_toolkit/datasets/ObjaverseXL.py \
  tests/data_toolkit/test_leaf_worker_contracts.py \
  tests/data_toolkit/test_dataset_adapters.py
git commit -m "fix: retain PBR asset failure evidence"
```

### Task 3: Route commands and validators by family

**Files:**
- Modify: `data_toolkit/pipeline/orchestrator.py:896-1433,3240-3696,4230-4400`
- Test: `tests/data_toolkit/test_orchestrator.py`

**Interfaces:**
- Produces: `PipelineServices._candidate_assets(context, family=None) -> tuple[str, ...]`
- Produces: `PipelineServices._command_families(command_name) -> tuple[str, ...]`
- Produces: `PipelineServices._exclude_stage_failure(context, asset_sha, command_name, reason) -> None`
- Consumes: Task 1 `record_family_exclusion` and ledger state.

- [ ] **Step 1: Write failing routing tests**

```python
def _family_services(isolated_config, tmp_path):
    context = ShardContext.for_test(
        tmp_path / "family-routing", "ABO", "ABO-00000"
    )
    assets = ("a" * 64, "b" * 64)
    write_instances(context, assets)
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )
    runner = services.runner
    runner.active_context = context
    runner.active_checkpoint = PipelineCheckpoint(context.shard_id)
    runner.active_checkpoint_path = tmp_path / "checkpoint.json"
    runner._active_quality_assets = assets
    runner._active_instances_sha256 = sha256(
        context.instances.read_bytes()
    ).hexdigest()
    runner._active_quality_ledger = orchestrator_module._empty_quality_ledger(
        context
    )
    runner._active_quality_ledger_path = tmp_path / "quality.json"
    orchestrator_module._save_quality_ledger(
        runner._active_quality_ledger_path, runner._active_quality_ledger
    )
    return services, context, runner, assets


def test_pbr_exclusion_keeps_geometry_commands_eligible(
    isolated_config, tmp_path
):
    services, context, runner, assets = _family_services(
        isolated_config, tmp_path
    )
    full_pbr, geometry_only = assets
    runner.record_family_exclusion(
        geometry_only,
        ("PBR-256", "PBR-512", "PBR-1024"),
        category="unsupported_shader",
        stage="dump_pbr",
        reason="Material is not supported",
        attempts=1,
    )
    assert services._candidate_assets(context, "shape-256") == (
        full_pbr, geometry_only
    )
    assert services._candidate_assets(context, "SS-64") == (
        full_pbr, geometry_only
    )
    assert services._candidate_assets(context, "PBR-256") == (full_pbr,)


def test_shape_resolution_failure_cascades_only_to_dependents(
    isolated_config, tmp_path
):
    services, context, runner, assets = _family_services(
        isolated_config, tmp_path
    )
    asset = assets[0]
    services._exclude_stage_failure(
        context, asset, "encode_shape_256", "missing shape latent"
    )
    assert asset not in services._candidate_assets(context, "shape-256")
    assert asset not in services._candidate_assets(context, "PBR-256")
    assert asset in services._candidate_assets(context, "shape-512")
    assert asset in services._candidate_assets(context, "SS-64")
```

- [ ] **Step 2: Run tests and verify RED**

```bash
conda run --no-capture-output -n pixal3d python -m pytest \
  tests/data_toolkit/test_orchestrator.py \
  -k 'pbr_exclusion_keeps or shape_resolution_failure' -v
```

Expected: FAIL because command and family candidates still use the global terminal map.

- [ ] **Step 3: Implement command-family routing**

Map commands exactly:

```python
def _command_families(self, name: str) -> tuple[str, ...]:
    if name == "dump_pbr":
        return tuple(f"PBR-{r}" for r in self.config.targets.resolutions)
    if name.startswith("voxelize_pbr_") or name.startswith("encode_pbr_"):
        return (f"PBR-{name.rsplit('_', 1)[1]}",)
    if name.startswith("dual_grid_") or name.startswith("encode_shape_"):
        return (f"shape-{name.rsplit('_', 1)[1]}",)
    if name == f"encode_ss_{self.config.targets.ss_resolution}":
        return (f"SS-{self.config.targets.ss_resolution}",)
    return ()
```

`_command_for_eligible_assets` writes a command-specific instance file using
the union of candidate assets for its mapped families. Shared commands use all
non-terminal source candidates. Validators call `_exclude_stage_failure` for
family stages and retain `record_asset_outcome` for source, mesh, stats, and
render failures.

At `dump_pbr`, read the new records, validate one row per selected SHA, and
write the recorded category/reason to all PBR families. Do not classify a
missing record as `unsupported_shader`.

At final validation, validate each family independently. Record global
`completed` when the asset remains included in at least one non-common family;
otherwise write a global failure with category `no_eligible_training_family`.

- [ ] **Step 4: Run focused routing and resume tests**

```bash
conda run --no-capture-output -n pixal3d python -m pytest \
  tests/data_toolkit/test_orchestrator.py \
  -k 'eligible or family or dump_pbr or validate_outputs or resume' -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data_toolkit/pipeline/orchestrator.py tests/data_toolkit/test_orchestrator.py
git commit -m "feat: route preprocessing by pack family"
```

### Task 4: Version manifests with exact included identities

**Files:**
- Modify: `data_toolkit/pipeline/packing.py:20-70,208-360,470-755`
- Modify: `data_toolkit/pipeline/orchestrator.py:3650-4150`
- Test: `tests/data_toolkit/test_packing.py`
- Test: `tests/data_toolkit/test_orchestrator.py`

**Interfaces:**
- Produces: `PackManifest.schema_version == 2`
- Produces: `PackManifest.included_asset_sha256s: tuple[str, ...]`
- Changes: `publish_pack(..., included_asset_sha256s_by_family: Mapping[str, tuple[str, ...]], ...)`

- [ ] **Step 1: Write failing manifest and subset tests**

```python
def test_pack_manifest_records_frozen_and_included_scopes(tmp_path):
    frozen = ("a" * 64, "b" * 64)
    manifest = build_pack(
        tmp_path,
        [],
        tmp_path / "pack.tar",
        "shard",
        batch_id="batch000",
        family="PBR-256",
        config_hash="c" * 64,
        tool_commit="deadbeef",
        asset_sha256s=frozen,
        included_asset_sha256s=(frozen[0],),
        completed_count=1,
        quarantined_count=1,
    )
    assert manifest.schema_version == 2
    assert manifest.asset_sha256s == frozen
    assert manifest.included_asset_sha256s == (frozen[0],)


def test_publish_pack_accepts_distinct_family_included_scopes(tmp_path):
    source = tmp_path / "source"
    members = _family_members(source)
    first, second = "a" * 64, "b" * 64
    included = {
        family: (first, second) for family in PACK_FAMILIES
    }
    for family in ("PBR-256", "PBR-512", "PBR-1024"):
        included[family] = (first,)
    manifests = publish_pack(
        tmp_path / "data2",
        source,
        members,
        "ABO-00000",
        source="ABO",
        batch_id="batch000",
        config_hash="c" * 64,
        tool_commit="deadbeef",
        asset_sha256s=(first, second),
        included_asset_sha256s_by_family=included,
    )
    by_family = {manifest.family: manifest for manifest in manifests}
    assert by_family["PBR-256"].included_asset_sha256s == (first,)
    assert by_family["shape-256"].included_asset_sha256s == (first, second)
```

- [ ] **Step 2: Run tests and verify RED**

```bash
conda run --no-capture-output -n pixal3d python -m pytest \
  tests/data_toolkit/test_packing.py tests/data_toolkit/test_orchestrator.py \
  -k 'included_scope or subset_of_matching_shape' -v
```

Expected: FAIL because schema version and included identities are absent.

- [ ] **Step 3: Implement schema-v2 manifests and per-family publication**

Extend the dataclass:

```python
@dataclass(frozen=True)
class PackManifest:
    schema_version: int
    shard_id: str
    batch_id: str
    family: str
    config_hash: str
    tool_commit: str
    asset_sha256s: tuple[str, ...]
    included_asset_sha256s: tuple[str, ...]
    completed_count: int
    quarantined_count: int
    created_at: str
    validated_at: str
    pack_sha256: str
    members: tuple[PackMember, ...]
    gate: str = "production"
```

Validate that frozen and included scopes are sorted and unique, included is a
subset of frozen, and counts equal included/excluded sizes. Accept schema-v1
manifests by deriving `included_asset_sha256s` from the legacy completed member
paths only in read/verification code; never publish schema 1.

Change orchestrator membership derivation to return
`Mapping[str, tuple[str, ...]]`. Build members independently per family, set
`common` to the union of non-common included sets, and enforce every PBR/shape
subset before publication.

- [ ] **Step 4: Run packing, publication, and raw archive tests**

```bash
conda run --no-capture-output -n pixal3d python -m pytest \
  tests/data_toolkit/test_packing.py \
  tests/data_toolkit/test_orchestrator.py \
  -k 'pack or publish or archive or family' -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data_toolkit/pipeline/packing.py data_toolkit/pipeline/orchestrator.py \
  tests/data_toolkit/test_packing.py tests/data_toolkit/test_orchestrator.py
git commit -m "feat: publish exact family memberships"
```

### Task 5: Enforce family membership in runtime audit and reports

**Files:**
- Modify: `data_toolkit/pipeline/runtime.py:1460-1585,1820-1945`
- Modify: `data_toolkit/pipeline/reporting.py:380-430,680-790`
- Modify: `data_toolkit/pipeline/cli.py`
- Test: `tests/data_toolkit/test_reporting.py`
- Test: `tests/data_toolkit/test_cli.py`
- Test: `tests/data_toolkit/test_pipeline_integration.py`

**Interfaces:**
- Audit consumes manifest `included_asset_sha256s` and ledger `family_exclusions`.
- Reports produce `family_counts[family] = {included, excluded}` separately from global quarantine counts.
- Produces: `validate_family_memberships(included: Mapping[str, set[str]], config) -> None`.

- [ ] **Step 1: Write failing audit/report tests**

```python
def test_report_accepts_family_counts_separate_from_global_quality():
    report = valid_report_payload()
    report["quality"]["quarantined_assets"] = 2
    report["handoff"]["family_counts"] = {
        "shape-256": {"included": 18, "excluded": 2},
        "PBR-256": {"included": 15, "excluded": 5},
    }
    validate_gate_report(report)
    assert report["quality"]["quarantined_assets"] == 2
    assert report["handoff"]["family_counts"]["PBR-256"] == {
        "included": 15,
        "excluded": 5,
    }


def test_audit_rejects_pbr_identity_missing_from_shape(config):
    included = {family: set() for family in PACK_FAMILIES}
    included["shape-256"] = {"a" * 64}
    included["PBR-256"] = {"b" * 64}
    with pytest.raises(ArtifactValidationError, match="PBR.*shape"):
        validate_family_memberships(included, config)
```

- [ ] **Step 2: Run tests and verify RED**

```bash
conda run --no-capture-output -n pixal3d python -m pytest \
  tests/data_toolkit/test_reporting.py tests/data_toolkit/test_cli.py \
  tests/data_toolkit/test_pipeline_integration.py \
  -k 'family_counts or pbr_identity' -v
```

Expected: FAIL because audit assumes identical asset sets and reports no per-family counts.

- [ ] **Step 3: Implement held-artifact and report validation**

For every batch, load all eight manifests, validate the common union, and
enforce:

```python
for resolution in config.targets.resolutions:
    shape = included[f"shape-{resolution}"]
    pbr = included[f"PBR-{resolution}"]
    if not pbr <= shape:
        raise ArtifactValidationError(
            f"PBR-{resolution} membership is not a shape-{resolution} subset"
        )
non_common_union = set().union(
    *(values for family, values in included.items() if family != "common")
)
if included["common"] != non_common_union:
    raise ArtifactValidationError("common membership is not the family union")
```

Read the ledger and require every frozen-but-not-included identity to have a
global quarantine or matching family exclusion. Derive report family counts
from held manifest identities rather than candidate input fields.

- [ ] **Step 4: Run CLI, report, runtime, and integration tests**

```bash
conda run --no-capture-output -n pixal3d python -m pytest \
  tests/data_toolkit/test_reporting.py tests/data_toolkit/test_cli.py \
  tests/data_toolkit/test_pipeline_integration.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data_toolkit/pipeline/runtime.py data_toolkit/pipeline/reporting.py \
  data_toolkit/pipeline/cli.py tests/data_toolkit/test_reporting.py \
  tests/data_toolkit/test_cli.py tests/data_toolkit/test_pipeline_integration.py
git commit -m "feat: audit family-scoped handoff"
```

### Task 6: Full regression, migration, and ObjaverseXL smoke

**Files:**
- Modify: `data_toolkit/README.md`
- Modify: `docs/data_preprocessing_runbook_ko.md`
- Modify: `docs/superpowers/plans/2026-07-18-paper-faithful-family-eligibility.md`
- Runtime recovery: `/root/data2/pixal3d/control/recovery/objaversexl-current-rerun-20260718-HlPwoz/`
- Runtime frozen scope: `/root/data2/pixal3d/control/qualification/smoke/shards/ObjaverseXL_github/ObjaverseXL_github-00000/`

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: one clean new-commit 20-asset smoke publication and passing audit.

- [ ] **Step 1: Run the complete automated suite**

```bash
conda run --no-capture-output -n pixal3d python -m compileall -q data_toolkit
conda run --no-capture-output -n pixal3d python -m pytest tests/data_toolkit -q
```

Expected: all tests pass with no collection or runtime errors.

- [ ] **Step 2: Document operator-visible family behavior**

Update both runbooks with these exact rules:

```text
Unsupported standard-PBR parsing excludes only PBR family outputs.
Shape and SS outputs remain eligible when their own validators pass.
Pack manifests record both frozen asset_sha256s and exact included_asset_sha256s.
Never infer global quarantine counts from a PBR pack's excluded count.
```

- [ ] **Step 3: Commit the code-complete documentation**

```bash
git add data_toolkit/README.md docs/data_preprocessing_runbook_ko.md \
  docs/superpowers/plans/2026-07-18-paper-faithful-family-eligibility.md
git commit -m "docs: explain paper-faithful family filtering"
```

- [ ] **Step 4: Preserve the interrupted old-contract execution state**

Create a timestamped child directory beneath the existing recovery root. Move
only the verified current-rerun checkpoint, quality ledger, partial
qualification local tree, and current-rerun prepared/raw publications into it.
Write a sorted SHA-256 inventory before and after the move and verify the two
inventories match. Preserve the frozen `batch*.txt` files and their manifest.

Read-only resolution commands before moving:

```bash
find /root/data2/pixal3d/control/qualification/smoke \
  /root/data2/pixal3d/prepared/qualification/smoke \
  /root/data3/pixal3d/archive/qualification/smoke \
  /root/node17/data/pixal3d/preprocess/qualification/smoke \
  -path '*ObjaverseXL_github-00000*' -print
```

Expected: every target is batch-scoped and no frozen batch text file is in the move set.

- [ ] **Step 5: Run the clean frozen 20-asset smoke**

```bash
env PYTHONPATH=. conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli run \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --gate smoke --source ObjaverseXL_github \
  --shard ObjaverseXL_github-00000 --count 20
```

Expected: same frozen SHA scope, all batches terminal, geometry-valid unsupported-PBR assets present in shape/SS packs and absent from PBR packs.

- [ ] **Step 6: Audit the new publication**

```bash
env PYTHONPATH=. conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli audit \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --gate smoke --source ObjaverseXL_github \
  --shard ObjaverseXL_github-00000
```

Expected: exit 0, one tool commit across all new packs, exact family memberships, and no unexplained omitted identity.

- [ ] **Step 7: Record observed counts and continue the gate sequence**

Record geometry, SS, per-resolution shape/PBR, global quarantine, and exclusion
category counts in the Korean runbook. Then resume the next pending source/gate
from the existing preprocessing plan; do not hard-code the provisional 18/15/2
diagnostic expectation as a pass condition.

## Self-review

- Spec coverage: family dependencies, durable error evidence, command routing,
  exact manifest membership, audit/report invariants, migration, and smoke are
  each covered by one task.
- Placeholder scan: the plan contains no deferred implementation marker; each
  task names interfaces, failing tests, implementation behavior, verification,
  and commit scope.
- Type consistency: family names use the existing `PACK_FAMILIES` spelling;
  manifest included identities are tuples; ledger exclusions are nested
  mappings; runner/service method names are consistent across tasks.
