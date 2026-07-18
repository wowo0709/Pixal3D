# Gate Evidence And Full Preprocessing Launch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the approved smoke/pilot evidence contract and start the full five-source production download/preprocessing sequence without bypassing audits or changing the paper-faithful FP32 data contract.

**Architecture:** Add a held-evidence collector that derives measurements from frozen scopes, schema-v2 publications, schema-v3 quality ledgers, and actual resource telemetry. Keep FP32 active, pass smoke and pilot, then use one resumable runner to process canonical production shards in fixed source order with an audit after every shard.

**Tech Stack:** Python 3.11, pandas, PyTorch 2.8.0+cu128, CUDA 12.8, pytest, existing `data_toolkit.pipeline`.

## Global Constraints

- CUDA 12.8 and PyTorch 2.8 or newer are mandatory.
- Local scratch is `/root/node17/data/pixal3d`; data2 and data3 roots remain `/root/data2/pixal3d` and `/root/data3/pixal3d`.
- Keep `targets.latent_dtype: float32`; a dtype change requires explicit user approval and a new config hash.
- Preserve frozen SHA lists. Recover old mutable artifacts before current-commit reruns.
- Process sequentially and audit before advancing.
- Global quarantine and family exclusions remain durable; evidence must never turn them into fabricated successes.
- Source admission is at least 90% terminal usable assets with zero smoke schema/infrastructure failures.

---

### Task 1: Align Gate Decisions With FP32 And Quality Policy

**Files:**
- Modify: `data_toolkit/pipeline/runtime.py`
- Modify: `data_toolkit/pipeline/reporting.py`
- Test: `tests/data_toolkit/test_cli.py`
- Test: `tests/data_toolkit/test_reporting.py`

**Interfaces:**
- Consumes: held measurement/FP16/telemetry artifacts and `PipelineConfig.targets.latent_dtype`.
- Produces: `fp16_gate_summary(measurements, latent_dtype) -> dict` and the approved 90% smoke admission rule.

- [ ] **Step 1: Write failing FP32 tests**

Add this assertion and a float16 companion that still requires the existing six groups of 32 unique assets:

```python
def test_fp32_gate_does_not_require_fp16_qualification():
    frame = pd.DataFrame(columns={
        "sha256", "family", "resolution", "fp16_abs_error",
        "coordinates_match", "fp16_finite", "decode_degradation_percent",
    })
    assert fp16_gate_summary(frame, "float32") == {
        "dtype": "float32",
        "required": False,
        "passed": True,
        "families": {},
    }
```

- [ ] **Step 2: Write failing quality and telemetry tests**

Build five-source smoke measurements with two provider failures among 20 GitHub assets and no schema failures; require a pass at exactly 90%. Change one row to `schema_failure` and require failure. Also require old timezone-aware telemetry to remain valid while a fresh evidence manifest binds it; reject naive or future telemetry.

- [ ] **Step 3: Run RED**

```bash
conda run --no-capture-output -n pixal3d \
  python -m pytest tests/data_toolkit/test_reporting.py tests/data_toolkit/test_cli.py \
  -k 'fp32 or smoke_threshold or historical_telemetry' -v
```

- [ ] **Step 4: Implement the minimal policy**

```python
def fp16_gate_summary(measurements: pd.DataFrame, latent_dtype: str) -> dict:
    columns = {
        "sha256", "family", "resolution", "fp16_abs_error",
        "coordinates_match", "fp16_finite", "decode_degradation_percent",
    }
    if latent_dtype == "float32":
        if set(measurements.columns) != columns or not measurements.empty:
            raise ReportValidationError(
                "FP32 gate requires header-only FP16 evidence"
            )
        return {
            "dtype": "float32", "required": False,
            "passed": True, "families": {},
        }
    families = fp16_family_summary(measurements)
    return {
        "dtype": "float16", "required": True,
        "passed": all(item["passed"] for item in families.values()),
        "families": families,
    }
```

Use `parity["passed"]` in the gate decision. For smoke require overall and per-source failure rate at most 10% and zero overall/per-source schema failures. Retain the current pilot/production thresholds. Telemetry rows must be timezone-aware and no more than five minutes in the future, but may be older than 24 hours; only the evidence manifest keeps the 24-hour freshness check.

- [ ] **Step 5: Run GREEN and commit**

```bash
conda run --no-capture-output -n pixal3d \
  python -m pytest tests/data_toolkit/test_reporting.py tests/data_toolkit/test_cli.py -q
conda run --no-capture-output -n pixal3d python -m pytest tests/data_toolkit -q
git add data_toolkit/pipeline/runtime.py data_toolkit/pipeline/reporting.py \
  tests/data_toolkit/test_cli.py tests/data_toolkit/test_reporting.py
git commit -m "fix: align gate admission with FP32 policy"
```

---

### Task 2: Collect Checksum-Bound Evidence From Actual Runs

**Files:**
- Create: `data_toolkit/pipeline/evidence.py`
- Modify: `data_toolkit/pipeline/cli.py`
- Test: `tests/data_toolkit/test_evidence.py`
- Test: `tests/data_toolkit/test_cli.py`

**Interfaces:**
- Consumes: frozen batches, validated publication manifests, schema-v3 quality ledgers, and `control/telemetry/resources.jsonl`.
- Produces: `GateEvidenceCollector(config).collect(gate) -> tuple[Path, Path, Path, Path]`.

- [ ] **Step 1: Write failing segmentation tests**

Create complete telemetry sequences beginning with `stage_raw` and ending with successful `cleanup_local`, containing `build_packs` and `archive_raw`. Require the latest exact number of complete segments and reject missing boundaries.

- [ ] **Step 2: Write failing accounting tests**

Use schema-v2 manifests with differing family membership. Require each tar's exact byte size to be allocated only over `included_asset_sha256s`, with allocations summing exactly to the tar size. Raw bytes are allocated only over `common` membership.

- [ ] **Step 3: Write a failing integration test**

Require exactly these measurement fields, durable quarantine categories, `failure_category="none"` for completed assets, a header-only FP16 CSV for FP32, scope-bound telemetry, and correct SHA-256 values in the schema-v2 evidence manifest:

```python
MEASUREMENT_COLUMNS = (
    "sha256", "source", "shard_id", "outcome", "failure_category",
    "elapsed_seconds", "peak_local_bytes", "final_local_bytes",
    "final_data2_bytes", "final_data3_bytes",
)
```

- [ ] **Step 4: Run RED**

```bash
conda run --no-capture-output -n pixal3d \
  python -m pytest tests/data_toolkit/test_evidence.py -v
```

Expected: import failure because the collector does not exist.

- [ ] **Step 5: Implement the collector**

```python
REQUIRED_SEGMENT_COMMANDS = frozenset({
    "stage_raw", "build_packs", "archive_raw", "cleanup_local",
})

class GateEvidenceCollector:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def collect(self, gate: str) -> tuple[Path, Path, Path, Path]:
        """Validate publications, derive held rows, and publish manifest last."""
```

Implementation rules:

- Reopen frozen scopes and publications through the existing runtime validators.
- Split telemetry into complete batch segments and match them to ordered frozen batches.
- Divide observed segment elapsed time by frozen batch count.
- Derive peak local bytes from the segment's observed `local_free_gib` range and divide by batch count.
- Allocate exact tar bytes over included assets, preserving total tar overhead.
- Set final local bytes to zero only for a segment with successful cleanup and a passing publication audit.
- Read outcome/category from the schema-v3 ledger without rewriting it.
- Copy selected telemetry rows and add canonical `gate` and `source`.
- Write a header-only FP16 CSV for FP32; reject float16 collection without real decoder evidence.
- Atomically write measurements, FP16, and telemetry, compute SHA-256, and atomically publish the manifest last.

- [ ] **Step 6: Add the CLI**

Add `evidence --config PATH --gate {smoke,pilot,production}`. It must not freeze scopes or initialize asset providers.

- [ ] **Step 7: Run GREEN and commit**

```bash
conda run --no-capture-output -n pixal3d \
  python -m pytest tests/data_toolkit/test_evidence.py tests/data_toolkit/test_cli.py -q
conda run --no-capture-output -n pixal3d python -m pytest tests/data_toolkit -q
git diff --check
git add data_toolkit/pipeline/evidence.py data_toolkit/pipeline/cli.py \
  tests/data_toolkit/test_evidence.py tests/data_toolkit/test_cli.py
git commit -m "feat: collect held preprocessing gate evidence"
```

---

### Task 3: Finish Current-Contract Smoke And Pass Its Report

**Files:**
- Runtime: `/root/data2/pixal3d/control/qualification/smoke/**`
- Recovery: `/root/data2/pixal3d/control/recovery/**`
- Recovery: `/root/data3/pixal3d/recovery/**`
- Modify: `docs/data_preprocessing_runbook_ko.md`

**Interfaces:**
- Consumes: existing immutable frozen SHA lists and current commit.
- Produces: schema-v2 publications/schema-v3 ledgers for every source and a passed smoke report.

- [ ] **Step 1: Finish and audit the active 3D-FUTURE rerun**

```bash
conda run --no-capture-output -n pixal3d python -m data_toolkit.pipeline.cli audit \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --gate smoke --source 3D-FUTURE --shard 3D-FUTURE-00000
```

- [ ] **Step 2: Recover and rerun HSSD, ABO, and Sketchfab sequentially**

Preserve each frozen `shards/<source>/<shard>` directory. Move only checkpoints, ledger, packs/index, archives, and scratch into timestamped recovery roots. Run HSSD and ABO with frozen count 9 and Sketchfab with frozen count 20, then audit each immediately:

```bash
declare -A smoke_counts=([HSSD]=9 [ABO]=9 [ObjaverseXL_sketchfab]=20)
for source in HSSD ABO ObjaverseXL_sketchfab; do
  count="${smoke_counts[$source]}"
  conda run --no-capture-output -n pixal3d \
    python -m data_toolkit.pipeline.cli run \
    --config data_toolkit/configs/multiview_preprocess.yaml \
    --gate smoke --source "$source" --shard "$source-00000" --count "$count"
  conda run --no-capture-output -n pixal3d \
    python -m data_toolkit.pipeline.cli audit \
    --config data_toolkit/configs/multiview_preprocess.yaml \
    --gate smoke --source "$source" --shard "$source-00000"
done
```

- [ ] **Step 3: Refresh hardware evidence and produce the report**

```bash
conda run --no-capture-output -n pixal3d python -m data_toolkit.pipeline.cli hardware-preflight \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --bootstrap-peak-local-gib 350
conda run --no-capture-output -n pixal3d python -m data_toolkit.pipeline.cli evidence \
  --config data_toolkit/configs/multiview_preprocess.yaml --gate smoke
conda run --no-capture-output -n pixal3d python -m data_toolkit.pipeline.cli report \
  --config data_toolkit/configs/multiview_preprocess.yaml --gate smoke
```

- [ ] **Step 4: Record current commit, counts, recovery paths, report hashes, and commands in the Korean runbook; commit**

---

### Task 4: Run The 1,000-Asset Pilot

**Files:**
- Runtime: `/root/data2/pixal3d/control/qualification/pilot/**`
- Runtime: `/root/data2/pixal3d/control/reports/gates/pilot.json`
- Modify: `docs/data_preprocessing_runbook_ko.md`

**Interfaces:**
- Consumes: passed smoke report with active config hash.
- Produces: 1,000 terminal attempts, audited publications, capacity/ETA evidence, and production admission.

- [ ] **Step 1: Freeze 200 assets per source**

Use equal source coverage to keep per-source quality/capacity estimates independently meaningful:

```bash
for source in ObjaverseXL_sketchfab ObjaverseXL_github ABO HSSD 3D-FUTURE; do
  conda run --no-capture-output -n pixal3d python -m data_toolkit.pipeline.cli plan \
    --config data_toolkit/configs/multiview_preprocess.yaml --gate pilot \
    --source "$source" --shard "$source-00000" --count 200
done
```

- [ ] **Step 2: Run `--count 200` and audit each source sequentially in fixed source order**

Resume exact frozen scopes after interruption. Do not replan from current free space.

- [ ] **Step 3: Refresh hardware evidence when needed, collect pilot evidence, and derive the pilot report**

```bash
conda run --no-capture-output -n pixal3d python -m data_toolkit.pipeline.cli evidence \
  --config data_toolkit/configs/multiview_preprocess.yaml --gate pilot
conda run --no-capture-output -n pixal3d python -m data_toolkit.pipeline.cli report \
  --config data_toolkit/configs/multiview_preprocess.yaml --gate pilot
```

- [ ] **Step 4: Stop for user review only if a guard trips**

Stop if projected data2 exceeds 16 TiB, data3 exceeds 26 TiB, any source is below 90%, schema failure exceeds 5%, or checksum/hardware/audit fails. Keep FP32 regardless of parity unless the user explicitly approves a dtype change.

- [ ] **Step 5: Record per-source quality, p95 local/final bytes, projected totals, ETA, and report hash in the runbook**

---

### Task 5: Add And Launch A Resumable Full Runner

**Files:**
- Create: `data_toolkit/pipeline/full_run.py`
- Modify: `data_toolkit/pipeline/cli.py`
- Test: `tests/data_toolkit/test_full_run.py`
- Modify: `docs/data_preprocessing_runbook_ko.md`

**Interfaces:**
- Consumes: passed smoke/pilot reports and canonical training registry.
- Produces: `FullProductionRunner.run()`, processing all canonical shards in audited source order.

- [ ] **Step 1: Write failing ordering and resume tests**

Use two Sketchfab shards and one per remaining source. Require `run` then `audit` for each shard, all Sketchfab before GitHub, then ABO, HSSD, 3D-FUTURE. Require `count=None`. Simulate interruption and require the same frozen shard to be revalidated before advancement.

- [ ] **Step 2: Implement the runner**

```python
SOURCE_ORDER = (
    "ObjaverseXL_sketchfab", "ObjaverseXL_github",
    "ABO", "HSSD", "3D-FUTURE",
)

class FullProductionRunner:
    def __init__(self, config: PipelineConfig, services: PipelineServices):
        self.config = config
        self.services = services

    def run(self) -> None:
        read_gate_report(self.config, "smoke")
        read_gate_report(self.config, "pilot")
        registry = SafeRegistryStore(
            self.config.paths.data2_root / "control/assets.parquet", self.config
        ).load()
        for source in SOURCE_ORDER:
            shards = sorted(registry.loc[
                registry["owner_source"] == source, "shard_id"
            ].unique())
            for shard in shards:
                self.services.run("production", source, shard, None)
                self.services.audit("production", source, shard)
```

Add `full-run --config`. It initializes the existing mutating runtime, rejects a count, and propagates the first stop before another shard is scheduled.

- [ ] **Step 3: Run tests and commit**

```bash
conda run --no-capture-output -n pixal3d \
  python -m pytest tests/data_toolkit/test_full_run.py tests/data_toolkit/test_cli.py -q
conda run --no-capture-output -n pixal3d python -m pytest tests/data_toolkit -q
git diff --check
git add data_toolkit/pipeline/full_run.py data_toolkit/pipeline/cli.py \
  tests/data_toolkit/test_full_run.py tests/data_toolkit/test_cli.py \
  docs/data_preprocessing_runbook_ko.md
git commit -m "feat: run full preprocessing in audited shard order"
```

- [ ] **Step 4: Launch after the pilot passes**

```bash
cd /root/dev/Pixal3D/.worktrees/multiview-preprocess
env PYTHONPATH=. conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli full-run \
  --config data_toolkit/configs/multiview_preprocess.yaml
```

Downloads happen on demand. ABO reuses `/root/data2/pixal3d/raw/ABO/raw/abo-3dmodels.tar`. A restart uses the identical command and exact frozen production scopes.

- [ ] **Step 5: After all shards audit, collect/report production and require checksum-bound `control/splits/training_handoff.json`**

---

## Final Verification

- [ ] Run `python -m compileall -q data_toolkit`.
- [ ] Run the full `tests/data_toolkit` suite and `git diff --check`.
- [ ] Verify Torch >=2.8, CUDA 12.8, CUDA available, and seven GPUs.
- [ ] Verify every report/evidence manifest has the active config hash.
- [ ] Verify no production command has `--count`.
- [ ] Verify preprocessing does not overlap model training.
