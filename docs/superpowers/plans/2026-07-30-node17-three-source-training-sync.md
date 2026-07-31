# Node17 Three-Source Training Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Synchronize Node17 with Node16's current multi-view fine-tuning code and publish a CPU-verified ABO + 3D-FUTURE + HSSD training input for all four models on Node17.

**Architecture:** Apply only the reviewed Node16 model/config/test delta to the clean Node17 worktree. Add a small shared configuration-policy module, an isolated resumable HSSD transfer/promote module, and a Node17-local publication orchestrator; reuse existing source preflight, combined-manifest, and configured DataLoader boundaries.

**Tech Stack:** Python 3.11, PyTorch 2.8, pytest, JSON/SHA-256 evidence, rsync over SSH, existing Pixal3D training materialization/preflight modules, Git.

## Global Constraints

- Node16 `/home/youngwoo/Pixal3D-multiview` is the code/config source of truth.
- Modify only `/root/dev/Pixal3D/.worktrees/multiview-model-extension`; never modify the dirty `/root/dev/Pixal3D` checkout.
- Do not import Node16 `ops/`, root-level design documents, caches, logs, generated outputs, or `.superpowers` scratch state.
- Preserve batch policies exactly: `ss64=8/4`, `shape512=8/4`, `shape1024=2/1`, `pbr1024=2/1`, six GPUs.
- Preserve `max_steps=20000`, `num_workers=2`, `i_print=10`, `i_log=10`, `i_sample=1000`, `i_save=1000`, and `max_checkpoints=3`.
- Preserve `snapshot_dataset_on_start=false` for `ss64`; do not add that key to the other three configs.
- Node17 runtime config generation may change only `/file2/youngwoo/pixal3d` to `/root/data2/pixal3d` and `/file3/youngwoo/pixal3d` to `/root/data3/pixal3d`.
- Reuse Node17 ABO and 3D-FUTURE without rematerialization or a full population rescan.
- Transfer only the completed HSSD materialization; do not run raw HSSD preprocessing.
- All readiness checks run with `CUDA_VISIBLE_DEVICES=""` and must leave CUDA uninitialized.
- All production publication is create-only; failures retain hidden staging for inspection and leave the canonical HSSD path absent.
- Do not start training, initialize six-GPU DDP, contact W&B, or write checkpoints.
- Every code task receives an independent spec-compliance review followed by a code-quality review before the next task.

---

### Task 1: Synchronize Node16 Fine-Tuning Config and Trainer Behavior

**Files:**
- Modify: `configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json`
- Modify: `configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json`
- Modify: `configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json`
- Modify: `configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json`
- Modify: `pixal3d/trainers/basic.py`
- Modify: `tests/multiview/test_configs.py`
- Modify: `tests/multiview/test_train_smoke_override.py`
- Modify: `tests/multiview/test_wandb_multiview.py`

**Interfaces:**
- Consumes: Node16 reference files under `/home/youngwoo/Pixal3D-multiview` and the approved policy in this plan.
- Produces: a `snapshot_dataset_on_start: bool = True` constructor option on `BasicTrainer`, rank-safe `BasicTrainer.snapshot`, current four config files, and regression coverage used by every later task.

- [ ] **Step 1: Add the authoritative failing config assertions**

Update `test_four_configs_use_batchwide_k_and_matching_checkpoints` to assert:

```python
assert trainer_args["num_workers"] == 2
assert trainer_args["i_print"] == 10
assert trainer_args["i_log"] == 10
assert trainer_args["i_sample"] == 1000
assert trainer_args["i_save"] == 1000
assert trainer_args["max_checkpoints"] == 3
if stage == "ss64":
    assert trainer_args["snapshot_dataset_on_start"] is False
else:
    assert "snapshot_dataset_on_start" not in trainer_args
```

- [ ] **Step 2: Add the Node16 trainer regression tests**

Bring in the three Node16 regressions. The fresh-start regression is:

```python
def test_ckpt_none_starts_fresh_even_when_output_has_a_latest_checkpoint(
    tmp_path,
):
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    (ckpt_dir / "misc_step0002000.pt").touch()
    config = edict({
        "load_dir": str(tmp_path),
        "ckpt": "none",
    })

    resolved = find_ckpt(config)

    assert resolved.load_ckpt is None
```

The startup-snapshot regression is:

```python
def test_run_can_skip_startup_dataset_snapshot_but_keeps_model_snapshot():
    trainer = object.__new__(BasicTrainer)
    trainer.is_master = True
    trainer.i_sample = 1000
    trainer.snapshot_dataset_on_start = False
    trainer.snapshot_num_samples = 1
    trainer.snapshot_batch_size = 1
    trainer.step = 0
    trainer.max_steps = 0
    trainer.world_size = 1
    trainer.writer = None
    calls = []
    trainer.snapshot_dataset = lambda **_kwargs: calls.append(
        "snapshot_dataset"
    )
    trainer.snapshot = lambda **kwargs: calls.append(
        ("snapshot", kwargs.get("suffix"))
    )

    BasicTrainer.run(trainer)

    assert "snapshot_dataset" not in calls
    assert ("snapshot", "init") in calls
```

The rank-zero failure regression is:

```python
def test_master_only_snapshot_reaches_barrier_when_rank_zero_sampling_fails(
    monkeypatch,
):
    trainer = object.__new__(BasicTrainer)
    trainer.is_master = True
    trainer.world_size = 6
    trainer.step = 0
    trainer.mix_precision_mode = None
    trainer.mix_precision_dtype = torch.bfloat16
    trainer.run_snapshot = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AttributeError("render failed")
    )
    barriers = []
    monkeypatch.setattr(basic.dist, "barrier", lambda: barriers.append(True))

    with pytest.raises(AttributeError, match="render failed"):
        BasicTrainer.snapshot(
            trainer, suffix="failure", num_samples=1, batch_size=1
        )
    assert barriers == [True]
```

Also add the W&B submission-output regression:

```python
assert "[W&B] Logged snapshot images at step 7:" in output
for key in sorted(payload):
    assert key in output
```

- [ ] **Step 3: Run the focused tests and verify the intended failures**

Run:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_configs.py \
  tests/multiview/test_train_smoke_override.py \
  tests/multiview/test_wandb_multiview.py -x -q
```

Expected: FAIL because the checked-in configs still contain the old
`14/-1/2000/5` values or because `BasicTrainer` does not yet accept
`snapshot_dataset_on_start`.

- [ ] **Step 4: Apply the reviewed Node16 config values**

Change only the fields listed in Global Constraints. In `ss64`, insert:

```json
"snapshot_dataset_on_start": false,
"num_workers": 2
```

For every stage, set:

```json
"i_print": 10,
"i_log": 10,
"i_sample": 1000,
"i_save": 1000,
"max_checkpoints": 3
```

Do not change model paths, dataset selection, batch/split, optimizer, or
attention settings.

- [ ] **Step 5: Apply the reviewed Node16 `BasicTrainer` behavior**

Add and store the constructor argument:

```python
snapshot_dataset_on_start=True,
self.snapshot_dataset_on_start = snapshot_dataset_on_start
```

Place the argument after `snapshot_num_samples` and assign it immediately
after `self.snapshot_num_samples`.

Wrap the master-only snapshot body in `try/finally` with exactly one matching
barrier:

```python
try:
    samples = self.run_snapshot(
        num_samples, batch_size=batch_size, verbose=verbose
    )
    # Keep the current metadata extraction, CUDA cleanup, and visualization
    # loop inside this try block.
finally:
    dist.barrier()
```

Catch visualization failures as `Exception`, retain the CUDA cleanup, and
print successful W&B image submission:

```python
keys = ", ".join(sorted(wandb_images))
print(f"[W&B] Logged snapshot images at step {self.step}: {keys}")
```

Gate only the startup dataset snapshot:

```python
if self.i_sample != -1 and self.snapshot_dataset_on_start:
    self.snapshot_dataset(
        num_samples=self.snapshot_num_samples,
        batch_size=self.snapshot_batch_size,
    )
elif self.i_sample != -1:
    print("[INFO] Startup dataset snapshot disabled.")
```

The existing initial model snapshot remains enabled when `i_sample != -1`.

- [ ] **Step 6: Run focused trainer/config tests**

Run the Step 3 command.

Expected: PASS, with integration checkpoint tests skipped only when their
released weight files are absent.

- [ ] **Step 7: Commit**

```bash
git add configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json \
  configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json \
  configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  pixal3d/trainers/basic.py \
  tests/multiview/test_configs.py \
  tests/multiview/test_train_smoke_override.py \
  tests/multiview/test_wandb_multiview.py
git commit -m "sync: align fine-tuning trainer with Node16"
```

---

### Task 2: Synchronize the Installed `utils3d` Camera API

**Files:**
- Modify: `pixal3d/datasets/flexi_dual_grid.py`
- Modify: `pixal3d/datasets/sparse_structure_latent.py`
- Modify: `pixal3d/datasets/sparse_voxel_pbr.py`
- Modify: `pixal3d/datasets/structured_latent.py`
- Modify: `pixal3d/datasets/structured_latent_shape.py`
- Modify: `pixal3d/datasets/structured_latent_svpbr.py`
- Modify: `pixal3d/trainers/vae/pbr_vae.py`
- Modify: `pixal3d/trainers/vae/shape_vae.py`
- Modify: `pixal3d/utils/render_utils.py`
- Modify: `tests/multiview/test_projection_geometry.py`

**Interfaces:**
- Consumes: installed `utils3d.torch.intrinsics_from_fov(*, fov_x, fov_y)`.
- Produces: all camera construction paths use the installed API while preserving square-FOV behavior.

- [ ] **Step 1: Add API and geometry regressions**

Add:

```python
def test_python_sources_use_current_utils3d_intrinsics_api():
    repo_root = Path(__file__).resolve().parents[2]
    obsolete_name = "intrinsics_from_fov" + "_xy"
    obsolete = [
        path.relative_to(repo_root)
        for path in repo_root.rglob("*.py")
        if obsolete_name in path.read_text()
    ]
    assert obsolete == []


def test_current_utils3d_square_fov_intrinsics_are_centered():
    fov = torch.tensor(torch.pi / 2)
    intrinsics = utils3d.torch.intrinsics_from_fov(
        fov_x=fov, fov_y=fov
    )
    assert intrinsics.shape == (3, 3)
    assert torch.allclose(
        intrinsics[:2, 2], torch.tensor([0.5, 0.5])
    )
```

- [ ] **Step 2: Run the regression and verify failure**

Run:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_projection_geometry.py -x -q
```

Expected: FAIL with a non-empty list of Python sources containing the obsolete
API.

- [ ] **Step 3: Replace every reviewed obsolete call**

In the nine source files listed above, replace:

```python
utils3d.torch.intrinsics_from_fov_xy(fov, fov)
```

or its equivalent tensor arguments with:

```python
utils3d.torch.intrinsics_from_fov(
    fov_x=fov,
    fov_y=fov,
)
```

Preserve each call's existing tensor device and dtype. Apply no other
rendering or camera change.

- [ ] **Step 4: Run geometry and source-scan tests**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pixal3d/datasets/flexi_dual_grid.py \
  pixal3d/datasets/sparse_structure_latent.py \
  pixal3d/datasets/sparse_voxel_pbr.py \
  pixal3d/datasets/structured_latent.py \
  pixal3d/datasets/structured_latent_shape.py \
  pixal3d/datasets/structured_latent_svpbr.py \
  pixal3d/trainers/vae/pbr_vae.py \
  pixal3d/trainers/vae/shape_vae.py \
  pixal3d/utils/render_utils.py \
  tests/multiview/test_projection_geometry.py
git commit -m "fix: use current utils3d intrinsics API"
```

---

### Task 3: Centralize Current Training Policy and Generate Node17 Runtime Configs

**Files:**
- Create: `data_toolkit/pipeline/training_config_policy.py`
- Modify: `data_toolkit/pipeline/node16_training_prepare.py`
- Modify: `tests/multiview/test_node16_training_prepare.py`
- Create: `tests/multiview/test_training_config_policy.py`

**Interfaces:**
- Consumes: ordered `Mapping[str, Path]` keyed by `STAGES`.
- Produces:
  - `StageTrainingPolicy(batch_size_per_gpu: int, batch_split: int, six_gpu_global_batch: int)`;
  - `validate_finetuning_configs(configs: Mapping[str, Path]) -> dict[str, dict[str, object]]`;
  - `rebase_json_paths(value: object, replacements: Sequence[tuple[str, str]]) -> object`;
  - `create_node17_runtime_configs(configs: Mapping[str, Path], output_root: Path) -> dict[str, Path]`;
  - `node17_runtime_config_evidence(runtime_configs: Mapping[str, Path], source_configs: Mapping[str, Path]) -> dict[str, dict[str, object]]`.

- [ ] **Step 1: Write failing policy and path-rebase tests**

Cover exact policy, typed values, create-only behavior, and path-only
transformation:

```python
def test_node17_runtime_configs_change_only_machine_path_prefixes(tmp_path):
    outputs = create_node17_runtime_configs(CONFIGS, tmp_path / "runtime")
    for stage, output in outputs.items():
        source = json.loads(CONFIGS[stage].read_text())
        runtime = json.loads(output.read_text())
        assert runtime["trainer"]["args"] == source["trainer"]["args"]
        assert "/file2/youngwoo/pixal3d" not in output.read_text()
        assert "/file3/youngwoo/pixal3d" not in output.read_text()
        restored = rebase_json_paths(
            runtime,
            (
                ("/root/data2/pixal3d", "/file2/youngwoo/pixal3d"),
                ("/root/data3/pixal3d", "/file3/youngwoo/pixal3d"),
            ),
        )
        assert restored == source
```

Also assert the output filenames end in `.node17.json`, partial existing
output is rejected, different existing bytes are rejected, and a complete
identical existing set is reused.

- [ ] **Step 2: Update Node16 preparation tests to the current source policy**

Change source-policy assertions and fixtures to:

```python
{
    "num_workers": 2,
    "i_print": 10,
    "i_log": 10,
    "i_sample": 1000,
    "i_save": 1000,
    "max_checkpoints": 3,
}
```

Keep the historical Node16 runtime transform of only `num_workers: 2 -> 1`;
update its evidence expectations to `save_interval=1000`,
`retained_checkpoints=3`, `snapshot_interval=1000`, and
`startup_dataset_snapshot=(stage != "ss64")`.

- [ ] **Step 3: Run policy tests and verify failure**

Run:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_training_config_policy.py \
  tests/multiview/test_node16_training_prepare.py -x -q
```

Expected: FAIL because `training_config_policy` is absent or the Node16
validator still requires `i_sample=-1`, `i_save=2000`, and
`max_checkpoints=5`.

- [ ] **Step 4: Implement typed current-policy validation**

Define:

```python
@dataclass(frozen=True)
class StageTrainingPolicy:
    batch_size_per_gpu: int
    batch_split: int
    six_gpu_global_batch: int


STAGE_POLICIES = {
    "ss64": StageTrainingPolicy(8, 4, 48),
    "shape512": StageTrainingPolicy(8, 4, 48),
    "shape1024": StageTrainingPolicy(2, 1, 12),
    "pbr1024": StageTrainingPolicy(2, 1, 12),
}
```

`validate_finetuning_configs` must reject bool-as-int values and enforce every
Global Constraint field. It must require
`snapshot_dataset_on_start is False` only for `ss64` and require the key to be
absent for the other stages.

- [ ] **Step 5: Implement create-only Node17 path rebasing**

Recursively transform string values only when they equal a source prefix or
begin with `source_prefix + "/"`. Use:

```python
NODE17_PATH_REPLACEMENTS = (
    ("/file2/youngwoo/pixal3d", "/root/data2/pixal3d"),
    ("/file3/youngwoo/pixal3d", "/root/data3/pixal3d"),
)
```

Write canonical, sorted, newline-terminated JSON using exclusive creation.
Validate an existing complete output set byte-for-byte; reject partial or
different sets. `node17_runtime_config_evidence` reverses the two prefixes and
proves equality with each source config before reporting SHA-256 and policy.

- [ ] **Step 6: Update the Node16 validator without changing its path contract**

Use the shared validator for source config semantics. Keep
`.node16-workers1.json` filenames and the single `num_workers=1` override.
Update only the stale interval/retention/snapshot evidence fields identified
in Step 2.

- [ ] **Step 7: Run policy and Node16 preparation tests**

Run the Step 3 command.

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add data_toolkit/pipeline/training_config_policy.py \
  data_toolkit/pipeline/node16_training_prepare.py \
  tests/multiview/test_training_config_policy.py \
  tests/multiview/test_node16_training_prepare.py
git commit -m "feat: add Node17 runtime config policy"
```

---

### Task 4: Implement Resumable HSSD Transfer, Strict Preflight, and Promotion

**Files:**
- Create: `data_toolkit/pipeline/node17_hssd_transfer.py`
- Create: `tests/multiview/test_node17_hssd_transfer.py`

**Interfaces:**
- Consumes:
  - `Node17HssdTransferPaths(source_host: str, source_port: int, source_root: Path, data2_root: Path, production_root: Path, staging_root: Path)`;
  - runtime config mapping from Task 3;
  - `command_runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]]` for unit isolation.
- Produces:
  - `TreeInventory(file_count: int, logical_bytes: int)`;
  - `plan_hssd_transfer(paths) -> dict[str, object]`;
  - `transfer_and_publish_hssd(paths, runtime_configs, command_runner=subprocess.run) -> HssdTransferResult`.

Define the path type as:

```python
@dataclass(frozen=True)
class Node17HssdTransferPaths:
    source_host: str
    source_port: int
    source_root: Path
    data2_root: Path
    production_root: Path
    staging_root: Path

    @property
    def canonical_root(self) -> Path:
        return self.production_root / "hssd"
```

Define the result type as:

```python
@dataclass(frozen=True)
class HssdTransferResult:
    source_inventory: TreeInventory
    target_inventory: TreeInventory
    original_materialization_sha256: dict[str, str]
    canonical_materialization_sha256: dict[str, str]
    training_data: Path
    training_data_sha256: str
    stage_counts: dict[str, int]
    elapsed_seconds: dict[str, float]
```

- [ ] **Step 1: Write failing path, admission, and rsync command tests**

Test canonical absolute roots, exact source identity, canonical-HSSD
conflicts, local free-space admission, and the generated commands:

```python
assert transfer_command[:4] == [
    "rsync", "-a", "--partial", "--info=progress2"
]
assert "--include=/ss64/***" in transfer_command
assert "--include=/shape512/***" in transfer_command
assert "--include=/shape1024/***" in transfer_command
assert "--include=/pbr1024/***" in transfer_command
assert transfer_command[-1] == str(paths.staging_root) + "/"
```

The verification command must add `--checksum`, `--dry-run`,
`--itemize-changes`, and `--delete`; any non-empty itemized output fails.

- [ ] **Step 2: Write failing JSON rebase and preflight-result remap tests**

Create synthetic materialization evidence containing Node16 local and shared
paths. Assert the first transform maps:

```text
/home/youngwoo/data/pixal3d/train/production/hssd
  -> str(paths.staging_root)
/file2/youngwoo/pixal3d
  -> /root/data2/pixal3d
```

After a synthetic strict preflight result, assert the promotion transform maps
`str(paths.staging_root)` to `str(paths.canonical_root)` and changes no count,
scope, exclusion, pack digest, tool commit, or validation count.

- [ ] **Step 3: Write failing promotion rollback tests**

Use temporary same-filesystem directories. Force the publisher or final chain
validator to raise after `os.rename`. Assert:

```python
assert not paths.canonical_root.exists()
assert paths.staging_root.exists()
```

Also assert a successful promotion creates exactly the four stage
directories, `publication/report.json`, `publication/handoff.json`, and
`training_data.json`.

- [ ] **Step 4: Run transfer tests and verify module absence**

Run:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_node17_hssd_transfer.py -x -q
```

Expected: FAIL because `node17_hssd_transfer` is absent.

- [ ] **Step 5: Implement validation, inventory, and resumable transfer**

Use subprocess argument lists, never `shell=True`. Validate:

```python
source_host == "youngwoo@n16.unist.info"
source_port == 55555
source_root == Path(
    "/home/youngwoo/data/pixal3d/train/production/hssd"
)
canonical_root == production_root / "hssd"
staging_root.parent == production_root
staging_root.name == ".hssd-node16-transfer"
```

Query remote file count/logical bytes before local creation. Require
`logical_bytes + max(logical_bytes // 10, 10 * 1024**3)` free. Permit an
existing safe staging directory for resume, but reject symlinks and any
pre-existing canonical HSSD directory unless its full source chain already
validates.

- [ ] **Step 6: Implement exact transfer verification and evidence rebasing**

Rsync only the four stage directories. Run the checksum dry-run and require
zero itemized differences. Record source inventory, target inventory, and
the four original `materialization.json` SHA-256 values before rewriting.

Recursively replace only exact path prefixes and atomically rewrite each
materialization JSON. Reject an old Node16 prefix remaining anywhere in a
rewritten document.

- [ ] **Step 7: Implement strict staged preflight and rollback-safe promotion**

Run `training_preflight.preflight_stage` for all four stages against staging
with CUDA hidden. Rebase each `StagePreflight.materialization_bytes` and
`.root` from staging to canonical while retaining validated counts and scope.
Rewrite the four on-disk materialization files to the same canonical bytes.

Atomically rename staging to canonical, call
`training_preflight.publish_source_handoff`, then call
`validate_source_training_data("HSSD", canonical/training_data.json)`.
On any exception after rename, rename canonical back to staging and re-raise.
Return evidence containing original/rebased materialization digests, inventory,
published chain digests, stage counts, and elapsed times.

- [ ] **Step 8: Run transfer tests**

Run the Step 4 command.

Expected: PASS without network access and without CUDA initialization.

- [ ] **Step 9: Commit**

```bash
git add data_toolkit/pipeline/node17_hssd_transfer.py \
  tests/multiview/test_node17_hssd_transfer.py
git commit -m "feat: add resumable Node17 HSSD transfer"
```

---

### Task 5: Orchestrate Node17 Three-Source Publication and Readiness Evidence

**Files:**
- Create: `data_toolkit/pipeline/node17_training_prepare.py`
- Create: `scripts/prepare_node17_training.py`
- Create: `tests/multiview/test_node17_training_prepare.py`
- Create: `docs/node17_three_source_training_runbook_ko.md`

**Interfaces:**
- Consumes:
  - `Node17PreparationPaths.from_roots(data2_root: Path, local_root: Path, repo_root: Path)`;
  - `transfer_and_publish_hssd` from Task 4;
  - `create_node17_runtime_configs` from Task 3.
- Produces:
  - `plan_node17_training(paths) -> dict[str, object]`;
  - `prepare_node17_training(paths) -> Path`;
  - CLI plan mode by default and mutation only with `--execute`;
  - immutable `/root/node17/data/pixal3d/train/production/node17-preparation-evidence/report.json`.

- [ ] **Step 1: Write failing root and plan-mode tests**

Assert defaults:

```python
assert paths.production_root == Path(
    "/root/node17/data/pixal3d/train/production"
)
assert paths.runtime_config_root == Path(
    "/root/node17/data/pixal3d/train/runtime-configs"
)
assert paths.hssd_training_data == (
    paths.production_root / "hssd/training_data.json"
)
assert paths.combined_training_data == (
    paths.production_root
    / "abo-3d-future-hssd/training_data.json"
)
```

Plan mode must validate roots/configs/disk and perform no mkdir, transfer,
publication, CUDA import, or W&B call.

- [ ] **Step 2: Write failing orchestration-order and source-reuse tests**

Mock task boundaries and require this order:

```text
validate and record the clean Git HEAD revision
create/validate Node17 runtime configs
transfer + full-preflight + publish HSSD
validate ABO immutable publication evidence
validate 3D-FUTURE immutable publication evidence
preflight HSSD standalone configured loaders
publish three-source combined manifest
preflight combined configured loaders
write immutable final report
```

Assert ABO and 3D-FUTURE use `validate_source_training_data` only and never
call `materialize_stage` or `preflight_stage`.

- [ ] **Step 3: Write failing report and launch-command tests**

Require the report to include revision, runtime config digests, transfer
inventory, HSSD original/rebased evidence digests, all source-chain digests,
per-stage source counts, standalone and combined DataLoader results, elapsed
times, and four launch commands.

Each command must use the common arguments:

```text
/opt/conda/envs/pixal3d/bin/python train.py
--training_data /root/node17/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json
--num_gpus 6 --use_wandb
```

The four exact `--config` values are:

```text
/root/node17/data/pixal3d/train/runtime-configs/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.node17.json
/root/node17/data/pixal3d/train/runtime-configs/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.node17.json
/root/node17/data/pixal3d/train/runtime-configs/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.node17.json
/root/node17/data/pixal3d/train/runtime-configs/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.node17.json
```

The command generator must create no log directory; the runbook performs
`mkdir -p /root/node17/data/pixal3d/training-logs` before `tee`.

- [ ] **Step 4: Run Node17 preparation tests and verify failure**

Run:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_node17_training_prepare.py -x -q
```

Expected: FAIL because the module and CLI are absent.

- [ ] **Step 5: Implement create-only local orchestration**

Use existing functions:

```python
validate_source_training_data
publish_combined_training_data
preflight_multisource_stage
```

Check CUDA before and after each HSSD/combined stage. Publish sources in exact
order `ABO`, `3D-FUTURE`, `HSSD`. Reject a partial combined directory. Reuse
an existing complete identical combined manifest only after strict manifest
resolution.

- [ ] **Step 6: Implement immutable report validation**

Before reusing or creating the report, re-hash every referenced artifact and
re-resolve every source and combined manifest. Ignore only volatile disk-free
and elapsed-time values when comparing a rerun with an existing report.
Reject a partial evidence directory or any changed invariant.

- [ ] **Step 7: Implement the CPU-only CLI**

Defaults:

```text
--data2-root /root/data2/pixal3d
--local-root /root/node17/data/pixal3d
--repo-root /root/dev/Pixal3D/.worktrees/multiview-model-extension
--source-host youngwoo@n16.unist.info
--source-port 55555
--source-root /home/youngwoo/data/pixal3d/train/production/hssd
```

Require `CUDA_VISIBLE_DEVICES=""` and `PYTHONDONTWRITEBYTECODE=1` at import
time. Print the plan as JSON unless `--execute` is supplied.

- [ ] **Step 8: Write the Korean Node17 runbook**

Document exact plan, execute, resume, evidence-check, and four training
commands. Begin launch setup with:

```bash
mkdir -p /root/node17/data/pixal3d/training-logs
cd /root/dev/Pixal3D/.worktrees/multiview-model-extension
export PYTHONPATH=.
```

State that execution does not start training or W&B.

- [ ] **Step 9: Run preparation, manifest, entrypoint, and CLI tests**

Run:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_node17_training_prepare.py \
  tests/multiview/test_training_manifest.py \
  tests/multiview/test_multisource_preflight.py \
  tests/multiview/test_training_entrypoint.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add data_toolkit/pipeline/node17_training_prepare.py \
  scripts/prepare_node17_training.py \
  tests/multiview/test_node17_training_prepare.py \
  docs/node17_three_source_training_runbook_ko.md
git commit -m "feat: prepare Node17 three-source training"
```

---

### Task 6: Run the Full Code Verification Gate

**Files:**
- Modify only files required to correct failures attributable to Tasks 1-5.
- Create: `docs/superpowers/reports/2026-07-30-node17-code-sync-verification.md`

**Interfaces:**
- Consumes: committed Tasks 1-5.
- Produces: one clean reviewed revision eligible for production-data execution.

- [ ] **Step 1: Verify the Node16 review scope**

Run checksum comparison for the 18 reviewed Node16 files. Confirm that every
difference is either an exact Node16 change or the deliberate
`test_configs.py` correction to `1000/1000/3`. Confirm no `ops/`, root-level
July 31 documents, cache, or scratch file entered Git.

- [ ] **Step 2: Run the complete CPU-only multi-view suite**

Run:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
/opt/conda/envs/pixal3d/bin/python -m pytest tests/multiview -q
```

Expected: PASS; only explicitly optional integration tests may skip.

- [ ] **Step 3: Run static and Git hygiene checks**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m compileall -q \
  data_toolkit/pipeline pixal3d scripts
git diff --check
git status --short
```

Expected: compile succeeds, `git diff --check` is silent, and status contains
only the verification report before its commit.

- [ ] **Step 4: Record and commit verification evidence**

Record the exact revision, commands, test counts, skips, elapsed time, and
Node16 checksum-scope result.

```bash
git add docs/superpowers/reports/2026-07-30-node17-code-sync-verification.md
git commit -m "docs: verify Node17 code synchronization"
```

---

### Task 7: Transfer HSSD and Publish the Node17 Three-Source Input

**Files:**
- Create outside Git: `/root/node17/data/pixal3d/train/production/.hssd-node16-transfer`
- Create outside Git: `/root/node17/data/pixal3d/train/production/hssd`
- Create outside Git: `/root/node17/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json`
- Create outside Git: `/root/node17/data/pixal3d/train/runtime-configs/*.node17.json`
- Create outside Git: `/root/node17/data/pixal3d/train/production/node17-preparation-evidence/report.json`
- Create: `docs/superpowers/reports/2026-07-30-node17-three-source-training-readiness.md`

**Interfaces:**
- Consumes: the reviewed Task 6 Git revision and key-based Node16 SSH access.
- Produces: canonical Node17 HSSD and three-source training publications plus reproducible evidence; no GPU process.

- [ ] **Step 1: Audit before mutation**

Run:

```bash
git status --short
df -B1 /root/node17/data
test ! -e /root/node17/data/pixal3d/train/production/hssd
test ! -e /root/node17/data/pixal3d/train/production/abo-3d-future-hssd
ssh -p 55555 -o BatchMode=yes youngwoo@n16.unist.info \
  'test -f /home/youngwoo/data/pixal3d/train/production/hssd/training_data.json'
```

Expected: clean Git status, sufficient local disk, absent canonical Node17
targets, and successful source existence check.

- [ ] **Step 2: Run non-mutating preparation plan**

Run:

```bash
cd /root/dev/Pixal3D/.worktrees/multiview-model-extension
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
/opt/conda/envs/pixal3d/bin/python scripts/prepare_node17_training.py
```

Expected: JSON plan showing the exact source, staging/canonical paths, remote
inventory, disk admission, configs, and combined output without creating
directories.

- [ ] **Step 3: Execute transfer and full CPU-only preflight**

Run:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
/opt/conda/envs/pixal3d/bin/python scripts/prepare_node17_training.py \
  --execute 2>&1 | tee \
  /root/node17/data/pixal3d/node17-three-source-prepare.log
```

Expected: resumable HSSD rsync, checksum verification, four full strict HSSD
stage preflights, standalone HSSD loader checks, three-source publication,
and four combined loader checks. No CUDA or W&B initialization occurs.

- [ ] **Step 4: Validate final immutable artifacts**

Run:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
/opt/conda/envs/pixal3d/bin/python scripts/prepare_node17_training.py

CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
/opt/conda/envs/pixal3d/bin/python scripts/preflight_multisource_training.py \
  --training-data \
  /root/node17/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json
```

Expected: plan reports complete reusable publications and the independent
configured DataLoader preflight passes all four stages.

- [ ] **Step 5: Confirm no training process and inspect GPU state**

Run:

```bash
pgrep -af 'train.py|torchrun' || true
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory \
  --format=csv,noheader
```

Expected: this workflow introduced no `train.py` or `torchrun` process.
Existing unrelated GPU processes are reported but not touched.

- [ ] **Step 6: Record readiness evidence and commit**

Record:

- reviewed Git revision;
- source/target HSSD counts and logical bytes;
- checksum verification result;
- original and Node17 materialization digests;
- ABO, 3D-FUTURE, HSSD, and combined manifest digests;
- every stage/source count;
- standalone and combined loader results;
- runtime-config digests;
- total transfer/preflight time;
- the four exact launch commands.

```bash
git add docs/superpowers/reports/2026-07-30-node17-three-source-training-readiness.md
git commit -m "docs: record Node17 three-source readiness"
```

---

### Task 8: Final Cumulative Review and Handoff

**Files:**
- Modify only files needed to resolve findings.

**Interfaces:**
- Consumes: the complete committed range from the design commit through Task 7.
- Produces: final reviewed code/data readiness with exact training commands.

- [ ] **Step 1: Dispatch a cumulative spec-compliance review**

Review every acceptance criterion in
`docs/superpowers/specs/2026-07-30-node17-three-source-training-sync-design.md`
against commits, tests, and production evidence. Reject any claim supported
only by sampled HSSD validation.

- [ ] **Step 2: Dispatch a cumulative code-quality and safety review**

Review path validation, subprocess argument construction, exclusive creation,
resume behavior, atomic rename/rollback, digest binding, CUDA guards, source
reuse, and report invariant validation.

- [ ] **Step 3: Correct findings with focused regression tests**

For each accepted finding, first add a regression that fails for the
demonstrated issue, apply the minimum fix, run the focused test, and rerun:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
/opt/conda/envs/pixal3d/bin/python -m pytest tests/multiview -q
```

- [ ] **Step 4: Revalidate production evidence after any fix**

Run the non-mutating Node17 preparation plan. If a fix affects evidence,
manifest resolution, loaders, or path binding, rerun the affected read-only
validation. Do not replace create-only publications.

- [ ] **Step 5: Deliver the final handoff**

Report the final Git revision, canonical three-source manifest, runtime
configs, per-stage batches, evidence report, test result, transfer/preflight
duration, and the four copy-paste training commands. State explicitly that
training and W&B have not been started.
