# Multi-view Checkpoint Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Disable snapshots, standardize 5,000-step checkpointing and `batch_split = 1`, and retain only the five latest complete checkpoints for all four multi-view fine-tuning configs.

**Architecture:** Add an opt-in `max_checkpoints` trainer argument. Retention-enabled saves run synchronously so a `misc_stepXXXXXXX.pt` completion marker is written only after model and EMA files succeed; pruning then removes all files for obsolete complete steps. Configs without the option preserve the existing asynchronous save behavior.

**Tech Stack:** Python, PyTorch checkpoint I/O, pytest, JSON training configs.

## Global Constraints

- All four multi-view configs set `batch_split` to exactly `1`.
- All four multi-view configs set `i_sample` to exactly `-1`.
- All four multi-view configs set `i_save` to exactly `5000`.
- All four multi-view configs set `max_checkpoints` to exactly `5`.
- Do not change model architecture, dataset behavior, loss, optimizer, or `max_steps`.
- `misc_stepXXXXXXX.pt` is the only complete-checkpoint index.
- Pruning runs only after every file for the current checkpoint saves successfully.
- A failed save must not delete any older complete checkpoint.
- Configs without `max_checkpoints` preserve existing save behavior and never prune.

---

### Task 1: Standardize multi-view outputs and add safe checkpoint retention

**Files:**
- Modify: `pixal3d/trainers/basic.py`
- Modify: `configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json`
- Modify: `configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json`
- Modify: `configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json`
- Modify: `configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json`
- Modify: `tests/multiview/test_configs.py`
- Create: `tests/multiview/test_checkpoint_retention.py`

**Interfaces:**
- Consumes: `BasicTrainer.save(non_blocking: bool = True)` and the existing `ckpts/*_stepXXXXXXX.pt` filename layout.
- Produces: optional `BasicTrainer.__init__(..., max_checkpoints: int | None = None, ...)` behavior and a private pruning helper that indexes complete steps from `misc_stepXXXXXXX.pt`.

- [ ] **Step 1: Write failing config-policy tests**

Extend the four-config loop in `tests/multiview/test_configs.py` with:

```python
assert trainer_args["batch_split"] == 1
assert trainer_args["i_sample"] == -1
assert trainer_args["i_save"] == 5000
assert trainer_args["max_checkpoints"] == 5
```

- [ ] **Step 2: Write failing retention tests**

Create `tests/multiview/test_checkpoint_retention.py` with small CPU-only trainer fixtures that verify:

```python
def test_retention_keeps_latest_five_complete_checkpoints(...):
    # Save a sixth complete step and assert all model, EMA, and misc files
    # for only the oldest complete step are removed.

def test_failed_save_keeps_all_previous_complete_checkpoints(...):
    # Make torch.save raise while writing the new step and assert every
    # pre-existing complete checkpoint remains.

def test_incomplete_step_is_not_a_retention_marker(...):
    # Add model/EMA files without misc for a newer step and assert it is
    # not counted among the five complete checkpoints.

@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5, "5"])
def test_max_checkpoints_requires_a_positive_integer(invalid):
    # Exercise the validation without initializing GPU training state.
```

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
conda run --no-capture-output -n pixal3d python -m pytest -q \
  tests/multiview/test_configs.py::test_four_configs_use_batchwide_k_and_matching_checkpoints \
  tests/multiview/test_checkpoint_retention.py
```

Expected: FAIL because config values and `BasicTrainer.max_checkpoints` retention do not exist yet.

- [ ] **Step 4: Implement the minimal trainer behavior**

In `BasicTrainer.__init__`, add `max_checkpoints=None`, reject values for which:

```python
max_checkpoints is not None and (
    isinstance(max_checkpoints, bool)
    or not isinstance(max_checkpoints, int)
    or max_checkpoints <= 0
)
```

Store the validated option. In `save`, keep the current asynchronous code path unchanged when it is `None`. When configured, save model files, then EMA files, then misc synchronously. After all saves return successfully, find complete steps using the exact regex `^misc_step(\d+)\.pt$`, sort numerically, and for every obsolete complete step delete files matching `*_step{step:07d}.pt`.

- [ ] **Step 5: Apply the exact config values**

Set this policy in every config listed in the task:

```json
"batch_split": 1,
"i_sample": -1,
"i_save": 5000,
"max_checkpoints": 5
```

Leave `max_steps` and all model, dataset, optimizer, and loss values unchanged.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
conda run --no-capture-output -n pixal3d python -m pytest -q \
  tests/multiview/test_configs.py::test_four_configs_use_batchwide_k_and_matching_checkpoints \
  tests/multiview/test_checkpoint_retention.py \
  tests/multiview/test_train_smoke_override.py
```

Expected: all selected tests pass.

- [ ] **Step 7: Run static verification**

Run:

```bash
conda run --no-capture-output -n pixal3d python -m compileall -q \
  pixal3d/trainers/basic.py tests/multiview/test_checkpoint_retention.py
git diff --check
```

Expected: both commands exit with status 0.

- [ ] **Step 8: Commit**

```bash
git add \
  pixal3d/trainers/basic.py \
  configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json \
  configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json \
  configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  tests/multiview/test_configs.py \
  tests/multiview/test_checkpoint_retention.py
git commit -m "feat: bound multiview checkpoint retention"
```
