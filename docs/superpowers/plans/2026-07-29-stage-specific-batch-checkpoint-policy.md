# Stage-Specific Batch and Checkpoint Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Shape1024 and PBR1024 default to per-GPU batch 2, split 1 and global batch 12, while retaining batch 8, split 4 and global batch 48 for SS64 and Shape512, and save all four stages every 2,000 steps.

**Architecture:** Encode the heterogeneous batch policy directly in the four production JSON configs and in one explicit stage-to-policy test mapping. Keep the Node16 measurements as historical evidence, while adding a current-default section and marking the old 1024-stage duration estimates non-authoritative for the new defaults.

**Tech Stack:** JSON training configs, Python/pytest, Markdown reports.

## Global Constraints

- All runs use six GPUs and `max_steps=20000`.
- `ss64`: `batch_size_per_gpu=8`, `batch_split=4`, global batch `48`.
- `shape512`: `batch_size_per_gpu=8`, `batch_split=4`, global batch `48`.
- `shape1024`: `batch_size_per_gpu=2`, `batch_split=1`, global batch `12`.
- `pbr1024`: `batch_size_per_gpu=2`, `batch_split=1`, global batch `12`.
- All four stages use `i_save=2000`, `i_sample=-1`, and `max_checkpoints=5`.
- Existing model, optimizer, data, input-checkpoint, and output paths remain unchanged.
- Existing batch-8/split-4 profiles remain historical evidence and are not relabelled as measurements of the new 1024-stage defaults.
- This plan does not run GPU profiles or start fine-tuning.

---

### Task 1: Encode and test the production config policy

**Files:**
- Modify: `tests/multiview/test_configs.py`
- Modify: `configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json`
- Modify: `configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json`
- Modify: `configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json`
- Modify: `configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json`

**Interfaces:**
- Consumes: the existing `CONFIGS` stage-to-path mapping in `tests/multiview/test_configs.py`.
- Produces: four production configs and a stage-specific regression contract used by the report.

- [ ] **Step 1: Write the failing policy assertions**

Add this mapping next to `OUTPUT_DIRS`:

```python
BATCH_POLICIES = {
    "ss64": (8, 4),
    "shape512": (8, 4),
    "shape1024": (2, 1),
    "pbr1024": (2, 1),
}
```

Replace the common batch assertions inside
`test_four_configs_use_batchwide_k_and_matching_checkpoints` with:

```python
expected_batch, expected_split = BATCH_POLICIES[stage]
assert trainer_args["batch_size_per_gpu"] == expected_batch
assert trainer_args["batch_split"] == expected_split
assert expected_batch * 6 == (48 if stage in {"ss64", "shape512"} else 12)
```

Change the save assertion to:

```python
assert trainer_args["i_save"] == 2000
```

- [ ] **Step 2: Run the focused test and observe RED**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/multiview/test_configs.py::test_four_configs_use_batchwide_k_and_matching_checkpoints
```

Expected: FAIL because the two 1024 configs still contain batch 8/split 4
and all four configs still contain `i_save=5000`.

- [ ] **Step 3: Apply the minimal JSON changes**

Set these exact nested values:

```text
ss64 trainer.args.batch_size_per_gpu = 8
ss64 trainer.args.batch_split = 4
ss64 trainer.args.i_save = 2000

shape512 trainer.args.batch_size_per_gpu = 8
shape512 trainer.args.batch_split = 4
shape512 trainer.args.i_save = 2000

shape1024 trainer.args.batch_size_per_gpu = 2
shape1024 trainer.args.batch_split = 1
shape1024 trainer.args.i_save = 2000

pbr1024 trainer.args.batch_size_per_gpu = 2
pbr1024 trainer.args.batch_split = 1
pbr1024 trainer.args.i_save = 2000
```

Do not modify any other JSON key or value.

- [ ] **Step 4: Verify GREEN and adjacent contracts**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/multiview/test_configs.py \
  tests/multiview/test_checkpoint_retention.py \
  tests/multiview/test_train_smoke_override.py
git diff --check
```

Expected: all runnable tests pass; checkpoint-materialization integration
cases may skip when their external files are absent.

- [ ] **Step 5: Commit**

```bash
git add \
  tests/multiview/test_configs.py \
  configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json \
  configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json \
  configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json
git commit -m "config: set stage-specific multiview batches"
```

---

### Task 2: Update the profile report for the new defaults

**Files:**
- Modify: `docs/superpowers/reports/2026-07-28-node16-multiview-profile.md`

**Interfaces:**
- Consumes: the reviewed stage-to-policy mapping from Task 1 and the retained
  Node16 batch-8/split-4 evidence already in the report.
- Produces: an unambiguous operator-facing distinction between current
  production defaults and historical profile results.

- [ ] **Step 1: Add the current production-default table**

Near the report introduction, add:

```markdown
## Current production defaults

| Stage | Batch/GPU | Split | Six-GPU global batch | Checkpoint interval |
|---|---:|---:|---:|---:|
| `ss64` | 8 | 4 | 48 | 2,000 |
| `shape512` | 8 | 4 | 48 | 2,000 |
| `shape1024` | 2 | 1 | 12 | 2,000 |
| `pbr1024` | 2 | 1 | 12 | 2,000 |

All four stages keep 20,000 optimizer steps, snapshots disabled, and only
the latest five checkpoints. The measurements below were collected before
this policy change and retain their exact measured batch labels.
```

- [ ] **Step 2: Mark the profile matrices as historical**

Rename or preface the relevant matrix sections so they explicitly say the
measurements use batch 8, split 2/4, six GPUs, and global batch 48 for the
split-4 profiles. Do not alter any recorded step time, VRAM value, error, raw
evidence path, or allocator conclusion.

- [ ] **Step 3: Qualify the 20k duration estimates**

Replace the duration-section introduction with language that states:

```markdown
The SS64 and Shape512 rows still match their current batch-8/split-4 defaults.
The Shape1024 and PBR1024 rows are historical batch-8/split-4 estimates and
are not authoritative for the new batch-2/split-1, global-batch-12 defaults.
New duration estimates for those stages require a separate profile.
```

Retain the exact historical arithmetic and retain the PBR allocator note as a
property of the measured historical run, not as an unmeasured claim about the
new default.

- [ ] **Step 4: Verify report consistency**

Run:

```bash
rg -n \
  'Current production defaults|global batch 12|historical|not authoritative|i_save|2,000' \
  docs/superpowers/reports/2026-07-28-node16-multiview-profile.md
/opt/conda/envs/pixal3d/bin/python -m pytest -q tests/multiview/test_configs.py
git diff --check
git status --short --branch
```

Expected: the new table and historical qualifications are present, config
tests pass, and only the intended report change is uncommitted.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/reports/2026-07-28-node16-multiview-profile.md
git commit -m "docs: update multiview training defaults"
```

