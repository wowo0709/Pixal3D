# W&B Anchor-View Naming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ambiguous `gt_view` snapshot/media names with explicit `anchor_view` names without changing rendering behavior.

**Architecture:** Keep the existing snapshot data flow and camera selection. Rename the visualization result keys at the dataset boundary, then update `BasicTrainer.snapshot` combined-image detection and tests so local filenames and W&B keys follow the same convention.

**Tech Stack:** Python, PyTorch, pytest, Weights & Biases

## Global Constraints

- Do not emit compatibility aliases for the old names.
- Do not change rendering, model inputs, camera selection, training, or checkpoint behavior.
- Preserve `sample_gt_multiview`.
- Generated output uses `sample_anchor_view`; ground-truth output uses `sample_gt_anchor_view`.
- PBR attribute keys append `_{attr}` to those same names.

---

### Task 1: Rename Anchor-View Snapshot and W&B Keys

**Files:**
- Modify: `pixal3d/datasets/sparse_structure_latent.py`
- Modify: `pixal3d/datasets/structured_latent_shape.py`
- Modify: `pixal3d/datasets/structured_latent_svpbr.py`
- Modify: `pixal3d/trainers/basic.py`
- Modify: `tests/multiview/test_wandb_multiview.py`

**Interfaces:**
- Consumes: dataset `visualize_sample()` dictionaries and `BasicTrainer.snapshot()`.
- Produces: `sample_anchor_view`, `sample_gt_anchor_view`, `combined_anchor_views`, and their PBR attribute variants as local snapshot and W&B keys.

- [ ] **Step 1: Write failing naming tests**

Update focused tests so shape/sparse snapshots require:

```python
assert "sample_anchor_view" in image_payload
assert "sample_gt_anchor_view" in image_payload
assert "sample_gt_view" not in image_payload
assert "sample_gt_gt_view" not in image_payload
```

Add equivalent PBR assertions for:

```python
assert "sample_anchor_view_base_color" in image_payload
assert "sample_gt_anchor_view_base_color" in image_payload
assert "combined_anchor_views_base_color" in image_payload
```

- [ ] **Step 2: Verify RED**

Run:

```bash
conda run --no-capture-output -n pixal3d python -m pytest -q tests/multiview/test_wandb_multiview.py
```

Expected: failures showing the implementation still emits `gt_view` and
`combined_views` names.

- [ ] **Step 3: Implement minimal renaming**

Rename dataset visualization return keys from `gt_view` to `anchor_view`.
Update PBR `gt_view_{attr}` keys to `anchor_view_{attr}`. Update
`BasicTrainer.snapshot()` combined-image detection and output names:

```python
combo1_keys = [
    "image",
    "sample_anchor_view",
    "sample_gt_anchor_view",
]
wandb_images["samples/combined_anchor_views"] = ...
```

Use the corresponding attribute-suffixed names for PBR.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
conda run --no-capture-output -n pixal3d python -m pytest -q tests/multiview/test_wandb_multiview.py
```

Expected: all focused tests pass.

- [ ] **Step 5: Run available regression coverage**

Run:

```bash
conda run --no-capture-output -n pixal3d python -m pytest -q tests/multiview
```

Expected: full suite passes when CUDA is available. If CUDA is unavailable
during collection, record the exact failure and retain the successful focused
CPU-only result.

- [ ] **Step 6: Commit**

```bash
git add pixal3d/datasets/sparse_structure_latent.py \
  pixal3d/datasets/structured_latent_shape.py \
  pixal3d/datasets/structured_latent_svpbr.py \
  pixal3d/trainers/basic.py \
  tests/multiview/test_wandb_multiview.py
git commit -m "refactor: rename wandb anchor view media"
```
