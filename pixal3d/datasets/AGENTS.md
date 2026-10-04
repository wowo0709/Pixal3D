# Training Datasets

## Overview

Datasets combine artifact metadata, latent arrays and condition renders. Multi-view variants select a cached anchor latent and align the ordered image/camera conditions to that anchor.

## Where to Look

| Contract | File |
|----------|------|
| Root parsing, metadata merging, skip lists | `components.py:StandardDatasetBase` |
| Matching RGBA condition image and camera | `components.py:ViewImageConditionedMixin` |
| Dense SS latent and view variants | `sparse_structure_latent.py` |
| Generic sparse latent collation | `structured_latent.py:SLat` |
| Shape flow data and visualization | `structured_latent_shape.py` |
| Texture latent plus shape condition | `structured_latent_svpbr.py` |
| VAE grid/voxel inputs | `flexi_dual_grid.py`, `sparse_voxel_pbr.py` |

## Metadata and View Pairing

- `StandardDatasetBase` accepts JSON directory mappings or legacy comma-separated roots. Supplied view configs use dictionaries keyed by artifact type.
- Dictionary mode merges artifact `metadata.csv` files by `sha256`, excluding `base` except for aesthetic scores. Base completion flags must not override artifact completion columns.
- Skip-list entries use `dataset_name/sha256`. Aesthetic filtering may be disabled by dataset name and is skipped when its column is absent.
- View classes choose `view_idx` in `[0,num_views)` and set `_current_view_idx` and `_current_latent_dir` before the condition mixin reads that same frame.
- Latents live at `<latent_root>/<sha256>/viewNN.npz`; scale sidecars are `viewNN_scale.json` with `total_scale`. Condition renders use `<render_root>/<sha256>/transforms.json` and indexed `frames`.
- Condition images must provide alpha: the mixin resizes RGBA, converts RGB to `[3,H,W]` floats, and multiplies by alpha.
- Samples carry `camera_angle_x`, `camera_distance` from transform translation norm, and `mesh_scale` from the selected latent sidecar.
- Single-view latents are already in their anchor frame. Multi-view conditions must include matching `transform_matrix [B,K,4,4]` to project the other views into that frame; do not apply the old single-view omission rule to MV inputs.
- SS metadata filtering checks existing per-view columns, preferring `view_scale` naming. It does not prove all configured view columns exist; check actual artifacts when a randomly selected view fails to load.

## Tensor and Collation Contracts

- SS `.npz` uses key `z`; collation produces dense `x_0` suitable for the sparse-structure flow model.
- Shape/generic sparse `.npz` uses integer `coords [N,3]` and float `feats [N,C]`. Channel normalization is applied during loading and reversed by decoding/visualization.
- Sparse collation prefixes a batch index, builds `SparseTensor` as `x_0`, and preserves per-instance layout slices. With `split_size`, it returns a list of packs formed by token-balanced grouping.
- PBR view loading reads shape and texture latents for the same view and asserts exact coordinate equality. Collation returns texture `x_0` and shape `concat_cond` with identical sparse coordinates.
- Snapshot helpers lazily load pretrained decoders onto CUDA. Inspect decoder paths and normalization alongside the dataset class when checking training snapshots.

## Multi-view Fine-tuning

- `MultiViewImageConditionedMixin` reads eight prepared condition views. Cached
  target anchors are view00/view01, matching latent and scale sidecars. The anchor
  is first in `view_indices`; images, FOV, distance and transforms use that order.
- The new recipe samples one batch-wide K uniformly from 1 through 6. Original
  collaborator configs retain their 2-through-6 default. Collation must preserve
  one K across every accumulation pack.
- Dense conditions are `[B,K,3,H,W]`; cameras have the same B/K axes and transforms
  add `[4,4]`. Sparse collation prefixes latent coordinates with the batch index.
- `batch_split` counts accumulation packs. Sparse token balancing can produce
  unequal asset counts per pack; retain the native mean-over-packs objective.
- Mixed-source preparation uses the common eligible asset set for all four
  stages, checks both cached anchors, and rejects high-token or misaligned
  Shape1024/PBR1024 coordinates in metadata. Train/validation SHA sets are disjoint.
