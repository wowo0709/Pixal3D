# Pixal3D Paper-Faithful Multi-View Extension Design

**Date:** 2026-07-17

**Status:** Approved

**Scope:** Extend the current TRELLIS.2/DINOv3 Pixal3D single-view implementation to calibrated, variable-view conditioning without changing the denoiser architecture or introducing a learned fusion module.

## Objective

Implement the multi-view extension described in the Pixal3D paper and its official `paper` branch while retaining the current repository's TRELLIS.2 backbone, DINOv3 encoder, NAF feature path, view-aligned latent representation, and three-stage SS/Shape/PBR cascade.

The extension accepts a batch of calibrated views, projects every view into the first view's pixel-aligned 3D frame, and computes an arithmetic mean across the view dimension. It must preserve the existing single-view behavior at `K=1` and load the existing single-view denoiser checkpoints without adding model parameters.

## Sources of Truth and Precedence

1. The current `master` implementation defines the backbone, DINOv3/NAF feature representation, projection-channel layout, denoiser interfaces, cascade, checkpoints, and training infrastructure.
2. Pixal3D paper Section 3.2.3 defines multi-view fusion as simple averaging of per-view back-projected feature volumes with known cameras.
3. The official `paper` branch implementation `DinoEncoderProjMultiView` defines the anchor-relative camera calculation and averages both projected features and global tokens.
4. The approved preprocessing design defines eight rendered conditions and two target anchors.

When the current code and the paper branch differ outside multi-view geometry and aggregation, retain the current code. In particular, do not replace DINOv3 with DINOv2 and do not replace TRELLIS.2 with Direct3D-S2.

Primary references:

- [Paper, Section 3.2.3 and implementation details](https://arxiv.org/html/2605.10922)
- [Official paper-branch conditioner](https://github.com/TencentARC/Pixal3D/blob/paper/pixal3d/models/conditional_encoders/dinov2_project_grid.py#L556-L749)
- [Released final-stage checkpoints](https://huggingface.co/TencentARC/Pixal3D/tree/main/ckpts)

## Fixed Requirements

- Runtime environment is conda environment `pixal3d` with Python 3.11, PyTorch 2.8 or newer, and CUDA 12.8.
- Inputs are calibrated: every view supplies `camera_angle_x`, camera distance, and a finite camera-to-world `transform_matrix`.
- The first condition view is the anchor and defines the generated latent coordinate frame.
- Training data supplies eight condition views; `view00` and `view01` remain the only target-latent anchors.
- One `K` is sampled per collated training batch, not per sample.
- Training samples `K` uniformly from the integers 2 through 6, matching the paper.
- Every sample in a collated batch uses the same `K`; padding and `view_mask` are forbidden in the baseline.
- Inference supports `K=1` through `K=8` with one shared `K` per inference batch.
- Local scratch and training materialization use `/root/node17/data/pixal3d`.
- Training starts from the released or locally materialized single-view denoiser weights.

## Scope

### Included

- Multi-view condition loading for SS, Shape, and PBR view-aligned datasets.
- Batch-wide random selection of 2 to 6 views with the anchor kept at index zero.
- Paper-branch anchor-relative camera transforms.
- Per-view DINOv3/NAF projection followed by arithmetic mean.
- Arithmetic mean of DINOv3 CLS and register tokens across views.
- Existing trainer condition flow and visualization compatibility for five-dimensional image batches.
- Existing cascade inference from calibrated `transforms.json` files.
- Single-view regression compatibility and multi-view unit/integration tests.
- Separate multi-view fine-tuning configs initialized from single-view checkpoints.
- Toys4K evaluation with 2, 4, and 6 input views.

### Excluded

- Learned gates, view attention, Set Transformers, pose embeddings, confidence prediction, or view selection networks.
- Visibility-weighted, alpha-weighted, depth-weighted, or occlusion-aware fusion.
- New denoiser blocks or changes to `ProjectAttention`/`SparseProjectAttention`.
- Feature caching across cascade stages.
- Camera pose estimation for multi-view inputs.
- Per-sample ragged K, padding, or a `view_mask`.
- Changing DINOv3, NAF, TRELLIS.2, the latent encoders, or the decoder architecture.
- Expanding the training distribution to K=1, K=7, or K=8 before the paper-faithful baseline is measured.
- Performance improvements before functional parity and baseline evaluation.

## Data Contract

For one asset, the multi-view dataset returns:

```text
cond              float32 [V, 3, H, W], V=8 before collation
camera_angle_x    float32 [V]
camera_distance   float32 [V]
transform_matrix  float32 [V, 4, 4], camera-to-world
mesh_scale        float32 scalar, taken from the target anchor
view_idx          integer scalar, target-anchor render index
```

The target latent parent dataset first selects `view00` or `view01`. The multi-view mixin then orders the eight conditions as:

```text
[target anchor, random permutation of the other seven views]
```

The collate function samples one integer `K` in `[2, 6]`, slices the first K entries from every view-dependent field in every sample, then delegates latent collation to the existing dataset implementation. The resulting batch contract is:

```text
cond              float32 [B, K, 3, H, W]
camera_angle_x    float32 [B, K]
camera_distance   float32 [B, K]
transform_matrix  float32 [B, K, 4, 4]
mesh_scale        float32 [B]
```

The preprocessing pack format does not store DINO features or fused volumes. It continues to store images, `transforms.json`, target-anchor latents, and scale files.

## Camera Alignment

Let `T_i` be the camera-to-world transform for view `i`, `T_0` the first/anchor view transform, `d_0` the anchor distance, and `F(d_0)` the current `ProjGrid.front_view_transform_matrix` with its camera translation set from `d_0`.

The official paper-branch calculation is retained exactly:

```python
relative_i = torch.linalg.inv(T_0) @ T_i
projection_transform_i = F(d_0) @ relative_i
```

All matrix inversion and multiplication occur in FP32 with autocast disabled. The result is cast only as required by the existing projection path.

For `K=1`, `relative_0` is identity and `projection_transform_0` equals the current fixed front-view transform. This is the basis of the single-view equivalence requirement.

## Conditioner Architecture

`DinoV3ProjFeatureExtractor` remains the only projection conditioner class. Its existing four-dimensional input behavior remains intact, and a five-dimensional branch is added:

```text
single view: image [B, 3, H, W]
multi-view: image [B, K, 3, H, W]
```

For multi-view input:

1. Compute paper-branch projection transforms relative to the first view.
2. Process each view through the unchanged single-view DINOv3/NAF path with that view's aligned projection transform. The baseline deliberately keeps this sequential path because it is the smallest natural extension of the current implementation; view batching is a later performance concern.
3. Stack projected features as `[B, K, R^3, C_proj]` and calculate `mean(dim=1)`.
4. Stack global tokens as `[B, K, T, C_global]` and calculate `mean(dim=1)`.
5. Return the existing `(z_global, z_proj)` interface.

The current `ProjGrid` computes a valid mask but samples with border padding and does not apply the mask. The multi-view baseline preserves that behavior because the paper implementation also uses unweighted averaging.

No trainable parameter is introduced. The denoiser continues to receive:

```python
{"global": z_global, "proj": z_proj}
```

with the same shapes and channel counts as single-view conditioning.

## Trainer and Checkpoint Compatibility

`ImageConditionedProjMixin`, `ImageConditionedProjFlowMatchingCFGTrainer`, and `ImageConditionedProjSparseFlowMatchingCFGTrainer` remain the trainer classes. Their conditioning path already forwards camera tensors and `transform_matrix`; only five-dimensional visualization handling is added.

Snapshots and visual diagnostics display the first/anchor condition image and first camera values. They do not concatenate K views into a new visualization model input.

Multi-view training initializes the denoiser from the current single-view checkpoint. Because the conditioner adds no parameters and its output interface is unchanged:

- Denoiser state-dict keys and tensor shapes must remain unchanged.
- Strict denoiser loading must succeed after any existing compatibility remapping.
- Optimizer state is not imported from the single-view release.
- SS, Shape, and PBR are fine-tuned independently, following the current stage boundaries.

The first baseline fine-tunes the released final-resolution single-view models:

- SS 64.
- Shape 1024.
- PBR/texture 1024.

Lower-resolution data remains available for diagnosis and future staged retraining but is not required to establish the paper-faithful multi-view baseline.

## Inference Contract

The Python pipeline accepts either one `PIL.Image` with scalar camera parameters or a sequence of images with view-shaped parameters:

```text
images             sequence length K
camera_angle_x     length K
distance           length K
transform_matrix   [K, 4, 4]
mesh_scale         one scalar for the anchor-aligned object
```

The existing single-image API remains valid and is normalized internally to `K=1`.

The CLI keeps `--image` for the existing single-view/MoGe path and adds `--transforms` for calibrated multi-view inference. `--transforms` consumes the repository's existing render `transforms.json` format and resolves frame image paths relative to that file. Multi-view inference never invokes MoGe or estimates poses.

The first listed frame is the anchor. Reordering the manifest intentionally changes the generated coordinate frame.

## Validation and Acceptance

### Unit Tests

- Camera-transform helper matches the official paper-branch formula.
- Identity relative transform gives the current fixed front-view projection.
- `ProjGrid` accepts an explicit transform without changing its default path.
- K=1 multi-view output matches the single-view output within FP32 tolerance.
- Repeating one view K times and averaging matches the single-view result.
- Permuting non-anchor views does not change the arithmetic-mean result.
- Global and projected output shapes are unchanged.
- The conditioner exposes no new trainable parameters.
- One collated batch uses one K for every sample and K is always 2 through 6.
- The anchor remains first and all selected views are unique.
- SS, Shape, and PBR collation retain their existing latent structures.

### Integration Tests

- A two-asset synthetic batch completes condition encoding for dense SS and sparse Shape/PBR paths.
- Existing single-view configs and CLI parsing remain valid.
- New multi-view configs load the same denoiser architecture and point to single-view weights.
- Calibrated K=1 and K=2 inference condition construction succeeds without model-shape changes.

### Runtime Gates

1. Finish preprocessing Task 13 hardware preflight.
2. Produce and audit the 100-asset smoke pack.
3. Load smoke samples for both target anchors and verify K=2 and K=6 projection visualizations.
4. Run one optimizer step for SS 64, Shape 1024, and PBR 1024.
5. Confirm K=1 regression before multi-view fine-tuning.
6. Fine-tune the paper-faithful baseline with K sampled from 2 through 6.
7. Evaluate Toys4K at K=2, K=4, and K=6.

## Operational Sequencing

Code implementation and unit testing may proceed while data preprocessing is idle or running on disjoint resources. Full fine-tuning does not overlap Blender rendering, voxelization, latent encoding, packing, or archival. After preprocessing has stopped at a gate, SS, Shape, and PBR training read only their audited handoffs under `/root/node17/data/pixal3d/train/stage1/active`, `stage2/active`, and `stage3/active`, respectively.

## Completion Criteria

The baseline is complete when:

- All unit and integration tests pass in conda environment `pixal3d`.
- K=1 behavior is numerically equivalent to the existing single-view conditioner.
- SS, Shape, and PBR load single-view denoiser weights without architecture changes.
- Training uses one batch-wide K sampled uniformly from 2 through 6.
- Multi-view inference accepts calibrated K=1 through K=8 inputs.
- Toys4K reports are produced separately for K=2, K=4, and K=6.
- No learned fusion, view mask, visibility weighting, or unrelated model improvement is present.
