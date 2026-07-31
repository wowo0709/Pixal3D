# CorrAdapter-Style Multi-View Conditioning Audit

**Date:** 2026-07-31
**Pixal3D source:** `feature/multiview-correspondence-node11` at `3ff6fca5`
**CorrAdapter source:** `/root/dev/CorrAdapter` at `a269d22`
**Scope:** Inference-time correspondence-aware aggregation for the
Shape-512, Shape-1024, and PBR-1024 stages of multi-view Pixal3D

## Executive Summary

The current multi-view Pixal3D conditioner processes each calibrated view
through the unchanged single-view DINOv3/NAF projection path and then computes
an arithmetic mean of both global tokens and projected features. The mean is
performed in
`pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py::_forward_multiview`
by passing a lazy per-view generator to `_online_mean_tensor_groups`.

The mathematical intervention point for correspondence-aware routing is
immediately before that mean. A naive implementation that stacks dense
per-view volumes is not suitable for the 64-resolution stages: for `B=1`,
`K=4`, `R=64`, `C=2048`, the projected bf16 buffer alone is 4 GiB.

All target stages are sparse SLat stages, and their active coordinates are
already available to the inference pipeline. The recommended prototype
therefore adds an inference-only sparse-first aggregation path:

1. encode one view at a time;
2. immediately gather only the active Shape/PBR coordinates from its dense
   projected volume;
3. retain or stream the resulting sparse per-view features;
4. compute correspondence confidence and aggregate across views; and
5. return the unchanged `{"global": ..., "proj": SparseTensor(...)}` denoiser
   contract.

The existing `_forward_multiview` arithmetic-mean implementation remains the
default and the reference regression path.

CorrAdapter itself does not aggregate pre-denoiser DINO projection volumes.
Its released implementations add a bypass branch inside selected diffusion
self-attention blocks, construct correspondences from intermediate q/k
features, gather a local region around the matched token, and blend the
aligned value aggregation into the original attention output. The Pixal3D
prototype is therefore an application of CorrAdapter's inductive principle,
not a direct port of its architecture.

## Sources of Truth

### Pixal3D

- Multi-view design:
  `docs/superpowers/specs/2026-07-22-pixal3d-multiview-model-extension-design.md`
- Conditioner:
  `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py`
- Inference pipeline:
  `pixal3d/pipelines/pixal3d_image_to_3d.py`
- Dense projection attention:
  `pixal3d/modules/attention/proj_attention.py`
- Sparse projection attention:
  `pixal3d/modules/sparse/attention/proj_attention.py`
- Conditioner regression tests:
  `tests/multiview/test_conditioner.py`
- Online-mean tests:
  `tests/multiview/test_online_mean.py`
- Camera/projection tests:
  `tests/multiview/test_projection_geometry.py`

### CorrAdapter

- Project overview: `/root/dev/CorrAdapter/README.md`
- MVAdapter local correspondence branch:
  `/root/dev/CorrAdapter/image-conditioned/static/MVAdapter/mvadapter/models/attention_processor.py`
- SyncDreamer correspondence branch:
  `/root/dev/CorrAdapter/image-conditioned/static/SyncDreamer/ldm/modules/attention.py`
- Official paper: [Align Images Before You Generate](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Align_Images_Before_You_Generate_CVPR_2026_paper.html)

## Current Multi-View Input Contract

The calibrated multi-view conditioner receives:

| Value | Shape | Meaning |
| --- | --- | --- |
| `image` | `[B,K,3,H,W]` | ordered condition views |
| `camera_angle_x` | `[B,K]` | horizontal FOV per view |
| `distance` | `[B,K]` | camera distance per view |
| `mesh_scale` | `[B]` | anchor-aligned object scale |
| `transform_matrix` | `[B,K,4,4]` | camera-to-world transforms |

The first view is the anchor. For camera-to-world transform `T_i`, anchor
transform `T_0`, and the fixed front-view transform `F(d_0)`, the code uses:

```text
relative_0 = I
relative_i = inverse(T_0) @ T_i
projection_0 = F(d_0)
projection_i = F(d_0) @ relative_i
```

The inverse and matrix products run in FP32 with autocast disabled. This
behavior is covered by `tests/multiview/test_projection_geometry.py`.

## Current Feature Representation

`DinoV3ProjFeatureExtractor._forward_single_view` returns one global and one
spatial condition tensor.

### Global condition

DINOv3 output is split into:

- one CLS token;
- four register tokens; and
- spatial patch tokens.

The global condition is:

```text
z_global_i = concat(CLS_i, REG_i)
z_global_i: [B,5,1024]
```

### Projected low-resolution branch

Native DINOv3 patch tokens are reshaped into a 2D field and sampled at the
camera projection of the 3D grid:

```text
z_proj_lr_i: [B,R^3,1024]
```

### Projected high-resolution branch

NAF consumes the same native DINO field plus the RGB image as a guide. Its
output is sampled at the same projected 3D grid locations:

```text
z_proj_hr_i: [B,R^3,1024]
```

This is an NAF refinement of the native DINO field, not an independent
high-resolution DINO encoder.

### Within-view feature fusion

For all three target stages:

```text
z_proj_i = concat(z_proj_lr_i, z_proj_hr_i)
z_proj_i: [B,R^3,2048]
```

Low/high branch fusion is channel concatenation. It is not a mean.

## Current View Fusion

`_forward_multiview` constructs a one-shot generator over
`_forward_single_view` calls. `_online_mean_tensor_groups` consumes the
generator and returns:

```text
z_global = mean_i(z_global_i)  # [B,5,1024]
z_proj   = mean_i(z_proj_i)    # [B,R^3,2048]
```

The helper accumulates fp16/bf16 inputs in FP32 and casts the final mean back
to the input dtype. Existing tests establish:

- `K=1` equality with the single-view path;
- repeated-view equivalence;
- invariance to permutation of non-anchor views;
- numerical and gradient agreement with `torch.stack(...).mean(...)`; and
- lazy per-view iteration.

The current multi-view checkpoints are trained against this equal-mean
condition distribution. Equal mean is therefore the in-distribution baseline,
not merely a convenient implementation.

## Projection Coordinates, Depth, and Validity

`project_points_to_image_batch` returns:

```text
image_points: [B,R^3,2]
depth:        [B,R^3]
valid_mask:   [B,R^3]
```

`valid_mask` checks image bounds and positive camera depth. `ProjGrid.forward`
does not return or apply it. Instead, `sample_features` uses
`grid_sample(..., padding_mode="border")`. Invalid and out-of-frustum points
therefore receive border features in the current training and inference
distribution.

Consequences:

- a prototype must not silently introduce hard validity masking;
- validity, depth, and projected coordinates need an explicit diagnostics API
  if they are used later;
- a validity-aware experiment must be a named ablation; and
- validity is not occlusion visibility.

No rendered depth or exact surface visibility is available in the selected
mesh-free prototype.

## Denoiser Conditioning

Shape and PBR use `SparseProjectAttention`. Each block computes:

```text
global_out = CrossAttention(x, z_global)
proj_out   = Linear_2048_to_model_channels(z_proj)
output     = global_out + proj_out
```

Every block owns its own projection linear. The first and second 1024-channel
halves have different learned columns, so moving L into the H slot or H into
the L slot changes the function. The correspondence prototype must preserve:

- the `[L,H]` channel order;
- the 2048-channel width; and
- one shared per-view reliability weight for both halves in the first
  experiment.

The denoiser, ProjectAttention modules, state-dict keys, and condition shapes
do not need to change.

## Stage-Specific Paths

| Stage | Flow resolution | 3D projection grid | DINO input | NAF target | Max sparse tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shape-512 | 512 | `32^3` | 512 | 512 | 8,192 |
| Shape-1024 | 1024 | `64^3` | 1024 | 512 | 32,768 |
| PBR-1024 | 1024 | `64^3` | 1024 | 1024 | 32,768 |

The corresponding configs are:

- `configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json`
- `configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json`
- `configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json`

SS-64 is excluded from correspondence interventions.

### Inference sparse selection

`Pixal3DImageTo3DPipeline.get_proj_cond_shape` currently:

1. invokes the conditioner, which has already averaged dense per-view volumes;
2. reshapes `[B,R^3,2048]` to `[B,R,R,R,2048]`;
3. gathers the provided active coordinates; and
4. constructs a `SparseTensor`.

### Training sparse selection

`ImageConditionedProjMixin.encode_image_proj` follows the same ordering:
dense condition encoding and view mean first, sparse coordinate selection
second.

The selected prototype is inference-only. The training path remains unchanged.
The current inference pipeline uses `B=1`; the experimental sparse-first path
will reject larger batches rather than silently mishandle ragged sparse
coordinates.

For oracle-mask lookup and visualization, the experimental path must recompute
the active coordinates' `image_points`, `depth`, and `valid_mask` with
`project_points_to_image_batch` using the same scaled/rotated grid points and
the same per-view projection transform used by `ProjGrid`. This metadata is
diagnostic only in the first method. It must not change feature values or
weights unless the selected arm explicitly requests oracle-mask routing.

## Memory Audit

The following table counts only a bf16 projected feature buffer:

| Representation | Shape | Size |
| --- | --- | ---: |
| one dense Shape-512 view | `[1,32^3,2048]` | 128 MiB |
| four dense Shape-512 views | `[1,4,32^3,2048]` | 512 MiB |
| one dense 64-grid view | `[1,64^3,2048]` | 1 GiB |
| four dense 64-grid views | `[1,4,64^3,2048]` | 4 GiB |
| four sparse 8,192-token views | `[4,8192,2048]` | 128 MiB |
| four sparse 32,768-token views | `[4,32768,2048]` | 512 MiB |

These figures exclude DINO, NAF, temporary FP32 normalization/accumulation,
global tokens, the flow denoiser, and allocator fragmentation. The current
online mean also maintains an FP32 accumulator for half-precision inputs.

## Candidate Integration Points

### A. Dense conditioner-level aggregation

Modify `_forward_multiview` immediately before `_online_mean_tensor_groups`.

Advantages:

- mathematically direct;
- shared by training and inference; and
- minimal control-flow change.

Disadvantages:

- all-view confidence needs dense per-view state or a second feature pass;
- a naive `R=64`, `K=4` stack is 4 GiB before temporary tensors; and
- it changes the shared training path despite the prototype being
  inference-only.

### B. Sparse-first inference aggregation — selected

Add an explicit per-view sparse aggregation path for
`get_proj_cond_shape`. Each view is encoded sequentially, its active
coordinates are gathered immediately, and only sparse per-view features are
retained or chunked.

Advantages:

- preserves the default conditioner and training path;
- bounds retained view state by the actual SLat token count;
- keeps denoiser interfaces unchanged; and
- directly supports stage-local Shape/PBR experiments.

Disadvantages:

- the current single-view encoder still creates one dense projected volume at
  a time;
- inference code needs a deliberate experimental path rather than reusing
  ordinary `forward`; and
- transport will later need access to 2D feature maps and projection metadata.

### C. Denoiser-level CorrAdapter bypass

Add correspondence construction and aligned local aggregation inside
ProjectAttention or transformer self-attention.

Advantages:

- closer to the original CorrAdapter architecture;
- can use evolving denoising features; and
- may correct evidence at multiple flow steps.

Disadvantages:

- changes every denoising block/step;
- is substantially harder to isolate from the trained equal-mean conditioner;
- increases runtime and memory in the flow model; and
- does not answer the initial question about pre-denoiser multi-view evidence
  routing.

This approach is deferred.

## CorrAdapter Transferability

The official paper describes a bypass branch with:

1. a native correspondence constructor from diffusion intermediate features;
2. a confidence/reliability filter; and
3. an aligned-area aggregator that gathers messages only around matched
   regions.

The released MVAdapter integration:

- constructs local q/k/v projections;
- masks the same view;
- performs cross-view matching;
- chooses an argmax match after a softmax over candidate positions;
- filters by selected-match confidence;
- gathers a radius-`r` window around the match;
- aggregates values within the local window; and
- blends the local result with the original multi-view attention result.

This mechanism assumes a multi-image diffusion feature space in which
correspondences become available during generation. Pixal3D's projected
DINO/NAF features differ in three important ways:

1. they are frozen image-encoder features, not evolving flow hidden states;
2. cameras already map a nominal voxel to each image, while errors come from
   generated-view inconsistency and imperfect alignment; and
3. the current flow model receives only the already averaged condition.

The transferable idea is reliable local routing. The exact CorrAdapter module,
q/k cache schedule, row-wise search, and denoising-block placement are not
directly transferable.

## Safe Modification Boundary

The first implementation must:

- remain behind an explicit inference option;
- default to the existing equal-mean path;
- modify only Shape-512, Shape-1024, and PBR-1024 condition construction;
- preserve SS-64;
- preserve `[B,5,1024]` global output;
- preserve `[N_active,2048]` sparse projected features;
- preserve L/H channel order;
- use the same view weight for L and H;
- leave global tokens at arithmetic mean in the primary experiment;
- leave the denoiser and ProjectAttention unchanged; and
- fall back to uniform weights for `K=1`, non-finite scores, or degenerate
  normalization.

## Required Regression Gates

Before checkpoint experiments:

- existing 19 focused conditioner/mean/projection tests remain green;
- explicit equal-mean mode matches the current default;
- `K=1` matches the existing path;
- repeated views preserve the existing result;
- small-tensor sparse aggregation matches a naive reference;
- `alpha=0` is equal mean;
- identical per-view features produce uniform confidence;
- one synthetic outlier receives lower confidence;
- L/H shape, order, dtype, and device are preserved; and
- no new trainable parameter is introduced.

## Audit Conclusion

Correspondence-aware aggregation is feasible without modifying the flow model,
but the implementation should not stack dense per-view volumes. The correct
first prototype is sparse-first, projection-only, residual-to-uniform
aggregation at inference time. It should be evaluated against both the trained
equal-mean baseline and controlled oracle corruption masks before any
deformable transport or denoiser-level adapter is attempted.
