# Pixal3D Multi-View Model Extension Design

**Date:** 2026-07-22

**Status:** Approved

**Scope:** Extend the current TRELLIS.2/DINOv3 Pixal3D single-view inference
cascade to calibrated multi-view conditioning and fine-tune every flow model
that participates in the released cascade.

**Supersedes:** The model-scope and checkpoint sections of
`2026-07-17-pixal3d-paper-faithful-multiview-design.md`. In particular, the
older three-checkpoint baseline omitted the inference-critical Shape-512 flow
model. This design requires four independently fine-tuned checkpoints.

## Objective

Extend Pixal3D from one calibrated condition image to a variable number of
calibrated condition images without adding a learned fusion mechanism or
changing the denoiser architecture. Preserve the current TRELLIS.2 backbone,
DINOv3 and NAF feature paths, projection channel layout, latent encoders and
decoders, samplers, and three-stage SS/Shape/PBR cascade.

The extension must:

- place the target latent's view first and use it as the anchor coordinate
  frame;
- back-project every selected view into that anchor frame;
- fuse views by an arithmetic mean of projected features and global tokens;
- reproduce the current single-view result at `K=1`, checked in FP32 with
  `torch.testing.assert_close(rtol=1e-5, atol=1e-5)`;
- fine-tune SS-64, Shape-512, Shape-1024, and PBR-1024 from their corresponding
  released single-view weights;
- complete model validation before W&B integration, and complete W&B smoke
  validation before any long-running training.

The objective is a natural multi-view extension, not a performance-improvement
project. Learned fusion, visibility weighting, architecture additions, and
training curriculum expansion are excluded until this baseline is measured.

## Sources of Truth and Precedence

1. The current repository at base commit `fc26830338348e17929d7a734a4015ed4dca7bdd`
   defines the TRELLIS.2 backbone, DINOv3/NAF feature representation, denoiser
   interfaces, inference cascade, training infrastructure, and preprocessing
   handoff.
2. Pixal3D paper Section 3.2.3 defines multi-view fusion as an arithmetic mean
   of per-view back-projected feature volumes with known cameras.
3. The official `paper` branch implementation defines anchor-relative camera
   transforms and averages projected and global features across views.
4. The approved preprocessing contract defines eight condition views and two
   view-aligned target anchors, `view00` and `view01`.
5. The released pipeline configuration defines the inference-critical flow
   models: SS-64, Shape-512, Shape-1024, and PBR-1024.

When the current implementation and the paper branch differ outside
multi-view geometry and averaging, retain the current implementation. Do not
replace DINOv3 with DINOv2 or TRELLIS.2 with Direct3D-S2.

Primary references:

- [Pixal3D paper](https://arxiv.org/html/2605.10922)
- [Official paper-branch conditioner](https://github.com/TencentARC/Pixal3D/blob/paper/pixal3d/models/conditional_encoders/dinov2_project_grid.py)
- [Released pipeline configuration](https://huggingface.co/TencentARC/Pixal3D/blob/main/pipeline.json)
- [Released checkpoints](https://huggingface.co/TencentARC/Pixal3D/tree/main/ckpts)

## Current State

### Repository and model

- The isolated implementation worktree is
  `/root/dev/Pixal3D/.worktrees/multiview-model-extension` on branch
  `feature/multiview-model-extension`.
- The baseline suite passes: `774 passed` in the `pixal3d` conda environment.
- The current conditioner accepts only `[B, 3, H, W]` image tensors.
- `ProjGrid` rejects an explicit transform and constructs only the fixed
  front-view transform.
- `ViewImageConditionedMixin` returns one condition image associated with the
  selected target anchor and intentionally omits `transform_matrix`.
- The denoiser already consumes the stable condition interface
  `{"global": z_global, "proj": z_proj}`. This interface does not need to
  change.
- The released `1024_cascade` actually invokes SS-64, Shape-512, Shape-1024,
  and PBR-1024. Shape-512 generates the low-resolution structured latent and
  the coordinates used by Shape-1024.

### Preprocessing and development data

The 2026-07-22 observation found both node16 and node17 healthy and completing
ABO publication batches. Smoke and pilot reports were passed. Production was
still active, and the observed completed production `batch000` manifests had
`completed_count: 0`, so those packs are not accepted as development inputs.

The immutable qualification pilot packs contain 64 validated ABO assets with
eight condition images, camera transforms, and both target anchors. Model
implementation and validation use only these packs:

```text
/root/data2/pixal3d/prepared/qualification/pilot/
├── common/ABO/ABO-00000/batch000.tar
├── ss/64/ABO/ABO-00000/batch000.tar
├── shape/512/ABO/ABO-00000/batch000.tar
├── shape/1024/ABO/ABO-00000/batch000.tar
└── pbr/1024/ABO/ABO-00000/batch000.tar
```

They are materialized into isolated stage roots under:

```text
/root/node17/data/pixal3d/train/development/abo-pilot64/
├── ss64/active/          # common + SS-64
├── shape512/active/      # common + shape-512
├── shape1024/active/     # common + shape-1024
└── pbr1024/active/       # common + shape-1024 + PBR-1024
```

The exact latent directories are:

```text
ss_latents/ss_enc_conv3d_16l8_fp16_64_view
shape_latents/shape_enc_next_dc_f16c32_fp16_512_view
shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view
pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix
```

No test or development command reads an active preprocessing scratch tree.

## Fixed Requirements

- Runtime environment is conda environment `pixal3d` with Python 3.11,
  PyTorch 2.8 or newer, and CUDA 12.8.
- Every input view supplies a finite `camera_angle_x`, camera distance, and
  camera-to-world `transform_matrix`.
- The first condition view is always the target latent anchor and defines the
  generated latent coordinate frame.
- Training assets expose eight conditions; only `view00` and `view01` have
  target latents.
- One `K` is sampled for the entire collated batch, uniformly from integer
  values 2 through 6.
- Padding, ragged per-sample K, and `view_mask` are forbidden in the baseline.
- Calibrated inference accepts one shared K per batch for K=1 through K=8.
- Multi-view fusion is a plain arithmetic mean for projected features and
  global tokens.
- The extension adds no trainable parameter and changes no denoiser state-dict
  key or tensor shape.
- SS-64, Shape-512, Shape-1024, and PBR-1024 are independently initialized
  from their matching released single-view checkpoints.
- SS-32, Shape-256, PBR-256, and PBR-512 are not trained in this baseline.
- Model implementation, validation, W&B integration, and training occur in
  that order. A failed gate blocks every later gate.

## Approaches Considered

### Selected: independently fine-tune the four inference checkpoints

Load each released inference checkpoint into its unchanged denoiser and
fine-tune it with the shared multi-view conditioner and data contract. This is
the smallest complete extension of the released cascade, preserves every
resolution-specialized single-view checkpoint, and permits failures to be
isolated by stage.

### Rejected: retrain the full progressive curriculum

Running SS 32 to 64 and Shape/PBR 256 to 512 to 1024 is required when
reproducing the models from scratch, but the 256 models are not invoked by the
released inference pipeline. This adds substantial compute and confounds the
multi-view extension with curriculum retraining.

### Rejected: initialize Shape-1024 from a newly fine-tuned Shape-512

This would discard the released Shape-1024 specialization and entangle the
two experiments. Shape-512 and Shape-1024 instead start from their matching
released weights and are validated together in an end-to-end cascade smoke.

## Data Contract

For one asset before collation, the multi-view mixin returns:

```text
cond              float32 [8, 3, H, W]
camera_angle_x    float32 [8]
camera_distance   float32 [8]
transform_matrix  float32 [8, 4, 4], camera-to-world
mesh_scale        float32 scalar for the target anchor
view_idx          integer scalar, 0 or 1
view_indices      int64 [8], anchor first and all values unique
```

The target dataset first selects `view00` or `view01`. The condition mixin then
orders views as:

```text
[target anchor, random permutation of the other seven rendered views]
```

The collate helper samples one K in `[2, 6]`, slices every view-shaped field to
that K for every sample, and delegates target latent collation to the existing
dense or sparse dataset implementation. The batch contract is:

```text
cond              float32 [B, K, 3, H, W]
camera_angle_x    float32 [B, K]
camera_distance   float32 [B, K]
transform_matrix  float32 [B, K, 4, 4]
mesh_scale        float32 [B]
view_indices      int64 [B, K]
```

SS, Shape, and PBR keep their existing `x_0`, `concat_cond`, coordinate,
feature, and load-balancing representations. The preprocessing pack format is
unchanged and never stores DINO features or fused volumes.

## Camera Alignment

Let `T_i` be the camera-to-world matrix of selected view i, `T_0` the anchor
matrix, `d_0` the anchor distance, and `F(d_0)` the current
`ProjGrid.front_view_transform_matrix` with its camera translation set from
`d_0`.

The projection transform is:

```python
relative_i = torch.linalg.inv(T_0) @ T_i
projection_transform_i = F(d_0) @ relative_i
```

Inversion and matrix multiplication run in FP32 with autocast disabled.
Inputs must be finite, and the anchor matrix must be invertible. The result is
cast only where the current projection path requires it.

For K=1, `relative_0` is identity and `projection_transform_0` is exactly the
current fixed front-view projection. This defines the single-view regression
gate.

## Conditioner Architecture

`DinoV3ProjFeatureExtractor` remains the single conditioner implementation.
Its four-dimensional path remains intact, and it adds one five-dimensional
path:

```text
single view: image [B, 3, H, W]
multi-view:  image [B, K, 3, H, W]
```

The five-dimensional path performs the following operations:

1. Validate all image and camera batch shapes.
2. Compute anchor-relative projection transforms in FP32.
3. Run each selected view through the unchanged DINOv3/NAF and projection
   path using that view's aligned transform.
4. Stack projected features as `[B, K, R^3, C_proj]` and apply `mean(dim=1)`.
5. Stack DINOv3 CLS and register tokens as `[B, K, T, C_global]` and apply
   `mean(dim=1)`.
6. Return the existing `(z_global, z_proj)` interface.

Views are processed sequentially in the first baseline. Flattening B and K,
feature caching, or parallel per-view encoding is a later performance concern.
The existing projection valid-mask and border-padding behavior remains
unchanged; no visibility or alpha weighting is introduced.

## Dataset and Trainer Boundaries

A new anchor-first multi-view condition mixin is shared by these dataset
classes:

- `MultiViewImageConditionedSparseStructureLatentView`
- `MultiViewImageConditionedSLatShapeView`
- `MultiViewImageConditionedSLatPbrView`

The Shape dataset class is shared by the Shape-512 and Shape-1024 configs.
Existing single-view dataset classes and configs remain valid.

The current projection trainer classes remain unchanged in role. They receive
five-dimensional conditions through the existing `get_cond` path. Snapshot
helpers use the first condition image and first camera values when a
single-image visualization is required; no camera tensor is sliced before
condition encoding.

## Checkpoints and Fine-Tuning Configurations

The four source weights are:

```text
ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors
ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors
ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors
ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors
```

They are materialized without key rewriting under:

```text
/root/node17/data/pixal3d/train/checkpoints/single_view/
```

The baseline adds four configuration variants:

```text
configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json
configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json
configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json
configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json
```

Each variant changes only the dataset class, multi-view K arguments, effective
batch sizing needed for memory, source checkpoint path, and logging intervals
needed for the smoke gate. Model and trainer definitions remain identical to
their single-view source config.

## Inference Contract

The existing single-image CLI and MoGe camera-estimation path remain valid.
Calibrated multi-view inference adds a `--transforms` input that consumes the
existing `transforms.json` render format and resolves frame image paths relative
to the manifest.

The Python pipeline accepts:

```text
images             sequence length K
camera_angle_x     float32 [K]
distance           float32 [K]
transform_matrix   float32 [K, 4, 4]
mesh_scale         one anchor scalar
```

The first frame is the anchor. The same ordered view set is propagated through
SS-64, Shape-512, Shape-1024, and PBR-1024. Reordering the first frame
intentionally changes the generated coordinate frame. Multi-view inference
never estimates missing camera poses.

## Error Handling

The following contract violations fail explicitly and never fall back to a
single view:

- a development-training manifest has other than exactly eight frames, or a
  calibrated inference manifest has fewer than one or more than eight frames;
- any selected image, FOV, distance, transform, target latent, or scale file is
  missing;
- the anchor target view and condition index zero differ;
- an image or camera field has an unexpected shape;
- a camera value is non-finite or an anchor transform is singular;
- selected view indices contain duplicates;
- samples inside a collated batch do not share K;
- K is outside 2 through 6 in training or 1 through 8 in inference;
- a released checkpoint has an unexpected or shape-incompatible denoiser key.

Dataset errors identify source, asset SHA, anchor, and failing view. The
baseline does not add a new fallback/retry policy to the model datasets.

## Validation Gates

Validation completes before any W&B change is enabled.

### Gate 1: mathematical and single-view regression

- Anchor-relative matrices match the official paper formula.
- An identity relative transform reproduces the fixed front-view transform.
- The explicit-transform and default `ProjGrid` paths match for the anchor.
- K=1 conditioner outputs match the existing single-view outputs in FP32 under
  `torch.testing.assert_close(rtol=1e-5, atol=1e-5)`.
- Repeating one view K times and averaging reproduces the single-view output.
- Permuting non-anchor views does not change the mean.
- Global and projected output shapes are unchanged.
- No trainable parameter or denoiser state-dict key is added.

### Gate 2: dataset and pilot integration

- The ABO pilot64 materialization validates against pack manifests before use.
- SS-64, Shape-512, Shape-1024, and PBR-1024 each load both target anchors.
- Every anchor is condition index zero and all selected indices are unique.
- One batch-wide K is observed for K=2 and K=6.
- Dense SS and sparse Shape/PBR collation retain their existing structures.

### Gate 3: checkpoint and GPU function

- All four released denoisers load with exact parameter tensor shapes,
  `unexpected_keys == []`, and no missing key other than the known
  non-parameter buffer `rope_phases`.
- Every model completes K=2 and K=6 condition encoding.
- Every model completes one forward, backward, and optimizer step with finite
  loss, gradients, and updated parameters.
- A saved smoke checkpoint reloads successfully.
- Calibrated K=1 and K=2 pipeline construction succeeds through the complete
  four-model cascade without a condition shape change.

While preprocessing is active, GPU validation may use only GPU0 after a fresh
resource check. Otherwise it waits until the workers are drained. Validation
must not reduce preprocessing capacity implicitly.

## W&B Gate

W&B uses the existing rank-zero initialization, scalar logger, sample writer,
and config artifact flow. The baseline project name is
`pixal3d-multiview`; authentication and entity come from the operator's W&B
environment.

The four run names are:

```text
mv-baseline-ss64
mv-baseline-shape512
mv-baseline-shape1024
mv-baseline-pbr1024
```

Existing loss, learning rate, gradient, checkpoint, config, generated sample,
and ground-truth logging remains. Multi-view support adds:

- selected K under `multiview/k`;
- an ordered input-view grid with the anchor visibly marked under
  `samples/input_views`;
- the target anchor image under `samples/anchor`;
- stage name, dataset name, and asset SHA in captions;
- generated and ground-truth views through the existing sample logger.

First run W&B in offline mode and verify config and image serialization. Then
run one online optimizer step per stage. No long training starts until all four
runs show the correct input ordering, stage identity, finite scalar values, and
sample images.

## Training Gate and Order

Training proceeds sequentially:

1. SS-64
2. Shape-512
3. Shape-1024
4. PBR-1024

Each stage first runs an exactly 10-optimizer-step pilot64 smoke. It logs
scalars on every step, writes generated samples at steps 5 and 10, and saves
and reloads a checkpoint at step 10. The smoke must demonstrate:

- K values sampled only from 2 through 6;
- finite losses and gradients;
- checkpoint save and reload;
- correct W&B anchor and input-view ordering;
- preserved K=1 regression;
- K=2, K=4, and K=6 inference smoke for the completed cascade portion.

Each denoiser starts from its matching released single-view checkpoint, not
from another newly fine-tuned resolution. A stage that fails its smoke blocks
the next stage.

Long-running training starts only after preprocessing workers on node16 and
node17 are drained, no Blender/voxel/latent/packing task is active, the selected
training packs pass audit, and W&B has passed. Full training swaps only
`data_dir` from the pilot64 roots to audited stage handoffs; model code and
configuration semantics do not change.

## File Boundaries

- `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py`:
  projection math, explicit transforms, 4D/5D conditioner dispatch, arithmetic
  averaging, and anchor visualization helper.
- `pixal3d/datasets/components.py`: anchor-first image/camera loading and
  batch-wide K slicing.
- `pixal3d/datasets/sparse_structure_latent.py`: multi-view SS dataset class.
- `pixal3d/datasets/structured_latent_shape.py`: shared multi-view Shape-512
  and Shape-1024 dataset class.
- `pixal3d/datasets/structured_latent_svpbr.py`: multi-view PBR-1024 dataset
  class.
- `pixal3d/datasets/__init__.py`: lazy registrations.
- `pixal3d/trainers/flow_matching/flow_matching.py` and
  `sparse_flow_matching.py`: anchor-only snapshot compatibility.
- `pixal3d/trainers/basic.py`: ordered input-view W&B visualization and K
  metadata.
- `pixal3d/pipelines/pixal3d_image_to_3d.py`: shared calibrated view set across
  all four inference flow models.
- `inference.py`: `transforms.json` CLI input while retaining `--image`.
- `configs/gen/*_proj_multiview_*.json`: four fine-tuning configurations.
- `tests/multiview/`: geometry, conditioner, dataset, checkpoint, pipeline,
  CLI, trainer snapshot, W&B serialization, and pilot integration tests.
- `README.md`: multi-view data, validation, W&B, and training commands.

## Excluded Work

- learned gates, view attention, Set Transformers, pose embeddings, confidence
  prediction, or view-selection networks;
- visibility-, alpha-, depth-, or occlusion-weighted fusion;
- new denoiser or projection-attention blocks;
- per-sample ragged K, padding, or masks;
- DINO/NAF feature caching or batched-view performance optimization;
- camera pose estimation for multi-view inputs;
- SS-32, Shape-256, PBR-256, or PBR-512 multi-view training;
- hyperparameter search, loss changes, new regularizers, or architectural
  performance improvements;
- overlapping long model training with preprocessing.

## Completion Criteria

The baseline is complete only when:

- the full repository test suite and all new multi-view tests pass;
- K=1 matches the existing single-view conditioner in FP32 under
  `torch.testing.assert_close(rtol=1e-5, atol=1e-5)`;
- SS-64, Shape-512, Shape-1024, and PBR-1024 strictly load their released
  single-view denoisers without architecture changes;
- training uses one batch-wide K sampled uniformly from 2 through 6;
- calibrated inference accepts K=1 through K=8 and routes the same view set
  through all four flow models;
- the audited ABO pilot64 data completes dataset, optimizer-step, checkpoint,
  and inference smoke gates;
- the four W&B smoke runs show correct anchors, views, metadata, finite metrics,
  and samples;
- no excluded fusion, masking, weighting, curriculum, or performance feature is
  present.
