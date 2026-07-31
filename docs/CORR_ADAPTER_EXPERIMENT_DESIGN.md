# Multi-View Pixal3D Correspondence-Aware Conditioning Design

**Date:** 2026-07-31
**Status:** Stage 1 prototype implemented and CPU-regression verified;
checkpoint-dependent 3D experiment pending
**Branch:** `feature/multiview-correspondence-node11`
**Depends on:** `docs/CORR_ADAPTER_AUDIT.md`

## Objective

Determine whether correspondence-aware view routing is a useful inductive bias
for multi-view Pixal3D when independently generated geometry-conditioned input
views contain inconsistent or hallucinated evidence.

The immediate causal question is:

> Can voxel-wise multi-view consensus suppress inconsistent projected evidence
> while preserving valid and view-unique content better than the multi-view
> model's trained arithmetic mean?

This design does not claim to solve multi-view inconsistency. It defines a
small inference-time prototype and the controls needed to decide whether the
research direction has measurable headroom.

## Research Context

Each VLM-deformed condition view is generated independently from its
corresponding calibrated source view. The output is a newly synthesized image,
not a deterministic warp. Consequently:

- camera/view identity is inherited approximately from the source;
- exact source-to-generated pixel flow is unavailable;
- parts may be inserted, deleted, or moved;
- local patterns may be hallucinated differently in each view; and
- same-voxel camera projection does not guarantee semantic correspondence.

The available foreground mask may be used for controlled corruption, scoring,
and visualization. Hard-zero foreground/background feature masking is not part
of the primary method because it changes the condition distribution learned by
Pixal3D.

The anatomical last mesh is not used. It is an approximate foot/hand/head
proxy that requires image-to-mesh pose alignment and cannot provide exact
visibility or correspondence ground truth.

## Relationship to CorrAdapter

CorrAdapter adds a bypass branch inside multi-image diffusion transformer
blocks. It constructs correspondences from intermediate diffusion features and
aggregates messages around matched regions. The present method adopts its
reliable-routing principle but acts earlier, on calibrated DINOv3/NAF evidence
before the Pixal3D flow model.

This distinction must appear in every report:

- **CorrAdapter:** denoising-time native correspondence and aligned-area
  attention aggregation.
- **This prototype:** pre-denoiser voxel-wise confidence routing of
  camera-projected image features.

Calling the prototype a direct CorrAdapter port would be inaccurate.

## Prior Projection-Conditioning Evidence

A preceding inference-time causal ablation used the public single-view
Pixal3D checkpoint, six images, and seed 42. It is recorded on branch
`codex/projection-feature-ablation` at commit `9f898cca`, in
`docs/experiments/2026-07-28-projection-conditioning-causal-ablation.md`.
It is not evidence from the new multi-view checkpoints, but it motivates the
control structure of this experiment.

The ablation found:

- native DINO L and NAF-refined H features had stage-level mean cosine
  similarity of approximately `0.9955–0.9965`;
- Shape-512 and Shape-1024 projection linear low/high weight-norm ratios were
  approximately `6.25` and `54.19`, while PBR-1024 was `1.20`;
- low-slot activation accounted for `80.9%` and `93.6%` in the two shape
  stages but `50.2%` in PBR;
- `[L,0]` remained closer to `[L,H]` than `[0,H]`, largely because the trained
  shape checkpoints route the first slot more strongly; and
- projection-only generation preserved much more input-specific signal than
  global-only generation under that deliberately out-of-distribution
  intervention.

These observations do not establish how the multi-view checkpoints will
behave. They do justify three conservative choices:

1. preserve the learned `[L,H]` order and width;
2. use one view reliability for both halves initially; and
3. isolate projection routing before modifying global-token fusion.

## Scope Decomposition

The research program contains two separate implementation phases.

### Stage 1 — current implementation scope

- sparse-first per-view feature access;
- same-voxel multi-view consensus confidence;
- residual confidence-aware aggregation;
- controlled corruption and oracle masks;
- feature-level and stage-local 3D evaluation; and
- projection-only routing with global tokens held at equal mean.

### Stage 2 — gated future scope

- deformable back-projection;
- per-view 2D sampling offsets;
- 3D-neighbor offset continuity;
- controlled oracle warp offsets; and
- confidence-residual transport.

Stage 2 receives a separate implementation plan only if Stage 1 demonstrates
oracle headroom and a non-trivial relationship between confidence and
corruption. Mesh proxy utilization is removed from both phases.

## Target Models

Use only stage-matched, multi-view-fine-tuned checkpoints from the
`feature/multiview-model-extension` model family.

| Stage | Intervention | Upstream state held fixed |
| --- | --- | --- |
| SS-64 | none | produces or supplies fixed sparse coordinates |
| Shape-512 | Stage 1 routing | identical SS coordinates |
| Shape-1024 | Stage 1 routing | identical SS coordinates and Shape-512 input |
| PBR-1024 | Stage 1 routing | identical Shape-1024 SLat |

The checkpoints are still training and will be supplied later. Public
single-view Pixal3D weights must not substitute for them.

Initial inference settings:

- `K=4`;
- one seed, fixed to `42` unless the checkpoint handoff specifies otherwise;
- identical cameras, noise, CFG, samplers, upstream latents, and postprocessing
  across paired arms; and
- stage-local experiments before a cumulative cascade.

## Selected Architecture

### Default path

Ordinary inference continues to call the existing multi-view conditioner and
uses arithmetic mean for global and projected conditions.

### Experimental sparse-first path

For Shape/PBR stages with active coordinates `Q`:

1. compute anchor-relative camera transforms exactly as the default path;
2. encode each view with the unchanged single-view DINOv3/NAF path;
3. reshape that view's dense `[B,R^3,2048]` projection;
4. gather only `Q`, producing `[N_active,2048]`;
5. retain all per-view sparse features in one `[K,N_active,2048]` tensor;
6. aggregate views with the selected strategy;
7. arithmetic-mean the five global tokens for the primary experiment; and
8. return the existing sparse condition dictionary.

The flow model and ProjectAttention remain unchanged.

The first implementation supports pipeline inference with `B=1`. It raises a
clear error for a larger batch. General ragged sparse batching is outside this
prototype.

### Implemented interfaces and defaults

The implemented code paths are:

- `ProjectionAggregationConfig`;
- `ProjectionAggregationDiagnostics`;
- `aggregate_projected_features`;
- `aggregate_global_features`;
- `DinoV3ProjFeatureExtractor.iter_view_features`;
- `ProjGrid.project_grid_points`;
- `Pixal3DImageTo3DPipeline.get_proj_cond_shape`; and
- `inference.py::build_projection_aggregation_configs`.

`ProjectionAggregationConfig` defaults to:

```text
mode="mean"
alpha=0.5
temperature=0.1
chunk_size=4096
global_mode="mean"
```

CLI defaults are:

```text
--projection_aggregation  omitted / None
--projection_stages       shape512,shape1024,pbr1024
--projection_alpha        0.5
--projection_temperature  0.1
```

The CLI aggregation choices are `mean` and `consensus`. Oracle routing
requires programmatic `mode="oracle"` plus masks, and projection-weighted
global fusion requires programmatic `global_mode="projection_weights"`.
Omitting `--projection_aggregation` leaves `projection_aggregation=None` and
uses the original dense/default conditioner path. Explicit `mean` selects the
sparse-first path but returns an FP32-accumulated arithmetic mean cast back to
the source dtype.

This inference prototype supports only `pipeline_type="1024_cascade"`.
Enabling aggregation without an explicit resolution selects 1024. An explicit
1536 aggregation request is rejected, while omission of aggregation preserves
the ordinary defaults (1024 with low-VRAM mode, otherwise 1536). R=96/1536 is
intentionally unsupported and is outside the reported memory budget; enabling
it requires a later measured memory study.

Programmatic oracle masks must have shape `[K,1,H,W]`, use the same view order,
and share the already-preprocessed coordinate frame of the supplied images.
The pipeline only resizes masks to conditioner resolution. It rejects
`oracle_masks` with `preprocess_image=True`; callers must provide aligned
image/mask pairs with `preprocess_image=False`.

The B=1 limitation is enforced from sparse coordinates:
`_flat_sparse_indices` rejects any nonzero coordinate batch index. The path
still creates one transient dense `[1,R^3,2048]` projection per view and
retains a `[K,N_active,2048]` source-dtype tensor. `chunk_size` bounds only
FP32 scoring and fusion temporaries, not that retained tensor. Retained sparse
bytes scale as `K * N_active * C * bytes_per_element`. For K=4, C=2048, and
bf16, this is 128 MiB at 8,192 tokens, 512 MiB at 32,768, and 768 MiB at the
configured 49,152-token threshold. The threshold is not a hard ceiling: at
the forced 1024 fallback the count can remain higher, up to the R=64 lattice
bound of 262,144 tokens and 4 GiB of retained sparse state. Separately, the
one-view-at-a-time transient dense R=64 projection is 1 GiB. All figures
exclude diagnostics, model state, temporary FP32 buffers, and allocator
overhead. They apply to the 1024 cascade, not the unsupported R=96/1536 path.

## Feature and Weight Definitions

For active voxel `q` and view `i`:

```text
F_i(q) = [L_i(q), H_i(q)]
L_i(q), H_i(q) in R^1024
F_i(q) in R^2048
```

The first experiment derives one reliability value per view and voxel, then
applies it to the whole concatenated feature. It does not route L and H
independently.

### Leave-one-out consensus

Normalize L and H separately to avoid branch norm dominating the similarity:

```text
l_i(q) = normalize(L_i(q))
h_i(q) = normalize(H_i(q))
```

For `K > 1`, form leave-one-out prototypes:

```text
mu_L,-i(q) = normalize(sum_{j != i} l_j(q))
mu_H,-i(q) = normalize(sum_{j != i} h_j(q))
```

The initial confidence logit is:

```text
s_i(q) =
0.5 cos(l_i(q), mu_L,-i(q))
+
0.5 cos(h_i(q), mu_H,-i(q))
```

The coefficient is fixed at `0.5/0.5` in the first prototype. It is not tuned
from final 3D results.

### Residual-to-uniform routing

```text
r_i(q) = softmax_i(s_i(q) / tau)
w_i(q) = (1 - alpha) / K + alpha r_i(q)
F_new(q) = sum_i w_i(q) F_i(q)
```

Properties:

- `sum_i w_i(q) = 1`;
- `alpha=0` exactly recovers arithmetic mean;
- feature magnitude remains a convex combination of observed view features;
- `alpha=1` is a stress condition, not the primary method; and
- invalid or non-finite normalization falls back to `1/K`.

The initial feature-only diagnostic may compare a small declared set of
`tau`/`alpha` values. One pair must be frozen before 3D generation. Final
renders must not be used for repeated hyperparameter selection.

## Global-Condition Policy

Primary Stage 1 experiments keep:

```text
z_global = mean_i(z_global_i)
```

Only projected features are routed. This isolates the research question and
retains the global condition distribution used for multi-view fine-tuning.

After the primary result, one diagnostic may convert voxel confidence into a
view scalar and use it to weight corresponding CLS/register tokens. This is a
separate arm, not part of the initial method. Per-view global tokens are never
concatenated into a new token sequence.

The diagnostic is implemented as
`global_mode="projection_weights"` by averaging voxel weights into one scalar
per view. It remains programmatic and is not the CLI/default primary path.

## Stage 1 Comparison Arms

| Arm | Projection fusion | Global fusion | Purpose |
| --- | --- | --- | --- |
| S0 | CLI omitted, or explicit `mode="mean"` | `global_mode="mean"` | trained baseline and explicit-mean regression control |
| S1 | `mode="consensus", alpha=0.5, temperature=0.1` | `global_mode="mean"` | primary method |
| S2 | `mode="consensus", alpha=1.0` | `global_mode="mean"` | distribution-shift stress test |
| S3 | `mode="oracle", alpha=0.5` plus masks | `global_mode="mean"` | realistic oracle headroom |
| S4 | `mode="oracle", alpha=1.0` plus masks | `global_mode="mean"` | controlled upper bound |
| S5 | residual consensus | `global_mode="projection_weights"` | global leakage diagnostic |

The minimal first 3D gate is S0, S1, and S3. S2, S4, and S5 are run only after
the basic pipeline and artifact checks pass.

## Controlled Corruption

Oracle masks are generated by the experiment tooling, not supplied by the
user and not manually inferred on real VLM outputs.

For clean image `I`, mask `M`, and deterministic corruption `C`:

```text
I_corrupt = (1 - M) I + M C(I)
```

The prototype creates:

1. **local material/color change** — deterministic hue, saturation, or
   brightness modification inside a foreground patch;
2. **hallucinated pattern** — procedural or donor logo/sole/decal composite;
3. **local deletion** — a selected pattern or part replaced by a deterministic
   background-like/inpainted patch;
4. **local affine warp** — translation, rotation, scale, or shear with a saved
   transform; and
5. **smooth local warp** — a bounded, smoothed displacement field with saved
   forward/inverse coordinate maps.

Every generated sample stores:

- source image hash;
- corruption seed and parameters;
- exact binary or soft mask;
- forward transform/displacement field when applicable;
- inverse sampling grid;
- out-of-bounds/hole mask; and
- a contact-sheet preview.

The first controlled experiment corrupts one non-anchor view among `K=4`.
Anchor corruption is a named stress case. Real VLM images have neither oracle
mask nor oracle correspondence.

## Oracle Mask Routing

For the projected coordinate `p_i(q)`:

```text
m_i(q) = bilinear_sample(M_i, p_i(q))
a_i(q) = 1 - m_i(q)
```

Pure oracle weights normalize `a_i(q)` across views. If all reliabilities are
zero, weights fall back to uniform rather than producing an empty condition.
Residual oracle routing mixes the normalized oracle weights with `1/K` using
the same alpha convention as S1.

The projected mask coordinate is computed from the same rotated/scaled
`ProjGrid.grid_points` and anchor-relative projection matrix as feature
sampling. The method does not infer it from rendered output or a mesh proxy.

Oracle routing answers only:

> How much improvement is possible if the corrupted image region is known?

It is not a deployable method.

## Data Tracks and Progressive Gates

### Track A — controlled

- 10–20 calibrated objects;
- four views per object;
- one corrupted view initially;
- exact masks for all cases; and
- exact warp offsets for affine/smooth warp cases.

### Track B — real VLM deformation

- independently synthesized view images;
- source view identity and cameras retained;
- no oracle masks or flows; and
- original views used only for evaluation and visualization.

Track B starts only if Track A shows measurable oracle headroom.

### Gate A — synthetic unit and feature tests

- no flow checkpoint required;
- validate weights, masks, offsets, and sparse aggregation;
- measure clean-reference feature error; and
- reject methods that harm identical/clean views.

Implementation-level Gate A checks are complete for aggregation math,
projected mask sampling, projection metadata, sparse/default mean equality,
non-finite fallbacks, and unchanged trainable-parameter count. Dataset-level
clean-reference metrics have not been run.

### Gate B — Shape-512 stage-local pilot

- fixed SS coordinates;
- 3–5 representative objects;
- S0, S1, and S3 only; and
- verify artifacts and qualitative readability.

Shape-512 is first because its projection grid and sparse token budget are
smaller than the 1024 stages.

### Gate C — Shape-1024 and PBR-1024

- run only methods that pass Gate B;
- hold upstream SLat state fixed; and
- isolate geometry and appearance effects.

### Gate D — cumulative cascade

Apply the selected method to Shape-512, Shape-1024, and PBR-1024 only after
stage-local effects are understood.

### Gate E — real VLM inputs

Test consensus without oracle assistance and report failures caused by
large deformation, deletion, insertion, low overlap, or split consensus.

## Feature-Level Evaluation

Use the clean calibrated equal-mean aggregate as the controlled reference.

Required metrics:

- fused-feature cosine error;
- fused-feature L2 drift;
- L/H norm statistics before and after routing;
- corrupted-view weight mass inside the projected oracle region;
- clean-region feature preservation;
- correct unique-view retention;
- weight entropy and per-view mean;
- anchor versus non-anchor weight;
- consensus-to-oracle gap; and
- uniform deviation on all-clean inputs.

The central preservation question is:

> Does the method remove wrong evidence without rejecting evidence that is
> correct but visible in only one view?

Consistency alone is not a sufficient success metric.

## 3D Evaluation

No multi-view checkpoint-dependent 3D experiment has been run. No checkpoint
identity has been supplied or guessed, and there is no GPU/3D validation,
render comparison, or 3D quality result for this prototype. The following
remains the required protocol after checkpoint handoff.

When checkpoints are available, record:

- conditioning-view silhouette IoU;
- SSIM and LPIPS;
- foreground RGB error;
- DINO image similarity;
- held-out/novel-view consistency;
- MEt3R if its official implementation and camera/input contract are
  validated;
- SLat feature drift;
- final mesh/PBR qualitative comparison; and
- descriptive vertex, face, and component counts.

Without aligned 3D ground truth, Chamfer distance, normal consistency, and
mesh statistics are diagnostic changes from a baseline, not reconstruction
accuracy.

## Required Visualizations

The report must embed:

- source, clean, corrupted, and real VLM view grids;
- corruption masks and warp fields;
- projected voxel/pixel overlays;
- per-view confidence and aggregation-weight maps;
- clean-versus-corrupted fused-feature error maps;
- equal-mean, consensus, and oracle stage-local renders;
- L/H feature statistics;
- weight histograms and entropy;
- correct-unique-view failure cases; and
- final cascade comparisons.

A text-only report is not acceptable.

## Distribution-Shift Controls

The following choices limit but do not eliminate inference-time distribution
shift:

- weights are normalized across views;
- output is a convex combination of observed features;
- residual-to-uniform routing exposes a continuous path to the trained mean;
- L/H channel width and order are unchanged;
- global tokens remain at their trained arithmetic mean initially;
- no foreground or validity hard-zeroing is hidden in the method; and
- clean and repeated-view controls are mandatory.

Pure consensus and pure oracle rejection are stress/ceiling conditions and
must be labeled as such.

## Stage 2 Decision Gate

Do not implement deformable back-projection until both conditions hold:

1. S3 shows that suppressing known corruption improves feature-level or 3D
   outcomes; and
2. S1 confidence is statistically or qualitatively related to the controlled
   corruption rather than merely view novelty.

If the gate passes, Stage 2 will compare:

- Stage 1 weight-only routing;
- unregularized soft transport;
- 3D-neighbor smooth transport;
- confidence-residual transport; and
- exact oracle transport for controlled warps.

The proposed offset objective is:

```text
L =
L_consensus
+
lambda_smooth sum_(q,r in E) rho(delta_i(q) - delta_i(r))
+
lambda_magnitude sum_q ||delta_i(q)||^2
```

Only offsets may be optimized for a fixed 5–10 iterations. Pixal3D, DINO, and
NAF parameters remain frozen. This future phase requires a separate approved
implementation spec.

## Explicit Non-Goals

The current implementation does not include:

- SS-64 aggregation changes;
- flow-model or conditioner training;
- learned reliability networks;
- CorrAdapter* LoRA/LoFTR training;
- denoiser-level attention bypasses;
- source-to-VLM local-window support search;
- foreground hard masking;
- hard projection-validity masking;
- mesh/depth/occlusion weighting;
- last-mesh alignment;
- per-branch L/H routing;
- full test-time model optimization; or
- claims of solving multi-view inconsistency.

## Checkpoint Handoff

Before any 3D experiment, record for Shape-512, Shape-1024, and PBR-1024:

- absolute checkpoint path;
- file hash;
- training step;
- raw versus EMA state;
- source config;
- strict load result; and
- one equal-mean regression artifact.

No checkpoint path is guessed, and no public single-view checkpoint is used as
a substitute.

As of this implementation verification, that handoff has not occurred.

## Reproducibility Manifest

Each run records:

- code commit;
- checkpoint identity;
- object and view identifiers;
- source/generated/corrupted image hashes;
- camera transforms;
- K and seed;
- corruption mask/warp hashes;
- stage and method;
- alpha and temperature;
- sparse-coordinate or fixed upstream-SLat hash;
- sampler and CFG settings;
- artifact paths; and
- completion/failure state.

## Implementation Verification Evidence

Focused CPU regression:

```bash
PYTHONPATH=/tmp/pixal3d-flex-gemm-stub.gIVSQA${PYTHONPATH:+:$PYTHONPATH} \
  conda run -n pixal3d python -m pytest \
  tests/multiview/test_projection_aggregation.py \
  tests/multiview/test_conditioner.py \
  tests/multiview/test_online_mean.py \
  tests/multiview/test_projection_geometry.py \
  tests/multiview/test_pipeline_inputs.py \
  tests/multiview/test_inference_manifest.py -v
```

Result after final review fixes: 93 passed, 0 skipped, no warnings, in 4.57s.

Full CPU regression:

```bash
CUDA_VISIBLE_DEVICES='' \
  conda run -n pixal3d python -m pytest tests/multiview -q
```

Result after final review fixes: 806 passed, 32 skipped, and 6 identical
`wandb` `DeprecationWarning`s in 58.40s.

The successful no-driver run used an external import-only `flex_gemm` stub at
`/tmp/pixal3d-flex-gemm-stub.gIVSQA`; a temporary `.pth` outside the repository
made it visible to child collection processes. The repository-pinned
`wandb==0.26.1` dependency was installed into the `pixal3d` conda environment.
Neither environment adjustment is a production-code change.

The focused suite proves that the CLI is opt-in and that explicit sparse-first
mean is bitwise equal to the production `_online_mean_tensor_groups` reference
for nontrivial synthetic bf16 per-view values with active-coordinate gathering
and a grid-resolution override. It also proves `alpha=0` exactly recovers mean,
projection subset geometry matches the full grid, invalid and non-finite
weights fall back safely, consensus diagnostic weights do not change when an
oracle mask is supplied, and the conditioner adds no trainable parameter.
These are CPU synthetic interface/numerical regressions, not real DINO/GPU,
checkpoint, or 3D evidence.

## Success and Stop Criteria

Proceed beyond Stage 1 only if:

- the oracle demonstrates useful headroom;
- consensus moves corrupted features toward the clean reference;
- clean-view damage remains controlled;
- correct unique evidence is not systematically rejected; and
- at least one stage-local 3D result improves without relying only on a
  consistency metric.

Stop or redesign if:

- oracle suppression has no effect;
- consensus confidence is unrelated to corruption;
- clean results degrade materially;
- improvements arise only from hard feature attenuation;
- the method rejects low-overlap or unique correct views; or
- checkpoint behavior cannot be reproduced under equal mean.

Negative outcomes are valid research results and must be reported without
inflating the claim.

After checkpoints and the gated experiments are complete, results,
visualizations, failure cases, and research interpretation are written to
`docs/CORR_ADAPTER_PROTOTYPE_REPORT.md`.
