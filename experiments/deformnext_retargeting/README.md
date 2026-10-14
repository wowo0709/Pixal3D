# DeformNeXt retargeting: native multi-view revision

This implementation starts from `origin/dev-jaehyeok` commit `04f9883`.
The SS64 step 4210 and PBR1024 step 16850 mixed385 checkpoints contain
both frozen base weights and trained LoRA adapters. Continuation loads the
complete state strictly and starts a fresh optimizer. DeformNeXt uses its
own train/validation split; mixed385 is checkpoint lineage, not deformation
training data.

## Conditioning contract

| Arm | Source projection | Additional module |
| --- | --- | --- |
| V1 | Original camera-aligned multi-view projection and mean | None |
| B1 | Frozen Omega cosine Top-1 | None |
| B2 | Shared frozen Q2C selects one Top-1 feature | None; no score averaging |
| B3 | Q2C score weights candidates and source views | Matching-aware attention |
| C1 | B3 | Retargeter with native target query; no deformation channel |
| C2 | B3 | Retargeter additionally receives 2048-D deformation context |

Every arm has one geometry-only M1 anchor and three textured M0 source
views. Target RGB is used for supervision/auditing, never as conditioning.
V1 has `deform_conditioner: null` and reads no transport cache. It uses the
native `dev-jaehyeok` global and projection means without modification.

Semantic arms preserve the native mean of all four global features. Their
projection is `(anchor_projection + 3 * transported_source_evidence) / 4`.
The anchor uses the canonical camera returned by the native multi-view
relative-camera transform. Corrected compact projection maps convert
latent anchor-frame positions into target patch addresses. The original
large descriptor/value caches are immutable; the loader overrides only
projection maps using the versioned sidecar schema.

C1/C2 have a zero-initialized final residual layer and a nonzero gate. A
zero gate together with a zero output layer would prevent either from
learning. C1 rejects deformation tensors entirely. Q2C is shared and
frozen so B2-to-B3 measures feature consumption rather than different
matching checkpoints.

The original `runs/` experiment used a different, incorrectly labelled
V1 transport path and noncanonical projections. Those checkpoints and
metrics are historical diagnostic artifacts and **must not be used as
valid vanilla/ablation results**. Corrected experiments use a separate
`runs_native_v2/` root and separate W&B IDs, initialized from the original
mixed385 checkpoints rather than the historical deformation checkpoints.

## Validation and acceptance checks

`verify_gpu.py` loads the real base+LoRA checkpoints and verifies:

- V1 conditioning and denoiser output are exactly equal to the native MV
  reference on the same tensors.
- Training and inference conditions match for all six arms. V1/Top-1
  require zero error. Weighted CUDA scatter aggregation permits absolute
  error at most 1e-5 due to floating-point atomic sum order.
- Losses are finite and backward reaches LoRA in every arm, and reaches
  both retargeting adapters.

CPU regressions independently verify canonical addresses against the
native projector, source/anchor averaging, absence of V1 transport I/O,
strict checkpoint continuation, and nonzero retargeter learning gradients.
Production SS and three-rank PBR also run full-batch ten-update smoke
checks with checkpoint save before launch. These checks establish correct
execution, not sufficient generation quality.

Validation runs before the first update and at each epoch boundary with
fixed samples, seeds, 12 Euler steps and CFG 3. SS reports voxel IoU,
precision, recall and F1 on four generated assets, plus loss on 32 assets.
PBR uses two generated assets, one 256px anchor render and 16 loss assets.
It reports foreground PSNR/base-color PSNR, crop SSIM/LPIPS and legacy
whole-image metrics. Whole-image metrics include background and must not
be used alone. PBR generation is conditioned on GT shape/support, so
`gt_shape_conditioned_alpha_iou` is a diagnostic, not a geometry score.
Displayed GT is decoder reconstruction of GT latent, not the raw reference
render. A four-asset depth-anchor/target-render audit gave mask IoU above
0.996; this is a sampled audit, not a dataset-wide proof.

A generation baseline is recorded at step zero to distinguish checkpoint
capability from fine-tuning changes. Loss reduction or nonzero gradients
alone do not establish successful deformation generation.

## Launching

Generate recipes with explicit retained PBR counts and versioned outputs:

```bash
python experiments/deformnext_retargeting/configure.py \
  --data-root /path/to/retargeting-data \
  --output-root /path/to/runs_native_v2 \
  --config-root /path/to/configs-native-v2 \
  --training-assets 9149 --pbr-training-assets 9137 \
  --ss-checkpoint /path/to/ss64-mixed385.pt \
  --pbr-checkpoint /path/to/pbr1024-mixed385.pt \
  --q2c-checkpoint /path/to/shared-q2c.pt
python -m experiments.deformnext_retargeting.projection_maps \
  --manifest /path/to/train.jsonl --output-root /path/to/projection_maps_v2
CUDA_VISIBLE_DEVICES=<assigned-device> PYTHONPATH=. \
  python -m experiments.deformnext_retargeting.verify_gpu \
  --stage ss64 --config-root /path/to/configs-native-v2 \
  --output-root /path/to/verification
```

Run the projection builder for validation too, and GPU verification for
PBR before launching its campaign. Assign physical GPUs explicitly at
launch. Current resolved allocation is SS on one GPU, global batch 96;
PBR on three GPUs, global batch 24. SS V1 microbatch 96 passed ten updates
with peak allocated about 87.3 GiB on a 95.6 GiB GPU. Semantic SS arms
benchmark their own maximum microbatch before training; global batch
stays 96. PBR uses four balanced accumulation packs per rank (average
microbatch 2), because sparse token counts vary across assets.

`run_campaign.py` requires passing GPU reports, checks native config
schema and validates completion before proceeding. Training errors stop
the queue; file-server sync errors are recorded for deferred retry and do
not discard local checkpoints. Data and generated artifacts belong outside
Git. Model construction and decoder dependencies require the verified
Pixal3D environment and cached pretrained weights.

## SS supervision audit (anchor-v3)

A controlled four-asset diagnostic found that the legacy SS labels derived
from the exported Shape512 artifacts do not agree with the M1 anchor frame.
Reprojecting their decoded GT gave mask IoU 0.071–0.332. Increasing sampling
steps or replacing the depth input with target RGB did not fix the low
scores. Re-encoding SS from the canonical M1 Shape1024 support raised GT
mask IoU to 0.563–0.848. With the checkpoint, input images, noise and sampler
held fixed, mean per-asset prediction IoU rose from 0.02795 to 0.30474.
This comparison isolates a supervision/evaluation alignment error; it is
not a fine-tuning improvement.

The encoder uses binary occupancy of the canonical 64³ M1 support and the
original pretrained SS encoder, with deterministic posterior means. It
writes a new schema, source-coordinate hash and versioned manifests without
overwriting original labels. Train has 9,149 assets, validation 508. Configure
SS with `--ss-manifest-root <manifests_anchor_v3>` and
`--ss-latent-schema deformnext-ss64-m1-anchor-v3`. The loader rejects old
manifest/NPZ labels when that schema is required. SS restarts from the
original mixed385 checkpoint under `runs_ss_anchor_v3`; it does not resume
the 24 updates trained against invalid SS targets. The subsequent world-frame
audit below supersedes the earlier assumption that Shape1024/PBR was aligned.

`diagnose_ss.py` produces controlled diagnostics only. Its RGB oracle cases
use target texture to test the cause and must never enter training or the
ablation table. `build_ss_anchor_latents.py` publishes versioned labels and
manifests only after complete encoding. The sampled audit does not prove
perfect dataset-wide silhouette reconstruction at SS64 resolution.

## Render-world audit and v4 correction

The whole-dataset audit checked 9,657 assets: Shape/PBR coordinates match
exactly, SS hashes identify the corresponding Shape support, all latent
features are finite, and train/validation have no overlap in sample, pair,
source UID or source cluster. Those checks prove internal consistency; they
do not prove agreement with the original M1 rendering frame.

Independent GT decoding/rendering on 32 category-stratified validation
assets found mean Shape mask IoU 0.73376 against the original depth anchor.
The manifest incorrectly inherited constant scale 0.96 from legacy SS.
Correcting only scale improved the mean to 0.84224, but left substantial
errors, including Scissors at 0.30194.

The actual cause is `dump_pbr.main()` calling `normalize_scene()`: each
deformed target is independently centered and resized before serialization.
The builder used that normalized dump with the original cameras without
restoring its world frame. SS anchor-v3 inherits the same discrepancy from
Shape1024. Equal sparse support cannot detect this shared error.

Restore dump vertices using the original target bounding-box center and
maximum extent before native camera transformation and encoding. With this
inverse transform, raw-mesh mask IoU over the same 32 assets is mean
0.9999965, minimum 0.9999235. This verifies world-frame restoration before
VAE reconstruction; it does not certify decoded v4 latents or full-dataset
registration. A subsequent real encode/decode audit of the same 32 assets
gave mean Shape mask IoU 0.9974700, minimum 0.9765059. SS decoded occupancy
matched its pre-encoder 64³ support exactly (voxel IoU 1.0 for all 32).
SS mask IoU against the high-resolution anchor is mean 0.8064105, minimum
0.5114658; coarse support loses thin details despite correct registration.
Four guided PBR attribute reconstructions were finite. These are sampled
checks; full-cache integrity and training preflight must finish before launch.

`target_frame.py` preserves immutable original dumps and material data.
`build_world_manifests.py` writes separate pending manifests, new Shape/PBR
paths and per-asset scales. `build_pbr_latents.py` writes schema and scale
metadata; the PBR loader rejects mismatched schemas/scales. SS must then be
re-encoded from the new support and semantic projection maps rebuilt using
the new scales. Retain the original split and never condition on target RGB.

Both `runs_native_v2` and `runs_ss_anchor_v3` are invalid for final results;
the launcher rejects `INVALIDATED_GT_ALIGNMENT.json` in a run's ancestors.
The SS step-6 and PBR step-129 checkpoints were saved before stopping and
are diagnostic artifacts only. New experiments must start from the original
mixed385 checkpoints, after decoded-GT, six-arm and batch checks pass again.

`audit_registration.py` records full integrity checks, independent mesh
reprojection, GT VAE reconstruction, and fixed category panels. Report
`assets_checked: 0` explicitly denotes a rerender-only audit.

`rebuild_world_cache.py` rebuilds Shape/PBR on explicitly assigned GPUs,
then SS and semantic projection maps, audits the complete split and 32 GT
reconstructions, and synchronizes new artifacts to file4. It never starts
training. Its final status is `CACHE_AUDIT_PASS_TRAINING_PREFLIGHT_PENDING`.
Fresh six-arm GPU verification, full-batch checks, a step-zero baseline and
recounting PBR's retained token-filtered assets remain required. Original
Shape512 export labels are not used by the SS/PBR experiment.
