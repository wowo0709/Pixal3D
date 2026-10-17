# DeformNeXt retargeting: R9 / native-preserving revision v5

Code lineage is `origin/dev-jaehyeok` (`04f9883`). The SS step 4210 and PBR
step 16850 mixed385 checkpoints include frozen base and trained LoRA weights.
Continuation loads both strictly with a fresh optimizer. Training data is the
DeformNeXt split: SS 9,149 train / 508 validation, PBR 9,139 train after the
32,768-token filter. Mixed385's 80k assets are initialization lineage only.

## Current contract

**2026-10-17 SS matching-v2 campaign:** see the
[experiment plan](../../docs/superpowers/reports/2026-10-17-ss-matching-v2-experiment-plan-ko.md)
and [execution evidence](../../docs/superpowers/reports/2026-10-17-ss-matching-v2-execution-ko.md).
`matching_v2_setup.py` freezes cluster-disjoint calibration and nested common
evaluation panels. `matching_v2_proposals.py` measures expanded RAW proposal
recall; `train_matching_v2.py` trains independent H1/H2/H3 heads. Calibration
selects a fixed final head, temperature and separately supervised reliability.
`matching_v2_finalize.py` builds preaggregated inference evidence only after
GT-transport screening passes. Failed scientific gates hold SS runs and do not
silently substitute a matcher or change fusion strength. Existing weights are
re-evaluated; vanilla V1 is not retrained.

`matching_v2_queue.py --work-root <runtime> --gpus <assigned-physical-IDs>`
executes the frozen runtime `queue-jobs.json`. GPU assignments are explicit;
light head/cache jobs may share with bounded evaluation, while SS training is
exclusive. CPU jobs have no visible GPUs. The new `matching-v2` conditioner is
SS-only, preserves native output at initialization, rejects whole updates
including adapter residuals, and physically adds deformation channels only
for C2. It does not change the historical promoted R9 scorer contract.

Fixed-panel SS evaluation supports per-asset seeds independent of panel order,
batched generation, per-asset voxel counts and source-cluster comparisons.
`matching_v2_storage.py` retains file4 outputs; `matching_v2_reporter.py` logs
paired voxel metrics/images and explicit completion/hold/failure outcomes.

**2026-10-16 correction:** conditioning is edited **M1 RGB**, followed by three
M0 RGB views from different cameras. For the current target-camera-8 split,
the four cameras are `[8,9,10,11]`: edited input states are `[M1,M0,M0,M0]`,
source-only states are `[M0,M0,M0,M0]`. Source-only uses the original M0 image
at camera 8; it never removes a view or duplicates another camera. The old
`geometry_depth + M0 cameras [8,9,10]` implementation was a mistake. Its V1,
arm results, D1 head, and image-dependent caches are historical depth runs,
not evidence for the intended RGB experiment. Dataset loaders reject them.

`prepare_rgb_manifest.py` verifies the original render roles, four distinct
cameras, anchor pose, RGBA files and per-input content fingerprint. It reuses
audited world-v4 M1 geometry latents while versioning all image-dependent
caches under a separate RGB namespace. Since Omega mixes views jointly,
all four descriptor streams are recomputed. Native DINO/NAF values, topology
visibility, RAW matches and the seven-epoch D1 head are also refreshed. Old
local cache copies may be evicted only after a checksum-verified file4 mirror
and an atomically completed RGB replacement. Original file4 artifacts remain.

`run_rgb_restart.py` runs the paired four-view zero-shot stage controls, fresh
RGB V1, then B1–C2 after `prepare_rgb_matching.py` publishes verified readiness.
Each arm starts independently from mixed385 base+LoRA. W&B hardware monitoring
is disabled; training metrics and images remain enabled. PBR stage controls
still fix GT target shape and are explicitly labelled as such; they do not
establish end-to-end geometry reconstruction performance.

`pixal3d/modules/r9_matching.py` vendors the selected original R9 scorer
**verbatim**, with source and AST checksums. The fixed promoted tuple is
`L12 FRAME RAW 1024 -> GLOBAL64 -> P0-RAW -> M0-L12 -> D1-Q2C`.
D1 is one query-only contextual attention layer (four heads of width 32),
followed by the original bounded residual scalar head: 928,453 parameters.
`verify_r9_authority.py` compares initialization, forward and parameter
gradients to an explicit historical source, including nonzero learned gates.
The old 2048-D pairwise MLP is available only as `implementation=legacy-v4`
for historical diagnostics. Its checkpoints cannot load into the R9 path.

Matching uses only FRAME 1024-D. Existing FRAME+GLOBAL 2048-D Omega caches
remain immutable and supply the first branch directly. C2 separately retains
RAW FRAME+GLOBAL 2048-D context; it is not a 2048-D matching key. FP32 proposal
and D1 math are shared by offline, training and inference, including AMP runs.

| Arm | Semantic selection | Added adaptation |
| --- | --- | --- |
| V1 | Native spatial multi-view projection | Native LoRA continuation |
| B1 | RAW FRAME cosine Top-1 | Common zero-start scalar fusion gate |
| B2 | Frozen shared D1 Top-1 | Same gate; no score-weighted feature averaging |
| B3 | Frozen D1 categorical score attention | Same gate |
| C1 | B3 | Zero-output retargeter with native target query |
| C2 | C1 | Additional RAW 2048-D deformation context channels |

All semantic arms preserve native global conditioning. Their projection is
`C_native + tanh(g) * valid * delta + (3/4) * valid * R`, where
`delta = (C_anchor + 3*E)/4 - C_native`, `g=0` initially and the retargeter's
final output layer is zero. The retargeter is outside the zero semantic gate;
its output layer and the gate both receive gradients on the first update.
C1 has no deformation channels. C2 adds them physically. PBR semantic delta
modifies DINO LR channels only; native NAF HR conditioning remains present.
The retargeter can learn an HR residual from its initially zero output.

Feature attention inherits R8's train-only fixed temperature **0.005**, also
used by the promoted R9 recipe. This temperature is recorded in the cache
contract and shared by online attention and C2 context construction. The new
topology listwise head objective uses its own unscaled logits.
Categorical attention is invariant to query-wise score offsets. Candidate
weights are global probabilities; view weights report their marginal mass.
Source validity and the edited RGB input's alpha mask identify unavailable
queries, which retain native conditioning. A source pool with fewer than 64
valid sites also uses native fallback in every semantic arm: invalid background
tokens never become a useful padded D1 context. The historical depth-input scan found 130 such
train assets and no validation assets. They still participate in native LoRA
training. This is not a learned visibility or NULL classifier. `source-off` disables the semantic residual; the native
multi-view source path remains present.

## Teacher, cache and checkpoint separation

`topology_labels.py` transfers M1 barycentrics into M0 vertex positions, then
checks source projection, z-buffer depth (0.005) and input alpha. It records
input content hashes, role/split and raster agreement. Transparent materials
are checked by observed-alpha containment, allowing opaque raster overfill.
Opaque z-buffer visibility is conservative around cutouts: it can discard a
point visible through a transparent foreground surface. It is not a full
material-aware ray tracer. These labels never enter inference conditioning.

`train_q2c.py` requires those teacher files; it cannot fall back to projecting
X1 unchanged into M0 cameras. It distinguishes visible proposal misses and
unobservable queries, uses a fixed final checkpoint and records objective
provenance. **Topology listwise supervision is a new DeformNeXt objective;
it is distinct from the original R9 controlled RGB task loss.** Exact scorer
reproduction does not establish a new SS/PBR generation improvement.

The initial three-epoch CPU candidate reached validation Top-1 0.481319
(RAW 0.338762) and was preserved under `q2c_r9_v5/history`. It did not include
optimizer state. The user subsequently requested five to seven epochs; the
current production head therefore trains a fresh, fixed-seed **seven epochs**
on its explicitly assigned GPU. B2–C2 wait for that fixed-final checkpoint.
Epochs 3/5/7 are compared diagnostically, without selecting a best epoch.
Every new epoch saves weights, AdamW state and Python/NumPy/Torch/CUDA RNG.
`--resume-state` restores a complete epoch snapshot and rejects legacy
weight-only files or changed populations/recipes/devices. Optional
`--wandb-project/--wandb-entity/--wandb-id/--wandb-name` enable head logging.
For GPU head jobs, set `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
`NUMPY_MADVISE_HUGEPAGE=0` and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before importing NumPy/Torch.
The head transfers only FRAME keys through pinned asynchronous buffers;
cache preparation also pins its full FRAME+GLOBAL context. This avoids the
observed pageable-transfer and host huge-page compaction stalls while keeping
feature values unchanged. The actual GPU7 run measured 0.026–0.028 s/update
over its first 500 updates (roughly 35–45 minutes for seven epochs including
validation; preliminary estimate). Its current W&B ID is
`deformnext-r9-v5-d1-head-7ep-final`.

`train_q2c --device cpu` supports the same FP32, fixed-seed, three-epoch recipe
without sharing a fully occupied generation GPU. GPU sharing caused severe
warm-up starvation despite ample free memory. A 12-asset CPU probe with four
threads averaged 0.127 s/update over its last ten updates. Checkpoints record
the compute device; CPU and CUDA rounding need not be bit-identical. The
production frozen weight still receives actual cache/online and native-start
verification before B2–C2 run.

Full construction certified 9,650 teachers: 9,142 train and all 508 validation.
Seven train rows failed the unchanged raster-agreement criterion and are
recorded explicitly; only head supervision excludes them. Main SS/PBR training
populations remain unchanged. B1 has its own complete 9,657-row
`deformnext-raw-top1-v5` cache, requiring no teacher or learned matching head.

Learned matches use `deformnext-match-r9-v5`, exact matcher contract and weight
hash, categorical aggregation and 2048-D C2 context. New dataset configs reject
legacy caches and require world-v4 geometry/projection sidecars. Defaults point
to `manifests_world_v4`, `projection_maps_world_v4`, `matches_r9_v5` and a
separate `runs_r9_v5`; existing world-v4 latents need no reconstruction.

Portable semantic recipes start with `release_status=audit-hold`.
`run_campaign.py` permits a bounded one-epoch `pilot-ready` run after exact
native-start proof; B1 additionally requires its RAW cache contract, and B2–C2
require the exact frozen matcher hash. An epoch-one execution/quality gate
checks finite metrics, the identical panel and major regressions against both
step zero and same-epoch native V1. Passing continues to five epochs with the
same optimizer, sampler and RNG; failure preserves the checkpoint and writes
`PILOT_HOLD.json`. This gate does not establish a module improvement. The
durable `run_r9_pipeline` queue attaches to B1 and runs B2/B3/C1/C2 independently
from the original mixed385 initialization, with a shared fixed-final D1 head.
An explicit `--matching-gpus` assignment places cache workers and cache/online
verification on additional devices, disjoint from generation. No extra GPUs
are selected automatically. `--attach-running-campaigns` validates the existing
stage campaign PID files and attaches to them without relaunching training.
The old `runs_world_v4` semantic campaign remains under `SEMANTIC_AUDIT_HOLD`.
Its native V1 remains a valid baseline. Original invalid-geometry runs remain
historical diagnostics and must not be resumed.

## Verification and evaluation

Use the verified Pixal3D environment and explicitly assigned physical GPUs.
Run modules from the repository root with `PYTHONPATH=.` and `ATTN_BACKEND=sdpa`.

- `verify_r9_authority`: original/vendored scorer state, forward, backward parity.
- `verify_native_start`: actual mixed385 SS/PBR checkpoint, all-arm condition,
  denoiser and same-noise 12-step generation identity, plus native LoRA/gate/
  adapter gradients. Use `--q2c-checkpoint` to verify the actual frozen matcher.
- `verify_match_cache`: real offline/online routing under BF16 AMP with an
  active nonzero C2 adapter.
- `verify_conditioner_ddp`: two updates and exact parameter synchronization
  on explicitly assigned ranks, including the fusion gate.

Step-zero and epoch-end quick metrics remain SS voxel IoU/precision/recall/F1
and PBR foreground/base-color PSNR plus cropped SSIM/LPIPS. Generation panels
are small trend checks, not final population metrics. PBR uses GT shape/support;
alpha IoU is not generated-geometry quality. The PBR renderer uses common AO
random rays for comparisons and preserves the surrounding training RNG.

Target batch recipes remain SS global 96 on one GPU (split 1) and PBR global
24 on three GPUs (per-rank 8, two packs capped at four assets). Actual B1 v5
profiling measured SS micro 96 at 90.14 GiB allocated peak, roughly 36 s/update;
PBR micro 8 exhausted memory, while micro 4 used 64.50 GiB on the high-token
stress panel. Each subsequent arm receives a fresh production-batch probe.
These are separate from the historical v4 measurements below.

## Historical registration and runtime audits

The following records explain geometry corrections and earlier runs. Current
semantic contracts and launch policy are the v5 definitions above.

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

## World-v4 launch preflight

All four versioned caches contain 9,657 assets and were synchronized to
file4 after the complete integrity audit passed. Re-encoding changed PBR's
token-filtered train count to 9,139 (10 exclusions); SS retains 9,149.
Six-arm GPU parity and backward checks passed again for both stages.
Production-batch ten-update/save tests passed for SS microbatch 96 on GPU 0
and PBR global batch 24 on GPUs 1–3. A second PBR test passed with two
balanced accumulation packs per rank (average microbatch 4), peak allocated
79.53 GiB across ranks. SS peak allocated was 87.33 GiB.

New campaigns use separate `runs_world_v4` outputs and W&B IDs, initialize
from the original mixed385 weights, validate at step zero and every epoch,
and queue V1/B1/B2/B3/C1/C2. `--archive-checkpoints` copies a completed
arm to file4, verifies remote checkpoint contents using rsync checksums,
stores an archive receipt on both sides, then evicts duplicate local cache
checkpoint files. Failed sync/verification retains local files. Completed
archived arms are skipped on relaunch; resuming or extending one requires
retrieving its checkpoint first. This keeps the six-arm queue within local
disk capacity while preserving every saved epoch on file4.

## PBR image-pack cap correction

The initial world-v4 PBR V1 attempt stopped after update 28: token-only
balancing split the next rank-1 batch into 2/6 assets, although the configured
average microbatch was 4. NAF's dense image attention attempted a 24 GiB
allocation on the six-image pack. Sparse token balance cannot bound that
image memory. The collator now limits each pack to `ceil(local_batch / splits)`
assets while balancing token load within the capacity. Regression cases
cover the pathological 1/7 token-balanced split and real image collation.

Re-encoded geometry also changes token counts; the PBR loader reads actual
versioned NPZ support even when a manifest contains older counts. World-v4
train/validation manifests were refreshed and synchronized to file4 with
previous counts retained for provenance. The current counts are 9,139
retained PBR training assets and all 508 validation assets.

A three-rank two-update/save stress test on the 24 largest retained assets
(maximum 31,900 tokens per asset) passed with 4/4 image packs on every rank;
peak allocated was 64.46 GiB. The old attempt has `FAILED_ATTEMPT.json` and
failed progress status. PBR restarts from the original mixed385 checkpoint
under local `v1-cap4`, with a separate W&B ID ending `v1-cap4`, global batch
24 and two accumulation packs. The same bounded collator applies to all
subsequent PBR ablation arms. SS continues its existing world-v4 run.

## Native projection retargeting follow-up (2026-10-17)

The old SS/PBR campaign is stopped and preserved. The raw matching-v2 ladder
remains held by its GT transport screen. A separate retargeting follow-up
uses the positive common-panel C1 result to test native bilinear projection
with support-normalized fallback: P1 retains the old frozen R9 attention;
P2 changes only the matcher bundle after fixed-final calibration selection.
Both start independently from the original SS mixed385 base+LoRA weights.

Reference R9 evidence is preaggregated in bounded chunks, with FP32 weighted
values and an online parity audit, to remove the Top64 value expansion from
SS batch96. `initialization_rng=r9-compatible` preserves the original
retargeter initialization despite removing the frozen scorer from training.
Neither cache constructor reads GT teachers. Each run repeats real native
identity/backward and largest-microbatch two-update checks before training.

See the [diagnostic protocol](../../docs/superpowers/reports/2026-10-17-native-transport-projection-audit-ko.md)
and [follow-up learning plan](../../docs/superpowers/reports/2026-10-17-ss-native-retargeting-followup-ko.md).
Runtime configs and caches live outside Git under the campaign's
`retarget-native-v3/` namespace. Native bilinear is opt-in; existing
nearest-projection recipes retain their behavior.
