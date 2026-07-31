# Correspondence Controlled Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic, checkpoint-independent projection/mask sampling and C4/C5 forward-warp convention utilities needed for controlled oracle experiments.

**Architecture:** Keep geometry diagnostics outside `ProjGrid.forward()` so the trained baseline remains untouched. Reuse the existing camera projection primitive, expose its coordinates and validity as a diagnostic value object, and define all controlled warp artifacts around a documented clean-pixel forward map `T(p)=p+u(p)` with a separately stored inverse sampling grid.

**Tech Stack:** Python 3.11, PyTorch 2.8, pytest

## Global Constraints

- These utilities belong to `pixal3d.experiments.correspondence`; they do not change model conditioning, denoiser, checkpoints, or state dicts.
- Projection validity is returned for diagnostics but is not multiplied into baseline feature aggregation.
- Oracle mask values are sampled at the same normalized projection coordinates, with zero padding outside the mask image; validity remains a separate tensor.
- Pixel coordinates are `(x,y)` with integer pixel centers at `(0,0)`.
- Normalized coordinates use `align_corners=False`: `x_n=2*(x+0.5)/W-1`, `y_n=2*(y+0.5)/H-1`.
- Stored warp convention is forward clean-pixel displacement: `T(p)=p+u(p)` and `delta_oracle(p)=u(p)`.
- Image resampling may use an inverse grid, but it must never relabel that inverse grid as oracle forward correspondence.
- Identity offsets are exactly zero; translation offsets equal the supplied translation.
- Out-of-bound samples are explicitly marked invalid; no border replication is used for controlled warp artifacts.
- C4/C5 utilities are controlled-data oracles only and must not be applied to actual VLM outputs as oracle correspondence.
- No mesh, source-image local search, foreground hard masking, Stage 2 optimization, GPU experiment, or 3D generation is part of this plan.
- Every production behavior follows RED → GREEN TDD.

---

### Task 1: Projection diagnostics and oracle-mask sampling

**Files:**
- Create: `pixal3d/experiments/correspondence/projection.py`
- Modify: `pixal3d/experiments/correspondence/__init__.py`
- Test: `tests/multiview/test_correspondence_projection.py`

**Interfaces:**
- Consumes existing
  `project_points_to_image_batch(points_3d, transform_matrix, camera_angle_x, resolution)`.
- Produces:

```python
@dataclass(frozen=True)
class MultiviewProjection:
    pixel_coordinates: torch.Tensor       # [B,K,N,2]
    normalized_coordinates: torch.Tensor  # [B,K,N,2]
    depth: torch.Tensor                   # [B,K,N]
    valid_mask: torch.Tensor              # [B,K,N], bool


def project_points_for_views(
    points_3d: torch.Tensor,
    projection_transforms: torch.Tensor,
    camera_angle_x: torch.Tensor,
    *,
    resolution: int,
) -> MultiviewProjection:
    ...


def sample_oracle_masks(
    masks: torch.Tensor,
    normalized_coordinates: torch.Tensor,
) -> torch.Tensor:
    ...


def oracle_reliability(sampled_masks: torch.Tensor) -> torch.Tensor:
    ...
```

- [ ] **Step 1: Write failing coordinate/shape tests**

Create literal fixtures for `B=1`, `K=2`, `N=3`. Use identity-like valid
camera transforms accepted by the existing projector. Compare
`project_points_for_views()` against two direct calls to the existing
single-batch projection function and independently compute normalized
coordinates with:

```python
expected_norm = (expected_pixels + 0.5) / resolution * 2 - 1
```

Assert exact output ranks, boolean validity, and that a point behind the
camera is invalid.

- [ ] **Step 2: Run projection tests and verify RED**

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_correspondence_projection.py \
  -k "project_points" -q
```

Expected: import failure because `projection.py` does not exist.

- [ ] **Step 3: Implement multiview projection wrapper**

Validate:

- points `[N,3]` or `[B,N,3]`;
- transforms finite `[B,K,4,4]`;
- FOV finite positive `[B,K]`;
- integer `resolution >= 1`;
- batched points have matching B.

Flatten B and K only for the call into
`project_points_to_image_batch()`, expanding points per view without
changing values. Reshape all outputs back to `[B,K,N,...]`. Compute
normalized coordinates with the exact `align_corners=False` formula above.
Return validity without applying it to coordinates or features.

- [ ] **Step 4: Run projection tests and verify GREEN**

Run the Step 2 command.

Expected: all selected tests pass.

- [ ] **Step 5: Write failing mask-sampling tests**

Use a `3x3` mask with a single center value one and hand-authored normalized
queries for:

- center pixel `(1,1)` → sampled value exactly one;
- corner pixel `(0,0)` on an all-one mask → exactly one;
- far outside coordinate `(2,2)` → exactly zero due to zero padding.

Assert `sample_oracle_masks()` returns `[B,K,N]`, preserves FP32, and does
not consume or change a separately supplied validity tensor. Add rejection
tests for non-floating masks, values outside `[0,1]`, shape mismatch, and
non-finite coordinates.

Test `oracle_reliability(torch.tensor([0.0,0.25,1.0]))` equals
`[1.0,0.75,0.0]` exactly and rejects values outside `[0,1]`.

- [ ] **Step 6: Run mask tests and verify RED**

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_correspondence_projection.py \
  -k "oracle or mask" -q
```

Expected: failures because mask APIs are missing.

- [ ] **Step 7: Implement mask sampling and reliability**

Require masks `[B,K,H,W]` and coordinates `[B,K,N,2]`. Flatten B/K, call
`torch.nn.functional.grid_sample` with:

```python
mode="bilinear"
padding_mode="zeros"
align_corners=False
```

Return `[B,K,N]` in the mask dtype. Do not accept or apply projection
validity. Reliability is the elementwise `1-mask` in the input floating
dtype after finite/range validation.

- [ ] **Step 8: Run focused and full regressions**

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_correspondence_projection.py -q
```

Then:

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview -q
```

Expected: focused and full suites pass.

- [ ] **Step 9: Commit Task 1**

```bash
git add \
  pixal3d/experiments/correspondence/__init__.py \
  pixal3d/experiments/correspondence/projection.py \
  tests/multiview/test_correspondence_projection.py
git commit -m "feat: add correspondence projection diagnostics"
```

---

### Task 2: Forward-field warp convention and controlled C4/C5 artifacts

**Files:**
- Create: `pixal3d/experiments/correspondence/warps.py`
- Modify: `pixal3d/experiments/correspondence/__init__.py`
- Test: `tests/multiview/test_correspondence_warps.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class ControlledWarp:
    warped_image: torch.Tensor       # [C,H,W]
    affected_mask: torch.Tensor      # [H,W], bool
    forward_field_px: torch.Tensor   # [H,W,2], (dx,dy)
    inverse_grid_norm: torch.Tensor  # [H,W,2]
    invalid_mask: torch.Tensor       # [H,W], bool


def pixel_to_normalized(points_px: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    ...


def normalized_to_pixel(points_norm: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    ...


def affine_forward_field(
    matrix: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    ...


def affine_inverse_grid(
    matrix: torch.Tensor,
    *,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...


def invert_forward_field(
    forward_field_px: torch.Tensor,
    *,
    iterations: int = 12,
    convergence_tolerance_px: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...


def warp_affine(
    image: torch.Tensor,
    affected_mask: torch.Tensor,
    matrix: torch.Tensor,
) -> ControlledWarp:
    ...


def warp_with_forward_field(
    image: torch.Tensor,
    affected_mask: torch.Tensor,
    forward_field_px: torch.Tensor,
    *,
    inverse_iterations: int = 12,
    convergence_tolerance_px: float = 1e-3,
) -> ControlledWarp:
    ...


def smooth_random_forward_field(
    affected_mask: torch.Tensor,
    *,
    seed: int,
    max_displacement_px: float,
    coarse_size: int = 5,
    blur_kernel_size: int = 9,
    max_gradient: float = 0.5,
) -> torch.Tensor:
    ...
```

- [ ] **Step 1: Write failing pixel/normalized conversion tests**

For `H=3`, `W=5`, hand-check:

```python
points = torch.tensor([[0.0, 0.0], [4.0, 2.0], [2.0, 1.0]])
expected = torch.tensor([
    [-0.8, -2.0 / 3.0],
    [ 0.8,  2.0 / 3.0],
    [ 0.0,  0.0],
])
```

Assert forward conversion equals `expected`, inverse conversion reconstructs
`points`, dtype/device are preserved, and invalid height/width/rank are
rejected.

- [ ] **Step 2: Run conversion tests and verify RED**

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_correspondence_warps.py \
  -k "pixel_normalized" -q
```

Expected: import failure because `warps.py` does not exist.

- [ ] **Step 3: Implement coordinate conversion**

Implement exactly:

```text
x_n = 2*(x+0.5)/W - 1
y_n = 2*(y+0.5)/H - 1
x = (x_n+1)*W/2 - 0.5
y = (y_n+1)*H/2 - 0.5
```

Require floating `[...,2]` tensors and positive integer sizes.

- [ ] **Step 4: Run conversion tests and verify GREEN**

Run the Step 2 command.

- [ ] **Step 5: Write failing identity/translation field tests**

Use a `5x5` one-channel image with a marker at clean pixel `(1,2)`.

- identity affine matrix `[[1,0,0],[0,1,0]]` passed to `warp_affine()`
  yields exact zero forward field, an exact matrix-derived identity inverse
  grid, no invalid pixels, and unchanged image;
- translation matrix `[[1,0,1],[0,1,-1]]` passed to `warp_affine()` yields
  forward field `(dx=1,dy=-1)` everywhere, an inverse grid derived from the
  exact inverse affine matrix, and moves the marker to `(2,1)` in the warped
  image;
- passing that same constant translation field to
  `invert_forward_field(iterations=12)` yields the same interior inverse
  coordinates and invalid border mask as the exact affine inverse, proving
  the numerical solver does not exhibit even/odd zero-padding oscillation;
- the translation artifact stores the forward field, not the negated inverse
  sampling displacement;
- pixels whose inverse source coordinate is outside `[0,W-1]x[0,H-1]` are
  true in `invalid_mask` and are zero in the warped image.

Use an all-true affected mask for these convention tests.

- [ ] **Step 6: Run identity/translation tests and verify RED**

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_correspondence_warps.py \
  -k "identity or translation" -q
```

Expected: failures because field and warp APIs are missing.

- [ ] **Step 7: Implement exact affine fields/inversion and smooth-field inversion**

Build a pixel-center mesh `[H,W,2]`. `affine_forward_field()` applies the
forward `2x3` matrix to homogeneous clean pixel coordinates and subtracts
the clean coordinates.

`affine_inverse_grid()` augments the `2x3` forward matrix to a homogeneous
`3x3` matrix, rejects singular matrices, computes its exact inverse in FP32,
and applies that inverse to every destination pixel. It returns normalized
source coordinates and an invalid mask based on whether the exact source
pixel is outside `[0,W-1]x[0,H-1]`.

`warp_affine()` must use `affine_forward_field()` for the stored oracle
forward displacement and `affine_inverse_grid()` for resampling. It must not
route affine transforms through iterative field inversion.

For non-affine smooth fields, `invert_forward_field()` solves
`p_src + u(p_src) = p_dst` by fixed-point iteration:

```text
p_src_0 = p_dst
p_src_{t+1} = p_dst - sample(u, p_src_t)
```

Sample displacement during the numerical solver with bilinear
`grid_sample`, `padding_mode="border"`, and `align_corners=False`. Border
extension is a solver convention only: it prevents an out-of-bounds
candidate from reading a fictitious zero displacement and oscillating back
inside. It does not make an out-of-bounds source valid.

After the final iteration, sample the border-extended displacement once
more and compute:

```text
residual = ||p_src + u(p_src) - p_dst||_2
```

The returned invalid mask is true if the final source pixel is outside the
image or residual exceeds `convergence_tolerance_px`. Validate that
`iterations >= 1` and the tolerance is finite and positive.

`warp_with_forward_field()` validates finite floating image/field and bool
mask shapes, obtains the inverse grid, samples with bilinear/zero padding,
and composites only the affected destination region:

```text
warped = where(affected_mask, sampled, original)
```

Set affected invalid pixels to zero. Return detached values only when inputs
are detached; do not force a device transfer.

- [ ] **Step 8: Run identity/translation tests and verify GREEN**

Run the Step 6 command.

- [ ] **Step 9: Write failing smooth-field tests**

For a fixed boolean central mask and seed `20260731`, assert:

- repeated calls produce bit-identical fields;
- outside-mask displacement is exactly zero;
- max vector magnitude is `<= max_displacement_px + 1e-5`;
- field shape is `[H,W,2]`, FP32, finite;
- `max_displacement_px=0` yields exact zero;
- maximum horizontal or vertical neighboring displacement-vector difference
  is `<= max_gradient + 1e-5`;
- invalid even blur kernel, coarse size below two, negative displacement,
  non-positive/invalid gradient limit, and empty mask are rejected.

Warp a coordinate-ramp image with a small deterministic field and assert the
artifact is finite, has the documented shapes, and preserves image values
outside the affected destination mask.

- [ ] **Step 10: Run smooth tests and verify RED**

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_correspondence_warps.py \
  -k "smooth" -q
```

Expected: failures because smooth field generation is missing.

- [ ] **Step 11: Implement deterministic smooth field generation**

Use a local CPU `torch.Generator().manual_seed(seed)` to sample a
`[1,2,coarse_size,coarse_size]` normal field. Bilinearly resize to H/W,
apply `torchvision.transforms.functional.gaussian_blur` with the odd kernel,
permute to `[H,W,2]`, multiply by the boolean mask, and rescale globally so
both conditions hold:

```text
max ||u(p)||_2 <= max_displacement_px
max_{horizontal/vertical neighbors} ||u(p)-u(q)||_2 <= max_gradient
```

Use one global non-increasing scale factor so direction and smoothness are
preserved. Return FP32 on the mask device. Do not change global RNG state.

- [ ] **Step 12: Run focused and full regressions**

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_correspondence_warps.py \
  tests/multiview/test_correspondence_projection.py -q
```

Then:

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview -q
```

Expected: all tests pass.

- [ ] **Step 13: Commit Task 2**

```bash
git add \
  pixal3d/experiments/correspondence/__init__.py \
  pixal3d/experiments/correspondence/warps.py \
  tests/multiview/test_correspondence_warps.py
git commit -m "feat: add controlled correspondence warps"
```

---

## Plan boundary

This plan establishes projection/oracle-mask sampling and controlled C4/C5
geometry conventions. C1 color/material changes, C2 decals, C3 deletion,
artifact serialization/contact sheets, real dataset selection, feature Gate A,
checkpoint loading, 3D generation, and Stage 2 transport remain separate
plans. No output from this plan is a result from a trained flow model.
