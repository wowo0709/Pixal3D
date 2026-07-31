# Multi-view Correspondence Stage 1 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a checkpoint-independent, inference-time Stage 1 consensus aggregation foundation that preserves the existing equal-mean path and exposes reproducible feature-level diagnostics.

**Architecture:** Keep the current `_online_mean_tensor_groups()` path as an exact fast path. Put parameter-free consensus math in a focused experiment module, integrate it immediately before the current view mean disappears, and evaluate fused features with pure tensor metrics. Non-default modes cache per-view projection tensors off-device and aggregate voxel chunks on a caller-selected compute device; no denoiser or checkpoint schema changes.

**Tech Stack:** Python 3.11, PyTorch, dataclasses, pytest

## Global Constraints

- The implementation target is the multi-view code on `feature/multiview-correspondence`, not the single-view repository behavior.
- SS64 is not an aggregation intervention target; Shape512, Shape1024, and PBR1024 are the eventual targets.
- Default aggregation remains the existing arithmetic mean and must call the existing `_online_mean_tensor_groups()` path.
- `alpha=0` must call the same existing arithmetic-mean path.
- K=1 must call the same existing arithmetic-mean path.
- Global output remains `[B,5,1024]`.
- Projection output remains `[B,R^3,2048]` in the real target models.
- L/H projection order remains `[low, high]`; one scalar view weight is applied to both halves.
- The denoiser architecture, ProjectAttention input, and flow-model state dict must not change.
- Projection `valid_mask` must not be added to S0/S1 weighting.
- New aggregation code is parameter-free and inference-only.
- NaN/non-finite confidence rows fall back to uniform view weights.
- All view weights sum to one along the view dimension.
- No GPU experiment, checkpoint selection, model training, mesh proxy, transport, or actual-VLM oracle is part of this plan.
- Every production behavior is implemented with RED → GREEN TDD evidence.

---

### Task 1: Pure consensus aggregation kernel

**Files:**
- Create: `pixal3d/experiments/__init__.py`
- Create: `pixal3d/experiments/correspondence/__init__.py`
- Create: `pixal3d/experiments/correspondence/aggregation.py`
- Test: `tests/multiview/test_correspondence_aggregation.py`

**Interfaces:**
- Consumes: A sequence of K tensors, each shaped `[B,N,2D]`, with L in `[..., :D]` and H in `[..., D:]`.
- Produces:
  - `ProjectionAggregationDiagnostics(scores, weights, entropy, fallback_mask)`
  - `residual_to_uniform_weights(scores, *, alpha, temperature)`
  - `aggregate_consensus_projection_naive(stacked_features, *, alpha, temperature)`
  - `aggregate_consensus_projection(view_features, *, alpha, temperature, chunk_size, compute_device, output_device)`
- `scores` and `weights` use shape `[B,K,N]`; `entropy` and `fallback_mask` use `[B,N]`.

- [ ] **Step 1: Write failing tests for residual weighting and fallback**

Add tests with hand-derived expectations:

```python
def test_alpha_zero_returns_uniform_weights():
    scores = torch.tensor([[[3.0], [1.0], [-2.0], [0.5]]])
    diagnostics = residual_to_uniform_weights(
        scores, alpha=0.0, temperature=0.2
    )
    torch.testing.assert_close(
        diagnostics.weights,
        torch.full((1, 4, 1), 0.25),
        rtol=0,
        atol=0,
    )


def test_nonfinite_score_row_uses_uniform_fallback():
    scores = torch.tensor([[[1.0], [float("nan")], [0.0], [-1.0]]])
    diagnostics = residual_to_uniform_weights(
        scores, alpha=1.0, temperature=0.2
    )
    assert diagnostics.fallback_mask.item()
    torch.testing.assert_close(
        diagnostics.weights,
        torch.full((1, 4, 1), 0.25),
        rtol=0,
        atol=0,
    )
```

- [ ] **Step 2: Run the residual tests and verify RED**

Run:

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_correspondence_aggregation.py \
  -k "alpha_zero or nonfinite" -q
```

Expected: collection/import failure because
`pixal3d.experiments.correspondence.aggregation` does not exist.

- [ ] **Step 3: Implement diagnostic types and residual weighting**

In `aggregation.py`, add:

```python
@dataclass(frozen=True)
class ProjectionAggregationDiagnostics:
    scores: torch.Tensor
    weights: torch.Tensor
    entropy: torch.Tensor
    fallback_mask: torch.Tensor


def residual_to_uniform_weights(
    scores: torch.Tensor,
    *,
    alpha: float,
    temperature: float,
) -> ProjectionAggregationDiagnostics:
    ...
```

Validate `[B,K,N]`, `0 <= alpha <= 1`, finite positive temperature, and
`K >= 1`. Compute softmax along dimension 1. If any score in a `[B,:,N]`
row is non-finite, replace the entire row with exact uniform weights. Compute
entropy as `-(w * log(clamp_min(w, finfo.tiny))).sum(dim=1)`. Preserve the
score device; return floating weights in FP32.

- [ ] **Step 4: Run residual tests and verify GREEN**

Run the command from Step 2.

Expected: both selected tests pass with no warnings.

- [ ] **Step 5: Write failing tests for consensus behavior**

Add separate tests proving:

```python
def test_one_opposite_outlier_receives_less_weight_than_three_agreeing_views():
    good = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
    outlier = torch.tensor([[[-1.0, 0.0, -1.0, 0.0]]])
    stacked = torch.stack([good, good, good, outlier], dim=1)
    result, diagnostics = aggregate_consensus_projection_naive(
        stacked, alpha=1.0, temperature=0.2
    )
    assert diagnostics.weights[0, 3, 0] < diagnostics.weights[0, 0, 0]
    assert result[0, 0, 0] > 0
    assert result[0, 0, 2] > 0


def test_identical_views_produce_uniform_weights_and_input_feature():
    feature = torch.tensor([[[2.0, -1.0, 0.5, 4.0]]])
    stacked = feature[:, None].repeat(1, 4, 1, 1)
    result, diagnostics = aggregate_consensus_projection_naive(
        stacked, alpha=1.0, temperature=0.2
    )
    torch.testing.assert_close(result, feature, rtol=0, atol=0)
    torch.testing.assert_close(
        diagnostics.weights,
        torch.full((1, 4, 1), 0.25),
        rtol=0,
        atol=0,
    )
```

Also test K=1, odd channel rejection, non-floating input rejection, shape
mismatch, invalid alpha, and invalid temperature.

- [ ] **Step 6: Run consensus tests and verify RED**

Run:

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_correspondence_aggregation.py \
  -k "outlier or identical or rejects or k1" -q
```

Expected: failures because the aggregation functions are missing.

- [ ] **Step 7: Implement the naive LOO consensus reference**

Implement `aggregate_consensus_projection_naive()`:

1. Validate finite floating `[B,K,N,C]` input and even `C`.
2. Split at `D=C//2`.
3. L2-normalize L and H with `torch.nn.functional.normalize(..., eps=1e-6)`.
4. For each view, construct the leave-one-out prototype from the normalized
   sum of the other views, then normalize it.
5. Average L and H cosine agreement with coefficients exactly `0.5` and
   `0.5`.
6. Call `residual_to_uniform_weights()`.
7. Apply the same `[B,K,N,1]` weight to the full concatenated `[L,H]`
   feature and sum over K.
8. K=1 returns the sole feature with score zero, weight one, entropy zero,
   and no fallback.

- [ ] **Step 8: Run all naive tests and verify GREEN**

Run:

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_correspondence_aggregation.py -q
```

Expected: all current tests pass without warnings.

- [ ] **Step 9: Write a failing chunk-equivalence test**

Use a seeded small tensor and non-divisible chunk size:

```python
def test_chunked_aggregation_matches_naive_reference():
    generator = torch.Generator().manual_seed(20260731)
    stacked = torch.randn(2, 4, 11, 8, generator=generator)
    expected, expected_diag = aggregate_consensus_projection_naive(
        stacked, alpha=0.35, temperature=0.17
    )
    actual, actual_diag = aggregate_consensus_projection(
        [stacked[:, i].clone() for i in range(4)],
        alpha=0.35,
        temperature=0.17,
        chunk_size=3,
        compute_device="cpu",
        output_device="cpu",
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_diag.scores, expected_diag.scores)
    torch.testing.assert_close(actual_diag.weights, expected_diag.weights)
    torch.testing.assert_close(actual_diag.entropy, expected_diag.entropy)
    assert torch.equal(actual_diag.fallback_mask, expected_diag.fallback_mask)
```

- [ ] **Step 10: Run the chunk test and verify RED**

Run:

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_correspondence_aggregation.py \
  -k chunked -q
```

Expected: failure because `aggregate_consensus_projection()` is missing.

- [ ] **Step 11: Implement sequence-based chunk aggregation**

Do not stack full view tensors. Validate the sequence once, allocate the
fused output on `output_device`, and allocate diagnostics on CPU. For each
`[start:end]` voxel chunk:

1. Move only that slice from each view to `compute_device`.
2. Stack the K small slices and call the naive reference.
3. Copy the fused chunk to `output_device`.
4. Copy score/weight/entropy/fallback chunks to CPU.

Preserve the input projection dtype in the fused result. The diagnostic
score and weight tensors are FP32. Reject `chunk_size < 1`.

- [ ] **Step 12: Run focused and full tests**

Run:

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_correspondence_aggregation.py -q
```

Then:

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview -q
```

Expected: aggregation tests and the existing multi-view suite pass.

- [ ] **Step 13: Commit Task 1**

```bash
git add \
  pixal3d/experiments/__init__.py \
  pixal3d/experiments/correspondence/__init__.py \
  pixal3d/experiments/correspondence/aggregation.py \
  tests/multiview/test_correspondence_aggregation.py
git commit -m "feat: add chunked multiview consensus aggregation"
```

---

### Task 2: Integrate explicit aggregation policy into the conditioner

**Files:**
- Modify: `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py:481-530`
- Modify: `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py:689-727`
- Modify: `tests/multiview/test_conditioner.py`

**Interfaces:**
- Consumes from Task 1:
  - `aggregate_consensus_projection(view_features, *, alpha, temperature, chunk_size, compute_device, output_device)`
  - `ProjectionAggregationDiagnostics`
- Produces:
  - Constructor option `multiview_aggregation: Optional[Mapping[str, object]] = None`
  - Read-only latest diagnostics attribute
    `last_multiview_aggregation_diagnostics: Optional[ProjectionAggregationDiagnostics]`
- Accepted policy mappings:
  - absent/`None` or `{"mode": "equal_mean"}`
  - `{"mode": "consensus", "alpha": float, "temperature": float,
     "chunk_size": int, "cache_device": "cpu"}`

- [ ] **Step 1: Write failing policy-validation tests**

Extend the harness so its `__init__` sets
`self.multiview_aggregation = None` and
`self.last_multiview_aggregation_diagnostics = None`.

Add tests proving:

- missing mode is rejected when a mapping is supplied;
- unsupported mode is rejected;
- consensus requires explicit `alpha`, `temperature`, and `chunk_size`;
- only `cache_device="cpu"` is accepted by the first prototype;
- invalid mapping values fail before any view extraction.

Use a harness counter in `_forward_single_view()` to prove invalid policy
extracts zero views.

- [ ] **Step 2: Run validation tests and verify RED**

Run:

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_conditioner.py \
  -k "aggregation_policy" -q
```

Expected: failures because policy parsing does not exist.

- [ ] **Step 3: Add constructor state and a private policy parser**

Add the optional constructor argument without changing existing config
requirements. Store a plain copied mapping rather than an `nn.Module` or
`nn.Parameter`. Add a private parser that returns one of:

```python
("equal_mean", None)
("consensus", {
    "alpha": validated_float,
    "temperature": validated_float,
    "chunk_size": validated_int,
    "cache_device": "cpu",
})
```

Use Task 1 validation rules for alpha, temperature, and chunk size. Parsing
must not load DINO/NAF or alter the model state dict.

- [ ] **Step 4: Run validation tests and verify GREEN**

Run the Step 2 command.

Expected: all selected validation tests pass.

- [ ] **Step 5: Write failing exact-bypass regression tests**

Add tests that monkeypatch `_online_mean_tensor_groups` and assert it
receives the original one-shot view generator in all three cases:

1. policy absent/default;
2. `{"mode": "equal_mean"}`;
3. consensus with `alpha=0`.

Add K=1 consensus as a fourth exact bypass. Assert
`last_multiview_aggregation_diagnostics is None` for every bypass.

- [ ] **Step 6: Run bypass tests and verify RED**

Run:

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_conditioner.py \
  -k "exact_bypass" -q
```

Expected: at least the configured consensus cases fail because all policies
currently ignore the configuration.

- [ ] **Step 7: Implement the exact fast path**

In `_forward_multiview()`, parse the policy after camera validation and
projection-matrix construction. If mode is equal mean, alpha is zero, or
K is one:

1. set latest diagnostics to `None`;
2. build the existing generator expression unchanged;
3. return `_online_mean_tensor_groups(view_features)`.

Do not materialize a list in this path.

- [ ] **Step 8: Run bypass tests and verify GREEN**

Run the Step 6 command.

Expected: all exact bypass tests pass.

- [ ] **Step 9: Write a failing conditioner consensus test**

Create a test harness whose view projection output has an even channel
width and three agreeing views plus one opposite outlier. Configure:

```python
model.multiview_aggregation = {
    "mode": "consensus",
    "alpha": 1.0,
    "temperature": 0.2,
    "chunk_size": 1,
    "cache_device": "cpu",
}
```

Assert:

- global output equals arithmetic mean;
- projected output favors the agreeing feature;
- final projection shape/dtype/device are preserved;
- diagnostics are on CPU and weights sum exactly to one within FP32
  tolerance;
- the outlier receives lower weight;
- no trainable parameter or state-dict key was added.

- [ ] **Step 10: Run the conditioner consensus test and verify RED**

Run:

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_conditioner.py \
  -k "configured_consensus" -q
```

Expected: failure because non-default conditioner aggregation is missing.

- [ ] **Step 11: Implement CPU-cache/chunk integration**

For non-default consensus:

1. sequentially call `_forward_single_view()`;
2. retain the small per-view global tensors for arithmetic mean;
3. detach and copy each projection tensor to CPU in its original dtype;
4. call Task 1 chunk aggregation with `compute_device=image.device` and
   `output_device=image.device`;
5. compute global arithmetic mean through `_online_mean_tensor_groups()`;
6. store detached CPU diagnostics in
   `last_multiview_aggregation_diagnostics`;
7. return exactly `(z_global, z_proj)`.

Do not call or apply projection `valid_mask`.

- [ ] **Step 12: Run focused and full regression tests**

Run:

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_conditioner.py \
  tests/multiview/test_correspondence_aggregation.py -q
```

Then:

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview -q
```

Expected: all tests pass; existing default lazy-stream test remains green.

- [ ] **Step 13: Commit Task 2**

```bash
git add \
  pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py \
  tests/multiview/test_conditioner.py
git commit -m "feat: integrate multiview aggregation policy"
```

---

### Task 3: Feature-level metric primitives

**Files:**
- Create: `pixal3d/experiments/correspondence/metrics.py`
- Modify: `pixal3d/experiments/correspondence/__init__.py`
- Test: `tests/multiview/test_correspondence_metrics.py`

**Interfaces:**
- Consumes:
  - fused clean/candidate tensors `[B,N,C]`
  - weights `[B,K,N]`
  - optional boolean active mask `[B,N]`
  - optional boolean corrupted-region mask `[B,N]`
  - integer `corrupted_view_index`
- Produces:
  - `feature_cosine_error(candidate, reference, mask=None) -> torch.Tensor`
  - `feature_l2_drift(candidate, reference, mask=None) -> torch.Tensor`
  - `branch_norms(features) -> tuple[torch.Tensor, torch.Tensor]`
  - `weight_diagnostics(weights, *, active_mask=None,
     corrupted_region_mask=None, corrupted_view_index=None) -> dict[str, torch.Tensor]`

- [ ] **Step 1: Write failing metric tests with literal expectations**

Use hand-derived fixtures:

```python
def test_feature_cosine_error_uses_only_selected_voxels():
    reference = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    candidate = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    mask = torch.tensor([[False, True]])
    assert feature_cosine_error(candidate, reference, mask).item() == 1.0


def test_feature_l2_drift_is_mean_voxel_norm():
    reference = torch.zeros(1, 2, 2)
    candidate = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
    assert feature_l2_drift(candidate, reference).item() == 2.5
```

Add tests for L/H branch norm split, all-false mask rejection, shape mismatch,
odd channel rejection, invalid weight normalization, entropy, per-view
means, uniform deviation, and corrupt-view mass inside/outside the supplied
region.

- [ ] **Step 2: Run metric tests and verify RED**

Run:

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_correspondence_metrics.py -q
```

Expected: import failure because `metrics.py` does not exist.

- [ ] **Step 3: Implement metric primitives**

Implement pure tensor functions with these rules:

- validate floating tensors, exact shape agreement, and finite values;
- default mask selects all `[B,N]` positions;
- reject a mask selecting zero positions;
- cosine error is `mean(1 - cosine_similarity)` over selected voxels;
- L2 drift is the mean per-voxel Euclidean norm;
- branch norms return mean per-voxel L2 norm for the first and second channel
  halves;
- weights must be finite `[B,K,N]`, non-negative, and sum to one over K
  within `atol=1e-5`, `rtol=1e-5`;
- weight diagnostics include entropy, normalized entropy
  (`entropy/log(K)`, zero for K=1), mean weight per view, mean absolute
  deviation from `1/K`, and optional corrupt-view mass inside/outside the
  region.

Return scalar metrics in FP32 and per-view means as FP32 tensors.

- [ ] **Step 4: Run focused and full tests**

Run:

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_correspondence_metrics.py \
  tests/multiview/test_correspondence_aggregation.py \
  tests/multiview/test_conditioner.py -q
```

Then:

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add \
  pixal3d/experiments/correspondence/__init__.py \
  pixal3d/experiments/correspondence/metrics.py \
  tests/multiview/test_correspondence_metrics.py
git commit -m "feat: add correspondence feature diagnostics"
```

---

## Plan boundary

This plan intentionally stops before controlled corruption generation,
oracle-mask projection, experiment manifests, checkpoint loading, stage-local
3D generation, and Stage 2 transport. Those are separate plans because they
have independent data/geometry and execution failure modes. Completion of
this plan means the Stage 1 tensor mechanism is implemented, integrated,
unit-tested, and ready to consume deterministic controlled-corruption data.
