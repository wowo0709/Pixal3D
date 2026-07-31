# Multi-View Correspondence Conditioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, inference-only sparse-first aggregation path that can reproduce equal-mean Pixal3D conditioning and compare residual consensus or oracle-mask routing without changing any flow-model weights or condition shapes.

**Architecture:** Extract the existing per-view conditioner iteration into a reusable generator while keeping the default forward path byte-for-byte equivalent at its public boundary. A focused pure-PyTorch aggregation module computes voxel-wise weights over sparse per-view features. The image-to-3D pipeline invokes this path only when an explicit per-stage configuration is supplied; otherwise it retains the trained dense arithmetic-mean path.

**Tech Stack:** Python 3, PyTorch, PIL, NumPy, pytest, existing Pixal3D `SparseTensor`

## Global Constraints

- Work only on `feature/multiview-correspondence-node11`.
- Keep the ordinary inference and all training paths on the existing arithmetic mean.
- Do not change SS-64 conditioning.
- Support experimental pipeline inference only for `B=1`.
- Preserve global condition shape `[1,5,1024]`.
- Preserve sparse projected condition shape `[N_active,2048]`.
- Preserve `[L,H]` channel order and apply one view weight to both 1024-channel halves.
- Keep global CLS/register tokens on arithmetic mean unless the named `projection_weights` diagnostic is explicitly selected.
- Do not add trainable parameters or modify flow, ProjectAttention, DINOv3, or NAF weights.
- Do not hard-mask foreground, projection validity, depth, or occlusion.
- Do not use a mesh proxy, local source search, deformable transport, or test-time model optimization.
- Accumulate half-precision means and weighted sums in FP32, then cast back to the input dtype.
- Fall back to uniform view weights for `K=1`, all-zero oracle reliability, or non-finite confidence.
- Public single-view checkpoints are not substitutes for the pending multi-view checkpoints.

---

## File Structure

- Create `pixal3d/pipelines/projection_aggregation.py`: immutable experiment configuration, pure consensus/oracle weight computation, sparse feature fusion, and diagnostics.
- Modify `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py`: reusable per-view iterator and projection-coordinate API shared by default and experimental paths.
- Modify `pixal3d/pipelines/pixal3d_image_to_3d.py`: B=1 sparse gather, per-stage opt-in routing, oracle-mask sampling, and caller-owned diagnostics.
- Modify `inference.py`: explicit non-oracle CLI controls for mean/consensus experiments.
- Create `tests/multiview/test_projection_aggregation.py`: mathematical unit tests.
- Modify `tests/multiview/test_conditioner.py`: iterator/default-path regression tests.
- Modify `tests/multiview/test_projection_geometry.py`: projection metadata regression tests.
- Modify `tests/multiview/test_pipeline_inputs.py`: sparse-first pipeline contract and stage-routing tests.
- Modify `tests/multiview/test_inference_manifest.py`: CLI and `run_inference` option-forwarding tests.

### Task 1: Pure sparse-view aggregation mathematics

**Files:**

- Create: `pixal3d/pipelines/projection_aggregation.py`
- Create: `tests/multiview/test_projection_aggregation.py`

**Interfaces:**

- Consumes: per-view sparse features `features: Tensor[K,N,C]`, optional projected corruption probabilities `projected_corruption: Tensor[K,N]`.
- Produces: `ProjectionAggregationConfig`, `ProjectionAggregationDiagnostics`, `aggregate_projected_features(...) -> tuple[Tensor[N,C], ProjectionAggregationDiagnostics]`, and `aggregate_global_features(...) -> Tensor[1,T,D]`.

- [ ] **Step 1: Write configuration validation tests**

```python
import pytest

from pixal3d.pipelines.projection_aggregation import (
    ProjectionAggregationConfig,
)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "invalid"}, "mode"),
        ({"alpha": -0.1}, "alpha"),
        ({"alpha": 1.1}, "alpha"),
        ({"temperature": 0.0}, "temperature"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"global_mode": "invalid"}, "global_mode"),
    ],
)
def test_projection_aggregation_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ProjectionAggregationConfig(**kwargs)
```

- [ ] **Step 2: Run the validation test and verify that the missing module fails**

Run:

```bash
pytest tests/multiview/test_projection_aggregation.py::test_projection_aggregation_config_rejects_invalid_values -v
```

Expected: FAIL during collection with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the immutable configuration and diagnostic types**

```python
from dataclasses import dataclass
from typing import Literal, Optional

import torch


AggregationMode = Literal["mean", "consensus", "oracle"]
GlobalAggregationMode = Literal["mean", "projection_weights"]


@dataclass(frozen=True)
class ProjectionAggregationConfig:
    mode: AggregationMode = "mean"
    alpha: float = 0.5
    temperature: float = 0.1
    chunk_size: int = 4096
    global_mode: GlobalAggregationMode = "mean"

    def __post_init__(self) -> None:
        if self.mode not in {"mean", "consensus", "oracle"}:
            raise ValueError("mode must be mean, consensus, or oracle")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.global_mode not in {"mean", "projection_weights"}:
            raise ValueError("global_mode must be mean or projection_weights")


@dataclass(frozen=True)
class ProjectionAggregationDiagnostics:
    scores: Optional[torch.Tensor]
    weights: torch.Tensor
    projected_corruption: Optional[torch.Tensor]
```

- [ ] **Step 4: Run the validation test and verify that it passes**

Run:

```bash
pytest tests/multiview/test_projection_aggregation.py::test_projection_aggregation_config_rejects_invalid_values -v
```

Expected: PASS.

- [ ] **Step 5: Write failing tests for equal mean, consensus, and fallback behavior**

```python
import torch

from pixal3d.pipelines.projection_aggregation import (
    ProjectionAggregationConfig,
    aggregate_projected_features,
)


def test_mean_matches_fp32_reference_and_preserves_dtype():
    features = torch.tensor(
        [
            [[1.0, 3.0, 10.0, 12.0], [2.0, 4.0, 20.0, 22.0]],
            [[5.0, 7.0, 14.0, 16.0], [6.0, 8.0, 24.0, 26.0]],
        ],
        dtype=torch.bfloat16,
    )
    fused, diagnostics = aggregate_projected_features(
        features,
        ProjectionAggregationConfig(mode="mean"),
    )
    expected = features.float().mean(dim=0).to(torch.bfloat16)
    assert torch.equal(fused, expected)
    assert fused.dtype == features.dtype
    torch.testing.assert_close(
        diagnostics.weights,
        torch.full((2, 2), 0.5),
        rtol=0,
        atol=0,
    )


def test_alpha_zero_consensus_is_exact_equal_mean():
    generator = torch.Generator().manual_seed(20260731)
    features = torch.randn(4, 7, 8, generator=generator).to(torch.bfloat16)
    mean, _ = aggregate_projected_features(
        features,
        ProjectionAggregationConfig(mode="mean"),
    )
    residual, _ = aggregate_projected_features(
        features,
        ProjectionAggregationConfig(
            mode="consensus", alpha=0.0, temperature=0.2, chunk_size=3
        ),
    )
    assert torch.equal(residual, mean)


def test_alpha_zero_oracle_is_exact_equal_mean():
    generator = torch.Generator().manual_seed(20260731)
    features = torch.randn(4, 7, 8, generator=generator).to(torch.bfloat16)
    corruption = torch.randint(
        0, 2, (4, 7), generator=generator
    ).float()
    mean, _ = aggregate_projected_features(
        features,
        ProjectionAggregationConfig(mode="mean"),
    )
    residual, _ = aggregate_projected_features(
        features,
        ProjectionAggregationConfig(mode="oracle", alpha=0.0, chunk_size=3),
        projected_corruption=corruption,
    )
    assert torch.equal(residual, mean)


def test_consensus_downweights_one_outlier_for_both_feature_halves():
    inlier = torch.tensor([1.0, 0.0, 1.0, 0.0])
    outlier = torch.tensor([0.0, 1.0, 0.0, 1.0])
    features = torch.stack([inlier, inlier, inlier, outlier])[:, None, :]
    _, diagnostics = aggregate_projected_features(
        features,
        ProjectionAggregationConfig(
            mode="consensus", alpha=1.0, temperature=0.05
        ),
    )
    assert diagnostics.weights[3, 0] < diagnostics.weights[:3, 0].min()
    torch.testing.assert_close(
        diagnostics.weights[:, 0].sum(),
        torch.tensor(1.0),
    )


def test_identical_and_k1_features_use_uniform_weights():
    identical = torch.ones(4, 3, 8)
    _, repeated = aggregate_projected_features(
        identical,
        ProjectionAggregationConfig(mode="consensus", alpha=1.0),
    )
    torch.testing.assert_close(repeated.weights, torch.full((4, 3), 0.25))

    one_view = torch.randn(1, 3, 8)
    fused, single = aggregate_projected_features(
        one_view,
        ProjectionAggregationConfig(mode="consensus", alpha=1.0),
    )
    assert torch.equal(fused, one_view[0])
    assert torch.equal(single.weights, torch.ones(1, 3))


def test_nonfinite_consensus_scores_fall_back_to_uniform_weights():
    features = torch.ones(4, 2, 8)
    features[3, :, 0] = float("nan")
    _, diagnostics = aggregate_projected_features(
        features,
        ProjectionAggregationConfig(mode="consensus", alpha=1.0),
    )
    torch.testing.assert_close(
        diagnostics.weights, torch.full((4, 2), 0.25)
    )
```

- [ ] **Step 6: Run the new mathematical tests and verify that the missing functions fail**

Run:

```bash
pytest tests/multiview/test_projection_aggregation.py -v
```

Expected: FAIL because `aggregate_projected_features` is not defined.

- [ ] **Step 7: Implement chunked leave-one-out consensus and FP32 weighted fusion**

```python
def _uniform_weights(
    num_views: int,
    num_tokens: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.full(
        (num_views, num_tokens),
        1.0 / num_views,
        dtype=torch.float32,
        device=device,
    )


def _safe_normalize(features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    norms = torch.linalg.vector_norm(features, dim=-1, keepdim=True)
    return features / norms.clamp_min(eps)


def _consensus_scores(features: torch.Tensor) -> torch.Tensor:
    if features.shape[-1] % 2:
        raise ValueError("projected feature width must split evenly into L/H")
    split = features.shape[-1] // 2
    low = _safe_normalize(features[..., :split].float())
    high = _safe_normalize(features[..., split:].float())

    def branch_scores(branch: torch.Tensor) -> torch.Tensor:
        prototype = _safe_normalize(branch.sum(dim=0, keepdim=True) - branch)
        return (branch * prototype).sum(dim=-1)

    scores = 0.5 * branch_scores(low) + 0.5 * branch_scores(high)
    return torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))


def _residual_weights(
    routing: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    uniform = torch.full_like(routing, 1.0 / routing.shape[0])
    return (1.0 - alpha) * uniform + alpha * routing


def aggregate_projected_features(
    features: torch.Tensor,
    config: ProjectionAggregationConfig,
    *,
    projected_corruption: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, ProjectionAggregationDiagnostics]:
    if features.ndim != 3:
        raise ValueError("features must have shape [K, N, C]")
    num_views, num_tokens, _ = features.shape
    if num_views < 1:
        raise ValueError("at least one view is required")
    if config.mode == "oracle":
        if projected_corruption is None:
            raise ValueError("oracle mode requires projected_corruption")
        if projected_corruption.shape != (num_views, num_tokens):
            raise ValueError("projected_corruption must have shape [K, N]")

    fused_chunks = []
    score_chunks = []
    weight_chunks = []
    for start in range(0, num_tokens, config.chunk_size):
        stop = min(start + config.chunk_size, num_tokens)
        chunk = features[:, start:stop]
        uniform = _uniform_weights(
            num_views, stop - start, device=features.device
        )
        scores = None
        exact_mean = (
            num_views == 1
            or config.mode == "mean"
            or config.alpha == 0.0
        )
        if exact_mean:
            weights = uniform
        elif config.mode == "consensus":
            scores = _consensus_scores(chunk)
            routing = torch.softmax(scores / config.temperature, dim=0)
            finite = torch.isfinite(routing).all(dim=0, keepdim=True)
            routing = torch.where(finite, routing, uniform)
            weights = _residual_weights(routing, config.alpha)
        else:
            reliability = (
                1.0 - projected_corruption[:, start:stop].float()
            ).clamp(0.0, 1.0)
            denominator = reliability.sum(dim=0, keepdim=True)
            routing = reliability / denominator.clamp_min(1e-6)
            routing = torch.where(denominator > 0, routing, uniform)
            weights = _residual_weights(routing, config.alpha)
        fused = (
            chunk.float().mean(dim=0)
            if exact_mean
            else (chunk.float() * weights[..., None]).sum(dim=0)
        )
        fused_chunks.append(fused.to(features.dtype))
        weight_chunks.append(weights)
        if scores is not None:
            score_chunks.append(scores)

    diagnostics = ProjectionAggregationDiagnostics(
        scores=torch.cat(score_chunks, dim=1) if score_chunks else None,
        weights=torch.cat(weight_chunks, dim=1),
        projected_corruption=projected_corruption,
    )
    return torch.cat(fused_chunks, dim=0), diagnostics
```

- [ ] **Step 8: Add global mean and explicit projection-weight diagnostic tests**

```python
from pixal3d.pipelines.projection_aggregation import aggregate_global_features


def test_global_features_default_to_fp32_arithmetic_mean():
    features = torch.arange(4 * 5 * 6).reshape(4, 5, 6).to(torch.bfloat16)
    weights = torch.tensor(
        [[0.7, 0.7], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]
    )
    actual = aggregate_global_features(features, weights, mode="mean")
    expected = features.float().mean(dim=0, keepdim=True).to(torch.bfloat16)
    assert torch.equal(actual, expected)


def test_projection_weight_global_mode_reduces_voxel_weights_to_view_weights():
    features = torch.tensor([[[1.0]], [[3.0]]])
    weights = torch.tensor([[0.75, 0.75], [0.25, 0.25]])
    actual = aggregate_global_features(
        features, weights, mode="projection_weights"
    )
    torch.testing.assert_close(actual, torch.tensor([[[1.5]]]))
```

- [ ] **Step 9: Implement global aggregation and run all aggregation tests**

```python
def aggregate_global_features(
    features: torch.Tensor,
    projection_weights: torch.Tensor,
    *,
    mode: GlobalAggregationMode,
) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError("global features must have shape [K, T, D]")
    if projection_weights.shape[0] != features.shape[0]:
        raise ValueError("global and projected view counts must match")
    if mode == "mean":
        fused = features.float().mean(dim=0, keepdim=True)
    else:
        view_weights = projection_weights.float().mean(dim=1)
        denominator = view_weights.sum()
        if not torch.isfinite(denominator) or denominator <= 0:
            view_weights = torch.full_like(
                view_weights, 1.0 / view_weights.numel()
            )
        else:
            view_weights = view_weights / denominator
        fused = (
            features.float() * view_weights[:, None, None]
        ).sum(dim=0, keepdim=True)
    return fused.to(features.dtype)
```

Run:

```bash
pytest tests/multiview/test_projection_aggregation.py -v
```

Expected: PASS.

- [ ] **Step 10: Commit the pure aggregation unit**

```bash
git add pixal3d/pipelines/projection_aggregation.py tests/multiview/test_projection_aggregation.py
git commit -m "feat: add sparse projection aggregation math"
```

### Task 2: Reusable per-view conditioner stream

**Files:**

- Modify: `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py:689-727`
- Modify: `tests/multiview/test_conditioner.py`

**Interfaces:**

- Consumes: the current single- or multi-view conditioner tensors and cameras.
- Produces: `DinoV3ProjFeatureExtractor.iter_view_features(...) -> Iterator[tuple[Tensor, Tensor]]`; the existing `_forward_multiview` consumes this iterator with `_online_mean_tensor_groups`.

- [ ] **Step 1: Write failing iterator equivalence and laziness tests**

```python
def test_iter_view_features_matches_current_per_view_projection_order():
    model = ConditionerHarness()
    image = torch.arange(48, dtype=torch.float32).reshape(1, 4, 3, 2, 2)
    camera = cameras(4)

    groups = list(model.iter_view_features(image, **camera))

    assert len(groups) == 4
    default_global, default_projected = model(image, **camera)
    expected_global = torch.stack([group[0] for group in groups]).mean(dim=0)
    expected_projected = torch.stack([group[1] for group in groups]).mean(dim=0)
    torch.testing.assert_close(default_global, expected_global)
    torch.testing.assert_close(default_projected, expected_projected)


def test_iter_view_features_supports_uncalibrated_k1_without_transform():
    model = ConditionerHarness()
    image = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    groups = list(
        model.iter_view_features(
            image,
            camera_angle_x=torch.tensor([0.7]),
            distance=torch.tensor([2.5]),
            mesh_scale=torch.ones(1),
            transform_matrix=None,
        )
    )
    assert len(groups) == 1
    expected = model._forward_single_view(
        image, torch.tensor([0.7]), torch.tensor([2.5]), torch.ones(1), None
    )
    for actual, reference in zip(groups[0], expected):
        torch.testing.assert_close(actual, reference)
```

- [ ] **Step 2: Run the new tests and verify the missing iterator failure**

Run:

```bash
pytest tests/multiview/test_conditioner.py -k "iter_view_features" -v
```

Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Extract validation and iteration without changing public forward behavior**

```python
def iter_view_features(
    self,
    image: torch.Tensor,
    camera_angle_x: torch.Tensor,
    distance: torch.Tensor,
    mesh_scale: torch.Tensor,
    transform_matrix: Optional[torch.Tensor],
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    if image.ndim == 4:
        yield self._forward_single_view(
            image, camera_angle_x, distance, mesh_scale, transform_matrix
        )
        return
    if image.ndim != 5:
        raise ValueError("image must have shape [B,C,H,W] or [B,K,C,H,W]")
    batch_size, num_views = image.shape[:2]
    expected_vector = (batch_size, num_views)
    if camera_angle_x is None or camera_angle_x.shape != expected_vector:
        raise ValueError("camera_angle_x must have shape [B, K]")
    if distance is None or distance.shape != expected_vector:
        raise ValueError("distance must have shape [B, K]")
    if mesh_scale is None or mesh_scale.shape != (batch_size,):
        raise ValueError("mesh_scale must have shape [B]")
    if transform_matrix is None:
        if num_views != 1:
            raise ValueError("transform_matrix must have shape [B, K, 4, 4]")
        yield self._forward_single_view(
            image[:, 0],
            camera_angle_x[:, 0],
            distance[:, 0],
            mesh_scale,
            None,
        )
        return
    if transform_matrix.shape != (batch_size, num_views, 4, 4):
        raise ValueError("transform_matrix must have shape [B, K, 4, 4]")
    if not torch.isfinite(mesh_scale).all() or torch.any(mesh_scale <= 0):
        raise ValueError("mesh_scale must be finite and positive")

    projection, _ = compute_multiview_projection_matrices(
        transform_matrix, distance, self.fixed_projection_transform
    )
    for view_index in range(num_views):
        yield self._forward_single_view(
            image[:, view_index],
            camera_angle_x[:, view_index],
            distance[:, view_index],
            mesh_scale,
            projection[:, view_index],
        )
```

Replace `_forward_multiview`'s local generator with:

```python
return _online_mean_tensor_groups(
    self.iter_view_features(
        image,
        camera_angle_x,
        distance,
        mesh_scale,
        transform_matrix,
    )
)
```

- [ ] **Step 4: Run conditioner and online-mean regressions**

Run:

```bash
pytest tests/multiview/test_conditioner.py tests/multiview/test_online_mean.py -v
```

Expected: all tests PASS, including exact K=1 and lazy-stream tests.

- [ ] **Step 5: Commit the per-view stream**

```bash
git add pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py tests/multiview/test_conditioner.py
git commit -m "refactor: expose multiview conditioner stream"
```

### Task 3: Projection metadata for active sparse coordinates

**Files:**

- Modify: `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py:259-349`
- Modify: `tests/multiview/test_projection_geometry.py`

**Interfaces:**

- Consumes: optional flattened grid indices and the exact camera arguments used for feature projection.
- Produces: `ProjGrid.project_grid_points(...) -> tuple[pixel_xy, depth, valid_mask, ndc_xy]`; `ProjGrid.forward` samples from its returned NDC coordinates.

- [ ] **Step 1: Write failing full-grid and indexed-grid equivalence tests**

```python
def test_project_grid_points_indexed_subset_matches_full_grid():
    grid = ProjGrid(grid_resolution=3, image_resolution=12)
    fov = torch.tensor([0.7])
    distance = torch.tensor([2.5])
    scale = torch.tensor([1.0])
    indices = torch.tensor([0, 4, 26])

    full = grid.project_grid_points(fov, distance, scale)
    subset = grid.project_grid_points(
        fov, distance, scale, point_indices=indices
    )

    for subset_value, full_value in zip(subset, full):
        torch.testing.assert_close(subset_value, full_value[:, indices])


def test_proj_grid_forward_uses_public_projection_ndc():
    grid = ProjGrid(grid_resolution=2, image_resolution=8)
    features = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)
    fov = torch.tensor([0.7])
    distance = torch.tensor([2.5])
    scale = torch.tensor([1.0])
    _, _, _, ndc = grid.project_grid_points(fov, distance, scale)
    expected = sample_features(
        features.permute(0, 3, 1, 2), ndc
    ).permute(0, 2, 1)
    actual = grid(features, fov, distance, scale)
    torch.testing.assert_close(actual, expected)
```

- [ ] **Step 2: Run the projection tests and verify the missing method failure**

Run:

```bash
pytest tests/multiview/test_projection_geometry.py -k "project_grid_points" -v
```

Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement the shared projection method and delegate `forward` to it**

```python
def project_grid_points(
    self,
    camera_angle_x: torch.Tensor,
    distance: torch.Tensor,
    mesh_scale: torch.Tensor,
    transform_matrix: Optional[torch.Tensor] = None,
    *,
    point_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = camera_angle_x.shape[0]
    grid_points = self.grid_points
    if point_indices is not None:
        grid_points = grid_points[point_indices.long()]
    grid_points = grid_points.expand(batch_size, -1, -1)
    grid_points = grid_points / mesh_scale[:, None, None] / 2
    if transform_matrix is None:
        transform_matrix = self.front_view_transform_matrix.expand(
            batch_size, -1, -1
        ).clone()
        transform_matrix[:, 1, 3] = -distance
    elif (
        transform_matrix.shape != (batch_size, 4, 4)
        or not torch.isfinite(transform_matrix).all()
    ):
        raise ValueError(
            "transform_matrix must be finite with shape [B, 4, 4]"
        )
    pixel_xy, depth, valid = project_points_to_image_batch(
        grid_points,
        transform_matrix,
        camera_angle_x,
        self.image_resolution,
    )
    ndc_xy = (pixel_xy + 0.5) / self.image_resolution * 2 - 1
    return pixel_xy, depth, valid, ndc_xy
```

In `ProjGrid.forward`, replace duplicated point scaling and projection with:

```python
_, _, _, image_points_norm = self.project_grid_points(
    camera_angle_x,
    distance,
    mesh_scale,
    transform_matrix,
)
```

- [ ] **Step 4: Run all projection and conditioner tests**

Run:

```bash
pytest tests/multiview/test_projection_geometry.py tests/multiview/test_conditioner.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit projection metadata support**

```bash
git add pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py tests/multiview/test_projection_geometry.py
git commit -m "feat: expose projection grid metadata"
```

### Task 4: Sparse-first pipeline condition builder

**Files:**

- Modify: `pixal3d/pipelines/pixal3d_image_to_3d.py:307-394`
- Modify: `tests/multiview/test_pipeline_inputs.py`

**Interfaces:**

- Consumes: `ProjectionAggregationConfig | None`, optional oracle masks `[K,1,H,W]`, and optional caller-owned diagnostics dictionary.
- Produces: the unchanged condition dictionary with `global: Tensor[1,T,D]` and `proj: SparseTensor[N,C]`.

- [ ] **Step 1: Extend the recording conditioner and write a failing explicit-mean equivalence test**

Replace `RecordingGrid` usage in the test harness with the real projection grid and add a deterministic per-view stream:

```python
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    ProjGrid,
)


class RecordingConditioner(torch.nn.Module):
    def __init__(self, image_size=8, grid_resolution=2):
        super().__init__()
        self.image_size = image_size
        self.grid_resolution = grid_resolution
        self.proj_grid = ProjGrid(grid_resolution, image_size)
        self.calls = []

    @property
    def fixed_projection_transform(self):
        return self.proj_grid.front_view_transform_matrix

    def iter_view_features(self, image, **camera):
        num_views = image.shape[1] if image.ndim == 5 else 1
        for view_index in range(num_views):
            value = float(view_index + 1)
            yield (
                torch.full((1, 5, 4), value, device=image.device),
                torch.full(
                    (1, self.grid_resolution ** 3, 4),
                    value,
                    device=image.device,
                ),
            )

    def forward(self, image, **camera):
        self.calls.append(
            {
                "image": image.detach().clone(),
                "camera_angle_x": camera["camera_angle_x"].detach().clone(),
                "distance": camera["distance"].detach().clone(),
                "mesh_scale": camera["mesh_scale"].detach().clone(),
                "transform_matrix": (
                    None
                    if camera["transform_matrix"] is None
                    else camera["transform_matrix"].detach().clone()
                ),
            }
        )
        groups = list(self.iter_view_features(image, **camera))
        return tuple(
            torch.stack([group[index] for group in groups])
            .float()
            .mean(dim=0)
            for index in range(2)
        )
```

Then add:

```python
from pixal3d.pipelines.projection_aggregation import (
    ProjectionAggregationConfig,
)


def test_sparse_first_explicit_mean_matches_default_dense_mean():
    pipeline = Pixal3DImageTo3DPipeline()
    pipeline._device = "cpu"
    pipeline.low_vram = False
    conditioner = RecordingConditioner(grid_resolution=2)
    images = [image("red"), image("blue")]
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 1, 1]], dtype=torch.int32
    )
    transforms = torch.eye(4).repeat(1, 2, 1, 1)
    cameras = {
        "camera_angle_x": torch.tensor([[0.7, 0.8]]),
        "distance": torch.tensor([[2.5, 2.7]]),
        "mesh_scale": torch.tensor([1.0]),
        "transform_matrix": transforms,
    }

    default = pipeline.get_proj_cond_shape(
        conditioner, images, coords, **cameras
    )
    experimental = pipeline.get_proj_cond_shape(
        conditioner,
        images,
        coords,
        aggregation_config=ProjectionAggregationConfig(mode="mean"),
        **cameras,
    )

    torch.testing.assert_close(
        experimental["cond"]["global"], default["cond"]["global"]
    )
    torch.testing.assert_close(
        experimental["cond"]["proj"].feats,
        default["cond"]["proj"].feats,
    )
    assert experimental["cond"]["proj"].feats.shape == (2, 4)
    assert torch.equal(
        experimental["neg_cond"]["proj"].coords,
        coords,
    )
    assert torch.count_nonzero(
        experimental["neg_cond"]["proj"].feats
    ) == 0
```

- [ ] **Step 2: Write B=1, shape, negative-condition, and restore-on-error tests**

```python
def test_sparse_first_rejects_nonzero_sparse_batch_indices():
    pipeline = Pixal3DImageTo3DPipeline()
    pipeline._device = "cpu"
    pipeline.low_vram = False
    conditioner = RecordingConditioner()
    coords = torch.tensor([[1, 0, 0, 0]], dtype=torch.int32)
    with pytest.raises(ValueError, match="B=1"):
        pipeline.get_proj_cond_shape(
            conditioner,
            [image("red")],
            coords,
            0.7,
            2.5,
            1.0,
            aggregation_config=ProjectionAggregationConfig(mode="mean"),
        )


def test_sparse_first_restores_grid_override_after_iterator_error():
    pipeline = Pixal3DImageTo3DPipeline()
    pipeline._device = "cpu"
    pipeline.low_vram = False
    conditioner = FailingIteratorConditioner()
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="synthetic iterator failure"):
        pipeline.get_proj_cond_shape(
            conditioner,
            [image("red")],
            coords,
            0.7,
            2.5,
            1.0,
            grid_resolution_override=3,
            aggregation_config=ProjectionAggregationConfig(mode="mean"),
        )
    assert conditioner.grid_resolution == 2
    assert conditioner.proj_grid.grid_resolution == 2
```

Define the deterministic failure harness directly above the test:

```python
class FailingIteratorConditioner(RecordingConditioner):
    def iter_view_features(self, *args, **kwargs):
        raise RuntimeError("synthetic iterator failure")
        yield
```

- [ ] **Step 3: Run the pipeline tests and verify that the new argument fails**

Run:

```bash
pytest tests/multiview/test_pipeline_inputs.py -k "sparse_first" -v
```

Expected: FAIL because `get_proj_cond_shape` does not accept `aggregation_config`.

- [ ] **Step 4: Add flat sparse-index validation and gather helpers**

In `pixal3d_image_to_3d.py`:

```python
def _flat_sparse_indices(coords: torch.Tensor, grid_resolution: int) -> torch.Tensor:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("coords must have shape [N, 4]")
    if torch.any(coords[:, 0] != 0):
        raise ValueError("experimental projection aggregation supports B=1")
    xyz = coords[:, 1:].long()
    if torch.any(xyz < 0) or torch.any(xyz >= grid_resolution):
        raise ValueError("sparse coordinates are outside the projection grid")
    return (
        xyz[:, 0] * grid_resolution * grid_resolution
        + xyz[:, 1] * grid_resolution
        + xyz[:, 2]
    )


def _gather_sparse_projection(
    dense: torch.Tensor,
    flat_indices: torch.Tensor,
) -> torch.Tensor:
    if dense.ndim != 3 or dense.shape[0] != 1:
        raise ValueError("experimental projection aggregation supports B=1")
    return dense[0, flat_indices]
```

- [ ] **Step 5: Add the opt-in sparse-first branch with guaranteed grid restoration**

At the start of `get_proj_cond_shape`, preserve the current body as the `aggregation_config is None` branch. For the experimental branch:

```python
view_global = []
per_view_sparse = None
projected_corruption = None
original_proj_grid = image_cond_model.proj_grid
try:
    flat_indices = _flat_sparse_indices(coords, image_cond_model.grid_resolution)
    for view_index, (global_features, dense_features) in enumerate(
        image_cond_model.iter_view_features(
            image_tensor,
            camera_angle_x,
            distance,
            mesh_scale,
            transform_matrix,
        )
    ):
        view_global.append(global_features[0])
        sparse_features = _gather_sparse_projection(
            dense_features, flat_indices
        )
        if per_view_sparse is None:
            per_view_sparse = torch.empty(
                (
                    num_views,
                    sparse_features.shape[0],
                    sparse_features.shape[1],
                ),
                dtype=sparse_features.dtype,
                device=sparse_features.device,
            )
        per_view_sparse[view_index].copy_(sparse_features)
    if per_view_sparse is None:
        raise ValueError("at least one view is required")
    fused_sparse, aggregation_diagnostics = aggregate_projected_features(
        per_view_sparse,
        aggregation_config,
        projected_corruption=projected_corruption,
    )
    fused_global = aggregate_global_features(
        torch.stack(view_global, dim=0),
        aggregation_diagnostics.weights,
        mode=aggregation_config.global_mode,
    )
finally:
    if grid_resolution_override is not None:
        image_cond_model.grid_resolution = orig_grid_res
        image_cond_model.proj_grid = original_proj_grid
```

Construct positive and negative conditions with the same `SparseTensor` shape as the default branch. If a diagnostics dictionary was passed, populate it with detached CPU `scores`, `weights`, `projected_corruption`, and the active coordinates; do not retain GPU tensors in pipeline state.

- [ ] **Step 6: Run all pipeline input tests**

Run:

```bash
pytest tests/multiview/test_pipeline_inputs.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit the sparse-first condition builder**

```bash
git add pixal3d/pipelines/pixal3d_image_to_3d.py tests/multiview/test_pipeline_inputs.py
git commit -m "feat: add sparse-first projection conditioning"
```

### Task 5: Oracle mask projection and pipeline diagnostics

**Files:**

- Modify: `pixal3d/pipelines/pixal3d_image_to_3d.py`
- Modify: `tests/multiview/test_pipeline_inputs.py`

**Interfaces:**

- Consumes: `oracle_masks: Tensor[K,1,H,W]` where `1` denotes a known corrupted pixel.
- Produces: projected corruption probability `[K,N_active]` sampled with the same projection NDC and border behavior as feature projection; depth and validity are diagnostics only.

- [ ] **Step 1: Write a failing projection-alignment test**

```python
def test_oracle_mask_uses_same_active_projection_coordinates(monkeypatch):
    pipeline = Pixal3DImageTo3DPipeline()
    pipeline._device = "cpu"
    pipeline.low_vram = False
    conditioner = RecordingConditioner(grid_resolution=2)
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    masks = torch.zeros(2, 1, 8, 8)
    masks[1, :, 4:, 4:] = 1.0
    diagnostics = {}
    transforms = torch.eye(4).repeat(1, 2, 1, 1)

    pipeline.get_proj_cond_shape(
        conditioner,
        [image("red"), image("blue")],
        coords,
        camera_angle_x=torch.tensor([[0.7, 0.7]]),
        distance=torch.tensor([[2.5, 2.5]]),
        mesh_scale=torch.tensor([1.0]),
        transform_matrix=transforms,
        aggregation_config=ProjectionAggregationConfig(
            mode="oracle", alpha=1.0
        ),
        oracle_masks=masks,
        diagnostics=diagnostics,
    )

    assert diagnostics["projected_corruption"].shape == (2, 1)
    assert diagnostics["pixel_xy"].shape == (2, 1, 2)
    assert diagnostics["depth"].shape == (2, 1)
    assert diagnostics["valid_mask"].shape == (2, 1)
```

- [ ] **Step 2: Run the test and verify the missing oracle path failure**

Run:

```bash
pytest tests/multiview/test_pipeline_inputs.py::test_oracle_mask_uses_same_active_projection_coordinates -v
```

Expected: FAIL because oracle-mask arguments are not implemented.

- [ ] **Step 3: Implement active-coordinate projection and mask sampling**

Whenever `oracle_masks` is supplied, compute it for diagnostics regardless of aggregation mode; only `mode="oracle"` consumes it when constructing weights. If oracle mode is selected without masks, raise `ValueError("oracle mode requires oracle_masks")`. Use `compute_multiview_projection_matrices`, `ProjGrid.project_grid_points`, and `sample_features` from the conditioner module:

```python
projection, _ = compute_multiview_projection_matrices(
    transform_matrix,
    distance,
    image_cond_model.fixed_projection_transform,
)
pixel_xy, depth, valid_mask, ndc_xy = (
    image_cond_model.proj_grid.project_grid_points(
        camera_angle_x.reshape(-1),
        distance.reshape(-1),
        mesh_scale.expand(num_views),
        projection.reshape(num_views, 4, 4),
        point_indices=flat_indices,
    )
)
masks = oracle_masks.to(device=device, dtype=torch.float32)
if masks.shape[:2] != (num_views, 1):
    raise ValueError("oracle_masks must have shape [K, 1, H, W]")
if masks.shape[-2:] != (
    image_cond_model.proj_grid.image_resolution,
    image_cond_model.proj_grid.image_resolution,
):
    masks = torch.nn.functional.interpolate(
        masks,
        size=(
            image_cond_model.proj_grid.image_resolution,
            image_cond_model.proj_grid.image_resolution,
        ),
        mode="bilinear",
        align_corners=False,
    )
projected_corruption = sample_features(masks, ndc_xy).squeeze(1)
```

For uncalibrated K=1, call `project_grid_points` with `transform_matrix=None`. Do not multiply by `valid_mask`.

Add a regression proving that a diagnostic oracle mask cannot leak into consensus weights:

```python
def test_consensus_oracle_mask_is_diagnostic_only():
    pipeline = Pixal3DImageTo3DPipeline()
    pipeline._device = "cpu"
    pipeline.low_vram = False
    conditioner = RecordingConditioner(grid_resolution=2)
    images = [image("red"), image("blue")]
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    cameras = {
        "camera_angle_x": torch.tensor([[0.7, 0.7]]),
        "distance": torch.tensor([[2.5, 2.5]]),
        "mesh_scale": torch.tensor([1.0]),
        "transform_matrix": torch.eye(4).repeat(1, 2, 1, 1),
    }
    masks = torch.zeros(2, 1, 8, 8)
    masks[1, :, 4:, 4:] = 1.0
    config = ProjectionAggregationConfig(
        mode="consensus", alpha=1.0, temperature=0.1
    )
    without_mask = pipeline.get_proj_cond_shape(
        conditioner,
        images,
        coords,
        aggregation_config=config,
        oracle_masks=None,
        diagnostics={},
        **cameras,
    )
    diagnostics = {}
    with_mask = pipeline.get_proj_cond_shape(
        conditioner,
        images,
        coords,
        aggregation_config=config,
        oracle_masks=masks,
        diagnostics=diagnostics,
        **cameras,
    )
    torch.testing.assert_close(
        with_mask["cond"]["proj"].feats,
        without_mask["cond"]["proj"].feats,
    )
    assert diagnostics["projected_corruption"].shape == (
        len(images), coords.shape[0]
    )
```

- [ ] **Step 4: Add oracle rejection and all-zero fallback assertions**

```python
def test_oracle_routing_rejects_corrupted_view_and_falls_back_if_all_corrupt():
    features = torch.tensor([[[1.0, 1.0]], [[9.0, 9.0]]])
    masks = torch.tensor([[0.0], [1.0]])
    fused, diagnostics = aggregate_projected_features(
        features,
        ProjectionAggregationConfig(mode="oracle", alpha=1.0),
        projected_corruption=masks,
    )
    torch.testing.assert_close(fused, torch.tensor([[1.0, 1.0]]))
    torch.testing.assert_close(
        diagnostics.weights[:, 0], torch.tensor([1.0, 0.0])
    )

    all_corrupt, diagnostics = aggregate_projected_features(
        features,
        ProjectionAggregationConfig(mode="oracle", alpha=1.0),
        projected_corruption=torch.ones(2, 1),
    )
    torch.testing.assert_close(all_corrupt, torch.tensor([[5.0, 5.0]]))
    torch.testing.assert_close(
        diagnostics.weights[:, 0], torch.tensor([0.5, 0.5])
    )
```

- [ ] **Step 5: Run aggregation and pipeline tests**

Run:

```bash
pytest tests/multiview/test_projection_aggregation.py tests/multiview/test_pipeline_inputs.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit oracle routing**

```bash
git add pixal3d/pipelines/pixal3d_image_to_3d.py tests/multiview/test_pipeline_inputs.py tests/multiview/test_projection_aggregation.py
git commit -m "feat: add projected oracle mask routing"
```

### Task 6: Per-stage pipeline routing without SS intervention

**Files:**

- Modify: `pixal3d/pipelines/pixal3d_image_to_3d.py:708-889`
- Modify: `tests/multiview/test_pipeline_inputs.py`

**Interfaces:**

- Consumes: `projection_aggregation: Mapping[str, ProjectionAggregationConfig] | None`, `oracle_masks: Tensor[K,1,H,W] | None`, and `projection_diagnostics: MutableMapping[str, dict] | None`.
- Produces: `_projection_stage_arguments(...)` for exact keys `shape512`, `shape1024`, and `pbr1024`; `run` expands those arguments only into its three `get_proj_cond_shape` calls, while SS continues to use `get_proj_cond_ss`.

- [ ] **Step 1: Write a failing stage-key validation test**

```python
def test_projection_stage_arguments_reject_unknown_stage():
    with pytest.raises(ValueError, match="shape512, shape1024, pbr1024"):
        _projection_stage_arguments(
            {"ss64": ProjectionAggregationConfig(mode="consensus")},
            oracle_masks=None,
            diagnostics=None,
        )
```

- [ ] **Step 2: Write a failing propagation test with a mocked pipeline cascade**

```python
def test_projection_stage_arguments_route_only_named_stages():
    configs = {
        "shape512": ProjectionAggregationConfig(mode="consensus"),
        "pbr1024": ProjectionAggregationConfig(mode="mean"),
    }
    diagnostics = {}
    masks = torch.zeros(4, 1, 8, 8)
    arguments = _projection_stage_arguments(
        configs,
        oracle_masks=masks,
        diagnostics=diagnostics,
    )
    assert arguments["shape512"]["aggregation_config"] is configs["shape512"]
    assert arguments["shape1024"]["aggregation_config"] is None
    assert arguments["pbr1024"]["aggregation_config"] is configs["pbr1024"]
    assert arguments["shape512"]["oracle_masks"] is masks
    assert arguments["shape1024"]["oracle_masks"] is None
    assert set(diagnostics) == {"shape512", "pbr1024"}
```

- [ ] **Step 3: Run the stage-routing tests and verify the signature failure**

Run:

```bash
pytest tests/multiview/test_pipeline_inputs.py -k "projection_stage_arguments" -v
```

Expected: FAIL because `run` does not accept the new arguments.

- [ ] **Step 4: Add exact stage lookup and propagate only to Shape/PBR calls**

```python
def _projection_stage_arguments(configs, *, oracle_masks, diagnostics):
    allowed = {"shape512", "shape1024", "pbr1024"}
    unknown = set(configs or {}) - allowed
    if unknown:
        raise ValueError(
            "projection aggregation stages must be shape512, "
            "shape1024, or pbr1024"
        )
    arguments = {}
    for stage in ("shape512", "shape1024", "pbr1024"):
        config = (configs or {}).get(stage)
        stage_diagnostics = None
        if config is not None and diagnostics is not None:
            stage_diagnostics = diagnostics.setdefault(stage, {})
        arguments[stage] = {
            "aggregation_config": config,
            "oracle_masks": oracle_masks if config is not None else None,
            "diagnostics": stage_diagnostics,
        }
    return arguments
```

Extend `run` with:

```python
projection_aggregation: Optional[
    Mapping[str, ProjectionAggregationConfig]
] = None,
oracle_masks: Optional[torch.Tensor] = None,
projection_diagnostics: Optional[MutableMapping[str, dict]] = None,
```

Build `stage_arguments = _projection_stage_arguments(...)` after normalizing the cameras. At each of the three existing `get_proj_cond_shape` calls, expand `**stage_arguments["shape512"]`, `**stage_arguments["shape1024"]`, or `**stage_arguments["pbr1024"]`. Do not add arguments to `get_proj_cond_ss`.

- [ ] **Step 5: Run all pipeline tests**

Run:

```bash
pytest tests/multiview/test_pipeline_inputs.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit stage-local routing**

```bash
git add pixal3d/pipelines/pixal3d_image_to_3d.py tests/multiview/test_pipeline_inputs.py
git commit -m "feat: route correspondence aggregation by stage"
```

### Task 7: Non-oracle inference CLI controls

**Files:**

- Modify: `inference.py:358-494`
- Modify: `tests/multiview/test_inference_manifest.py`

**Interfaces:**

- Consumes: optional CLI `--projection_aggregation`, `--projection_stages`, `--projection_alpha`, `--projection_temperature`.
- Produces: per-stage `ProjectionAggregationConfig` mapping passed to `pipeline.run`; oracle mode remains available only through controlled experiment tooling because ordinary manifests have no oracle mask.

- [ ] **Step 1: Write failing parser default and explicit-consensus tests**

```python
def test_projection_aggregation_cli_is_opt_in():
    args = build_parser().parse_args(["--image", "input.png"])
    assert args.projection_aggregation is None
    assert args.projection_stages == "shape512,shape1024,pbr1024"
    assert args.projection_alpha == 0.5
    assert args.projection_temperature == 0.1


def test_projection_aggregation_cli_parses_consensus():
    args = build_parser().parse_args(
        [
            "--transforms",
            "transforms.json",
            "--projection_aggregation",
            "consensus",
            "--projection_stages",
            "shape512,pbr1024",
            "--projection_alpha",
            "0.25",
            "--projection_temperature",
            "0.2",
        ]
    )
    assert args.projection_aggregation == "consensus"
    assert args.projection_stages == "shape512,pbr1024"
    assert args.projection_alpha == 0.25
    assert args.projection_temperature == 0.2
```

- [ ] **Step 2: Run parser tests and verify missing arguments**

Run:

```bash
pytest tests/multiview/test_inference_manifest.py -k "projection_aggregation_cli" -v
```

Expected: FAIL with missing namespace attributes.

- [ ] **Step 3: Add parser arguments and a pure stage-config builder**

```python
def build_projection_aggregation_configs(
    *,
    mode,
    stages,
    alpha,
    temperature,
):
    if mode is None:
        return None
    stage_names = [value.strip() for value in stages.split(",") if value.strip()]
    allowed = {"shape512", "shape1024", "pbr1024"}
    if not stage_names or set(stage_names) - allowed:
        raise ValueError(
            "projection stages must be a comma-separated subset of "
            "shape512,shape1024,pbr1024"
        )
    config = ProjectionAggregationConfig(
        mode=mode,
        alpha=alpha,
        temperature=temperature,
    )
    return {stage: config for stage in stage_names}
```

Parser arguments:

```python
parser.add_argument(
    "--projection_aggregation",
    choices=("mean", "consensus"),
    default=None,
)
parser.add_argument(
    "--projection_stages",
    default="shape512,shape1024,pbr1024",
)
parser.add_argument("--projection_alpha", type=float, default=0.5)
parser.add_argument("--projection_temperature", type=float, default=0.1)
```

- [ ] **Step 4: Write and pass an option-forwarding test**

```python
def test_run_inference_forwards_projection_aggregation(tmp_path, monkeypatch):
    pipeline = RecordingPipeline()
    monkeypatch.setattr(inference, "init_pipeline", lambda *args, **kwargs: pipeline)
    monkeypatch.setattr(
        inference.o_voxel.postprocess,
        "to_glb",
        lambda **kwargs: RecordingGlb(),
    )
    image_path = tmp_path / "input.png"
    Image.new("RGBA", (4, 4)).save(image_path)

    inference.run_inference(
        image_path=str(image_path),
        output_path=str(tmp_path / "output.glb"),
        manual_fov=0.7,
        projection_aggregation={
            "shape512": ProjectionAggregationConfig(mode="consensus")
        },
    )

    assert pipeline.run_kwargs["projection_aggregation"]["shape512"].mode == "consensus"
```

Extend `run_inference` with an optional mapping and pass it to `pipeline.run`. Update the existing test harness with this exact method body:

```python
def run(self, image, *, camera_params, **kwargs):
    self.camera_params = camera_params
    self.run_kwargs = kwargs
    mesh = type(
        "Mesh",
        (),
        {
            "vertices": None,
            "faces": None,
            "attrs": None,
            "coords": None,
        },
    )()
    return [mesh], (None, None, 1)
```

- [ ] **Step 5: Run inference manifest tests**

Run:

```bash
pytest tests/multiview/test_inference_manifest.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit CLI support**

```bash
git add inference.py tests/multiview/test_inference_manifest.py
git commit -m "feat: expose consensus inference controls"
```

### Task 8: Full regression and documentation synchronization

**Files:**

- Modify: `docs/CORR_ADAPTER_AUDIT.md`
- Modify: `docs/CORR_ADAPTER_EXPERIMENT_DESIGN.md`

**Interfaces:**

- Consumes: completed implementation and test evidence.
- Produces: exact implemented paths, config defaults, limitations, and verification commands in the audit/design documents.

- [ ] **Step 1: Run focused CPU regressions**

Run:

```bash
pytest \
  tests/multiview/test_projection_aggregation.py \
  tests/multiview/test_conditioner.py \
  tests/multiview/test_online_mean.py \
  tests/multiview/test_projection_geometry.py \
  tests/multiview/test_pipeline_inputs.py \
  tests/multiview/test_inference_manifest.py \
  -v
```

Expected: PASS with no skipped tests introduced by this feature.

- [ ] **Step 2: Run the full multi-view CPU test suite**

Run:

```bash
pytest tests/multiview -q
```

Expected: PASS. Existing environment-dependent skips may remain unchanged.

- [ ] **Step 3: Verify source hygiene and absence of trainable changes**

Run:

```bash
git diff --check
git status --short
rg -n "ProjectionAggregationConfig|iter_view_features|project_grid_points" \
  pixal3d inference.py tests/multiview
```

Expected: no whitespace errors; only intentional files differ; all three interfaces have implementation and test references.

- [ ] **Step 4: Update audit and design documents with implementation evidence**

Record:

- the final function and class names;
- explicit CLI defaults;
- B=1 limitation;
- default-path and explicit-mean regression results;
- focused and full test counts;
- memory behavior measured from retained `[K,N,C]` sparse tensors; and
- the fact that no checkpoint-dependent 3D result has been run.

- [ ] **Step 5: Commit documentation evidence**

```bash
git add docs/CORR_ADAPTER_AUDIT.md docs/CORR_ADAPTER_EXPERIMENT_DESIGN.md
git commit -m "docs: record correspondence conditioning implementation"
```

- [ ] **Step 6: Verify the final branch state**

Run:

```bash
git status --short --branch
git log --oneline --decorate -8
```

Expected: clean worktree on `feature/multiview-correspondence-node11`, with one intentional commit per task.
