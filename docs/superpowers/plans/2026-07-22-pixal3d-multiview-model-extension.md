# Pixal3D Multi-View Model Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the released Pixal3D single-view cascade to calibrated multi-view conditioning and independently fine-tune SS-64, Shape-512, Shape-1024, and PBR-1024 from their matching released weights.

**Architecture:** Preserve the current TRELLIS.2 denoisers, DINOv3/NAF feature paths, projection channel layouts, samplers, latent encoders/decoders, and cascade. Add anchor-relative camera transforms and a five-dimensional conditioner path that runs the unchanged per-view encoder/projection path sequentially and arithmetic-means projected and global features; datasets return eight anchor-first calibrated views and select one batch-wide K from 2 through 6.

**Tech Stack:** Python 3.11, PyTorch 2.8+, CUDA 12.8, torchvision, transformers DINOv3, NAF, safetensors, Pillow, NumPy, pandas, pytest, W&B, and the existing Pixal3D/TRELLIS.2 modules.

## Global Constraints

- Work only in `/root/dev/Pixal3D/.worktrees/multiview-model-extension` on branch `feature/multiview-model-extension`.
- Run Python and pytest with `conda run --no-capture-output -n pixal3d`; the required runtime is Python 3.11, PyTorch 2.8 or newer, and CUDA 12.8.
- Preserve all existing single-view classes, configs, APIs, denoiser parameters, checkpoint keys, `ProjectAttention`, and `SparseProjectAttention`.
- The first condition is the target latent anchor; training targets are only `view00` or `view01`.
- Compute `relative = inverse(anchor_c2w) @ view_c2w`, then `projection = fixed_anchor(distance_anchor) @ relative` in FP32 with autocast disabled.
- Use an arithmetic mean for projected features and DINOv3 CLS/register global tokens; add no trainable fusion, view weights, masks, padding, pose embeddings, visibility/depth/alpha weighting, or new loss.
- Training uses exactly eight available condition views and one uniformly sampled batch-wide K in integer `[2, 6]`; inference accepts K in `[1, 8]`.
- K=1 regression uses `torch.testing.assert_close(rtol=1e-5, atol=1e-5)` in FP32.
- Fine-tune exactly SS-64, Shape-512, Shape-1024, and PBR-1024, each from its matching released single-view checkpoint; do not train SS-32, Shape-256, PBR-256, or PBR-512.
- Checkpoints must load with exact parameter shapes, `unexpected_keys == []`, and no missing state-dict key except `rope_phases`.
- Use only the immutable ABO pilot64 packs under `/root/data2/pixal3d/prepared/qualification/pilot` until audited production handoffs exist; never read an active preprocessing scratch tree.
- Complete Tasks 1-10 and the model-validation gate in Task 11 before enabling W&B code in Task 12. Complete the W&B gate before the training gate in Task 13.
- While preprocessing is active, GPU validation and pilot smoke may use only physical GPU0 after a fresh resource check. Long training waits for node16 and node17 workers to drain and for all Blender/voxel/latent/packing processes to stop.
- Follow TDD for every code change and make one focused commit per task.

## File Map

- `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py`: FP32 anchor-relative matrices, explicit `ProjGrid` transforms, 4D/5D DINOv3 conditioner dispatch, arithmetic means, anchor helpers, and multi-view W&B grids.
- `pixal3d/datasets/components.py`: strict eight-frame condition loading, anchor-first ordering, camera tensors, scale loading, and batch-wide K slicing.
- `pixal3d/datasets/sparse_structure_latent.py`: dense SS multi-view dataset class and collation.
- `pixal3d/datasets/structured_latent_shape.py`: sparse Shape-512/Shape-1024 multi-view dataset class and collation.
- `pixal3d/datasets/structured_latent_svpbr.py`: sparse PBR-1024 multi-view dataset class and collation.
- `pixal3d/datasets/__init__.py`: lazy registrations for the three new dataset classes.
- `pixal3d/trainers/flow_matching/flow_matching.py`: dense snapshot anchor cameras and W&B metadata captions.
- `pixal3d/trainers/flow_matching/sparse_flow_matching.py`: sparse snapshot anchor cameras and W&B metadata captions.
- `pixal3d/trainers/basic.py`: exact `multiview/k` step logging.
- `pixal3d/pipelines/pixal3d_image_to_3d.py`: K=1/K>1 calibrated input normalization and propagation through all four inference flow models.
- `inference.py`: strict calibrated `transforms.json` CLI input while retaining the existing `--image`/MoGe path.
- `train.py`: explicit bounded-smoke override without changing long-training config semantics.
- `scripts/materialize_multiview_pilot.py`: checksum-verified, non-overwriting pilot64 materialization and dataset metadata generation.
- `scripts/materialize_multiview_checkpoints.py`: four released safetensors downloads/conversions without key rewriting.
- `configs/gen/*_proj_multiview_*.json`: four independently initialized fine-tuning configurations.
- `tests/multiview/`: geometry, conditioner, dataset, materialization, configs, checkpoint, pipeline, CLI, trainer, W&B, and opt-in GPU coverage.
- `README.md`: exact validation, W&B, pilot smoke, and long-training runbook.

---

### Task 1: Add FP32 Anchor-Relative Projection Geometry

**Files:**
- Create: `tests/multiview/test_projection_geometry.py`
- Modify: `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py:142-226`

**Interfaces:**
- Consumes: `transform_matrix: Tensor[B,K,4,4]`, `distance: Tensor[B,K]`, and `fixed_transform: Tensor[4,4]`.
- Produces: `compute_multiview_projection_matrices(...) -> tuple[Tensor[B,K,4,4], Tensor[B,K,4,4]]` and an explicit-transform `ProjGrid.forward` path.

- [ ] **Step 1: Write the failing geometry tests**

```python
import pytest
import torch

from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    ProjGrid,
    compute_multiview_projection_matrices,
)


RTOL = 1e-5
ATOL = 1e-5


def test_projection_matrices_match_anchor_relative_formula():
    anchor = torch.eye(4)
    anchor[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    second = torch.eye(4)
    second[:3, 3] = torch.tensor([-2.0, 1.5, 4.0])
    transforms = torch.stack([anchor, second])[None]
    distances = torch.tensor([[3.75, 4.72]])
    fixed = ProjGrid(2, 8).front_view_transform_matrix

    projection, relative = compute_multiview_projection_matrices(
        transforms, distances, fixed
    )

    expected_relative = torch.linalg.inv(anchor) @ transforms[0]
    expected_fixed = fixed.clone()
    expected_fixed[1, 3] = -distances[0, 0]
    torch.testing.assert_close(relative[0], expected_relative, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        projection[0], expected_fixed @ expected_relative, rtol=RTOL, atol=ATOL
    )


def test_anchor_projection_equals_current_fixed_front_view():
    grid = ProjGrid(grid_resolution=2, image_resolution=8)
    transforms = torch.eye(4).reshape(1, 1, 4, 4)
    distances = torch.tensor([[2.5]])
    projection, relative = compute_multiview_projection_matrices(
        transforms, distances, grid.front_view_transform_matrix
    )
    expected = grid.front_view_transform_matrix.clone()
    expected[1, 3] = -2.5
    torch.testing.assert_close(relative, transforms, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(projection[0, 0], expected, rtol=RTOL, atol=ATOL)


def test_proj_grid_default_and_explicit_anchor_paths_match():
    grid = ProjGrid(grid_resolution=2, image_resolution=8)
    features = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)
    fov = torch.tensor([0.7])
    distance = torch.tensor([2.5])
    scale = torch.tensor([1.0])
    explicit = grid.front_view_transform_matrix[None].clone()
    explicit[:, 1, 3] = -distance
    expected = grid(features, fov, distance, scale)
    actual = grid(features, fov, distance, scale, explicit)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    ("transforms", "message"),
    [
        (torch.full((1, 2, 4, 4), float("nan")), "finite"),
        (torch.zeros(1, 2, 4, 4), "invertible"),
    ],
)
def test_projection_rejects_invalid_anchor_camera(transforms, message):
    with pytest.raises(ValueError, match=message):
        compute_multiview_projection_matrices(
            transforms, torch.ones(1, 2), torch.eye(4)
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_projection_geometry.py -q`

Expected: collection fails because `compute_multiview_projection_matrices` does not exist; after the import exists, explicit projection fails at the current `assert transform_matrix is None`.

- [ ] **Step 3: Implement the exact FP32 transform helper**

Add immediately above `ProjGrid`:

```python
def compute_multiview_projection_matrices(
    transform_matrix: torch.Tensor,
    distance: torch.Tensor,
    fixed_transform: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if transform_matrix.ndim != 4 or transform_matrix.shape[-2:] != (4, 4):
        raise ValueError("transform_matrix must have shape [B, K, 4, 4]")
    if distance.shape != transform_matrix.shape[:2]:
        raise ValueError("distance must have shape [B, K]")
    if not torch.isfinite(transform_matrix).all() or not torch.isfinite(distance).all():
        raise ValueError("camera transforms and distances must be finite")
    if not torch.isfinite(fixed_transform).all() or fixed_transform.shape != (4, 4):
        raise ValueError("fixed_transform must be a finite [4, 4] matrix")

    batch_size, num_views = transform_matrix.shape[:2]
    device_type = transform_matrix.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        transforms = transform_matrix.float()
        anchors = transforms[:, 0]
        anchor_inverse, info = torch.linalg.inv_ex(anchors)
        if torch.any(info != 0):
            raise ValueError("anchor transform must be invertible")
        relative = anchor_inverse[:, None] @ transforms
        fixed = fixed_transform.float().expand(batch_size, 4, 4).clone()
        fixed[:, 1, 3] = -distance[:, 0].float()
        projection = fixed[:, None] @ relative
    return projection, relative
```

In `ProjGrid.forward`, remove only `assert transform_matrix is None`. Retain the current default branch, and validate an explicit transform before projection:

```python
if transform_matrix is None:
    transform_matrix = self.front_view_transform_matrix.expand(B, -1, -1).clone()
    transform_matrix[:, 1, 3] = -distance
elif transform_matrix.shape != (B, 4, 4) or not torch.isfinite(transform_matrix).all():
    raise ValueError("transform_matrix must be finite with shape [B, 4, 4]")
```

- [ ] **Step 4: Run the focused tests**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_projection_geometry.py -q`

Expected: `5 passed`.

- [ ] **Step 5: Commit the geometry boundary**

```bash
git add tests/multiview/test_projection_geometry.py pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py
git commit -m "feat: add calibrated multiview projection geometry"
```

---

### Task 2: Extend DINOv3 Projection Conditioning to Five Dimensions

**Files:**
- Create: `tests/multiview/test_conditioner.py`
- Modify: `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py:344-566`

**Interfaces:**
- Consumes: Task 1 geometry, `image[B,K,3,H,W]`, cameras `[B,K]`, anchor scale `[B]`, and transforms `[B,K,4,4]`.
- Produces: unchanged `forward(...) -> (z_global[B,T,D], z_proj[B,R^3,C])` for both 4D and 5D inputs.

- [ ] **Step 1: Write a parameter-free failing conditioner harness**

```python
import pytest
import torch
import torch.nn as nn

from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    DinoV3ProjFeatureExtractor,
)


class ConditionerHarness(DinoV3ProjFeatureExtractor):
    def __init__(self):
        nn.Module.__init__(self)
        self.register_buffer("front", torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]))

    @property
    def fixed_projection_transform(self):
        return self.front

    def _forward_single_view(
        self, image, camera_angle_x, distance, mesh_scale, transform_matrix
    ):
        global_feature = image.mean(dim=(-2, -1))[:, None, :]
        projected = transform_matrix[:, :1, :3].reshape(image.shape[0], 1, 12)
        return global_feature, projected


def cameras(k):
    transforms = torch.eye(4).repeat(1, k, 1, 1)
    transforms[0, :, 0, 3] = torch.arange(k, dtype=torch.float32)
    return {
        "camera_angle_x": torch.full((1, k), 0.7),
        "distance": torch.full((1, k), 2.5),
        "mesh_scale": torch.ones(1),
        "transform_matrix": transforms,
    }


def test_k1_matches_single_view_with_exact_fp32_gate():
    model = ConditionerHarness()
    image = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    camera = cameras(1)
    actual = model(image[:, None], **camera)
    fixed = model.front[None].clone()
    fixed[:, 1, 3] = -2.5
    expected = model._forward_single_view(
        image, camera["camera_angle_x"][:, 0], camera["distance"][:, 0],
        camera["mesh_scale"], fixed,
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-5)


def test_repeated_view_mean_matches_single_view():
    model = ConditionerHarness()
    image = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    repeated = image[:, None].repeat(1, 4, 1, 1, 1)
    camera = cameras(4)
    camera["transform_matrix"][:] = torch.eye(4)
    actual = model(repeated, **camera)
    expected = model(image[:, None], **cameras(1))
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-5)


def test_non_anchor_permutation_preserves_arithmetic_mean():
    model = ConditionerHarness()
    image = torch.arange(48, dtype=torch.float32).reshape(1, 4, 3, 2, 2)
    camera = cameras(4)
    first = model(image, **camera)
    order = torch.tensor([0, 3, 1, 2])
    second = model(
        image[:, order],
        camera_angle_x=camera["camera_angle_x"][:, order],
        distance=camera["distance"][:, order],
        mesh_scale=camera["mesh_scale"],
        transform_matrix=camera["transform_matrix"][:, order],
    )
    torch.testing.assert_close(first[0], second[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(first[1], second[1], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("key", ["camera_angle_x", "distance", "transform_matrix"])
def test_multiview_rejects_misaligned_camera_shapes(key):
    camera = cameras(2)
    camera[key] = camera[key][:, :1]
    with pytest.raises(ValueError, match=key):
        ConditionerHarness()(torch.zeros(1, 2, 3, 2, 2), **camera)


def test_multiview_conditioner_adds_no_trainable_parameter():
    assert [p for p in ConditionerHarness().parameters() if p.requires_grad] == []
```

- [ ] **Step 2: Run the test and verify RED**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_conditioner.py -q`

Expected: the current public `forward` rejects the five-dimensional tensor.

- [ ] **Step 3: Preserve the complete 4D implementation behind a private boundary**

Rename the current `DinoV3ProjFeatureExtractor.forward` method to `_forward_single_view` without changing any statement in its body, and expose:

```python
@property
def fixed_projection_transform(self) -> torch.Tensor:
    return self.proj_grid.front_view_transform_matrix
```

- [ ] **Step 4: Add the sequential 5D branch and public dispatcher**

```python
def _forward_multiview(
    self,
    image: torch.Tensor,
    camera_angle_x: torch.Tensor,
    distance: torch.Tensor,
    mesh_scale: torch.Tensor,
    transform_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if image.ndim != 5:
        raise ValueError("multi-view image must have shape [B, K, C, H, W]")
    batch_size, num_views = image.shape[:2]
    expected_vector = (batch_size, num_views)
    if camera_angle_x is None or camera_angle_x.shape != expected_vector:
        raise ValueError("camera_angle_x must have shape [B, K]")
    if distance is None or distance.shape != expected_vector:
        raise ValueError("distance must have shape [B, K]")
    if mesh_scale is None or mesh_scale.shape != (batch_size,):
        raise ValueError("mesh_scale must have shape [B]")
    if transform_matrix is None or transform_matrix.shape != (
        batch_size, num_views, 4, 4
    ):
        raise ValueError("transform_matrix must have shape [B, K, 4, 4]")
    if not torch.isfinite(mesh_scale).all() or torch.any(mesh_scale <= 0):
        raise ValueError("mesh_scale must be finite and positive")

    projection, _ = compute_multiview_projection_matrices(
        transform_matrix, distance, self.fixed_projection_transform
    )
    global_views = []
    projected_views = []
    for view_index in range(num_views):
        global_feature, projected_feature = self._forward_single_view(
            image[:, view_index],
            camera_angle_x[:, view_index],
            distance[:, view_index],
            mesh_scale,
            projection[:, view_index],
        )
        global_views.append(global_feature)
        projected_views.append(projected_feature)
    return (
        torch.stack(global_views, dim=1).mean(dim=1),
        torch.stack(projected_views, dim=1).mean(dim=1),
    )


def forward(
    self,
    image: Union[torch.Tensor, List[Image.Image]],
    camera_angle_x: Optional[torch.Tensor] = None,
    distance: Optional[torch.Tensor] = None,
    mesh_scale: Optional[torch.Tensor] = None,
    transform_matrix: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(image, torch.Tensor) and image.ndim == 5:
        return self._forward_multiview(
            image, camera_angle_x, distance, mesh_scale, transform_matrix
        )
    return self._forward_single_view(
        image, camera_angle_x, distance, mesh_scale, transform_matrix
    )
```

Do not flatten B×K, cache features, alter NAF, or modify `DinoV3VaeProjFeatureExtractor` in this baseline.

- [ ] **Step 5: Run conditioner and geometry tests**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_conditioner.py tests/multiview/test_projection_geometry.py -q`

Expected: `12 passed`.

- [ ] **Step 6: Commit the conditioner extension**

```bash
git add tests/multiview/test_conditioner.py pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py
git commit -m "feat: average calibrated multiview image features"
```

---

### Task 3: Load Strict Anchor-First Calibrated Conditions

**Files:**
- Create: `tests/multiview/test_dataset_conditions.py`
- Modify: `pixal3d/datasets/components.py:128-290`

**Interfaces:**
- Consumes: the parent dataset's `_current_view_idx`, `_current_latent_dir`, exactly eight render frames, and the target anchor scale JSON.
- Produces: `load_anchor_first_conditions(...) -> dict` with `cond[8,3,H,W]`, camera vectors `[8]`, transforms `[8,4,4]`, and unique anchor-first indices `[8]`; `MultiViewImageConditionedMixin.get_instance` adds scalar `mesh_scale`.

- [ ] **Step 1: Write strict loader tests with a real temporary render fixture**

```python
import json

import numpy as np
import pytest
import torch
from PIL import Image

from pixal3d.datasets.components import load_anchor_first_conditions


def write_render_fixture(root, num_views=8):
    frames = []
    for index in range(num_views):
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[..., index % 3] = 10 + index
        rgba[..., 3] = 255
        Image.fromarray(rgba, mode="RGBA").save(root / f"{index:03d}.png")
        transform = np.eye(4, dtype=np.float32)
        transform[0, 3] = index
        transform[2, 3] = 2.0
        frames.append({
            "file_path": f"{index:03d}.png",
            "camera_angle_x": 0.5 + index * 0.01,
            "transform_matrix": transform.tolist(),
        })
    (root / "transforms.json").write_text(json.dumps({"frames": frames}))


def test_loader_keeps_anchor_first_and_all_views_unique(tmp_path):
    write_render_fixture(tmp_path)
    result = load_anchor_first_conditions(
        tmp_path,
        anchor_index=1,
        image_size=4,
        other_view_indices=[7, 4, 0, 2, 3, 5, 6],
    )
    assert result["view_indices"].tolist() == [1, 7, 4, 0, 2, 3, 5, 6]
    assert result["cond"].shape == (8, 3, 4, 4)
    assert result["camera_angle_x"].shape == (8,)
    assert result["camera_distance"].shape == (8,)
    assert result["transform_matrix"].shape == (8, 4, 4)
    assert torch.equal(
        result["transform_matrix"][0, :3, 3], torch.tensor([1.0, 0.0, 2.0])
    )


def test_loader_requires_exactly_eight_training_frames(tmp_path):
    write_render_fixture(tmp_path, num_views=7)
    with pytest.raises(ValueError, match="exactly eight"):
        load_anchor_first_conditions(
            tmp_path,
            anchor_index=1,
            image_size=4,
            other_view_indices=[0, 2, 3, 4, 5, 6],
        )


def test_loader_rejects_duplicate_anchor(tmp_path):
    write_render_fixture(tmp_path)
    with pytest.raises(ValueError, match="anchor"):
        load_anchor_first_conditions(
            tmp_path,
            anchor_index=1,
            image_size=4,
            other_view_indices=[1, 0, 2, 3, 4, 5, 6],
        )


def test_loader_rejects_missing_or_nonfinite_camera(tmp_path):
    write_render_fixture(tmp_path)
    manifest = json.loads((tmp_path / "transforms.json").read_text())
    manifest["frames"][3]["transform_matrix"][0][0] = float("nan")
    (tmp_path / "transforms.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="finite"):
        load_anchor_first_conditions(
            tmp_path,
            anchor_index=0,
            image_size=4,
            other_view_indices=[1, 2, 3, 4, 5, 6, 7],
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_dataset_conditions.py -q`

Expected: collection fails because `load_anchor_first_conditions` does not exist.

- [ ] **Step 3: Implement RGBA loading and strict camera alignment**

Add the required `Path` and `Sequence` imports, then add beside the existing condition mixins:

```python
def _load_rgba_condition(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as source:
        rgba_image = source.convert("RGBA").resize(
            (image_size, image_size), Image.Resampling.LANCZOS
        )
        rgba = torch.from_numpy(np.asarray(rgba_image).copy()).float() / 255.0
    rgb = rgba[..., :3].permute(2, 0, 1)
    return rgb * rgba[..., 3].unsqueeze(0)


def load_anchor_first_conditions(
    image_root: Union[str, os.PathLike],
    *,
    anchor_index: int,
    image_size: int,
    other_view_indices: Sequence[int],
) -> Dict[str, torch.Tensor]:
    image_root = Path(image_root).resolve()
    manifest_path = image_root / "transforms.json"
    metadata = json.loads(manifest_path.read_text())
    frames = metadata.get("frames")
    if not isinstance(frames, list) or len(frames) != 8:
        raise ValueError("development-training manifest must contain exactly eight frames")
    if anchor_index not in (0, 1):
        raise ValueError("target anchor must be view00 or view01")
    order = [anchor_index, *other_view_indices]
    if anchor_index in other_view_indices:
        raise ValueError("anchor must not occur in other_view_indices")
    if len(order) != 8 or len(set(order)) != 8 or sorted(order) != list(range(8)):
        raise ValueError("view order must contain all eight render indices exactly once")

    images = []
    angles = []
    distances = []
    transforms = []
    for view_index in order:
        frame = frames[view_index]
        image_path = (image_root / frame["file_path"]).resolve()
        if not image_path.is_relative_to(image_root) or not image_path.is_file():
            raise FileNotFoundError(f"missing condition image for view {view_index}: {image_path}")
        angle = frame.get("camera_angle_x", metadata.get("camera_angle_x"))
        if angle is None or not np.isfinite(float(angle)):
            raise ValueError(f"camera_angle_x must be finite for view {view_index}")
        transform = torch.as_tensor(frame.get("transform_matrix"), dtype=torch.float32)
        if transform.shape != (4, 4) or not torch.isfinite(transform).all():
            raise ValueError(f"transform_matrix must be finite [4, 4] for view {view_index}")
        images.append(_load_rgba_condition(image_path, image_size))
        angles.append(float(angle))
        distances.append(torch.linalg.vector_norm(transform[:3, 3]))
        transforms.append(transform)
    return {
        "cond": torch.stack(images),
        "camera_angle_x": torch.tensor(angles, dtype=torch.float32),
        "camera_distance": torch.stack(distances).float(),
        "transform_matrix": torch.stack(transforms),
        "view_indices": torch.tensor(order, dtype=torch.int64),
    }
```

- [ ] **Step 4: Add the anchor-first multi-view mixin without modifying either single-view mixin**

```python
class MultiViewImageConditionedMixin:
    def __init__(
        self,
        roots,
        *,
        image_size=518,
        condition_num_views=8,
        min_condition_views=2,
        max_condition_views=6,
        **kwargs,
    ):
        if condition_num_views != 8:
            raise ValueError("development training requires condition_num_views=8")
        if not 2 <= min_condition_views <= max_condition_views <= 6:
            raise ValueError("training view bounds must satisfy 2 <= min <= max <= 6")
        self.image_size = image_size
        self.condition_num_views = condition_num_views
        self.min_condition_views = min_condition_views
        self.max_condition_views = max_condition_views
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata, dataset_name=None):
        metadata, stats = super().filter_metadata(metadata, dataset_name=dataset_name)
        metadata = metadata[metadata["cond_rendered"].notna()]
        stats["Cond rendered"] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        anchor_index = self._current_view_idx
        other_indices = np.random.permutation(
            [index for index in range(8) if index != anchor_index]
        ).tolist()
        try:
            pack.update(load_anchor_first_conditions(
                os.path.join(root["render_cond"], instance),
                anchor_index=anchor_index,
                image_size=self.image_size,
                other_view_indices=other_indices,
            ))
            scale_path = Path(self._current_latent_dir) / f"view{anchor_index:02d}_scale.json"
            scale = json.loads(scale_path.read_text()).get("total_scale")
            if scale is None or not np.isfinite(float(scale)) or float(scale) <= 0:
                raise ValueError(f"total_scale must be finite and positive: {scale_path}")
            pack["mesh_scale"] = torch.tensor(float(scale), dtype=torch.float32)
        except Exception as error:
            source = getattr(self, "_current_dataset_name", "unknown")
            raise RuntimeError(
                f"source={source} asset={instance} anchor=view{anchor_index:02d}: {error}"
            ) from error
        return pack
```

In `StandardDatasetBase.__getitem__`, assign `self._current_dataset_name = dataset_name` immediately before `get_instance` so raised errors include the source. Do not change its existing retry policy.

- [ ] **Step 5: Run the loader tests**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_dataset_conditions.py -q`

Expected: `4 passed`.

- [ ] **Step 6: Commit strict condition loading**

```bash
git add tests/multiview/test_dataset_conditions.py pixal3d/datasets/components.py
git commit -m "feat: load anchor-first calibrated training views"
```

---

### Task 4: Select One K Per Batch and Register All Stage Datasets

**Files:**
- Create: `tests/multiview/test_dataset_collation.py`
- Modify: `pixal3d/datasets/components.py`
- Modify: `pixal3d/datasets/sparse_structure_latent.py:399-408`
- Modify: `pixal3d/datasets/structured_latent_shape.py:393-402`
- Modify: `pixal3d/datasets/structured_latent_svpbr.py:655-666`
- Modify: `pixal3d/datasets/__init__.py:3-28`

**Interfaces:**
- Consumes: Task 3 per-sample eight-view fields.
- Produces: `slice_condition_views(batch, K)`, one uniform batch-wide K, and three registered multi-view dataset names shared by four model stages.

- [ ] **Step 1: Write failing K and registration tests**

```python
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from pixal3d import datasets
from pixal3d.datasets.components import (
    MultiViewImageConditionedMixin,
    slice_condition_views,
)
from pixal3d.datasets.sparse_structure_latent import (
    MultiViewImageConditionedSparseStructureLatentView,
)
from pixal3d.datasets.structured_latent_shape import (
    MultiViewImageConditionedSLatShapeView,
)
from pixal3d.datasets.structured_latent_svpbr import (
    MultiViewImageConditionedSLatPbrView,
)


VIEW_KEYS = (
    "cond", "camera_angle_x", "camera_distance", "transform_matrix", "view_indices"
)


def condition_sample(offset):
    return {
        "cond": torch.arange(offset, offset + 24).reshape(8, 3, 1, 1).float(),
        "camera_angle_x": torch.arange(8).float() + offset,
        "camera_distance": torch.arange(8).float() + 2.0,
        "transform_matrix": torch.eye(4).repeat(8, 1, 1),
        "view_indices": torch.arange(8),
        "mesh_scale": torch.tensor(1.0),
    }


def selector():
    value = SimpleNamespace(min_condition_views=2, max_condition_views=6)
    value.select_batch_condition_views = MethodType(
        MultiViewImageConditionedMixin.select_batch_condition_views, value
    )
    return value


def test_slice_is_batchwide_and_does_not_mutate_sources():
    source = [condition_sample(0), condition_sample(100)]
    sliced = slice_condition_views(source, 4)
    for item in sliced:
        for key in VIEW_KEYS:
            assert item[key].shape[0] == 4
    assert source[0]["cond"].shape[0] == 8
    assert sliced[0]["mesh_scale"].ndim == 0


@pytest.mark.parametrize("num_views", [2, 6])
def test_dense_shape_and_pbr_collators_keep_one_endpoint_k(monkeypatch, num_views):
    monkeypatch.setattr(np.random, "randint", lambda low, high: num_views)
    dense = condition_sample(0)
    dense["x_0"] = torch.zeros(2, 2, 2, 2)
    shape = condition_sample(0)
    shape.update({
        "coords": torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int32),
        "feats": torch.zeros(2, 32),
    })
    pbr = condition_sample(0)
    pbr.update({
        "coords": torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int32),
        "pbr_feats": torch.zeros(2, 32),
        "shape_feats": torch.ones(2, 32),
    })
    dense_pack = MultiViewImageConditionedSparseStructureLatentView.collate_fn(
        selector(), [dense, dense]
    )
    shape_pack = MultiViewImageConditionedSLatShapeView.collate_fn(
        selector(), [shape, shape]
    )
    pbr_pack = MultiViewImageConditionedSLatPbrView.collate_fn(
        selector(), [pbr, pbr]
    )
    assert dense_pack["cond"].shape[:2] == (2, num_views)
    assert shape_pack["cond"].shape[:2] == (2, num_views)
    assert pbr_pack["cond"].shape[:2] == (2, num_views)
    assert shape_pack["x_0"].shape[0] == 2
    assert pbr_pack["concat_cond"].shape[0] == 2


def test_all_multiview_dataset_names_are_registered():
    expected = {
        "MultiViewImageConditionedSparseStructureLatentView":
            MultiViewImageConditionedSparseStructureLatentView,
        "MultiViewImageConditionedSLatShapeView":
            MultiViewImageConditionedSLatShapeView,
        "MultiViewImageConditionedSLatPbrView":
            MultiViewImageConditionedSLatPbrView,
    }
    for name, class_object in expected.items():
        assert getattr(datasets, name) is class_object
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_dataset_collation.py -q`

Expected: collection fails because the slicing helper and new dataset classes do not exist.

- [ ] **Step 3: Add immutable view slicing and uniform batch K selection**

```python
MULTIVIEW_CONDITION_KEYS = (
    "cond",
    "camera_angle_x",
    "camera_distance",
    "transform_matrix",
    "view_indices",
)


def slice_condition_views(
    batch: Sequence[Dict[str, Any]], num_views: int
) -> List[Dict[str, Any]]:
    if not 1 <= num_views <= 8:
        raise ValueError("num_views must be between 1 and 8")
    sliced = []
    for source in batch:
        item = dict(source)
        for key in MULTIVIEW_CONDITION_KEYS:
            if key not in source or source[key].shape[0] < num_views:
                raise ValueError(f"{key} does not contain {num_views} aligned views")
            item[key] = source[key][:num_views]
        sliced.append(item)
    return sliced
```

Add to `MultiViewImageConditionedMixin`:

```python
def select_batch_condition_views(self, batch):
    num_views = int(np.random.randint(
        self.min_condition_views, self.max_condition_views + 1
    ))
    return slice_condition_views(batch, num_views)
```

- [ ] **Step 4: Add three thin classes that delegate latent collation unchanged**

```python
# sparse_structure_latent.py
from torch.utils.data._utils.collate import default_collate
from .components import MultiViewImageConditionedMixin


class MultiViewImageConditionedSparseStructureLatentView(
    MultiViewImageConditionedMixin, SparseStructureLatentView
):
    def collate_fn(self, batch, split_size=None):
        if split_size is not None:
            raise ValueError("dense sparse-structure collation does not use split_size")
        return default_collate(self.select_batch_condition_views(batch))


# structured_latent_shape.py
class MultiViewImageConditionedSLatShapeView(
    MultiViewImageConditionedMixin, SLatShapeView
):
    def collate_fn(self, batch, split_size=None):
        return SLatShapeView.collate_fn(
            self.select_batch_condition_views(batch), split_size=split_size
        )


# structured_latent_svpbr.py
class MultiViewImageConditionedSLatPbrView(
    MultiViewImageConditionedMixin, SLatPbrView
):
    def collate_fn(self, batch, split_size=None):
        return SLatPbrView.collate_fn(
            self.select_batch_condition_views(batch), split_size=split_size
        )
```

Add the exact names to `pixal3d/datasets/__init__.py`:

```python
'MultiViewImageConditionedSparseStructureLatentView': 'sparse_structure_latent',
'MultiViewImageConditionedSLatShapeView': 'structured_latent_shape',
'MultiViewImageConditionedSLatPbrView': 'structured_latent_svpbr',
```

- [ ] **Step 5: Run all dataset tests**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_dataset_conditions.py tests/multiview/test_dataset_collation.py -q`

Expected: `8 passed`.

- [ ] **Step 6: Commit dataset collation and registration**

```bash
git add tests/multiview/test_dataset_collation.py pixal3d/datasets/components.py pixal3d/datasets/sparse_structure_latent.py pixal3d/datasets/structured_latent_shape.py pixal3d/datasets/structured_latent_svpbr.py pixal3d/datasets/__init__.py
git commit -m "feat: collate one view count across each training batch"
```

---

### Task 5: Preserve Trainer Snapshots with 5D Conditions

**Files:**
- Create: `tests/multiview/test_trainer_views.py`
- Modify: `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py:1460-1530`
- Modify: `pixal3d/trainers/flow_matching/flow_matching.py:446-560`
- Modify: `pixal3d/trainers/flow_matching/sparse_flow_matching.py:431-515`

**Interfaces:**
- Consumes: 4D/5D condition tensors and scalar/view-vector cameras.
- Produces: `anchor_condition_image` and `anchor_camera_value`; condition encoding continues to receive all K views, while target rendering uses only anchor index zero.

- [ ] **Step 1: Write failing helper tests**

```python
import torch

from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    anchor_camera_value,
    anchor_condition_image,
)


def test_anchor_condition_handles_single_and_multiview():
    single = torch.zeros(2, 3, 4, 4)
    multi = torch.arange(2 * 6 * 3 * 4 * 4).reshape(2, 6, 3, 4, 4)
    assert anchor_condition_image(single) is single
    assert torch.equal(anchor_condition_image(multi), multi[:, 0])


def test_anchor_camera_handles_scalar_and_view_vectors():
    single = torch.tensor([2.0, 3.0])
    multi = torch.tensor([[2.0, 4.0], [3.0, 5.0]])
    assert anchor_camera_value(single) is single
    assert torch.equal(anchor_camera_value(multi), torch.tensor([2.0, 3.0]))
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_trainer_views.py -q`

Expected: collection fails for the two missing helpers.

- [ ] **Step 3: Add non-learned anchor helpers and use them only for visualization**

```python
def anchor_condition_image(cond: torch.Tensor) -> torch.Tensor:
    return cond[:, 0] if cond.ndim == 5 else cond


def anchor_camera_value(value: torch.Tensor) -> torch.Tensor:
    return value[:, 0] if value.ndim > 1 else value
```

For now, change `ImageConditionedProjMixin.vis_cond` to return the anchor under the existing `image` key:

```python
def vis_cond(self, cond, **kwargs):
    return {"image": {"value": anchor_condition_image(cond), "type": "image"}}
```

Task 12 will add the ordered multi-view grid only after model validation passes.

- [ ] **Step 4: Normalize only target-rendering camera values in dense and sparse snapshots**

In `flow_matching.py`, wrap the concatenated `camera_distance` and `camera_angle_x` with `anchor_camera_value` before adding them to `sample_gt_value` and `sample_value`. In `sparse_flow_matching.py`, apply this complete block only after `sample_gt` and `sample` dictionaries are built:

```python
for key in ("camera_angle_x", "camera_distance"):
    if key in sample_gt:
        sample_gt[key] = anchor_camera_value(sample_gt[key])
        sample[key] = anchor_camera_value(sample[key])
```

Import `anchor_camera_value` from the projection mixin module. Do not slice `cond`, `transform_matrix`, or camera tensors before `get_cond`/`get_inference_cond`.

- [ ] **Step 5: Run snapshot, conditioner, and dataset tests**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_trainer_views.py tests/multiview/test_conditioner.py tests/multiview/test_dataset_collation.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit snapshot compatibility**

```bash
git add tests/multiview/test_trainer_views.py pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py pixal3d/trainers/flow_matching/flow_matching.py pixal3d/trainers/flow_matching/sparse_flow_matching.py
git commit -m "fix: keep multiview trainer snapshots anchor aligned"
```

---

### Task 6: Materialize the Audited ABO Pilot64 into Four Isolated Stage Roots

**Files:**
- Create: `scripts/materialize_multiview_pilot.py`
- Create: `tests/multiview/test_pilot_materialization.py`
- Runtime output: `/root/node17/data/pixal3d/train/development/abo-pilot64/{ss64,shape512,shape1024,pbr1024}/active`

**Interfaces:**
- Consumes: the five approved `batch000.tar` files and adjacent manifests under `qualification/pilot`.
- Produces: four non-overwritten stage roots with generated `metadata.csv` files and a `materialization.json` evidence file; every input pack is verified before extraction.

- [ ] **Step 1: Write a failing small-archive materialization test**

```python
import csv
import io
import json
import tarfile
from hashlib import sha256

from scripts.materialize_multiview_pilot import (
    extract_verified_pack,
    write_metadata,
)


def write_pack(path, members):
    manifest_members = []
    with tarfile.open(path, "w") as bundle:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
            manifest_members.append({
                "path": name,
                "size": len(payload),
                "sha256": sha256(payload).hexdigest(),
            })
    manifest = {
        "pack_sha256": sha256(path.read_bytes()).hexdigest(),
        "asset_sha256s": ["a" * 64],
        "included_asset_sha256s": ["a" * 64],
        "completed_count": 1,
        "members": manifest_members,
    }
    manifest_path = path.with_suffix(".tar.manifest.json")
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def test_verified_extract_and_metadata_are_non_destructive(tmp_path):
    pack = tmp_path / "one.tar"
    manifest = write_pack(pack, {"renders_cond/" + "a" * 64 + "/000.png": b"png"})
    destination = tmp_path / "active"
    destination.mkdir()
    extract_verified_pack(pack, manifest, destination)
    write_metadata(destination / "renders_cond", ["a" * 64], {"cond_rendered": True})
    with (destination / "renders_cond" / "metadata.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [{"sha256": "a" * 64, "cond_rendered": "True"}]


def test_verified_extract_rejects_path_traversal(tmp_path):
    pack = tmp_path / "unsafe.tar"
    manifest = write_pack(pack, {"../escape": b"bad"})
    try:
        extract_verified_pack(pack, manifest, tmp_path / "active")
    except ValueError as error:
        assert "unsafe" in str(error)
    else:
        raise AssertionError("unsafe tar member was extracted")
```

- [ ] **Step 2: Run the test and verify RED**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_pilot_materialization.py -q`

Expected: import failure because the materializer does not exist.

- [ ] **Step 3: Implement verified extraction and metadata generation**

Create the script with these exact constants and functions. Use `data_toolkit.pipeline.packing.verify_pack` before reading a tar, manually copy only regular files, and never add an overwrite option:

```python
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tarfile
import tempfile
from hashlib import sha256
from pathlib import Path

from data_toolkit.pipeline.packing import verify_pack


PREPARED = Path("/root/data2/pixal3d/prepared/qualification/pilot")
OUTPUT = Path("/root/node17/data/pixal3d/train/development/abo-pilot64")
SOURCE = "ABO"
SHARD = "ABO-00000"
BATCH = "batch000.tar"
FAMILIES = {
    "common": PREPARED / "common" / SOURCE / SHARD / BATCH,
    "ss64": PREPARED / "ss" / "64" / SOURCE / SHARD / BATCH,
    "shape512": PREPARED / "shape" / "512" / SOURCE / SHARD / BATCH,
    "shape1024": PREPARED / "shape" / "1024" / SOURCE / SHARD / BATCH,
    "pbr1024": PREPARED / "pbr" / "1024" / SOURCE / SHARD / BATCH,
}
STAGES = {
    "ss64": ("common", "ss64"),
    "shape512": ("common", "shape512"),
    "shape1024": ("common", "shape1024"),
    "pbr1024": ("common", "shape1024", "pbr1024"),
}
LATENT_DIRS = {
    "ss64": "ss_latents/ss_enc_conv3d_16l8_fp16_64_view",
    "shape512": "shape_latents/shape_enc_next_dc_f16c32_fp16_512_view",
    "shape1024": "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view",
    "pbr1024": "pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix",
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_verified_pack(pack: Path, manifest_path: Path, destination: Path) -> None:
    verify_pack(pack, manifest_path)
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(pack, "r:") as bundle:
        for member in bundle:
            member_path = (destination / member.name).resolve()
            if not member_path.is_relative_to(destination):
                raise ValueError(f"unsafe tar member: {member.name}")
            if member.isdir():
                member_path.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"unsafe non-regular tar member: {member.name}")
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"unreadable tar member: {member.name}")
            member_path.parent.mkdir(parents=True, exist_ok=True)
            with source, member_path.open("wb") as target:
                shutil.copyfileobj(source, target)


def write_metadata(root: Path, assets: list[str], values: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fields = ["sha256", *values]
    with (root / "metadata.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for asset in assets:
            writer.writerow({"sha256": asset, **values})


def manifest_for(pack: Path) -> Path:
    return pack.with_suffix(".tar.manifest.json")


def materialize_stage(stage: str, output_root: Path) -> Path:
    final = output_root / stage / "active"
    if final.exists():
        raise FileExistsError(f"refusing to overwrite existing stage root: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".materializing-", dir=final.parent))
    evidence = {"stage": stage, "packs": []}
    try:
        asset_scope = None
        for family in STAGES[stage]:
            pack = FAMILIES[family]
            manifest_path = manifest_for(pack)
            manifest = json.loads(manifest_path.read_text())
            assets = list(manifest["included_asset_sha256s"])
            if manifest.get("completed_count") != 64 or len(assets) != 64:
                raise ValueError(f"{family} is not a complete 64-asset pilot pack")
            if asset_scope is None:
                asset_scope = assets
            elif assets != asset_scope:
                raise ValueError(f"asset scope differs for family {family}")
            extract_verified_pack(pack, manifest_path, temporary)
            evidence["packs"].append({
                "family": family,
                "pack": str(pack),
                "pack_sha256": file_sha256(pack),
                "manifest": str(manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
            })

        write_metadata(temporary / "renders_cond", asset_scope, {"cond_rendered": True})
        if stage == "ss64":
            write_metadata(temporary / LATENT_DIRS[stage], asset_scope, {
                "ss_latent_view_scale00_encoded": True,
                "ss_latent_view_scale01_encoded": True,
            })
        if stage in ("shape512", "shape1024"):
            write_metadata(temporary / LATENT_DIRS[stage], asset_scope, {
                "shape_latent_view00_encoded": True,
                "shape_latent_view01_encoded": True,
            })
        if stage == "pbr1024":
            write_metadata(temporary / LATENT_DIRS["shape1024"], asset_scope, {
                "shape_latent_view00_encoded": True,
                "shape_latent_view01_encoded": True,
            })
            write_metadata(temporary / LATENT_DIRS[stage], asset_scope, {
                "pbr_latent_view00_encoded": True,
                "pbr_latent_view01_encoded": True,
            })
        evidence["asset_count"] = len(asset_scope)
        (temporary / "materialization.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--stage", choices=tuple(STAGES), action="append")
    args = parser.parse_args()
    for stage in args.stage or list(STAGES):
        print(materialize_stage(stage, args.output_root))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run materializer unit tests**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_pilot_materialization.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Materialize the four immutable pilot roots**

First verify no target already exists:

Run: `find /root/node17/data/pixal3d/train/development/abo-pilot64 -maxdepth 3 -type d -name active -print 2>/dev/null`

Expected: no output. If output exists, inspect its `materialization.json`; do not overwrite or delete it implicitly.

Run: `conda run --no-capture-output -n pixal3d python scripts/materialize_multiview_pilot.py`

Expected: exactly four `.../active` paths are printed and every `materialization.json` reports `asset_count: 64`.

- [ ] **Step 6: Commit the reproducible materializer**

```bash
git add scripts/materialize_multiview_pilot.py tests/multiview/test_pilot_materialization.py
git commit -m "feat: materialize audited multiview pilot data"
```

---

### Task 7: Materialize Four Released Weights and Add Four Fine-Tuning Configs

**Files:**
- Create: `scripts/materialize_multiview_checkpoints.py`
- Create: `tests/multiview/test_checkpoints.py`
- Create: `tests/multiview/test_configs.py`
- Create: `configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json`
- Create: `configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json`
- Create: `configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json`
- Create: `configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json`
- Runtime output: `/root/node17/data/pixal3d/train/checkpoints/single_view/*.pt`

**Interfaces:**
- Consumes: four released `TencentARC/Pixal3D/ckpts/*.safetensors` files and the matching current single-view config variants.
- Produces: four `.pt` state dictionaries with identical keys/tensors and four configs whose denoiser/trainer definitions remain unchanged.

- [ ] **Step 1: Write failing conversion and config-invariant tests**

```python
# tests/multiview/test_checkpoints.py
import torch
from safetensors.torch import load_file, save_file

from scripts.materialize_multiview_checkpoints import convert_checkpoint


def test_checkpoint_conversion_preserves_every_key_and_tensor(tmp_path):
    source = tmp_path / "source.safetensors"
    target = tmp_path / "target.pt"
    expected = {
        "blocks.0.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
        "blocks.0.bias": torch.arange(2, dtype=torch.bfloat16),
    }
    save_file(expected, source)
    convert_checkpoint(source, target)
    actual = torch.load(target, map_location="cpu", weights_only=True)
    assert actual.keys() == expected.keys()
    for key in expected:
        assert torch.equal(actual[key], load_file(source)[key])
```

```python
# tests/multiview/test_configs.py
import json
from pathlib import Path


CONFIGS = {
    "ss64": Path("configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json"),
    "shape512": Path("configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json"),
    "shape1024": Path("configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json"),
    "pbr1024": Path("configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json"),
}
CHECKPOINTS = {
    "ss64": "ss_flow_img_dit_1_3B_64_bf16.pt",
    "shape512": "slat_flow_img2shape_dit_1_3B_512_bf16.pt",
    "shape1024": "slat_flow_img2shape_dit_1_3B_1024_bf16.pt",
    "pbr1024": "slat_flow_imgshape2tex_dit_1_3B_1024_bf16.pt",
}


def test_four_configs_use_batchwide_k_and_matching_checkpoints():
    datasets = {
        "ss64": "MultiViewImageConditionedSparseStructureLatentView",
        "shape512": "MultiViewImageConditionedSLatShapeView",
        "shape1024": "MultiViewImageConditionedSLatShapeView",
        "pbr1024": "MultiViewImageConditionedSLatPbrView",
    }
    for stage, path in CONFIGS.items():
        config = json.loads(path.read_text())
        dataset_args = config["dataset"]["args"]
        trainer_args = config["trainer"]["args"]
        assert config["dataset"]["name"] == datasets[stage]
        assert dataset_args["condition_num_views"] == 8
        assert dataset_args["min_condition_views"] == 2
        assert dataset_args["max_condition_views"] == 6
        assert trainer_args["batch_size_per_gpu"] == 1
        assert trainer_args["batch_split"] == 1
        assert trainer_args["multiview_stage"] == stage
        assert trainer_args["image_cond_model"]["name"] == "DinoV3ProjFeatureExtractor"
        assert trainer_args["finetune_ckpt"] == {
            "denoiser": "/root/node17/data/pixal3d/train/checkpoints/single_view/"
            + CHECKPOINTS[stage]
        }
        assert config["models"]["denoiser"]["args"]["image_attn_mode"] == "proj"


def test_shape_512_is_an_independent_inference_checkpoint():
    config = json.loads(CONFIGS["shape512"].read_text())
    assert config["models"]["denoiser"]["args"]["resolution"] == 32
    assert config["dataset"]["args"]["resolution"] == 512
    assert config["trainer"]["args"]["finetune_ckpt"]["denoiser"].endswith(
        "slat_flow_img2shape_dit_1_3B_512_bf16.pt"
    )
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_checkpoints.py tests/multiview/test_configs.py -q`

Expected: conversion-script import and all four config paths fail.

- [ ] **Step 3: Implement key-preserving download and conversion**

```python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


REPO_ID = "TencentARC/Pixal3D"
OUTPUT = Path("/root/node17/data/pixal3d/train/checkpoints/single_view")
CHECKPOINTS = {
    "ss_flow_img_dit_1_3B_64_bf16": "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
    "slat_flow_img2shape_dit_1_3B_512_bf16": "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
    "slat_flow_img2shape_dit_1_3B_1024_bf16": "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
    "slat_flow_imgshape2tex_dit_1_3B_1024_bf16": "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
}


def convert_checkpoint(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {target}")
    state_dict = load_file(str(source), device="cpu")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, target)
    reloaded = torch.load(target, map_location="cpu", weights_only=True)
    if reloaded.keys() != state_dict.keys():
        raise RuntimeError(f"checkpoint keys changed during conversion: {source}")
    for key in state_dict:
        if not torch.equal(reloaded[key], state_dict[key]):
            raise RuntimeError(f"checkpoint tensor changed during conversion: {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    for stem, filename in CHECKPOINTS.items():
        source = Path(hf_hub_download(repo_id=args.repo_id, filename=filename))
        target = args.output_root / f"{stem}.pt"
        convert_checkpoint(source, target)
        print(target)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create four configs by preserving each exact source config**

Use `apply_patch` to create each new file from the stated source and change only the listed fields:

| Stage | Exact source config | New dataset | Released `.pt` |
|---|---|---|---|
| SS-64 | `ss_flow_img_dit_1_3B_32_bf16_proj_finetune_ft64.json` | `MultiViewImageConditionedSparseStructureLatentView` | `ss_flow_img_dit_1_3B_64_bf16.pt` |
| Shape-512 | `slat_flow_img2shape_dit_1_3B_256_bf16_proj_finetune_ft512.json` | `MultiViewImageConditionedSLatShapeView` | `slat_flow_img2shape_dit_1_3B_512_bf16.pt` |
| Shape-1024 | `slat_flow_img2shape_dit_1_3B_512_bf16_proj_finetune_ft1024.json` | `MultiViewImageConditionedSLatShapeView` | `slat_flow_img2shape_dit_1_3B_1024_bf16.pt` |
| PBR-1024 | `slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_finetune_ft1024.json` | `MultiViewImageConditionedSLatPbrView` | `slat_flow_imgshape2tex_dit_1_3B_1024_bf16.pt` |

For every new config, apply this exact dataset/trainer mutation while retaining all other model, normalization, optimizer, schedule, NAF, and decoder values from its source:

```json
"condition_num_views": 8,
"min_condition_views": 2,
"max_condition_views": 6
```

Remove the obsolete `load_camera_info` dataset argument where present. Set `batch_size_per_gpu` and `batch_split` to `1` in all four files. Set the stage field to these exact literals:

| Config | Exact field |
|---|---|
| SS-64 | `"multiview_stage": "ss64"` |
| Shape-512 | `"multiview_stage": "shape512"` |
| Shape-1024 | `"multiview_stage": "shape1024"` |
| PBR-1024 | `"multiview_stage": "pbr1024"` |

Point `trainer.args.finetune_ckpt.denoiser` to the exact absolute `.pt` path from `tests/multiview/test_configs.py`. Leave `max_steps`, `i_log`, `i_sample`, and `i_save` at their long-training source values; Task 10 adds an explicit smoke-only runtime override.

- [ ] **Step 5: Run conversion and config unit tests**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_checkpoints.py tests/multiview/test_configs.py -q`

Expected: `3 passed`.

- [ ] **Step 6: Download and convert all four released weights**

Run: `conda run --no-capture-output -n pixal3d python scripts/materialize_multiview_checkpoints.py`

Expected: four `.pt` paths are printed directly under `/root/node17/data/pixal3d/train/checkpoints/single_view`; no target is overwritten.

- [ ] **Step 7: Add and run the opt-in strict architecture gate**

Append to `tests/multiview/test_configs.py`:

```python
import gc
import os

import pytest
import torch

from pixal3d import models


@pytest.mark.integration
@pytest.mark.parametrize("stage", tuple(CONFIGS))
def test_released_checkpoint_strictly_matches_denoiser(stage):
    config = json.loads(CONFIGS[stage].read_text())
    checkpoint = Path(config["trainer"]["args"]["finetune_ckpt"]["denoiser"])
    if not checkpoint.exists():
        if os.environ.get("PIXAL3D_REQUIRE_CHECKPOINTS") == "1":
            pytest.fail(f"required checkpoint is missing: {checkpoint}")
        pytest.skip(f"checkpoint has not been materialized: {checkpoint}")
    model_config = config["models"]["denoiser"]
    denoiser = getattr(models, model_config["name"])(**model_config["args"])
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = denoiser.load_state_dict(state_dict, strict=False)
    assert set(incompatible.missing_keys) <= {"rope_phases"}
    assert incompatible.unexpected_keys == []
    del state_dict, denoiser
    gc.collect()
```

Run: `PIXAL3D_REQUIRE_CHECKPOINTS=1 conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_configs.py -q -m integration`

Expected: `4 passed`; no test is skipped, no unexpected key exists, and tensor-shape mismatch raises rather than being rewritten.

- [ ] **Step 8: Commit all four independent model configurations**

```bash
git add scripts/materialize_multiview_checkpoints.py tests/multiview/test_checkpoints.py tests/multiview/test_configs.py configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json
git commit -m "feat: configure four multiview flow checkpoints"
```

---

### Task 8: Propagate One Calibrated View Set Through All Four Pipeline Models

**Files:**
- Create: `tests/multiview/test_pipeline_inputs.py`
- Modify: `pixal3d/pipelines/pixal3d_image_to_3d.py:188-295`
- Modify: `pixal3d/pipelines/pixal3d_image_to_3d.py:609-780`

**Interfaces:**
- Consumes: one PIL image/scalar cameras or K PIL images/per-view cameras and transforms.
- Produces: `normalize_calibrated_views`, `pil_views_to_tensor`, and the existing `{'cond', 'neg_cond'}` dictionaries for SS-64, Shape-512, Shape-1024, and PBR-1024.

- [ ] **Step 1: Write failing input-normalization tests**

```python
import pytest
import torch
from PIL import Image

from pixal3d.pipelines.pixal3d_image_to_3d import normalize_calibrated_views


def image(color):
    return Image.new("RGB", (8, 8), color=color)


def test_uncalibrated_single_view_normalizes_to_legacy_k1():
    images, cameras = normalize_calibrated_views(
        image("red"),
        {"camera_angle_x": 0.7, "distance": 2.5, "mesh_scale": 1.0},
    )
    assert len(images) == 1
    assert cameras["camera_angle_x"].shape == (1, 1)
    assert cameras["distance"].shape == (1, 1)
    assert cameras["mesh_scale"].shape == (1,)
    assert cameras["transform_matrix"] is None


def test_calibrated_multiview_preserves_order_and_shapes():
    transforms = torch.eye(4).repeat(2, 1, 1)
    images, cameras = normalize_calibrated_views(
        [image("red"), image("blue")],
        {
            "camera_angle_x": [0.7, 0.8],
            "distance": [2.5, 2.7],
            "mesh_scale": 1.0,
            "transform_matrix": transforms,
        },
    )
    assert [view.getpixel((0, 0)) for view in images] == [(255, 0, 0), (0, 0, 255)]
    assert cameras["camera_angle_x"].shape == (1, 2)
    assert cameras["distance"].shape == (1, 2)
    assert cameras["transform_matrix"].shape == (1, 2, 4, 4)


@pytest.mark.parametrize("num_views", [0, 9])
def test_inference_rejects_view_counts_outside_one_to_eight(num_views):
    with pytest.raises(ValueError, match="between 1 and 8"):
        normalize_calibrated_views(
            [image("red")] * num_views,
            {"camera_angle_x": [0.7] * num_views, "distance": [2.5] * num_views},
        )


def test_multiview_requires_calibrated_transforms():
    with pytest.raises(ValueError, match="transform_matrix"):
        normalize_calibrated_views(
            [image("red"), image("blue")],
            {"camera_angle_x": [0.7, 0.8], "distance": [2.5, 2.7]},
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_pipeline_inputs.py -q`

Expected: collection fails because `normalize_calibrated_views` does not exist.

- [ ] **Step 3: Implement one strict pipeline input boundary**

Add at module scope:

```python
def normalize_calibrated_views(image, camera_params):
    images = list(image) if isinstance(image, (list, tuple)) else [image]
    if not 1 <= len(images) <= 8:
        raise ValueError("Pixal3D inference requires between 1 and 8 views")
    if not all(isinstance(view, Image.Image) for view in images):
        raise TypeError("every inference view must be a PIL image")
    num_views = len(images)

    def vector(name):
        value = camera_params[name]
        values = [value] if np.isscalar(value) else list(value)
        if len(values) != num_views:
            raise ValueError(f"{name} must contain one value per view")
        tensor = torch.tensor(values, dtype=torch.float32).reshape(1, num_views)
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be finite")
        return tensor

    transforms = camera_params.get("transform_matrix")
    if transforms is not None:
        transforms = torch.as_tensor(transforms, dtype=torch.float32)
        if transforms.shape != (num_views, 4, 4) or not torch.isfinite(transforms).all():
            raise ValueError("transform_matrix must be finite with shape [K, 4, 4]")
        transforms = transforms.unsqueeze(0)
    if num_views > 1 and transforms is None:
        raise ValueError("calibrated multi-view inference requires transform_matrix")
    mesh_scale = torch.tensor(
        [float(camera_params.get("mesh_scale", 1.0))], dtype=torch.float32
    )
    if not torch.isfinite(mesh_scale).all() or torch.any(mesh_scale <= 0):
        raise ValueError("mesh_scale must be finite and positive")
    return images, {
        "camera_angle_x": vector("camera_angle_x"),
        "distance": vector("distance"),
        "mesh_scale": mesh_scale,
        "transform_matrix": transforms,
    }


def pil_views_to_tensor(images, image_size, device):
    tensors = []
    for view in images:
        resized = view.resize((image_size, image_size), Image.Resampling.LANCZOS)
        array = np.asarray(resized.convert("RGB"), dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array.copy()).permute(2, 0, 1))
    return torch.stack(tensors).unsqueeze(0).to(device)
```

- [ ] **Step 4: Extend both projection condition builders without changing sparse lookup**

Add `transform_matrix=None` to `get_proj_cond_ss` and `get_proj_cond_shape`. At the top of each builder, replace scalar tensor creation with:

```python
num_views = len(image)
image_tensor = pil_views_to_tensor(image, image_cond_model.image_size, device)
camera_angle_x = torch.as_tensor(camera_angle_x, dtype=torch.float32, device=device).reshape(1, num_views)
distance = torch.as_tensor(distance, dtype=torch.float32, device=device).reshape(1, num_views)
mesh_scale = torch.as_tensor(mesh_scale, dtype=torch.float32, device=device).reshape(1)
if transform_matrix is not None:
    transform_matrix = torch.as_tensor(
        transform_matrix, dtype=torch.float32, device=device
    ).reshape(1, num_views, 4, 4)
if num_views == 1 and transform_matrix is None:
    image_tensor = image_tensor[:, 0]
    camera_angle_x = camera_angle_x[:, 0]
    distance = distance[:, 0]
z_global, z_proj = image_cond_model(
    image_tensor,
    camera_angle_x=camera_angle_x,
    distance=distance,
    mesh_scale=mesh_scale,
    transform_matrix=transform_matrix,
)
```

In `get_proj_cond_shape`, preserve the existing `grid_resolution_override`, dense-grid reshape, sparse coordinate extraction, and restoration code byte-for-byte around this replacement.

- [ ] **Step 5: Normalize once in `run` and pass the same ordered bundle to all four condition builders**

Replace the current scalar extraction and image preprocessing block with:

```python
images, cameras = normalize_calibrated_views(image, camera_params)
if preprocess_image:
    images = [self.preprocess_image(view) for view in images]
camera_angle_x = cameras["camera_angle_x"]
distance = cameras["distance"]
mesh_scale = cameras["mesh_scale"]
transform_matrix = cameras["transform_matrix"]
torch.manual_seed(seed)
```

Replace each `[image]` argument with `images` and pass `transform_matrix=transform_matrix` in the SS-64, Shape-512, Shape-1024, and PBR-1024 condition calls. Do not change sampler inputs or cache DINOv3/NAF features.

- [ ] **Step 6: Add a four-conditioner propagation test**

Append a recording conditioner and call both condition builders for the four inference roles:

```python
class RecordingConditioner(torch.nn.Module):
    def __init__(self, image_size=8, grid_resolution=2):
        super().__init__()
        self.image_size = image_size
        self.grid_resolution = grid_resolution
        self.proj_grid = type("Grid", (), {"image_resolution": image_size})()
        self.calls = []

    def forward(self, image, **camera):
        self.calls.append((image.shape, camera["transform_matrix"].shape))
        batch = image.shape[0]
        return torch.zeros(batch, 5, 4), torch.zeros(batch, self.grid_resolution ** 3, 4)


def test_all_four_inference_conditioners_receive_the_same_k2_bundle():
    from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline

    pipeline = Pixal3DImageTo3DPipeline()
    pipeline._device = "cpu"
    pipeline.low_vram = False
    conditioners = [RecordingConditioner() for _ in range(4)]
    pipeline.image_cond_model_ss = conditioners[0]
    images = [image("red"), image("blue")]
    transforms = torch.eye(4).repeat(2, 1, 1)
    cameras = {"camera_angle_x": [[0.7, 0.8]], "distance": [[2.5, 2.7]],
               "mesh_scale": [1.0], "transform_matrix": transforms[None]}
    pipeline.get_proj_cond_ss(images, **cameras)
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    for conditioner in conditioners[1:]:
        pipeline.get_proj_cond_shape(conditioner, images, coords, **cameras)
    assert [call[0][1] for model in conditioners for call in model.calls] == [2, 2, 2, 2]
    assert [call[1][1] for model in conditioners for call in model.calls] == [2, 2, 2, 2]
```

- [ ] **Step 7: Run pipeline, conditioner, and single-view import tests**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_pipeline_inputs.py tests/multiview/test_conditioner.py -q`

Expected: all tests pass.

Run: `conda run --no-capture-output -n pixal3d python inference.py --help`

Expected: exit 0 and the existing `--image` option is still present.

- [ ] **Step 8: Commit four-model cascade propagation**

```bash
git add tests/multiview/test_pipeline_inputs.py pixal3d/pipelines/pixal3d_image_to_3d.py
git commit -m "feat: propagate calibrated views through the full cascade"
```

---

### Task 9: Add Strict Calibrated `transforms.json` Inference

**Files:**
- Create: `tests/multiview/test_inference_manifest.py`
- Modify: `inference.py:1-250`
- Modify: `inference.py` argument parser

**Interfaces:**
- Consumes: an existing render-format manifest with one through eight ordered frames.
- Produces: `load_calibrated_manifest(path, mesh_scale) -> tuple[list[Image.Image], dict]` and mutually exclusive `--image`/`--transforms` CLI inputs.

- [ ] **Step 1: Write failing manifest tests**

```python
import json

import numpy as np
import pytest
import torch
from PIL import Image

from inference import load_calibrated_manifest, load_flow_overrides


def test_manifest_preserves_first-frame_anchor_order(tmp_path):
    frames = []
    for index in range(2):
        Image.new("RGBA", (4, 4), color=(index * 10, 0, 0, 255)).save(
            tmp_path / f"{index:03d}.png"
        )
        transform = np.eye(4, dtype=np.float32)
        transform[0, 3] = index
        transform[2, 3] = 2.0
        frames.append({
            "file_path": f"{index:03d}.png",
            "camera_angle_x": 0.7,
            "transform_matrix": transform.tolist(),
        })
    manifest = tmp_path / "transforms.json"
    manifest.write_text(json.dumps({"frames": frames}))
    images, cameras = load_calibrated_manifest(manifest, mesh_scale=1.25)
    assert len(images) == 2
    assert cameras["camera_angle_x"] == [0.7, 0.7]
    assert cameras["distance"] == pytest.approx([2.0, np.sqrt(5.0)])
    assert np.asarray(cameras["transform_matrix"]).shape == (2, 4, 4)
    assert cameras["mesh_scale"] == 1.25


def test_manifest_rejects_more_than_eight_frames(tmp_path):
    (tmp_path / "transforms.json").write_text(json.dumps({"frames": [{}] * 9}))
    with pytest.raises(ValueError, match="between 1 and 8"):
        load_calibrated_manifest(tmp_path / "transforms.json", mesh_scale=1.0)


def test_manifest_rejects_frame_path_escape(tmp_path):
    manifest = tmp_path / "transforms.json"
    manifest.write_text(json.dumps({"frames": [{
        "file_path": "../outside.png",
        "camera_angle_x": 0.7,
        "transform_matrix": np.eye(4).tolist(),
    }]}))
    with pytest.raises(ValueError, match="inside"):
        load_calibrated_manifest(manifest, mesh_scale=1.0)


def test_flow_overrides_load_the_four_exact_pipeline_models(tmp_path):
    model_keys = (
        "sparse_structure_flow_model",
        "shape_slat_flow_model_512",
        "shape_slat_flow_model_1024",
        "tex_slat_flow_model_1024",
    )
    pipeline = type("Pipeline", (), {})()
    pipeline.models = {key: torch.nn.Linear(2, 2, bias=False) for key in model_keys}
    overrides = {}
    for index, key in enumerate(model_keys):
        checkpoint = tmp_path / f"{key}.pt"
        torch.save({"weight": torch.full((2, 2), float(index + 1))}, checkpoint)
        overrides[key] = checkpoint
    load_flow_overrides(pipeline, overrides)
    for index, key in enumerate(model_keys):
        assert torch.equal(
            pipeline.models[key].weight,
            torch.full((2, 2), float(index + 1)),
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_inference_manifest.py -q`

Expected: collection fails because `load_calibrated_manifest` does not exist.

- [ ] **Step 3: Implement strict manifest loading**

Add `json`, `Path`, and `Optional` imports, then add:

```python
def load_calibrated_manifest(path, *, mesh_scale):
    path = Path(path).resolve()
    metadata = json.loads(path.read_text())
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= 8:
        raise ValueError("transforms.json must contain between 1 and 8 frames")
    images = []
    angles = []
    distances = []
    transforms = []
    for index, frame in enumerate(frames):
        image_path = (path.parent / frame["file_path"]).resolve()
        if not image_path.is_relative_to(path.parent):
            raise ValueError("frame file_path must remain inside the manifest directory")
        if not image_path.is_file():
            raise FileNotFoundError(f"missing calibrated frame {index}: {image_path}")
        angle = frame.get("camera_angle_x", metadata.get("camera_angle_x"))
        if angle is None or not np.isfinite(float(angle)):
            raise ValueError(f"camera_angle_x must be finite for frame {index}")
        transform = np.asarray(frame.get("transform_matrix"), dtype=np.float32)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"transform_matrix must be finite [4, 4] for frame {index}")
        images.append(Image.open(image_path).convert("RGBA"))
        angles.append(float(angle))
        distances.append(float(np.linalg.norm(transform[:3, 3])))
        transforms.append(transform)
    return images, {
        "camera_angle_x": angles,
        "distance": distances,
        "mesh_scale": float(mesh_scale),
        "transform_matrix": np.stack(transforms),
    }
```

- [ ] **Step 4: Keep the single-image branch unchanged and add a calibrated branch**

Change `run_inference` to accept `image_path: Optional[str]` and `transforms_path: Optional[str] = None`. Before the current single-image preprocessing/MoGe block, branch as follows:

```python
if transforms_path is not None:
    print(f"[Inference] Loading calibrated views: {transforms_path}")
    images, camera_params = load_calibrated_manifest(
        transforms_path, mesh_scale=mesh_scale
    )
    image_preprocessed = [pipeline.preprocess_image(view) for view in images]
else:
    if image_path is None:
        raise ValueError("image_path is required without transforms_path")
    print(f"[Inference] Processing image: {image_path}")
    img = Image.open(image_path)
    image_preprocessed = pipeline.preprocess_image(img)
    tmp_path = os.path.join(
        os.path.dirname(os.path.abspath(output_path)),
        f"_tmp_preprocessed_{int(time.time() * 1000)}.png",
    )
    image_preprocessed.save(tmp_path)
    if manual_fov > 0:
        camera_angle_x = float(manual_fov)
        grid_point = torch.tensor([-1.0, 0.0, 0.0])
        distance = distance_from_fov(
            camera_angle_x,
            grid_point,
            torch.tensor([
                0 - extend_pixel,
                image_resolution - 1 + extend_pixel,
            ]),
            mesh_scale,
            image_resolution,
        )["distance_from_x"]
        camera_params = {
            "camera_angle_x": camera_angle_x,
            "distance": distance,
            "mesh_scale": mesh_scale,
        }
        print(
            f"[Inference] Using manual FOV: {math.degrees(manual_fov):.2f}° "
            f"({manual_fov:.4f} rad), distance={distance:.4f}"
        )
    else:
        print("[MoGe-2] Loading model for camera estimation...")
        moge_model = load_moge_model(device="cuda")
        print("[Inference] Estimating camera parameters...")
        camera_params = get_camera_params_wild_moge(
            tmp_path,
            moge_model,
            device="cuda",
            mesh_scale=mesh_scale,
            extend_pixel=extend_pixel,
            image_resolution=image_resolution,
        )
        print(
            f"  camera_angle_x={camera_params['camera_angle_x']:.4f}, "
            f"distance={camera_params['distance']:.4f}"
        )
        moge_model.cpu()
        del moge_model
        torch.cuda.empty_cache()
    os.remove(tmp_path)
```

The existing sampler and export body remains below the branch and passes `image_preprocessed` to `pipeline.run` in either form.

- [ ] **Step 5: Make CLI inputs mutually exclusive**

```python
inputs = parser.add_mutually_exclusive_group(required=True)
inputs.add_argument(
    "--image", help="Single input image; camera may be estimated with MoGe-2"
)
inputs.add_argument(
    "--transforms",
    help="Calibrated transforms.json with the first frame as anchor",
)
```

Pass `transforms_path=args.transforms` in the final `run_inference` call; retain all existing CLI options.

- [ ] **Step 6: Add strict optional fine-tuned flow checkpoint overrides**

Add this exact mapping and loader after `init_pipeline`:

```python
FLOW_MODEL_KEYS = (
    "sparse_structure_flow_model",
    "shape_slat_flow_model_512",
    "shape_slat_flow_model_1024",
    "tex_slat_flow_model_1024",
)


def load_flow_overrides(pipeline, overrides):
    unknown = set(overrides) - set(FLOW_MODEL_KEYS)
    if unknown:
        raise ValueError(f"unknown flow checkpoint overrides: {sorted(unknown)}")
    for model_key in FLOW_MODEL_KEYS:
        checkpoint = overrides.get(model_key)
        if checkpoint is None:
            continue
        checkpoint = Path(checkpoint)
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        incompatible = pipeline.models[model_key].load_state_dict(state_dict, strict=False)
        missing = set(incompatible.missing_keys)
        if missing - {"rope_phases"} or incompatible.unexpected_keys:
            raise RuntimeError(
                f"incompatible {model_key} checkpoint {checkpoint}: "
                f"missing={sorted(missing)} unexpected={incompatible.unexpected_keys}"
            )
        print(f"[Pipeline] Loaded {model_key}: {checkpoint}")
```

Add `flow_checkpoints: Optional[dict] = None` to `run_inference` and call `load_flow_overrides(pipeline, flow_checkpoints or {})` immediately after `init_pipeline`. Add these four CLI arguments:

```python
parser.add_argument("--ss_ckpt")
parser.add_argument("--shape512_ckpt")
parser.add_argument("--shape1024_ckpt")
parser.add_argument("--pbr1024_ckpt")
parser.add_argument("--mesh_scale", type=float, default=1.0)
```

Construct and filter the following mapping immediately before the final `run_inference` call:

```python
flow_checkpoints = {
    "sparse_structure_flow_model": args.ss_ckpt,
    "shape_slat_flow_model_512": args.shape512_ckpt,
    "shape_slat_flow_model_1024": args.shape1024_ckpt,
    "tex_slat_flow_model_1024": args.pbr1024_ckpt,
}
flow_checkpoints = {
    model_key: checkpoint
    for model_key, checkpoint in flow_checkpoints.items()
    if checkpoint is not None
}
```

Pass `flow_checkpoints=flow_checkpoints` and `mesh_scale=args.mesh_scale`. This permits stage-by-stage smoke inference with the newly trained stages and released weights for unfinished stages without exporting a new model repository.

- [ ] **Step 7: Run manifest, override, and CLI regression tests**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_inference_manifest.py tests/multiview/test_pipeline_inputs.py -q`

Expected: all manifest and four-model override tests pass.

Run: `conda run --no-capture-output -n pixal3d python inference.py --help`

Expected: usage contains `(--image IMAGE | --transforms TRANSFORMS)` and all four checkpoint flags.

- [ ] **Step 8: Commit calibrated manifest inference**

```bash
git add tests/multiview/test_inference_manifest.py inference.py
git commit -m "feat: add calibrated multiview inference manifests"
```

---

### Task 10: Add an Explicit Bounded-Smoke Runtime Override

**Files:**
- Create: `tests/multiview/test_train_smoke_override.py`
- Modify: `train.py:150-205`

**Interfaces:**
- Consumes: an already loaded experiment config and `--smoke_steps` equal to 1 or 10.
- Produces: smoke-only trainer intervals while leaving the committed long-training JSON unchanged.

- [ ] **Step 1: Write a failing pure config test**

```python
from easydict import EasyDict as edict

from train import apply_smoke_overrides


def test_ten_step_smoke_logs_every_step_samples_five_and_ten_and_saves_ten():
    config = edict({"trainer": {"args": {
        "max_steps": 1_000_000,
        "i_log": 5,
        "i_sample": 250,
        "i_save": 1000,
    }}})
    apply_smoke_overrides(config, 10)
    assert config.trainer.args.max_steps == 10
    assert config.trainer.args.i_log == 1
    assert config.trainer.args.i_sample == 5
    assert config.trainer.args.i_save == 10


def test_one_step_online_gate_samples_and_saves_step_one():
    config = edict({"trainer": {"args": {
        "max_steps": 1_000_000,
        "i_log": 5,
        "i_sample": 250,
        "i_save": 1000,
    }}})
    apply_smoke_overrides(config, 1)
    assert config.trainer.args.max_steps == 1
    assert config.trainer.args.i_log == 1
    assert config.trainer.args.i_sample == 1
    assert config.trainer.args.i_save == 1
```

- [ ] **Step 2: Run the test and verify RED**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_train_smoke_override.py -q`

Expected: import failure for `apply_smoke_overrides`.

- [ ] **Step 3: Implement and apply the smoke-only override**

```python
def apply_smoke_overrides(cfg, smoke_steps):
    if smoke_steps is None:
        return cfg
    if smoke_steps not in (1, 10):
        raise ValueError("smoke_steps must be 1 or 10")
    cfg.trainer.args.max_steps = smoke_steps
    cfg.trainer.args.i_log = 1
    cfg.trainer.args.i_sample = 1 if smoke_steps == 1 else 5
    cfg.trainer.args.i_save = smoke_steps
    cfg.trainer.args.snapshot_batch_size = 1
    cfg.trainer.args.snapshot_num_samples = 1
    cfg.trainer.args.num_workers = 0
    cfg.trainer.args.prefetch_data = False
    return cfg
```

Add `parser.add_argument('--smoke_steps', type=int, choices=(1, 10))`. After command-line/config merge and before printing/saving the resolved config, call `apply_smoke_overrides(cfg, opt.smoke_steps)`. Because the JSON config is merged after CLI arguments, read the value from `opt.smoke_steps`, not `cfg.smoke_steps`.

- [ ] **Step 4: Run smoke override and CLI tests**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_train_smoke_override.py -q`

Expected: `2 passed`.

Run: `conda run --no-capture-output -n pixal3d python train.py --help | rg -- '--smoke_steps'`

Expected: one help line documents `{1,10}`.

- [ ] **Step 5: Commit the bounded smoke control**

```bash
git add tests/multiview/test_train_smoke_override.py train.py
git commit -m "feat: add bounded multiview training smoke mode"
```

---

### Task 11: Pass the Complete Model Validation Gate Before W&B Changes

**Files:**
- Create: `tests/multiview/test_pilot_dataset.py`
- Create: `tests/multiview/test_gpu_conditioning.py`
- Runtime output: `/root/node17/data/pixal3d/train/reports/multiview/model-validation`

**Interfaces:**
- Consumes: Tasks 1-10, four materialized pilot roots, and four converted released checkpoints.
- Produces: unit/pilot/checkpoint/GPU evidence; no W&B code or run is allowed until every step passes.

- [ ] **Step 1: Add an opt-in real-pilot dataset gate for all four stages and both anchors**

```python
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from pixal3d import datasets
from tests.multiview.test_configs import CONFIGS


PILOT = Path("/root/node17/data/pixal3d/train/development/abo-pilot64")
STAGE_ROOTS = {
    "ss64": {
        "base": PILOT / "ss64/active",
        "render_cond": PILOT / "ss64/active/renders_cond",
        "ss_latent": PILOT / "ss64/active/ss_latents/ss_enc_conv3d_16l8_fp16_64_view",
    },
    "shape512": {
        "base": PILOT / "shape512/active",
        "render_cond": PILOT / "shape512/active/renders_cond",
        "shape_latent": PILOT / "shape512/active/shape_latents/shape_enc_next_dc_f16c32_fp16_512_view",
    },
    "shape1024": {
        "base": PILOT / "shape1024/active",
        "render_cond": PILOT / "shape1024/active/renders_cond",
        "shape_latent": PILOT / "shape1024/active/shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view",
    },
    "pbr1024": {
        "base": PILOT / "pbr1024/active",
        "render_cond": PILOT / "pbr1024/active/renders_cond",
        "shape_latent": PILOT / "pbr1024/active/shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view",
        "pbr_latent": PILOT / "pbr1024/active/pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix",
    },
}


def require_pilot():
    if not PILOT.exists():
        if os.environ.get("PIXAL3D_REQUIRE_PILOT") == "1":
            pytest.fail(f"pilot root is missing: {PILOT}")
        pytest.skip(f"pilot root is missing: {PILOT}")


def make_dataset(stage):
    config = json.loads(CONFIGS[stage].read_text())
    roots = {"ABO": {key: str(value) for key, value in STAGE_ROOTS[stage].items()}}
    return getattr(datasets, config["dataset"]["name"])(
        json.dumps(roots), **config["dataset"]["args"]
    )


@pytest.mark.integration
@pytest.mark.parametrize("stage", tuple(CONFIGS))
@pytest.mark.parametrize("anchor", [0, 1])
def test_pilot_stage_loads_anchor_first(stage, anchor, monkeypatch):
    require_pilot()
    dataset = make_dataset(stage)
    assert len(dataset) == 64
    root, asset, _ = dataset.instances[0]
    monkeypatch.setattr(np.random, "randint", lambda low, high: anchor)
    pack = dataset.get_instance(root, asset)
    assert pack["view_idx"] == anchor
    assert pack["view_indices"][0].item() == anchor
    assert sorted(pack["view_indices"].tolist()) == list(range(8))
    assert pack["cond"].shape[0] == 8
    assert pack["transform_matrix"].shape == (8, 4, 4)
    assert torch.isfinite(pack["cond"]).all()
    assert torch.isfinite(pack["transform_matrix"]).all()


@pytest.mark.integration
@pytest.mark.parametrize("stage", tuple(CONFIGS))
@pytest.mark.parametrize("num_views", [2, 6])
def test_pilot_collation_forces_batchwide_endpoint(stage, num_views, monkeypatch):
    require_pilot()
    dataset = make_dataset(stage)
    first = dataset[0]
    second = dataset[1]
    monkeypatch.setattr(np.random, "randint", lambda low, high: num_views)
    batch = dataset.collate_fn([first, second])
    assert batch["cond"].shape[:2] == (2, num_views)
    assert batch["camera_angle_x"].shape == (2, num_views)
    assert batch["transform_matrix"].shape == (2, num_views, 4, 4)
```

- [ ] **Step 2: Run math, dataset, config, and full repository tests on CPU**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview -q -m 'not gpu and not integration'`

Expected: zero failures and zero unexpected skips.

Run: `PIXAL3D_REQUIRE_PILOT=1 PIXAL3D_REQUIRE_CHECKPOINTS=1 conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_pilot_dataset.py tests/multiview/test_configs.py -q -m integration`

Expected: `20 passed`—16 pilot anchor/collation cases and four checkpoint cases—with no skip.

Run: `conda run --no-capture-output -n pixal3d python -m pytest -q`

Expected: all baseline and new tests pass.

- [ ] **Step 3: Add the opt-in real DINOv3/NAF K=2/K=6 condition gate**

```python
import json
import os
from pathlib import Path

import pytest
import torch

from pixal3d.datasets.components import load_anchor_first_conditions
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    DinoV3ProjFeatureExtractor,
)
from tests.multiview.test_configs import CONFIGS
from tests.multiview.test_pilot_dataset import PILOT, STAGE_ROOTS


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize("stage", tuple(CONFIGS))
@pytest.mark.parametrize("num_views", [2, 6])
def test_real_conditioner_outputs_are_finite(stage, num_views):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if os.environ.get("PIXAL3D_RUN_GPU_SMOKE") != "1":
        pytest.skip("set PIXAL3D_RUN_GPU_SMOKE=1 after the GPU0 resource gate")
    config = json.loads(CONFIGS[stage].read_text())
    roots = STAGE_ROOTS[stage]
    asset = sorted(path.name for path in roots["render_cond"].iterdir() if path.is_dir())[0]
    conditions = load_anchor_first_conditions(
        roots["render_cond"] / asset,
        anchor_index=0,
        image_size=config["dataset"]["args"]["image_size"],
        other_view_indices=[1, 2, 3, 4, 5, 6, 7],
    )
    latent_key = "pbr_latent" if stage == "pbr1024" else (
        "ss_latent" if stage == "ss64" else "shape_latent"
    )
    scale = json.loads((roots[latent_key] / asset / "view00_scale.json").read_text())
    model_args = config["trainer"]["args"]["image_cond_model"]["args"]
    conditioner = DinoV3ProjFeatureExtractor(**model_args).cuda().eval()
    with torch.no_grad():
        global_feature, projected_feature = conditioner(
            conditions["cond"][:num_views][None].cuda(),
            camera_angle_x=conditions["camera_angle_x"][:num_views][None].cuda(),
            distance=conditions["camera_distance"][:num_views][None].cuda(),
            mesh_scale=torch.tensor([scale["total_scale"]], device="cuda"),
            transform_matrix=conditions["transform_matrix"][:num_views][None].cuda(),
        )
    assert torch.isfinite(global_feature).all()
    assert torch.isfinite(projected_feature).all()
    assert global_feature.shape[0] == projected_feature.shape[0] == 1


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize("stage", tuple(CONFIGS))
def test_real_k1_matches_existing_single_view_path(stage):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if os.environ.get("PIXAL3D_RUN_GPU_SMOKE") != "1":
        pytest.skip("set PIXAL3D_RUN_GPU_SMOKE=1 after the GPU0 resource gate")
    config = json.loads(CONFIGS[stage].read_text())
    roots = STAGE_ROOTS[stage]
    asset = sorted(path.name for path in roots["render_cond"].iterdir() if path.is_dir())[0]
    conditions = load_anchor_first_conditions(
        roots["render_cond"] / asset,
        anchor_index=0,
        image_size=config["dataset"]["args"]["image_size"],
        other_view_indices=[1, 2, 3, 4, 5, 6, 7],
    )
    latent_key = "pbr_latent" if stage == "pbr1024" else (
        "ss_latent" if stage == "ss64" else "shape_latent"
    )
    scale = json.loads((roots[latent_key] / asset / "view00_scale.json").read_text())
    conditioner = DinoV3ProjFeatureExtractor(
        **config["trainer"]["args"]["image_cond_model"]["args"]
    ).cuda().eval()
    image = conditions["cond"][0][None].cuda()
    angle = conditions["camera_angle_x"][0][None].cuda()
    distance = conditions["camera_distance"][0][None].cuda()
    mesh_scale = torch.tensor([scale["total_scale"]], device="cuda")
    transform = conditions["transform_matrix"][0][None, None].cuda()
    with torch.no_grad():
        legacy_global, legacy_proj = conditioner(
            image,
            camera_angle_x=angle,
            distance=distance,
            mesh_scale=mesh_scale,
        )
        k1_global, k1_proj = conditioner(
            image[:, None],
            camera_angle_x=angle[:, None],
            distance=distance[:, None],
            mesh_scale=mesh_scale,
            transform_matrix=transform,
        )
    torch.testing.assert_close(k1_global.float(), legacy_global.float(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k1_proj.float(), legacy_proj.float(), rtol=1e-5, atol=1e-5)
```

- [ ] **Step 4: Perform the fresh GPU0 and preprocessing-process admission check**

Run:

```bash
conda run --no-capture-output -n pixal3d python -m data_toolkit.pipeline.cli workers \
  --config data_toolkit/configs/multiview_preprocess.yaml --action status
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits
pgrep -af 'blender|voxelize|encode_.*latent|dual_grid_view|packing|production_worker'
```

Expected while preprocessing is active: node16/node17 status is understood, physical GPU0 has enough free VRAM for the selected stage, and no preprocessing command owns GPU0. If GPU0 is occupied or resource state is ambiguous, stop here and wait; do not drain or kill a worker implicitly.

- [ ] **Step 5: Run real K=2/K=6 encoding sequentially on physical GPU0**

Run: `CUDA_VISIBLE_DEVICES=0 PIXAL3D_RUN_GPU_SMOKE=1 conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_gpu_conditioning.py -q -m 'gpu and integration'`

Expected: `12 passed`, executed sequentially; four real K=1 regressions and eight K=2/K=6 finite-output cases, with no NaN, Inf, or CUDA OOM.

- [ ] **Step 6: Define the exact pilot data-dir JSON values for trainer validation**

Use these literal shell variables for the next steps:

```bash
PIXAL3D_SS64_DATA='{"ABO":{"base":"/root/node17/data/pixal3d/train/development/abo-pilot64/ss64/active","render_cond":"/root/node17/data/pixal3d/train/development/abo-pilot64/ss64/active/renders_cond","ss_latent":"/root/node17/data/pixal3d/train/development/abo-pilot64/ss64/active/ss_latents/ss_enc_conv3d_16l8_fp16_64_view"}}'
PIXAL3D_SHAPE512_DATA='{"ABO":{"base":"/root/node17/data/pixal3d/train/development/abo-pilot64/shape512/active","render_cond":"/root/node17/data/pixal3d/train/development/abo-pilot64/shape512/active/renders_cond","shape_latent":"/root/node17/data/pixal3d/train/development/abo-pilot64/shape512/active/shape_latents/shape_enc_next_dc_f16c32_fp16_512_view"}}'
PIXAL3D_SHAPE1024_DATA='{"ABO":{"base":"/root/node17/data/pixal3d/train/development/abo-pilot64/shape1024/active","render_cond":"/root/node17/data/pixal3d/train/development/abo-pilot64/shape1024/active/renders_cond","shape_latent":"/root/node17/data/pixal3d/train/development/abo-pilot64/shape1024/active/shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view"}}'
PIXAL3D_PBR1024_DATA='{"ABO":{"base":"/root/node17/data/pixal3d/train/development/abo-pilot64/pbr1024/active","render_cond":"/root/node17/data/pixal3d/train/development/abo-pilot64/pbr1024/active/renders_cond","shape_latent":"/root/node17/data/pixal3d/train/development/abo-pilot64/pbr1024/active/shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view","pbr_latent":"/root/node17/data/pixal3d/train/development/abo-pilot64/pbr1024/active/pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix"}}'
```

- [ ] **Step 7: Run one real optimizer step for every released inference model**

Run the following commands one at a time, repeating the Step 4 admission check before each command:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json \
  --output_dir /root/node17/data/pixal3d/train/reports/multiview/model-validation/ss64 \
  --data_dir "$PIXAL3D_SS64_DATA" --num_gpus 1 --smoke_steps 1 --auto_retry 0

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json \
  --output_dir /root/node17/data/pixal3d/train/reports/multiview/model-validation/shape512 \
  --data_dir "$PIXAL3D_SHAPE512_DATA" --num_gpus 1 --smoke_steps 1 --auto_retry 0

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  --output_dir /root/node17/data/pixal3d/train/reports/multiview/model-validation/shape1024 \
  --data_dir "$PIXAL3D_SHAPE1024_DATA" --num_gpus 1 --smoke_steps 1 --auto_retry 0

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  --output_dir /root/node17/data/pixal3d/train/reports/multiview/model-validation/pbr1024 \
  --data_dir "$PIXAL3D_PBR1024_DATA" --num_gpus 1 --smoke_steps 1 --auto_retry 0
```

Expected for each: one finite loss, finite gradient norm, parameters updated, and `denoiser_step0000001.pt`, EMA, and misc checkpoint files written. Any checkpoint-key warning, NaN/Inf, CUDA error, or skipped optimizer update fails the gate.

- [ ] **Step 8: Reload every saved smoke checkpoint without taking another step**

Run these four commands sequentially:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json \
  --output_dir /root/node17/data/pixal3d/train/reports/multiview/model-validation/ss64 \
  --load_dir /root/node17/data/pixal3d/train/reports/multiview/model-validation/ss64 \
  --data_dir "$PIXAL3D_SS64_DATA" --num_gpus 1 --ckpt 1 --tryrun --auto_retry 0

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json \
  --output_dir /root/node17/data/pixal3d/train/reports/multiview/model-validation/shape512 \
  --load_dir /root/node17/data/pixal3d/train/reports/multiview/model-validation/shape512 \
  --data_dir "$PIXAL3D_SHAPE512_DATA" --num_gpus 1 --ckpt 1 --tryrun --auto_retry 0

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  --output_dir /root/node17/data/pixal3d/train/reports/multiview/model-validation/shape1024 \
  --load_dir /root/node17/data/pixal3d/train/reports/multiview/model-validation/shape1024 \
  --data_dir "$PIXAL3D_SHAPE1024_DATA" --num_gpus 1 --ckpt 1 --tryrun --auto_retry 0

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  --output_dir /root/node17/data/pixal3d/train/reports/multiview/model-validation/pbr1024 \
  --load_dir /root/node17/data/pixal3d/train/reports/multiview/model-validation/pbr1024 \
  --data_dir "$PIXAL3D_PBR1024_DATA" --num_gpus 1 --ckpt 1 --tryrun --auto_retry 0
```

Expected: trainer initialization prints `Loading checkpoint from step 1... Done.` and exits 0 without training.

- [ ] **Step 9: Close the model gate with K=1/K=2 pipeline and scope checks**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_projection_geometry.py tests/multiview/test_conditioner.py tests/multiview/test_pipeline_inputs.py -q`

Expected: K=1, K=2, repeated-view, permutation, and all-four-conditioner tests pass under the exact FP32 tolerances.

Run: `rg -n 'view_mask|SetTransformer|learned.*fusion|visibility.*weight|pose_embedding|depth.*weight|alpha.*weight' pixal3d configs/gen`

Expected: no new baseline implementation match.

Run: `git diff --check && git status --short`

Expected: no whitespace errors; only intentional uncommitted validation-test changes remain.

- [ ] **Step 10: Commit opt-in validation coverage**

```bash
git add tests/multiview/test_pilot_dataset.py tests/multiview/test_gpu_conditioning.py
git commit -m "test: gate all four multiview flow models"
```

Do not begin Task 12 unless Steps 1-10 all pass.

---

### Task 12: Connect Multi-View W&B Visualization After Model Validation

**Files:**
- Create: `tests/multiview/test_wandb_multiview.py`
- Modify: `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py`
- Modify: `pixal3d/trainers/flow_matching/flow_matching.py`
- Modify: `pixal3d/trainers/flow_matching/sparse_flow_matching.py`
- Modify: `pixal3d/trainers/basic.py:992-1160`

**Interfaces:**
- Consumes: a collated 5D condition, `view_indices`, `multiview_stage`, dataset name, and asset SHA.
- Produces: W&B keys `multiview/k`, `samples/input_views`, and `samples/anchor`, while preserving existing loss/LR/gradient/generated/ground-truth logs.

- [ ] **Step 1: Write failing grid, scalar-key, metadata, and offline-serialization tests**

```python
import os

import numpy as np
import torch

from pixal3d.trainers.basic import batch_multiview_k
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    format_multiview_metadata,
    make_anchor_marked_view_grid,
)


def test_ordered_grid_marks_only_anchor_and_keeps_view_order():
    condition = torch.zeros(1, 3, 3, 6, 6)
    condition[:, 0] = 0.1
    condition[:, 1] = 0.2
    condition[:, 2] = 0.3
    grid = make_anchor_marked_view_grid(condition, border=1)
    assert grid.shape == (1, 3, 6, 18)
    assert torch.all(grid[:, 0, 0, :6] == 1.0)
    assert torch.all(grid[:, 1:, 0, :6] == 0.0)
    assert torch.allclose(grid[:, :, 1:-1, 7:11], torch.full((1, 3, 4, 4), 0.2))
    assert torch.allclose(grid[:, :, 1:-1, 13:17], torch.full((1, 3, 4, 4), 0.3))


def test_batch_multiview_k_returns_one_shared_k():
    data = [{"cond": torch.zeros(2, 4, 3, 8, 8)}]
    assert batch_multiview_k(data) == 4


def test_metadata_contains_stage_dataset_sha_k_anchor_and_order():
    caption = format_multiview_metadata(
        "shape512", "ABO", "a" * 64, torch.tensor([1, 7, 4, 0])
    )
    assert caption == (
        "stage=shape512 dataset=ABO sha=" + "a" * 64
        + " K=4 anchor=view01 views=[1,7,4,0]"
    )


def test_wandb_offline_serializes_exact_scalar_and_image_keys(tmp_path, monkeypatch):
    import wandb

    monkeypatch.setenv("WANDB_MODE", "offline")
    run = wandb.init(
        project="pixal3d-multiview",
        name="serialization-test",
        dir=str(tmp_path),
        config={"stage": "ss64", "dataset": "ABO"},
    )
    run.log({
        "multiview/k": 2,
        "samples/input_views": wandb.Image(np.zeros((8, 16, 3), dtype=np.uint8)),
        "samples/anchor": wandb.Image(np.zeros((8, 8, 3), dtype=np.uint8)),
    }, step=1)
    run.finish()
    assert list(tmp_path.glob("wandb/offline-run-*"))
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `WANDB_MODE=offline conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_wandb_multiview.py -q`

Expected: imports fail for the three new helpers.

- [ ] **Step 3: Create the anchor-marked ordered grid and exact caption formatter**

```python
def make_anchor_marked_view_grid(
    cond: torch.Tensor, border: int = 4
) -> torch.Tensor:
    if cond.ndim == 4:
        return cond
    if cond.ndim != 5:
        raise ValueError("condition must have shape [B,C,H,W] or [B,K,C,H,W]")
    batch_size, num_views, channels, height, width = cond.shape
    if channels != 3:
        raise ValueError("condition visualization requires RGB views")
    edge = min(border, height // 2, width // 2)
    views = cond.detach().clone()
    anchor = views[:, 0]
    for region in (
        (slice(None), slice(None), slice(0, edge), slice(None)),
        (slice(None), slice(None), slice(height - edge, height), slice(None)),
        (slice(None), slice(None), slice(None), slice(0, edge)),
        (slice(None), slice(None), slice(None), slice(width - edge, width)),
    ):
        anchor[region] = 0.0
        red_region = (region[0], 0, region[2], region[3])
        anchor[red_region] = 1.0
    views[:, 0] = anchor
    return views.permute(0, 2, 3, 1, 4).reshape(
        batch_size, channels, height, num_views * width
    )


def format_multiview_metadata(stage, dataset, sha, view_indices) -> str:
    indices = [int(value) for value in torch.as_tensor(view_indices).tolist()]
    order = ",".join(str(value) for value in indices)
    return (
        f"stage={stage} dataset={dataset} sha={sha} K={len(indices)} "
        f"anchor=view{indices[0]:02d} views=[{order}]"
    )
```

- [ ] **Step 4: Extend projection trainer visualization without removing legacy keys**

Change the mixin initializer to accept and store `multiview_stage: Optional[str] = None` before calling `super`. Replace `vis_cond` with:

```python
def vis_cond(self, cond, **kwargs):
    anchor = anchor_condition_image(cond)
    result = {"image": {"value": anchor, "type": "image"}}
    if cond.ndim == 5:
        result["input_views"] = {
            "value": make_anchor_marked_view_grid(cond),
            "type": "image",
        }
        result["anchor"] = {"value": anchor, "type": "image"}
    return result
```

Keeping `image` preserves the existing combined generated/ground-truth writers; the added keys are automatically logged by `BasicTrainer.snapshot` as `samples/input_views` and `samples/anchor`.

- [ ] **Step 5: Add exact batch K to normal scalar logs**

Add at module scope in `basic.py`:

```python
def batch_multiview_k(data_list):
    counts = {
        int(micro_batch["cond"].shape[1])
        for micro_batch in data_list
        if isinstance(micro_batch, dict)
        and isinstance(micro_batch.get("cond"), torch.Tensor)
        and micro_batch["cond"].ndim == 5
    }
    if not counts:
        return None
    if len(counts) != 1:
        raise ValueError("all micro-batches in one optimizer step must share K")
    return counts.pop()
```

At the end of `BasicTrainer.run_step`, before returning `step_log`, add:

```python
num_views = batch_multiview_k(data_list)
if num_views is not None:
    step_log["multiview"] = {"k": num_views}
```

`save_logs` already flattens this to the exact W&B scalar key `multiview/k`.

- [ ] **Step 6: Include stage/dataset/asset/view order in dense and sparse snapshot captions**

In every projection `run_snapshot` metadata loop in `flow_matching.py` and `sparse_flow_matching.py`, replace `dataset/sha` construction with:

```python
view_indices = data.get("view_indices")
for sample_index in range(batch):
    if view_indices is None:
        sample_metadata.append(
            f"stage={self.multiview_stage} dataset={data['_dataset_name'][sample_index]} "
            f"sha={data['_sha256'][sample_index]}"
        )
    else:
        sample_metadata.append(format_multiview_metadata(
            self.multiview_stage,
            data["_dataset_name"][sample_index],
            data["_sha256"][sample_index],
            view_indices[sample_index],
        ))
```

Use the corresponding actual batch bound (`batch` in dense loops, `min(num_samples, ...)` in sparse loops) and import `format_multiview_metadata`. Keep metadata removal after this block.

- [ ] **Step 7: Run offline W&B and all trainer regression tests**

Run: `WANDB_MODE=offline conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_wandb_multiview.py tests/multiview/test_trainer_views.py tests/multiview/test_train_smoke_override.py -q`

Expected: all tests pass; an offline run directory contains serialized config, `multiview/k`, `samples/input_views`, and `samples/anchor` data.

- [ ] **Step 8: Commit W&B integration**

```bash
git add tests/multiview/test_wandb_multiview.py pixal3d/trainers/basic.py pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py pixal3d/trainers/flow_matching/flow_matching.py pixal3d/trainers/flow_matching/sparse_flow_matching.py
git commit -m "feat: visualize multiview conditioning in wandb"
```

- [ ] **Step 9: Run one online optimizer step per stage on GPU0**

Repeat Task 11 Step 4 immediately before each command. Reuse its four `PIXAL3D_*_DATA` variables, then run sequentially:

```bash
WANDB_DIR=/root/node17/data/pixal3d/wandb CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json \
  --output_dir /root/node17/data/pixal3d/train/runs/multiview/ss64 \
  --data_dir "$PIXAL3D_SS64_DATA" --num_gpus 1 --smoke_steps 1 --auto_retry 0 \
  --use_wandb --wandb_project pixal3d-multiview --wandb_name mv-baseline-ss64 \
  --wandb_id mv-baseline-ss64-v1

WANDB_DIR=/root/node17/data/pixal3d/wandb CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json \
  --output_dir /root/node17/data/pixal3d/train/runs/multiview/shape512 \
  --data_dir "$PIXAL3D_SHAPE512_DATA" --num_gpus 1 --smoke_steps 1 --auto_retry 0 \
  --use_wandb --wandb_project pixal3d-multiview --wandb_name mv-baseline-shape512 \
  --wandb_id mv-baseline-shape512-v1

WANDB_DIR=/root/node17/data/pixal3d/wandb CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  --output_dir /root/node17/data/pixal3d/train/runs/multiview/shape1024 \
  --data_dir "$PIXAL3D_SHAPE1024_DATA" --num_gpus 1 --smoke_steps 1 --auto_retry 0 \
  --use_wandb --wandb_project pixal3d-multiview --wandb_name mv-baseline-shape1024 \
  --wandb_id mv-baseline-shape1024-v1

WANDB_DIR=/root/node17/data/pixal3d/wandb CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  --output_dir /root/node17/data/pixal3d/train/runs/multiview/pbr1024 \
  --data_dir "$PIXAL3D_PBR1024_DATA" --num_gpus 1 --smoke_steps 1 --auto_retry 0 \
  --use_wandb --wandb_project pixal3d-multiview --wandb_name mv-baseline-pbr1024 \
  --wandb_id mv-baseline-pbr1024-v1
```

Expected in each online run: correct stage/config artifact, finite `loss/*`, learning rate and gradient values, integer `multiview/k` in `[2,6]`, ordered `samples/input_views` with a red anchor border, `samples/anchor`, generated/ground-truth images, and captions containing the stage, `ABO`, asset SHA, K, anchor, and view order. Stop before Task 13 if any field is absent or misordered.

---

### Task 13: Run the Ordered 10-Step Pilot Smokes, Inference Smokes, and Full Training Handoff

**Files:**
- Modify: `README.md`
- Runtime input: four pilot roots, then audited `/root/node17/data/pixal3d/train/stage{1,2,3}/active` handoffs.
- Runtime output: `/root/node17/data/pixal3d/train/runs/multiview/{ss64,shape512,shape1024,pbr1024}` and K=2/K=4/K=6 GLBs.

**Interfaces:**
- Consumes: four one-step online runs from Task 12.
- Produces: exactly ten total pilot optimizer steps per stage, checkpoint reload evidence, stage-by-stage calibrated inference, then resumed long training in SS-64 → Shape-512 → Shape-1024 → PBR-1024 order.

- [ ] **Step 1: Resume SS-64 from step 1 through exactly step 10**

Repeat Task 11 Step 4, then run:

```bash
WANDB_DIR=/root/node17/data/pixal3d/wandb CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json \
  --output_dir /root/node17/data/pixal3d/train/runs/multiview/ss64 \
  --load_dir /root/node17/data/pixal3d/train/runs/multiview/ss64 --ckpt 1 \
  --data_dir "$PIXAL3D_SS64_DATA" --num_gpus 1 --smoke_steps 10 --auto_retry 0 \
  --use_wandb --wandb_project pixal3d-multiview --wandb_name mv-baseline-ss64 \
  --wandb_id mv-baseline-ss64-v1
```

Expected: steps 2-10 complete, scalars log every step, samples exist at steps 5 and 10, and step-10 model/EMA/misc checkpoints exist. Then run this reload-only gate:

```bash
WANDB_DIR=/root/node17/data/pixal3d/wandb CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json \
  --output_dir /root/node17/data/pixal3d/train/runs/multiview/ss64 \
  --load_dir /root/node17/data/pixal3d/train/runs/multiview/ss64 --ckpt 10 \
  --data_dir "$PIXAL3D_SS64_DATA" --num_gpus 1 --tryrun --auto_retry 0 \
  --use_wandb --wandb_project pixal3d-multiview --wandb_name mv-baseline-ss64 \
  --wandb_id mv-baseline-ss64-v1
```

Expected: initialization exits 0 without taking another optimizer step.

- [ ] **Step 2: Resume Shape-512 from step 1 through exactly step 10**

Run:

```bash
WANDB_DIR=/root/node17/data/pixal3d/wandb CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json \
  --output_dir /root/node17/data/pixal3d/train/runs/multiview/shape512 \
  --load_dir /root/node17/data/pixal3d/train/runs/multiview/shape512 --ckpt 1 \
  --data_dir "$PIXAL3D_SHAPE512_DATA" --num_gpus 1 --smoke_steps 10 --auto_retry 0 \
  --use_wandb --wandb_project pixal3d-multiview --wandb_name mv-baseline-shape512 \
  --wandb_id mv-baseline-shape512-v1

WANDB_DIR=/root/node17/data/pixal3d/wandb CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json \
  --output_dir /root/node17/data/pixal3d/train/runs/multiview/shape512 \
  --load_dir /root/node17/data/pixal3d/train/runs/multiview/shape512 --ckpt 10 \
  --data_dir "$PIXAL3D_SHAPE512_DATA" --num_gpus 1 --tryrun --auto_retry 0 \
  --use_wandb --wandb_project pixal3d-multiview --wandb_name mv-baseline-shape512 \
  --wandb_id mv-baseline-shape512-v1
```

Expected: the same step-5/step-10 finite metrics, samples, save, and step-10 reload gate as SS-64. The source is the released Shape-512 weight, not the newly trained SS-64 or any Shape-256 weight.

- [ ] **Step 3: Resume Shape-1024 from step 1 through exactly step 10**

Run:

```bash
WANDB_DIR=/root/node17/data/pixal3d/wandb CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  --output_dir /root/node17/data/pixal3d/train/runs/multiview/shape1024 \
  --load_dir /root/node17/data/pixal3d/train/runs/multiview/shape1024 --ckpt 1 \
  --data_dir "$PIXAL3D_SHAPE1024_DATA" --num_gpus 1 --smoke_steps 10 --auto_retry 0 \
  --use_wandb --wandb_project pixal3d-multiview --wandb_name mv-baseline-shape1024 \
  --wandb_id mv-baseline-shape1024-v1

WANDB_DIR=/root/node17/data/pixal3d/wandb CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  --output_dir /root/node17/data/pixal3d/train/runs/multiview/shape1024 \
  --load_dir /root/node17/data/pixal3d/train/runs/multiview/shape1024 --ckpt 10 \
  --data_dir "$PIXAL3D_SHAPE1024_DATA" --num_gpus 1 --tryrun --auto_retry 0 \
  --use_wandb --wandb_project pixal3d-multiview --wandb_name mv-baseline-shape1024 \
  --wandb_id mv-baseline-shape1024-v1
```

Expected: the same gate. Shape-1024 continues from its released Shape-1024 weight and must not initialize from the new Shape-512 run.

- [ ] **Step 4: Resume PBR-1024 from step 1 through exactly step 10**

Run:

```bash
WANDB_DIR=/root/node17/data/pixal3d/wandb CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  --output_dir /root/node17/data/pixal3d/train/runs/multiview/pbr1024 \
  --load_dir /root/node17/data/pixal3d/train/runs/multiview/pbr1024 --ckpt 1 \
  --data_dir "$PIXAL3D_PBR1024_DATA" --num_gpus 1 --smoke_steps 10 --auto_retry 0 \
  --use_wandb --wandb_project pixal3d-multiview --wandb_name mv-baseline-pbr1024 \
  --wandb_id mv-baseline-pbr1024-v1

WANDB_DIR=/root/node17/data/pixal3d/wandb CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  --output_dir /root/node17/data/pixal3d/train/runs/multiview/pbr1024 \
  --load_dir /root/node17/data/pixal3d/train/runs/multiview/pbr1024 --ckpt 10 \
  --data_dir "$PIXAL3D_PBR1024_DATA" --num_gpus 1 --tryrun --auto_retry 0 \
  --use_wandb --wandb_project pixal3d-multiview --wandb_name mv-baseline-pbr1024 \
  --wandb_id mv-baseline-pbr1024-v1
```

Expected: the same gate, with `concat_cond` retained and finite PBR loss/gradients.

- [ ] **Step 5: Create deterministic K=2/K=4/K=6 calibrated smoke manifests**

Use the first audited pilot asset as the fixed anchor source:

```bash
PIXAL3D_EVAL_ASSET=0006d4c69de70f84754df85c2ec0a34514223f941500f080c84c19da7e137998
PIXAL3D_EVAL_SOURCE_DIR=/root/node17/data/pixal3d/train/development/abo-pilot64/ss64/active/renders_cond/$PIXAL3D_EVAL_ASSET
PIXAL3D_EVAL_SOURCE=$PIXAL3D_EVAL_SOURCE_DIR/transforms.json
PIXAL3D_EVAL_ROOT=/root/node17/data/pixal3d/train/eval/multiview-pilot/$PIXAL3D_EVAL_ASSET
if [ -e "$PIXAL3D_EVAL_ROOT" ]; then
  printf 'refusing to overwrite existing evaluation root: %s\n' "$PIXAL3D_EVAL_ROOT" >&2
  exit 1
fi
mkdir -p "$PIXAL3D_EVAL_ROOT/k2" "$PIXAL3D_EVAL_ROOT/k4" "$PIXAL3D_EVAL_ROOT/k6"
for COUNT in 2 4 6; do
  conda run --no-capture-output -n pixal3d python -c \
    'import json,sys; data=json.load(open(sys.argv[1])); data["frames"]=data["frames"][:int(sys.argv[2])]; json.dump(data,sys.stdout,indent=2)' \
    "$PIXAL3D_EVAL_SOURCE" "$COUNT" > "$PIXAL3D_EVAL_ROOT/k$COUNT/transforms.json"
done
for COUNT in 2 4 6; do
  for INDEX in $(seq 0 $((COUNT - 1))); do
    printf -v VIEW '%03d' "$INDEX"
    ln -s "$PIXAL3D_EVAL_SOURCE_DIR/$VIEW.png" "$PIXAL3D_EVAL_ROOT/k$COUNT/$VIEW.png"
  done
done
```

The command refuses an existing evaluation root; inspect it instead of deleting or overwriting it. Read `view00_scale.json` from the SS-64 pilot latent and store its `total_scale` as `PIXAL3D_EVAL_SCALE`:

```bash
PIXAL3D_EVAL_SCALE=$(conda run --no-capture-output -n pixal3d python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["total_scale"])' \
  /root/node17/data/pixal3d/train/development/abo-pilot64/ss64/active/ss_latents/ss_enc_conv3d_16l8_fp16_64_view/$PIXAL3D_EVAL_ASSET/view00_scale.json)
```

- [ ] **Step 6: Run K=2/K=4/K=6 inference with all four step-10 checkpoints**

Run sequentially after a fresh GPU0 check:

```bash
for COUNT in 2 4 6; do
  CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pixal3d python inference.py \
    --transforms "$PIXAL3D_EVAL_ROOT/k$COUNT/transforms.json" \
    --mesh_scale "$PIXAL3D_EVAL_SCALE" \
    --ss_ckpt /root/node17/data/pixal3d/train/runs/multiview/ss64/ckpts/denoiser_ema0.9999_step0000010.pt \
    --shape512_ckpt /root/node17/data/pixal3d/train/runs/multiview/shape512/ckpts/denoiser_ema0.9999_step0000010.pt \
    --shape1024_ckpt /root/node17/data/pixal3d/train/runs/multiview/shape1024/ckpts/denoiser_ema0.9999_step0000010.pt \
    --pbr1024_ckpt /root/node17/data/pixal3d/train/runs/multiview/pbr1024/ckpts/denoiser_ema0.9999_step0000010.pt \
    --output "/root/node17/data/pixal3d/train/runs/multiview/inference/k$COUNT.glb" \
    --low_vram --resolution 1024 --seed 42
done
```

Expected: three non-empty GLBs, no camera estimation, identical first-frame anchor, and the same ordered K-view bundle reported for all four flow models. This gate verifies behavior only; do not add a new metric or model feature in response to output quality.

- [ ] **Step 7: Verify the pilot W&B runs before admitting full data**

For each of the four exact run names, confirm:

```text
steps: 1 through 10 present
multiview/k: every value is an integer in [2, 6]
loss and gradients: finite at every step
samples/input_views: step 5 and 10, red border on first view only
samples/anchor: equals the first unmarked condition view
captions: stage, ABO, 64-character asset SHA, K, anchor, ordered indices
checkpoint artifact/files: step 10 reload succeeded
```

Any failure blocks the next step.

- [ ] **Step 8: Require preprocessing drain and audited full handoffs**

Do not issue drain/kill commands as part of this plan. Wait for the preprocessing owner to drain node16/node17, then verify:

```bash
conda run --no-capture-output -n pixal3d python -m data_toolkit.pipeline.cli workers \
  --config data_toolkit/configs/multiview_preprocess.yaml --action status
pgrep -af 'blender|voxelize|encode_.*latent|dual_grid_view|packing|production_worker'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits
test -s /root/data2/pixal3d/control/splits/training_handoff.json
```

Expected: both workers are drained/removed as coordinated, process search has no active preprocessing task, GPUs have no preprocessing owner, and the immutable training handoff exists and passes its existing audit command. If any condition is false, full training remains blocked while pilot results stay valid.

- [ ] **Step 9: Point the same four configs at audited full stage roots**

Use these exact full-data mappings after the handoff audit confirms the directories:

```bash
PIXAL3D_FULL_SS64_DATA='{"Handoff":{"base":"/root/node17/data/pixal3d/train/stage1/active","render_cond":"/root/node17/data/pixal3d/train/stage1/active/renders_cond","ss_latent":"/root/node17/data/pixal3d/train/stage1/active/ss_latents/ss_enc_conv3d_16l8_fp16_64_view"}}'
PIXAL3D_FULL_SHAPE512_DATA='{"Handoff":{"base":"/root/node17/data/pixal3d/train/stage2/active","render_cond":"/root/node17/data/pixal3d/train/stage2/active/renders_cond","shape_latent":"/root/node17/data/pixal3d/train/stage2/active/shape_latents/shape_enc_next_dc_f16c32_fp16_512_view"}}'
PIXAL3D_FULL_SHAPE1024_DATA='{"Handoff":{"base":"/root/node17/data/pixal3d/train/stage2/active","render_cond":"/root/node17/data/pixal3d/train/stage2/active/renders_cond","shape_latent":"/root/node17/data/pixal3d/train/stage2/active/shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view"}}'
PIXAL3D_FULL_PBR1024_DATA='{"Handoff":{"base":"/root/node17/data/pixal3d/train/stage3/active","render_cond":"/root/node17/data/pixal3d/train/stage3/active/renders_cond","shape_latent":"/root/node17/data/pixal3d/train/stage3/active/shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view","pbr_latent":"/root/node17/data/pixal3d/train/stage3/active/pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix"}}'
```

Instantiate each dataset with `--tryrun` and require nonzero length before training.

- [ ] **Step 10: Resume long training sequentially from the verified step-10 pilot checkpoints**

Run SS-64 first:

```bash
WANDB_DIR=/root/node17/data/pixal3d/wandb \
conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json \
  --output_dir /root/node17/data/pixal3d/train/runs/multiview/ss64 \
  --load_dir /root/node17/data/pixal3d/train/runs/multiview/ss64 --ckpt 10 \
  --data_dir "$PIXAL3D_FULL_SS64_DATA" --num_gpus 7 --auto_retry 3 \
  --use_wandb --wandb_project pixal3d-multiview --wandb_name mv-baseline-ss64 \
  --wandb_id mv-baseline-ss64-v1
```

After SS-64 reaches its configured completion or an explicitly approved stopping gate, repeat the Step 8 GPU/process checks and run Shape-512:

```bash
WANDB_DIR=/root/node17/data/pixal3d/wandb \
conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json \
  --output_dir /root/node17/data/pixal3d/train/runs/multiview/shape512 \
  --load_dir /root/node17/data/pixal3d/train/runs/multiview/shape512 --ckpt 10 \
  --data_dir "$PIXAL3D_FULL_SHAPE512_DATA" --num_gpus 7 --auto_retry 3 \
  --use_wandb --wandb_project pixal3d-multiview --wandb_name mv-baseline-shape512 \
  --wandb_id mv-baseline-shape512-v1
```

After the same completion/stopping gate and fresh Step 8 checks, run Shape-1024:

```bash
WANDB_DIR=/root/node17/data/pixal3d/wandb \
conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  --output_dir /root/node17/data/pixal3d/train/runs/multiview/shape1024 \
  --load_dir /root/node17/data/pixal3d/train/runs/multiview/shape1024 --ckpt 10 \
  --data_dir "$PIXAL3D_FULL_SHAPE1024_DATA" --num_gpus 7 --auto_retry 3 \
  --use_wandb --wandb_project pixal3d-multiview --wandb_name mv-baseline-shape1024 \
  --wandb_id mv-baseline-shape1024-v1
```

After the same completion/stopping gate and fresh Step 8 checks, run PBR-1024:

```bash
WANDB_DIR=/root/node17/data/pixal3d/wandb \
conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  --output_dir /root/node17/data/pixal3d/train/runs/multiview/pbr1024 \
  --load_dir /root/node17/data/pixal3d/train/runs/multiview/pbr1024 --ckpt 10 \
  --data_dir "$PIXAL3D_FULL_PBR1024_DATA" --num_gpus 7 --auto_retry 3 \
  --use_wandb --wandb_project pixal3d-multiview --wandb_name mv-baseline-pbr1024 \
  --wandb_id mv-baseline-pbr1024-v1
```

The committed configs restore their original long `max_steps` and logging intervals. All four W&B runs resume their pilot histories, and every stage loads only its own step-10 checkpoint; never initialize one stage from another stage's new checkpoint.

- [ ] **Step 11: Document exact commands and baseline exclusions**

Add to `README.md`:

- the four model/config/checkpoint mapping;
- pilot materialization and strict validation commands;
- calibrated `--transforms` inference plus four optional fine-tuned checkpoint flags;
- W&B project and exact run names;
- one-step online gate, step-10 pilot resume, K=2/K=4/K=6 inference smoke, and full-training order;
- the requirement that full data is audited and preprocessing is drained;
- K=1..8 inference, K=2..6 batch-wide training, anchor-first semantics, and arithmetic mean fusion;
- explicit exclusions: learned fusion, weights/masks, camera estimation for multi-view, feature caching, progressive SS-32/Shape-256/PBR-256/PBR-512 training, and hyperparameter/loss/architecture improvements.

- [ ] **Step 12: Run final verification and commit the runbook**

Run: `conda run --no-capture-output -n pixal3d python -m pytest -q`

Expected: zero failures.

Run: `git diff --check && rg -n 'T[B]D|T[O]DO|FIXM[E]|implement[ ]later|fill[ ]in' README.md`

Expected: no whitespace error and no unresolved plan marker.

```bash
git add README.md
git commit -m "docs: add multiview validation and training runbook"
```

---

## Plan Self-Review Checklist

- [x] Tasks 1-5 implement only the approved camera geometry, 5D dispatch, arithmetic means, anchor-first data, batch-wide K, and snapshot compatibility.
- [x] Shape-512 is present as an independently initialized model in config, checkpoint, validation, W&B, pilot, inference, and long-training steps.
- [x] Tasks 6-7 use only audited pilot64 packs and exact matching released checkpoints without key rewriting.
- [x] Tasks 8-10 preserve K=1 and the existing single-image/MoGe path while propagating K=1..8 through all four flow models.
- [x] Task 11 completes math, pilot, strict checkpoint, K=1/K=2/K=6 conditioner, optimizer, save/reload, full-suite, and scope gates before W&B code.
- [x] Task 12 retains existing logging and adds exact `multiview/k`, `samples/input_views`, `samples/anchor`, and stage/dataset/SHA/order captions.
- [x] Task 13 enforces SS-64 → Shape-512 → Shape-1024 → PBR-1024 and blocks full training until preprocessing drains and the full handoff audits.
- [x] No task adds learned fusion, weighting, masking, pose estimation, caching, new metrics, curriculum models, losses, or denoiser architecture changes.
- [x] Every production behavior change has a failing test, minimal implementation, passing command, and focused commit.
