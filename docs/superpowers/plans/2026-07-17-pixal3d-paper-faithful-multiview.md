# Pixal3D Paper-Faithful Multi-View Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the current Pixal3D single-view projection conditioner to calibrated, batch-wide variable-K multi-view conditioning by porting the official paper implementation's anchor-relative transforms and arithmetic averaging.

**Architecture:** Keep the current TRELLIS.2 denoisers, DINOv3/NAF features, `ProjGrid`, trainer classes, and SS/Shape/PBR cascade. Add a five-dimensional branch to `DinoV3ProjFeatureExtractor`, load anchor-first camera batches from the existing eight-view data, and mean-reduce per-view global and projected features without learned fusion or masks.

**Tech Stack:** Python 3.11, PyTorch 2.8+, CUDA 12.8, torchvision, transformers DINOv3, NATTEN/NAF, pytest, Pillow, NumPy, existing Pixal3D/TRELLIS.2 modules.

## Global Constraints

- Work in conda environment `pixal3d`; run Python and pytest with `conda run --no-capture-output -n pixal3d`.
- Preserve the current single-view configs, APIs, checkpoint keys, denoiser modules, `ProjectAttention`, and `SparseProjectAttention`.
- Use the first condition view as the anchor.
- Use the official paper-branch transform: `relative = inverse(anchor_c2w) @ view_c2w`, then `projection = fixed_anchor @ relative`.
- Use a plain arithmetic mean for both projected features and global tokens.
- Add no learned fusion, pose embedding, confidence, visibility weighting, padding, or `view_mask`.
- Select one K per collated batch, uniformly from integer values 2 through 6; all samples in that batch share K.
- Accept calibrated K=1 through K=8 at inference.
- Keep the current DINOv3 and NAF channel layout; do not port DINOv2 or Direct3D-S2.
- Use `/root/node17/data/pixal3d` for local scratch and checkpoint materialization. Use `train/stage1/active`, `train/stage2/active`, and `train/stage3/active` for the SS, Shape, and PBR handoffs.
- Follow TDD for every behavior change and make one focused commit per task.

## File Map

- `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py`: explicit transforms, paper-relative camera math, 4D/5D conditioner dispatch, arithmetic averaging, and anchor-only visualization helpers.
- `pixal3d/datasets/components.py`: anchor-first eight-view image/camera loading and batch-wide K slicing helpers.
- `pixal3d/datasets/sparse_structure_latent.py`: dense SS multi-view dataset class and collation.
- `pixal3d/datasets/structured_latent_shape.py`: sparse Shape multi-view dataset class and collation.
- `pixal3d/datasets/structured_latent_svpbr.py`: sparse PBR multi-view dataset class and collation.
- `pixal3d/datasets/__init__.py`: lazy registrations for the three multi-view dataset classes.
- `pixal3d/trainers/flow_matching/flow_matching.py`: anchor-only dense snapshot camera values.
- `pixal3d/trainers/flow_matching/sparse_flow_matching.py`: anchor-only sparse snapshot camera values.
- `pixal3d/pipelines/pixal3d_image_to_3d.py`: single/multi-view condition normalization and cascade propagation.
- `inference.py`: calibrated `transforms.json` input while preserving existing `--image` behavior.
- `configs/gen/*_proj_multiview_*.json`: three final-resolution multi-view fine-tuning configs.
- `tests/multiview/`: geometry, conditioner, dataset, config, pipeline, CLI, and checkpoint compatibility tests.
- `README.md`: multi-view training/inference commands and baseline limitations.

---

### Task 1: Port the Official Anchor-Relative Projection Math

**Files:**
- Create: `tests/multiview/test_projection_geometry.py`
- Modify: `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py:27-232`

**Interfaces:**
- Consumes: camera-to-world tensors `transform_matrix: Tensor[B,K,4,4]`, distances `distance: Tensor[B,K]`, and `ProjGrid.front_view_transform_matrix`.
- Produces: `compute_multiview_projection_matrices(transform_matrix, distance, fixed_transform) -> tuple[Tensor[B,K,4,4], Tensor[B,K,4,4]]` and explicit-transform support in `ProjGrid.forward`.

- [ ] **Step 1: Write failing geometry tests**

```python
import torch

from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    ProjGrid,
    compute_multiview_projection_matrices,
)


def test_multiview_projection_matrices_match_paper_formula():
    anchor = torch.eye(4)
    anchor[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    second = torch.eye(4)
    second[:3, 3] = torch.tensor([-2.0, 1.5, 4.0])
    transforms = torch.stack([anchor, second])[None]
    distances = torch.tensor([[3.75, 4.72]])
    fixed = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -2.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    projection, relative = compute_multiview_projection_matrices(
        transforms, distances, fixed
    )

    expected_relative = torch.linalg.inv(anchor) @ transforms[0]
    expected_fixed = fixed.clone()
    expected_fixed[1, 3] = -distances[0, 0]
    expected_projection = expected_fixed @ expected_relative
    torch.testing.assert_close(relative[0], expected_relative)
    torch.testing.assert_close(projection[0], expected_projection)


def test_anchor_projection_is_current_fixed_front_view():
    grid = ProjGrid(grid_resolution=2, image_resolution=8)
    transforms = torch.eye(4).reshape(1, 1, 4, 4)
    distances = torch.tensor([[2.5]])
    projection, relative = compute_multiview_projection_matrices(
        transforms, distances, grid.front_view_transform_matrix
    )
    expected = grid.front_view_transform_matrix.clone()
    expected[1, 3] = -2.5
    torch.testing.assert_close(relative, torch.eye(4).reshape(1, 1, 4, 4))
    torch.testing.assert_close(projection[0, 0], expected)


def test_proj_grid_default_and_explicit_anchor_transform_match():
    grid = ProjGrid(grid_resolution=2, image_resolution=8)
    features = torch.arange(1 * 2 * 2 * 3, dtype=torch.float32).reshape(1, 2, 2, 3)
    fov = torch.tensor([0.7])
    distance = torch.tensor([2.5])
    scale = torch.tensor([1.0])
    explicit = grid.front_view_transform_matrix[None].clone()
    explicit[:, 1, 3] = -distance
    default_output = grid(features, fov, distance, scale)
    explicit_output = grid(features, fov, distance, scale, explicit)
    torch.testing.assert_close(explicit_output, default_output)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_projection_geometry.py -q`

Expected: collection fails because `compute_multiview_projection_matrices` does not exist; after importing is fixed, the explicit-transform test fails at `assert transform_matrix is None`.

- [ ] **Step 3: Add the exact paper-branch transform helper**

Add above `ProjGrid`:

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
    batch_size, num_views = transform_matrix.shape[:2]
    with torch.amp.autocast("cuda", enabled=False):
        transforms = transform_matrix.float()
        anchor = transforms[:, :1].expand(batch_size, num_views, 4, 4)
        relative = torch.linalg.inv(anchor.reshape(-1, 4, 4)) @ transforms.reshape(-1, 4, 4)
        relative = relative.reshape(batch_size, num_views, 4, 4)
        fixed = fixed_transform.float().expand(batch_size, 4, 4).clone()
        fixed[:, 1, 3] = -distance[:, 0].float()
        projection = fixed[:, None] @ relative
    return projection, relative
```

Remove only the `assert transform_matrix is None` line from `ProjGrid.forward`. Preserve the existing default transform branch and all sampling behavior.

- [ ] **Step 4: Run geometry tests and the existing projection import smoke test**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_projection_geometry.py -q`

Expected: `3 passed`.

Run: `conda run --no-capture-output -n pixal3d python -c "from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ProjGrid, compute_multiview_projection_matrices; print('projection imports ok')"`

Expected: `projection imports ok`.

- [ ] **Step 5: Commit geometry support**

```bash
git add tests/multiview/test_projection_geometry.py pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py
git commit -m "feat: add paper-faithful multiview projection geometry"
```

---

### Task 2: Extend the Existing DINOv3 Conditioner with Arithmetic Multi-View Averaging

**Files:**
- Create: `tests/multiview/test_conditioner.py`
- Modify: `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py:344-566`

**Interfaces:**
- Consumes: Task 1 `compute_multiview_projection_matrices`, image tensor `[B,K,3,H,W]`, cameras `[B,K]`, scale `[B]`, transforms `[B,K,4,4]`.
- Produces: `_forward_single_view(...)`, `_forward_multiview(...)`, and the existing `forward(...) -> (z_global, z_proj)` with shapes `[B,T,D]` and `[B,R^3,C_proj]`.

- [ ] **Step 1: Write failing dispatch and mean tests with a parameter-free harness**

```python
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


def camera_batch(k):
    transforms = torch.eye(4).repeat(1, k, 1, 1)
    transforms[0, :, 0, 3] = torch.arange(k, dtype=torch.float32)
    return {
        "camera_angle_x": torch.full((1, k), 0.7),
        "distance": torch.full((1, k), 2.5),
        "mesh_scale": torch.ones(1),
        "transform_matrix": transforms,
    }


def test_k1_multiview_dispatch_matches_single_view():
    model = ConditionerHarness()
    image = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    cameras = camera_batch(1)
    multi = model(image[:, None], **cameras)
    fixed = model.front[None].clone()
    fixed[:, 1, 3] = -2.5
    single = model._forward_single_view(
        image,
        cameras["camera_angle_x"][:, 0],
        cameras["distance"][:, 0],
        cameras["mesh_scale"],
        fixed,
    )
    torch.testing.assert_close(multi[0], single[0])
    torch.testing.assert_close(multi[1], single[1])


def test_duplicate_views_average_to_single_view_output():
    model = ConditionerHarness()
    image = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    images = image[:, None].expand(1, 4, 3, 2, 2).clone()
    cameras = camera_batch(4)
    cameras["transform_matrix"][:] = torch.eye(4)
    global_feature, projected = model(images, **cameras)
    expected_global, expected_projected = model(images[:, 0:1], **camera_batch(1))
    torch.testing.assert_close(global_feature, expected_global)
    torch.testing.assert_close(projected, expected_projected)


def test_non_anchor_view_permutation_preserves_mean():
    model = ConditionerHarness()
    images = torch.arange(1 * 4 * 3 * 2 * 2, dtype=torch.float32).reshape(1, 4, 3, 2, 2)
    cameras = camera_batch(4)
    first = model(images, **cameras)
    order = torch.tensor([0, 3, 1, 2])
    permuted_cameras = {
        "camera_angle_x": cameras["camera_angle_x"][:, order],
        "distance": cameras["distance"][:, order],
        "mesh_scale": cameras["mesh_scale"],
        "transform_matrix": cameras["transform_matrix"][:, order],
    }
    second = model(images[:, order], **permuted_cameras)
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])


def test_harness_adds_no_trainable_parameters():
    assert [p for p in ConditionerHarness().parameters() if p.requires_grad] == []
```

- [ ] **Step 2: Run conditioner tests and verify RED**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_conditioner.py -q`

Expected: tests fail because the existing `forward` rejects a five-dimensional tensor and `_forward_single_view` is not the single-view dispatch boundary.

- [ ] **Step 3: Refactor the existing body without changing single-view math**

Rename the existing method only; keep its complete body unchanged:

```diff
-    def forward(
+    def _forward_single_view(
         self,
         image: Union[torch.Tensor, List[Image.Image]],
         camera_angle_x: Optional[torch.Tensor] = None,
         distance: Optional[torch.Tensor] = None,
         mesh_scale: Optional[torch.Tensor] = None,
         transform_matrix: Optional[torch.Tensor] = None,
     ) -> Tuple[torch.Tensor, torch.Tensor]:
```

Expose the current fixed transform through:

```python
@property
def fixed_projection_transform(self) -> torch.Tensor:
    return self.proj_grid.front_view_transform_matrix
```

Insert this complete public dispatcher after `_forward_multiview`:

```python
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

- [ ] **Step 4: Implement the paper-faithful multi-view branch using the single-view path**

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
    if camera_angle_x is None or distance is None or mesh_scale is None:
        raise ValueError("multi-view camera_angle_x, distance, and mesh_scale are required")
    if transform_matrix is None:
        raise ValueError("calibrated multi-view transform_matrix is required")
    if camera_angle_x.shape != (batch_size, num_views):
        raise ValueError("camera_angle_x must have shape [B, K]")
    if distance.shape != (batch_size, num_views):
        raise ValueError("distance must have shape [B, K]")
    if mesh_scale.shape != (batch_size,):
        raise ValueError("mesh_scale must have shape [B]")
    if transform_matrix.shape != (batch_size, num_views, 4, 4):
        raise ValueError("transform_matrix must have shape [B, K, 4, 4]")
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
    z_global = torch.stack(global_views, dim=1).mean(dim=1)
    z_proj = torch.stack(projected_views, dim=1).mean(dim=1)
    return z_global, z_proj
```

This sequential implementation intentionally reuses the current single-view DINOv3/NAF code. Do not add batching, caching, masks, or learned weights in this task.

- [ ] **Step 5: Run conditioner and geometry tests**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_conditioner.py tests/multiview/test_projection_geometry.py -q`

Expected: `7 passed`.

- [ ] **Step 6: Commit conditioner support**

```bash
git add tests/multiview/test_conditioner.py pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py
git commit -m "feat: average projected features across calibrated views"
```

---

### Task 3: Load Anchor-First Multi-View Conditions

**Files:**
- Create: `tests/multiview/test_dataset_conditions.py`
- Modify: `pixal3d/datasets/components.py:197-349`

**Interfaces:**
- Consumes: target anchor stored by the parent dataset in `_current_view_idx` and `_current_latent_dir`, eight render frames, and anchor scale JSON.
- Produces: per-sample `cond[V,3,H,W]`, camera vectors `[V]`, `transform_matrix[V,4,4]`, scalar `mesh_scale`, with anchor at position zero.

- [ ] **Step 1: Write failing tests around a real temporary render fixture**

```python
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pixal3d.datasets.components import load_anchor_first_conditions


def write_render_fixture(root: Path, num_views: int = 8):
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


def test_anchor_first_conditions_keep_all_views_unique(tmp_path):
    write_render_fixture(tmp_path)
    result = load_anchor_first_conditions(
        tmp_path, anchor_index=1, image_size=4, other_view_indices=[7, 4, 0, 2, 3, 5, 6]
    )
    assert result["view_indices"].tolist() == [1, 7, 4, 0, 2, 3, 5, 6]
    assert result["cond"].shape == (8, 3, 4, 4)
    assert result["camera_angle_x"].shape == (8,)
    assert result["camera_distance"].shape == (8,)
    assert result["transform_matrix"].shape == (8, 4, 4)
    assert torch.equal(result["transform_matrix"][0, :3, 3], torch.tensor([1.0, 0.0, 2.0]))


def test_condition_loader_rejects_anchor_in_other_views(tmp_path):
    write_render_fixture(tmp_path)
    try:
        load_anchor_first_conditions(
            tmp_path, anchor_index=1, image_size=4, other_view_indices=[1, 0, 2, 3, 4, 5, 6]
        )
    except ValueError as error:
        assert "anchor" in str(error)
    else:
        raise AssertionError("duplicate anchor was accepted")
```

- [ ] **Step 2: Run fixture tests and verify RED**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_dataset_conditions.py -q`

Expected: collection fails because `load_anchor_first_conditions` does not exist.

- [ ] **Step 3: Add a reusable non-learned loader beside the existing mixins**

```python
def _load_rgba_condition(path: str, image_size: int) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGBA").resize(
            (image_size, image_size), Image.Resampling.LANCZOS
        )
        rgba = torch.from_numpy(np.array(image)).float() / 255.0
    rgb = rgba[..., :3].permute(2, 0, 1)
    alpha = rgba[..., 3]
    return rgb * alpha.unsqueeze(0)


def load_anchor_first_conditions(
    image_root: Union[str, os.PathLike],
    *,
    anchor_index: int,
    image_size: int,
    other_view_indices: Sequence[int],
) -> Dict[str, torch.Tensor]:
    image_root = os.fspath(image_root)
    with open(os.path.join(image_root, "transforms.json")) as stream:
        metadata = json.load(stream)
    frames = metadata["frames"]
    order = [anchor_index, *other_view_indices]
    if anchor_index in other_view_indices:
        raise ValueError("anchor must not occur in other_view_indices")
    if len(order) != len(set(order)) or sorted(order) != list(range(len(frames))):
        raise ValueError("view order must contain every render exactly once")
    images = []
    angles = []
    distances = []
    transforms_out = []
    for index in order:
        frame = frames[index]
        images.append(_load_rgba_condition(os.path.join(image_root, frame["file_path"]), image_size))
        angle = frame.get("camera_angle_x", metadata.get("camera_angle_x"))
        if angle is None:
            raise KeyError("camera_angle_x is required for every view")
        transform = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        angles.append(float(angle))
        distances.append(torch.linalg.vector_norm(transform[:3, 3]))
        transforms_out.append(transform)
    return {
        "cond": torch.stack(images),
        "camera_angle_x": torch.tensor(angles, dtype=torch.float32),
        "camera_distance": torch.stack(distances).float(),
        "transform_matrix": torch.stack(transforms_out),
        "view_indices": torch.tensor(order, dtype=torch.int64),
    }
```

- [ ] **Step 4: Add `MultiViewImageConditionedMixin` without changing single-view mixins**

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
        if not 1 <= min_condition_views <= max_condition_views <= condition_num_views:
            raise ValueError("condition view bounds must satisfy 1 <= min <= max <= total")
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
        other_indices = [i for i in range(self.condition_num_views) if i != anchor_index]
        other_indices = np.random.permutation(other_indices).tolist()
        condition = load_anchor_first_conditions(
            os.path.join(root["render_cond"], instance),
            anchor_index=anchor_index,
            image_size=self.image_size,
            other_view_indices=other_indices,
        )
        pack.update(condition)
        scale_path = os.path.join(
            self._current_latent_dir, f"view{anchor_index:02d}_scale.json"
        )
        with open(scale_path) as stream:
            scale_data = json.load(stream)
        pack["mesh_scale"] = torch.tensor(float(scale_data["total_scale"]), dtype=torch.float32)
        return pack
```

- [ ] **Step 5: Run condition-loader tests**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_dataset_conditions.py -q`

Expected: `2 passed`.

- [ ] **Step 6: Commit condition loading**

```bash
git add tests/multiview/test_dataset_conditions.py pixal3d/datasets/components.py
git commit -m "feat: load anchor-first calibrated condition views"
```

---

### Task 4: Select One K Per Batch and Register Multi-View Dataset Classes

**Files:**
- Create: `tests/multiview/test_dataset_collation.py`
- Modify: `pixal3d/datasets/components.py`
- Modify: `pixal3d/datasets/sparse_structure_latent.py:399-408`
- Modify: `pixal3d/datasets/structured_latent_shape.py:393-402`
- Modify: `pixal3d/datasets/structured_latent_svpbr.py:655-666`
- Modify: `pixal3d/datasets/__init__.py:3-23`

**Interfaces:**
- Consumes: Task 3 per-sample eight-view fields.
- Produces: one batch-wide K and the three registered dataset names `MultiViewImageConditionedSparseStructureLatentView`, `MultiViewImageConditionedSLatShapeView`, and `MultiViewImageConditionedSLatPbrView`.

- [ ] **Step 1: Write failing batch-wide K tests**

```python
import pytest
import torch

from pixal3d import datasets
from pixal3d.datasets.components import slice_condition_views
from pixal3d.datasets.sparse_structure_latent import (
    MultiViewImageConditionedSparseStructureLatentView,
)
from pixal3d.datasets.structured_latent_shape import (
    MultiViewImageConditionedSLatShapeView,
)
from pixal3d.datasets.structured_latent_svpbr import (
    MultiViewImageConditionedSLatPbrView,
)


VIEW_KEYS = ("cond", "camera_angle_x", "camera_distance", "transform_matrix", "view_indices")


def sample(offset):
    return {
        "cond": torch.arange(offset, offset + 8 * 3).reshape(8, 3, 1, 1).float(),
        "camera_angle_x": torch.arange(8).float() + offset,
        "camera_distance": torch.arange(8).float() + 2.0,
        "transform_matrix": torch.eye(4).repeat(8, 1, 1),
        "view_indices": torch.arange(8),
        "mesh_scale": torch.tensor(1.0),
    }


def test_slice_condition_views_uses_one_k_for_whole_batch():
    batch = slice_condition_views([sample(0), sample(100)], 4)
    for item in batch:
        for key in VIEW_KEYS:
            assert item[key].shape[0] == 4
    assert batch[0]["mesh_scale"].ndim == 0
    assert batch[1]["mesh_scale"].ndim == 0


def test_slice_condition_views_does_not_mutate_source_samples():
    source = [sample(0), sample(100)]
    sliced = slice_condition_views(source, 2)
    assert source[0]["cond"].shape[0] == 8
    assert sliced[0]["cond"].shape[0] == 2


def test_multiview_dataset_names_are_registered():
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

- [ ] **Step 2: Run collation tests and verify RED**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_dataset_collation.py -q`

Expected: collection fails because `slice_condition_views` and the multi-view dataset classes do not exist.

- [ ] **Step 3: Add immutable slicing and batch K selection to the mixin**

```python
MULTIVIEW_CONDITION_KEYS = (
    "cond",
    "camera_angle_x",
    "camera_distance",
    "transform_matrix",
    "view_indices",
)


def slice_condition_views(batch: Sequence[Dict[str, Any]], num_views: int):
    sliced = []
    for source in batch:
        item = dict(source)
        for key in MULTIVIEW_CONDITION_KEYS:
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

- [ ] **Step 4: Add the three thin dataset classes with existing collation**

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
        selected = self.select_batch_condition_views(batch)
        return SLatShapeView.collate_fn(selected, split_size=split_size)


# structured_latent_svpbr.py
class MultiViewImageConditionedSLatPbrView(
    MultiViewImageConditionedMixin, SLatPbrView
):
    def collate_fn(self, batch, split_size=None):
        selected = self.select_batch_condition_views(batch)
        return SLatPbrView.collate_fn(selected, split_size=split_size)
```

Register the exact class names in `pixal3d/datasets/__init__.py`:

```python
__attributes.update({
    "MultiViewImageConditionedSparseStructureLatentView": "sparse_structure_latent",
    "MultiViewImageConditionedSLatShapeView": "structured_latent_shape",
    "MultiViewImageConditionedSLatPbrView": "structured_latent_svpbr",
})
```

- [ ] **Step 5: Test K endpoints and existing sparse latent structures**

Append these exact endpoint tests. Calling the methods with a lightweight selector avoids dataset metadata I/O while exercising the real dense and sparse collation functions:

```python
from types import MethodType, SimpleNamespace

import numpy as np

from pixal3d.datasets.components import MultiViewImageConditionedMixin


def selector():
    value = SimpleNamespace(min_condition_views=2, max_condition_views=6)
    value.select_batch_condition_views = MethodType(
        MultiViewImageConditionedMixin.select_batch_condition_views, value
    )
    return value


def dense_sample(offset):
    value = sample(offset)
    value["x_0"] = torch.full((2, 2, 2, 2), float(offset))
    return value


def shape_sample(offset):
    value = sample(offset)
    value["coords"] = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int32)
    value["feats"] = torch.full((2, 32), float(offset))
    return value


def pbr_sample(offset):
    value = sample(offset)
    value["coords"] = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int32)
    value["pbr_feats"] = torch.full((2, 32), float(offset))
    value["shape_feats"] = torch.full((2, 32), float(offset + 1))
    return value


@pytest.mark.parametrize("num_views", [2, 6])
def test_all_stage_collators_use_the_same_sampled_endpoint(monkeypatch, num_views):
    monkeypatch.setattr(np.random, "randint", lambda low, high: num_views)
    dense_pack = MultiViewImageConditionedSparseStructureLatentView.collate_fn(
        selector(), [dense_sample(0), dense_sample(100)]
    )
    shape_pack = MultiViewImageConditionedSLatShapeView.collate_fn(
        selector(), [shape_sample(0), shape_sample(100)]
    )
    pbr_pack = MultiViewImageConditionedSLatPbrView.collate_fn(
        selector(), [pbr_sample(0), pbr_sample(100)]
    )
    assert dense_pack["cond"].shape[:2] == (2, num_views)
    assert shape_pack["cond"].shape[:2] == (2, num_views)
    assert pbr_pack["cond"].shape[:2] == (2, num_views)
    assert shape_pack["x_0"].shape[0] == 2
    assert pbr_pack["concat_cond"].shape[0] == 2
```

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_dataset_collation.py -q`

Expected: all collation tests pass.

- [ ] **Step 6: Commit dataset collation**

```bash
git add tests/multiview/test_dataset_collation.py pixal3d/datasets/components.py pixal3d/datasets/sparse_structure_latent.py pixal3d/datasets/structured_latent_shape.py pixal3d/datasets/structured_latent_svpbr.py pixal3d/datasets/__init__.py
git commit -m "feat: collate one random view count per training batch"
```

---

### Task 5: Keep Trainer Snapshots Compatible with Five-Dimensional Conditions

**Files:**
- Create: `tests/multiview/test_trainer_views.py`
- Modify: `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py:1362-1530`
- Modify: `pixal3d/trainers/flow_matching/flow_matching.py:445-590`
- Modify: `pixal3d/trainers/flow_matching/sparse_flow_matching.py:430-550`

**Interfaces:**
- Consumes: collated condition `[B,K,3,H,W]` and camera vectors `[B,K]`.
- Produces: unchanged conditioning for training, anchor-only `[B,3,H,W]` images and `[B]` cameras for visualization.

- [ ] **Step 1: Write failing anchor-view helper tests**

```python
import torch

from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    anchor_camera_value,
    anchor_condition_image,
)


def test_anchor_condition_image_handles_single_and_multiview():
    single = torch.zeros(2, 3, 4, 4)
    multi = torch.arange(2 * 6 * 3 * 4 * 4).reshape(2, 6, 3, 4, 4)
    assert anchor_condition_image(single) is single
    assert torch.equal(anchor_condition_image(multi), multi[:, 0])


def test_anchor_camera_value_handles_scalar_and_view_vectors():
    single = torch.tensor([2.0, 3.0])
    multi = torch.tensor([[2.0, 4.0], [3.0, 5.0]])
    assert anchor_camera_value(single) is single
    assert torch.equal(anchor_camera_value(multi), torch.tensor([2.0, 3.0]))
```

- [ ] **Step 2: Run tests and verify RED**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_trainer_views.py -q`

Expected: import failure for both helpers.

- [ ] **Step 3: Add non-learned anchor visualization helpers**

```python
def anchor_condition_image(cond: torch.Tensor) -> torch.Tensor:
    return cond[:, 0] if cond.ndim == 5 else cond


def anchor_camera_value(value: torch.Tensor) -> torch.Tensor:
    return value[:, 0] if value.ndim > 1 else value
```

Apply these exact changes to `ImageConditionedProjMixin`:

```diff
     def vis_cond(self, cond, **kwargs):
         """Visualize the conditioning data."""
-        return {'image': {'value': cond, 'type': 'image'}}
+        return {"image": {"value": anchor_condition_image(cond), "type": "image"}}
@@
         if camera_info is None:
             print("Warning: No camera info available for projection visualization")
             return None
+        cond = anchor_condition_image(cond)
+        camera_info["camera_angle_x"] = anchor_camera_value(
+            camera_info["camera_angle_x"]
+        )
+        camera_info["distance"] = anchor_camera_value(camera_info["distance"])
+        camera_info["transform_matrix"] = None
         return module.visualize_projection(
```

This reproduces the existing fixed-front projection for the anchor diagnostic.

- [ ] **Step 4: Slice camera values only where snapshots render the target anchor**

In `flow_matching.py`, slice only after concatenating the values used by rendered snapshots:

```diff
-            camera_distance = torch.cat(camera_distances, dim=0)
+            camera_distance = anchor_camera_value(
+                torch.cat(camera_distances, dim=0)
+            )
@@
-            camera_angle_x = torch.cat(camera_angles, dim=0)
+            camera_angle_x = anchor_camera_value(torch.cat(camera_angles, dim=0))
```

Import `anchor_camera_value` from the projection mixin module. In `sparse_flow_matching.py`, normalize only the snapshot dictionaries after they are built:

```python
for key in ("camera_angle_x", "camera_distance"):
    if key in sample_gt:
        sample_gt[key] = anchor_camera_value(sample_gt[key])
        sample[key] = anchor_camera_value(sample[key])
```

Do not slice camera tensors before `get_cond` or `get_inference_cond`.

- [ ] **Step 5: Run trainer helper and conditioner tests**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_trainer_views.py tests/multiview/test_conditioner.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit snapshot compatibility**

```bash
git add tests/multiview/test_trainer_views.py pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py pixal3d/trainers/flow_matching/flow_matching.py pixal3d/trainers/flow_matching/sparse_flow_matching.py
git commit -m "fix: visualize the multiview anchor in training snapshots"
```

---

### Task 6: Materialize Single-View Weights and Add Final-Resolution Multi-View Configs

**Files:**
- Create: `tests/multiview/test_configs.py`
- Create: `configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json`
- Create: `configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json`
- Create: `configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json`
- Modify: `README.md`
- Runtime output: `/root/node17/data/pixal3d/train/checkpoints/single_view/*.pt`

**Interfaces:**
- Consumes: official `TencentARC/Pixal3D` final SS/Shape/PBR safetensors and Task 4 dataset registrations.
- Produces: three `.pt` denoiser state dictionaries and three multi-view training configs that retain current model/trainer definitions.

- [ ] **Step 1: Write failing config invariants**

```python
import json
from pathlib import Path


CONFIGS = {
    "ss": Path("configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json"),
    "shape": Path("configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json"),
    "pbr": Path("configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json"),
}


def test_multiview_configs_use_existing_models_and_batchwide_k():
    expected_datasets = {
        "ss": "MultiViewImageConditionedSparseStructureLatentView",
        "shape": "MultiViewImageConditionedSLatShapeView",
        "pbr": "MultiViewImageConditionedSLatPbrView",
    }
    for stage, path in CONFIGS.items():
        config = json.loads(path.read_text())
        args = config["dataset"]["args"]
        assert config["dataset"]["name"] == expected_datasets[stage]
        assert args["condition_num_views"] == 8
        assert args["min_condition_views"] == 2
        assert args["max_condition_views"] == 6
        assert config["trainer"]["args"]["image_cond_model"]["name"] == "DinoV3ProjFeatureExtractor"
        assert config["trainer"]["args"]["batch_size_per_gpu"] == 1
        assert config["trainer"]["args"]["batch_split"] == 1
        assert config["models"]["denoiser"]["args"]["image_attn_mode"] == "proj"


def test_multiview_configs_use_exact_single_view_checkpoint_paths():
    expected = {
        "ss": "/root/node17/data/pixal3d/train/checkpoints/single_view/ss_flow_img_dit_1_3B_64_bf16.pt",
        "shape": "/root/node17/data/pixal3d/train/checkpoints/single_view/slat_flow_img2shape_dit_1_3B_1024_bf16.pt",
        "pbr": "/root/node17/data/pixal3d/train/checkpoints/single_view/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.pt",
    }
    for stage, path in CONFIGS.items():
        config = json.loads(path.read_text())
        assert config["trainer"]["args"]["finetune_ckpt"] == {"denoiser": expected[stage]}
```

- [ ] **Step 2: Run config tests and verify RED**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_configs.py -q`

Expected: three config files are missing.

- [ ] **Step 3: Create exact final-resolution config variants**

Copy the three exact source files to the target paths in the **Files** block, then apply these diffs. Preserve every field not shown and do not modify the source files.

```diff
# configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_finetune_ft64.json
# -> configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json
-        "name": "ViewImageConditionedSparseStructureLatentView",
+        "name": "MultiViewImageConditionedSparseStructureLatentView",
         "args": {
             "min_aesthetic_score": 4.5,
             "image_size": 512,
             "num_views": 2,
-            "load_camera_info": true,
+            "condition_num_views": 8,
+            "min_condition_views": 2,
+            "max_condition_views": 6,
@@
-            "batch_size_per_gpu": 8,
-            "batch_split": 2,
+            "batch_size_per_gpu": 1,
+            "batch_split": 1,
```

Replace the SS `trainer.args.finetune_ckpt` object with:

```json
{"denoiser": "/root/node17/data/pixal3d/train/checkpoints/single_view/ss_flow_img_dit_1_3B_64_bf16.pt"}
```

```diff
# configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_finetune_ft1024.json
# -> configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json
-        "name": "ViewImageConditionedSLatShapeView",
+        "name": "MultiViewImageConditionedSLatShapeView",
         "args": {
@@
             "num_views": 2,
+            "condition_num_views": 8,
+            "min_condition_views": 2,
+            "max_condition_views": 6,
@@
-            "batch_size_per_gpu": 2,
-            "batch_split": 2,
+            "batch_size_per_gpu": 1,
+            "batch_split": 1,
```

Replace the Shape `trainer.args.finetune_ckpt` object with:

```json
{"denoiser": "/root/node17/data/pixal3d/train/checkpoints/single_view/slat_flow_img2shape_dit_1_3B_1024_bf16.pt"}
```

```diff
# configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_finetune_ft1024.json
# -> configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json
-        "name": "ViewImageConditionedSLatPbrView",
+        "name": "MultiViewImageConditionedSLatPbrView",
         "args": {
@@
             "num_views": 2,
+            "condition_num_views": 8,
+            "min_condition_views": 2,
+            "max_condition_views": 6,
@@
-            "batch_size_per_gpu": 2,
-            "batch_split": 2,
+            "batch_size_per_gpu": 1,
+            "batch_split": 1,
```

Replace the PBR `trainer.args.finetune_ckpt` object with:

```json
{"denoiser": "/root/node17/data/pixal3d/train/checkpoints/single_view/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.pt"}
```

- [ ] **Step 4: Download and convert released weights without changing keys**

Run:

```bash
huggingface-cli download TencentARC/Pixal3D \
  ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors \
  ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors \
  ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors \
  --local-dir /root/node17/data/pixal3d/train/checkpoints/single_view/hf
```

Run:

```bash
conda run --no-capture-output -n pixal3d python -c "from pathlib import Path; import torch; from safetensors.torch import load_file; source=Path('/root/node17/data/pixal3d/train/checkpoints/single_view/hf/ckpts'); target=source.parents[1]; mapping={'ss_flow_img_dit_1_3B_64_bf16':'ss_flow_img_dit_1_3B_64_bf16','slat_flow_img2shape_dit_1_3B_1024_bf16':'slat_flow_img2shape_dit_1_3B_1024_bf16','slat_flow_imgshape2tex_dit_1_3B_1024_bf16':'slat_flow_imgshape2tex_dit_1_3B_1024_bf16'}; [torch.save(load_file(str(source / (src + '.safetensors')), device='cpu'), target / (dst + '.pt')) for src, dst in mapping.items()]"
```

Expected: three `.pt` files exist directly under `/root/node17/data/pixal3d/train/checkpoints/single_view` and retain the safetensors keys unchanged.

- [ ] **Step 5: Run config tests and a strict architecture compatibility gate**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_configs.py -q`

Expected: `2 passed`.

Append this opt-in checkpoint gate. It uses the exact model constructor from each config and requires all released denoiser keys to match except the repository's allowed `rope_phases` buffer:

```python
import gc
import os

import pytest
import torch

from pixal3d import models


@pytest.mark.parametrize("stage", ["ss", "shape", "pbr"])
def test_released_checkpoint_matches_configured_denoiser(stage):
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

Run: `PIXAL3D_REQUIRE_CHECKPOINTS=1 conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_configs.py -q -k released_checkpoint`

Expected: `3 passed`; no checkpoint test is skipped.

- [ ] **Step 6: Document the three-stage fine-tuning entry points**

Add these exact commands:

```bash
conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json \
  --output_dir /root/node17/data/pixal3d/train/runs/multiview/ss64 \
  --data_dir '{"Handoff":{"base":"/root/node17/data/pixal3d/train/stage1/active","ss_latent":"/root/node17/data/pixal3d/train/stage1/active/ss_latents/ss_enc_conv3d_16l8_fp16_64_view","render_cond":"/root/node17/data/pixal3d/train/stage1/active/renders_cond"}}' \
  --num_gpus 7

conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  --output_dir /root/node17/data/pixal3d/train/runs/multiview/shape1024 \
  --data_dir '{"Handoff":{"base":"/root/node17/data/pixal3d/train/stage2/active","shape_latent":"/root/node17/data/pixal3d/train/stage2/active/shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view","render_cond":"/root/node17/data/pixal3d/train/stage2/active/renders_cond"}}' \
  --num_gpus 7

conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  --output_dir /root/node17/data/pixal3d/train/runs/multiview/pbr1024 \
  --data_dir '{"Handoff":{"base":"/root/node17/data/pixal3d/train/stage3/active","shape_latent":"/root/node17/data/pixal3d/train/stage3/active/shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view","pbr_latent":"/root/node17/data/pixal3d/train/stage3/active/pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix","render_cond":"/root/node17/data/pixal3d/train/stage3/active/renders_cond"}}' \
  --num_gpus 7
```

State that full fine-tuning begins only after preprocessing is stopped and its handoff audit passes.

- [ ] **Step 7: Commit configs and documentation**

```bash
git add tests/multiview/test_configs.py configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json README.md
git commit -m "feat: configure final-resolution multiview finetuning"
```

---

### Task 7: Extend the Existing Cascade Pipeline to Calibrated Multi-View Inputs

**Files:**
- Create: `tests/multiview/test_pipeline_inputs.py`
- Modify: `pixal3d/pipelines/pixal3d_image_to_3d.py:188-295`
- Modify: `pixal3d/pipelines/pixal3d_image_to_3d.py:609-780`

**Interfaces:**
- Consumes: one image/scalar cameras or K images/vector cameras with `[K,4,4]` transforms.
- Produces: the same condition dictionaries and generated `MeshWithVoxel` list as the single-view pipeline.

- [ ] **Step 1: Write failing input-normalization tests**

```python
import torch
from PIL import Image

from pixal3d.pipelines.pixal3d_image_to_3d import normalize_calibrated_views


def image(color):
    return Image.new("RGB", (8, 8), color=color)


def test_single_view_is_normalized_to_k1():
    images, cameras = normalize_calibrated_views(
        image("red"),
        {"camera_angle_x": 0.7, "distance": 2.5, "mesh_scale": 1.0},
    )
    assert len(images) == 1
    assert cameras["camera_angle_x"].shape == (1, 1)
    assert cameras["distance"].shape == (1, 1)
    assert cameras["mesh_scale"].shape == (1,)
    assert cameras["transform_matrix"] is None


def test_multiview_requires_matching_calibrated_vectors():
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
    assert len(images) == 2
    assert cameras["camera_angle_x"].shape == (1, 2)
    assert cameras["distance"].shape == (1, 2)
    assert cameras["transform_matrix"].shape == (1, 2, 4, 4)
```

- [ ] **Step 2: Run pipeline input tests and verify RED**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_pipeline_inputs.py -q`

Expected: import failure for `normalize_calibrated_views`.

- [ ] **Step 3: Implement one normalization boundary**

```python
def normalize_calibrated_views(image, camera_params):
    images = list(image) if isinstance(image, (list, tuple)) else [image]
    if not images or len(images) > 8:
        raise ValueError("Pixal3D inference requires between 1 and 8 views")
    num_views = len(images)

    def vector(name):
        value = camera_params[name]
        values = [value] if np.isscalar(value) else list(value)
        if len(values) != num_views:
            raise ValueError(f"{name} must contain one value per view")
        return torch.tensor(values, dtype=torch.float32).reshape(1, num_views)

    transforms = camera_params.get("transform_matrix")
    if transforms is not None:
        transforms = torch.as_tensor(transforms, dtype=torch.float32)
        if transforms.shape != (num_views, 4, 4):
            raise ValueError("transform_matrix must have shape [K, 4, 4]")
        transforms = transforms.unsqueeze(0)
    if num_views > 1 and transforms is None:
        raise ValueError("calibrated multi-view inference requires transform_matrix")
    cameras = {
        "camera_angle_x": vector("camera_angle_x"),
        "distance": vector("distance"),
        "mesh_scale": torch.tensor(
            [float(camera_params.get("mesh_scale", 1.0))], dtype=torch.float32
        ),
        "transform_matrix": transforms,
    }
    return images, cameras
```

- [ ] **Step 4: Convert PIL views at the pipeline boundary and extend both condition builders**

Add these module-level helpers beside `normalize_calibrated_views` so the conditioner itself continues to accept tensors or its existing flat single-view PIL list:

```python
def pil_views_to_tensor(images, image_size, device):
    tensors = []
    for image in images:
        resized = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        array = np.asarray(resized.convert("RGB"), dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(tensors, dim=0).unsqueeze(0).to(device)


def condition_camera_tensors(
    num_views,
    device,
    camera_angle_x,
    distance,
    mesh_scale,
    transform_matrix,
):
    def view_vector(name, value):
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        tensor = tensor.reshape(1, -1)
        if tensor.shape != (1, num_views):
            raise ValueError(f"{name} must contain one value per view")
        return tensor

    scale = torch.as_tensor(mesh_scale, dtype=torch.float32, device=device).reshape(-1)
    if scale.shape != (1,):
        raise ValueError("mesh_scale must contain one anchor scale")
    transforms = None
    if transform_matrix is not None:
        transforms = torch.as_tensor(
            transform_matrix, dtype=torch.float32, device=device
        ).reshape(1, num_views, 4, 4)
    return {
        "camera_angle_x": view_vector("camera_angle_x", camera_angle_x),
        "distance": view_vector("distance", distance),
        "mesh_scale": scale,
        "transform_matrix": transforms,
    }
```

Apply these exact signature and call-site changes in `get_proj_cond_ss`:

```diff
     def get_proj_cond_ss(
         self,
         image: list,
-        camera_angle_x: float = 0.8575560450553894,
-        distance: float = 2.0,
+        camera_angle_x=0.8575560450553894,
+        distance=2.0,
         mesh_scale: float = 1.0,
+        transform_matrix=None,
@@
-        cam_angle = torch.tensor([camera_angle_x], device=device)
-        dist_tensor = torch.tensor([distance], device=device)
-        scale_tensor = torch.tensor([mesh_scale], device=device)
+        image_tensor = pil_views_to_tensor(image, image_cond_model.image_size, device)
+        cameras = condition_camera_tensors(
+            len(image), device, camera_angle_x, distance, mesh_scale, transform_matrix
+        )
+        if len(image) == 1 and cameras["transform_matrix"] is None:
+            image_tensor = image_tensor[:, 0]
+            cameras["camera_angle_x"] = cameras["camera_angle_x"][:, 0]
+            cameras["distance"] = cameras["distance"][:, 0]
         z_global, z_proj = image_cond_model(
-            image, camera_angle_x=cam_angle, distance=dist_tensor, mesh_scale=scale_tensor,
+            image_tensor,
+            camera_angle_x=cameras["camera_angle_x"],
+            distance=cameras["distance"],
+            mesh_scale=cameras["mesh_scale"],
+            transform_matrix=cameras["transform_matrix"],
         )
```

Apply the same boundary in `get_proj_cond_shape`, leaving its sparse-grid lookup unchanged:

```diff
     def get_proj_cond_shape(
@@
-        camera_angle_x: float = 0.8575560450553894,
-        distance: float = 2.0,
+        camera_angle_x=0.8575560450553894,
+        distance=2.0,
         mesh_scale: float = 1.0,
+        transform_matrix=None,
         grid_resolution_override: int = None,
@@
-        cam_angle = torch.tensor([camera_angle_x], device=device)
-        dist_tensor = torch.tensor([distance], device=device)
-        scale_tensor = torch.tensor([mesh_scale], device=device)
+        image_tensor = pil_views_to_tensor(image, image_cond_model.image_size, device)
+        cameras = condition_camera_tensors(
+            len(image), device, camera_angle_x, distance, mesh_scale, transform_matrix
+        )
+        if len(image) == 1 and cameras["transform_matrix"] is None:
+            image_tensor = image_tensor[:, 0]
+            cameras["camera_angle_x"] = cameras["camera_angle_x"][:, 0]
+            cameras["distance"] = cameras["distance"][:, 0]
         z_global, z_proj = image_cond_model(
-            image, camera_angle_x=cam_angle, distance=dist_tensor, mesh_scale=scale_tensor,
+            image_tensor,
+            camera_angle_x=cameras["camera_angle_x"],
+            distance=cameras["distance"],
+            mesh_scale=cameras["mesh_scale"],
+            transform_matrix=cameras["transform_matrix"],
         )
```

- [ ] **Step 5: Propagate one normalized camera bundle through every cascade stage**

Replace the scalar extraction/preprocessing block at the beginning of `run`:

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

Replace `[image]` with `images` in all four condition-builder calls and add the transform argument each time:

```python
cond_ss = self.get_proj_cond_ss(
    images,
    camera_angle_x=camera_angle_x,
    distance=distance,
    mesh_scale=mesh_scale,
    transform_matrix=transform_matrix,
)

cond_shape_lr = self.get_proj_cond_shape(
    self.image_cond_model_shape_512,
    images,
    coords,
    camera_angle_x=camera_angle_x,
    distance=distance,
    mesh_scale=mesh_scale,
    transform_matrix=transform_matrix,
)

cond_shape_hr = self.get_proj_cond_shape(
    self.image_cond_model_shape_1024,
    images,
    hr_coords_unique,
    camera_angle_x=camera_angle_x,
    distance=distance,
    mesh_scale=mesh_scale,
    transform_matrix=transform_matrix,
    grid_resolution_override=actual_grid_res,
)

cond_tex = self.get_proj_cond_shape(
    self.image_cond_model_tex_1024,
    images,
    shape_slat.coords,
    camera_angle_x=camera_angle_x,
    distance=distance,
    mesh_scale=mesh_scale,
    transform_matrix=transform_matrix,
    grid_resolution_override=tex_grid_res,
)
```

Do not cache DINOv3 or NAF features across these four calls.

- [ ] **Step 6: Run pipeline and conditioner tests**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_pipeline_inputs.py tests/multiview/test_conditioner.py -q`

Expected: all tests pass.

- [ ] **Step 7: Verify the existing single-view CLI still imports the pipeline**

Run: `conda run --no-capture-output -n pixal3d python inference.py --help`

Expected: exit 0 and the existing `--image` option remains documented.

- [ ] **Step 8: Commit pipeline support**

```bash
git add tests/multiview/test_pipeline_inputs.py pixal3d/pipelines/pixal3d_image_to_3d.py pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py
git commit -m "feat: run the Pixal3D cascade from calibrated views"
```

---

### Task 8: Add Calibrated `transforms.json` Inference Without Changing Single-View MoGe

**Files:**
- Create: `tests/multiview/test_inference_manifest.py`
- Modify: `inference.py:115-220`
- Modify: `inference.py` argument parser at the end of the file
- Modify: `README.md`

**Interfaces:**
- Consumes: existing render `transforms.json` with one to eight frames.
- Produces: `load_calibrated_manifest(path) -> tuple[list[Image.Image], dict]` and CLI `--transforms PATH` mutually exclusive with `--image`.

- [ ] **Step 1: Write failing manifest tests**

```python
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from inference import load_calibrated_manifest


def test_load_calibrated_manifest_preserves_frame_order(tmp_path):
    frames = []
    for index in range(2):
        Image.new("RGBA", (4, 4), color=(index * 10, 0, 0, 255)).save(tmp_path / f"{index:03d}.png")
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
    images, cameras = load_calibrated_manifest(manifest)
    assert len(images) == 2
    assert cameras["camera_angle_x"] == [0.7, 0.7]
    assert cameras["distance"] == pytest.approx([2.0, np.sqrt(5.0)])
    assert np.asarray(cameras["transform_matrix"]).shape == (2, 4, 4)
```

- [ ] **Step 2: Run manifest tests and verify RED**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_inference_manifest.py -q`

Expected: import failure for `load_calibrated_manifest`.

- [ ] **Step 3: Implement strict existing-format manifest loading**

```python
def load_calibrated_manifest(path):
    path = Path(path).resolve()
    metadata = json.loads(path.read_text())
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= 8:
        raise ValueError("transforms.json must contain between 1 and 8 frames")
    images = []
    angles = []
    distances = []
    transforms_out = []
    for frame in frames:
        image_path = (path.parent / frame["file_path"]).resolve()
        if not image_path.is_relative_to(path.parent):
            raise ValueError("frame file_path must remain inside the manifest directory")
        images.append(Image.open(image_path).convert("RGBA"))
        angle = frame.get("camera_angle_x", metadata.get("camera_angle_x"))
        if angle is None:
            raise KeyError("camera_angle_x is required for every frame")
        transform = np.asarray(frame["transform_matrix"], dtype=np.float32)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("every transform_matrix must be finite and 4 by 4")
        angles.append(float(angle))
        distances.append(float(np.linalg.norm(transform[:3, 3])))
        transforms_out.append(transform)
    return images, {
        "camera_angle_x": angles,
        "distance": distances,
        "mesh_scale": 1.0,
        "transform_matrix": np.stack(transforms_out),
}
```

Add the required production imports:

```python
import json
from pathlib import Path
from typing import Optional
```

- [ ] **Step 4: Add a mutually exclusive CLI input group**

Change the first argument of `run_inference` and add the manifest argument:

```diff
 def run_inference(
-    image_path: str,
+    image_path: Optional[str],
     output_path: str,
+    transforms_path: Optional[str] = None,
```

Replace the current input preprocessing and camera-estimation block with this branch. The single-image body is the current MoGe/manual-FOV logic, moved without mathematical changes:

```python
if transforms_path is not None:
    print(f"[Inference] Loading calibrated views: {transforms_path}")
    images, camera_params = load_calibrated_manifest(transforms_path)
    image_preprocessed = [pipeline.preprocess_image(image) for image in images]
else:
    if image_path is None:
        raise ValueError("image_path is required when transforms_path is not set")
    print(f"[Inference] Processing image: {image_path}")
    image = Image.open(image_path)
    image_preprocessed = pipeline.preprocess_image(image)
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
            torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
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

Keep the existing sampler and export code below this branch; it now passes either the single preprocessed image or the calibrated image list to `pipeline.run`.

Apply this exact parser and final-call diff; all arguments not shown remain unchanged:

```diff
     parser = argparse.ArgumentParser(description="Pixal3D Inference: Image to GLB")
-    parser.add_argument("--image", type=str, required=True, help="Path to input image")
+    inputs = parser.add_mutually_exclusive_group(required=True)
+    inputs.add_argument(
+        "--image", help="Single input image; camera may be estimated with MoGe-2"
+    )
+    inputs.add_argument(
+        "--transforms",
+        help="Calibrated multi-view transforms.json; first frame is anchor",
+    )
@@
     run_inference(
         image_path=args.image,
+        transforms_path=args.transforms,
         output_path=args.output,
```

- [ ] **Step 5: Run manifest tests and CLI parser checks**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview/test_inference_manifest.py -q`

Expected: all manifest tests pass.

Run: `conda run --no-capture-output -n pixal3d python inference.py --help`

Expected: usage displays `(--image IMAGE | --transforms TRANSFORMS)`.

- [ ] **Step 6: Document calibrated inference**

Add:

```bash
conda activate pixal3d
python inference.py \
  --transforms /root/node17/data/pixal3d/eval/Toys4k/example/transforms.json \
  --output /root/node17/data/pixal3d/train/runs/multiview/example.glb
```

State that the first frame is the anchor and multi-view input never estimates camera pose.

- [ ] **Step 7: Commit manifest inference**

```bash
git add tests/multiview/test_inference_manifest.py inference.py README.md
git commit -m "feat: add calibrated multiview inference manifest"
```

---

### Task 9: Run Regression, Smoke Handoff, and Paper-View Evaluation Gates

**Files:**
- Create: `tests/multiview/test_integration.py`
- Modify: `data_toolkit/README.md`
- Runtime input: `/root/node17/data/pixal3d/train/stage{1,2,3}/active`
- Runtime output: `/root/node17/data/pixal3d/train/reports/multiview`

**Interfaces:**
- Consumes: Tasks 1-8, audited 100-asset smoke packs, and Toys4K conditions.
- Produces: K=1 regression evidence, K=2/K=6 optimizer-step evidence, and K=2/K=4/K=6 evaluation reports.

- [ ] **Step 1: Add a lightweight integration test without downloading DINO weights**

Use this complete synthetic dense-path integration test; it exercises the real batch-wide selector, real dense collator, and the same five-dimensional conditioner dispatch used by all stages without loading DINO weights:

```python
from types import MethodType, SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from pixal3d.datasets.components import MultiViewImageConditionedMixin
from pixal3d.datasets.sparse_structure_latent import (
    MultiViewImageConditionedSparseStructureLatentView,
)
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    DinoV3ProjFeatureExtractor,
)


class IntegrationConditioner(DinoV3ProjFeatureExtractor):
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


def integration_sample(offset):
    transforms = torch.eye(4).repeat(8, 1, 1)
    transforms[:, 0, 3] = torch.arange(8, dtype=torch.float32)
    return {
        "x_0": torch.full((2, 2, 2, 2), float(offset)),
        "cond": torch.arange(offset, offset + 8 * 12).reshape(8, 3, 2, 2).float(),
        "camera_angle_x": torch.full((8,), 0.7),
        "camera_distance": torch.linalg.vector_norm(transforms[:, :3, 3], dim=-1) + 2.5,
        "transform_matrix": transforms,
        "mesh_scale": torch.tensor(1.0),
        "view_indices": torch.arange(8),
    }


def test_dense_batch_reaches_multiview_condition_dictionary(monkeypatch):
    monkeypatch.setattr(np.random, "randint", lambda low, high: 4)
    selector = SimpleNamespace(min_condition_views=2, max_condition_views=6)
    selector.select_batch_condition_views = MethodType(
        MultiViewImageConditionedMixin.select_batch_condition_views, selector
    )
    batch = MultiViewImageConditionedSparseStructureLatentView.collate_fn(
        selector, [integration_sample(0), integration_sample(100)]
    )
    global_feature, projected_feature = IntegrationConditioner()(
        batch["cond"],
        camera_angle_x=batch["camera_angle_x"],
        distance=batch["camera_distance"],
        mesh_scale=batch["mesh_scale"],
        transform_matrix=batch["transform_matrix"],
    )
    condition = {"global": global_feature, "proj": projected_feature}
    assert set(condition) == {"global", "proj"}
    assert condition["global"].shape[0] == 2
    assert condition["proj"].shape[0] == 2
    assert batch["cond"].shape[:2] == (2, 4)
```

- [ ] **Step 2: Run the complete unit suite**

Run: `conda run --no-capture-output -n pixal3d python -m pytest tests/multiview tests/data_toolkit -q`

Expected: zero failures.

- [ ] **Step 3: Verify code-quality and scope constraints**

Run: `git diff --check`

Expected: no output.

Run: `rg -n "view_mask|SetTransformer|learned.*fusion|visibility.*weight|pose_embedding" pixal3d configs/gen`

Expected: no new baseline implementation references.

- [ ] **Step 4: Finish the data hardware preflight and 100-asset smoke gate**

Run the existing preprocessing plan Tasks 13 and 14 using:

```bash
conda run --no-capture-output -n pixal3d python -m data_toolkit.pipeline.cli hardware-preflight \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --bootstrap-peak-local-gib 350
conda run --no-capture-output -n pixal3d python -m data_toolkit.pipeline.cli plan \
  --config data_toolkit/configs/multiview_preprocess.yaml --gate smoke
conda run --no-capture-output -n pixal3d python -m data_toolkit.pipeline.cli run \
  --config data_toolkit/configs/multiview_preprocess.yaml --gate smoke
conda run --no-capture-output -n pixal3d python -m data_toolkit.pipeline.cli audit \
  --config data_toolkit/configs/multiview_preprocess.yaml --gate smoke
```

Expected: hardware and smoke reports pass with local artifacts under `/root/node17/data/pixal3d`; no training starts during these commands.

- [ ] **Step 5: Materialize one smoke handoff and validate both target anchors**

Materialize the first audited smoke batch with these exact commands:

```bash
mkdir -p /root/node17/data/pixal3d/train/stage1/active
tar -xf /root/data2/pixal3d/prepared/common/ObjaverseXL_sketchfab/ObjaverseXL_sketchfab-00000/batch000.tar -C /root/node17/data/pixal3d/train/stage1/active
tar -xf /root/data2/pixal3d/prepared/ss/64/ObjaverseXL_sketchfab/ObjaverseXL_sketchfab-00000/batch000.tar -C /root/node17/data/pixal3d/train/stage1/active

mkdir -p /root/node17/data/pixal3d/train/stage2/active
tar -xf /root/data2/pixal3d/prepared/common/ObjaverseXL_sketchfab/ObjaverseXL_sketchfab-00000/batch000.tar -C /root/node17/data/pixal3d/train/stage2/active
for resolution in 256 512 1024; do
  tar -xf "/root/data2/pixal3d/prepared/shape/${resolution}/ObjaverseXL_sketchfab/ObjaverseXL_sketchfab-00000/batch000.tar" -C /root/node17/data/pixal3d/train/stage2/active
done

mkdir -p /root/node17/data/pixal3d/train/stage3/active
tar -xf /root/data2/pixal3d/prepared/common/ObjaverseXL_sketchfab/ObjaverseXL_sketchfab-00000/batch000.tar -C /root/node17/data/pixal3d/train/stage3/active
for resolution in 256 512 1024; do
  tar -xf "/root/data2/pixal3d/prepared/shape/${resolution}/ObjaverseXL_sketchfab/ObjaverseXL_sketchfab-00000/batch000.tar" -C /root/node17/data/pixal3d/train/stage3/active
  tar -xf "/root/data2/pixal3d/prepared/pbr/${resolution}/ObjaverseXL_sketchfab/ObjaverseXL_sketchfab-00000/batch000.tar" -C /root/node17/data/pixal3d/train/stage3/active
done
```

Load one `view00` and one `view01` sample from every stage, force K=2 and K=6 at collation, and verify every transform and conditioner output with `torch.isfinite(...).all()` before promoting the handoff.

- [ ] **Step 6: Run K=1 numerical regression on real DINOv3/NAF**

With one audited smoke asset and fixed FP32 inputs, run the original four-dimensional conditioner and the new five-dimensional K=1 branch for SS, Shape 1024, and PBR 1024. Require:

```python
torch.testing.assert_close(new_global, old_global, rtol=1e-5, atol=1e-6)
torch.testing.assert_close(new_proj, old_proj, rtol=1e-5, atol=1e-6)
```

Record results in `/root/node17/data/pixal3d/train/reports/multiview/k1_regression.json`.

- [ ] **Step 7: Run one optimizer step for each final-resolution stage**

Run each Task 6 config with one smoke handoff batch and terminate after the first completed optimizer step. Require finite loss, no checkpoint-key warning, and GPU memory below the physical limit on every rank. Store logs under `/root/node17/data/pixal3d/train/reports/multiview/optimizer_smoke`.

- [ ] **Step 8: Fine-tune only after preprocessing stops at a handoff gate**

Start SS 64, Shape 1024, and PBR 1024 independently from their single-view weights. Keep K uniformly batch-sampled from 2 through 6. Do not run Blender, voxelization, latent encoding, packing, or archival while a training job is active.

- [ ] **Step 9: Evaluate Toys4K at the paper's three view counts**

For every Toys4K asset, keep frame zero as anchor and evaluate prefixes of length 2, 4, and 6. Write separate reports:

```text
/root/node17/data/pixal3d/train/reports/multiview/toys4k_k2.json
/root/node17/data/pixal3d/train/reports/multiview/toys4k_k4.json
/root/node17/data/pixal3d/train/reports/multiview/toys4k_k6.json
```

Do not add new metrics before the paper-aligned CD, EMD, and F-Score reports are complete.

- [ ] **Step 10: Commit integration coverage and runbook**

```bash
git add tests/multiview/test_integration.py data_toolkit/README.md
git commit -m "test: gate paper-faithful multiview training"
```

---

## Plan Self-Review Checklist

- [x] Every design requirement maps to Tasks 1-9.
- [x] No task changes the denoiser architecture or adds learned fusion.
- [x] Camera transforms and both arithmetic means match the official paper branch.
- [x] Batch-wide K=2 through 6 is implemented without padding or masks.
- [x] K=1 through K=8 calibrated inference is covered.
- [x] Existing single-view files and CLI behavior remain covered by regression tests.
- [x] SS, Shape, and PBR each have an exact final-resolution config and checkpoint path.
- [x] Preprocessing and full training never overlap.
- [x] Every production-code step follows a failing-test, minimal-change, passing-test cycle.
- [x] All runtime paths use `/root/node17/data/pixal3d`.
