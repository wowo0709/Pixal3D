# Controlled Correspondence Feature Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate deterministic multi-view corruptions with exact oracle metadata and run the checkpoint-independent Gate A comparison of equal mean, consensus routing, and oracle routing at the projected-feature level.

**Architecture:** A small experiment package creates controlled image variants from calibrated clean views and foreground masks, saving every mask and transform needed for an oracle ceiling. Pure metric and visualization helpers compare clean-reference sparse features against corrupted equal-mean, consensus, and oracle conditions. A CLI runner invokes only the image conditioners and never samples a flow model, so it can validate routing behavior before the pending multi-view flow checkpoints arrive.

**Tech Stack:** Python 3, PyTorch, NumPy, PIL, OpenCV, pytest, JSON

## Global Constraints

- Work only on `feature/multiview-correspondence-node11`.
- This plan implements Gate A only; it does not run Shape/PBR flow generation.
- Use exactly four calibrated views by default and corrupt one non-anchor view.
- Use seed `42` by default.
- Generate oracle masks and controlled warp metadata in the tooling; do not require user-authored oracle data.
- Foreground masks select and score corruption regions but never hard-zero Pixal3D features.
- Do not infer oracle masks or correspondences for real VLM-generated images.
- Do not use the last mesh, rendered depth, local source-feature search, learned reliability, or deformation transport.
- Store source hashes, corruption parameters, exact masks, forward maps, inverse sampling grids, and hole masks.
- Preserve calibrated view order and camera metadata exactly.
- Compare clean-reference equal mean against corrupted S0, S1, and S3 before running stress arms.
- Do not substitute public single-view flow checkpoints for the pending multi-view checkpoints.

---

## File Structure

- Create `pixal3d/experiments/__init__.py`: experiment package marker.
- Create `pixal3d/experiments/correspondence/__init__.py`: public controlled-corruption and metric exports.
- Create `pixal3d/experiments/correspondence/corruptions.py`: deterministic corruption generation and exact coordinate maps.
- Create `pixal3d/experiments/correspondence/artifacts.py`: hashes, manifests, array/image serialization, and contact sheets.
- Create `pixal3d/experiments/correspondence/metrics.py`: feature-level preservation and routing metrics.
- Create `pixal3d/experiments/correspondence/visualization.py`: sparse-volume slice heatmaps and feature/weight panels.
- Create `scripts/generate_correspondence_cases.py`: controlled dataset CLI.
- Create `scripts/run_correspondence_feature_gate.py`: image-conditioner-only Gate A runner.
- Create `tests/multiview/test_correspondence_corruptions.py`: deterministic corruption and coordinate-map tests.
- Create `tests/multiview/test_correspondence_artifacts.py`: bundle and path-safety tests.
- Create `tests/multiview/test_correspondence_metrics.py`: metric and visualization tests.
- Create `tests/multiview/test_correspondence_feature_gate.py`: runner parsing, validation, and mocked-conditioner tests.

### Task 1: Controlled corruption data contract and deterministic region selection

**Files:**

- Create: `pixal3d/experiments/__init__.py`
- Create: `pixal3d/experiments/correspondence/__init__.py`
- Create: `pixal3d/experiments/correspondence/corruptions.py`
- Create: `tests/multiview/test_correspondence_corruptions.py`

**Interfaces:**

- Consumes: RGB `PIL.Image`, foreground `PIL.Image` or `ndarray[H,W]`, case name, and integer seed.
- Produces: immutable `ControlledCorruption` with RGB image, soft mask, source-to-destination forward map, destination-to-source inverse grid, hole mask, and JSON-safe parameters.

- [ ] **Step 1: Write failing contract and empty-foreground tests**

```python
import numpy as np
import pytest
from PIL import Image

from pixal3d.experiments.correspondence.corruptions import (
    ControlledCorruption,
    make_identity_coordinate_map,
    select_foreground_patch,
)


def test_identity_coordinate_map_uses_pixel_xy_order():
    coordinates = make_identity_coordinate_map(height=2, width=3)
    assert coordinates.shape == (2, 3, 2)
    np.testing.assert_array_equal(coordinates[1, 2], np.array([2.0, 1.0]))


def test_select_foreground_patch_is_deterministic_and_inside_foreground():
    foreground = np.zeros((16, 16), dtype=np.float32)
    foreground[4:12, 3:13] = 1.0
    first = select_foreground_patch(foreground, seed=42, fraction=0.35)
    second = select_foreground_patch(foreground, seed=42, fraction=0.35)
    np.testing.assert_array_equal(first, second)
    assert first.shape == foreground.shape
    assert first.max() == 1.0
    assert np.all(first[foreground == 0] == 0)


def test_select_foreground_patch_rejects_empty_mask():
    with pytest.raises(ValueError, match="foreground mask is empty"):
        select_foreground_patch(np.zeros((8, 8), dtype=np.float32), seed=42)
```

- [ ] **Step 2: Run the tests and verify that the package is missing**

Run:

```bash
pytest tests/multiview/test_correspondence_corruptions.py -v
```

Expected: FAIL during import.

- [ ] **Step 3: Implement the data class, identity map, and deterministic patch**

```python
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ControlledCorruption:
    name: str
    image: Image.Image
    mask: np.ndarray
    forward_map: np.ndarray
    inverse_grid: np.ndarray
    hole_mask: np.ndarray
    parameters: dict[str, Any]

    def __post_init__(self) -> None:
        width, height = self.image.size
        if self.image.mode != "RGB":
            raise ValueError("corruption image must be RGB")
        if self.mask.shape != (height, width):
            raise ValueError("mask shape must match image")
        if self.forward_map.shape != (height, width, 2):
            raise ValueError("forward_map must have shape [H, W, 2]")
        if self.inverse_grid.shape != (height, width, 2):
            raise ValueError("inverse_grid must have shape [H, W, 2]")
        if self.hole_mask.shape != (height, width):
            raise ValueError("hole_mask shape must match image")


def make_identity_coordinate_map(height: int, width: int) -> np.ndarray:
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        indexing="xy",
    )
    return np.stack((x, y), axis=-1)


def _foreground_array(foreground, size: tuple[int, int]) -> np.ndarray:
    if isinstance(foreground, Image.Image):
        mask = np.asarray(
            foreground.convert("L").resize(size, Image.Resampling.NEAREST),
            dtype=np.float32,
        ) / 255.0
    else:
        mask = np.asarray(foreground, dtype=np.float32)
        if mask.shape != (size[1], size[0]):
            raise ValueError("foreground mask shape must match image")
    return np.clip(mask, 0.0, 1.0)


def select_foreground_patch(
    foreground: np.ndarray,
    *,
    seed: int,
    fraction: float = 0.3,
) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    foreground = np.asarray(foreground, dtype=np.float32)
    candidates = np.argwhere(foreground >= 0.5)
    if candidates.size == 0:
        raise ValueError("foreground mask is empty")
    generator = np.random.default_rng(seed)
    center_y, center_x = candidates[generator.integers(len(candidates))]
    y0, x0 = candidates.min(axis=0)
    y1, x1 = candidates.max(axis=0) + 1
    patch_h = max(1, int(round((y1 - y0) * fraction)))
    patch_w = max(1, int(round((x1 - x0) * fraction)))
    top = int(np.clip(center_y - patch_h // 2, 0, foreground.shape[0] - patch_h))
    left = int(np.clip(center_x - patch_w // 2, 0, foreground.shape[1] - patch_w))
    patch = np.zeros_like(foreground, dtype=np.float32)
    patch[top : top + patch_h, left : left + patch_w] = 1.0
    return patch * (foreground >= 0.5)
```

- [ ] **Step 4: Run contract tests**

Run:

```bash
pytest tests/multiview/test_correspondence_corruptions.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit the corruption contract**

```bash
git add pixal3d/experiments tests/multiview/test_correspondence_corruptions.py
git commit -m "feat: define controlled corruption contract"
```

### Task 2: Photometric, pattern, and deletion oracle cases

**Files:**

- Modify: `pixal3d/experiments/correspondence/corruptions.py`
- Modify: `pixal3d/experiments/correspondence/__init__.py`
- Modify: `tests/multiview/test_correspondence_corruptions.py`

**Interfaces:**

- Consumes: clean RGB image, foreground mask, seed, and one of `material`, `pattern`, `deletion`.
- Produces: a `ControlledCorruption` whose non-warp coordinate maps are identity and whose exact changed-pixel mask is nonzero only inside foreground.

- [ ] **Step 1: Write failing deterministic and locality tests**

```python
from pixal3d.experiments.correspondence.corruptions import (
    create_controlled_corruption,
)


@pytest.mark.parametrize("case", ["material", "pattern", "deletion"])
def test_nonwarp_corruptions_are_deterministic_local_and_nonempty(case):
    clean = Image.new("RGB", (32, 24), color=(80, 120, 160))
    foreground = np.zeros((24, 32), dtype=np.float32)
    foreground[3:21, 4:28] = 1.0

    first = create_controlled_corruption(
        clean, foreground, case=case, seed=42
    )
    second = create_controlled_corruption(
        clean, foreground, case=case, seed=42
    )

    np.testing.assert_array_equal(np.asarray(first.image), np.asarray(second.image))
    np.testing.assert_array_equal(first.mask, second.mask)
    assert first.mask.sum() > 0
    assert np.all(first.mask[foreground == 0] == 0)
    assert np.any(np.asarray(first.image) != np.asarray(clean))
    identity = make_identity_coordinate_map(24, 32)
    np.testing.assert_array_equal(first.forward_map, identity)
    np.testing.assert_array_equal(first.inverse_grid, identity)
    assert not first.hole_mask.any()


def test_unknown_corruption_case_is_rejected():
    with pytest.raises(ValueError, match="material, pattern, deletion, affine, smooth_warp"):
        create_controlled_corruption(
            Image.new("RGB", (8, 8)),
            np.ones((8, 8), dtype=np.float32),
            case="unknown",
            seed=42,
        )
```

- [ ] **Step 2: Run the non-warp tests and verify the missing factory**

Run:

```bash
pytest tests/multiview/test_correspondence_corruptions.py -k "nonwarp or unknown" -v
```

Expected: FAIL because `create_controlled_corruption` is not defined.

- [ ] **Step 3: Implement deterministic masked compositing**

```python
def _composite(
    clean: np.ndarray,
    replacement: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    alpha = mask[..., None].astype(np.float32)
    result = (
        clean.astype(np.float32) * (1.0 - alpha)
        + replacement.astype(np.float32) * alpha
    )
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def _material_replacement(clean: np.ndarray) -> np.ndarray:
    hsv = np.asarray(Image.fromarray(clean).convert("HSV"), dtype=np.uint8).copy()
    hsv[..., 0] = (hsv[..., 0].astype(np.uint16) + 37) % 256
    hsv[..., 1] = np.clip(
        hsv[..., 1].astype(np.float32) * 1.25, 0, 255
    ).astype(np.uint8)
    hsv[..., 2] = np.clip(
        hsv[..., 2].astype(np.float32) * 0.8 + 24, 0, 255
    ).astype(np.uint8)
    return np.asarray(Image.fromarray(hsv, mode="HSV").convert("RGB"))


def _pattern_replacement(clean: np.ndarray) -> np.ndarray:
    height, width = clean.shape[:2]
    x, y = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    checker = ((x // 4 + y // 4) % 2)[..., None]
    colors = np.array([[235, 45, 55], [25, 220, 215]], dtype=np.uint8)
    return colors[checker[..., 0]]


def _deletion_replacement(clean: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ring = np.logical_and(
        cv2.dilate((mask > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0,
        mask == 0,
    )
    samples = clean[ring]
    local_median = np.median(samples, axis=0) if len(samples) else np.median(
        clean.reshape(-1, 3), axis=0
    )
    fill = np.clip(local_median * 0.65 + 16.0, 0, 255)
    replacement = np.empty_like(clean)
    replacement[...] = np.rint(fill).astype(np.uint8)
    return replacement
```

- [ ] **Step 4: Implement the public factory for the three non-warp cases**

```python
def create_controlled_corruption(
    image: Image.Image,
    foreground,
    *,
    case: str,
    seed: int,
    fraction: float = 0.3,
    max_displacement: float = 6.0,
) -> ControlledCorruption:
    allowed = {"material", "pattern", "deletion", "affine", "smooth_warp"}
    if case not in allowed:
        raise ValueError(
            "case must be material, pattern, deletion, affine, or smooth_warp"
        )
    clean_image = image.convert("RGB")
    clean = np.asarray(clean_image, dtype=np.uint8)
    foreground_array = _foreground_array(foreground, clean_image.size)
    patch = select_foreground_patch(
        foreground_array, seed=seed, fraction=fraction
    )
    if case in {"affine", "smooth_warp"}:
        return _create_warp_corruption(
            clean_image,
            patch,
            case=case,
            seed=seed,
            max_displacement=max_displacement,
        )
    replacements = {
        "material": _material_replacement(clean),
        "pattern": _pattern_replacement(clean),
        "deletion": _deletion_replacement(clean, patch),
    }
    corrupted = _composite(clean, replacements[case], patch)
    identity = make_identity_coordinate_map(*clean.shape[:2])
    return ControlledCorruption(
        name=case,
        image=Image.fromarray(corrupted, mode="RGB"),
        mask=patch.astype(np.float32),
        forward_map=identity,
        inverse_grid=identity.copy(),
        hole_mask=np.zeros(patch.shape, dtype=np.uint8),
        parameters={"seed": seed, "fraction": fraction},
    )
```

Import `cv2` and export `create_controlled_corruption` from the package `__init__.py`.

- [ ] **Step 5: Run all corruption tests**

Run:

```bash
pytest tests/multiview/test_correspondence_corruptions.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit the non-warp cases**

```bash
git add pixal3d/experiments/correspondence tests/multiview/test_correspondence_corruptions.py
git commit -m "feat: add controlled appearance corruptions"
```

### Task 3: Affine and smooth controlled warps with oracle maps

**Files:**

- Modify: `pixal3d/experiments/correspondence/corruptions.py`
- Modify: `tests/multiview/test_correspondence_corruptions.py`

**Interfaces:**

- Consumes: clean image, selected patch, seed, and maximum displacement.
- Produces: bounded source-to-destination forward maps and numerically inverted destination-to-source sampling grids in absolute pixel `(x,y)` coordinates.

- [ ] **Step 1: Write failing affine round-trip and smooth-bound tests**

```python
@pytest.mark.parametrize("case", ["affine", "smooth_warp"])
def test_warp_coordinate_maps_round_trip_on_valid_pixels(case):
    width, height = 40, 32
    x, y = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    clean = np.stack(
        (x * 5 % 256, y * 7 % 256, (x + y) * 3 % 256), axis=-1
    ).astype(np.uint8)
    foreground = np.zeros((height, width), dtype=np.float32)
    foreground[4:28, 5:35] = 1.0
    result = create_controlled_corruption(
        Image.fromarray(clean),
        foreground,
        case=case,
        seed=42,
        max_displacement=4.0,
    )

    source = result.inverse_grid
    forward_x = cv2.remap(
        result.forward_map[..., 0],
        source[..., 0],
        source[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-100,
    )
    forward_y = cv2.remap(
        result.forward_map[..., 1],
        source[..., 0],
        source[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-100,
    )
    destination = make_identity_coordinate_map(height, width)
    valid = result.hole_mask == 0
    error = np.linalg.norm(
        np.stack((forward_x, forward_y), axis=-1) - destination,
        axis=-1,
    )
    assert np.quantile(error[valid], 0.99) < 0.35
    assert result.mask.sum() > 0


def test_smooth_warp_is_bounded_and_spatially_smooth():
    result = create_controlled_corruption(
        Image.new("RGB", (48, 48), color=(80, 120, 160)),
        np.pad(np.ones((32, 32), dtype=np.float32), 8),
        case="smooth_warp",
        seed=42,
        max_displacement=5.0,
    )
    identity = make_identity_coordinate_map(48, 48)
    displacement = result.forward_map - identity
    assert np.linalg.norm(displacement, axis=-1).max() <= 5.01
    neighbor_jump = np.linalg.norm(
        displacement[:, 1:] - displacement[:, :-1], axis=-1
    )
    assert np.quantile(neighbor_jump, 0.99) < 1.5
```

- [ ] **Step 2: Run the warp tests and verify the private implementation is missing**

Run:

```bash
pytest tests/multiview/test_correspondence_corruptions.py -k "warp" -v
```

Expected: FAIL because `_create_warp_corruption` is not defined.

- [ ] **Step 3: Implement affine and smooth source displacement fields**

```python
def _affine_forward_map(
    height: int,
    width: int,
    mask: np.ndarray,
    *,
    seed: int,
    max_displacement: float,
) -> tuple[np.ndarray, dict]:
    generator = np.random.default_rng(seed)
    identity = make_identity_coordinate_map(height, width)
    support = np.argwhere(mask > 0)
    center_y, center_x = support.mean(axis=0)
    angle = float(generator.uniform(-8.0, 8.0))
    scale = float(generator.uniform(0.94, 1.06))
    translation = generator.uniform(
        -max_displacement, max_displacement, size=2
    ).astype(np.float32)
    matrix = cv2.getRotationMatrix2D(
        (float(center_x), float(center_y)), angle, scale
    ).astype(np.float32)
    matrix[:, 2] += translation
    homogeneous = np.concatenate(
        (identity, np.ones((height, width, 1), dtype=np.float32)), axis=-1
    )
    transformed = homogeneous @ matrix.T
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.5)[..., None]
    forward = identity * (1.0 - soft) + transformed * soft
    displacement = forward - identity
    magnitude = np.linalg.norm(displacement, axis=-1, keepdims=True)
    displacement *= np.minimum(
        1.0, max_displacement / np.maximum(magnitude, 1e-6)
    )
    return identity + displacement, {
        "angle_degrees": angle,
        "scale": scale,
        "translation_xy": translation.tolist(),
    }


def _smooth_forward_map(
    height: int,
    width: int,
    mask: np.ndarray,
    *,
    seed: int,
    max_displacement: float,
) -> tuple[np.ndarray, dict]:
    generator = np.random.default_rng(seed)
    coarse = generator.normal(0.0, 1.0, size=(5, 5, 2)).astype(np.float32)
    displacement = cv2.resize(
        coarse, (width, height), interpolation=cv2.INTER_CUBIC
    )
    displacement = cv2.GaussianBlur(displacement, (0, 0), 3.0)
    magnitude = np.linalg.norm(displacement, axis=-1, keepdims=True)
    displacement = displacement / np.maximum(magnitude.max(), 1e-6)
    displacement *= max_displacement
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2.0)[..., None]
    displacement *= soft
    return make_identity_coordinate_map(height, width) + displacement, {
        "coarse_grid": [5, 5],
        "max_displacement": max_displacement,
    }
```

- [ ] **Step 4: Implement fixed-point inversion and image remapping**

```python
def _invert_forward_map(
    forward_map: np.ndarray,
    *,
    iterations: int = 20,
) -> np.ndarray:
    height, width = forward_map.shape[:2]
    identity = make_identity_coordinate_map(height, width)
    displacement = forward_map - identity
    source = identity.copy()
    for _ in range(iterations):
        sampled_x = cv2.remap(
            displacement[..., 0],
            source[..., 0],
            source[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        sampled_y = cv2.remap(
            displacement[..., 1],
            source[..., 0],
            source[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        source = identity - np.stack((sampled_x, sampled_y), axis=-1)
    return source.astype(np.float32)


def _create_warp_corruption(
    image: Image.Image,
    mask: np.ndarray,
    *,
    case: str,
    seed: int,
    max_displacement: float,
) -> ControlledCorruption:
    clean = np.asarray(image, dtype=np.uint8)
    height, width = clean.shape[:2]
    if max_displacement <= 0:
        raise ValueError("max_displacement must be positive")
    if case == "affine":
        forward, parameters = _affine_forward_map(
            height,
            width,
            mask,
            seed=seed,
            max_displacement=max_displacement,
        )
    else:
        forward, parameters = _smooth_forward_map(
            height,
            width,
            mask,
            seed=seed,
            max_displacement=max_displacement,
        )
    inverse = _invert_forward_map(forward)
    holes = (
        (inverse[..., 0] < 0)
        | (inverse[..., 0] > width - 1)
        | (inverse[..., 1] < 0)
        | (inverse[..., 1] > height - 1)
    )
    warped = cv2.remap(
        clean,
        inverse[..., 0],
        inverse[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    identity = make_identity_coordinate_map(height, width)
    source_support = (
        np.linalg.norm(forward - identity, axis=-1) > 0.05
    ).astype(np.float32)
    destination_support = cv2.remap(
        source_support,
        inverse[..., 0],
        inverse[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    oracle_mask = np.maximum(
        destination_support, holes.astype(np.float32)
    )
    parameters.update(
        {
            "seed": seed,
            "max_displacement": max_displacement,
            "inverse_iterations": 20,
        }
    )
    return ControlledCorruption(
        name=case,
        image=Image.fromarray(warped, mode="RGB"),
        mask=oracle_mask.astype(np.float32),
        forward_map=forward.astype(np.float32),
        inverse_grid=inverse,
        hole_mask=holes.astype(np.uint8),
        parameters=parameters,
    )
```

- [ ] **Step 5: Run corruption tests**

Run:

```bash
pytest tests/multiview/test_correspondence_corruptions.py -v
```

Expected: PASS, including the 99th-percentile round-trip bound.

- [ ] **Step 6: Commit controlled warps**

```bash
git add pixal3d/experiments/correspondence/corruptions.py tests/multiview/test_correspondence_corruptions.py
git commit -m "feat: add controlled warp oracle maps"
```

### Task 4: Reproducible corruption bundles and contact sheets

**Files:**

- Create: `pixal3d/experiments/correspondence/artifacts.py`
- Create: `scripts/generate_correspondence_cases.py`
- Create: `tests/multiview/test_correspondence_artifacts.py`

**Interfaces:**

- Consumes: a calibrated `transforms.json`, case list, target view index, seed, and output directory.
- Produces: one self-contained directory per case with copied clean views, one corrupted view, exact oracle arrays, a calibrated manifest, `case.json`, SHA-256 hashes, and `contact_sheet.png`.

- [ ] **Step 1: Write failing safe-manifest and bundle-content tests**

```python
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixal3d.experiments.correspondence.artifacts import (
    load_controlled_source_manifest,
    write_corruption_bundle,
)
from pixal3d.experiments.correspondence.corruptions import (
    create_controlled_corruption,
)


def _write_source_manifest(tmp_path: Path) -> Path:
    frames = []
    for index in range(4):
        path = tmp_path / f"{index:03d}.png"
        Image.new("RGBA", (16, 16), color=(40 * index, 80, 120, 255)).save(path)
        frames.append(
            {
                "file_path": path.name,
                "camera_angle_x": 0.7,
                "transform_matrix": np.eye(4).tolist(),
            }
        )
    manifest = tmp_path / "transforms.json"
    manifest.write_text(json.dumps({"frames": frames}))
    return manifest


def test_source_manifest_rejects_escaping_frame_path(tmp_path):
    manifest = tmp_path / "transforms.json"
    manifest.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "file_path": "../outside.png",
                        "camera_angle_x": 0.7,
                        "transform_matrix": np.eye(4).tolist(),
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="inside"):
        load_controlled_source_manifest(manifest)


def test_bundle_contains_exact_oracle_artifacts_and_hashes(tmp_path):
    manifest = _write_source_manifest(tmp_path)
    source = load_controlled_source_manifest(manifest)
    result = create_controlled_corruption(
        source.images[2],
        source.foreground_masks[2],
        case="pattern",
        seed=42,
    )
    output = tmp_path / "bundle"
    write_corruption_bundle(
        source,
        result,
        target_view=2,
        output_dir=output,
    )

    expected = {
        "transforms.json",
        "case.json",
        "oracle_mask.png",
        "forward_map.npy",
        "inverse_grid.npy",
        "hole_mask.png",
        "warp_field.png",
        "contact_sheet.png",
        "views/000.png",
        "views/001.png",
        "views/002.png",
        "views/003.png",
    }
    assert expected <= {
        str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()
    }
    metadata = json.loads((output / "case.json").read_text())
    assert metadata["target_view"] == 2
    assert metadata["case"] == "pattern"
    assert len(metadata["sha256"]["source_view"]) == 64
    assert len(metadata["sha256"]["corrupted_view"]) == 64
```

- [ ] **Step 2: Run artifact tests and verify the missing module**

Run:

```bash
pytest tests/multiview/test_correspondence_artifacts.py -v
```

Expected: FAIL during import.

- [ ] **Step 3: Implement safe source loading and alpha-derived foreground masks**

```python
@dataclass(frozen=True)
class ControlledSource:
    manifest_path: Path
    metadata: dict
    images: tuple[Image.Image, ...]
    foreground_masks: tuple[Image.Image, ...]


def load_controlled_source_manifest(path: Path) -> ControlledSource:
    path = path.resolve()
    metadata = json.loads(path.read_text())
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= 8:
        raise ValueError("manifest must contain between 1 and 8 frames")
    images = []
    masks = []
    for index, frame in enumerate(frames):
        relative = frame.get("file_path")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"frame {index} requires file_path")
        image_path = (path.parent / relative).resolve()
        if not image_path.is_relative_to(path.parent):
            raise ValueError("frame file_path must remain inside manifest directory")
        with Image.open(image_path) as opened:
            rgba = opened.convert("RGBA")
            images.append(rgba.convert("RGB"))
            masks.append(rgba.getchannel("A"))
    return ControlledSource(path, metadata, tuple(images), tuple(masks))
```

- [ ] **Step 4: Implement atomic bundle serialization and contact sheet creation**

Use these exact hash and contact-sheet helpers:

```python
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_image(image: Image.Image) -> str:
    array = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return hashlib.sha256(array.tobytes()).hexdigest()


def make_contact_sheet(
    clean_views: Sequence[Image.Image],
    corrupted_views: Sequence[Image.Image],
    mask: Image.Image,
) -> Image.Image:
    tiles = [
        *(image.convert("RGB") for image in clean_views),
        *(image.convert("RGB") for image in corrupted_views),
        mask.convert("RGB"),
    ]
    width = max(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (width * len(tiles), height), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (index * width, 0))
    return sheet


def render_warp_field(forward_map: np.ndarray) -> Image.Image:
    height, width = forward_map.shape[:2]
    displacement = forward_map - make_identity_coordinate_map(height, width)
    angle = (
        np.arctan2(displacement[..., 1], displacement[..., 0])
        + np.pi
    ) / (2 * np.pi)
    magnitude = np.linalg.norm(displacement, axis=-1)
    scale = max(float(np.quantile(magnitude, 0.99)), 1e-6)
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[..., 0] = np.clip(angle * 179, 0, 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude / scale * 255, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(rgb, mode="RGB")
```

Implement atomic serialization with:

```python
def write_corruption_bundle(
    source: ControlledSource,
    corruption: ControlledCorruption,
    *,
    target_view: int,
    output_dir: Path,
) -> None:
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"output bundle already exists: {output_dir}")
    if not 0 <= target_view < len(source.images):
        raise ValueError("target_view is outside the calibrated view range")

    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-",
        dir=output_dir.parent,
    ) as temporary:
        root = Path(temporary) / "bundle"
        views_dir = root / "views"
        views_dir.mkdir(parents=True)
        corrupted_views = list(source.images)
        corrupted_views[target_view] = corruption.image
        for index, image in enumerate(corrupted_views):
            image.save(views_dir / f"{index:03d}.png")

        mask_image = Image.fromarray(
            np.clip(np.rint(corruption.mask * 255), 0, 255).astype(np.uint8),
            mode="L",
        )
        mask_image.save(root / "oracle_mask.png")
        Image.fromarray(
            corruption.hole_mask.astype(np.uint8) * 255, mode="L"
        ).save(root / "hole_mask.png")
        np.save(root / "forward_map.npy", corruption.forward_map)
        np.save(root / "inverse_grid.npy", corruption.inverse_grid)
        render_warp_field(corruption.forward_map).save(
            root / "warp_field.png"
        )
        make_contact_sheet(
            source.images, corrupted_views, mask_image
        ).save(root / "contact_sheet.png")

        manifest = copy.deepcopy(source.metadata)
        for index, frame in enumerate(manifest["frames"]):
            frame["file_path"] = f"views/{index:03d}.png"
        manifest["frames"][target_view]["oracle_mask_path"] = (
            "oracle_mask.png"
        )
        manifest["controlled_case_path"] = "case.json"
        (root / "transforms.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )

        metadata = {
            "case": corruption.name,
            "target_view": target_view,
            "parameters": corruption.parameters,
            "sha256": {
                "source_view": _sha256_image(source.images[target_view]),
                "corrupted_view": _sha256_image(corruption.image),
                "oracle_mask": sha256_file(root / "oracle_mask.png"),
                "forward_map": sha256_file(root / "forward_map.npy"),
                "inverse_grid": sha256_file(root / "inverse_grid.npy"),
                "hole_mask": sha256_file(root / "hole_mask.png"),
                "warp_field": sha256_file(root / "warp_field.png"),
            },
        }
        (root / "case.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True)
        )
        root.replace(output_dir)
```

This changes only bundle-local frame paths; all camera values and frame order remain exact copies. The ignored `oracle_mask_path` and `controlled_case_path` fields make oracle provenance explicit without changing ordinary inference parsing.

- [ ] **Step 5: Add the dataset-generation CLI**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transforms", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cases",
        default="material,pattern,deletion,affine,smooth_warp",
    )
    parser.add_argument("--target-view", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=0.3)
    parser.add_argument("--max-displacement", type=float, default=6.0)
    parser.add_argument("--allow-anchor-corruption", action="store_true")
    return parser
```

Add this programmatic entry point and have `main` call it:

```python
def generate_cases(
    *,
    transforms: Path,
    output_dir: Path,
    cases: Sequence[str],
    target_view: int,
    seed: int,
    fraction: float,
    max_displacement: float,
    allow_anchor_corruption: bool,
) -> tuple[Path, ...]:
    source = load_controlled_source_manifest(transforms)
    if not 0 <= target_view < len(source.images):
        raise ValueError("target_view is outside the calibrated view range")
    if target_view == 0 and not allow_anchor_corruption:
        raise ValueError(
            "anchor corruption requires allow_anchor_corruption=True"
        )
    completed = []
    for case in cases:
        result = create_controlled_corruption(
            source.images[target_view],
            source.foreground_masks[target_view],
            case=case,
            seed=seed,
            fraction=fraction,
            max_displacement=max_displacement,
        )
        case_dir = output_dir / case
        write_corruption_bundle(
            source,
            result,
            target_view=target_view,
            output_dir=case_dir,
        )
        completed.append(case_dir / "case.json")
    return tuple(completed)
```

`main` splits `--cases` on commas, calls `generate_cases`, and prints every returned `case.json` path.

- [ ] **Step 6: Run artifact tests and CLI help**

Run:

```bash
pytest tests/multiview/test_correspondence_artifacts.py -v
python scripts/generate_correspondence_cases.py --help
```

Expected: tests PASS and CLI exits `0`.

- [ ] **Step 7: Commit bundle tooling**

```bash
git add pixal3d/experiments/correspondence/artifacts.py scripts/generate_correspondence_cases.py tests/multiview/test_correspondence_artifacts.py
git commit -m "feat: package controlled correspondence cases"
```

### Task 5: Feature preservation metrics and sparse-volume visualizations

**Files:**

- Create: `pixal3d/experiments/correspondence/metrics.py`
- Create: `pixal3d/experiments/correspondence/visualization.py`
- Create: `tests/multiview/test_correspondence_metrics.py`

**Interfaces:**

- Consumes: clean-reference and candidate sparse features `[N,C]`, per-view weights `[K,N]`, projected oracle corruption `[K,N]`, sparse coords `[N,4]`, and grid resolution.
- Produces: JSON-safe scalar metrics and RGB slice panels.

- [ ] **Step 1: Write failing feature metric tests**

```python
import math

import torch

from pixal3d.experiments.correspondence.metrics import (
    compute_feature_metrics,
)


def test_feature_metrics_are_zero_for_clean_reference():
    reference = torch.tensor([[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0]])
    weights = torch.full((4, 2), 0.25)
    projected = torch.zeros(4, 2)
    metrics = compute_feature_metrics(
        reference,
        reference,
        weights=weights,
        projected_corruption=projected,
        corrupted_view=1,
    )
    assert metrics["fused_cosine_error"] == pytest.approx(0.0)
    assert metrics["fused_l2_drift"] == pytest.approx(0.0)
    assert metrics["uniform_weight_deviation"] == pytest.approx(0.0)
    assert metrics["weight_entropy"] == pytest.approx(math.log(4.0))


def test_corrupted_region_weight_mass_uses_only_projected_oracle_support():
    reference = torch.ones(2, 4)
    candidate = reference.clone()
    weights = torch.tensor(
        [[0.25, 0.25], [0.05, 0.4], [0.35, 0.175], [0.35, 0.175]]
    )
    projected = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    )
    metrics = compute_feature_metrics(
        reference,
        candidate,
        weights=weights,
        projected_corruption=projected,
        corrupted_view=1,
    )
    assert metrics["corrupted_view_weight_inside_oracle"] == pytest.approx(0.05)
    assert metrics["corrupted_view_weight_outside_oracle"] == pytest.approx(0.4)
```

- [ ] **Step 2: Implement metrics with separate L/H norms and finite JSON scalars**

```python
def compute_feature_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    weights: torch.Tensor,
    projected_corruption: torch.Tensor,
    corrupted_view: int,
) -> dict[str, float]:
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("reference and candidate must share shape [N, C]")
    if reference.shape[1] % 2:
        raise ValueError("feature width must split evenly into L/H")
    cosine = torch.nn.functional.cosine_similarity(
        reference.float(), candidate.float(), dim=-1, eps=1e-6
    )
    l2 = torch.linalg.vector_norm(
        candidate.float() - reference.float(), dim=-1
    )
    split = reference.shape[1] // 2
    oracle = projected_corruption[corrupted_view] >= 0.5
    corrupted_weights = weights[corrupted_view].float()

    def selected_mean(values, selected):
        return values[selected].mean() if torch.any(selected) else torch.tensor(
            float("nan"), device=values.device
        )

    entropy = -(weights.float().clamp_min(1e-12).log() * weights.float()).sum(dim=0)
    uniform = 1.0 / weights.shape[0]
    result = {
        "fused_cosine_error": (1.0 - cosine).mean().item(),
        "fused_l2_drift": l2.mean().item(),
        "reference_low_norm": reference[:, :split].float().norm(dim=-1).mean().item(),
        "candidate_low_norm": candidate[:, :split].float().norm(dim=-1).mean().item(),
        "reference_high_norm": reference[:, split:].float().norm(dim=-1).mean().item(),
        "candidate_high_norm": candidate[:, split:].float().norm(dim=-1).mean().item(),
        "corrupted_view_weight_inside_oracle": selected_mean(
            corrupted_weights, oracle
        ).item(),
        "corrupted_view_weight_outside_oracle": selected_mean(
            corrupted_weights, ~oracle
        ).item(),
        "weight_entropy": entropy.mean().item(),
        "uniform_weight_deviation": (
            weights.float() - uniform
        ).abs().mean().item(),
    }
    return result
```

The JSON writer converts non-finite region metrics to `None`; it does not replace them with zero.

- [ ] **Step 3: Write failing sparse-slice visualization tests**

```python
from PIL import Image

from pixal3d.experiments.correspondence.visualization import (
    render_projected_overlay,
    render_sparse_scalar_slices,
    render_view_weight_panel,
    render_weight_histogram,
)


def test_sparse_scalar_slices_return_rgb_image_without_changing_input():
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2]], dtype=torch.int32
    )
    values = torch.tensor([0.0, 0.5, 1.0])
    original = values.clone()
    image = render_sparse_scalar_slices(values, coords, grid_resolution=3)
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.width == 9
    assert image.height == 3
    assert torch.equal(values, original)


def test_view_weight_panel_contains_one_row_per_view():
    coords = torch.tensor([[0, 1, 1, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
    panel = render_view_weight_panel(weights, coords, grid_resolution=3)
    assert panel.mode == "RGB"
    assert panel.height == 12


def test_projected_overlay_and_histogram_preserve_requested_canvas():
    source = Image.new("RGB", (16, 12), "white")
    pixel_xy = torch.tensor([[2.0, 3.0], [10.0, 8.0]])
    values = torch.tensor([0.0, 1.0])
    overlay = render_projected_overlay(source, pixel_xy, values)
    histogram = render_weight_histogram(
        torch.tensor([[0.1, 0.2], [0.8, 0.9]])
    )
    assert overlay.size == source.size
    assert overlay.mode == "RGB"
    assert histogram.size == (256, 128)
    assert histogram.mode == "RGB"
```

- [ ] **Step 4: Implement deterministic OpenCV heatmaps of center XY/XZ/YZ slices**

```python
from PIL import Image, ImageDraw


def _dense_scalar(
    values: torch.Tensor,
    coords: torch.Tensor,
    grid_resolution: int,
) -> np.ndarray:
    dense = np.full(
        (grid_resolution, grid_resolution, grid_resolution),
        np.nan,
        dtype=np.float32,
    )
    xyz = coords[:, 1:].detach().cpu().long().numpy()
    dense[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = (
        values.detach().cpu().float().numpy()
    )
    return dense


def _heatmap(slice_values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(slice_values)
    normalized = np.zeros_like(slice_values, dtype=np.uint8)
    if finite.any():
        low, high = np.nanpercentile(slice_values, [1, 99])
        scale = max(float(high - low), 1e-6)
        normalized[finite] = np.clip(
            (slice_values[finite] - low) / scale * 255, 0, 255
        ).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
    colored[~finite] = 0
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def render_sparse_scalar_slices(values, coords, grid_resolution):
    dense = _dense_scalar(values, coords, grid_resolution)
    center = grid_resolution // 2
    slices = (
        dense[:, :, center],
        dense[:, center, :],
        dense[center, :, :],
    )
    return Image.fromarray(
        np.concatenate([_heatmap(value) for value in slices], axis=1),
        mode="RGB",
    )


def render_view_weight_panel(weights, coords, grid_resolution):
    rows = [
        np.asarray(
            render_sparse_scalar_slices(weight, coords, grid_resolution)
        )
        for weight in weights
    ]
    return Image.fromarray(np.concatenate(rows, axis=0), mode="RGB")


def render_projected_overlay(image, pixel_xy, values):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    points = pixel_xy.detach().cpu().float().numpy()
    scalar = values.detach().cpu().float().numpy()
    low, high = np.quantile(scalar, [0.01, 0.99])
    normalized = np.clip(
        (scalar - low) / max(float(high - low), 1e-6), 0, 1
    )
    colors = cv2.applyColorMap(
        np.rint(normalized * 255).astype(np.uint8)[:, None],
        cv2.COLORMAP_VIRIDIS,
    )[:, 0, ::-1]
    for (x, y), color in zip(points, colors):
        draw.ellipse(
            (float(x) - 2, float(y) - 2, float(x) + 2, float(y) + 2),
            fill=tuple(int(channel) for channel in color),
        )
    return canvas


def render_weight_histogram(weights, *, width=256, height=128):
    values = weights.detach().cpu().float().numpy().reshape(-1)
    counts, _ = np.histogram(values, bins=32, range=(0.0, 1.0))
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    maximum = max(int(counts.max()), 1)
    bin_width = width / len(counts)
    for index, count in enumerate(counts):
        left = int(round(index * bin_width))
        right = int(round((index + 1) * bin_width)) - 1
        top = height - int(round(count / maximum * (height - 1)))
        cv2.rectangle(canvas, (left, top), (right, height - 1), (40, 90, 220), -1)
    return Image.fromarray(canvas, mode="RGB")
```

- [ ] **Step 5: Run metric and visualization tests**

Run:

```bash
pytest tests/multiview/test_correspondence_metrics.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit metrics and visualizations**

```bash
git add pixal3d/experiments/correspondence/metrics.py pixal3d/experiments/correspondence/visualization.py tests/multiview/test_correspondence_metrics.py
git commit -m "feat: add correspondence feature diagnostics"
```

### Task 6: Image-conditioner-only Gate A runner

**Files:**

- Create: `scripts/run_correspondence_feature_gate.py`
- Create: `tests/multiview/test_correspondence_feature_gate.py`
- Modify: `pixal3d/experiments/correspondence/artifacts.py`

**Interfaces:**

- Consumes: clean calibrated manifest, controlled case bundle, safe sparse coordinate tensor, Pixal3D model path, stage, alpha, temperature, and output directory.
- Produces: S0/S1/S3 sparse conditions, feature metrics, diagnostics tensors saved on CPU, weight/error slice panels, and a reproducibility `run.json`; no flow model is sampled.

- [ ] **Step 1: Write failing parser and coordinate validation tests**

```python
from pathlib import Path

import pytest
import torch

from scripts.run_correspondence_feature_gate import (
    build_parser,
    load_sparse_coords,
)


def test_feature_gate_parser_defaults_to_gate_a_settings():
    args = build_parser().parse_args(
        [
            "--clean-transforms",
            "clean/transforms.json",
            "--case-dir",
            "cases/pattern",
            "--coords",
            "coords.pt",
            "--model-path",
            "model",
            "--stage",
            "shape512",
            "--output-dir",
            "results",
        ]
    )
    assert args.seed == 42
    assert args.alpha == 0.5
    assert args.temperature == 0.1
    assert args.stage == "shape512"


@pytest.mark.parametrize(
    "coords",
    [
        torch.tensor([[1, 0, 0, 0]]),
        torch.tensor([[0, -1, 0, 0]]),
        torch.tensor([[0, 32, 0, 0]]),
        torch.zeros(2, 3),
    ],
)
def test_load_sparse_coords_rejects_invalid_shape_batch_or_bounds(
    tmp_path, coords
):
    path = tmp_path / "coords.pt"
    torch.save(coords, path)
    with pytest.raises(ValueError, match="coords"):
        load_sparse_coords(path, grid_resolution=32)
```

- [ ] **Step 2: Implement safe coordinate loading and exact CLI**

```python
STAGES = {
    "shape512": ("image_cond_model_shape_512", 32),
    "shape1024": ("image_cond_model_shape_1024", 64),
    "pbr1024": ("image_cond_model_tex_1024", 64),
}


def load_sparse_coords(path: Path, *, grid_resolution: int) -> torch.Tensor:
    coords = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(coords, torch.Tensor):
        raise ValueError("coords file must contain a tensor")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("coords must have shape [N, 4]")
    coords = coords.to(torch.int32)
    if torch.any(coords[:, 0] != 0):
        raise ValueError("coords must use batch index zero")
    if torch.any(coords[:, 1:] < 0) or torch.any(
        coords[:, 1:] >= grid_resolution
    ):
        raise ValueError("coords must lie inside the stage projection grid")
    return coords


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-transforms", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--coords", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-vram", action="store_true")
    return parser
```

- [ ] **Step 3: Write a failing mocked S0/S1/S3 runner test**

Define these complete test helpers before the test:

```python
class RecordingFeaturePipeline:
    def __init__(self):
        self._device = "cpu"
        self.image_cond_model_shape_512 = object()
        self.image_cond_model_shape_1024 = object()
        self.image_cond_model_tex_1024 = object()
        self.calls = []

    @property
    def device(self):
        return self._device

    def preprocess_image(self, image):
        return image.convert("RGB")

    def get_proj_cond_shape(
        self,
        conditioner,
        images,
        coords,
        *,
        aggregation_config,
        oracle_masks,
        diagnostics,
        **camera,
    ):
        mode = None if aggregation_config is None else aggregation_config.mode
        call_index = len(self.calls)
        self.calls.append({"mode": mode})
        values = (0.0, 0.0, 2.0, 1.0, 0.2)
        features = torch.full((coords.shape[0], 4), values[call_index])
        if diagnostics is not None:
            num_views = len(images)
            weights = torch.full(
                (num_views, coords.shape[0]), 1.0 / num_views
            )
            if call_index == 3:
                weights[1] = 0.1
                weights[[0, 2, 3]] = 0.3
            elif call_index == 4:
                weights[1] = 0.0
                weights[[0, 2, 3]] = 1.0 / 3.0
            projected = torch.zeros(num_views, coords.shape[0])
            projected[1] = 1.0
            pixel_xy = torch.stack(
                (
                    torch.linspace(1, 6, coords.shape[0]),
                    torch.linspace(1, 6, coords.shape[0]),
                ),
                dim=-1,
            ).expand(num_views, -1, -1)
            diagnostics.update(
                {
                    "weights": weights,
                    "scores": torch.zeros_like(weights),
                    "projected_corruption": projected,
                    "pixel_xy": pixel_xy,
                    "coords": coords.cpu(),
                }
            )
        projected = type("Projected", (), {"feats": features})()
        return {
            "cond": {"global": torch.zeros(1, 5, 4), "proj": projected}
        }


def write_feature_gate_fixture(tmp_path):
    clean_dir = tmp_path / "clean"
    case_dir = tmp_path / "case"
    (clean_dir / "views").mkdir(parents=True)
    (case_dir / "views").mkdir(parents=True)
    frames = []
    for index in range(4):
        name = f"{index:03d}.png"
        clean = Image.new("RGBA", (8, 8), (30 * index, 80, 120, 255))
        clean.save(clean_dir / "views" / name)
        corrupted = clean.copy()
        if index == 1:
            corrupted.paste((240, 20, 20, 255), (2, 2, 6, 6))
        corrupted.save(case_dir / "views" / name)
        transform = torch.eye(4)
        transform[2, 3] = 2.0 + index * 0.1
        frames.append(
            {
                "file_path": f"views/{name}",
                "camera_angle_x": 0.7,
                "transform_matrix": transform.tolist(),
            }
        )
    manifest = {"frames": frames, "mesh_scale": 1.0}
    clean_transforms = clean_dir / "transforms.json"
    clean_transforms.write_text(json.dumps(manifest))
    (case_dir / "transforms.json").write_text(json.dumps(manifest))
    mask = Image.new("L", (8, 8), 0)
    mask.paste(255, (2, 2, 6, 6))
    mask.save(case_dir / "oracle_mask.png")
    (case_dir / "case.json").write_text(
        json.dumps({"case": "pattern", "target_view": 1})
    )
    coords_path = tmp_path / "coords.pt"
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2]],
        dtype=torch.int32,
    )
    torch.save(coords, coords_path)
    return {
        "clean_transforms": clean_transforms,
        "case_dir": case_dir,
        "coords_path": coords_path,
        "model_path": str(tmp_path / "model"),
        "output_dir": tmp_path / "result",
    }
```

```python
def test_feature_gate_runs_clean_reference_s0_s1_s3(
    tmp_path, monkeypatch
):
    pipeline = RecordingFeaturePipeline()
    monkeypatch.setattr(
        feature_gate,
        "init_pipeline",
        lambda *args, **kwargs: pipeline,
    )
    inputs = write_feature_gate_fixture(tmp_path)

    result = feature_gate.run_feature_gate(
        **inputs,
        stage="shape512",
        alpha=0.5,
        temperature=0.1,
        seed=42,
        low_vram=False,
    )

    assert [call["mode"] for call in pipeline.calls] == [
        None,
        "consensus",
        None,
        "consensus",
        "oracle",
    ]
    assert set(result["metrics"]) == {"S0", "S1", "S3"}
    assert (inputs["output_dir"] / "run.json").is_file()
    assert (inputs["output_dir"] / "weights_S1.png").is_file()
    assert (inputs["output_dir"] / "weights_S3.png").is_file()
    assert (inputs["output_dir"] / "weights_clean_S1.png").is_file()
    assert (inputs["output_dir"] / "weight_histogram_S1.png").is_file()
    assert (inputs["output_dir"] / "projection_overlay_S1.png").is_file()
    assert (inputs["output_dir"] / "feature_error.png").is_file()
```

- [ ] **Step 4: Implement the five conditioner calls without sampling flow**

```python
def _encode(
    pipeline,
    conditioner,
    images,
    cameras,
    coords,
    *,
    config,
    oracle_masks,
    diagnostics,
):
    return pipeline.get_proj_cond_shape(
        conditioner,
        images,
        coords.to(pipeline.device),
        camera_angle_x=cameras["camera_angle_x"],
        distance=cameras["distance"],
        mesh_scale=cameras["mesh_scale"],
        transform_matrix=cameras["transform_matrix"],
        aggregation_config=config,
        oracle_masks=oracle_masks,
        diagnostics=diagnostics,
    )


clean_reference = _encode(
    pipeline,
    conditioner,
    clean_images,
    clean_cameras,
    coords,
    config=None,
    oracle_masks=None,
    diagnostics=None,
)
clean_s1_diagnostics = {}
clean_consensus = _encode(
    pipeline,
    conditioner,
    clean_images,
    clean_cameras,
    coords,
    config=ProjectionAggregationConfig(
        mode="consensus", alpha=alpha, temperature=temperature
    ),
    oracle_masks=oracle_masks,
    diagnostics=clean_s1_diagnostics,
)
corrupted_s0 = _encode(
    pipeline,
    conditioner,
    corrupted_images,
    corrupted_cameras,
    coords,
    config=None,
    oracle_masks=None,
    diagnostics=None,
)
corrupted_s1 = _encode(
    pipeline,
    conditioner,
    corrupted_images,
    corrupted_cameras,
    coords,
    config=ProjectionAggregationConfig(
        mode="consensus", alpha=alpha, temperature=temperature
    ),
    oracle_masks=oracle_masks,
    diagnostics=s1_diagnostics,
)
corrupted_s3 = _encode(
    pipeline,
    conditioner,
    corrupted_images,
    corrupted_cameras,
    coords,
    config=ProjectionAggregationConfig(
        mode="oracle", alpha=alpha, temperature=temperature
    ),
    oracle_masks=oracle_masks,
    diagnostics=s3_diagnostics,
)
```

The runner uses `load_calibrated_manifest` and `normalize_calibrated_views` for both manifests, asserts that every camera tensor is identical, and preprocesses both view lists identically. It creates `oracle_masks` as zeros for all views except `case.json["target_view"]`, where it loads `oracle_mask.png`. It selects the conditioner using `STAGES[stage][0]` and never calls `pipeline.run`, a sampler, or a flow model.

- [ ] **Step 5: Save metrics, diagnostics, visualizations, and reproducibility metadata**

For each arm, call `compute_feature_metrics` against the clean reference. Save:

```text
metrics.json
diagnostics.pt
weights_S1.png
weights_S3.png
weights_clean_S1.png
weight_histogram_S1.png
weight_histogram_S3.png
projection_overlay_S1.png
projection_overlay_S3.png
feature_error.png
run.json
```

Use the S3 projected oracle mask when scoring S0, uniform `[K,N]` weights for S0, and each arm's own diagnostics for S1/S3. Also record clean-consensus feature drift, minimum per-view mean weight, and entropy under the `clean_control` key; this is the explicit guard against suppressing correct but view-unique clean evidence.

Render the weight histograms with `render_weight_histogram`. Render the corrupted target view's projected points with `render_projected_overlay`, coloring each point by that view's aggregation weight. These overlays use `diagnostics["pixel_xy"][target_view]`; validity is displayed separately in saved diagnostics and never filters points.

`run.json` contains:

```python
{
    "code_commit": subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip(),
    "model_path": str(Path(model_path).resolve()),
    "stage": stage,
    "seed": seed,
    "alpha": alpha,
    "temperature": temperature,
    "clean_manifest_sha256": sha256_file(clean_transforms),
    "case_metadata_sha256": sha256_file(case_dir / "case.json"),
    "coords_sha256": sha256_file(coords_path),
    "arms": ["S0", "S1", "S3"],
    "flow_sampled": False,
}
```

Do not write CUDA tensors directly; detach and move diagnostic tensors to CPU before `torch.save`.

- [ ] **Step 6: Run the Gate A tests and CLI help**

Run:

```bash
pytest tests/multiview/test_correspondence_feature_gate.py -v
python scripts/run_correspondence_feature_gate.py --help
```

Expected: tests PASS and CLI exits `0`.

- [ ] **Step 7: Commit the Gate A runner**

```bash
git add scripts/run_correspondence_feature_gate.py tests/multiview/test_correspondence_feature_gate.py pixal3d/experiments/correspondence/artifacts.py
git commit -m "feat: add correspondence feature gate runner"
```

### Task 7: End-to-end CPU fixture and documentation

**Files:**

- Modify: `tests/multiview/test_correspondence_feature_gate.py`
- Modify: `docs/CORR_ADAPTER_EXPERIMENT_DESIGN.md`
- Create: `docs/CORR_ADAPTER_GATE_A_RUNBOOK.md`

**Interfaces:**

- Consumes: completed corruption and feature-gate tooling.
- Produces: one tiny deterministic CPU bundle/diagnostic fixture and an exact runbook for real four-view data.

- [ ] **Step 1: Add an end-to-end tiny fixture test**

The test creates four 16×16 calibrated RGBA images, generates a `pattern` corruption for view `1`, writes the bundle, runs the mocked feature gate on eight coordinates, and asserts:

```python
def test_tiny_controlled_case_to_feature_report(tmp_path, monkeypatch):
    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    frames = []
    for index in range(4):
        name = f"{index:03d}.png"
        Image.new(
            "RGBA", (16, 16), (40 * index, 80, 120, 255)
        ).save(clean_root / name)
        transform = torch.eye(4)
        transform[2, 3] = 2.0 + 0.1 * index
        frames.append(
            {
                "file_path": name,
                "camera_angle_x": 0.7,
                "transform_matrix": transform.tolist(),
            }
        )
    clean_manifest = clean_root / "transforms.json"
    clean_manifest.write_text(
        json.dumps({"frames": frames, "mesh_scale": 1.0})
    )
    generated_root = tmp_path / "cases"
    generate_cases(
        transforms=clean_manifest,
        output_dir=generated_root,
        cases=("pattern",),
        target_view=1,
        seed=42,
        fraction=0.3,
        max_displacement=4.0,
        allow_anchor_corruption=False,
    )
    case_dir = generated_root / "pattern"
    coords_path = tmp_path / "coords.pt"
    torch.save(
        torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 1],
                [0, 2, 2, 2],
                [0, 3, 3, 3],
                [0, 4, 4, 4],
                [0, 5, 5, 5],
                [0, 6, 6, 6],
                [0, 7, 7, 7],
            ],
            dtype=torch.int32,
        ),
        coords_path,
    )
    pipeline = RecordingFeaturePipeline()
    monkeypatch.setattr(
        feature_gate,
        "init_pipeline",
        lambda *args, **kwargs: pipeline,
    )
    result = feature_gate.run_feature_gate(
        clean_transforms=clean_manifest,
        case_dir=case_dir,
        coords_path=coords_path,
        model_path=str(tmp_path / "model"),
        output_dir=tmp_path / "result",
        stage="shape512",
        alpha=0.5,
        temperature=0.1,
        seed=42,
        low_vram=False,
    )
    assert result["metrics"]["S3"]["fused_l2_drift"] <= (
        result["metrics"]["S0"]["fused_l2_drift"]
    )
    assert (tmp_path / "result" / "run.json").is_file()
```

- [ ] **Step 2: Run all correspondence unit and integration tests**

Run:

```bash
pytest \
  tests/multiview/test_projection_aggregation.py \
  tests/multiview/test_correspondence_corruptions.py \
  tests/multiview/test_correspondence_artifacts.py \
  tests/multiview/test_correspondence_metrics.py \
  tests/multiview/test_correspondence_feature_gate.py \
  -v
```

Expected: PASS.

- [ ] **Step 3: Run the full multi-view suite and source checks**

Run:

```bash
pytest tests/multiview -q
git diff --check
python scripts/generate_correspondence_cases.py --help
python scripts/run_correspondence_feature_gate.py --help
```

Expected: all tests PASS subject only to pre-existing environment-dependent skips; both CLIs exit `0`; no whitespace errors.

- [ ] **Step 4: Write the Gate A runbook**

Document exact commands:

```bash
python scripts/generate_correspondence_cases.py \
  --transforms /abs/path/to/clean/transforms.json \
  --output-dir /abs/path/to/cases/object_id \
  --target-view 1 \
  --seed 42

python scripts/run_correspondence_feature_gate.py \
  --clean-transforms /abs/path/to/clean/transforms.json \
  --case-dir /abs/path/to/cases/object_id/pattern \
  --coords /abs/path/to/shape512_coords.pt \
  --model-path /abs/path/to/pixal3d_model \
  --stage shape512 \
  --output-dir /abs/path/to/results/object_id/pattern/shape512 \
  --alpha 0.5 \
  --temperature 0.1 \
  --seed 42
```

Also document:

- required RGBA foreground alpha or explicit foreground-mask extension;
- target view and anchor conventions;
- coordinate tensor shape and bounds;
- S0/S1/S3 definitions;
- how to inspect `metrics.json`, weight panels, and feature error;
- why no 3D conclusion can be drawn from Gate A alone; and
- the Stage 2 gate from the approved design.

- [ ] **Step 5: Synchronize the design status**

Update `docs/CORR_ADAPTER_EXPERIMENT_DESIGN.md` with implemented Gate A paths, test counts, and the remaining blockers: pending multi-view Shape-512/Shape-1024/PBR-1024 checkpoint identities and fixed upstream latents for Gate B.

- [ ] **Step 6: Commit runbook and evidence**

```bash
git add docs/CORR_ADAPTER_GATE_A_RUNBOOK.md docs/CORR_ADAPTER_EXPERIMENT_DESIGN.md tests/multiview/test_correspondence_feature_gate.py
git commit -m "docs: add correspondence gate A runbook"
```

- [ ] **Step 7: Verify final clean state**

Run:

```bash
git status --short --branch
git log --oneline --decorate -12
```

Expected: clean worktree on `feature/multiview-correspondence-node11`; no checkpoint-dependent experiment is represented as completed.
