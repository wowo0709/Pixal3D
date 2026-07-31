# Controlled Corruption CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build a dataset-agnostic `--transforms` CLI that resolves foreground masks, generates C1–C3 controlled corruptions, writes resumable artifact bundles/contact sheets, and passes a synthetic calibrated end-to-end test.

**Architecture:** Reuse the Pixal3D calibrated manifest contract but keep experiment parsing in `pixal3d.experiments.correspondence`. Resolve masks by explicit manifest path, then meaningful RGBA alpha, then a lazy injected BiRefNet provider. Pure tensor corruption functions create their own oracle changed-region masks; a separate bundle layer hashes inputs and atomically writes images, masks, metadata, and contact sheets.

**Tech Stack:** Python 3.11, PyTorch, torchvision, Pillow, JSON, hashlib, argparse, pytest

## Global Constraints

- No actual Track A directory or dataset is selected or executed.
- CLI input is `--transforms <path>` and output is an explicit `--output-dir <path>`.
- Manifest preserves ordered `frames`, each `file_path`, finite `[4,4]` transform, and frame/top-level `camera_angle_x`.
- Default K is 4; use the first K ordered frames without permutation and record original frame indices.
- `--mesh-scale` is required, finite, and positive; distance is the float64 norm of transform translation.
- Manifest paths, explicit mask paths, and output paths are validated; inputs cannot escape the manifest directory.
- Foreground priority is `foreground_mask_path` → meaningful RGBA alpha → lazy existing Pixal3D BiRefNet/rembg provider.
- Explicit/alpha/rembg masks are saved with source hash and provenance; empty masks fail.
- Foreground masks select corruption/evaluation regions only and never zero Pixal3D features.
- Oracle corruption masks represent pixels actually changed by C1/C2/C3, not the whole foreground.
- Default corruption is non-anchor view index 1; seed 42; view/camera metadata never changes.
- No flow checkpoint, GPU/3D generation, training, mesh/depth, Stage 2 transport, or VLM oracle logic.
- Every production behavior uses RED → GREEN TDD.

---

### Task 1: Calibrated input and foreground-mask provenance

**Files:**
- Create: `pixal3d/experiments/correspondence/inputs.py`
- Modify: `pixal3d/experiments/correspondence/__init__.py`
- Test: `tests/multiview/test_correspondence_inputs.py`

**Interfaces:**
```python
@dataclass(frozen=True)
class CalibratedView:
    frame_index: int
    image_path: Path
    image: Image.Image
    camera_angle_x: float
    distance: float
    transform_matrix: torch.Tensor
    source_sha256: str

@dataclass(frozen=True)
class ForegroundMask:
    mask: torch.Tensor          # bool [H,W]
    provenance: str             # explicit | alpha | rembg
    source_sha256: str

def load_calibrated_views(path: Path, *, mesh_scale: float, num_views: int = 4) -> tuple[list[CalibratedView], float]: ...
def resolve_foreground_mask(view: CalibratedView, frame: dict, *, rembg_provider=None) -> ForegroundMask: ...
```

- [ ] Write failing tests for ordered first-K loading, frame/top-level FOV, finite transforms, required positive mesh scale, translation-norm distance, path containment, and insufficient frames.
- [ ] Run `pytest tests/multiview/test_correspondence_inputs.py -k calibrated -q`; verify RED import failure.
- [ ] Implement the loader without importing `inference.py`; validate JSON directly and preserve frame order.
- [ ] Run calibrated tests GREEN.
- [ ] Write failing mask-priority tests: explicit `foreground_mask_path`; nontrivial RGBA alpha; opaque RGB/RGBA invokes injected rembg exactly once; empty explicit/alpha/rembg fails; source/provenance hashes are stable.
- [ ] Implement mask resolution. Alpha is meaningful when it contains at least one zero/non-opaque pixel and at least one foreground pixel. RemBG is lazy and must be injected in tests; production default wraps `pixal3d.pipelines.rembg.BiRefNet` only when reached.
- [ ] Run focused tests and full `tests/multiview`; commit `feat: add calibrated corruption inputs`.

---

### Task 2: Pure C1–C3 corruption generators

**Files:**
- Create: `pixal3d/experiments/correspondence/corruptions.py`
- Modify: `pixal3d/experiments/correspondence/__init__.py`
- Test: `tests/multiview/test_correspondence_corruptions.py`

**Interfaces:**
```python
@dataclass(frozen=True)
class ControlledCorruption:
    image: torch.Tensor         # float32 [3,H,W], [0,1]
    oracle_mask: torch.Tensor   # bool [H,W], actual changed pixels
    kind: str                   # c1_color | c2_pattern | c3_deletion
    parameters: dict

def sample_foreground_region(foreground: torch.Tensor, *, seed: int, area_fraction: float = 0.12) -> torch.Tensor: ...
def corrupt_local_color(image, foreground, region, *, hue, saturation, brightness) -> ControlledCorruption: ...
def corrupt_procedural_pattern(image, foreground, region, *, seed, pattern="sole") -> ControlledCorruption: ...
def corrupt_local_deletion(image, foreground, region) -> ControlledCorruption: ...
```

- [ ] Write RED tests for deterministic nonempty region selection fully inside foreground and explicit failure for empty/too-small foreground.
- [ ] Implement region selection with a local CPU generator; choose a foreground center and deterministic rectangle, intersect with foreground, never alter global RNG.
- [ ] Write RED tests for C1 actual-change oracle mask, unchanged outside region, same shape/range, and parameter validation. Implement masked torchvision hue/saturation/brightness adjustment; oracle mask is `region & any(abs(out-in)>1/255)`.
- [ ] Write RED tests for deterministic C2 `sole`, `stripes`, and `logo` procedural patterns, foreground containment, and an intentionally semantic-similar high-cosine-style pattern fixture. Implement patterns without external logo assets; alpha composite only within region and derive oracle mask from actual difference.
- [ ] Write RED tests for C3 deletion. Fill from median non-foreground RGB, falling back to border median when no background exists; oracle mask is actual difference only.
- [ ] Run corruption tests, foundation-focused tests, full suite; commit `feat: add controlled c1 c2 c3 corruptions`.

---

### Task 3: Artifact bundles, hashes, manifest, and contact sheets

**Files:**
- Create: `pixal3d/experiments/correspondence/artifacts.py`
- Modify: `pixal3d/experiments/correspondence/__init__.py`
- Test: `tests/multiview/test_correspondence_artifacts.py`

**Bundle contract:**
```text
<output>/<run_id>/
  manifest.json
  views/view_00/source.png
  views/view_00/foreground_mask.png
  corruptions/c1/view_01/image.png
  corruptions/c1/view_01/oracle_mask.png
  contact_sheet.png
```

- [ ] RED-test atomic creation in a temporary output, SHA-256 for source/mask/corrupted/oracle files, relative paths, K/seed/mesh scale/cameras/transforms/view indices, corruption parameters, mask provenance, completed/failed status, and refusal to silently overwrite a hash-mismatched complete run.
- [ ] Implement canonical JSON (`sort_keys=True`, finite numbers only), PNG writing, temp-directory then rename, and resumable hash validation.
- [ ] RED-test a contact sheet containing ordered source, foreground overlay, corrupted image, and oracle overlay for every C1–C3 arm with stable dimensions/labels.
- [ ] Implement Pillow-only contact sheet generation and failure contact-sheet support; masks are visualization overlays only.
- [ ] Run focused/full tests; commit `feat: add corruption artifact bundles`.

---

### Task 4: Dataset-independent CLI and synthetic calibrated E2E

**Files:**
- Create: `scripts/prepare_correspondence_corruptions.py`
- Create: `tests/multiview/test_correspondence_corruption_cli.py`
- Create: `tests/fixtures/correspondence/README.md`

**CLI:**
```text
python scripts/prepare_correspondence_corruptions.py \
  --transforms /path/to/transforms.json \
  --output-dir /path/to/output \
  --mesh-scale 1.0 \
  --num-views 4 \
  --corrupt-view-index 1 \
  --seed 42
```

- [ ] RED-test parser validation: required transforms/output/mesh scale; K default 4; non-anchor default index 1; index range; no flow/GPU/model arguments.
- [ ] Implement CLI orchestration for all C1–C3 arms, mask resolution, bundle/contact sheet, and nonzero exit with failed manifest on empty mask or invalid input.
- [ ] Build a synthetic temporary fixture in the test: four ordered RGBA images, distinct finite camera transforms, frame/top-level FOV coverage, explicit mask on one frame, alpha masks on remaining frames. Do not commit binary fixtures.
- [ ] Run CLI twice by subprocess. Assert first success, second resumable validation, unchanged source/camera order, translation-norm distances, all artifacts/hashes, nonempty foreground/oracle masks, C1–C3 changed-only masks, and contact sheet.
- [ ] Add negative synthetic tests for empty masks, path escape, insufficient K, and hash mismatch.
- [ ] Run focused CLI E2E and full `tests/multiview`; `git diff --check`; commit `feat: add controlled corruption cli`.

---

## Execution boundary

Completion produces code, synthetic tests, and a path-agnostic CLI only. It does not run an arbitrary real dataset. The final report must show the required input layout and CLI example while marking Track A execution and all checkpoint/GPU stages as waiting for user-provided paths/checkpoints.
