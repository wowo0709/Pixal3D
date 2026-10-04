# Pixal3D Library

## Overview

The library combines TRELLIS.2-style dense/sparse models with Pixal3D projection conditioning, mesh/voxel representations, renderers and custom trainers.

## Routing

| Change | Location | Local guidance |
|--------|----------|----------------|
| Denoiser/VAE architecture | `models/` | `models/AGENTS.md` |
| Sample loading and collation | `datasets/` | `datasets/AGENTS.md` |
| Sparse operators and storage | `modules/sparse/` | `modules/sparse/AGENTS.md` |
| Lifecycle/objective/projection features | `trainers/` | `trainers/AGENTS.md` |
| Cascade and guidance sampling | `pipelines/` | `pipelines/AGENTS.md` |
| Dense attention and DiT blocks | `modules/attention/`, `modules/transformer/` | Attention config and projection modules |
| Mesh/PBR/voxel visualization | `renderers/`, `utils/render_utils.py` | `get_renderer` selects by representation |
| Mesh and voxel result contracts | `representations/mesh/base.py`, `representations/voxel/voxel_model.py` | Decoder and export consumers |
| Distributed I/O, elastic execution, batching | `utils/dist_utils.py`, `utils/elastic_utils.py`, `utils/data_utils.py` | Shared with trainers/models |

## Package and Checkpoint APIs

- `models`, `datasets`, `trainers`, and `pipelines` expose lazy `__attributes` registries with module-level `__getattr__`. Config-facing names belong in the relevant registry when adding a class.
- `models.from_pretrained(path)` expects paired `<path>.json` and `<path>.safetensors`; otherwise it resolves a Hugging Face repo plus checkpoint prefix. JSON holds `name` and `args`; weights load with `strict=False`.
- Pipeline loading uses directory/repo `pipeline.json` and its model mapping. Training `.pt` checkpoints follow a different format and are loaded by trainers.
- `pixal3d/__init__.py` imports several package namespaces; import success depends on installed rendering/sparse extensions as well as PyTorch.

## Cross-Subsystem Contracts

- Dense sparse-structure latents and sparse structured latents use different denoisers, trainers and collation. Follow the matching family instead of converting layouts implicitly.
- Multi-view conditions use `[B,K,3,H,W]` images and matching camera arrays; the first view defines the anchor frame. View features are projected into that frame and averaged before the existing denoiser. K=1 uses the single-view computation path.
- Projection features live in `trainers/flow_matching/mixins/image_conditioned_proj.py` and are also used by inference. Changes there affect both consumers.
- Attention/backend selection is split between dense `modules/attention/config.py` and sparse `modules/sparse/config.py`. Dense `naive` attention is not a supported sparse backend.
- `representations/mesh/base.py` defines mesh-with-voxel results consumed by rendering and `o_voxel` export; preserve coordinate frame, voxel grid resolution and attribute layout together.
- `utils/render_utils.py` and `utils/vis_utils.py` are visualization helpers; renderer implementations own CUDA rasterization details. Dataset snapshot rendering is another consumer, so inspect both before changing return keys.

## Verification Boundaries

- Relevant CPU tests live in `tests/multiview/`; actual GPU conditioning and sampling checks additionally require installed backend extensions and matching external assets.
- Config constructors are dynamic; inspect the selected registry entry and actual constructor rather than inferring usage from an import alone.
