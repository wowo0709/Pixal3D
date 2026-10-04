# Project Knowledge Base

Reviewed for the publication branch based on `3ff6fca` of `feature/multiview-model-extension`.

## Overview

Pixal3D generates textured 3D assets with pixel-aligned projection conditioning and a sparse-structure → shape → texture cascade. This checkout contains calibrated multi-view conditioning, the native TRELLIS.2-based training runtime, prepared view-aligned datasets, and our mixed-source fine-tuning tools.

## Structure

```text
Pixal3D_youngwoo/
├── inference.py       # Image, camera estimation, cascade, GLB export
├── app.py             # Gradio/API demo with index.html frontend
├── train.py           # JSON-configured training and multiprocessing
├── configs/gen/       # Original flow configs and portable fine-tuning recipes
├── experiments/multiview_finetune/ # Prepared data, resolved recipes, train/debug/sample
├── result_viewer/      # CPU-only saved-result server and comparison generation
├── pixal3d/           # Models, sparse operators, training and inference
├── data_toolkit/      # Asset processing, Blender renders, latent encoding
└── assets/            # Example images, UI textures, HDR environments
```

## Where to Look

| Task | Location | Guide |
|------|----------|-------|
| Package APIs, rendering, representations, shared helpers | `pixal3d/` | `pixal3d/AGENTS.md` |
| Dense/sparse denoisers and VAEs | `pixal3d/models/` | Local `AGENTS.md` |
| Metadata, view pairing, training collation | `pixal3d/datasets/` | Local `AGENTS.md` |
| Sparse tensor layout and backend dispatch | `pixal3d/modules/sparse/` | Local `AGENTS.md` |
| Resume, optimization, flow objectives, image projection | `pixal3d/trainers/` | Local `AGENTS.md` |
| Camera-aware cascade, samplers, stage offloading | `pixal3d/pipelines/` | Local `AGENTS.md` |
| Dataset preparation and completion records | `data_toolkit/` | Local `AGENTS.md`, `README.md` |
| Environment and stage command examples | `README.md` | Check actual CLI/config for defaults |
| Mixed-source fine-tuning and its external inputs | `experiments/multiview_finetune/` | Local `README.md` and directly related tests |
| Saved-result visualization and generation | `result_viewer/` | Local `README.md` and `test/` |

## Code Map

AST-grep confirmed class definitions and dynamic constructor calls; Python AST supplied symbol inventory. LSP tools were unavailable; resolved reference centrality is unmeasured.

| Symbol | Location | Role |
|--------|----------|------|
| `main` | `train.py` | Resolve dataset/models/trainer; run or profile |
| `find_ckpt` | `train.py` | Choose resume step from `ckpts/misc_*.pt` |
| `run_inference` | `inference.py` | Preprocess, estimate camera, generate and export |
| `Pixal3DImageTo3DPipeline` | `pixal3d/pipelines/pixal3d_image_to_3d.py` | Three-stage projection-conditioned inference |
| `BasicTrainer` | `pixal3d/trainers/basic.py` | Shared optimizer, DDP, checkpoint and logging lifecycle |
| `DinoV3ProjFeatureExtractor` | `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py` | Shared training/inference image conditioning |
| `SparseTensor` | `pixal3d/modules/sparse/basic.py` | Batched sparse features, coordinates and spatial caches |

## Training Configuration

- `train.py` uses `cfg.update(cli)` followed by `cfg.update(config)`: JSON values override same-named CLI values. This is an independent runtime, not the neighboring repository's Kitsu entrypoint.
- Config sections `models`, `dataset`, and `trainer` name package-exported classes and constructor `args`. Maintain those names with the lazy registries.
- `--data_dir` for supplied view configs is a JSON string mapping dataset names to directory dictionaries. Structure needs `ss_latent`, `render_cond`; shape needs `shape_latent`, `render_cond`; texture additionally needs `pbr_latent`. `base` is optional and supplies original aesthetic/material metadata when available.
- The new fine-tuning recipe trains SS64, Shape512, Shape1024 and PBR1024 from matching official single-view weights. Original stage configs remain lineage records. Dataset directories, model arguments and decoder dependencies determine resolution; filenames alone are insufficient.
- Higher-resolution initialization uses `trainer.args.finetune_ckpt`, a model-name → checkpoint-path dictionary. Full resume uses `--load_dir` and `--ckpt`; see the trainer guide for saved state.
- Default `--num_gpus=-1` selects all visible GPUs. Explicitly constrain visible devices and worker count to the assigned resources before a launch.
- `--tryrun` still initializes data, CUDA models and the trainer, and writes run files; it only skips training/profile. `--auto_retry=0` disables the default retry loop.
- W&B is opt-in with `--use_wandb`; actual CLI default project is `pixal3d-training`. `WANDB_DIR` can place W&B files on local storage rather than S3 FUSE.

## Commands

Run from the repository root after preparing the documented environment:

```bash
python inference.py --image assets/images/0_img.png --output ./output.glb
ATTN_BACKEND=sdpa python inference.py --image assets/images/0_img.png --output ./output.glb --low_vram
python app.py --low_vram
python train.py --config <CONFIG_JSON> --output_dir <RUN_DIR> --data_dir '<DATA_DIR_JSON>'
```

## Environment and Verification

- `requirements.txt` supplements the TRELLIS.2 environment; README separately specifies `natten==0.21.0` and a particular `utils3d` wheel. Sparse operators and GLB export require compiled dependencies absent from this tree.
- `requirements-hfdemo.txt` has architecture-specific demo wheels; do not treat it as a portable environment recipe.
- Tests live under `tests/multiview/`, `tests/data_toolkit/` and `result_viewer/test/`. Run the affected CPU checks with `CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python -m pytest ...`; the multiview conftest initializes the sparse import backend without allocating CUDA. GPU checks require an assigned device and matching external weights/data.
- README describes upstream `main` versus `paper` implementations; this publication derives from `feature/multiview-model-extension`. The current code must not be reported as reproducing paper-branch results without checking that implementation.
- Inference defaults to `1536_cascade`, or `1024_cascade` in low-VRAM mode. Create the output parent directory beforehand: inference saves a temporary preprocessed image there before its final directory creation.
- `app.py` launches with `share=True`; starting it can expose a public demo link. `index.html` is the custom frontend, not a separate packaged web application.

## Publication and GPU Policy

- Discuss the intended scope and concrete call/config changes before implementing.
  An explicit request to apply an agreed plan authorizes its routine steps.
- Use only the exact server and physical GPUs assigned by the user. An idle GPU
  is not an assignment. Pass `CUDA_VISIBLE_DEVICES` at launch; single-GPU tools
  use logical `cuda:0`, and DDP process counts must match visible assigned GPUs.
  Never retain a past session's physical GPU allow-list in reusable code.
- Use the project's verified PyTorch environment (the lab default is Conda
  `torch`). Do not change global packages just to run a local experiment.
- Git owns source, portable recipes, documentation and direct regression tests.
  Data, weights, runtime environments, resolved machine configs, real catalogs,
  checkpoints and generated outputs belong under ignored external data/results.
- Current scripts import this checkout. `runtime_bundle/source` is a historical
  run snapshot and must not be a dependency of shared entrypoints.
- Preserve original dataset/camera/latent contracts; avoid path searches, silent
  fallback downloads or generic shape guessing. Let unexpected errors propagate.
- Put disposable checks under `temp/`. Clean only owned files, and never delete
  run artifacts or worktrees until their provenance, merge and clean state have
  been checked. Keep dataset/checkpoint/results storage independent of worktrees.
