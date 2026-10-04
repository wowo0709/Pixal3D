# Data Toolkit Guide

## Scope

`data_toolkit/` prepares source assets, rendered conditions, view-aligned voxel data, and VAE latents used by Pixal3D training. Follow the staged commands and dataset-specific arguments in [README.md](README.md); this guide records the data contracts between stages.

## Dataset Modules

- `datasets/ObjaverseXL.py` supports `sketchfab` and `github` sources; GitHub archive processing extracts the matching member into a temporary directory before invoking the stage processor.
- `datasets/ABO.py` downloads/extracts its archive and checks extracted file hashes against metadata.
- `datasets/TexVerse.py` expects local GLB files and metadata prepared by `build_texverse_metadata.py`; its generic metadata and download functions raise `NotImplementedError`.
- Modules also exist for HSSD, 3D-FUTURE and Toys4k. Source adapters and the pipeline registry own their names and download contracts.

## Metadata And Outputs

- `build_metadata.py` imports `datasets.<SUBSET>` dynamically. The module supplies `get_metadata`, `download`, and `foreach_instance` behaviors used by the processing scripts.
- Root `metadata.csv` is the input registry. Stage outputs live under their own directories, generally keyed by `sha256`; workers append `new_records/part_<rank>.csv` metadata, then `build_metadata.py` aggregates it into stage metadata and the root registry.
- Aggregation discovers resolution-specific voxel/grid directories and model-specific latent directories. Names ending `_view` or `_view_fix` are classified as multi-view latent variants.
- `--from_file` reconstructs raw-download metadata by scanning local files and matching identifiers to the root registry. Standard aggregation reads `new_records/part_*.csv` and moves processed records under timestamped names in `merged_records/`.
- `--from_merged_records` rebuilds metadata from files in `merged_records/`; both discovery and reads use that directory.

## View-Aligned Data Flow

- Render conditions provide `renders_cond/<sha256>/transforms.json`; its `frames` carry the camera transforms consumed by the view voxelizers.
- `dual_grid_view.py` writes per-view `.vxz` grids beneath `dual_grid_view_<resolution>/<sha256>/`; shape view encoding consumes these grids. Sparse-structure view encoding consumes shape latents.
- `voxelize_pbr_view.py` writes PBR voxel outputs under `pbr_voxels_view_fix_<resolution>/<sha256>/`.
- View outputs use `viewNN` filenames and include `viewNN_scale.json`. The shape latent encoder copies the corresponding scale JSON alongside each latent. Keep view indices aligned between rendered conditions, voxelization, and encoding.
- Explicit view selections accept comma-separated indices and inclusive ranges, such as `0,1` and `0-5`. When reproducibility across stages matters, pass the same explicit selection; defaults vary by script.
- The later view latent stages consume artifacts from prior stages: shape latents consume dual-grid views, and sparse-structure latents consume shape view latents. Refresh metadata between stages as documented in `README.md`.
- `encode_ss_latent_view.py` appends `_view` to `--shape_latent_name` internally. Pass the unsuffixed model name; the README example includes `_view` and would resolve a doubled suffix.

## Sharding And Resume

- Processing scripts partition the filtered metadata list into contiguous slices using `rank` and `world_size`; launch all intended ranks with a shared dataset root.
- Many stages use metadata flags and/or existing output files to skip completed work. Metadata is the completion index, so aggregate fresh `new_records` before relying on it for a subsequent stage.
- Dataset `foreach_instance` implementations catch per-object failures, print an error, and may omit failed records from the returned metadata; inspect stage output summaries and records when a run is incomplete.

## Setup

`setup.sh` installs the toolkit's Python packages with `pip`; it does not activate or create an environment. The root README documents the separate TRELLIS.2 environment and additional runtime dependencies used by model-backed encoding.

## MV Branch Pipeline

- `pipeline/` owns preparation orchestration, manifests, eligibility, packing and
  source-specific training materialization; its APIs differ from raw stage CLIs.
  Read the local README and matching tests under `tests/data_toolkit/` before edits.
- `experiments/multiview_finetune/prepare.py` consumes already completed batch
  packs. It streams each NFS tar into local staging, verifies member sizes/hashes,
  publishes completed bundles and derives the common four-stage eligible metadata.
  It does not regenerate renders/latents or copy the tar archives locally.
- Completion and eligibility audit files are dataset artifacts. Preserve original
  archives and their manifests; generated metadata must not invent missing scores.
