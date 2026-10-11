# Pixal3D result viewer

The HTTP viewer compares saved calibrated inputs, textured GLBs, base-color
renders, and normal renders. Serving uses the Python standard library on CPU;
`generate_cases.py` separately runs GPU inference with source from this checkout.

## Artifact layout

Keep source and recipe configs in the checkout. Prepared data and weights belong
under `data/pixal3d/` or an explicit external path; run outputs, logs, selection
manifests, and viewer catalogs belong under `results/` or an external run path.
The tracked `cases.example.json` documents the catalog schema, without real
assets or machine paths. Copy it to your run and replace its example entries.

Each catalog case has a unique `id`, `title`, and a list of results. Each result
has its own `id`, display `label`, artifact `directory`, GLB filename `glb`, and
`training_steps` (zero for untouched pretrained weights). Result counts can vary
between cases. A relative result directory is resolved from the catalog's parent,
not the working directory. Absolute result paths remain valid.

A result directory must contain `sample_manifest.json`, `base_color_views.png`,
`normal_views.png`, and its specified GLB. The manifest supplies the asset ID,
ordered input view indices, calibrated `input_manifest`, seed, resolution,
sampler settings, and checkpoint provenance. Relative `input_manifest` paths
are resolved from the result directory; input frame `file_path` entries are
resolved from that input manifest's parent. Existing absolute provenance paths
are preserved. All results in one case must share an asset ID and be complete.
The viewer reads the inputs and outputs in place; it does not copy artifacts.

Only registered files and local UI assets are served. Metadata is rebuilt on
server startup; artifact bytes are read on each request. Saved renders show the
native mesh, while the GLB shows the remeshed, decimated export with baked
textures. These qualitative comparisons are not held-out evaluation scores.

## Generate comparisons

Run from the repository root in the installed Pixal3D inference environment:

```bash
CUDA_VISIBLE_DEVICES=<assigned-id-or-uuid> python result_viewer/generate_cases.py \
  --selection results/my-run/selection.json \
  --output-dir results/my-run/comparisons \
  --weights-root data/pixal3d/checkpoints \
  --official-root /path/to/TencentARC-Pixal3D-snapshot \
  --finetuned-root results/my-finetune \
  --checkpoint-step 10 --start-index 0 --count 16
```

Exactly one GPU must be explicitly assigned through `CUDA_VISIBLE_DEVICES`;
physical IDs, GPU UUIDs, and MIG UUIDs all use logical `cuda:0`. There is no
resource selection or default memory reservation. An optional
`--memory-budget-gib` limits the PyTorch allocator; other libraries may allocate
separately. Generation requires fresh per-case output directories.
`--finetuned-only` produces only the fine-tuned MV variant.

`--weights-root` defaults to `data/pixal3d/checkpoints` under this checkout and
must contain `download_report.json`. The report has `revisions` and `files`:
four denoiser stems with `safetensors` paths, `ss_decoder`, `shape_decoder`,
`pbr_decoder` paths without extensions, and a `dino` snapshot directory.
Report-relative paths resolve from the weights root; absolute paths are used
as recorded. Supply a report that points to your installed weights.
`--dino-model` overrides its DINO directory. `--official-root` contains
`pipeline.json` and `ckpts/` and overrides the four official flow weight paths.
Without it, the official config snapshot is located by the report's recorded
TencentARC revision in the Hugging Face cache (`HF_HUB_CACHE`, legacy
`HUGGINGFACE_HUB_CACHE`, or `HF_HOME`/`XDG_CACHE_HOME`). This does not search for
or download missing weights.

`--finetuned-root` contains `ss64/`, `shape512/`, `shape1024/`, and `pbr1024/`,
each with the saved native `config.json` and `ckpts/denoiser_stepNNNNNNN.pt`.
The saved config reconstructs the denoiser and any LoRA adapters; checkpoints
contain both the frozen base and adapters. `--checkpoint-step` selects the native
checkpoint number and records that number in output provenance; its default 10
supports earlier ten-update comparisons. Use the actual selected checkpoint
step when registering `training_steps` in the catalog.

The selection JSON supplies `scope` and a `cases` list. Each case contains `id`,
`asset`, `input_manifest`, ordered `view_indices`, and `mesh_scale`, plus any
scale/anchor provenance you want carried into the saved manifest. A relative
input manifest resolves from the selection's parent. Select the input order
before evaluating generated results. `conditioning.py` reorders images, FOVs,
distances, and camera transforms together, composites alpha on black, and keeps
calibrated image dimensions. SV uses the first input and original single-view
conditioning branch; pretrained MV and fine-tuned MV use the ordered views and
camera transforms. Generation fixes seed 42, 12 sampler steps, `1024_cascade`,
and 32768 maximum tokens. The extractor alone uses BF16 autocast, preserving
the sparse decoders' native FP16 execution. Fine-tuned denoisers are loaded
inside inference mode because sampling creates inference-tensor RoPE buffers.

New saved snapshots use the native Y-up anchor frame with front/right/back/left
renders. The o_voxel GLB export applies an axis conversion; the viewer restores
Y-up with a +90° X rotation without rewriting exports. For a non-original
anchor, record whether `mesh_scale` is mesh-derived or estimated; the occupied
latent-cell estimator in `conditioning.py` is approximate.

## Serve and connect

After generation, register completed outputs in your run's catalog and start:

```bash
python result_viewer/server.py \
  --catalog results/my-run/cases.json --host 127.0.0.1 --port 8765
```

The default catalog is `results/result_viewer/cases.json` under the checkout.
Restart the server after editing the catalog. For a remote server, forward the
port through the container alias you use:

```bash
ssh -N -L 8765:127.0.0.1:8765 <container-ssh-alias>
```

Open `http://127.0.0.1:8765`. Drag to orbit, scroll to zoom, Shift-drag to pan.
Camera synchronization copies orientation, radius, target, and field of view.
Reset restores the common camera. Input thumbnails and saved renders open an
image dialog; click the enlarged image for native size, or Escape to close it.

The UI and `@google/model-viewer` 4.0.0 are served locally, with its Apache 2.0
license in `vendor/LICENSE`. GLBs embed their WebP textures, so no CDN or remote
texture requests are needed. Camera APIs follow the
[upstream controls implementation](https://github.com/google/model-viewer/blob/v4.0.0/packages/model-viewer/src/features/controls.ts).

## Verification

From the repository root:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python -m pytest -q result_viewer/test
python result_viewer/generate_cases.py --help
```

For actual GLBs, run the server and use the already installed Playwright and
Chromium runtime; this check does not install packages:

```bash
VIEWER_URL=http://127.0.0.1:8765 \
VIEWER_TEST_OUTPUT=results/my-run/browser-checks \
node result_viewer/test/browser_checks.cjs
```

`NODE_PATH` can expose an existing Playwright installation and
`VIEWER_BROWSER_EXECUTABLE` can select an installed Chromium. The default output
is checkout-relative `temp/result-viewer-browser`. Checks use each catalog
case's actual result count, including two-result debug comparisons, and cover
textures, orbit/zoom/pan synchronization, independent cameras, reset, image
dialogs, render tabs, GLB download, mobile rendering, loading errors, and no
external network requests. `--smoke` checks page loading only.
