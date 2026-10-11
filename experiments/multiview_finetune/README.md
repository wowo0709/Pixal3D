# Mixed-source multi-view fine-tuning

This experiment fine-tunes official single-view Pixal3D denoisers with calibrated
multi-view conditioning using the implementation derived from `3ff6fca` on
`feature/multiview-model-extension`. It is a local adaptation, not an official
multi-view pretrained release or a reproduction of paper metrics. The original
`configs/gen/*multiview*.json` configs remain the native model/dataset/trainer
sources. The four small JSON files in `configs/gen/multiview_finetune/` describe
recipe derivation; run `configure.py` to produce runnable native configs outside
the source config directory.

| Recipe | Native source resolution → trained resolution | Official SV initialization | EBS | Nominal assets per pack |
|---|---|---|---|---|
| SS64 | SS32 → SS64 | `ss_flow_img_dit_1_3B_64_bf16` | 96 | 3 |
| Shape512 | Shape256 → Shape512 | `slat_flow_img2shape_dit_1_3B_512_bf16` | 96 | 2 |
| Shape1024 | Shape512 → Shape1024 | `slat_flow_img2shape_dit_1_3B_1024_bf16` | 24 | 1 |
| PBR1024 | PBR512 → PBR1024 | `slat_flow_imgshape2tex_dit_1_3B_1024_bf16` | 24 | 1 |

PBR1024 additionally requires aligned Shape1024 latents. All stages use frozen
DINO/NAF and native frozen decoder dependencies. The recipe halves the original
LR, uses 5% linear warmup followed by a constant LR, samples K uniformly from
1 through 6, and trains all denoiser parameters. It preserves native AdamW,
mean-over-microbatches loss, BF16 AMP, FP32 optimizer state, adaptive gradient
clipping, and EMA 0.9999. `batch_size_per_gpu` is the full loader batch;
`batch_split` is the number of accumulation packs (32/48/24/24 by default).
Sparse packs balance token loads and can contain unequal numbers of assets.
Five epochs means `5 * floor(actual_common_eligible_training_assets / EBS)`
optimizer updates, with a save at each epoch boundary and five checkpoint sets.
The recorded full-corpus preparation had 21,314 eligible training assets; this
count is evidence from that dataset, not a constant used to resolve new configs.

## External setup

Run from this checkout, using a Python environment installed according to the
[root environment instructions](../../README.md). Training and sampling require
CUDA PyTorch, the native attention/sparse kernels, `flex_gemm`, `o_voxel`, the
rendering/export dependencies, and DINO/NAF dependencies. CPU tests use the same
environment with `CUDA_VISIBLE_DEVICES=''`; their `conftest.py` patches the
`flex_gemm` import-time hardware query without initializing CUDA. Data, converted
weights, resolved JSONs, Python environments, and results belong in ignored
`data/pixal3d` / `results` directories or external storage.

Set paths and select a GPU only after obtaining its assignment:

```bash
export PYTHON=/path/to/runtime/bin/python
export DATA_ROOT=/path/to/prepared/mixed100
export WEIGHTS_ROOT=/path/to/converted/single_view
export DINO_MODEL=/path/to/dino/snapshot  # or camenduru/dinov3-vitl16-pretrain-lvd1689m
export OUTPUT_ROOT=/path/to/results/multiview
export CONFIG_DIR="$OUTPUT_ROOT/configs"
export DOWNLOAD_REPORT=/path/to/checkpoints/download_report.json
export ASSIGNED_GPU=GPU-your-assigned-uuid  # a physical GPU index also works
mkdir -p "$OUTPUT_ROOT"
```

The default data root is this checkout's `data/pixal3d/mixed100`, weights root is
`data/pixal3d/pretrained/single_view`, DINO is the Hub identifier above, and output
root is `results/multiview`. No machine-specific GPU assignment is stored in the
code. Single-GPU tools require exactly one explicit `CUDA_VISIBLE_DEVICES` entry,
including CUDA/MIG UUIDs, and address it internally as `cuda:0`. The full EBS
recipe assumes one training worker. The training wrapper enforces one assigned
visible device before runtime imports, rejects `--num_gpus` values other than 1,
and supplies `--num_gpus 1` when omitted. Use `--auto_retry 0` for bounded runs.
Resolved stage configs default to `<output-root>/<stage>`, matching the sample
and checkpoint-verifier result layout.

### Weights

Download official snapshots into storage you control with `huggingface_hub`, or
reuse complete local snapshots. The original experiment used these revisions:

- `TencentARC/Pixal3D`: `b0cb2e1b794cab9aa0ac38a95d794a4d9337437f`
- `camenduru/dinov3-vitl16-pretrain-lvd1689m`: `3c276edd87d6f6e569ff0c4400e086807d0f3881`
- `microsoft/TRELLIS-image-large`: `25e0d31ffbebe4b5a97464dd851910efc3002d96`
- `microsoft/TRELLIS.2-4B`: `af44b45f2e35a493886929c6d786e563ec68364d`

For example, `snapshot_download(repo_id='TencentARC/Pixal3D', revision=..., local_dir='/path/to/official/pixal3d')`
obtains the pipeline JSON and checkpoint architecture JSONs/safetensors. Convert
the four official denoisers with the existing source script, explicitly
overriding its historical output default:

```bash
CUDA_VISIBLE_DEVICES='' "$PYTHON" scripts/materialize_multiview_checkpoints.py \
  --output-root "$WEIGHTS_ROOT"
```

This downloads the official safetensors and creates native `.pt` state dicts,
checking every tensor after conversion. Existing targets are never overwritten.
For pre-downloaded snapshots, its `convert_checkpoint(source, target)` function
can convert each local `ckpts/<stem>.safetensors` to `<weights-root>/<stem>.pt`.
Record the source snapshot revision and file paths in the report below. The
source materializer uses the Hub's default repository revision;
the pinned revision list above documents the original experiment.

Sampling needs `pipeline.json`, the four `ckpts/<stem>.json` architecture files,
and frozen decoder `.json`/`.safetensors` pairs. Create an external
`download_report.json` with `files` entries for each denoiser stem containing
`safetensors` and `pt` paths, and decoder prefixes (without extensions):

```json
{
  "files": {
    "ss_flow_img_dit_1_3B_64_bf16": {
      "safetensors": "/path/to/official/pixal3d/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
      "pt": "/path/to/converted/single_view/ss_flow_img_dit_1_3B_64_bf16.pt"
    },
    "ss_decoder": "/path/to/trellis-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
    "shape_decoder": "/path/to/trellis2/ckpts/shape_dec_next_dc_f16c32_fp16",
    "pbr_decoder": "/path/to/trellis2/ckpts/tex_dec_next_dc_f16c32_fp16"
  }
}
```

Add the other three stem entries for provenance. `sample.py --official-snapshot`
can select the pipeline snapshot explicitly; otherwise it locates the snapshot
from the report's SS safetensors path, without assuming a Hugging Face cache
layout. The report still supplies decoder paths. Training decoder identifiers
remain native Hub identifiers and may populate the caller's configured HF cache.
DINO/NAF remain frozen; `--dino` accepts either a Hub identifier or a local
snapshot (sampling can override the saved DINO path when relocating a run).

### Prepared dataset

#### Extending an existing split

An expanded corpus reuses the original extracted batch directories and fixed
validation split. It is a dataset extension for the same SV-to-MV recipes,
not a new conditioning protocol. Keep its selection and summaries in a separate
external profile directory so historical selections and resolved runs remain
inspectable. `--train-batches` specifies final counts, including existing bundles;
the source proportions may change when adding higher-quality Sketchfab data.

```bash
"$PYTHON" experiments/multiview_finetune/prepare.py select \
  --source /path/to/prepared/archive_root --destination "$DATA_ROOT" \
  --base-selection "$DATA_ROOT/selection.json" \
  --selection "$PROFILE_ROOT/selection.json" \
  --train-batches ObjaverseXL_sketchfab=455 ABO=10 3D-FUTURE=5 HSSD=5
"$PYTHON" experiments/multiview_finetune/prepare.py extract \
  --selection "$PROFILE_ROOT/selection.json" --workers 4 --reserve-gib 100
"$PYTHON" experiments/multiview_finetune/prepare.py metadata \
  --selection "$PROFILE_ROOT/selection.json" --workers 4
```

This example adds 375 Sketchfab train bundles to the original 100-bundle train
corpus. Asset counts are measured after the common eligibility audit, not assumed
from bundle counts. Existing rows retain their split and manifest identities;
new bundles must be disjoint from all existing train and validation asset SHAs.
Tar files are streamed from NFS without local archive copies. Extraction skips
completed matching bundles and checks pending payload plus the requested free-space
reserve before writing new ones. Interrupted `.staging` directories require
inspection; they are not silently reused or overwritten.

After metadata completes, resolve the appropriate recipe with
`configure.py --data-root "$DATA_ROOT" --selection "$PROFILE_ROOT/selection.json"`
and the explicit weight/DINO/result paths described above. Use a new
`--output-config-dir`; the actual common-eligible count determines epoch/update
budgets. The four stage roots share the same batch files; data is not duplicated
per model.

The staging input is the prepared multi-source archive collection, not raw
images or meshes. Required families are `common`, `ss/64`, `shape/512`,
`shape/1024`, and `pbr/1024`; selection also requires the original `shape/256`,
`pbr/256`, and `pbr/512` family bundles. Each tar has a `.tar.manifest.json`
sidecar listing member path, size, SHA256, and included asset SHAs. Selection
chooses 80 Sketchfab, 10 ABO, 5 3D-FUTURE, and 5 HSSD training bundles plus one
held-out bundle per source, with seed 42 and disjoint asset SHAs.

```bash
"$PYTHON" experiments/multiview_finetune/prepare.py select \
  --source /path/to/prepared/archive_root --destination "$DATA_ROOT" \
  --selection "$DATA_ROOT/selection.json"
"$PYTHON" experiments/multiview_finetune/prepare.py extract \
  --selection "$DATA_ROOT/selection.json" --workers 4
"$PYTHON" experiments/multiview_finetune/prepare.py metadata \
  --selection "$DATA_ROOT/selection.json" --workers 4
"$PYTHON" experiments/multiview_finetune/configure.py \
  --data-root "$DATA_ROOT" --weights-root "$WEIGHTS_ROOT" --dino "$DINO_MODEL" \
  --output-root "$OUTPUT_ROOT" --output-config-dir "$CONFIG_DIR"
```

Extraction streams tars directly, verifies every member's size/SHA256, and
publishes only verified batch directories. Metadata uses verified completion,
maximum token counts over both cached anchors (view00/view01), and matching
shape/PBR coordinates. All four stage loaders receive the common eligible asset
intersection. Missing original aesthetic/material metadata is not invented.
`configure.py` reads `selection.json` and each batch's `metadata_report.json`;
it writes four native stage configs, eight train/validation roots mappings, and
`dataset_summary.json` only under `CONFIG_DIR`. Directory mappings stay native
because the older combined-training-manifest API does not support Sketchfab.
Sampling is a shuffled concatenation without source reweighting, so bundle
proportions are not exact asset proportions.

## Verification, probes, and training

```bash
CUDA_VISIBLE_DEVICES='' "$PYTHON" experiments/multiview_finetune/verify_loaders.py \
  --configs-dir "$CONFIG_DIR" --output "$OUTPUT_ROOT/loader_report.json"
CUDA_VISIBLE_DEVICES="$ASSIGNED_GPU" "$PYTHON" experiments/multiview_finetune/verify_conditioner.py \
  --configs-dir "$CONFIG_DIR" --output "$OUTPUT_ROOT/conditioner_report.json"
CUDA_VISIBLE_DEVICES="$ASSIGNED_GPU" "$PYTHON" experiments/multiview_finetune/probe.py shape512 \
  --config "$CONFIG_DIR/shape512.json" --sizes 1,2 --stress --headroom-gib 6 \
  --output "$OUTPUT_ROOT/probe_shape512"
CUDA_VISIBLE_DEVICES="$ASSIGNED_GPU" "$PYTHON" experiments/multiview_finetune/train.py \
  --config "$CONFIG_DIR/shape512.json" --output_dir "$OUTPUT_ROOT/shape512" \
  --num_gpus 1 --auto_retry 0
```

Loader checks use resolved training mappings; conditioner checks use the paired
validation mapping and saved DINO configuration. `--roots-dir` overrides their
mappings if needed. The probe reads the matching resolved stage config and
runs two native optimizer updates per candidate with real high-token data, K=6,
initialized Adam state, and resident full-EBS input images. `--stress` locks
initial Elastic activation retention at its upper bound; the second update
records Flash Attention profiler events. The optional `--headroom-gib` is a
caller-chosen margin (default 0), not a machine capacity assumption. Reassess
packs on the assigned device before choosing a different nominal microbatch.

The experiment training wrapper records actual batch/VRAM/LR timing and adds
raw-weight held-out flow losses after saves. Main configs evaluate all held-out
assets; debug configs can set `validation.assets_per_source` explicitly. K=1/4
losses share noise/t and disable conditioning dropout, then restore RNG and model
mode. EMA weights remain separately available. The wrapper raises its file
limit to 8192 because full sparse EBS packs exhausted the default 1024 limit.

For a bounded debug run, write a **separate external** native config with
`max_steps=2`, `i_save=2`, `num_workers=0`, and
`validation.assets_per_source=4`. Set `PIXAL3D_DEBUG_VIEWS=1,6,4`; then resume with
`max_steps=3`, `i_save=3`, `--load_dir <stage-run> --ckpt 2`. Persistent workers
would retain the old K, which is why these debug runs use workers=0. Native
`run()` saves only on `i_save` boundaries. Production uses workers=2, random K,
and no `PIXAL3D_DEBUG_VIEWS` variable. The debug sequence proves functional
training/resume, not model quality.

## Saved checkpoints and sampling

Store matched runs under `<result-root>/{ss64,shape512,shape1024,pbr1024}`. Each
stage keeps its native `config.json` and `ckpts`. Tools also accept
`--configs-dir` to read explicitly selected `<stage>.json` files instead.

```bash
CUDA_VISIBLE_DEVICES='' "$PYTHON" experiments/multiview_finetune/verify_checkpoints.py \
  --result /path/to/matched_stage_runs --step 3
CUDA_VISIBLE_DEVICES="$ASSIGNED_GPU" "$PYTHON" experiments/multiview_finetune/sample.py \
  --result /path/to/matched_stage_runs --step 3 --download-report "$DOWNLOAD_REPORT" \
  --batch "$DATA_ROOT/ABO/ABO-00000/batch001" --asset <held-out-sha> \
  --views 1 4 --anchor 1 --output-root "$OUTPUT_ROOT/samples"
```

The checkpoint verifier checks finite raw/EMA/FP32 optimizer state, changes from
matching official initialization, and the native scheduler/sampler resume state
at the requested update, including later epochs and completed warmup. The
sampling step defaults to 3 as an explicit debug default; use `--step <update>`
and optionally `--ema 0.9999`. All four stages use the requested checkpoint step.
For matched epochs with different stage update counts, pass explicit
`--stage-steps SS64 SHAPE512 SHAPE1024 PBR1024`, which overrides the shared step
and records the actual per-stage updates and checkpoint paths in the sample
manifest. For the recorded 21,314-asset five-epoch preparation, those final
updates are `1110 1110 4440 4440`. Verify each distinct step separately:

```bash
CUDA_VISIBLE_DEVICES='' "$PYTHON" experiments/multiview_finetune/verify_checkpoints.py \
  --result /path/to/matched_stage_runs --stage ss64 shape512 --step 1110
CUDA_VISIBLE_DEVICES='' "$PYTHON" experiments/multiview_finetune/verify_checkpoints.py \
  --result /path/to/matched_stage_runs --stage shape1024 pbr1024 --step 4440
# Add to the sample command above for these matched final epochs:
# --stage-steps 1110 1110 4440 4440
```

Sampling retains native 1024 cascade math, stage-local BF16 DINO/NAF AMP,
prepared calibrated cameras, black alpha composite, and exact first-anchor
scale; it does not crop calibrated images. `--anchor 1` places that cached frame
first and reorders all camera values identically. `--views 1 4` reuses one loaded
pipeline. Outputs include a GLB, rendered views, latents, and a provenance
manifest. Optional `--memory-budget-gib` and `--min-free-gib` set an explicit
allocator budget and free-memory preflight; they default to unset.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. "$PYTHON" -m pytest \
  tests/multiview/test_mixed_finetune_*.py
```
