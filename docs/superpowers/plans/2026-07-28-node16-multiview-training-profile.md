# Node16 Multi-View Fine-Tuning Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configure all four multi-view flow models for 20,000 steps at six-GPU global batch 48, test per-GPU batch 8 with split 2, select a stable attention backend on Node16, and calculate measured 20k training durations.

**Architecture:** Keep production configuration changes small and test-enforced. Use the existing isolated Node16 training checkout and 64-instance ABO pilot data for bounded six-GPU profiles, while evaluating external FlashAttention in a cloned conda environment so the active `pixal3d` environment and preprocessing workers remain untouched.

**Tech Stack:** Python 3.11, PyTorch 2.8.0+cu128, CUDA 12.8, pytest, JSON configs, six RTX PRO 6000 Blackwell GPUs, SSH, conda.

## Global Constraints

- The four stages are exactly `ss64`, `shape512`, `shape1024`, and `pbr1024`.
- All production configs use `max_steps=20000`, `batch_size_per_gpu=8`, and `batch_split=4`.
- Six GPUs give global batch size `8 * 6 = 48`.
- All production configs keep `i_sample=-1`, `i_save=5000`, and `max_checkpoints=5`.
- Do not change model architecture, dataset behavior, loss, optimizer, learning rate, or condition-view range.
- Node16 outputs use `/file3/youngwoo/pixal3d/ckpts/{stage}`.
- Initial checkpoints use `/file3/youngwoo/pixal3d/train/checkpoints/single_view/{checkpoint}`.
- Never modify `/home/youngwoo/Pixal3D` or active preprocessing processes.
- Profiling uses the separate `/home/youngwoo/Pixal3D-training` checkout and no W&B, snapshots, or checkpoint saves.
- Profile the worst configured condition-view count, `K=6`.
- External FlashAttention is tested only in a cloned environment; PyTorch fused SDPA is the fallback.

---

### Task 1: Enforce the 20k global-batch-48 production policy

**Files:**
- Modify: `tests/multiview/test_configs.py`
- Modify: `configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json`
- Modify: `configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json`
- Modify: `configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json`
- Modify: `configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json`

**Interfaces:**
- Consumes: the existing `CONFIGS`, `CHECKPOINTS`, and `OUTPUT_DIRS` policy tables in `tests/multiview/test_configs.py`.
- Produces: four launch-ready JSON configs with identical step, batch, snapshot, save, retention, and Node16 shared-path policy.

- [ ] **Step 1: Update the config-policy test and verify RED**

Change `OUTPUT_DIRS` to:

```python
OUTPUT_DIRS = {
    stage: f"/file3/youngwoo/pixal3d/ckpts/{stage}"
    for stage in CONFIGS
}
```

Change the trainer assertions to:

```python
assert trainer_args["batch_size_per_gpu"] == 8
assert trainer_args["batch_split"] == 4
assert trainer_args["max_steps"] == 20_000
```

Change the checkpoint assertion to:

```python
assert trainer_args["finetune_ckpt"] == {
    "denoiser": (
        "/file3/youngwoo/pixal3d/train/checkpoints/single_view/"
        + CHECKPOINTS[stage]
    )
}
```

Keep the existing assertions for `i_sample`, `i_save`, `max_checkpoints`,
condition views, model type, and projection attention.

Run:

```bash
conda run --no-capture-output -n pixal3d python -m pytest -q \
  tests/multiview/test_configs.py::test_four_configs_use_batchwide_k_and_matching_checkpoints
```

Expected: FAIL on the old production values.

- [ ] **Step 2: Apply the exact production values**

In every listed config set:

```json
"max_steps": 20000,
"batch_size_per_gpu": 8,
"batch_split": 4
```

Set the output and checkpoint roots exactly as required by the global
constraints. Leave `i_sample=-1`, `i_save=5000`, and `max_checkpoints=5`
unchanged.

- [ ] **Step 3: Verify GREEN and run static checks**

Run:

```bash
conda run --no-capture-output -n pixal3d python -m pytest -q \
  tests/multiview/test_configs.py \
  tests/multiview/test_train_smoke_override.py
git diff --check
```

Expected: all selected tests pass and `git diff --check` exits 0.

- [ ] **Step 4: Commit**

```bash
git add \
  tests/multiview/test_configs.py \
  configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json \
  configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json \
  configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json
git commit -m "config: set multiview 20k global batch 48"
```

---

### Task 2: Evaluate a Blackwell-compatible external FlashAttention backend

**Files:**
- Create: `docs/superpowers/reports/2026-07-28-node16-attention-backend.md`

**Interfaces:**
- Consumes: Node16 environment `/home/youngwoo/miniconda3/envs/pixal3d`, GPU capability `sm_120`, and the repository `ATTN_BACKEND`/`SPARSE_ATTN_BACKEND` switches.
- Produces: an evidence-backed selection between PyTorch fused SDPA and an official external FlashAttention build.

- [ ] **Step 1: Record the immutable baseline**

On Node16 run:

```bash
source /home/youngwoo/miniconda3/etc/profile.d/conda.sh
conda activate pixal3d
python - <<'PY'
import importlib.util
import torch

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("capability", torch.cuda.get_device_capability(0))
print("external_flash_attn", bool(importlib.util.find_spec("flash_attn")))
print("torch_flash_available", torch.backends.cuda.is_flash_attention_available())
print("torch_flash_enabled", torch.backends.cuda.flash_sdp_enabled())
PY
```

Expected baseline: Torch `2.8.0+cu128`, CUDA `12.8`, capability `(12, 0)`,
no external package, and PyTorch fused FlashAttention available and enabled.

- [ ] **Step 2: Resolve build prerequisites without changing the active environment**

Resolve the first CUDA 12.8-capable toolkit from these exact candidates:

```bash
CUDA_ROOT=
for root in /usr/local/cuda-12.8 /usr/local/cuda /opt/cuda
do
  if test -x "$root/bin/nvcc" &&
     "$root/bin/nvcc" --version | grep -q 'release 12.8'
  then
    CUDA_ROOT=$root
    break
  fi
done
printf 'CUDA_ROOT=%s\n' "$CUDA_ROOT"
```

Resolve and record the current official source commit:

```bash
FLASH_SHA=$(
  git ls-remote \
    https://github.com/Dao-AILab/flash-attention.git \
    refs/heads/main |
  awk '{print $1}'
)
printf 'FLASH_SHA=%s\n' "$FLASH_SHA"
```

If no CUDA 12.8-capable `nvcc` exists, record external build as unavailable
and retain PyTorch SDPA. Do not install a CUDA toolkit or modify the active
environment.

- [ ] **Step 3: Clone the environment and attempt an sm_120-only official build**

Only when a suitable `nvcc` exists:

```bash
conda create -y \
  --clone /home/youngwoo/miniconda3/envs/pixal3d \
  --prefix /home/youngwoo/miniconda3/envs/pixal3d-flash
conda activate /home/youngwoo/miniconda3/envs/pixal3d-flash
python -m pip install packaging psutil ninja
git clone https://github.com/Dao-AILab/flash-attention.git \
  /home/youngwoo/data/pixal3d/tools/flash-attention-sm120
cd /home/youngwoo/data/pixal3d/tools/flash-attention-sm120
CUDA_ROOT=$(
  for root in /usr/local/cuda-12.8 /usr/local/cuda /opt/cuda
  do
    if test -x "$root/bin/nvcc" &&
       "$root/bin/nvcc" --version | grep -q 'release 12.8'
    then
      printf '%s\n' "$root"
      break
    fi
  done
)
FLASH_SHA=$(
  git ls-remote \
    https://github.com/Dao-AILab/flash-attention.git \
    refs/heads/main |
  awk '{print $1}'
)
git checkout "$FLASH_SHA"
CUDA_HOME="$CUDA_ROOT" \
FLASH_ATTN_CUDA_ARCHS=120 \
MAX_JOBS=8 \
NVCC_THREADS=2 \
python -m pip install . --no-build-isolation
```

Record `CUDA_ROOT` and `FLASH_SHA` verbatim in the report before building.

- [ ] **Step 4: Run correctness and backend smoke tests**

In `pixal3d-flash`, test BF16 forward and backward on the Node16 GPU:

```python
import torch
from flash_attn import flash_attn_func

q = torch.randn(2, 1024, 12, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
y = flash_attn_func(q, k, v)
assert y.shape == q.shape
y.float().square().mean().backward()
assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
```

Then run a one-step SS profile with `ATTN_BACKEND=flash_attn`. If import,
kernel launch, forward, or backward fails on `sm_120`, retain SDPA.

- [ ] **Step 5: Select and document the backend**

Compare the successful external one-step result with the measured SDPA SS
baseline using identical data, `K=6`, batch, and split. Select external
FlashAttention only if it is stable and at least 5% faster without raising
peak physical VRAM. Otherwise select SDPA.

Write the exact environment, source SHA, commands, result, selected backend,
and fallback reason to:

`docs/superpowers/reports/2026-07-28-node16-attention-backend.md`

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/reports/2026-07-28-node16-attention-backend.md
git commit -m "docs: record node16 attention backend validation"
```

---

### Task 3: Stage the four-model Node16 profiling inputs

**Files:**
- Create: `docs/superpowers/reports/2026-07-28-node16-profile-inputs.md`

**Interfaces:**
- Consumes: four Node17 pilot roots under `/root/node17/data/pixal3d/train/development/abo-pilot64` and four single-view checkpoints under `/root/node17/data/pixal3d/train/checkpoints/single_view`.
- Produces: verified Node16 pilot roots under `/home/youngwoo/data/pixal3d/train/profile-input/abo-pilot64/{stage}` and shared checkpoints under `/file3/youngwoo/pixal3d/train/checkpoints/single_view`.

- [ ] **Step 1: Verify source counts and hashes**

For each pilot stage verify exactly 64 instance directories exist in every
required stage component. Record byte sizes and file counts. Compute
SHA-256 for these four checkpoints:

```text
ss_flow_img_dit_1_3B_64_bf16.pt
slat_flow_img2shape_dit_1_3B_512_bf16.pt
slat_flow_img2shape_dit_1_3B_1024_bf16.pt
slat_flow_imgshape2tex_dit_1_3B_1024_bf16.pt
```

- [ ] **Step 2: Transfer into create-only targets**

Create the exact Node16 targets after confirming they do not already contain
different files. Transfer pilot data with low-priority tar streams over the
existing SSH control socket. Transfer checkpoints to:

```text
/file3/youngwoo/pixal3d/train/checkpoints/single_view
```

Do not overwrite a pre-existing checkpoint unless its SHA-256 already
matches the source.

- [ ] **Step 3: Verify destination evidence**

Recompute checkpoint SHA-256 values on Node16 and compare them byte-for-byte
with Step 1. Recount all pilot stage components and instantiate each dataset
without CUDA to verify it exposes 64 samples.

Write paths, sizes, counts, and hashes to:

`docs/superpowers/reports/2026-07-28-node16-profile-inputs.md`

- [ ] **Step 4: Deploy the reviewed source to the isolated checkout**

Transfer the current committed `HEAD` with `git archive` into
`/home/youngwoo/Pixal3D-training`, never
`/home/youngwoo/Pixal3D`. Compare SHA-256 for `train.py` and all four
production configs between local and Node16.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/reports/2026-07-28-node16-profile-inputs.md
git commit -m "docs: record node16 four-model profile inputs"
```

---

### Task 4: Profile `8/2` feasibility and `8/4` six-GPU throughput

**Files:**
- Create: `docs/superpowers/reports/2026-07-28-node16-multiview-profile.md`

**Interfaces:**
- Consumes: reviewed production configs, selected attention backend, staged 64-instance pilot data, and staged checkpoints.
- Produces: per-model `8/2` feasibility, production `8/4` timing/VRAM, and raw evidence paths.

- [ ] **Step 1: Generate bounded profile configs**

For every stage generate two untracked JSON configs derived mechanically
from its production config:

```json
"max_steps": 5,
"batch_size_per_gpu": 8,
"batch_split": 2
```

and:

```json
"max_steps": 5,
"batch_size_per_gpu": 8,
"batch_split": 4
```

Both variants must additionally use:

```json
"min_condition_views": 6,
"max_condition_views": 6,
"num_workers": 0,
"prefetch_data": false,
"i_print": 1,
"i_log": 1,
"i_sample": -1,
"i_save": 999999,
"max_checkpoints": null
```

Write them only under:

`/home/youngwoo/Pixal3D-training/.profile-configs`

- [ ] **Step 2: Run six-GPU `8/2` feasibility profiles**

Before each stage, record all GPU processes and free VRAM. Define the
following exact stage JSON values:

```bash
declare -A PROFILE_DATA_JSON
PROFILE_DATA_JSON[ss64]='{"ABO":{"base":"/home/youngwoo/data/pixal3d/train/profile-input/abo-pilot64/ss64","render_cond":"/home/youngwoo/data/pixal3d/train/profile-input/abo-pilot64/ss64/renders_cond","ss_latent":"/home/youngwoo/data/pixal3d/train/profile-input/abo-pilot64/ss64/ss_latents/ss_enc_conv3d_16l8_fp16_64_view"}}'
PROFILE_DATA_JSON[shape512]='{"ABO":{"base":"/home/youngwoo/data/pixal3d/train/profile-input/abo-pilot64/shape512","render_cond":"/home/youngwoo/data/pixal3d/train/profile-input/abo-pilot64/shape512/renders_cond","shape_latent":"/home/youngwoo/data/pixal3d/train/profile-input/abo-pilot64/shape512/shape_latents/shape_enc_next_dc_f16c32_fp16_512_view"}}'
PROFILE_DATA_JSON[shape1024]='{"ABO":{"base":"/home/youngwoo/data/pixal3d/train/profile-input/abo-pilot64/shape1024","render_cond":"/home/youngwoo/data/pixal3d/train/profile-input/abo-pilot64/shape1024/renders_cond","shape_latent":"/home/youngwoo/data/pixal3d/train/profile-input/abo-pilot64/shape1024/shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view"}}'
PROFILE_DATA_JSON[pbr1024]='{"ABO":{"base":"/home/youngwoo/data/pixal3d/train/profile-input/abo-pilot64/pbr1024","render_cond":"/home/youngwoo/data/pixal3d/train/profile-input/abo-pilot64/pbr1024/renders_cond","shape_latent":"/home/youngwoo/data/pixal3d/train/profile-input/abo-pilot64/pbr1024/shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view","pbr_latent":"/home/youngwoo/data/pixal3d/train/profile-input/abo-pilot64/pbr1024/pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix"}}'
```

For each stage assign `STAGE` to one of `ss64`, `shape512`, `shape1024`,
or `pbr1024`, then launch:

```bash
PROFILE_CONFIG="/home/youngwoo/Pixal3D-training/.profile-configs/${STAGE}-b8s2-k6-w6.json"
python train.py \
  --config "$PROFILE_CONFIG" \
  --data_dir "${PROFILE_DATA_JSON[$STAGE]}" \
  --num_gpus 6 \
  --auto_retry 0 \
  --ckpt none
```

Set the selected attention backend environment variable. Do not pass
`--use_wandb` or `--smoke_steps`. Poll physical VRAM on all GPUs every
0.2 seconds. Record whether all five steps complete or the exact OOM/error.

- [ ] **Step 3: Run six-GPU `8/4` production-candidate profiles**

Run the same command for the `8/4` config. Every stage must complete five
optimizer steps before its duration is considered measurable. Record all
five per-step times and peak physical VRAM for every GPU.

If a stage OOMs at `8/4`, stop that stage, preserve evidence, and report it
as a blocker rather than silently increasing split.

- [ ] **Step 4: Verify cleanup and write the report**

After every run verify no profile training or monitor process remains and
GPU usage returns to its pre-profile baseline.

Write:

- exact backend;
- stage/config;
- six-GPU world size;
- `8/2` pass/OOM;
- `8/4` five-step times;
- mean and last-four-step mean;
- peak physical VRAM per GPU;
- raw Node16 evidence directories

to `docs/superpowers/reports/2026-07-28-node16-multiview-profile.md`.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/reports/2026-07-28-node16-multiview-profile.md
git commit -m "docs: record node16 multiview batch profiles"
```

---

### Task 5: Calculate and verify four-model 20k durations

**Files:**
- Modify: `docs/superpowers/reports/2026-07-28-node16-multiview-profile.md`

**Interfaces:**
- Consumes: the five `8/4` per-step measurements for each stage.
- Produces: raw-compute and 10%-margin estimates for 20,000 optimizer steps.

- [ ] **Step 1: Calculate from the steady-state measurement**

For each stage calculate:

```text
steady_seconds = mean(step_2, step_3, step_4, step_5)
raw_seconds = steady_seconds * 20000
planning_seconds = raw_seconds * 1.10
```

Report hours and days to two decimal places. Do not derive one model's
duration from another model's timing.

- [ ] **Step 2: Cross-check arithmetic**

Run an independent `awk` or calculator pass over the raw log values and
confirm the report values differ by less than one second before rounding.

- [ ] **Step 3: Run final repository verification**

Run:

```bash
conda run --no-capture-output -n pixal3d python -m pytest -q \
  tests/multiview/test_configs.py \
  tests/multiview/test_train_smoke_override.py
git diff --check
git status --short --branch
```

Expected: selected tests pass, diff check exits 0, and only the intended
report amendment is uncommitted.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/reports/2026-07-28-node16-multiview-profile.md
git commit -m "docs: estimate 20k multiview fine-tuning time"
```
