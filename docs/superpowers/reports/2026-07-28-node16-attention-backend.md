# Node16 attention backend validation

Date: 2026-07-28 UTC
Host: `rvi-node016` (`n16.unist.info`, SSH port `55555`)
Decision: retain PyTorch scaled dot product attention with
`ATTN_BACKEND=sdpa` and `SPARSE_ATTN_BACKEND=sdpa`.

## Scope and safety constraints

This validation was read-only on the active checkout and active conda
environment. It did not alter `/home/youngwoo/Pixal3D`, did not install into
`/home/youngwoo/miniconda3/envs/pixal3d`, and did not stop, pause, or otherwise
interact with preprocessing. The authorized clone and source-build paths were
not created because the CUDA 12.8 build prerequisite was not met.

All Node16 access used:

```bash
ssh -S /tmp/pixal3d-n16-control.sock \
  -p 55555 -o BatchMode=yes youngwoo@n16.unist.info
```

## Immutable baseline

At `2026-07-28T17:04:28Z`, the baseline was collected with the exact Python
probe required by the plan:

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

Observed output:

```text
torch 2.8.0+cu128
cuda 12.8
capability (12, 0)
external_flash_attn False
torch_flash_available True
torch_flash_enabled True
```

An additional read-only capability query reported GPU4 as `(12, 0)` as well.
Thus the active environment has no external `flash_attn` distribution, while
the PyTorch backend API reports fused Flash SDP available and enabled. These
flags do not prove that a fused SDPA kernel executed; no SDPA
forward/backward smoke or SS profile completed.

The repository accepts `sdpa` for both environment switches. Its dense and
sparse attention implementations dispatch to
`torch.nn.functional.scaled_dot_product_attention`; preprocessing commands
already pin both switches to `sdpa`.

## External build prerequisite resolution

The exact CUDA-root resolution command was:

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

Observed output:

```text
CUDA_ROOT=
```

The candidate-by-candidate audit at `2026-07-28T17:05:01Z` found:

```text
/usr/local/cuda-12.8: absent
/usr/local/cuda: symlink to /usr/local/cuda-13.0/
nvcc: release 13.0, V13.0.88
/opt/cuda: absent
```

The official main-branch source commit was resolved without cloning:

```bash
FLASH_SHA=$(
  git ls-remote \
    https://github.com/Dao-AILab/flash-attention.git \
    refs/heads/main |
  awk '{print $1}'
)
printf 'FLASH_SHA=%s\n' "$FLASH_SHA"
```

Observed output:

```text
FLASH_SHA=14c377950125c70b7a9dabf9c561fca53715ac7d
```

Because none of the permitted candidates provides CUDA 12.8 `nvcc`, the plan's
terminal prerequisite rule applies: the external build is unavailable, no CUDA
toolkit may be installed, and PyTorch SDPA must be retained. Consequently these
authorized-but-conditional paths remained absent:

```text
/home/youngwoo/miniconda3/envs/pixal3d-flash
/home/youngwoo/data/pixal3d/tools/flash-attention-sm120
```

No `conda create`, `pip install`, `git clone`, or FlashAttention build command
was executed.

## Smoke/profile disposition

No external correctness smoke or one-step SS profile was run because no
external package could be built. There was therefore no successful external
one-step result eligible for the plan's five-percent speed and peak-physical-
VRAM comparison.

Two independent runtime guards also prevented a bounded training test:

- GPU4 was not free. At `2026-07-28T17:05:01Z` it had `591 MiB` allocated by
  PID `3440682`,
  `/home/youngwoo/data/pixal3d/tools/blender-4.5.1-linux-x64/blender`, belonging
  to the active preprocessing workload. A later verification briefly observed
  no compute context, but the guarded smoke command immediately saw one context
  and `2,695 MiB`; its `test "$GPU4_CONTEXTS" -eq 0` guard failed before Python
  launched. This confirmed that the apparent idle window was not stable.
- The required isolated SS root
  `/home/youngwoo/data/pixal3d/train/profile-input/abo-pilot64/ss64` was absent
  (its `abo-pilot64` parent was absent too). The checkpoint
  `/home/youngwoo/data/pixal3d/train/checkpoints/single_view/ss_flow_img_dit_1_3B_64_bf16.pt`
  was present as a 5,359,981,708-byte regular file.

The active preprocessing supervisor and workers were observed but not altered.
The aborted guarded smoke did not initialize Torch or allocate GPU memory.

## Backend selection

Select:

```bash
ATTN_BACKEND=sdpa
SPARSE_ATTN_BACKEND=sdpa
```

Fallback reason: an official external FlashAttention build could not be
attempted under the approved procedure because Node16 has CUDA 13.0 `nvcc`, not
CUDA 12.8 `nvcc`, at the permitted toolkit locations. The active Torch
`2.8.0+cu128` backend API reports fused Flash SDP available and enabled.
PyTorch SDPA was selected by the CUDA 12.8 external-build gate as the
non-invasive fallback; it was not runtime-executed or performance-validated in
this task.

Re-evaluation requires a CUDA 12.8-capable `nvcc` at one of the three permitted
locations, a free GPU4, and the isolated SS profile root. Only then should the
official source be built in the cloned environment and compared against the
same-data, `K=6`, same-batch, same-split SDPA baseline.
