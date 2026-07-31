# Node16 Multi-View Fine-Tuning Profile Design

## Goal

Prepare all four multi-view Pixal3D flow models for 20,000-step fine-tuning
on six Node16 GPUs with a global batch size of 48, establish whether
`batch_size_per_gpu=8, batch_split=2` is viable, and estimate end-to-end
training time from measured six-GPU throughput.

## Scope

The four stages are:

1. `ss64`
2. `shape512`
3. `shape1024`
4. `pbr1024`

This work changes training configuration and performs bounded profiling. It
does not start a production fine-tuning run, enable W&B, take snapshots, or
add model-quality improvements.

## Production Configuration

All four multi-view configuration files will use:

- `max_steps: 20000`
- `batch_size_per_gpu: 8`
- `batch_split: 4`
- six GPUs, giving global batch size `8 * 6 = 48`
- `i_sample: -1`
- `i_save: 5000`
- `max_checkpoints: 5`

Existing optimizer, data, model, conditioning-view, and checkpoint-retention
behavior remains unchanged.

Node16 training outputs will use shared data3 storage under
`/file3/youngwoo/pixal3d/ckpts`. Initial single-view checkpoints and bounded
profile inputs must be available through Node16-accessible paths. Active
preprocessing source trees and processes must not be modified.

## Attention Backend Strategy

Node16 and the current development environment both use Python 3.11.15,
PyTorch 2.8.0+cu128, CUDA 12.8, and the CXX11 ABI. Neither environment has
the external `flash-attn` package installed.

PyTorch reports its fused FlashAttention SDPA backend as available on the
RTX PRO 6000 Blackwell GPU. Therefore `ATTN_BACKEND=sdpa` is the stable
baseline, not an unfused attention baseline.

External FlashAttention will be evaluated without modifying the working
`pixal3d` environment:

1. Clone the Node16 environment to a separate `pixal3d-flash` environment.
2. Install only an official FlashAttention build that targets `sm_120`.
3. Run representative forward/backward correctness checks.
4. A/B test it against PyTorch SDPA on the SS model.
5. Promote the external backend only if it is stable and materially faster;
   otherwise retain PyTorch SDPA.

FlashAttention installation failure or lack of Blackwell kernel support is
not a blocker for the requested fine-tuning profiles because PyTorch fused
SDPA remains the supported fallback.

## VRAM and Throughput Validation

For every model, run bounded six-GPU profiles on the already-preprocessed
64-instance ABO pilot data:

1. Test `batch_size_per_gpu=8, batch_split=2`, which gives microbatch 4.
2. Record success or CUDA OOM and exact peak allocated/reserved/physical
   VRAM per GPU.
3. Run the production candidate
   `batch_size_per_gpu=8, batch_split=4`, which gives microbatch 2.
4. Use at least five optimizer steps for the successful production
   candidate so optimizer-state and allocator growth are represented.
5. Fix conditioning at `K=6` to measure the worst configured number of
   condition views.

No profile may use W&B, snapshots, or checkpoint saving. Profiling processes
must be isolated from active preprocessing and removed after each run.

## Time Estimate

For each model, calculate:

- mean time per step across the bounded six-GPU `8/4` profile;
- steady-state mean excluding the first warm-up step;
- `20000 * steady_state_seconds_per_step`;
- a planning estimate with a 10% operational margin for dataloader,
  logging, 5k checkpoint saves, and short interruptions.

Report both raw compute time and the 10%-margin planning time. Estimates do
not include queueing time or a restart after node failure.

## Verification

Configuration tests must assert all four files use the exact production
values and preserve snapshots/checkpoint retention requirements. Runtime
evidence must include backend selection, model/stage, GPU/world size,
per-GPU batch and split, completed step count, per-step timing, peak VRAM,
and any OOM traceback.

The final handoff must state which attention backend was selected, which
models passed `8/2`, and the measured 20k duration for all four models.
