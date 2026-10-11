# Trainers

## Overview

`BasicTrainer` owns training state; flow/VAE subclasses specialize objectives, conditioning, collation and snapshots.

## Where to look

| Task | Location | Notes |
|---|---|---|
| Shared training lifecycle | `basic.py` | Initialization, distributed wrapping, optimizer/EMA, resume and finetune, save/load. |
| Dense flow matching | `flow_matching/flow_matching.py` | Diffusion path, dense loss, snapshots and conditioned variants. |
| Sparse flow matching | `flow_matching/sparse_flow_matching.py` | Sparse noise/loss, balanced sampler, sparse collation. |
| Conditioning | `flow_matching/mixins/` | CFG, text, image, multi-image, and projected-image conditioning. |
| VAE objectives | `vae/` | Shape, PBR, and sparse-structure-specific loss and snapshot behavior. |

## Contracts

- Trainers inherit `BasicTrainer`; model entries are keyed by configured names (flow trainers use `denoiser`). Keep objective-specific behavior in the matching trainer subclass.
- Resume via `load_dir` and `step` restores model and EMA weights plus `misc_stepNNNNNNN.pt` state: optimizer, sampler, and configured scaler, LR scheduler, elastic controller, and gradient clip state.
- `finetune_ckpt` loads per-model weights into the current initialization; it does not restore optimizer/sampler training progress. Shape mismatches are retained from the current model, and missing-key policy is handled in `finetune_from()`.
- Checkpoint files are under `ckpts/`: `{name}_step{step:07d}.pt`, optional `{name}_ema{rate}_step{step:07d}.pt`, and `misc_step{step:07d}.pt`. Preserve names expected by `load()`.
- Mixed precision supports full-precision master parameters (`inflat_all`) and AMP; dtype is configured separately. EMA is maintained over master parameters.
- Base training uses distributed training support and DDP wrapping. Dataset batches flow through `load_data()` and then `training_losses(**mb_data)`; retain the mapping-shaped batch contract.
- Sparse flow expects `x_0: SparseTensor`. It creates noise by replacing features while retaining coordinates, and computes per-instance MSE over `x_0.layout` slices.
- Sparse flow dataloading calls `dataset.collate_fn(split_size=batch_split)` with `BalancedResumableSampler`; datasets used here must provide that collator and compatible split semantics.
- Conditioning mixins define the `cond` and extra keyword arguments passed to the denoiser. Preserve those signatures across training and inference/snapshot paths.
- `batch_size_per_gpu` must be divisible by `batch_split`. Dense loading splits batches; sparse collation supplies token-balanced packs for gradient accumulation.
- `ImageConditionedProjMixin` and `DinoV3ProjFeatureExtractor` live in `flow_matching/mixins/image_conditioned_proj.py`; extractor projection grid and feature-channel dimensions must match the denoiser stage.

## Gotchas

- `FlowMatchingTrainer` samples timesteps per batch and passes `t * 1000` to the denoiser; its target is velocity, not noise prediction.
- `SparseFlowMatchingTrainer` has specialized snapshot batching/device transfer because sparse tensors cannot follow the dense tensor path unchanged.
- New checkpoint state must be added consistently to save and load paths; otherwise resumed runs silently lack the intended state.

## Fine-tuning and Verification

- The full-finetune recipe keeps native AdamW, mean-over-microbatches loss, BF16 AMP
  with FP32 parameters/optimizer, adaptive clipping and EMA 0.9999. Frozen image
  encoders/VAEs are dependencies; all four flow denoisers are fine-tuned.
- `batch_size_per_gpu` is the full loader batch, not a forward microbatch.
  `batch_split` counts accumulation packs; EBS multiplies the per-rank batch by
  world size. Sparse packs are balanced by tokens but the loss is not token-weighted.
- Five-epoch recipe budgets use optimizer updates from `floor(assets / EBS)`,
  with 5% linear warmup then constant LR. The generated resolved config owns
  paths and actual asset counts; portable Git recipes do not encode one machine.
- Native `run()` saves only on `i_save` boundaries. The multiview runtime hooks
  additionally save at epoch boundaries, completion and graceful stop; LoRA
  periodic saves occur every 500 optimizer updates, independently of validation.
- LoRA freezes the BF16 base and optimizes FP32 adapters in self-attention,
  global cross-attention and `proj_linear`, with fixed activation checkpointing
  and no EMA. Resume reconstructs adapters before restoring the full checkpoint.
- `microbatch_to_device` retains accumulation packs on CPU, transferring only
  the active pack. Sparse device caches must not be shared back into CPU packs.
- The experiment wrapper installs runtime hooks in every spawned worker and
  records EBS/K/pack sizes/VRAM. Rank 0 evaluates held-out raw-weight flow losses
  at the configured validation interval using unwrapped models. Evaluation restores RNG/model mode
  and must not change optimizer, EMA or training sampler state. Debug K forcing requires workers=0;
  persistent workers do not observe later mutations of the parent dataset.
