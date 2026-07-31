# Multi-view Checkpoint Retention Design

## Goal

Apply one conservative training-output policy to all four multi-view fine-tuning stages:

- disable snapshots;
- save checkpoints every 5,000 optimizer steps;
- retain only the latest five complete checkpoints;
- use `batch_split = 1`.

This changes training operations only. It does not alter the model architecture, dataset sampling, optimization objective, or `max_steps`.

## Configuration

The following four checked-in multi-view configs use the same trainer values:

- `configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json`
- `configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json`
- `configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json`
- `configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json`

Each config sets:

```json
{
  "batch_split": 1,
  "i_sample": -1,
  "i_save": 5000,
  "max_checkpoints": 5
}
```

## Complete-checkpoint contract

`misc_stepXXXXXXX.pt` is the completion marker for a checkpoint step because it is written after the model and EMA files. Retention considers only steps that have this marker.

When `max_checkpoints` is configured, `BasicTrainer.save()` writes all files for the new step synchronously. Only after every `torch.save` call succeeds does it prune old complete steps. It keeps the five numerically newest complete steps and removes every `*_stepXXXXXXX.pt` file belonging to each obsolete complete step, including model, EMA, and misc files.

If any save fails, the exception propagates and pruning does not run. Existing complete checkpoints therefore remain available. A partial new step without its misc marker is not counted as complete.

## Compatibility

`max_checkpoints` defaults to `None`. Configurations that omit it retain the existing non-blocking save behavior and do not prune checkpoints.

A configured value must be a positive integer. Booleans, zero, negative numbers, and non-integers are rejected.
