# Stage-Specific Batch and Checkpoint Policy

## Goal

Change the four production multi-view fine-tuning configs so the two 1024
stages use a conservative per-GPU batch while the lower-resolution stages keep
their current throughput-oriented batch. Save checkpoints every 2,000 optimizer
steps for all four stages.

## Production policy

All runs use six GPUs and 20,000 optimizer steps.

| Stage | Batch per GPU | Batch split | Global batch | Checkpoint interval |
|---|---:|---:|---:|---:|
| `ss64` | 8 | 4 | 48 | 2,000 |
| `shape512` | 8 | 4 | 48 | 2,000 |
| `shape1024` | 2 | 1 | 12 | 2,000 |
| `pbr1024` | 2 | 1 | 12 | 2,000 |

All four configs keep:

- `max_steps=20000`;
- `i_sample=-1`;
- `max_checkpoints=5`;
- the existing model, optimizer, data, checkpoint input, and output paths.

At most the latest five periodic checkpoints remain. No snapshot policy,
model architecture, conditioning, attention backend, or training objective is
changed.

## Test contract

The config test must assert the exact stage-to-batch mapping above instead of
requiring one common batch policy. It must also assert `i_save=2000` for every
stage and preserve the existing assertions for steps, snapshots, retention,
checkpoints, and output paths.

The test must be observed failing against the current configs before the
production JSON files are changed.

## Report contract

The Node16 profile report must distinguish:

1. the new production defaults in the table above; and
2. the retained historical measurements collected with batch 8, split 4 and
   global batch 48.

Historical VRAM and duration measurements remain useful evidence and must not
be deleted or relabelled as measurements of the new 1024-stage defaults.
The existing Shape1024 and PBR1024 20k duration estimates must be marked as
non-authoritative for batch 2, split 1 until that configuration is separately
profiled.

This task does not run new GPU profiles or start fine-tuning.

## Success criteria

- The four production configs exactly match the policy table.
- Config tests cover the heterogeneous batch policy and 2,000-step saves.
- The report clearly states which measurements are historical.
- No unrelated code, preprocessing process, staged data, or checkpoint is
  changed.

