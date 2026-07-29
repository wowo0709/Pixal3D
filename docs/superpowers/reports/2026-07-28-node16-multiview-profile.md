# Node16 multiview batch profiles

Date: 2026-07-29 UTC
Host: `rvi-node016` (`n16.unist.info`, SSH port `55555`)
World size: 6 physical GPUs
Attention backend: `ATTN_BACKEND=sdpa`,
`SPARSE_ATTN_BACKEND=sdpa`

## Result

At the reviewed production batch size of 8 per GPU and exactly six condition
views:

- `ss64`: `8/2` OOM; `8/4` completed five steps.
- `shape512`: `8/2` and `8/4` both completed five steps.
- `shape1024`: `8/2` and `8/4` both OOM before step 1.
- `pbr1024`: `8/2` and `8/4` both OOM before step 1.

Thus the current `8/4` production candidate is measurable only for `ss64` and
`shape512`. `shape1024` and `pbr1024` are blockers at the specified production
batch policy. A separately labelled allocator diagnostic did not make either
blocked stage executable.

## Exact profile configuration

Eight untracked configs were generated under:

```text
/home/youngwoo/Pixal3D-training/.profile-configs
```

They were derived from the four reviewed production configs. The effective
fields were changed only at their actual nested schema locations:

```text
dataset.args.min_condition_views = 6
dataset.args.max_condition_views = 6
trainer.args.max_steps = 5
trainer.args.batch_size_per_gpu = 8
trainer.args.batch_split = 2 or 4
trainer.args.num_workers = 0
trainer.args.prefetch_data = false
trainer.args.i_print = 1
trainer.args.i_log = 1
trainer.args.i_sample = -1
trainer.args.i_save = 999999
trainer.args.max_checkpoints = null
```

CPU JSON assertions verified every field in all eight configs. Canonical hashes
of each profile's `models`, optimizer, and complete remainder after removing the
authorized fields matched its production source. Evidence:

```text
/home/youngwoo/Pixal3D-training/.profile-evidence/task4-20260729T0149Z/profile-config-validation.txt
```

The Task 4 plan's data JSON omitted the verified staging layer `active`. Its
literal paths produced an empty metadata frame before CUDA initialization. A
minimal CPU-only probe proved that the corrected `ss64/active` paths construct
the configured dataset with exactly 64 samples while CUDA remains unavailable
and uninitialized. All runs therefore used the corresponding
`<stage>/active/...` paths without moving or modifying staged data:

```text
/home/youngwoo/Pixal3D-training/.profile-evidence/task4-20260729T0149Z/ss64-active-path-cpu-probe.txt
```

The command shape for every production-matrix run was:

```bash
ATTN_BACKEND=sdpa \
SPARSE_ATTN_BACKEND=sdpa \
PYTHONPATH=/home/youngwoo/Pixal3D-training/.profile-deps:/home/youngwoo/Pixal3D-training \
python train.py \
  --config "$PROFILE_CONFIG" \
  --data_dir "$PROFILE_DATA_JSON" \
  --num_gpus 6 \
  --auto_retry 0 \
  --ckpt none
```

No run passed `--use_wandb`, `--smoke_steps`, or an alternate batch, split,
elastic-controller, or model setting. Snapshotting was disabled by
`i_sample=-1`; `i_save=999999` prevented checkpoint saves.

## Production matrix

Physical VRAM values are peak device `memory.used` in MiB, ordered GPU0 through
GPU5. They include the recorded coexistence baseline as required.

| Stage | Config | `8/2` result | `8/2` peak MiB, GPU0..5 | `8/4` result | `8/4` five step times, seconds | Mean, seconds | Last-four mean, seconds | `8/4` peak MiB, GPU0..5 |
|---|---|---|---|---|---|---:|---:|---|
| `ss64` | `ss64-b8s{2,4}-k6-w6.json` | 1/5, then OOM | 97,167 / 97,205 / 97,233 / 97,207 / 97,147 / 97,245 | 5/5 pass | 15.696 / 12.877 / 12.449 / 12.330 / 12.350 | 13.140 | 12.501 | 71,857 / 66,577 / 66,561 / 66,577 / 64,933 / 69,842 |
| `shape512` | `shape512-b8s{2,4}-k6-w6.json` | 5/5 pass | 64,623 / 60,303 / 59,203 / 62,041 / 64,749 / 64,552 | 5/5 pass | 21.791 / 17.420 / 16.373 / 17.289 / 16.070 | 17.789 | 16.788 | 57,141 / 57,343 / 59,205 / 56,675 / 58,566 / 54,519 |
| `shape1024` | `shape1024-b8s{2,4}-k6-w6.json` | OOM before step 1 | 86,053 / 80,655 / 80,639 / 97,041 / 79,013 / 96,938 | OOM before step 1; blocker | N/A | N/A | N/A | 83,927 / 87,047 / 78,499 / 93,959 / 89,824 / 77,389 |
| `pbr1024` | `pbr1024-b8s{2,4}-k6-w6.json` | OOM before step 1 | 94,961 / 88,251 / 87,983 / 87,405 / 96,234 / 97,218 | OOM before step 1; blocker | N/A | N/A | N/A | 79,787 / 81,055 / 96,949 / 90,813 / 72,750 / 96,603 |

Exact step durations came from the trainer's unrounded `time.step` records:

```text
/file3/youngwoo/pixal3d/ckpts/ss64/log_20260729_111154.txt
/file3/youngwoo/pixal3d/ckpts/shape512/log_20260729_112530.txt
```

The `ss64 8/4` compute-app monitor saw no Blender context. The `shape512 8/4`
run coexisted with transient Blender contexts throughout the one-second
compute-app snapshots, with at most about 4,024 MiB on an individual GPU. Its
last-four step times ranged from 16.070 to 17.420 seconds, with no isolated
large timing outlier, so no timing retry was performed. The measurement remains
explicitly a coexistence-baseline result.

## Exact OOM evidence

### `ss64 8/2`

The first optimizer step completed; the next forward pass requested another
96 MiB on GPU0 with only 73.75 MiB free.
The profile process held 93.27 GiB; stable reservation PIDs `4047349` and
`931568` held 808 MiB each. The physical monitor peaked between 97,147 and
97,245 MiB across all GPUs.

### `shape1024 8/2`

Multiview projection requested another 6 GiB on GPU5 with 312.38 MiB free.
The profile process held 89.88 GiB and the stable Nuclio PID `711845` held
4.77 GiB. Even excluding that external context, the reported free memory would
remain below the requested 6 GiB.

### `pbr1024 8/2`

NATTEN `cutlass-fna` requested another 20 GiB on GPU3 with 9.62 GiB free. The
profile process held 83.15 GiB, including 67.32 GiB allocated and 14.97 GiB
reserved but unallocated. This is recorded as a current-configuration OOM; no
unplanned allocator or split change was made.

### `shape1024 8/4`

Multiview projection requested another 24 GiB on GPU0 with 22.60 GiB free. The
profile process held 68.44 GiB, including 46.48 GiB allocated and 21.11 GiB
reserved but unallocated. The stable reservation contexts and a transient
2.33 GiB Blender context were also recorded. This is the baseline production
blocker.

### `pbr1024 8/4`

The same projection stack requested another 24 GiB on GPU0 with 20.00 GiB
free. The profile process held 71.21 GiB, including 46.49 GiB allocated and
23.86 GiB reserved but unallocated. This is the baseline production blocker.

## One-factor allocator diagnostics

These diagnostics do not replace the baseline verdicts above. They used the
same `8/4` configs and command, with only:

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

added and explicitly recorded.

| Stage | Result | Exact terminal allocation | Peak MiB, GPU0..5 | Evidence |
|---|---|---|---|---|
| `shape1024` | OOM before step 1 | requested 36 GiB with 31.80 GiB free; profile 54.98 GiB; external 4.77 GiB Nuclio plus 3.27 GiB Blender; only 812.25 MiB reserved-unallocated | 90,271 / 97,121 / 91,829 / 97,151 / 83,802 / 95,237 | `shape1024-b8s4-diagnostic-expandable-attempt2` |
| `pbr1024` | OOM before step 1 | requested 36 GiB with 35.28 GiB free; profile 54.90 GiB; external 4.77 GiB Nuclio; only 712.64 MiB reserved-unallocated | 85,693 / 96,001 / 85,715 / 97,247 / 75,901 / 97,162 | `pbr1024-b8s4-diagnostic-expandable-attempt2` |

The diagnostics sharply reduced reserved-but-unallocated memory but still
failed at the same multiview stack-copy operation. The current implementation,
not merely allocator fragmentation, prevents these stages from running at
`8/4` under the approved coexistence baseline.

## Dependency evidence

The reviewed production NAF path initially exposed two missing imports in the
active Node16 environment. The active environment was not modified. With
explicit authorization, exact isolated packages were installed only under:

```text
/home/youngwoo/Pixal3D-training/.profile-deps
```

The installed packages were `einops==0.8.0` and the official
`natten==0.21.1+torch280cu128` wheel installed with `--no-deps` from
`https://whl.natten.org`. Standalone assertions verified:

```text
NATTEN distribution: 0.21.1+torch280cu128
NATTEN module:       0.21.1
Torch:               2.8.0+cu128
Torch CUDA:          12.8
```

Without the profile-only `PYTHONPATH`, both packages remain absent from the
active environment. Evidence:

```text
/home/youngwoo/Pixal3D-training/.profile-evidence/task4-20260729T0149Z/profile-deps-install.txt
/home/youngwoo/Pixal3D-training/.profile-evidence/task4-20260729T0149Z/profile-deps-natten-install.txt
/home/youngwoo/Pixal3D-training/.profile-evidence/task4-20260729T0149Z/profile-deps-natten-corrected-verification.txt
```

The first end-to-end `shape512 8/2` run after the corrected verification
completed all five CUDA training steps.

## Invalid and non-launch attempts

Invalid evidence was retained rather than mixed with the matrix:

1. `ss64-b8s2-attempt1` used the plan's literal paths and failed before CUDA
   with `No view columns found in metadata: []`.
2. `ss64-b8s2-attempt2` exposed a config-generation placement error: top-level
   overrides left resolved trainer values at 20,000 steps, split 4, and 14
   workers. It was terminated, its one orphan rank/process group was cleaned,
   and fresh evidence proved no descendant remained before regeneration.
3. The configs were regenerated from production with the exact nested fields;
   model, optimizer, and unauthorized-remainder hashes then matched.
4. `shape512-b8s2-attempt3` stopped before step 1 on missing `einops`;
   `attempt4` stopped before step 1 on missing `natten`; `attempt5` is the valid
   five-step result.
5. Guard-failed directories contain audits only and never launched Python:
   `shape512-b8s2-attempt1`, `shape512-b8s2-attempt2`,
   `shape512-b8s4-attempt1`, `shape512-b8s4-attempt2`,
   `pbr1024-b8s4-attempt1`, and the first diagnostic audit for each blocked
   stage.

## Monitoring and coexistence

Before every launched run, all GPU compute contexts and full process state were
captured. A run launched only when every GPU had at least 90,112 MiB free and
utilization at most 20 percent. The stable baseline was:

- reservation PIDs `4047349` and `931568`, 808 MiB each on GPUs 0-3;
- Nuclio PID `711845`, about 4,886 MiB on GPU5;
- protected transient preprocessing Blender contexts, when present.

Physical `memory.used`, `memory.free`, and utilization for all six GPUs were
queried in a loop with `sleep 0.2` after each query. Because `nvidia-smi` query
runtime and contention are included in the timestamps, effective median
sample-to-sample cadence was approximately 0.40-0.76 seconds across the
reported runs, rather than a 0.20-second wall-clock cadence. Compute-app
ownership was sampled separately once per second. This limitation is preserved
in the raw evidence and should be considered when interpreting very short peak
transients.

Raw parsed matrix evidence:

```text
/home/youngwoo/Pixal3D-training/.profile-evidence/task4-20260729T0149Z/matrix-summary.json
```

## Cleanup and preservation

A fresh final audit at `2026-07-29T02:42:22Z` proved:

- no profile `train.py` parent;
- no profile runner or monitor;
- no multiprocessing spawn/resource-tracker helper;
- GPU memory returned to the stable baseline:
  `1,630 / 1,630 / 1,630 / 1,630 / 3 / 4,895 MiB`;
- preprocessing supervisor/worker PIDs
  `567993 / 567994 / 567996 / 1915976` remained present;
- the active preprocessing checkout's `train.py` SHA-256 remained
  `f1266f8f6608d58dee2f21b86a4061a5aca62bbd4c3dd457d1e56467839910f8`.

No checkpoint or snapshot was created. The trainer wrote only its ordinary
`command.txt`, `config.json`, model-summary, and step-log evidence in the four
configured output directories.

Cleanup evidence:

```text
/home/youngwoo/Pixal3D-training/.profile-evidence/task4-20260729T0149Z/final-cleanup-audit.txt
```

## Raw Node16 evidence directories

Base:

```text
/home/youngwoo/Pixal3D-training/.profile-evidence/task4-20260729T0149Z
```

Production-matrix result directories:

```text
ss64-b8s2-attempt3
shape512-b8s2-attempt5
shape1024-b8s2-attempt1
pbr1024-b8s2-attempt1
ss64-b8s4-attempt1
shape512-b8s4-attempt3
shape1024-b8s4-attempt1
pbr1024-b8s4-attempt2
```
