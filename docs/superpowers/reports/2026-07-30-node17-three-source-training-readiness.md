# Node17 three-source training readiness

Date: 2026-07-30 UTC

Reviewed code revision:
`34949b1422d4799eef68ab757430051cbf8da604`
(`docs: expand Node16 verification evidence`).

Historical-evidence validator revision:
`697a9d47dda9e79360f8c883abae85188e934aa6`
(`fix: bind historical validator ancestry`).

## Outcome and safety boundary

The create-only CPU workflow completed with exit code 0. Node17 now has a
canonical HSSD publication, the ABO + 3D-FUTURE + HSSD combined manifest,
four Node17 runtime configs, and an immutable preparation-evidence report.

ABO and 3D-FUTURE were not materialized or population-preflighted again.
Their existing report, handoff, and training-manifest chains were only
re-hashed and resolved. Only the four completed Node16 HSSD stage trees were
transferred.

`CUDA_VISIBLE_DEVICES=""` remained set for execution and every independent
readiness check. No `train.py`, `torchrun`, six-GPU DDP, W&B run, or
checkpoint write was started.

## Execution and transfer evidence

The execution ran from 2026-07-30 20:23:09 UTC through 23:14:33 UTC:

```text
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
/opt/conda/envs/pixal3d/bin/python \
  scripts/prepare_node17_training.py --execute
```

The source was exactly:

```text
youngwoo@n16.unist.info:55555
/home/youngwoo/data/pixal3d/train/production/hssd
```

The source and target inventories match exactly:

| Inventory | File count | Logical bytes |
| --- | ---: | ---: |
| Node16 source | 337,245 | 87,806,314,999 |
| Node17 target | 337,245 | 87,806,314,999 |

The resumable rsync took 857.503 seconds. The subsequent
`rsync --checksum --dry-run --itemize-changes --delete` exited 0 with empty
itemized output and took 49.326 seconds. The hidden staging path was then
fully preflighted and atomically promoted; it no longer exists after the
successful promotion.

The four full strict HSSD stage preflights took 9,358.468 seconds. HSSD
transfer, verification, rebase, strict preflight, and publication together
took 10,274.674 seconds. The complete orchestration took 10,281.816 seconds
(2 h 51 min 21.816 s).

Original Node16 and canonical Node17 materialization digests are:

| Stage | Original Node16 SHA-256 | Canonical Node17 SHA-256 |
| --- | --- | --- |
| `ss64` | `70f718287752b3d107d49b5bc5601fb8f467ad9f4bb3687970aeae38852678bb` | `a16cc854f3196da6853101b3fa13cee7d28a4718aae72c0bca11cf323a11b420` |
| `shape512` | `5e68c98a8006adf074d068d5571e1ce5ceed48f51f2ee3a76130a88853655e6c` | `229cb07ae0a47f440eb113968c8706f585cfaef4c34a75dd73232e18ff3c852f` |
| `shape1024` | `f85b6722c5e54e9f97abad8b588ead883af6252e1c024a2c1fed003e594719a3` | `3eda7409c6e2de539eb60dbbed09f3e2f86115ed6f2707f6a00a21ad8e1e8881` |
| `pbr1024` | `ce395c1167e787abaa58bb6a6004c2a669214eae126badf510c71e883c88fef7` | `9a6460ca450929715d9f733bdb0e67bf42e31ea3338e960fc63055714d2c13d3` |

## Immutable publication digests

| Publication | SHA-256 |
| --- | --- |
| ABO `training_data.json` | `fe107e817f33dbad6a12f65920862b2112240c0539aa2463689ecb920fa4bc61` |
| 3D-FUTURE `training_data.json` | `88b2ccffe41bad2fc83c543ca2acbf8c490e08a0f7a698145d260493b612d6a4` |
| HSSD `training_data.json` | `0777d272775dc2fe244d80df3b1251fb705a411b09b77941e2cab52ff626fa14` |
| Combined `training_data.json` | `c029be13b2ad2ae25b1d85522fd7a0cf7b2acf3e15a63b78621e4a4d04072bba` |
| Final evidence `report.json` | `87e1d3c5b3bd8867743d593ab9a88234afaf1d42d45a90b4da048b5001e9349e` |

Canonical paths:

```text
/root/node17/data/pixal3d/train/production/hssd/training_data.json
/root/node17/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json
/root/node17/data/pixal3d/train/production/node17-preparation-evidence/report.json
```

## Stage scope and configured loader checks

Every HSSD standalone loader check passed. It checked two boundary instances
per stage:

| Stage | HSSD count |
| --- | ---: |
| `ss64` | 6,078 |
| `shape512` | 6,059 |
| `shape1024` | 6,014 |
| `pbr1024` | 5,957 |

Every three-source combined loader check also passed. Each stage collated all
three sources and checked six boundary instances:

| Stage | ABO | 3D-FUTURE | HSSD | Total |
| --- | ---: | ---: | ---: | ---: |
| `ss64` | 3,660 | 8,495 | 6,078 | 18,233 |
| `shape512` | 3,628 | 8,502 | 6,059 | 18,189 |
| `shape1024` | 3,634 | 8,458 | 6,014 | 18,106 |
| `pbr1024` | 3,598 | 8,406 | 5,957 | 17,961 |

Sampling is `proportional-unweighted-concatenation` for all four stages.

After publication, a fresh process reran the non-mutating plan and the
independent combined configured DataLoader command:

```text
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
/opt/conda/envs/pixal3d/bin/python \
  scripts/preflight_multisource_training.py \
  --training-data \
  /root/node17/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json
```

It independently reproduced all four stage counts and passed every boundary
collation. The original exact-revision
`validate_node17_preparation_report(...)` also re-hashed every referenced
artifact and re-resolved all source and combined chains successfully before
the readiness-only delivery commit.

The final clean delivered descendant is reproducibly validated with:

```text
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
/opt/conda/envs/pixal3d/bin/python scripts/prepare_node17_training.py \
  --validate-evidence \
  --delivery-revision c3423fd23340b301940cd9c6ea084744caa77bd6 \
  --validator-revision 697a9d47dda9e79360f8c883abae85188e934aa6 \
  --report-sha256 87e1d3c5b3bd8867743d593ab9a88234afaf1d42d45a90b4da048b5001e9349e
```

This historical mode does not relax normal preparation or reuse: those paths
still require exact current HEAD. It requires a clean worktree, verifies all
three commits and ancestry, and verifies the raw immutable `report.json`
bytes against the independently supplied SHA-256 before parsing any report
field. It then enforces three exact path boundaries:

- execution evidence to delivery: only this Task's readiness/runbook docs;
- delivery to pinned validator: only the two delivery docs, the two
  validator/HSSD-transfer cores, CLI, and their two exact test paths;
- pinned validator to current HEAD: only the readiness/runbook docs.

Unrelated documentation, code, config, tests, missing/non-descendant
revisions, and dirty worktrees are rejected. After the path gate, all
artifacts are re-hashed and all source and combined chains are re-resolved
against the recorded execution revision.

## Runtime configs

| Stage | Batch/split | Six-GPU global batch | SHA-256 |
| --- | --- | ---: | --- |
| `ss64` | 8/4 | 48 | `7a06ffe930a830bf70a70b43d3791c9c43b432819ee1ac3bc5051767f77b674d` |
| `shape512` | 8/4 | 48 | `320758e80af2cd68b7353e940c8fa1c00e58f3bcbb23e816a04d1a51e646f704` |
| `shape1024` | 2/1 | 12 | `b461870cf667a45baddc63e8b38a0826114a529b3b40b5177cc1d1f828f18464` |
| `pbr1024` | 2/1 | 12 | `7ec3651e977bb11a0a13a42d2035dd13e45a1c85aaa9508dd030930d35c8eb2b` |

All four configs use 20,000 steps, two workers per rank, a 1,000-step
snapshot/save interval, and retain three checkpoints. Only `ss64` disables
the startup dataset snapshot.

## Process and GPU read-only audit

The strict process audit found zero `train.py` or `torchrun` processes after
preparation. The requested NVML compute-app query was attempted twice at
2026-07-30 23:16:32 UTC and returned:

```text
Failed to initialize NVML: Unknown Error
```

No recovery, reset, signal, or other GPU/process mutation was attempted.
The driver still exposed eight PCI GPU entries and `/dev/nvidia0` through
`/dev/nvidia7`. Therefore this report establishes that the workflow started
no training process, while current per-GPU compute-app memory could not be
observed through NVML at the audit instant.

## Exact launch commands

These commands are recorded for handoff only. They were not executed.

```bash
cd /root/dev/Pixal3D/.worktrees/multiview-model-extension
export PYTHONPATH=.
mkdir -p /root/node17/data/pixal3d/training-logs
```

`ss64`:

```bash
/opt/conda/envs/pixal3d/bin/python train.py \
  --config /root/node17/data/pixal3d/train/runtime-configs/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.node17.json \
  --training_data /root/node17/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json \
  --num_gpus 6 --use_wandb \
  2>&1 | tee -a /root/node17/data/pixal3d/training-logs/ss64.log
```

`shape512`:

```bash
/opt/conda/envs/pixal3d/bin/python train.py \
  --config /root/node17/data/pixal3d/train/runtime-configs/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.node17.json \
  --training_data /root/node17/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json \
  --num_gpus 6 --use_wandb \
  2>&1 | tee -a /root/node17/data/pixal3d/training-logs/shape512.log
```

`shape1024`:

```bash
/opt/conda/envs/pixal3d/bin/python train.py \
  --config /root/node17/data/pixal3d/train/runtime-configs/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.node17.json \
  --training_data /root/node17/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json \
  --num_gpus 6 --use_wandb \
  2>&1 | tee -a /root/node17/data/pixal3d/training-logs/shape1024.log
```

`pbr1024`:

```bash
/opt/conda/envs/pixal3d/bin/python train.py \
  --config /root/node17/data/pixal3d/train/runtime-configs/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.node17.json \
  --training_data /root/node17/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json \
  --num_gpus 6 --use_wandb \
  2>&1 | tee -a /root/node17/data/pixal3d/training-logs/pbr1024.log
```
