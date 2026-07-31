# Node16 four-model profile inputs

Date: 2026-07-29 UTC
Host: `rvi-node016` (`n16.unist.info`, SSH port `55555`)
Reviewed source HEAD: `23aa41dbeef33bd92323d4288510b56af7885fa2`

## Scope and safety

The four 64-asset pilot stages are staged at:

```text
/home/youngwoo/data/pixal3d/train/profile-input/abo-pilot64/{ss64,shape512,shape1024,pbr1024}
```

The four single-view checkpoints are staged at:

```text
/file3/youngwoo/pixal3d/train/checkpoints/single_view
```

The reviewed committed source was deployed with `git archive` only to:

```text
/home/youngwoo/Pixal3D-training
```

The active preprocessing checkout `/home/youngwoo/Pixal3D` was read-only
throughout. Its `train.py` SHA-256 remained
`f1266f8f6608d58dee2f21b86a4061a5aca62bbd4c3dd457d1e56467839910f8`,
and its preprocessing PID set remained `567993 567994 567996 1915976`
across deployment.

Initial target resolution proved that the canonical pilot root and all four
checkpoint files were absent. Transfers used Task-specific incoming paths,
low-priority I/O (`ionice -c3 nice -n 19`), exact destination checks, and
atomic promotion only after verification. No mismatched checkpoint was
overwritten.

After the original SSH control master blocked, completion used only:

```bash
ssh -S /tmp/pixal3d-n16-control-new.sock \
  -p 55555 -o BatchMode=yes youngwoo@n16.unist.info
```

## Pilot source and destination evidence

Node17 source:

```text
/root/node17/data/pixal3d/train/development/abo-pilot64
```

Node16 destination:

```text
/home/youngwoo/data/pixal3d/train/profile-input/abo-pilot64
```

Every required component has exactly 64 immediate instance directories on
both source and destination:

| Stage | Component relative to stage | Instances | Files | Bytes |
|---|---|---:|---:|---:|
| `ss64` | `active/renders_cond` | 64 | 577 | 82,956,776 |
| `ss64` | `active/ss_latents/ss_enc_conv3d_16l8_fp16_64_view` | 64 | 257 | 15,890,783 |
| `shape512` | `active/renders_cond` | 64 | 577 | 82,956,776 |
| `shape512` | `active/shape_latents/shape_enc_next_dc_f16c32_fp16_512_view` | 64 | 257 | 45,999,780 |
| `shape1024` | `active/renders_cond` | 64 | 577 | 82,956,776 |
| `shape1024` | `active/shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view` | 64 | 257 | 205,223,168 |
| `pbr1024` | `active/renders_cond` | 64 | 577 | 82,956,776 |
| `pbr1024` | `active/shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view` | 64 | 257 | 205,223,168 |
| `pbr1024` | `active/pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix` | 64 | 257 | 205,111,808 |

The stage totals and full sorted per-file SHA-256 manifest digests match
byte-for-byte between Node17 and Node16:

| Stage | Files | Bytes | Source and destination manifest SHA-256 |
|---|---:|---:|---|
| `ss64` | 835 | 98,848,491 | `5ccf55fef328867f886b36fee192e13b15cf758f4358d77cfa0cc632e2543234` |
| `shape512` | 835 | 128,957,504 | `56a7db7c50d7899f7d44599ba292bd1744a0f62f7ecdaf87e6496d9eede6e74f` |
| `shape1024` | 835 | 288,180,896 | `d6d9c42f6ffa37d90337d85f43db1f41a19fe4db4e737d1a8749ea8af9eb1ba2` |
| `pbr1024` | 1,093 | 493,300,217 | `7f2aa370bee2fe46e2841e3ea52c0a4fd8df3d9b484fcdb47034f56d2071ff4e` |

Manifest construction:

```bash
(cd "$stage_root" &&
  ionice -c3 nice -n 19 find . -type f -print0 |
  sort -z |
  xargs -0 sha256sum) |
sha256sum
```

The existing legacy SS source
`/home/youngwoo/data/pixal3d/train/profile-input/abo-pilot64-ss64`
was reused only after its complete 835-file manifest matched Node17
`ss64/active` at
`ad6afe7af4ef6449128385d06d36067666fcd14ec930f73cdf0650989da47c25`.
An interrupted first copy left 35 render instances and no latent tree.
Matching creation timestamps and empty sibling stages proved that partial was
the Task-created canonical target. Only that partial was removed. A
Node16-local low-priority tar copy was built in a create-only sibling, verified
against the legacy manifest, and atomically promoted to canonical `ss64`.

`shape512`, `shape1024`, and `pbr1024` were transferred directly from Node17
with low-priority tar streams over the restored SSH control socket. Each was
promoted only after its full manifest matched. No Task incoming artifact
remained after verification.

## Checkpoint evidence

Node17 source:

```text
/root/node17/data/pixal3d/train/checkpoints/single_view
```

Node16 destination:

```text
/file3/youngwoo/pixal3d/train/checkpoints/single_view
```

Each checkpoint was streamed at low priority to a unique create-only incoming
file, checked for exact size and SHA-256, and atomically promoted. A fresh
second pass on Node16 reproduced all four hashes:

| Checkpoint | Bytes | Source and destination SHA-256 |
|---|---:|---|
| `ss_flow_img_dit_1_3B_64_bf16.pt` | 5,359,981,708 | `9663332a48bb549ad2b9ace1bb4f50f26188186fb4b81bcfb8212dd70e307d45` |
| `slat_flow_img2shape_dit_1_3B_512_bf16.pt` | 5,546,929,495 | `12a55f09b52457ba0a5544e265d43f5d8423a8ea899bb6371ac3d61a4f7ff483` |
| `slat_flow_img2shape_dit_1_3B_1024_bf16.pt` | 5,546,930,201 | `eb0f2cb8309947d11cf200911acf9c8ac87e6b89e80e13d546952aea2c52a7e5` |
| `slat_flow_imgshape2tex_dit_1_3B_1024_bf16.pt` | 5,547,128,991 | `e3f73594e04d46a42853db5e0ba60d1b5b310b6b19e0e38a30d85139db9935a4` |

## CPU-only dataset construction

The deployed production configs were loaded on Node16 with:

```text
CUDA_VISIBLE_DEVICES=''
torch.cuda.is_available() == False
torch.cuda.is_initialized() == False
```

`flex_gemm` eagerly asks for a GPU model only to select an import-time kernel
table. The repository's production preflight pattern was followed: return the
`A100` table name only while importing dataset definitions, restore
`torch.cuda.get_device_name` immediately, and never invoke a kernel. CUDA was
uninitialized before, during, and after all dataset constructors.

| Stage | Configured dataset class | Samples |
|---|---|---:|
| `ss64` | `MultiViewImageConditionedSparseStructureLatentView` | 64 |
| `shape512` | `MultiViewImageConditionedSLatShapeView` | 64 |
| `shape1024` | `MultiViewImageConditionedSLatShapeView` | 64 |
| `pbr1024` | `MultiViewImageConditionedSLatPbrView` | 64 |

## Reviewed source deployment

Committed HEAD `23aa41dbeef33bd92323d4288510b56af7885fa2` was transferred by
`git archive --format=tar HEAD` into the isolated
`/home/youngwoo/Pixal3D-training` directory. The archive and destination were
audited for symlinks before extraction; neither contained one.

Fresh local and Node16 SHA-256 values match:

| File | SHA-256 |
|---|---|
| `train.py` | `229a53121e374ec2bd16eaee4128a060776ab39db875f8d1c83f3ee5f1bf922c` |
| `configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json` | `8874a0f457568ea475a877091a533ce036e7d94eaf0bc97c6fc57f814d4cc77c` |
| `configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json` | `60184eb23d0bbe0bf02fe2a1a07d1567f7afb5814f0bbdcef5144aa482c8ec88` |
| `configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json` | `571de55a37843100687f18d0d40f90fa8fe79fd894c27d1e7945156b3fd4ba2b` |
| `configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json` | `d0eee41c5f62600b45454ad4efd4b0106b58989d624d89f09954bf1afbd7379f` |

## Result

The isolated Node16 checkout now has verified 64-sample inputs and matching
single-view checkpoints for all four production models. The active checkout and
preprocessing workload were preserved.
