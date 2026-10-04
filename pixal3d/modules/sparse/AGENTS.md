# Sparse Modules

## Overview

Sparse operators share row-aligned features/coordinates, batched layouts and backend-specific spatial caches.

## Where to look

| Task | Location | Notes |
|---|---|---|
| Sparse values and variable-length sequences | `basic.py` | `SparseTensor`, `VarLenTensor`, batching and cache behavior. |
| Convolution backend choice | `config.py`, `conv/conv.py` | Runtime selection and backend dispatch. |
| Backend implementations | `conv/conv_flex_gemm.py`, `conv/conv_spconv.py`, `conv/conv_torchsparse.py` | Backend-specific kernels and coordinate rules. |
| Attention | `attention/` | Sparse attention and backend-specific attention paths. |
| Sparse transformer blocks | `transformer/` | Transformer layers preserve sparse structure. |
| Spatial resampling | `spatial/` | Down/up-sampling and spatial/channel transforms. |

## Contracts

- `SparseTensor` stores feature rows in `feats` and corresponding integer coordinates in `coords`; row counts must match. Coordinates use column 0 for batch index and subsequent columns for spatial coordinates.
- `shape` is batch size followed by feature dimensions. `spatial_shape` is derived from spatial coordinate maxima. `layout` contains contiguous row slices per batch and is used by sparse losses and attention.
- `SparseTensor.replace(feats)` retains coordinates and shares spatial caches. `.to(device, dtype)` converts features to the requested dtype and coordinates to the requested device.
- `VarLenTensor` stores concatenated feature rows and a slice layout per sequence. Attention context can be `VarLenTensor` or dense `torch.Tensor`; preserve batch ordering and lengths.
- Sparse convolution backend is selected by `SPARSE_CONV_BACKEND` (`none`, `spconv`, `torchsparse`, `flex_gemm`); attention backend uses `SPARSE_ATTN_BACKEND` or fallback `ATTN_BACKEND` (`xformers`, `flash_attn`, `flash_attn_3`, `flash_attn_4`, `sdpa`). Defaults are flex_gemm and flash_attn.
- `SparseConv3d` and `SparseInverseConv3d` import/cache implementations at construction but dispatch forward through global `config.CONV`. Set the backend before construction and keep it consistent during execution; `none` has no convolution implementation.

## Gotchas

- `replace(feats, coords)` does not invalidate spatial caches. If support/order changes, construct the appropriate new tensor and caches rather than assuming replacement rebuilds them.
- spconv and TorchSparse wrap their native sparse objects; direct assumptions about `data` representation are backend-dependent. Prefer `feats`, `coords`, and `replace()`.
- Debug assertions in `SparseTensor` check feature-coordinate row agreement, computed shape/layout, and contiguous batch groups when `SPARSE_DEBUG=1`.
- `SparseTensor` coordinates are expected grouped by batch for layout slicing; do not reorder coordinates independently of features.

## Accumulation and Memory

- Sparse packs preserve feature/coordinate order and may contain unequal asset
  counts under token balancing. Do not change loss weighting during memory tuning.
- Verify attention acceleration with profiler forward/backward events on the
  assigned GPU; backend configuration alone does not prove the executed kernel.
