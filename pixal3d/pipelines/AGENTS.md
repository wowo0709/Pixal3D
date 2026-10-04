# Pipelines

## Overview

`Pixal3DImageTo3DPipeline` samples structure, shape and texture with pixel-projected features, then decodes mesh-with-voxel results.

## Where to Look

| Behavior | File |
|----------|------|
| Pixal3D loading, preprocessing, cascade and decode | `pixal3d_image_to_3d.py` |
| Base model loading and devices | `base.py` |
| Backbone inference and texturing alternatives | `trellis2_image_to_3d.py`, `trellis2_texturing.py` |
| Euler velocity integration and sampler outputs | `samplers/flow_euler.py` |
| CFG and inclusive guidance interval | `samplers/classifier_free_guidance_mixin.py`, `samplers/guidance_interval_mixin.py` |
| Shared feature projection | `../trainers/flow_matching/mixins/image_conditioned_proj.py` |

## Camera and Conditioning

- `run()` needs `camera_angle_x` in radians and `distance`; `mesh_scale` defaults to 1. The inference wrapper estimates these using MoGe or manual FOV.
- Pretrained loading leaves the four stage image extractors unset; wrappers attach SS, shape-512, shape-1024 and texture-1024 extractors before running.
- Single-view projection uses the fixed front-view Blender camera. Calibrated MV inputs provide one camera transform/FOV/distance per image; the first camera anchors the latent frame, and projected/global features are averaged across views. Keep prepared camera order and anchor scale together.
- SS projected features remain dense. Shape/texture conditioning gathers them at sparse `[batch,x,y,z]` coordinates and returns a `SparseTensor` aligned to the sampled latent. Negative conditioning is zero-valued.
- `ProjGrid` uses bilinear `grid_sample` with border padding; its calculated validity mask is not applied. Out-of-frame features therefore use border values.

## Cascade and Memory

- Supported Pixal3D run types are `1024_cascade` and `1536_cascade`; both require 512/1024 shape flows and the 1024 texture flow.
- `run()` implements the configured SS flow grid → shape-512 → decoder coordinate upsampling → HR shape → shape-conditioned texture inline. The separate `sample_shape_slat_cascade` helper is not the run path.
- HR quantization lowers candidate resolution by 128 until the unique-coordinate count fits `max_num_tokens` or reaches the 1024 floor. The token limit is a soft cap; use returned actual resolution for export.
- Low-VRAM `to(device)` updates `_device`; stage code moves models/extractors onto that device and offloads them afterward. Preprocessing independently moves rembg on/off device.
- PIL-list feature extraction internally uses `.cuda()`. Inspect that path when targeting a non-default CUDA device.
- Grid-resolution overrides temporarily modify projection buffers and restore them on normal completion; these shared extractors are mutable stage state.

## Outputs and Sampling

- `run()` returns a mesh list, or `(mesh_list, (shape_slat, tex_slat, actual_resolution))` with `return_latent=True`.
- Decoded `MeshWithVoxel` uses origin `[-0.5,-0.5,-0.5]`, voxel size `1/resolution`, and coordinates without their batch column. PBR channels are base color `0:3`, metallic `3`, roughness `4`, alpha `5`.
- Euler samplers return `samples`, `pred_x_t`, and `pred_x_0` in an EasyDict. Caller sampler parameters override pipeline defaults.
- CFG combines positive/negative predictions; guidance interval disables it outside the inclusive interval. Keep those sampler settings separate for SS, shape and texture.

## Shared Inference Entry Points

- `inference.py` supports calibrated multi-view manifests and stage checkpoint
  overrides. Experiment sampling and viewer comparison generation use this
  checkout and explicit external dependencies, not an archived run's source.
- Preserve calibrated image dimensions and black alpha compositing; extra image
  cropping changes the intrinsics. Use the exact prepared scale for cached anchors.
- Keep BF16 autocast local to frozen DINO/NAF. Sparse decoders use native FP16;
  applying global BF16 autocast can promote mixed-dtype residuals to FP32.
- MV result snapshots use the native Y-up anchor frame. `o_voxel` applies an axis
  conversion during GLB export; the viewer restores the frame with its X rotation.
