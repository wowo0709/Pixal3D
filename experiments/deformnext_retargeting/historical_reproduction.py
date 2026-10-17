"""Faithful mathematical primitives from the positive historical SS studies.

Reference code is vendored here, not imported from a historical worktree.
GT routes are diagnostic-only; the four-input global/negative branches stay native.
Geometry factorial reference modes are historical controls, not the physically
channel-removed No-query arm used by the production study.
"""
from __future__ import annotations
from collections.abc import Mapping
from hashlib import sha256
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt

VIEW_IDS = (9, 10, 11)
MODES = {"r2_feat": (False, False), "r4_gate": (True, False),
         "r4_content": (False, True), "r4_both": (True, True)}


def identity_patch_grid(
    batch: int, views: int, height: int, width: int, *, device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Grid-sample coordinates for patch centers, with align_corners=False."""

    if min(batch, views, height, width) < 1:
        raise ValueError("batch, views and spatial shape must be positive")
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * 2 / height - 1
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * 2 / width - 1
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).expand(batch, views, height, width, 2).clone()


def pixel_to_patch_grid(pixel_xy: torch.Tensor, *, image_width: int, image_height: int) -> torch.Tensor:
    """Native pixel-center convention mapped to a grid_sample NDC grid."""

    if pixel_xy.shape[-1] != 2 or image_width < 1 or image_height < 1:
        raise ValueError("invalid pixel coordinates/image shape")
    scale = pixel_xy.new_tensor((2 / image_width, 2 / image_height))
    return (pixel_xy + 0.5) * scale - 1


def _sample_neighborhood(
    source: torch.Tensor, read_grid: torch.Tensor, *, radius: int,
) -> torch.Tensor:
    """Sample a fixed source-patch neighborhood around each read address."""

    batch, views, height, width, channels = source.shape
    if read_grid.shape != (batch, views, height, width, 2):
        raise ValueError("read_grid/source shape mismatch")
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    source_bv = source.reshape(batch * views, height, width, channels).permute(0, 3, 1, 2)
    base = read_grid.reshape(batch * views, height, width, 2).float()
    offsets = torch.tensor(
        [(dx * 2 / width, dy * 2 / height) for dy in range(-radius, radius + 1)
         for dx in range(-radius, radius + 1)],
        dtype=base.dtype, device=base.device,
    )
    count = offsets.shape[0]
    grid = (base[..., None, :] + offsets).reshape(batch * views, height, width * count, 2)
    sampled = F.grid_sample(
        source_bv.float(), grid, mode="bilinear", padding_mode="border", align_corners=False,
    )
    return sampled.permute(0, 2, 3, 1).reshape(batch, views, height, width, count, channels)


def project_static_source_maps(
    native_conditioner, source_maps: torch.Tensor,
    *, camera_angle_x: torch.Tensor, distance: torch.Tensor,
    mesh_scale: torch.Tensor, projection_matrices: torch.Tensor,
    write_valid: torch.Tensor | None = None,
    mask_camera_fov: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project selected feature maps and their ray validity with native ProjGrid.

    Returns [B,V,16^3,C] and [B,V,16^3,1].  The latter permits target-only
    fallback at invalid rays without inserting zero feature values into the
    flow condition.  The selected source projection is never directly written
    at a voxel address.
    """

    if source_maps.ndim != 5:
        raise ValueError("source_maps must be [B,V,H,W,C]")
    batch, views, height, width, _ = source_maps.shape
    if camera_angle_x.shape != (batch, views) or distance.shape != (batch, views):
        raise ValueError("camera vectors mismatch")
    if projection_matrices.shape != (batch, views, 4, 4) or mesh_scale.shape != (batch,):
        raise ValueError("camera/mesh scale mismatch")
    if write_valid is None:
        write_valid = torch.ones((batch, views, height, width), device=source_maps.device, dtype=torch.bool)
    projected, projected_valid = [], []
    for view in range(views):
        feature = native_conditioner.proj_grid(
            source_maps[:, view], camera_angle_x[:, view], distance[:, view], mesh_scale,
            projection_matrices[:, view],
        )
        valid_feature = native_conditioner.proj_grid(
            write_valid[:, view, ..., None].to(source_maps.dtype),
            camera_angle_x[:, view], distance[:, view], mesh_scale,
            projection_matrices[:, view],
        ).clamp(0, 1)
        if mask_camera_fov:
            from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
                project_points_to_image_batch,
            )
            grid = native_conditioner.proj_grid
            points = grid.grid_points.expand(batch, -1, -1)
            points = points / mesh_scale[:, None, None] / 2
            _pixel, _depth, camera_valid = project_points_to_image_batch(
                points, projection_matrices[:, view], camera_angle_x[:, view], grid.image_resolution,
            )
            camera_valid = camera_valid[..., None]
            feature = feature * camera_valid
            valid_feature = valid_feature * camera_valid
        projected.append(feature)
        projected_valid.append(valid_feature)
    return torch.stack(projected, dim=1), torch.stack(projected_valid, dim=1)


def target_first_static_projection(
    target_projection: torch.Tensor, source_projection: torch.Tensor,
    ray_validity: torch.Tensor, *, source_weight: float = 1.0,
) -> torch.Tensor:
    """Native four-view mean at identity; target fallback for missing rays.

    ``source_projection`` is the projection of a map already zeroed at
    invalid write rays.  It is therefore a *weighted numerator*, not a
    normalized conditional mean.  Multiplying it by validity again would
    square the soft bilinear mask at ray boundaries.
    """

    if source_projection.ndim != 4 or source_projection.shape[:1] != target_projection.shape[:1]:
        raise ValueError("source projection must have view axis")
    if source_projection.shape[2:] != target_projection.shape[1:]:
        raise ValueError("source/target projection shape mismatch")
    if ray_validity.shape != source_projection.shape[:-1] + (1,):
        raise ValueError("ray validity shape mismatch")
    if not 0 <= source_weight <= 1:
        raise ValueError("source_weight must lie in [0,1]")
    views = source_projection.shape[1]
    contribution = source_projection.float() - ray_validity.float() * target_projection[:, None].float()
    result = target_projection.float() + float(source_weight) * contribution.sum(dim=1) / (views + 1)
    return result.to(target_projection.dtype)


def route_tensors(route: dict, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return native grid_sample read locations and common source-on write mask."""
    source_pixel = np.asarray(route["source_pixel"], dtype=np.float32)
    valid = np.asarray(route["valid"], dtype=bool)
    if source_pixel.shape != (3, 1024, 2) or valid.shape != (3, 1024):
        raise ValueError("expected three full 32x32 source-camera routes")
    if not np.isfinite(source_pixel[valid]).all():
        raise ValueError("valid source read pixel is nonfinite")
    pixel = np.nan_to_num(source_pixel, nan=0.0, posinf=0.0, neginf=0.0)
    grid = pixel_to_patch_grid(
        torch.from_numpy(pixel).to(device), image_width=512, image_height=512,
    ).reshape(1, 3, 32, 32, 2)
    mask = torch.from_numpy(valid).to(device).reshape(1, 3, 32, 32)
    if not bool(torch.isfinite(grid).all()):
        raise ValueError("route grid is nonfinite")
    return grid, mask


def wrong_permutation(valid: torch.Tensor, *, pair_id: str) -> torch.Tensor:
    """Derange active read rays inside each pair/view without changing the multiset."""
    if valid.shape != (1, 3, 32, 32) or valid.dtype != torch.bool:
        raise ValueError("invalid write mask")
    permutation = torch.arange(1024, device=valid.device).reshape(1, 1, 32, 32).expand(
        1, 3, 32, 32,
    ).clone()
    for view in range(3):
        ids = valid[0, view].flatten().nonzero(as_tuple=False).flatten()
        count = ids.numel()
        if count < 2:
            raise ValueError(f"wrong control requires two active rays at view {VIEW_IDS[view]}")
        seed = int.from_bytes(
            sha256(f"{pair_id}|{view}|gt-corr-wrong".encode()).digest()[:8], "big",
        )
        order = torch.randperm(count, generator=torch.Generator(device="cpu").manual_seed(seed))
        order = order.to(ids.device)
        target = ids[order]
        donor = ids[order.roll(1)]
        permutation[0, view].flatten()[target] = donor
        if bool((target == donor).any()):
            raise RuntimeError("wrong route has fixed active slots")
    return permutation


def geometry_from_logits(original64: np.ndarray, desired64: np.ndarray) -> torch.Tensor:
    """Return [1,4096,9] O/D predicted geometry in native x/y/z order."""

    def branch(value: np.ndarray) -> torch.Tensor:
        if value.shape != (64, 64, 64) or not np.isfinite(value).all():
            raise ValueError("predicted SS logits must be finite [64,64,64]")
        support = value > 0
        if not support.any() or support.all():
            raise ValueError("predicted support is empty or full; SDF undefined")
        logit = torch.from_numpy(np.asarray(value, dtype=np.float32))[None, None]
        logit_channel = F.avg_pool3d(logit.clamp(-8, 8) / 8, 4, 4)
        probability = F.avg_pool3d(logit.sigmoid(), 4, 4)
        distance = distance_transform_edt(~support) - distance_transform_edt(support)
        sdf = torch.from_numpy(np.asarray(distance, dtype=np.float32))[None, None]
        sdf_channel = F.avg_pool3d(sdf.clamp(-16, 16) / 16, 4, 4)
        return torch.cat((logit_channel, probability, sdf_channel), dim=1)

    desired = branch(desired64)
    original = branch(original64)
    geometry = torch.cat((desired, original, desired - original), dim=1)
    result = geometry.permute(0, 2, 3, 4, 1).contiguous().reshape(1, 4096, 9)
    if not torch.isfinite(result).all():
        raise RuntimeError("predicted geometry contains nonfinite features")
    return result


class GeometryFactorialVoxelAdapter(nn.Module):
    """Shared-parameter feature, gate, content and gate+content variants."""

    def __init__(self, channels: int = 1024, hidden: int = 128, geometry_hidden: int = 32):
        super().__init__()
        self.channels = channels
        self.content_encoder = nn.Sequential(
            nn.LayerNorm(9), nn.Linear(9, geometry_hidden), nn.SiLU(),
            nn.Linear(geometry_hidden, geometry_hidden),
        )
        self.gate_encoder = nn.Sequential(
            nn.LayerNorm(9), nn.Linear(9, geometry_hidden), nn.SiLU(),
            nn.Linear(geometry_hidden, 1),
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(3 * channels + 3 + geometry_hidden),
            nn.Linear(3 * channels + 3 + geometry_hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, channels + 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    @staticmethod
    def coordinates(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        axis = torch.linspace(-1, 1, 16, device=device, dtype=dtype)
        return torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(1, 4096, 3)

    def forward(self, target_proj: torch.Tensor, mismatch_proj: torch.Tensor,
                geometry: torch.Tensor, *, mode: str):
        if mode not in MODES:
            raise ValueError(f"unknown factorial arm: {mode}")
        if (target_proj.shape != mismatch_proj.shape or target_proj.ndim != 3
                or target_proj.shape[1:] != (4096, self.channels)):
            raise ValueError("expected matching native [B,4096,C] SS projections")
        if geometry.shape != (target_proj.shape[0], 4096, 9):
            raise ValueError("expected matching predicted [B,4096,9] geometry")
        target = target_proj.float()
        mismatch = mismatch_proj.float()
        geo = geometry.to(device=target.device, dtype=torch.float32)
        gate_enabled, content_enabled = MODES[mode]
        gate_input = geo if gate_enabled else torch.zeros_like(geo)
        content_input = geo if content_enabled else torch.zeros_like(geo)
        gate = self.gate_encoder(gate_input).sigmoid()
        content = self.content_encoder(content_input)
        difference = mismatch - target
        xyz = self.coordinates(device=target.device, dtype=target.dtype).expand(target.shape[0], -1, -1)
        raw = self.residual(torch.cat((target, mismatch, difference, xyz, content), -1))
        source_gate = raw[..., :1].tanh()
        residual = raw[..., 1:].tanh()
        scale = mismatch.detach().square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-4)
        base_correction = source_gate * difference + scale * residual
        correction = gate * base_correction
        projected = mismatch_proj + correction.to(mismatch_proj.dtype)
        return projected, {"geometry_gate": gate, "source_gate": source_gate,
                           "base_correction": base_correction, "correction": correction}


def differentiable_flow_euler_samples(
    sampler,
    model,
    noise,
    condition,
    *,
    steps: int,
    rescale_t: float,
    checkpoint_model: bool = False,
    **inference_kwargs,
):
    """Unroll the native Euler update without the sampler's no-grad wrapper."""
    import numpy as np
    import torch

    if not isinstance(noise, torch.Tensor) or not noise.is_floating_point() or not torch.isfinite(noise).all():
        raise ValueError("noise must be a finite floating tensor")
    if not isinstance(condition, Mapping):
        raise TypeError("condition must be a mapping")
    if int(steps) < 1 or not np.isfinite(rescale_t) or float(rescale_t) <= 0:
        raise ValueError("Euler schedule requires positive steps and rescale_t")
    t_seq = np.linspace(1, 0, int(steps) + 1)
    t_seq = float(rescale_t) * t_seq / (1 + (float(rescale_t) - 1) * t_seq)
    sample = noise
    for t, t_prev in zip(t_seq[:-1], t_seq[1:]):
        if checkpoint_model:
            import torch.utils.checkpoint

            positive = condition.get("cond")
            negative = condition.get("neg_cond")
            if not isinstance(positive, Mapping) or set(positive) != {"global", "proj"}:
                raise ValueError("checkpointed condition requires dense global/proj tensors")
            tensors = [sample, positive["global"], positive["proj"]]
            has_negative = isinstance(negative, Mapping) and set(negative) == {"global", "proj"}
            if has_negative:
                tensors.extend([negative["global"], negative["proj"]])

            def _velocity(x_t, pos_global, pos_proj, *neg_values, _time=float(t)):
                local_condition = {"cond": {"global": pos_global, "proj": pos_proj}}
                if neg_values:
                    local_condition["neg_cond"] = {"global": neg_values[0], "proj": neg_values[1]}
                return sampler._get_model_prediction(
                    model, x_t, _time, **local_condition, **inference_kwargs,
                )[2]

            velocity = torch.utils.checkpoint.checkpoint(
                _velocity, *tensors, use_reentrant=False,
            )
        else:
            _x0, _eps, velocity = sampler._get_model_prediction(
                model, sample, float(t), **condition, **inference_kwargs,
            )
        sample = sample - float(t - t_prev) * velocity
    return sample


def pool_ss_logits(logits, *, resolution: int):
    """Pool native decoder logits while preserving thresholded OR semantics."""
    import torch
    import torch.nn.functional as F

    if not isinstance(logits, torch.Tensor) or logits.ndim != 5 or logits.shape[0] != 1 or logits.shape[1] != 1:
        raise ValueError("SS logits must have shape [1,1,R,R,R]")
    native_resolution = int(logits.shape[2])
    if logits.shape[3:] != (native_resolution, native_resolution):
        raise ValueError("SS logits must be cubic")
    if int(resolution) < 1 or native_resolution % int(resolution) != 0:
        raise ValueError("requested resolution must evenly divide native resolution")
    ratio = native_resolution // int(resolution)
    return logits if ratio == 1 else F.max_pool3d(logits, ratio, ratio, 0)


def coords_to_mask(coords, resolution: int, *, device=None):
    mask = torch.zeros((1, 1, resolution, resolution, resolution), dtype=torch.bool, device=device)
    index = torch.as_tensor(np.asarray(coords, dtype=np.int64), device=device)
    if index.numel():
        if index.ndim != 2 or index.shape[-1] != 3 or bool((index < 0).any()) or bool((index >= resolution).any()):
            raise ValueError("invalid SS coordinates")
        mask[0, 0, index[:, 0], index[:, 1], index[:, 2]] = True
    return mask


def make_loss_regions(authority: dict, matched: dict, *, device=None):
    """GT/CAGE masks: supervision and evaluator only, never query inputs."""
    target64 = coords_to_mask(authority["target64"], 64, device=device)
    free64 = coords_to_mask(authority["free64"], 64, device=device)
    surface64 = coords_to_mask(authority["surface64"], 64, device=device) & target64
    source64 = coords_to_mask(authority["source64"], 64, device=device)
    src64 = coords_to_mask(matched["matched_source_exclusive64"], 64, device=device) & target64
    stale64 = coords_to_mask(matched["stale_visible64"], 64, device=device) & source64 & ~target64
    target32 = F.max_pool3d(target64.float(), 2, 2).bool()
    free32 = (-F.max_pool3d(-free64.float(), 2, 2) > 0.5)
    surface32 = F.max_pool3d(surface64.float(), 2, 2).bool() & target32
    src32 = coords_to_mask(matched["matched_source_exclusive32"], 32, device=device) & target32
    stale32 = coords_to_mask(matched["stale_visible32"], 32, device=device) & ~target32
    for depth, source, stale, positive, free, surface in (
        (64, src64, stale64, target64, free64, surface64),
        (32, src32, stale32, target32, free32, surface32),
    ):
        if bool((source & ~positive).any()) or bool((stale & positive).any()):
            raise ValueError(f"source/stale region contradicts target SS{depth}")
        if bool((free & surface).any()):
            raise ValueError(f"visible FREE/SURFACE overlap at SS{depth}")
    return {
        64: {"source_positive": src64, "stale": stale64,
             "visible_positive": surface64, "visible_negative": free64},
        32: {"source_positive": src32, "stale": stale32,
             "visible_positive": surface32, "visible_negative": free32},
    }


def locked_endpoint_loss(logits64: torch.Tensor, regions: dict):
    """Exactly 0.5*(source+stale+balanced-visible) at each resolution."""
    if logits64.shape != (1, 1, 64, 64, 64):
        raise ValueError("pre-threshold logits must be [1,1,64,64,64]")
    logits = {64: logits64.float(), 32: F.max_pool3d(logits64.float(), 2, 2)}

    def region_bce(value, mask, label):
        if not bool(mask.any()):
            return value.sum() * 0
        return F.binary_cross_entropy_with_logits(value[mask], torch.full_like(value[mask], label))

    terms = {}
    total = logits64.float().sum() * 0
    for resolution in (64, 32):
        value = logits[resolution]
        masks = regions[resolution]
        source = region_bce(value, masks["source_positive"], 1.0)
        stale = region_bce(value, masks["stale"], 0.0)
        visible = (0.5 * region_bce(value, masks["visible_positive"], 1.0)
                   + 0.5 * region_bce(value, masks["visible_negative"], 0.0))
        terms[resolution] = {"source_positive": source, "stale": stale, "visible": visible}
        total = total + 0.5 * (source + stale + visible)
    return total, terms
