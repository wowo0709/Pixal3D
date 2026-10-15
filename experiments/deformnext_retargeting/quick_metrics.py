"""Small deterministic generation panels for epoch-level trend metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch


def stratified_indices(dataset, count: int) -> list[int]:
    """Select a stable category round-robin panel from a manifest dataset."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(dataset.rows):
        groups[row.get("category", "DeformNeXt")].append(index)
    selected = []
    depth = 0
    names = sorted(groups)
    while len(selected) < min(count, len(dataset)):
        added = False
        for name in names:
            if depth < len(groups[name]):
                selected.append(groups[name][depth])
                added = True
                if len(selected) == min(count, len(dataset)):
                    break
        if not added:
            break
        depth += 1
    return selected


def binary_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction, target = prediction.bool(), target.bool()
    intersection = (prediction & target).sum().double()
    pred_count = prediction.sum().double()
    target_count = target.sum().double()
    union = (prediction | target).sum().double()
    eps = torch.finfo(torch.float64).eps
    precision = intersection / (pred_count + eps)
    recall = intersection / (target_count + eps)
    return {
        "voxel_iou": float(intersection / (union + eps)),
        "voxel_precision": float(precision),
        "voxel_recall": float(recall),
        "voxel_f1": float(2 * precision * recall / (precision + recall + eps)),
    }


def _panel_image(prediction: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    rows = []
    for pred, gt in zip(prediction, target):
        diff = (pred - gt).abs()
        rows.append(torch.cat((gt, pred, diff), dim=-1))
    panel = torch.cat(rows, dim=-2).clamp(0, 1)
    return (panel.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def _sample(self, dataset, index: int, steps: int, seed: int):
    from pixal3d.modules import sparse as sp
    from pixal3d.utils.data_utils import recursive_to_device

    pack = dataset.collate_fn([dataset[index]])
    pack = pack if isinstance(pack, dict) else pack[0]
    pack = recursive_to_device(pack, self.device)
    pack.pop("_dataset_name", None)
    pack.pop("_sha256", None)
    target = pack.pop("x_0")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    noise = (
        target.replace(torch.randn_like(target.feats))
        if isinstance(target, sp.SparseTensor) else torch.randn_like(target)
    )
    if isinstance(target, sp.SparseTensor):
        pack["coords"] = target.coords
    args = self.get_inference_cond(**pack)
    result = self.get_sampler().sample(
        self.models["denoiser"], noise=noise, **args,
        steps=steps, guidance_strength=3.0, verbose=False,
    )
    return result.samples, target, pack


def _ss_metrics(self, dataset, indices: list[int], config: dict[str, Any]):
    predictions, targets = [], []
    for panel_index, index in enumerate(indices):
        pred, target, _ = _sample(
            self, dataset, index, config["sampling_steps"],
            config.get("seed", 42) + panel_index,
        )
        predictions.append(pred)
        targets.append(target)
    count = len(predictions)
    decoded = dataset.decode_latent(torch.cat(predictions + targets, dim=0), batch_size=2)
    pred_occ, target_occ = decoded[:count, 0] > 0, decoded[count:, 0] > 0
    metrics = binary_metrics(pred_occ, target_occ)
    pred_projection = pred_occ.float().amax(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    target_projection = target_occ.float().amax(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    return metrics, _panel_image(pred_projection, target_projection)


def _render_pbr_anchor(dataset, representations, cameras, resolution: int, sampling_seed: int = 0):
    import cv2
    import utils3d
    from pixal3d.renderers import EnvMap, PbrMeshRenderer
    from pixal3d.representations import MeshWithVoxel

    renderer = PbrMeshRenderer()
    renderer.rendering_options.resolution = resolution
    renderer.rendering_options.near = 1
    renderer.rendering_options.far = 100
    renderer.rendering_options.ssaa = 1
    renderer.rendering_options.peel_layers = 4
    envmap = EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread("assets/hdri/forest.exr", cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
        dtype=torch.float32, device="cuda",
    ))
    outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
    for representation, (fov, distance, scale) in zip(representations, cameras):
        scaled = MeshWithVoxel(
            vertices=representation.vertices / scale,
            faces=representation.faces,
            origin=(representation.origin / scale).tolist(),
            voxel_size=representation.voxel_size / scale,
            coords=representation.coords,
            attrs=representation.attrs,
            voxel_shape=representation.voxel_shape,
            layout=representation.layout,
        )
        device = representation.vertices.device
        extrinsics = utils3d.torch.extrinsics_look_at(
            torch.tensor([0.0, 0.0, distance], device=device),
            torch.zeros(3, device=device),
            torch.tensor([0.0, 1.0, 0.0], device=device),
        ).to(device)
        intrinsics = utils3d.torch.intrinsics_from_fov(
            fov_x=torch.tensor(fov, device=device),
            fov_y=torch.tensor(fov, device=device),
        ).to(device)
        half_size = 0.5 / scale
        renderer.rendering_options.near = max(0.01, distance - half_size - 0.5)
        renderer.rendering_options.far = distance + half_size + 0.5
        # PbrMeshRenderer samples random ambient-occlusion rays. Compare
        # models/GT with the same rays and leave training's RNG untouched.
        with torch.cuda.device(device), torch.random.fork_rng(devices=[device]):
            torch.cuda.manual_seed(sampling_seed)
            rendered = renderer.render(scaled, extrinsics, intrinsics, envmap=envmap)
        for key in ("shaded", "base_color", "alpha", "mask"):
            outputs[key].append(rendered[key])
    return {key: torch.stack(values) for key, values in outputs.items()}


def _pbr_metrics(self, dataset, indices: list[int], config: dict[str, Any]):
    from pixal3d.modules import sparse as sp
    from pixal3d.utils.loss_utils import lpips, psnr, ssim

    predictions, targets, shapes, cameras = [], [], [], []
    for panel_index, index in enumerate(indices):
        pred, target, pack = _sample(
            self, dataset, index, config["sampling_steps"],
            config.get("seed", 42) + panel_index,
        )
        predictions.append(pred)
        targets.append(target)
        shapes.append(pack["concat_cond"])
        cameras.append((
            float(pack["camera_angle_x"][0, 0]),
            float(pack["camera_distance"][0, 0]),
            float(pack["mesh_scale"][0]),
        ))
    count = len(predictions)
    pbr = sp.sparse_cat(predictions + targets)
    shape = sp.sparse_cat(shapes + shapes)
    # The pretrained sparse VAEs mix FP32 boundary layers and FP16 torso
    # convolutions. Training BF16 autocast overrides their native casts and
    # leaves flex_gemm with mismatched convolution operands.
    with torch.autocast("cuda", enabled=False):
        representations = dataset.decode_latent(pbr.float(), shape.float(), batch_size=1)
        renders = _render_pbr_anchor(
            dataset, representations, cameras + cameras,
            config.get("render_resolution", 256),
        )
        pred, gt = renders["shaded"][:count], renders["shaded"][count:]
        pred_base, gt_base = renders["base_color"][:count], renders["base_color"][count:]
        pred_alpha, gt_alpha = renders["alpha"][:count], renders["alpha"][count:]
        foreground = (pred_alpha > 0.5) | (gt_alpha > 0.5)
        foreground = foreground.reshape(count, 1, pred.shape[-2], pred.shape[-1])
        foreground_rgb = foreground.expand_as(pred)
        masked_mse = (pred - gt).square()[foreground_rgb].mean()
        masked_base_mse = (pred_base - gt_base).square()[foreground_rgb].mean()
        crops_pred, crops_gt = [], []
        for index in range(count):
            positions = foreground[index, 0].nonzero()
            if not len(positions):
                continue
            top, left = positions.amin(0).tolist()
            bottom, right = (positions.amax(0) + 1).tolist()
            for image, crops in ((pred, crops_pred), (gt, crops_gt)):
                crop = image[index:index+1, :, top:bottom, left:right]
                crops.append(torch.nn.functional.interpolate(
                    crop, (256, 256), mode='bilinear', align_corners=False))
        if not crops_pred:
            raise ValueError('PBR validation panel has no foreground')
        crop_pred, crop_gt = torch.cat(crops_pred), torch.cat(crops_gt)
        metrics = {
            "foreground_fraction": float(foreground.float().mean()),
            "foreground_psnr": float(-10 * masked_mse.clamp_min(1e-12).log10()),
            "foreground_base_color_psnr": float(-10 * masked_base_mse.clamp_min(1e-12).log10()),
            "crop_ssim": float(ssim(crop_pred, crop_gt)),
            "crop_lpips": float(lpips(crop_pred, crop_gt)),
            "render_psnr": float(psnr(pred, gt)),
            "render_ssim": float(ssim(pred, gt)),
            "render_lpips": float(lpips(pred, gt)),
            "base_color_psnr": float(psnr(pred_base, gt_base)),
            "base_color_ssim": float(ssim(pred_base, gt_base)),
            "gt_shape_conditioned_alpha_iou": binary_metrics(pred_alpha > 0.5, gt_alpha > 0.5)["voxel_iou"],
        }
        return metrics, _panel_image(pred, gt)


@torch.no_grad()
def evaluate_quick(self, dataset, config: dict[str, Any]):
    quick = config.get("quick_metrics") or {}
    if not quick.get("enabled", False):
        return {}, None, []
    indices = stratified_indices(dataset, int(quick["panel_size"]))
    with torch.autocast("cuda", dtype=self.mix_precision_dtype):
        if config["dataset_name"] == "DeformNeXtSS64":
            metrics, panel = _ss_metrics(self, dataset, indices, quick)
        elif config["dataset_name"] == "DeformNeXtPBR1024":
            metrics, panel = _pbr_metrics(self, dataset, indices, quick)
        else:
            return {}, None, []
    return metrics, panel, [dataset.rows[index]["sample_id"] for index in indices]
