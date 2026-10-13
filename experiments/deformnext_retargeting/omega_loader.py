"""Exact epoch-7 VGGT-Omega LoRA reconstruction used by cache builders."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = float(alpha) / int(rank)
        # Attention reads these Linear attributes directly when reshaping QKV.
        self.in_features = base.in_features
        self.out_features = base.out_features
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = nn.Linear(base.in_features, rank, bias=False, **factory)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False, **factory)

    def forward(self, value):
        return self.base(value) + self.scale * self.lora_b(self.lora_a(value))


def _inject(aggregator, family: str, rank: int, alpha: float):
    modules = {}
    families = ("frame_blocks", "inter_frame_blocks") if family == "attention" else ("frame_blocks",)
    projections = ("qkv", "proj") if family == "attention" else ("fc1", "fc2")
    container = "attn" if family == "attention" else "mlp"
    for family_name in families:
        for index, block in enumerate(getattr(aggregator, family_name)):
            parent = getattr(block, container)
            for projection_name in projections:
                base = getattr(parent, projection_name)
                wrapped = LoRALinear(base, rank, alpha)
                setattr(parent, projection_name, wrapped)
                modules[f"aggregator.{family_name}.{index}.{container}.{projection_name}"] = wrapped
    return modules


def _load_lora(modules, state):
    expected = {
        f"{name}.{suffix}" for name in modules
        for suffix in ("lora_a.weight", "lora_b.weight")
    }
    if set(state) != expected:
        raise RuntimeError(f"epoch-7 LoRA key mismatch: missing={len(expected-set(state))} extra={len(set(state)-expected)}")
    with torch.no_grad():
        for name, module in modules.items():
            module.lora_a.weight.copy_(state[f"{name}.lora_a.weight"])
            module.lora_b.weight.copy_(state[f"{name}.lora_b.weight"])


def load_epoch7(vggt_root: Path, base_checkpoint: Path, epoch7_checkpoint: Path, device="cuda"):
    sys.path.insert(0, str(vggt_root))
    from vggt_omega.models.vggt_omega import VGGTOmega

    # The released base checkpoint contains the camera and dense-head weights.
    # Keep those modules while reconstructing the model so the load stays exact;
    # cache generation below only executes the aggregator.
    model = VGGTOmega(enable_alignment=False)
    model.load_state_dict(torch.load(base_checkpoint, map_location="cpu", weights_only=True), strict=True)
    checkpoint = torch.load(epoch7_checkpoint, map_location="cpu", weights_only=False)
    attention = _inject(model.aggregator, "attention", rank=64, alpha=128.0)
    frame_ffn = _inject(model.aggregator, "ffn", rank=32, alpha=64.0)
    _load_lora(attention, checkpoint["lora"])
    _load_lora(frame_ffn, checkpoint["frame_ffn_lora"])
    model.requires_grad_(False).eval().to(device)
    return model, checkpoint


@torch.no_grad()
def layer12_patch_tokens(model, images: torch.Tensor) -> torch.Tensor:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast(device_type="cuda", dtype=dtype):
        layers, start = model.aggregator(images)
    layer12 = layers[11]
    if layer12 is None:
        raise RuntimeError("Omega L12 was not cached")
    return layer12[:, :, start:].float()
