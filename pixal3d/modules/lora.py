"""LoRA on the six attention/projection linears of each Pixal3D flow block."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Linear):
    """Frozen base linear plus FP32 adapters; [..., Cin] -> [..., Cout]."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        # Reuse the base parameters, without allocating another full weight.
        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight = base.weight.requires_grad_(False)
        self.bias = base.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features,
                                              device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank,
                                              device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)  # [..., Cout]
        h = F.linear(self.dropout(x).to(self.lora_A.dtype), self.lora_A)  # [..., rank]
        delta = F.linear(h, self.lora_B)  # [..., Cout]; AMP controls compute dtype
        return base + delta.to(base.dtype) * self.scaling


def inject_lora(model: nn.Module, rank: int, alpha: float, dropout: float = 0.0):
    """Freeze a flow denoiser and adapt self/global cross-attention and proj_linear."""
    model.requires_grad_(False)
    for block in model.blocks:
        cross = block.cross_attn.cross_attn_block
        targets = [(block.self_attn, 'to_qkv'), (block.self_attn, 'to_out'),
                   (cross, 'to_q'), (cross, 'to_kv'), (cross, 'to_out'),
                   (block.cross_attn, 'proj_linear')]
        for parent, name in targets:
            setattr(parent, name, LoRALinear(getattr(parent, name), rank, alpha, dropout))
