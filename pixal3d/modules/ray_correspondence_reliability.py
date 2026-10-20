"""Frozen, inference-only availability confidence for a native target patch ray."""
import hashlib
import math
from pathlib import Path

import torch
from torch import nn


def reliability_features(native, context, prior, valid, addresses):
    n, k = valid.shape
    if (native.shape != (n, 1024) or context.shape != native.shape
            or prior.shape != valid.shape or addresses.shape != valid.shape
            or valid.dtype != torch.bool or k != 64):
        raise ValueError('native ray and aligned semantic Top64 inference inputs required')
    if ((addresses < 0) | (addresses >= 3072)).any():
        raise ValueError('three-source candidate addresses required')
    with torch.autocast(device_type=native.device.type, enabled=False):
        native, context, prior = native.float(), context.float(), prior.float()
        if not (torch.isfinite(native).all() and torch.isfinite(context).all()
                and torch.isfinite(prior[valid]).all()):
            raise ValueError('finite actual inference observations required')
        def normalize(x):
            centered = x-x.mean(-1, keepdim=True)
            return centered/torch.sqrt(centered.square().mean(-1, keepdim=True)+1e-5)
        masked = prior.masked_fill(~valid, -torch.inf)
        maximum = masked.amax(-1, keepdim=True)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        p = torch.exp(masked-maximum).masked_fill(~valid, 0.)
        p = p/p.sum(-1, keepdim=True).clamp_min(1e-8)
        count = valid.sum(-1)
        top = masked.topk(2, dim=-1).values
        top = torch.where((count >= 2)[:, None], top, torch.zeros_like(top))
        entropy = -(p*p.clamp_min(1e-30).log()).sum(-1)
        mean = prior.masked_fill(~valid, 0.).sum(-1)/count.clamp_min(1)
        views = torch.stack([(p*(addresses//1024 == view)).sum(-1) for view in range(3)], -1)
        extra = torch.stack([maximum[:, 0]/20., (top[:, 0]-top[:, 1])/20., mean/20.,
            entropy/math.log(64), p.amax(-1), entropy.exp()/64., count/64., views.amax(-1),
            -(views*views.clamp_min(1e-30).log()).sum(-1)/math.log(3),
            native.square().mean(-1).sqrt(), context.square().mean(-1).sqrt(),
            (views > 0).sum(-1)/3.], -1)
        return torch.cat([normalize(native), normalize(context), extra], -1)


class RayCorrespondenceReliability(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2060, 64), nn.GELU(), nn.Linear(64, 1))
        # NaNs force callers to load actual TRAIN normalization before inference.
        self.register_buffer('summary_mean', torch.full((12,), torch.nan))
        self.register_buffer('summary_scale', torch.full((12,), torch.nan))
        self.requires_grad_(False)

    def load_pilot(self, path, expected_sha256):
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError('explicit reliability checkpoint SHA256 differs')
        state = torch.load(path, map_location='cpu', weights_only=True)
        if (set(state) != {'head', 'TRAIN_summary_mean', 'TRAIN_summary_scale', 'input_width'}
                or state['input_width'] != 2060):
            raise ValueError('fixed7 native-ray reliability checkpoint required')
        self.network.load_state_dict(state['head'], strict=True)
        if (state['TRAIN_summary_mean'].shape != (12,) or state['TRAIN_summary_scale'].shape != (12,)
                or not torch.isfinite(state['TRAIN_summary_mean']).all()
                or not torch.isfinite(state['TRAIN_summary_scale']).all()
                or not (state['TRAIN_summary_scale'] >= 1e-4).all()):
            raise ValueError('finite TRAIN-only summary normalization required')
        self.summary_mean.copy_(state['TRAIN_summary_mean'])
        self.summary_scale.copy_(state['TRAIN_summary_scale'])
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, native, context, prior, valid, addresses):
        if not (torch.isfinite(self.summary_mean).all() and (self.summary_scale > 0).all()):
            raise ValueError('load the actual trained reliability head before inference')
        with torch.autocast(device_type=native.device.type, enabled=False):
            x = reliability_features(native, context, prior, valid, addresses)
            x = torch.cat([x[:, :2048], (x[:, 2048:]-self.summary_mean)/self.summary_scale], -1)
            return self.network(x)[:, 0].sigmoid()*valid.any(-1)
