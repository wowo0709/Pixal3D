"""G2 content retargeting with a physically absent No-query input branch.

The enabled path represents the original R4-CONTENT function. Its geometry-
independent gate is written as one scalar, rather than a network fed zeros.
The ablated path removes the encoder and its residual input channels entirely.
"""
from __future__ import annotations

import torch
from torch import nn

from .historical_reproduction import GeometryFactorialVoxelAdapter


class GeometryContentRetargetAdapter(nn.Module):
    def __init__(self, *, with_query: bool, channels=1024, hidden=128, geometry_hidden=32):
        super().__init__()
        self.with_query = bool(with_query)
        self.channels = channels
        self.geometry_hidden = geometry_hidden if self.with_query else 0
        self.content_encoder = (nn.Sequential(nn.LayerNorm(9), nn.Linear(9, geometry_hidden),
            nn.SiLU(), nn.Linear(geometry_hidden, geometry_hidden)) if self.with_query else None)
        self.gate_logit = nn.Parameter(torch.zeros(()))
        width = 3 * channels + 3 + self.geometry_hidden
        self.residual = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden),
                                      nn.SiLU(), nn.Linear(hidden, channels + 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    @classmethod
    def from_historical_content(cls, original: GeometryFactorialVoxelAdapter):
        """Import a trained R4-CONTENT function exactly, never its zero-input arm.

        The historical R4-CONTENT gate saw zeros for every voxel, so it is a
        learned constant. Evaluating that constant once preserves inference.
        This is a functional checkpoint conversion, not identical optimization
        of the differently parameterized constant gate during future training.
        """
        first = original.residual[1]
        geometry_hidden = original.content_encoder[1].out_features
        result = cls(with_query=True, channels=original.channels, hidden=first.out_features,
                     geometry_hidden=geometry_hidden)
        device = next(original.parameters()).device
        result = result.to(device)
        result.content_encoder.load_state_dict(original.content_encoder.state_dict(), strict=True)
        result.residual.load_state_dict(original.residual.state_dict(), strict=True)
        with torch.no_grad():
            result.gate_logit.copy_(original.gate_encoder(torch.zeros(1, 9, device=device)).reshape(()))
        return result

    def forward(self, target_proj, mismatch_proj, geometry=None):
        if (target_proj.shape != mismatch_proj.shape or target_proj.ndim != 3
                or target_proj.shape[1:] != (4096, self.channels)):
            raise ValueError('expected native SS projections [B,4096,C]')
        if self.with_query:
            if geometry is None or geometry.shape != (*target_proj.shape[:2], 9):
                raise ValueError('query-enabled content requires predicted O/D [B,4096,9]')
        elif geometry is not None:
            raise ValueError('No-query physically has no geometry input; do not pass zero channels')
        target, mismatch = target_proj.float(), mismatch_proj.float()
        difference = mismatch - target
        xyz = GeometryFactorialVoxelAdapter.coordinates(device=target.device, dtype=target.dtype)
        features = [target, mismatch, difference, xyz.expand(len(target), -1, -1)]
        if self.with_query:
            features.append(self.content_encoder(geometry.to(device=target.device, dtype=torch.float32)))
        raw = self.residual(torch.cat(features, -1))
        source_gate, residual = raw[..., :1].tanh(), raw[..., 1:].tanh()
        scale = mismatch.detach().square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-4)
        correction = self.gate_logit.sigmoid() * (source_gate * difference + scale * residual)
        return mismatch_proj + correction.to(mismatch_proj.dtype), {
            'correction': correction, 'source_gate': source_gate,
            'geometry_channels': 9 if self.with_query else 0,
            'content_channels': self.geometry_hidden,
        }
