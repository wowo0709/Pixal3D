"""G2 content retargeting with a physically absent No-query input branch.

The historical-network option retains the original R4-CONTENT gate and its
optimizer parameterization. The scalar option preserves earlier pilot files;
it is a functional simplification, not the original training algorithm.
The ablated path removes the query encoder and its residual input columns.
"""
from __future__ import annotations

import torch
from torch import nn

from .historical_reproduction import GeometryFactorialVoxelAdapter


class GeometryContentRetargetAdapter(nn.Module):
    def __init__(self, *, with_query: bool, channels=1024, hidden=128, geometry_hidden=32,
                 gate_parameterization='scalar'):
        super().__init__()
        self.with_query = bool(with_query)
        self.channels = channels
        self.geometry_hidden = geometry_hidden if self.with_query else 0
        if gate_parameterization not in {'scalar','historical-network'}:
            raise ValueError('explicit scalar or historical-network gate required')
        self.gate_parameterization=gate_parameterization
        self.content_encoder = (nn.Sequential(nn.LayerNorm(9), nn.Linear(9, geometry_hidden),
            nn.SiLU(), nn.Linear(geometry_hidden, geometry_hidden)) if self.with_query else None)
        if gate_parameterization=='scalar':
            self.gate_logit = nn.Parameter(torch.zeros(()))
            self.constant_gate_network=None
        else:
            self.register_parameter('gate_logit',None)
            self.constant_gate_network=nn.Sequential(nn.LayerNorm(9),nn.Linear(9,geometry_hidden),
                nn.SiLU(),nn.Linear(geometry_hidden,1))
        width = 3 * channels + 3 + self.geometry_hidden
        self.residual = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden),
                                      nn.SiLU(), nn.Linear(hidden, channels + 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    @classmethod
    def from_historical_content(cls, original: GeometryFactorialVoxelAdapter, *, gate_parameterization='historical-network'):
        """Import a trained R4-CONTENT function exactly, never its zero-input arm.

        Keep the entire learned constant network by default. The explicit
        scalar conversion only preserves its inference function, not future
        optimization of the differently parameterized gate.
        """
        first = original.residual[1]
        geometry_hidden = original.content_encoder[1].out_features
        result = cls(with_query=True, channels=original.channels, hidden=first.out_features,
                     geometry_hidden=geometry_hidden,gate_parameterization=gate_parameterization)
        device = next(original.parameters()).device
        result = result.to(device)
        result.content_encoder.load_state_dict(original.content_encoder.state_dict(), strict=True)
        result.residual.load_state_dict(original.residual.state_dict(), strict=True)
        if gate_parameterization=='historical-network':
            result.constant_gate_network.load_state_dict(original.gate_encoder.state_dict(),strict=True)
        else:
            with torch.no_grad():
                result.gate_logit.copy_(original.gate_encoder(torch.zeros(1, 9, device=device)).reshape(()))
        return result

    @classmethod
    def initialized_from_reference(cls, *, with_query, channels=1024, hidden=128, geometry_hidden=32):
        """Keep original gate initialization/optimization; physically remove No-query.

        Both arms start with the same reference common weights. The No-query
        branch removes content encoder, LayerNorm entries and Linear columns.
        Constructing the destination must not consume extra initialization RNG.
        """
        reference=GeometryFactorialVoxelAdapter(channels=channels,hidden=hidden,geometry_hidden=geometry_hidden)
        with torch.random.fork_rng(devices=[]):
            result=cls(with_query=with_query,channels=channels,hidden=hidden,geometry_hidden=geometry_hidden,
                       gate_parameterization='historical-network')
        result.constant_gate_network.load_state_dict(reference.gate_encoder.state_dict(),strict=True)
        if with_query:result.content_encoder.load_state_dict(reference.content_encoder.state_dict(),strict=True)
        state=reference.residual.state_dict()
        if not with_query:
            width=3*channels+3
            state={**state,'0.weight':state['0.weight'][:width],'0.bias':state['0.bias'][:width],
                   '1.weight':state['1.weight'][:,:width]}
        result.residual.load_state_dict(state,strict=True)
        return result

    @classmethod
    def for_training(cls, *, with_query, gate_parameterization):
        if gate_parameterization=='historical-network':
            return cls.initialized_from_reference(with_query=with_query)
        return cls(with_query=with_query,gate_parameterization=gate_parameterization)

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
        # This is the original R4's geometry-independent learned constant gate.
        # It receives no caller geometry, in either arm. No-query still has no
        # content/query encoder or query columns in the residual's input.
        gate=(self.gate_logit.sigmoid() if self.constant_gate_network is None else
              self.constant_gate_network(target.new_zeros((*target.shape[:2],9))).sigmoid())
        difference = mismatch - target
        xyz = GeometryFactorialVoxelAdapter.coordinates(device=target.device, dtype=target.dtype)
        features = [target, mismatch, difference, xyz.expand(len(target), -1, -1)]
        if self.with_query:
            features.append(self.content_encoder(geometry.to(device=target.device, dtype=torch.float32)))
        raw = self.residual(torch.cat(features, -1))
        source_gate, residual = raw[..., :1].tanh(), raw[..., 1:].tanh()
        scale = mismatch.detach().square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-4)
        correction = gate * (source_gate * difference + scale * residual)
        return mismatch_proj + correction.to(mismatch_proj.dtype), {
            'correction': correction, 'source_gate': source_gate,
            'constant_gate':gate,
            'geometry_channels': 9 if self.with_query else 0,
            'content_channels': self.geometry_hidden,
        }
