"""Deliver transformed ray correspondence through the retargeting residual.

The native consumer scales semantic replacement by its learned fusion gate,
but adds the retargeting residual separately. Preserve that native contract,
the positive Top1 carrier, and the exact zero-output initial function.
"""
from dataclasses import replace

import torch

from .ray_deformation_query import RayDeformationQueryConditioner


class DirectRayRetargetingConditioner(RayDeformationQueryConditioner):
    def forward(self, **kwargs):
        if self.g_max <= 0:
            raise ValueError('positive inherited retargeting bound required')
        chunks = []

        def capture(module, inputs, output):
            chunks.append(output[0])

        # Non-reentrant checkpointing retains the true autograd graph. Capture
        # the actual head outputs instead of reconstructing a small residual by
        # subtracting large appearance vectors (which breaks exact initial0).
        handle = self.ray_readout.register_forward_hook(capture)
        try:
            result = super().forward(**kwargs)
        finally:
            handle.remove()
        valid = result.query_valid
        selected = valid.flatten().nonzero().flatten()
        delta = result.evidence.new_zeros(valid.numel(), self.value_dim)
        if chunks:
            delta = delta.index_copy(0, selected, torch.cat(chunks))
        elif len(selected):
            raise ValueError('active rays did not execute the correspondence head')
        delta = delta.reshape(*valid.shape, self.value_dim)
        correction = self.g_max * torch.tanh(delta / self.g_max)
        # This is a feature correction, not an unbounded source replacement.
        # The semantic replacement retains the original Top1 carrier exactly.
        values = kwargs['source_values'].float().flatten(1, 2)
        chosen = result.candidate_weights.argmax(-1, keepdim=True)
        address = result.candidate_indices.gather(2, chosen).squeeze(2)
        top1 = values.gather(1, address[..., None].expand(-1, -1, self.value_dim))
        top1 = top1 * valid[..., None]
        residual = result.retarget_residual + correction
        self.last_feature_attention_stats.update(
            direct_retargeted_feature_RMS=float(correction.detach().square().mean().sqrt()),
            direct_feature_bound=self.g_max, feature_correction_in_retargeting_residual=True,
            original_semantic_fusion_gate=float(result.fusion_scale.detach()))
        return replace(result, evidence=top1 + residual, retarget_residual=residual)
