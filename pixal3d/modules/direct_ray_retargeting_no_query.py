"""The direct correspondence delivery with deformation-query channels absent.

Keep the full candidate readout and bounded residual delivery. The retargeter
has only appearance inputs: no RAW deformation input or extra projection is
constructed. This is the structural control for Direct+C2, rather than a
saved-weight intervention that zeros its query observations or projection.
"""
from dataclasses import replace

import torch

from .ray_feature_attention import RayFeatureAttentionConditioner


class DirectRayRetargetingNoQueryConditioner(RayFeatureAttentionConditioner):
    def forward(self, **kwargs):
        if self.g_max <= 0:
            raise ValueError('positive inherited retargeting bound required')
        chunks = []

        def capture(module, inputs, output):
            chunks.append(output[0])

        # Match the existing Direct delivery, including its true checkpointed
        # autograd output. Keep that live implementation frozen in its own file.
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
        values = kwargs['source_values'].float().flatten(1, 2)
        chosen = result.candidate_weights.argmax(-1, keepdim=True)
        address = result.candidate_indices.gather(2, chosen).squeeze(2)
        top1 = values.gather(1, address[..., None].expand(-1, -1, self.value_dim))
        top1 = top1 * valid[..., None]
        residual = result.retarget_residual + correction
        self.last_feature_attention_stats.update(
            direct_retargeted_feature_RMS=float(correction.detach().square().mean().sqrt()),
            direct_feature_bound=self.g_max, feature_correction_in_retargeting_residual=True,
            original_semantic_fusion_gate=float(result.fusion_scale.detach()),
            physical_deformation_channels=0,
            retargeter_input_channels=self.retargeter.input_dim,
            deformation_query_module_present=False)
        return replace(result, evidence=top1 + residual, retarget_residual=residual)
