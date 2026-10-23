"""Read-only feature/R controls for a saved native PBR feature checkpoint."""
import importlib

import torch

from .pbr_source_transfer_controls import mask_source_transfer


CONTROLS = ('normal', 'feature-output-zero', 'wrong-projected-V-fixed-use',
            'R-off', 'both-source-contributions-off')


def feature_output_control(change, probability, control):
    if torch.is_grad_enabled() or control not in CONTROLS:
        raise ValueError('explicit no_grad saved feature control required')
    if change.ndim != 2 or probability.ndim != 3 or change.shape[0] != probability.shape[0]:
        raise ValueError('aligned actual feature output and head probabilities required')
    return (torch.zeros_like(change) if control == 'feature-output-zero' else change), probability


class SavedPBRFeatureEpochControl:
    """Preserve matching/QK/use; intervene on projected V, feature F, or delivery.

    Wrong V reuses the existing projected-V-only intervention. Only the
    top1-plus-feature arm is admitted: raw Top-K aggregation would need a
    separate control because the shared helper also permutes raw values there.
    """
    def __init__(self, module, control):
        if (module.training or module.appearance_mode != 'top1-plus-feature'
                or module.value_dim != 2048 or control not in CONTROLS):
            raise ValueError('saved eval PBR Top1-plus-feature model required')
        self.module, self.control = module, control
        self.versions = {n:p._version for n,p in module.named_parameters()}
        self.events, self.chunks = [], []
        self.pbr_module = importlib.import_module('pixal3d.modules.r_mediated_pbr_source')
        self.original_chunk = self.pbr_module._input_chunk
        self.pbr_module._input_chunk = self.input_chunk
        self.handle = module.register_forward_hook(self.after)

    def input_chunk(self, head, native, context, values, keys, batch, indices, valid, prior,
                    control, raw_topk):
        from .r_mediated_native_runtime import tensor_sha
        if control != 'normal' or raw_topk or torch.is_grad_enabled() or head.training:
            raise ValueError('unmodified Top1-plus-feature eval chunk required')
        selected_control = ('wrong-retrieved-values-fixed-use'
                            if self.control == 'wrong-projected-V-fixed-use' else 'normal')
        change, aggregate, probability = self.original_chunk(
            head, native, context, values, keys, batch, indices, valid, prior,
            selected_control, False)
        delivered, same_probability = feature_output_control(change, probability, self.control)
        if same_probability is not probability:
            raise ValueError('feature intervention changed attention probability object')
        entropy = -(probability * probability.clamp_min(1e-30).log()).sum(-1)
        self.chunks.append(dict(probabilities_SHA256=tensor_sha(probability),
            input_feature_SHA256=tensor_sha(change), actual_feature_SHA256=tensor_sha(delivered),
            original_feature_RMS=float(change.square().mean().sqrt()),
            delivered_feature_RMS=float(delivered.square().mean().sqrt()),
            original_matching_inputs_SHA256={k:tensor_sha(v) for k,v in dict(
                query=native, context=context, candidate_ids=indices, valid=valid, prior=prior).items()},
            effective_K_per_head=float(entropy.exp().mean()),
            max_probability_mean=float(probability.amax(-1).mean())))
        return delivered, aggregate, probability

    def after(self, module, args, output):
        from .r_mediated_native_runtime import tensor_sha
        if (module.training or torch.is_grad_enabled()
                or self.versions != {n:p._version for n,p in module.named_parameters()}):
            raise ValueError('saved module weights or execution mode changed')
        masked = self.control if self.control in {'R-off','both-source-contributions-off'} else 'normal'
        changed = mask_source_transfer(output, masked)
        att = output.attention
        if not self.chunks:
            raise ValueError('actual feature readout must execute')
        prior = att.matching_logits[att.query_valid]
        probability = prior.softmax(-1)
        entropy = -(probability * probability.clamp_min(1e-30).log()).sum(-1)
        self.events.append(dict(control=self.control,
            fixed_source_fields={k:tensor_sha(v) for k,v in dict(
                candidate_indices=att.candidate_indices, matching_logits=att.matching_logits,
                candidate_weights=att.candidate_weights, Top1_carrier=att.base_evidence,
                source_use=output.use_probability, fusion_scale=att.fusion_scale).items()},
            computed_R_SHA256=tensor_sha(att.retarget_residual),
            computed_R_RMS=float(att.retarget_residual.square().mean().sqrt()),
            learned_fusion_gate=float(att.fusion_scale),
            matching_prior_effective_K=float(entropy.exp().mean()),
            matching_prior_max_probability_mean=float(probability.amax(-1).mean()),
            chunks=self.chunks, source_weights_unchanged=True,
            actual_output_numerator_RMS=float(changed.source_numerator.square().mean().sqrt())))
        self.chunks = []
        return changed

    def close(self):
        self.handle.remove()
        if self.pbr_module._input_chunk != self.input_chunk:
            raise ValueError('diagnostic chunk ownership changed')
        self.pbr_module._input_chunk = self.original_chunk
