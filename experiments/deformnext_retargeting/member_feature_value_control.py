"""Saved-weight projected-V intervention with actual same-forward checks."""
from contextlib import contextmanager
import math

import torch

from pixal3d.modules.member_feature_source_ray import MemberFeatureSourceRayConditioner


@contextmanager
def feature_value_control(module, events):
    if not isinstance(module, MemberFeatureSourceRayConditioner):
        raise ValueError('explicit four-head feature conditioner required')
    original, previous = module.forward, module.__dict__.get('forward')
    captured, gate_inputs, chunks = {}, [], []
    handles = []

    def capture(name):
        def hook(layer, args, output):
            captured[name] = output[2] if name == 'score' else output
        return hook

    def verify_weights(head, args, output):
        # Reconstruct the unchanged probability path from this forward's
        # actual Q/K/score tensors. No second matching/ray-pooling pass.
        valid = args[4]
        rows, candidates = valid.shape
        with torch.autocast(device_type=valid.device.type, enabled=False):
            q = captured['query'].reshape(rows, 4, 64)
            key = captured['key'].reshape(rows, candidates, 4, 64).transpose(1, 2)
            logits = (q.float()[:, :, None]*key.float()).sum(-1)/math.sqrt(64)
            logits = (logits+captured['score'][:, None].float()).masked_fill(~valid[:, None], -torch.inf)
            maximum = logits.amax(-1, keepdim=True)
            maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
            probability = torch.exp(logits-maximum).masked_fill(~valid[:, None], 0.)
            probability = probability/probability.sum(-1, keepdim=True).clamp_min(1e-8)
        if not torch.equal(output[1], probability):
            raise ValueError('projected-V intervention changed original four-head weights')
        chunks.append(dict(queries=rows, all_four_head_weights_exact_same_forward=True))
        captured.clear()

    def capture_gate(layer, args):
        gate_inputs.append(args)

    def controlled(**kwargs):
        if module.training or torch.is_grad_enabled() or kwargs.get('control', 'normal') != 'normal':
            raise ValueError('single saved-weight feature intervention requires eval/no_grad/normal input')
        captured.clear(); gate_inputs.clear(); chunks.clear()
        try:
            result = original(**dict(kwargs, control='wrong-feature-values-fixed-use'))
            if len(gate_inputs) != 1:
                raise ValueError('exactly one original source-use decision required')
            target, top1, valid = gate_inputs[0]
            if (not torch.equal(top1, result.attention.base_evidence)
                    or not torch.equal(valid, result.attention.query_valid)
                    or not torch.equal(result.use_probability, module.evidence_use.forward(target, top1, valid))):
                raise ValueError('original same-forward Top1/source-use decision changed')
            address = result.attention.candidate_indices.gather(
                -1, result.attention.candidate_weights.argmax(-1, keepdim=True)).squeeze(-1)
            expected = kwargs['source_values'].float().flatten(1, 2).gather(
                1, address[..., None].expand(-1, -1, 1024))*valid[..., None]
            if not torch.equal(top1, expected) or sum(row['queries'] for row in chunks) != int(valid.sum()):
                raise ValueError('actual Top1 carrier or four-head row coverage changed')
            events.append(dict(control='wrong-feature-values-fixed-use',
                original_matching_logits_and_attention_weights_exact=True,
                source_use_probability_exact_same_forward=True,
                original_Top1_feature_carrier_exact=True,
                projected_V_only=True, same_forward_reference=True,
                separate_matching_or_ray_pooling_reference_forward=False,
                queries=int(valid.sum()), chunks=list(chunks)))
            return result
        finally:
            captured.clear(); gate_inputs.clear(); chunks.clear()

    try:
        for name, layer in [('query', module.ray_readout.query), ('key', module.ray_readout.key),
                            ('score', module.ray_readout.score_selector)]:
            handles.append(layer.register_forward_hook(capture(name)))
        handles.append(module.ray_readout.register_forward_hook(verify_weights))
        handles.append(module.evidence_use.register_forward_pre_hook(capture_gate))
        module.forward = controlled
        yield
    finally:
        for handle in handles:
            handle.remove()
        if previous is None:
            module.__dict__.pop('forward', None)
        else:
            module.forward = previous
        captured.clear(); gate_inputs.clear(); chunks.clear()
