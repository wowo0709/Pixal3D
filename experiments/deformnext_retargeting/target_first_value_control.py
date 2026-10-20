"""Single-forward source-value intervention with the source-use decision fixed.

Candidate retrieval, decoder, logits and attention run on their original inputs.
Only their readout values are reassigned. The use gate receives the original
readout evidence from THAT SAME forward, avoiding an extra stochastic/atomic
ray-pooling pass and avoiding a confound from changing the use decision.
This is a saved-weight diagnostic, not a retrained correspondence ablation.
"""
from contextlib import contextmanager

import torch

from .task_attention_controls import task_value_context


@contextmanager
def fixed_use_value_control(conditioner, events):
    from pixal3d.modules.target_first_ray_v2 import TargetFirstRayConditionerV2

    if not isinstance(conditioner, TargetFirstRayConditionerV2):
        raise ValueError('corrected target-first V2 conditioner required')
    original_forward = conditioner.forward
    previous_forward = conditioner.__dict__.get('forward')
    chunks = []
    gate_calls = []
    gate_reference = []
    readout_events = []

    def capture(evidence, weights, logits):
        chunks.append((evidence.reshape(-1, 1024).detach(),
                       weights.reshape(-1, weights.shape[-1]).detach(),
                       logits.reshape(-1, logits.shape[-1]).detach()))

    def original_gate_evidence(module, args):
        target, moved_evidence, valid = args
        selected = valid.flatten().nonzero().flatten()
        baseline = torch.zeros_like(moved_evidence).flatten(0, 1)
        if len(selected):
            if not chunks or sum(len(row[0]) for row in chunks) != len(selected):
                raise ValueError('original readout rows must match this forward\'s active rays')
            baseline.index_copy_(0, selected, torch.cat([row[0] for row in chunks]))
        elif chunks:
            raise ValueError('empty-source forward cannot carry readouts from another asset')
        baseline = baseline.reshape_as(moved_evidence)
        gate_reference.append((target, baseline, valid))
        gate_calls.append(dict(
            gate_evidence_change_prevented_RMS=float(
                (baseline - moved_evidence).float().square().mean().sqrt()),
            active_rays=int(valid.sum())))
        return target, baseline, valid

    def controlled_forward(**kwargs):
        if conditioner.training or torch.is_grad_enabled():
            raise ValueError('fixed-use value intervention requires eval and no_grad')
        chunks.clear()
        gate_calls.clear()
        gate_reference.clear()
        readout_events.clear()
        try:
            result = original_forward(**kwargs)
            selected = result.attention.query_valid.flatten().nonzero().flatten()
            if len(gate_calls) != 1:
                raise ValueError('exactly one source-use decision required per conditioner call')
            # Bypass hooks, and reuse this forward's original evidence/query.
            # No second ray pooling/retrieval pass enters this comparison.
            reference_use = conditioner.evidence_use.forward(*gate_reference[0])
            if not torch.equal(result.use_probability, reference_use):
                raise ValueError('source-use probability differs from the same-forward original evidence decision')
            if len(selected):
                expected_weights = torch.cat([row[1] for row in chunks])
                expected_logits = torch.cat([row[2] for row in chunks])
                if (not torch.equal(result.attention.candidate_weights.flatten(0, 1)[selected], expected_weights)
                        or not torch.equal(result.attention.matching_logits.flatten(0, 1)[selected], expected_logits)):
                    raise ValueError('the value intervention changed original attention weights or logits')
            events.append(dict(
                control='wrong-value-address-fixed-use',
                **gate_calls[0], readouts=list(readout_events),
                original_matching_logits_and_attention_weights_exact=True,
                original_source_use_gate_input_retained=True,
                source_use_probability_exact_same_forward=True,
                same_forward_reference=True, extra_reference_forward=False,
                source_numerator_RMS=float(result.source_numerator.float().square().mean().sqrt()),
                source_use_probability_mean=float(result.use_probability.float().mean()),
                optimizer_updates=0))
            return result
        finally:
            # Do not retain GPU evidence from completed assets in the context.
            chunks.clear()
            gate_calls.clear()
            gate_reference.clear()
            readout_events.clear()

    handle = None
    previous_attention = conditioner.__dict__.get('_attention')
    original_attention = conditioner._attention
    if conditioner.task_attention is not None:
        def capture_attention(module, args, output):
            capture(*output)
        handle = conditioner.task_attention.register_forward_hook(capture_attention)
    else:
        def capture_static(logits, values, valid, view_index, views):
            evidence, weights, view_weights = original_attention(logits, values, valid, view_index, views)
            capture(evidence, weights, logits)
            return evidence, weights, view_weights
        conditioner._attention = capture_static
    gate_handle = conditioner.evidence_use.register_forward_pre_hook(original_gate_evidence)
    conditioner.forward = controlled_forward
    try:
        # The capture hook is installed first so that the intervention sees
        # the original readout and preserves its exact logits and weights.
        with task_value_context(conditioner, 'wrong-address', readout_events):
            yield
    finally:
        gate_handle.remove()
        if handle is not None:
            handle.remove()
        elif previous_attention is None:
            del conditioner._attention
        else:
            conditioner._attention = previous_attention
        if previous_forward is None:
            del conditioner.forward
        else:
            conditioner.forward = previous_forward
        chunks.clear()
        gate_calls.clear()
        gate_reference.clear()
        readout_events.clear()
