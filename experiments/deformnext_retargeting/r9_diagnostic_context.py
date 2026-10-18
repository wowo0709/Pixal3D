"""Saved-weight controls for the original R9 native C1 candidate.

This uses R9's real source-value tensors rather than pretending they are the
preaggregated matching-v2 evidence. No trained model or dataset is modified.
"""
from contextlib import contextmanager

import torch


DEFINITIONS = {
    'source-off': 'Disable additional R9 transport/adapter; retain native edited/source four-RGB conditioning.',
    'adapter-off': 'Remove only the learned retargeter output; keep semantic evidence, scores, gate and native RGB.',
    'wrong-object': 'Cycle donor DINO source values within each camera onto recipient foreground sites by ordinal resampling; retain recipient validity, Omega keys, target query, candidates and scores.',
    'wrong-address': 'Derange foreground DINO values within the same object and source camera; retain joint per-view value multisets, support, Omega keys, target query, candidates and scores.',
}


@contextmanager
def r9_diagnostic_context(conditioner, mode, events):
    if mode == 'normal':
        yield
        return
    if mode not in DEFINITIONS or getattr(conditioner, 'implementation', None) != 'r9-native-v5':
        raise ValueError('R9 diagnostics require the original r9-native-v5 conditioner')
    if mode == 'adapter-off':
        if conditioner.retargeter is None:
            raise ValueError('adapter-off requires the original R9 retargeter')
        def zero(module, inputs, output):
            events.append({'mode': mode, 'shape': list(output.shape)})
            return torch.zeros_like(output)
        hook = conditioner.retargeter.register_forward_hook(zero)
    else:
        def perturb(module, positional, keyword):
            if keyword.get('control', 'normal') != 'normal':
                raise ValueError('do not combine R9 path controls')
            changed = dict(keyword)
            if mode == 'source-off':
                changed['control'] = 'source-off'
                events.append({'mode': mode})
                return positional, changed
            values, valid = keyword['source_values'], keyword['source_valid']
            if values.ndim != 4 or valid.shape != values.shape[:3] or valid.dtype != torch.bool:
                raise ValueError('expected actual R9 BxVxS source values/validity')
            if mode == 'wrong-address':
                replacement = values.clone()
                changed_sites = 0
                for sample in range(len(values)):
                    for view in range(values.shape[1]):
                        index = torch.where(valid[sample, view])[0]
                        if not len(index):
                            continue
                        if len(index) < 2:
                            raise ValueError('wrong-address needs at least two foreground sites per active camera')
                        selected = index.roll(len(index)//2)
                        if torch.any(index == selected):
                            raise ValueError('source address permutation has fixed points')
                        replacement[sample, view, index] = values[sample, view, selected]
                        changed_sites += len(index)
                changed['source_values'] = replacement
                events.append({'mode': mode, 'changed_source_address_sites': changed_sites,
                    'foreground_sites': int(valid.sum()), 'per_view_joint_value_multisets_exact': True,
                    'recipient_support_and_Omega_keys_preserved': True})
                return positional, changed
            if values.shape[0] < 2:
                raise ValueError('R9 wrong-object needs at least two assets')
            replacement = values.clone()
            counts, zero_vectors = [], 0
            for recipient in range(len(values)):
                donor = (recipient-1) % len(values)
                per_view = []
                for view in range(values.shape[1]):
                    index = torch.where(valid[recipient, view])[0]
                    source = torch.where(valid[donor, view])[0]
                    per_view.append(len(index))
                    if not len(index):
                        continue
                    if not len(source):
                        raise ValueError('R9 wrong-object donor camera has no foreground')
                    ordinal = torch.arange(len(index), device=index.device)
                    selected = source[ordinal*len(source)//len(index)]
                    transported = values[donor, view, selected]
                    if not torch.isfinite(transported).all():
                        raise ValueError('nonfinite donor source values')
                    zero_vectors += int((transported.abs().sum(-1) == 0).sum())
                    replacement[recipient, view, index] = transported
                counts.append(per_view)
            if zero_vectors:
                raise ValueError('wrong-object would inject zero donor foreground values')
            changed['source_values'] = replacement
            events.append({'mode': mode, 'recipient_foreground_per_view': counts,
                'zero_donor_vectors': zero_vectors, 'recipient_support_and_Omega_keys_preserved': True})
            return positional, changed
        hook = conditioner.register_forward_pre_hook(perturb, with_kwargs=True)
    try:
        yield
    finally:
        hook.remove()
