"""Evaluation-only utility-query intervention on a frozen source-ray model.

Input-image alpha support changes which voxel features enter the utility mean.
It does not change the member posterior, R context, source-use context, carrier,
or ray projection/write support. This is not surface-visibility gating.
"""
from contextlib import contextmanager
import hashlib

import torch

from pixal3d.modules.member_feature_source_ray import MemberFeatureSourceRayConditioner
from pixal3d.modules.ray_group_query import pool_ray_queries


MODES = ('normal', 'observed-query-only')


def _digest(tensor):
    value = tensor.detach().contiguous().cpu()
    result = hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode())
    result.update(memoryview(value.numpy()))
    return result.hexdigest()


@contextmanager
def utility_query_support(module, events, *, mode):
    if type(module) is not MemberFeatureSourceRayConditioner or mode not in MODES:
        raise ValueError('explicit NoDQ four-head model and utility-query mode required')
    state, handles = {}, []

    def begin(layer, args, kwargs):
        if (state or args or layer.training or torch.is_grad_enabled()
                or kwargs.get('control', 'normal') != 'normal'):
            raise ValueError('single uncombined utility intervention requires eval/no_grad')
        full = pool_ray_queries(kwargs['native_voxels'], kwargs['ray_ids'], num_rays=1024)
        ids = torch.where(kwargs['observed_query'][:, None], kwargs['ray_ids'], -1)
        observed = pool_ray_queries(kwargs['native_voxels'], ids, num_rays=1024)
        active = observed.valid & (kwargs['source_valid'].flatten(1).sum(-1) >= 64)[:, None, None]
        selected = active.flatten().nonzero().flatten()
        state.update(inputs=kwargs, full=full, observed=observed, active=active,
                     selected=selected, cursor=0, priors=[], gate=[], retarget=[])

    def query(head, args, kwargs):
        if not state or kwargs or len(args) != 6:
            raise ValueError('actual six-argument RayFeatureReadout call required')
        count = len(args[0])
        start = state['cursor']
        take = state['selected'][start:start+count]
        full = state['full'].values.reshape(-1, 1024)[take]
        if len(take) != count or not torch.equal(args[0], full):
            raise ValueError('native full-ray utility or selected-row order changed')
        state['cursor'] += count
        state['priors'].append(args[5])
        if mode == 'normal':
            return None
        changed = state['observed'].values.reshape(-1, 1024)[take]
        # All other actual head arguments keep their original tensor objects.
        return (changed, *args[1:]), kwargs

    def gate(layer, args):
        state['gate'].append(args)

    def retarget(layer, args):
        state['retarget'].append(args[0][..., -1024:])

    def finish(layer, args, kwargs, result):
        selected = state['selected']
        attention = result.attention
        valid = state['active'].flatten(1)
        if (state['cursor'] != len(selected) or len(state['gate']) != 1
                or not torch.equal(attention.query_valid, valid)):
            raise ValueError('complete unchanged member-posterior row coverage required')
        target, top1, gate_valid = state['gate'][0]
        address = attention.candidate_indices.gather(
            -1, attention.matching_logits.argmax(-1, keepdim=True)).squeeze(-1)
        expected = state['inputs']['source_values'].float().flatten(1, 2).gather(
            1, address[..., None].expand(-1, -1, 1024))*valid[..., None]
        if (not torch.equal(top1, expected)
                or not torch.equal(attention.base_evidence, expected)
                or not torch.equal(gate_valid, valid)
                or not torch.equal(result.use_probability, module.evidence_use.forward(target, top1, valid))):
            raise ValueError('original same-forward Top1/source-use decision changed')
        if len(selected):
            prior = torch.cat(state['priors'])
            if (not torch.equal(prior, attention.matching_logits.reshape(-1, 64)[selected])
                    or len(state['retarget']) != 1
                    or not torch.equal(state['retarget'][0], target.reshape(-1, 1024)[selected])):
                raise ValueError('original matching prior or common R/use target context changed')
        elif state['priors'] or state['retarget']:
            raise ValueError('empty evidence cannot enter head or retargeter')
        full = state['full'].values.reshape(-1, 1024)[selected]
        observed = state['observed'].values.reshape(-1, 1024)[selected]
        counts = state['full'].voxel_count.flatten()[selected]
        seen = state['observed'].voxel_count.flatten()[selected]
        events.append(dict(
            mode=mode, queries=len(selected), actual_head_rows=state['cursor'],
            head_native_argument_only_replaced=mode != 'normal',
            other_five_head_arguments_passed_as_original_tensor_objects=True,
            original_full_ray_utility_and_row_order_exact=True,
            original_member_prior_exact_same_forward=True,
            original_Top1_feature_exact_same_forward=True,
            source_use_probability_exact_same_forward=True,
            original_R_and_source_use_target_context_exact=True,
            fixed_context_SHA256=dict(
                candidate_indices=_digest(attention.candidate_indices),
                matching_prior=_digest(attention.matching_logits),
                Top1_feature=_digest(top1), R_and_source_use_target=_digest(target),
                source_use_probability=_digest(result.use_probability)),
            excluded_input_alpha_fraction=float((1-seen.float()/counts).mean()) if len(selected) else None,
            utility_query_change_RMS=float((full-observed).square().mean().sqrt()) if len(selected) else None,
            ray_projection_and_voxel_write_support_unchanged=True,
            saved_weight_query_diagnostic_not_retrained_ablation=True))
        state.clear()

    try:
        handles.append(module.register_forward_pre_hook(begin, with_kwargs=True))
        handles.append(module.ray_readout.register_forward_pre_hook(query, with_kwargs=True))
        handles.append(module.evidence_use.register_forward_pre_hook(gate))
        handles.append(module.retargeter.register_forward_pre_hook(retarget))
        handles.append(module.register_forward_hook(finish, with_kwargs=True))
        yield
        if state:
            raise ValueError('unfinished utility-query intervention forward')
    finally:
        for handle in handles:
            handle.remove()
        state.clear()
