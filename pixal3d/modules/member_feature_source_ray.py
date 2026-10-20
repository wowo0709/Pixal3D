"""Historical Q/K/V feature correction on the new source-camera ray carrier.

This ports the actual four-head feature transform and direct residual delivery.
It is not a reproduction of the historical target-patch query or warmup stream.
The frozen member posterior supplies one64-key support per source-camera ray.
"""
import torch
from torch.utils.checkpoint import checkpoint

from .deform_conditioner import DeformationCondition
from .member_posterior_source_ray import MemberPosteriorSourceRayConditioner
from .member_ray_posterior import member_ray_posteriors
from .ray_feature_attention import RayFeatureReadout
from .ray_group_query import pool_ray_queries
from .target_first_ray_v2 import TargetFirstRayEvidence


def _feature_chunk(head, native, context, values, keys, batch, indices, valid, prior,
                   control):
    """Gather inside the checkpoint; intervention changes projected V only."""
    if control not in {'normal', 'wrong-feature-values-fixed-use'}:
        raise ValueError('explicit transformed-feature value control required')
    handle = None
    if control != 'normal':
        if torch.is_grad_enabled() or head.training:
            raise ValueError('saved-weight feature intervention requires eval and no_grad')
        count = valid.sum(-1, keepdim=True)
        position = torch.arange(valid.shape[-1], device=valid.device).expand_as(valid)
        ordered = position.masked_fill(~valid, valid.shape[-1]).sort(-1).values
        rank = valid.long().cumsum(-1)-1
        donor_rank = (rank+count//2) % count.clamp_min(1)
        donor = ordered.gather(1, donor_rank).clamp_max(valid.shape[-1]-1)

        def move_projected_values(module, args, output):
            moved = output.gather(1, donor[..., None].expand_as(output))
            return torch.where(valid[..., None], moved, torch.zeros_like(moved))

        # Scores, Q, K, and the four attention probabilities see original
        # inputs. Permuting linear-projected V isolates feature correspondence.
        handle = head.value.register_forward_hook(move_projected_values)
    try:
        return head(native, context, keys[batch[:, None], indices].float(),
                    values[batch[:, None], indices].float(), valid, prior)
    finally:
        if handle is not None:
            handle.remove()


class MemberFeatureSourceRayConditioner(MemberPosteriorSourceRayConditioner):
    """Actual Top1+R carrier plus bounded, initially zero Q/K/V correction.

    No deformation-query channels or projection exist in this arm. The
    independent source-use decision consumes the unchanged Top1 evidence.
    Both R and transformed-feature correction bypass semantic fusion gate g:
    native source-map delta = u*[g*(Top1-S)+R(Top1+delta,T)+bound(delta)].
    """
    def __init__(self, **kwargs):
        if (kwargs.get('ray_mode') != 'semantic-top1'
                or kwargs.get('task_attention') is not None):
            raise ValueError('explicit Top1 carrier without scalar attention required')
        super().__init__(**kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261021)
            self.ray_readout = RayFeatureReadout(1024)

    def load_utility(self, state):
        """Load the complete historical five-tensor utility, never partial keys."""
        if torch.count_nonzero(self.ray_readout.output.weight):
            raise ValueError('untouched zero-output feature branch required')
        expected = self.ray_readout.score_selector.state_dict()
        if (set(state) != set(expected)
                or any(state[key].shape != value.shape or not torch.isfinite(state[key]).all()
                       for key, value in expected.items())):
            raise ValueError('complete finite historical utility tensors required')
        self.ray_readout.score_selector.load_state_dict(state, strict=True)

    def forward(self, *, native_voxels, target_omega_voxels, target_native_voxels,
                frame_target_keys, target_patch_ids, ray_ids, observed_query,
                source_keys, source_values, source_valid, control='normal'):
        if (control not in {'normal', 'wrong-feature-values-fixed-use'}
                or native_voxels.shape != target_native_voxels.shape
                or native_voxels.shape != target_omega_voxels.shape
                or native_voxels.shape[-1] != 1024
                or source_values.shape != (native_voxels.shape[0], 3, 1024, 1024)
                or not self.g_max > 0):
            raise ValueError('aligned nativeSS/source-camera rays and positive bound required')
        if control != 'normal' and (self.training or torch.is_grad_enabled()):
            raise ValueError('feature correspondence intervention requires eval and no_grad')
        batch, rays, count = source_values.shape[0], 3072, 64
        utility = pool_ray_queries(native_voxels, ray_ids, num_rays=1024)
        target = pool_ray_queries(target_native_voxels, ray_ids, num_rays=1024)
        posterior = member_ray_posteriors(self.q2c, target_keys=frame_target_keys,
            source_keys=source_keys, source_valid=source_valid, ray_ids=ray_ids,
            patch_ids=target_patch_ids, observed_query=observed_query,
            temperature=self.score_temperature, chunk_size=self.ray_chunk_size)
        indices, valid = posterior.candidate_indices, posterior.candidate_valid
        selected = valid.any(-1).flatten().nonzero().flatten()
        flat_values, flat_keys = source_values.flatten(1, 2).float(), source_keys.flatten(1, 2)
        prior = posterior.probabilities.log().reshape(-1, count)
        query = utility.values.reshape(-1, 1024)
        context = posterior.contextual_query.reshape(-1, 1024)
        top1 = source_values.new_zeros(batch*rays, 1024, dtype=torch.float32)
        hard = source_values.new_zeros(batch*rays, count, dtype=torch.float32)
        appearances, changes, hard_weights, feature_weights = [], [], [], []
        for start in range(0, len(selected), self.ray_chunk_size):
            take = selected[start:start+self.ray_chunk_size]
            b = take//rays
            ids, mask = indices.reshape(-1, count)[take], valid.reshape(-1, count)[take]
            value, weight, _ = self._attention(prior[take][None],
                flat_values[b[:, None], ids][None], mask[None], (ids//1024)[None], 3)
            appearances.append(value[0]); hard_weights.append(weight[0])
            args = (self.ray_readout, query[take], context[take], flat_values,
                    flat_keys, b, ids, mask, prior[take], control)
            change, probability = (checkpoint(_feature_chunk, *args, use_reentrant=False)
                if torch.is_grad_enabled() else _feature_chunk(*args))
            changes.append(change); feature_weights.append(probability.detach())
        residual = torch.zeros_like(top1)
        target_query = target.values.reshape(batch*rays, 1024)
        if len(selected):
            appearance, change = torch.cat(appearances), torch.cat(changes)
            top1 = top1.index_copy(0, selected, appearance)
            hard = hard.index_copy(0, selected, torch.cat(hard_weights))
            retargeted = self.retargeter(torch.cat((appearance+change, target_query[selected]), -1), self.g_max)
            direct = self.g_max*torch.tanh(change/self.g_max)
            residual = residual.index_copy(0, selected, retargeted.float()+direct)
            self.last_direct_feature_correction = direct.detach()
        else:
            # All trainable tensors remain DDP-connected on an empty rank.
            top1 = top1+sum(p.reshape(-1)[0]*0 for p in self.parameters() if p.requires_grad)
            self.last_direct_feature_correction = source_values.new_empty((0, 1024), dtype=torch.float32)
        top1 = top1.reshape(batch, rays, 1024)
        residual = residual.reshape(batch, rays, 1024)
        hard = hard.reshape(batch, rays, count)
        query_valid = valid.any(-1)
        view_weights = top1.new_zeros(batch, rays, 3).scatter_add(2, indices//1024, hard)
        scale = torch.tanh(self.fusion_gate)
        attention = DeformationCondition(top1+residual, indices,
            prior.reshape(batch, rays, count), hard, view_weights, residual,
            query_valid, scale, base_evidence=top1)
        use = self.evidence_use(target_query.reshape(batch, rays, 1024), top1, query_valid)
        mass = use[..., None]*scale
        numerator = mass*top1+use[..., None]*residual
        # Only active four-head rows are retained, detached; no dense[B,3072,4,64] graph.
        self.last_feature_probabilities = (selected.detach(), torch.cat(feature_weights)
            if feature_weights else source_values.new_empty((0, 4, count), dtype=torch.float32))
        self._last_member_retained_mass = posterior.retained_mass.detach()
        self._last_distinct_member_count = posterior.distinct_member_count.detach()
        return TargetFirstRayEvidence(numerator.reshape(batch, 3, 1024, 1024),
            mass.reshape(batch, 3, 1024, 1), attention, use)
