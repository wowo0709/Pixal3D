"""Top-K composition on the unchanged, positive Top1+R target-ray carrier."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .deform_conditioner import DeformationAwareConditioner, DeformationCondition
from .task_matching_attention import TaskCandidateAttention, contextual_query


def _attend(head, native, context, values, batch, indices, valid, prior):
    return head(native, context, values[batch[:, None], indices], valid, prior)


class AnchorRayCompositionConditioner(DeformationAwareConditioner):
    """One query per edited target patch/ray; retain the original nearest gather.

    Only ray_mix and ray_composer's five tensors train. ray_mix is a projected
    parameter, not a raw clamp with a dead negative region. The original R,
    fusion and matcher are frozen and keep their original state-dict names.
    """

    def __init__(self, *, composer_chunk_size=256, **kwargs):
        super().__init__(**kwargs)
        if (self.arm != 'c1' or self.implementation != 'r9-native-v5'
                or self.projection_mode != 'nearest' or self.candidate_aggregation != 'top1'
                or self.task_attention is not None or self.correspondence_mode != 'semantic'
                or (self.key_dim, self.value_dim, self.query_dim, self.topk) != (1024, 1024, 1024, 64)):
            raise ValueError('unchanged nearest SS C1 Top1+R carrier required')
        if isinstance(composer_chunk_size, bool) or not isinstance(composer_chunk_size, int) or composer_chunk_size < 1:
            raise ValueError('positive composer query chunk required')
        self.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261020)
            self.ray_composer = TaskCandidateAttention(content_mode='residual-zero', initial_content_scale=16.)
        self.ray_mix = nn.Parameter(torch.zeros(()))
        self.composer_chunk_size = composer_chunk_size

    def reference_state(self):
        return {key: value for key, value in self.state_dict().items()
                if key != 'ray_mix' and not key.startswith('ray_composer.')}

    def load_reference(self, reference):
        if float(self.ray_mix.detach()) != 0 or set(reference) != set(self.reference_state()):
            raise ValueError('fresh zero-gate and complete original conditioner state required')
        merged = dict(self.state_dict()); merged.update(reference)
        self.load_state_dict(merged, strict=True)
        if any(not torch.equal(value.detach().cpu(), self.state_dict()[key].detach().cpu())
               for key, value in reference.items()):
            raise ValueError('original R/matcher/fusion was not loaded exactly')

    @torch.no_grad()
    def project_mix_(self):
        if not torch.isfinite(self.ray_mix):
            raise ValueError('nonfinite composition gate')
        self.ray_mix.clamp_(0., 1.)

    def install_gate_projection(self, optimizer):
        if not any(parameter is self.ray_mix for group in optimizer.param_groups for parameter in group['params']):
            raise ValueError('optimizer must own the composition gate')
        self.project_mix_()
        return optimizer.register_step_post_hook(lambda *_: self.project_mix_())

    def forward(self, *, source_values, source_valid, target_query, semantic_indices,
                matching_logits, candidate_valid, target_valid, target_keys, source_keys,
                control='normal', **unsupported):
        if unsupported or control != 'normal':
            raise ValueError('only declared RGB/Omega inputs and normal carrier accepted')
        if not bool(torch.isfinite(self.ray_mix)) or not 0 <= float(self.ray_mix.detach()) <= 1:
            raise ValueError('project the composition gate after every optimizer step')
        source_values = source_values.float()
        b, views, sites, width = source_values.shape
        q = target_query.shape[1]
        if (views != 3 or sites != 1024 or width != 1024 or target_query.shape != (b, q, 1024)
                or target_keys.shape != target_query.shape or source_keys.shape != source_values.shape
                or source_valid.shape != (b, views, sites) or target_valid.shape != (b, q)
                or semantic_indices.shape != (b, q, 64) or matching_logits.shape != semantic_indices.shape
                or candidate_valid.shape != semantic_indices.shape
                or source_valid.dtype != torch.bool or target_valid.dtype != torch.bool):
            raise ValueError('same target-patch rays and complete native source64 cache required')
        indices = semantic_indices.long()
        if ((indices < 0) | (indices >= views * sites)).any():
            raise ValueError('cached source candidate address out of range')
        flat_values = source_values.flatten(1, 2)
        flat_valid = source_valid.flatten(1)
        valid = flat_valid[:, None].expand(-1, q, -1).gather(2, indices) & candidate_valid.bool()
        valid = valid & target_valid[..., None] & (flat_valid.sum(-1) >= 64)[:, None, None]
        active = valid.any(-1)
        logits = matching_logits.to(source_values.dtype).masked_fill(~valid, -torch.inf)
        chosen = logits.argmax(-1, keepdim=True)
        hard = torch.zeros_like(logits).scatter(2, chosen, 1.) * active[..., None]
        address = indices.gather(2, chosen).squeeze(2)
        top1 = flat_values.gather(1, address[..., None].expand(-1, -1, 1024)) * active[..., None]
        selected = active.flatten().nonzero().flatten()
        soft = source_values.new_zeros(b*q, 1024)
        weights = source_values.new_zeros(b*q, 64)
        evidence_parts, probability_parts = [], []
        flat_keys = source_keys.flatten(1, 2)
        for begin in range(0, len(selected), self.composer_chunk_size):
            take = selected[begin:begin+self.composer_chunk_size];batch = take // q
            ids = indices.reshape(-1, 64)[take];mask = valid.reshape(-1, 64)[take]
            with torch.no_grad(), torch.autocast(device_type=source_values.device.type, enabled=False):
                keys = flat_keys[batch[:, None], ids].float()
                keys = torch.where(mask[..., None], keys, torch.zeros_like(keys))
                context = contextual_query(self.q2c.decoder, target_keys.reshape(-1, 1024)[take].float(), keys, mask)
            args = (self.ray_composer, target_query.reshape(-1, 1024)[take].float(), context,
                    flat_values, batch, ids, mask, logits.reshape(-1, 64)[take] / self.score_temperature)
            value, probability, _ = (checkpoint(_attend, *args, use_reentrant=False)
                if torch.is_grad_enabled() else _attend(*args))
            evidence_parts.append(value); probability_parts.append(probability)
        if len(selected):
            soft = soft.index_copy(0, selected, torch.cat(evidence_parts))
            weights = weights.index_copy(0, selected, torch.cat(probability_parts))
        else:
            link = sum(parameter.reshape(-1)[0]*0 for parameter in self.ray_composer.parameters())
            soft = soft + link; weights = weights + link
        soft = soft.reshape(b, q, 1024); weights = weights.reshape(b, q, 64)
        mixed = top1 + self.ray_mix * (soft-top1)
        combined = hard + self.ray_mix * (weights-hard)
        # Keep the original full[B,Q,2048] R call, dtype, mask and coefficients.
        residual = self.retargeter(torch.cat((mixed, target_query), -1), self.g_max)
        residual = residual * active[..., None]
        view_ids = torch.div(indices, sites, rounding_mode='floor')
        view_weights = logits.new_zeros(b, q, views).scatter_add(2, view_ids, combined)
        self.last_composition_stats = dict(gate=float(self.ray_mix.detach()), active_queries=int(active.sum()),
            target_patch_queries=q, source_candidates=64)
        return DeformationCondition(mixed+residual, indices, logits, combined, view_weights,
            residual, active, torch.tanh(self.fusion_gate))
