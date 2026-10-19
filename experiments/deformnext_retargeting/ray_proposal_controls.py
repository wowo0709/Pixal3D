"""Inference-only ray candidate union; retain the frozen ray query and readout math."""
from contextlib import contextmanager

import torch

from pixal3d.modules import ray_group_matching as runtime
from pixal3d.modules.r9_matching import frame_topk
from pixal3d.trainers.flow_matching.ray_group_matching import RayGroupImageConditionedProjFlowMatchingCFGTrainer

from .ray_candidate_union import merge_ray_patch_proposals


@torch.no_grad()
def constituent_patch_union(*, target_keys, source_keys, source_valid, ray_ids,
                           patch_ids, query_valid):
    """Retrieve from distinct RGB/Omega target patches, then merge into64 per ray.

    No depth, topology, teacher or target SS enters this argument boundary.
    Query_valid is the frozen native support, including its <64-source disable.
    """
    batch = len(source_keys)
    if (target_keys.shape != (batch, 1024, 1024) or source_keys.shape != (batch, 3, 1024, 1024)
            or source_valid.shape != (batch, 3, 1024) or ray_ids.ndim != 3
            or ray_ids.shape[:2] != (batch, 3) or patch_ids.shape != (batch, ray_ids.shape[-1])
            or query_valid.shape != (batch, 3072)):
        raise ValueError('native ray membership, RGB FRAME patches and support required')
    indices = torch.zeros(batch, 3072, 64, dtype=torch.long, device=source_keys.device)
    valid = torch.zeros_like(indices, dtype=torch.bool)
    for b in range(batch):
        if not query_valid[b].any():
            continue
        if int(source_valid[b].sum()) < 64:
            raise ValueError('native source-disable support differs')
        rays = ray_ids[b].detach().cpu()
        patches = patch_ids[b].detach().cpu()
        selected = patches[(rays >= 0).any(0)].unique(sorted=True)
        _, proposals = frame_topk(target_keys[b:b+1, selected.to(target_keys.device)],
            source_keys[b:b+1], source_valid[b:b+1], 64)
        patch_candidates = torch.zeros(1024, 64, dtype=torch.long)
        patch_candidates[selected] = proposals[0].cpu()
        merged, active = merge_ray_patch_proposals(rays, patches, patch_candidates,
            source_valid[b].flatten().detach().cpu())
        if not torch.equal(active, query_valid[b].cpu()):
            raise ValueError('union cannot change native query support')
        indices[b] = merged.to(indices.device)
        valid[b] = active[:, None].to(valid.device).expand(-1, 64)
    return indices, valid


@contextmanager
def ray_proposal_control(mode, events):
    """Change candidate proposals only inside this frozen inference process."""
    if mode not in {'identity', 'patch-union'}:
        raise ValueError('explicit identity or equal-budget patch-union control required')
    original_forward = runtime.RayGroupConditioner.forward
    original_proposals = runtime.semantic_ray_proposals
    original_inject = RayGroupImageConditionedProjFlowMatchingCFGTrainer._inject_deform_transport
    transport_context, forward_context = [], []

    def inject(self, condition, transport, coords, **kwargs):
        if {'positive_indices', 'positive_visible', 'depth', 'ss_latent', 'target_vertices'} & transport.keys():
            raise ValueError('teacher data cannot enter candidate proposals')
        transport_context.append(transport)
        try:
            return original_inject(self, condition, transport, coords, **kwargs)
        finally:
            transport_context.pop()

    def proposals(query, source, source_valid, query_valid, chunk_size):
        baseline_indices, baseline_valid = original_proposals(query, source, source_valid, query_valid, chunk_size)
        if not transport_context or not forward_context:
            raise ValueError('actual native transport and ray arguments required')
        transport, arguments = transport_context[-1], forward_context[-1]
        if mode == 'identity':
            indices, valid = baseline_indices, baseline_valid
        else:
            indices, valid = constituent_patch_union(target_keys=transport['target_keys'], source_keys=source,
                source_valid=source_valid, ray_ids=arguments['ray_ids'],
                patch_ids=transport['patch_grid'].flatten(1), query_valid=query_valid)
        if not torch.equal(valid, baseline_valid):
            raise ValueError('same64 validity and active query support required')
        active = valid.any(-1)
        if active.any():
            selected = indices[active]
            sorted_ids = selected.sort(-1).values
            if (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any():
                raise ValueError('each active ray requires64 distinct source addresses')
        events.append(dict(mode=mode, batch_assets=len(source), active_rays=int(active.sum()),
            candidate_count=64, same_query_support=True, same_pooled_ray_queries=True,
            changed_candidate_slots=int(((indices != baseline_indices) & valid).sum()),
            same_D1_scalar_utility_R_math=True, GT_read_by_control=False, optimizer_updates=0))
        return indices, valid

    def forward(self, **arguments):
        if self.training or torch.is_grad_enabled() or self.ray_mode != 'semantic-attention':
            raise ValueError('candidate controls require frozen inference Full conditioner')
        forward_context.append(arguments)
        try:
            return original_forward(self, **arguments)
        finally:
            forward_context.pop()

    runtime.RayGroupConditioner.forward = forward
    runtime.semantic_ray_proposals = proposals
    RayGroupImageConditionedProjFlowMatchingCFGTrainer._inject_deform_transport = inject
    try:
        yield
    finally:
        runtime.RayGroupConditioner.forward = original_forward
        runtime.semantic_ray_proposals = original_proposals
        RayGroupImageConditionedProjFlowMatchingCFGTrainer._inject_deform_transport = original_inject
