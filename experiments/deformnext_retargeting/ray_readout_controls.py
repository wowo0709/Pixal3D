"""Process-local, inference-only ray readout interventions with unchanged GT-free inputs."""
from __future__ import annotations

from contextlib import contextmanager

import torch

from pixal3d.modules.deform_conditioner import DeformationCondition
from pixal3d.modules.ray_group_matching import RayGroupCondition, RayGroupConditioner
from pixal3d.modules.ray_group_query import pool_ray_queries
from pixal3d.trainers.flow_matching.ray_group_matching import RayGroupImageConditionedProjFlowMatchingCFGTrainer

from .ray_patch_score_pool import score_pool_ray_patches


def rebuild_ray_readout(module, arguments, original, probabilities, *, matching_logits=None):
    """Keep addresses/support/parameters; recompute evidence and its same trained R."""
    source=arguments['source_values'].float();batch,views,sites,width=source.shape
    trace=original.attention;indices=trace.candidate_indices
    rays=views*sites;valid=trace.query_valid
    if probabilities.shape!=trace.candidate_weights.shape or not torch.isfinite(probabilities).all():
        raise ValueError('finite same-support ray probabilities required')
    if (probabilities<0).any() or not torch.allclose(probabilities[valid].sum(-1),torch.ones_like(probabilities[valid].sum(-1)),atol=1e-5,rtol=0):
        raise ValueError('normalized active-ray probabilities required')
    selected=valid.flatten().nonzero().flatten()
    flat=source.flatten(1,2)
    pieces=[]
    for start in range(0,len(selected),module.ray_chunk_size):
        ix=selected[start:start+module.ray_chunk_size];b=ix//rays
        ids=indices.reshape(-1,64)[ix]
        values=flat[b[:,None],ids]
        pieces.append((probabilities.reshape(-1,64)[ix,:,None]*values).sum(1))
    evidence=source.new_zeros(batch*rays,width,dtype=torch.float32)
    if len(selected):evidence=evidence.index_copy(0,selected,torch.cat(pieces))
    target=pool_ray_queries(arguments['target_native_voxels'],arguments['ray_ids'],num_rays=sites).values.reshape(-1,width)
    residual=torch.zeros_like(evidence)
    if len(selected):
        value=module.retargeter(torch.cat((evidence[selected],target[selected]),-1),module.g_max)
        residual=residual.index_copy(0,selected,value.float())
    evidence=evidence.reshape(batch,rays,width);residual=residual.reshape(batch,rays,width)
    gate=torch.tanh(module.fusion_gate)
    delta=gate*(evidence-source.reshape(batch,rays,width))+residual
    delta=torch.where(valid[...,None],delta,torch.zeros_like(delta))
    view_ids=torch.div(indices,sites,rounding_mode='floor')
    view_weights=evidence.new_zeros(batch,rays,views).scatter_add(2,view_ids,probabilities)
    logits=probabilities.clamp_min(torch.finfo(torch.float32).tiny).log().masked_fill(~valid[...,None],-torch.inf)
    if matching_logits is not None:logits=matching_logits
    result=DeformationCondition(evidence+residual,indices,logits,probabilities,view_weights,residual,valid,gate,base_evidence=evidence)
    return RayGroupCondition(delta.reshape(batch,views,sites,width),result)


@contextmanager
def ray_readout_control(mode, events):
    """Patch this evaluation process only; restore both functions on exit.

    identity reconstructs the original evidence/R to verify arithmetic before
    any quality experiment. prior-only and patch-score-pool keep the saved
    Full weights,source64 and support but replace the candidate weight readout.
    """
    if mode not in {'identity','prior-only','patch-score-pool'}:
        raise ValueError('explicit saved-weight ray readout control required')
    original_forward=RayGroupConditioner.forward
    original_inject=RayGroupImageConditionedProjFlowMatchingCFGTrainer._inject_deform_transport
    context=[]

    def inject(self,condition,transport,coords,**kwargs):
        if {'positive_indices','positive_visible','depth','ss_latent','target_vertices'}&transport.keys():
            raise ValueError('GT/teacher data cannot enter ray readout controls')
        context.append(transport)
        try:return original_inject(self,condition,transport,coords,**kwargs)
        finally:context.pop()

    def forward(self,**arguments):
        if self.training or torch.is_grad_enabled() or self.ray_mode!='semantic-attention':
            raise ValueError('readout controls require the frozen inference Full conditioner')
        priors=[]
        def capture(module,args,result):
            if len(args)!=5 or args[4] is None:raise ValueError('original scalar-prior inputs required')
            priors.append(torch.softmax(args[4].float().masked_fill(~args[3],-torch.inf),dim=-1))
        handle=self.task_attention.register_forward_hook(capture)
        try:original=original_forward(self,**arguments)
        finally:handle.remove()
        if mode=='identity':probabilities=original.attention.candidate_weights
        elif mode=='prior-only':
            probabilities=torch.zeros_like(original.attention.candidate_weights)
            if priors:probabilities[original.attention.query_valid]=torch.cat(priors)
        else:
            if not context:raise ValueError('original transport context with camera patch addresses required')
            transport=context[-1];weights=[]
            for batch in range(len(arguments['source_values'])):
                membership=arguments['ray_ids'][batch].detach().cpu().clone()
                # Preserve native matching disable branch on <64 RGB-valid sources.
                if not original.attention.query_valid[batch].any():membership.fill_(-1)
                probability,_=score_pool_ray_patches(self.q2c,
                    target_keys=transport['target_keys'][batch],source_keys=arguments['source_keys'][batch],
                    source_valid=arguments['source_valid'][batch],candidates=original.attention.candidate_indices[batch],
                    ray_ids=membership,patch_ids=transport['patch_grid'][batch].flatten().detach().cpu(),
                    temperature=self.score_temperature,chunk_size=self.ray_chunk_size)
                weights.append(probability)
            probabilities=torch.stack(weights)
        rebuilt=rebuild_ray_readout(self,arguments,original,probabilities,
            matching_logits=original.attention.matching_logits if mode=='identity' else None)
        if mode=='identity':
            for before,after in [(original.map_delta,rebuilt.map_delta),
                    (original.attention.base_evidence,rebuilt.attention.base_evidence),
                    (original.attention.retarget_residual,rebuilt.attention.retarget_residual)]:
                if not torch.equal(before,after):raise ValueError('identity readout reconstruction changed original arithmetic')
        events.append(dict(mode=mode,batch_assets=len(arguments['source_values']),
            active_rays=int(original.attention.query_valid.sum()),
            same_candidate_addresses=torch.equal(original.attention.candidate_indices,rebuilt.attention.candidate_indices),
            same_query_support=torch.equal(original.attention.query_valid,rebuilt.attention.query_valid),
            candidate_count=64,GT_read_by_control=False,optimizer_updates=0))
        return rebuilt

    RayGroupConditioner.forward=forward
    RayGroupImageConditionedProjFlowMatchingCFGTrainer._inject_deform_transport=inject
    try:yield
    finally:
        RayGroupConditioner.forward=original_forward
        RayGroupImageConditionedProjFlowMatchingCFGTrainer._inject_deform_transport=original_inject
