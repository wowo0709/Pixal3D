"""Target-conditioned source retargeting with a member-preserving ray prior."""
import torch
from torch.utils.checkpoint import checkpoint

from .deform_conditioner import DeformationCondition
from .member_ray_posterior import member_ray_posteriors
from .ray_group_matching import RayGroupConditioner,attend_semantic_chunk
from .ray_group_query import pool_ray_queries
from .target_first_ray import SourceEvidenceUse
from .target_first_ray_v2 import TargetFirstRayConditionerV2,TargetFirstRayEvidence


class MemberPosteriorSourceRayConditioner(TargetFirstRayConditionerV2):
    """Same R/use/readout modules, one posterior and one update per source ray.

    Top1 is the actual single correspondence FEATURE control. Fixed weighted
    and learned attention retain the same64 member-mixture keys and prior.
    Native target context/delivery use the full ray, without a visibility mask.
    """
    def __init__(self,**kwargs):
        if kwargs.get('ray_mode') not in {'semantic-top1','semantic-weighted','semantic-attention'}:
            raise ValueError('explicit member posterior semantic readout required')
        # Preserve the V2 evidence interface (including saved-value controls),
        # while admitting its additional fixed-weighted semantic readout.
        RayGroupConditioner.__init__(self,**kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261020)
            self.evidence_use=SourceEvidenceUse()

    def forward(self, *, native_voxels,target_omega_voxels,target_native_voxels,
                frame_target_keys,target_patch_ids,ray_ids,observed_query,
                source_keys,source_values,source_valid,control='normal'):
        if (control!='normal' or native_voxels.shape!=target_native_voxels.shape
                or native_voxels.shape!=target_omega_voxels.shape
                or native_voxels.shape[-1]!=1024
                or source_values.shape!=(native_voxels.shape[0],3,1024,1024)):
            raise ValueError('native SS/FRAME inputs and explicit saved-value diagnostics required')
        batch=source_values.shape[0];rays=3072;count=64
        utility=pool_ray_queries(native_voxels,ray_ids,num_rays=1024)
        target=pool_ray_queries(target_native_voxels,ray_ids,num_rays=1024)
        posterior=member_ray_posteriors(self.q2c,target_keys=frame_target_keys,
            source_keys=source_keys,source_valid=source_valid,ray_ids=ray_ids,
            patch_ids=target_patch_ids,observed_query=observed_query,
            temperature=self.score_temperature,chunk_size=self.ray_chunk_size)
        indices,valid=posterior.candidate_indices,posterior.candidate_valid
        selected=valid.any(-1).flatten().nonzero().flatten()
        evidence=source_values.new_zeros(batch*rays,1024,dtype=torch.float32)
        weights=source_values.new_zeros(batch*rays,count,dtype=torch.float32)
        logits=source_values.new_full((batch*rays,count),-torch.inf,dtype=torch.float32)
        flat=source_values.flatten(1,2).float();query=utility.values.reshape(-1,1024)
        context=posterior.contextual_query.reshape(-1,1024)
        # The prior is the normalized mixture, rather than the score of a
        # mean descriptor. Zero probability means absent mass, not a zero key.
        prior=posterior.probabilities.log().reshape(-1,count)
        parts=[];probabilities=[];scores=[]
        for start in range(0,len(selected),self.ray_chunk_size):
            ix=selected[start:start+self.ray_chunk_size];b=ix//rays
            ids=indices.reshape(-1,count)[ix];mask=valid.reshape(-1,count)[ix]
            if self.task_attention is None:
                values=flat[b[:,None],ids]
                e,p,_=self._attention(prior[ix][None],values[None],mask[None],
                    torch.div(ids,1024,rounding_mode='floor')[None],3)
                result=e[0],p[0],prior[ix]
            else:
                args=(self.task_attention,query[ix],context[ix],flat,b,ids,mask,
                    prior[ix] if self.task_attention_config['score_prior'] else None)
                result=(checkpoint(attend_semantic_chunk,*args,use_reentrant=False)
                    if torch.is_grad_enabled() else attend_semantic_chunk(*args))
            e,p,l=result;parts.append(e);probabilities.append(p);scores.append(l)
        if len(selected):
            evidence=evidence.index_copy(0,selected,torch.cat(parts))
            weights=weights.index_copy(0,selected,torch.cat(probabilities))
            logits=logits.index_copy(0,selected,torch.cat(scores))
        else:
            evidence=evidence+sum(p.reshape(-1)[0]*0 for p in self.parameters() if p.requires_grad)
        residual=evidence.new_zeros(batch*rays,1024)
        target_query=target.values.reshape(batch,rays,1024)
        if len(selected):
            value=self.retargeter(torch.cat((evidence[selected],target_query.flatten(0,1)[selected]),-1),self.g_max)
            residual=residual.index_copy(0,selected,value.float())
        evidence=evidence.reshape(batch,rays,1024);residual=residual.reshape(batch,rays,1024)
        weights=weights.reshape(batch,rays,count);query_valid=valid.any(-1)
        view_weights=evidence.new_zeros(batch,rays,3).scatter_add(2,
            torch.div(indices,1024,rounding_mode='floor'),weights)
        scale=torch.tanh(self.fusion_gate)
        attention=DeformationCondition(evidence+residual,indices,logits.reshape(batch,rays,count),
            weights,view_weights,residual,query_valid,scale,base_evidence=evidence)
        use=self.evidence_use(target_query,evidence,query_valid)
        mass=use[...,None]*scale
        numerator=mass*evidence+use[...,None]*residual
        self._last_member_retained_mass=posterior.retained_mass.detach()
        self._last_distinct_member_count=posterior.distinct_member_count.detach()
        return TargetFirstRayEvidence(numerator.reshape(batch,3,1024,1024),
            mass.reshape(batch,3,1024,1),attention,use)
