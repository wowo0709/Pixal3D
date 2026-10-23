"""RAW semantic/native-spatial Top1 controls with the common R-only delivery.

The source branch retains the same ten R/fusion/use tensors as R19. There is
no Q/K/V readout, score utility, Top-K value mixture, or dummy attention head.
The matching decoder and score head are physically absent in both modes.
"""
import torch

from .deform_conditioner import DeformationCondition
from .member_posterior_source_ray import MemberPosteriorSourceRayConditioner
from .ray_group_query import pool_ray_queries
from .source_contrast_member_feature import SourceContrastRetargeter
from .target_first_ray_v2 import TargetFirstRayEvidence


class RawCorrespondenceTop1SourceConditioner(MemberPosteriorSourceRayConditioner):
    def __init__(self, *, correspondence, **kwargs):
        if correspondence not in {'semantic','native-spatial'}:
            raise ValueError('explicit RAW semantic or original native-spatial mode required')
        if (kwargs.get('ray_mode') != 'semantic-top1'
                or kwargs.get('candidate_aggregation') != 'top1'
                or kwargs.get('task_attention') is not None):
            raise ValueError('semantic Top1 feature only; no uniform or learned Top-K head')
        super().__init__(**kwargs)
        self.q2c.eval()
        self.retargeter = SourceContrastRetargeter(self.retargeter)
        self._configure_matcher()
        self.correspondence_mode=correspondence
        self.register_buffer('raw_semantic_top1_source_schema' if correspondence=='semantic'
                             else 'raw_native_spatial_top1_source_schema',torch.tensor([1],dtype=torch.uint8))
        trainable = {n: p for n, p in self.named_parameters() if p.requires_grad}
        if (len(trainable) != 10 or self.task_attention is not None
                or any(n.startswith('ray_readout.') for n in self.state_dict())
                or self.retargeter.input_dim != 2048):
            raise ValueError('physical ten-tensor Top1 R-only source module required')

    def _configure_matcher(self):
        self.q2c=None
        self.matching_mode='RAW-cosine'

    def _semantic_posterior(self,**kwargs):
        if self.q2c is not None:raise ValueError('RAW matching must omit the decoder')
        from .raw_member_ray_posterior import raw_member_ray_posteriors
        return raw_member_ray_posteriors(**kwargs)

    def train(self, mode=True):
        torch.nn.Module.train(self,mode)
        if self.q2c is not None:self.q2c.eval()
        return self

    def forward(self, *, native_voxels, target_omega_voxels, target_native_voxels,
                frame_target_keys, target_patch_ids, ray_ids, observed_query,
                source_keys, source_values, source_valid, spatial_keys=None, spatial_values=None, control='normal'):
        if (control != 'normal' or native_voxels.shape != target_native_voxels.shape
                or native_voxels.shape != target_omega_voxels.shape or native_voxels.shape[-1] != 1024
                or source_values.shape != (native_voxels.shape[0], 3, 1024, 1024)):
            raise ValueError('aligned native SS/source-camera rays and unmodified Top1 feature control required')
        batch, rays, count = len(source_values), 3072, 64
        target = pool_ray_queries(target_native_voxels, ray_ids, num_rays=1024)
        if self.correspondence_mode=='semantic':
            if spatial_keys is not None or spatial_values is not None:
                raise ValueError('semantic matching must not consume spatial candidates')
            posterior=self._semantic_posterior(target_keys=frame_target_keys,
                source_keys=source_keys,source_valid=source_valid,ray_ids=ray_ids,
                patch_ids=target_patch_ids,observed_query=observed_query,
                temperature=self.score_temperature,chunk_size=self.ray_chunk_size)
        else:
            from .raw_spatial_member_posterior import raw_spatial_member_posteriors
            if spatial_values is None or spatial_keys is None or spatial_values.shape!=spatial_keys.shape:
                raise ValueError('actual original spatial key and feature candidates required')
            count=3
            posterior=raw_spatial_member_posteriors(target_keys=frame_target_keys,
                source_valid=source_valid,ray_ids=ray_ids,patch_ids=target_patch_ids,
                observed_query=observed_query,spatial_keys=spatial_keys,
                temperature=self.score_temperature,chunk_size=self.ray_chunk_size)
        indices, valid = posterior.candidate_indices, posterior.candidate_valid
        selected = valid.any(-1).flatten().nonzero().flatten()
        flat = source_values.flatten(1, 2).float()
        evidence = flat.new_zeros(batch * rays, 1024)
        hard = flat.new_zeros(batch * rays, count)
        prior = posterior.probabilities.log().reshape(-1, count)
        residual = torch.zeros_like(evidence)
        target_query = target.values.reshape(batch * rays, 1024)
        if len(selected):
            # Gather only the winning FEATURE, rather than building a Top-K
            # value tensor and assigning uniform or one-hot attention to it.
            slots = prior[selected].argmax(-1)
            winner = indices.reshape(-1, count)[selected].gather(1, slots[:, None])[:, 0]
            appearance=(flat[selected//rays,winner] if self.correspondence_mode=='semantic' else
                spatial_values.reshape(batch*rays,3,1024)[selected].gather(1,slots[:,None,None].expand(-1,1,1024))[:,0])
            evidence = evidence.index_copy(0, selected, appearance)
            weights = hard.new_zeros(len(selected), count).scatter_(1, slots[:, None], 1.)
            hard = hard.index_copy(0, selected, weights)
            own_source = flat.reshape(-1, 1024)[selected]
            with self.retargeter.native_source_context(own_source):
                retargeted = self.retargeter(torch.cat((appearance, target_query[selected]), -1), self.g_max)
            residual = residual.index_copy(0, selected, retargeted.float())
            self.last_R_input_appearance = appearance.detach()
            self.last_R_only_output = retargeted.detach().float()
        else:
            evidence = evidence + sum(p.reshape(-1)[0] * 0 for p in self.parameters() if p.requires_grad)
            self.last_R_input_appearance = flat.new_empty((0, 1024))
            self.last_R_only_output = flat.new_empty((0, 1024))
        evidence = evidence.reshape(batch, rays, 1024)
        residual = residual.reshape(batch, rays, 1024)
        hard = hard.reshape(batch, rays, count)
        query_valid = valid.any(-1)
        view_weights=(evidence.new_zeros(batch,rays,3).scatter_add(2,indices//1024,hard)
                      if self.correspondence_mode=='semantic' else hard)
        scale = torch.tanh(self.fusion_gate)
        attention = DeformationCondition(evidence + residual, indices, prior.reshape(batch, rays, count),
            hard, view_weights, residual, query_valid, scale, base_evidence=evidence)
        use = self.evidence_use(target_query.reshape(batch, rays, 1024), evidence, query_valid)
        mass = use[..., None] * scale
        numerator = mass * evidence + use[..., None] * residual
        self.last_source_reference_rows = selected.detach()
        self._last_member_retained_mass = posterior.retained_mass.detach()
        self._last_distinct_member_count = posterior.distinct_member_count.detach()
        return TargetFirstRayEvidence(numerator.reshape(batch, 3, 1024, 1024),
            mass.reshape(batch, 3, 1024, 1), attention, use)
