"""Frozen preaggregated matching evidence, with independent reliability gating."""
import torch
from torch import nn

from .deform_conditioner import DeformationCondition, _Retargeter


class MatchingV2Conditioner(nn.Module):
    def __init__(self, *, arm, value_dim=1024, query_dim=1024, deformation_dim=None,
                 hidden_dim=256, g_max=.25, reject=False, reject_threshold=0.,
                 implementation='matching-v2',projection_mode='nearest',initialization_rng='native'):
        super().__init__()
        if implementation!='matching-v2' or arm not in {'b2','b3','c1','c2'}:
            raise ValueError('invalid matching-v2 conditioner contract')
        if value_dim!=1024 or query_dim!=1024:
            raise ValueError('matching-v2 currently supports SS only')
        if (arm=='c2') != (deformation_dim==2048):
            raise ValueError('only C2 has physical RAW 2048-D deformation channels')
        if arm!='c2' and deformation_dim is not None:
            raise ValueError('C1/B arms must not contain deformation channels')
        self.implementation=implementation;self.arm=arm;self.value_dim=value_dim;self.query_dim=query_dim
        self.g_max=g_max;self.reject=bool(reject);self.reject_threshold=float(reject_threshold)
        if projection_mode not in {'nearest','native-bilinear'}:
            raise ValueError('unknown matching-v2 projection mode')
        self.projection_mode=projection_mode
        self.fusion_gate=nn.Parameter(torch.zeros(()))
        if initialization_rng=='r9-compatible':
            # Preaggregation removes the frozen scorer from the forward path.
            # Preserve its constructor RNG consumption so the retargeter has
            # the exact same initial parameters as the R9 reference recipe.
            from .r9_matching import PromotedR9Matcher
            discarded=PromotedR9Matcher()
            del discarded
        elif initialization_rng!='native':
            raise ValueError('unknown retargeter initialization RNG contract')
        self.retargeter=_Retargeter(value_dim+query_dim+(2048 if arm=='c2' else 0),value_dim,hidden_dim) if arm in {'c1','c2'} else None

    def forward(self, *, source_values=None, source_valid=None, evidence_top1, evidence_soft,
                p_use, cached_valid, target_valid, target_query=None, deformation=None,
                control='normal', **unused):
        if evidence_top1.shape!=evidence_soft.shape or evidence_top1.ndim!=3 or evidence_top1.shape[-1]!=1024:
            raise ValueError('matching-v2 evidence must be [B,Q,1024]')
        shape=evidence_top1.shape[:2]
        if p_use.shape!=shape or cached_valid.shape!=shape or target_valid.shape!=shape:
            raise ValueError('matching-v2 reliability shape differs')
        if not torch.isfinite(p_use).all() or torch.any((p_use<0)|(p_use>1)):
            raise ValueError('matching-v2 p_use must be a finite calibrated probability')
        valid=cached_valid.bool() & target_valid.bool()
        evidence=(evidence_top1 if self.arm=='b2' else evidence_soft).float()
        usage=p_use.float()*(p_use>=self.reject_threshold) if self.reject else torch.ones_like(p_use)
        residual=None
        if control=='source-off':
            valid=torch.zeros_like(valid)
        elif control=='wrong-corr':
            evidence=torch.roll(evidence,1,1)
        elif control not in {'normal','shuffled-deform'}:
            raise ValueError('unknown matching-v2 control')
        if self.retargeter is not None:
            if target_query is None or target_query.shape!=evidence.shape:
                raise ValueError('adapter requires native target query')
            parts=[evidence,target_query.float()]
            if self.arm=='c2':
                if deformation is None or deformation.shape!=(*shape,2048):
                    raise ValueError('C2 requires explicit RAW 2048-D context delta')
                if control=='shuffled-deform':deformation=torch.roll(deformation,1,0)
                parts.append(deformation.float())
            elif deformation is not None:
                raise ValueError('C1 physically excludes deformation channels')
            residual=self.retargeter(torch.cat(parts,-1),self.g_max)*valid[...,None]
            evidence=evidence+residual
        result=DeformationCondition(evidence,None,None,None,None,residual,valid,torch.tanh(self.fusion_gate))
        result.usage_scale=usage*valid
        return result
