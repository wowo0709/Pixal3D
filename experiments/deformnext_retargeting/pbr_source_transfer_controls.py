"""Saved-weight contribution masks for the native PBR source transport.

These change neither correspondence nor weights, and are not retrained arms.
The raw source module still executes; only its two delivery terms are masked.
"""
from dataclasses import replace

import torch

from pixal3d.modules.target_first_ray_v2 import TargetFirstRayEvidence


CONTROLS = ('normal', 'R-off', 'direct-top1-shift-off', 'both-source-contributions-off')


def mask_source_transfer(output, control):
    if torch.is_grad_enabled() or control not in CONTROLS:
        raise ValueError('explicit saved-weight, no_grad source-delivery control required')
    if not isinstance(output, TargetFirstRayEvidence):
        raise ValueError('actual structured native source output required')
    numerator, mass = output.source_numerator, output.semantic_weight
    if (numerator.ndim != 4 or numerator.shape[1:] != (3,1024,2048)
            or mass.shape != numerator.shape[:-1]+(1,)):
        raise ValueError('native PBR three32patch source maps and scalar masses required')
    batch = numerator.shape[0]
    evidence, residual = output.attention.base_evidence, output.attention.retarget_residual
    use, gate = output.use_probability, output.attention.fusion_scale
    if (evidence is None or residual is None or gate is None
            or evidence.shape != (batch,3072,2048) or residual.shape != evidence.shape
            or use.shape != (batch,3072) or gate.ndim != 0
            or any(t.dtype != torch.float32 or t.device != numerator.device
                   or not torch.isfinite(t).all() for t in (numerator,mass,evidence,residual,use,gate))
            or ((use<0)|(use>1)).any()):
        raise ValueError('finite native float32 Top1/R/use/gate decomposition required')
    original_mass = (use[...,None]*gate).reshape_as(mass)
    direct = (original_mass*evidence.reshape_as(numerator))
    adapted = (use[...,None]*residual).reshape_as(numerator)
    if not torch.equal(mass,original_mass) or not torch.equal(numerator,direct+adapted):
        raise ValueError('source numerator must exactly decompose as u*g*Top1 + u*R')
    if control == 'normal':
        return output
    if control == 'R-off':
        return replace(output,source_numerator=direct)
    if control == 'direct-top1-shift-off':
        return replace(output,source_numerator=adapted,semantic_weight=torch.zeros_like(mass))
    return replace(output,source_numerator=torch.zeros_like(numerator),
                   semantic_weight=torch.zeros_like(mass))


class SavedPBRSourceTransferControl:
    def __init__(self,module,control):
        if (module.training or control not in CONTROLS or module.appearance_mode!='top1-only'
                or module.value_dim!=2048):
            raise ValueError('saved eval physical PBR Top1-only model required')
        self.module,self.control=module,control
        self.versions={n:p._version for n,p in module.named_parameters()}
        self.events=[]
        self.handle=module.register_forward_hook(self.after)

    def after(self,module,args,output):
        from .r_mediated_native_runtime import tensor_sha
        if module.training or torch.is_grad_enabled() or self.versions!={n:p._version for n,p in module.named_parameters()}:
            raise ValueError('source model/versions changed in a saved-weight diagnostic')
        changed=mask_source_transfer(output,self.control)
        att=output.attention
        self.events.append(dict(control=self.control,
            fixed_source_fields={k:tensor_sha(v) for k,v in dict(candidate_indices=att.candidate_indices,
                matching_logits=att.matching_logits,candidate_weights=att.candidate_weights,
                Top1_carrier=att.base_evidence,computed_R=att.retarget_residual,
                source_use=output.use_probability,fusion_scale=att.fusion_scale).items()},
            actual_source_numerator_changed=not torch.equal(changed.source_numerator,output.source_numerator),
            actual_semantic_mass_changed=not torch.equal(changed.semantic_weight,output.semantic_weight),
            actual_output_numerator_RMS=float(changed.source_numerator.square().mean().sqrt()),
            original_output_numerator_RMS=float(output.source_numerator.square().mean().sqrt()),
            no_source_weight_or_correspondence_change=True))
        return changed

    def close(self):
        self.handle.remove()
