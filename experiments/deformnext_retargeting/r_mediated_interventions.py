"""Inference-only mechanism checks; these are not retrained ablation arms.

Every forward executes a normal shadow on the same inputs, checks fixed
candidate/score/carrier/use contracts, then discards the shadow. No labels or
optimizer are involved. Head interventions return raw and transformed values
under the same controlled probabilities.
"""
import torch
from torch.nn import functional as F

from pixal3d.modules.bounded_member_feature_source_ray import masked_probabilities
from pixal3d.modules.r_mediated_source_contrast import RMediatedSourceContrastConditioner


CONTROLS = ('normal', 'wrong-retrieved-values-fixed-use', 'semantic-prior-only',
            'top1-feature-only', 'no-transformed-value')


def controlled_readout(head, args, output, control):
    """Keep Q/K/values and native support; change only the named readout rule."""
    if head.training or torch.is_grad_enabled():
        raise ValueError('saved-weight interventions require eval and no_grad')
    if control in CONTROLS[:2]:
        return output
    native, context, keys, values, valid, prior = args
    change, probability = output
    if control == 'no-transformed-value':
        return torch.zeros_like(change), probability
    if control == 'semantic-prior-only':
        p = masked_probabilities(prior[:, None].expand(-1, 4, -1), valid)
    elif control == 'top1-feature-only':
        support = valid & torch.isfinite(prior)
        top1 = prior.masked_fill(~support, -torch.inf).argmax(-1, keepdim=True)
        p = torch.zeros_like(prior).scatter(1, top1, support.any(-1, keepdim=True).float())
        p = p[:, None].expand(-1, 4, -1)
    else:
        raise ValueError('explicit saved-weight readout control required')
    values = torch.where(valid[..., None], values, 0.)
    normalized = torch.cat([F.layer_norm(c, (1024,)) for c in values.split(1024, dim=-1)], -1)
    projected = head.value(normalized).reshape(len(native), valid.shape[-1], 4, 64).transpose(1, 2)
    with torch.autocast(device_type=native.device.type, enabled=False):
        combined = (p[..., None] * projected.float()).sum(-2).reshape(len(native), -1)
    return head.output(combined).float(), p


class SavedRMediatedIntervention:
    """Scoped forward hooks preserving the existing native runtime and weights."""

    def __init__(self, module, control):
        if (not isinstance(module, RMediatedSourceContrastConditioner)
                or module.appearance_mode != 'raw-topk-plus-feature' or control not in CONTROLS):
            raise ValueError('explicit raw-TopK R-mediated module and control required')
        self.module, self.control = module, control
        self.events = []
        self.in_shadow = False
        self.shadow = None
        self.parameter_versions = None
        self.handles = [module.register_forward_pre_hook(self.before, with_kwargs=True),
            module.ray_readout.register_forward_hook(self.readout),
            module.register_forward_hook(self.after, with_kwargs=True)]

    def readout(self, module, args, output):
        if self.in_shadow:
            return output
        return controlled_readout(module, args, output, self.control)

    def before(self, module, args, kwargs):
        if (module.training or torch.is_grad_enabled() or args
                or kwargs.get('control', 'normal') != 'normal' or self.shadow is not None):
            raise ValueError('native eval/no_grad keyword-only normal entry required')
        current = {n: p._version for n, p in module.named_parameters()}
        if self.parameter_versions is None:
            self.parameter_versions = current
        if current != self.parameter_versions:
            raise ValueError('parameters changed during read-only mechanism diagnosis')
        self.cpu_rng = torch.random.get_rng_state()
        self.cuda_rng = (torch.cuda.get_rng_state() if kwargs['native_voxels'].is_cuda else None)
        self.in_shadow = True
        try:
            normal = module.forward(**kwargs)
            self.shadow = (normal, module.last_feature_probabilities,
                           module.last_R_input_appearance, module.last_R_only_output,
                           module.last_source_reference_rows)
        finally:
            self.in_shadow = False
        actual = dict(kwargs)
        if self.control == 'wrong-retrieved-values-fixed-use':
            actual['control'] = self.control
        return args, actual

    def after(self, module, args, kwargs, output):
        normal, probability, appearance, residual, selected = self.shadow
        self.shadow = None
        fixed = dict(
            candidate_indices=torch.equal(normal.attention.candidate_indices, output.attention.candidate_indices),
            semantic_prior=torch.equal(normal.attention.matching_logits, output.attention.matching_logits),
            Top1_carrier=torch.equal(normal.attention.base_evidence, output.attention.base_evidence),
            source_use=torch.equal(normal.use_probability, output.use_probability),
            semantic_weight=torch.equal(normal.semantic_weight, output.semantic_weight),
            source_rows=torch.equal(selected, module.last_source_reference_rows),
            CPU_RNG=torch.equal(self.cpu_rng, torch.random.get_rng_state()),
            CUDA_RNG=(self.cuda_rng is None or torch.equal(self.cuda_rng, torch.cuda.get_rng_state())),
            parameter_versions=self.parameter_versions == {n: p._version for n, p in module.named_parameters()},
        )
        if not all(fixed.values()):
            raise ValueError('saved-weight intervention changed fixed inputs: ' + str(fixed))
        p_equal = torch.equal(probability[1], module.last_feature_probabilities[1])
        if self.control in {'normal', 'wrong-retrieved-values-fixed-use', 'no-transformed-value'} and not p_equal:
            raise ValueError('value-only intervention changed attention probabilities')
        R_equal = torch.equal(residual, module.last_R_only_output)
        if self.control == 'normal' and (not R_equal or not p_equal
                or not torch.equal(normal.source_numerator, output.source_numerator)
                or not torch.equal(appearance, module.last_R_input_appearance)):
            raise ValueError('normal shadow changed actual native module output')
        p = module.last_feature_probabilities[1]
        if self.control == 'top1-feature-only' and (not ((p == 0) | (p == 1)).all()
                or not ((p.sum(-1) == 0) | (p.sum(-1) == 1)).all()):
            raise ValueError('Top1-only must select one correspondence, never uniform weights')
        self.events.append(dict(control=self.control, fixed=fixed, active_source_rays=len(selected),
            attention_probabilities_exact_to_normal=p_equal,
            R_input_appearance_exact_to_normal=torch.equal(appearance, module.last_R_input_appearance),
            R_output_exact_to_normal=R_equal,
            R_output_difference_L1=float((residual-module.last_R_only_output).abs().sum()),
            transformed_feature_zero=(self.control == 'no-transformed-value'),
            Top1_only_not_uniform=(self.control == 'top1-feature-only')))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

