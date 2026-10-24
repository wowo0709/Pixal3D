"""Physical controls of the final shared-evidence candidates, without selection.

These constructors launch no training/evaluation. Native proof, capacity and
checkpoint/evaluator binding must precede actual science execution.
"""


NAME = 'SharedEvidenceAblationNativeFlowTrainer'
VARIANTS = {
    'A_FIXED_UNIFORM': ('E_FIXED_RAW', 'fixed-prior', 'uniform-k', 10),
    'A_FIXED_NO_R': ('E_FIXED_RAW', 'fixed-prior', 'without-retargeting', 5),
    'A_LEARNED_NO_R': ('E_LEARNED_RAW', 'learned-anchored', 'without-retargeting', 12),
    'A_LEARNED_NO_LOGIT_PRIOR': ('E_LEARNED_RAW', 'learned-anchored', 'without-logit-prior', 17),
}


def construct_conditioner(variant, kwargs):
    if variant not in VARIANTS:
        raise ValueError('one explicit physical shared-evidence control required')
    import torch
    from .consistent_evidence_native_recipe import construct_conditioner as parent_constructor
    from pixal3d.modules.shared_evidence_ablation import SharedEvidenceAblationConditioner
    parent, evidence_mode, ablation, count = VARIANTS[variant]
    devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        reference = parent_constructor(parent, kwargs)
    with torch.random.fork_rng(devices=devices):
        module = SharedEvidenceAblationConditioner(evidence_mode=evidence_mode, ablation=ablation, **kwargs)
    actual = {n: p for n, p in module.named_parameters() if p.requires_grad}
    original = {n: p for n, p in reference.named_parameters() if p.requires_grad}
    retained = {n: p for n, p in original.items() if not (ablation == 'without-retargeting' and n.startswith('retargeter.'))}
    if (len(actual) != count or set(actual) != set(retained)
            or any(not torch.equal(p, actual[n]) for n, p in retained.items())
            or any(p.requires_grad for p in module.q2c.parameters())
            or (ablation == 'without-retargeting' and module.retargeter is not None)
            or (ablation == 'uniform-k' and module.ray_readout is not None)):
        raise ValueError('same actual retained cold weights and only the declared physical change required')
    module._actual_initial_retained_parameters_equal_parent = True
    module._parent_initial_core10_equal_Top1 = reference._actual_initial_core10_equal_Top1
    return module


def build_trainer(variant):
    """Use the existing native SS/LoRA trainer with only the declared source."""
    if variant not in VARIANTS:
        raise ValueError('one explicit physical shared-evidence control required')
    from pixal3d.trainers.flow_matching.member_posterior_source_ray import MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer

    class SharedControlTrainer(MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer):
        def init_models_and_more(self, **kwargs):
            if 'ray_feature_utility' in kwargs:
                raise ValueError('no value transform or feature-teacher initializer in shared raw evidence')
            if self.image_cond_model is None:
                self._init_image_cond_model()
            if self.deform_conditioner is not None:
                raise ValueError('fresh source module required')
            module = construct_conditioner(variant, self.deform_conditioner_config)
            self.deform_conditioner = module.cuda()
            self.models['deform_conditioner'] = self.deform_conditioner
            super().init_models_and_more(**kwargs)
    return SharedControlTrainer
