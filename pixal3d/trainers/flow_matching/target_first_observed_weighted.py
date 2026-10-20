"""Native trainer for the same-query fixed-score Top64 control."""
from ...modules.target_first_observed_weighted import ObservedQueryWeightedTargetFirstRayConditioner
from .target_first_observed_query import ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer


class ObservedQueryWeightedTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer(ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer):
    def init_models_and_more(self, **kwargs):
        if self.image_cond_model is None:
            self._init_image_cond_model()
        if self.deform_conditioner is None:
            if self.deform_conditioner_config is None:
                raise ValueError('explicit fixed-score Top64 conditioner required')
            self.deform_conditioner = ObservedQueryWeightedTargetFirstRayConditioner(
                **self.deform_conditioner_config).cuda()
            self.models['deform_conditioner'] = self.deform_conditioner
        super().init_models_and_more(**kwargs)
