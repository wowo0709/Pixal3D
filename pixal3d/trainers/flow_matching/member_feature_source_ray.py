"""Explicit four-head source-ray trainer; native LoRA/optimizer lifecycle."""
import hashlib
import json
from pathlib import Path

import torch

from ...modules.member_feature_source_ray import MemberFeatureSourceRayConditioner
from .member_posterior_source_ray import MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer


def verify_utility(spec):
    if set(spec) != {'checkpoint', 'SHA256', 'completed'}:
        raise ValueError('explicit complete historical utility provenance required')
    checkpoint = Path(spec['checkpoint'])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    completed = json.loads(Path(spec['completed']).read_text())
    if (digest != spec['SHA256']
            or completed['status'] != 'PASS_FIXED_SEVEN_EPOCH_TARGET_PATCH_UTILITY_TRAINING'
            or completed['epochs'] != 7 or completed['optimizer_updates'] != 112
            or completed['pure_utility_export_SHA256'] != digest):
        raise ValueError('same complete seven-epoch historical feature utility required')
    return digest


class MemberFeatureSourceRayImageConditionedProjFlowMatchingCFGTrainer(MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer):
    def init_models_and_more(self, *, ray_feature_utility, **kwargs):
        digest = verify_utility(ray_feature_utility)
        if self.image_cond_model is None:
            self._init_image_cond_model()
        if self.deform_conditioner is None:
            self.deform_conditioner = MemberFeatureSourceRayConditioner(**self.deform_conditioner_config).cuda()
            self.deform_conditioner.load_utility(torch.load(
                ray_feature_utility['checkpoint'], map_location='cpu', weights_only=True))
            self.models['deform_conditioner'] = self.deform_conditioner
        super().init_models_and_more(**kwargs)
        module = self.deform_conditioner
        if (not isinstance(module, MemberFeatureSourceRayConditioner)
                or module.task_attention is not None
                or len(list(module.ray_readout.parameters())) != 9
                or sum(p.requires_grad for p in module.parameters()) != 19
                or module.retargeter.net[0].in_features != 2048):
            raise ValueError('actual nine-tensor QKV head and ten-tensor NoDQ source module required')
        self._ray_feature_utility_initialization = dict(
            status='PASS_COMPLETE_SEVEN_EPOCH_FEATURE_UTILITY_INITIALIZATION',
            SHA256=digest, checkpoint=ray_feature_utility['checkpoint'],
            epochs=7, utility_updates=112, joint_feature_head_trainable=True,
            source_ray_query=True, historical_target_patch_query_reproduced=False,
            physical_deformation_query_channels=0)
