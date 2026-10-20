"""Explicit physical Top1/spatial controls with the frozen native lifecycle."""
import torch

from ...modules.member_feature_controls import Top1FeatureSourceRayConditioner, SpatialFeatureSourceRayConditioner
from ...modules.native_prior_source_ray import project_native_prior_source_delta
from ...modules.ray_group_query import native_ray_membership
from ...modules.target_first_observed_spatial import native_spatial_ray_candidates
from ...modules.target_first_ray import add_source_ray_correction
from .member_feature_source_ray import verify_utility
from .member_posterior_source_ray import MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer
from .mixins.image_conditioned_proj import ImageConditionedProjMixin, compute_multiview_projection_matrices


class MemberFeatureControlImageConditionedProjFlowMatchingCFGTrainer(MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer):
    def init_models_and_more(self, *, feature_control, ray_feature_utility, **kwargs):
        choices = {'top1-feature': (Top1FeatureSourceRayConditioner, 12, 2),
                   'spatial-feature-attention': (SpatialFeatureSourceRayConditioner, 19, 9)}
        if feature_control not in choices:
            raise ValueError('one explicit matched physical feature control required')
        constructor, source_tensors, head_tensors = choices[feature_control]
        self.feature_control = feature_control
        digest = verify_utility(ray_feature_utility)
        if self.image_cond_model is None:
            self._init_image_cond_model()
        if self.deform_conditioner is None:
            self.deform_conditioner = constructor(**self.deform_conditioner_config).cuda()
            if feature_control == 'spatial-feature-attention':
                self.deform_conditioner.load_utility(torch.load(ray_feature_utility['checkpoint'], map_location='cpu', weights_only=True))
            self.models['deform_conditioner'] = self.deform_conditioner
        super().init_models_and_more(**kwargs)
        module = self.deform_conditioner
        if (not isinstance(module, constructor) or module.task_attention is not None
                or len(list(module.ray_readout.parameters())) != head_tensors
                or sum(p.requires_grad for p in module.parameters()) != source_tensors
                or module.retargeter.net[0].in_features != 2048):
            raise ValueError('physical control head/source counts or NoDQ width changed')
        self._ray_feature_utility_initialization = dict(
            status='PASS_COMPLETE_SEVEN_EPOCH_FEATURE_UTILITY_BINDING', SHA256=digest,
            checkpoint=ray_feature_utility['checkpoint'], epochs=7, utility_updates=112,
            utility_initialization_applied=feature_control != 'top1-feature',
            score_selector_physically_absent=feature_control == 'top1-feature',
            source_ray_query=True, historical_target_patch_query_reproduced=False,
            physical_deformation_query_channels=0, feature_control=feature_control)

    def _encode_deform_views(self, images, camera_info, transport, *, control='normal'):
        if self.feature_control == 'top1-feature':
            return super()._encode_deform_views(images, camera_info, transport, control=control)
        if (images.ndim != 5 or images.shape[1] != 4 or camera_info.get('coords') is not None
                or control != 'normal' or self.image_attn_mode != 'proj'):
            raise ValueError('native SS edited target RGB and three M0 source RGBs required')
        native, negative = ImageConditionedProjMixin._encode_deform_views(self, images, camera_info, None)
        if transport is None:
            return native, negative
        required = {'target_keys', 'source_keys', 'source_values', 'source_valid', 'target_query', 'target_valid', 'patch_grid'}
        if (not required <= transport.keys()
                or {'spatial_indices', 'semantic_indices', 'matching_logits', 'deformation'} & transport.keys()):
            raise ValueError('input features and original cameras only;no GT correspondence routes')
        extractor = getattr(self, '_image_cond_module', self.image_cond_model)
        matrices, _ = compute_multiview_projection_matrices(camera_info['transform_matrix'], camera_info['distance'], extractor.fixed_projection_transform)
        target, _ = self.encode_image_proj(images[:, 0], camera_angle_x=camera_info['camera_angle_x'][:, 0],
            distance=camera_info['distance'][:, 0], mesh_scale=camera_info['mesh_scale'], transform_matrix=matrices[:, 0], coords=None)
        rays = native_ray_membership(extractor.proj_grid, transform_matrix=camera_info['transform_matrix'],
            distance=camera_info['distance'], camera_angle_x=camera_info['camera_angle_x'], mesh_scale=camera_info['mesh_scale'], views=(1, 2, 3))
        patch = transport['patch_grid'].flatten(1).long()
        if (target['proj'].shape != (images.shape[0], 4096, 1024) or native['proj'].shape != target['proj'].shape
                or patch.shape != (images.shape[0], 4096) or transport['target_valid'].shape != (images.shape[0], 1024)
                or transport['target_valid'].dtype != torch.bool):
            raise ValueError('native SS16 grid and input-alpha32patch matching observations required')
        keys, values = native_spatial_ray_candidates(extractor.proj_grid, transport['source_keys'], transport['source_values'], rays, camera_info)
        gather = patch[..., None].expand(-1, -1, 1024)
        module = self.training_models.get('deform_conditioner', self.deform_conditioner) if hasattr(self, 'training_models') else self.deform_conditioner
        output = module(native_voxels=target['proj'], target_omega_voxels=transport['target_keys'].gather(1, gather),
            target_native_voxels=transport['target_query'].gather(1, gather), frame_target_keys=transport['target_keys'],
            target_patch_ids=patch, ray_ids=rays, observed_query=transport['target_valid'].gather(1, patch),
            source_keys=transport['source_keys'], source_values=transport['source_values'], source_valid=transport['source_valid'],
            spatial_keys=keys, spatial_values=values, control=control)
        correction = project_native_prior_source_delta(extractor.proj_grid, output.source_numerator, output.semantic_weight, transport['source_values'], camera_info)
        self._last_deform_condition = output.attention
        self._last_source_use_probability = output.use_probability.detach()
        return add_source_ray_correction(native, correction), negative
