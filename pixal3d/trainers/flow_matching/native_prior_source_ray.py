"""Native four-view initialization with target-conditioned source-ray updates."""
import torch

from ...modules.native_prior_source_ray import project_native_prior_source_delta
from ...modules.target_first_ray import add_source_ray_correction
from ...modules.target_first_observed_query import ObservedQueryTargetFirstRayConditioner
from ...modules.target_first_observed_weighted import ObservedQueryWeightedTargetFirstRayConditioner
from ...modules.target_first_observed_spatial import (
    ObservedQuerySpatialTargetFirstRayConditioner, native_spatial_ray_candidates,
)
from ...modules.ray_group_query import native_ray_membership
from .target_first_observed_query import ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer
from .mixins.image_conditioned_proj import ImageConditionedProjMixin, compute_multiview_projection_matrices


class NativePriorSourceRayImageConditionedProjFlowMatchingCFGTrainer(ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer):
    def init_models_and_more(self, *, ffn_chunk_assets=None, **kwargs):
        if self.image_cond_model is None:
            self._init_image_cond_model()
        if self.deform_conditioner is None:
            if self.deform_conditioner_config is None:
                raise ValueError('explicit native-prior source-ray conditioner required')
            mode = self.deform_conditioner_config.get('ray_mode')
            choices = {'semantic-top1': ObservedQueryTargetFirstRayConditioner,
                       'semantic-attention': ObservedQueryTargetFirstRayConditioner,
                       'semantic-weighted': ObservedQueryWeightedTargetFirstRayConditioner,
                       'native-spatial-attention': ObservedQuerySpatialTargetFirstRayConditioner}
            if mode not in choices:
                raise ValueError('explicit observed-query readout required')
            self.deform_conditioner = choices[mode](**self.deform_conditioner_config).cuda()
            self.models['deform_conditioner'] = self.deform_conditioner
        super().init_models_and_more(**kwargs)
        self._native_ffn_chunk_report = None
        if ffn_chunk_assets is not None:
            from ...modules.checkpointed_ffn_chunks import install_checkpointed_ffn_chunks
            self._native_ffn_chunk_report = install_checkpointed_ffn_chunks(
                self.models['denoiser'], chunk_assets=ffn_chunk_assets)

    def _encode_deform_views(self, images, camera_info, transport, *, control='normal'):
        if (images.ndim != 5 or images.shape[1] != 4 or camera_info.get('coords') is not None
                or control != 'normal' or self.image_attn_mode != 'proj'):
            raise ValueError('native dense SS with edited target RGB and three M0 source RGBs required')
        # This is the original full four-view global/projection/negative path.
        # No learned gate or source module is allowed to redefine its init.
        native, negative = ImageConditionedProjMixin._encode_deform_views(
            self, images, camera_info, None)
        if transport is None:
            return native, negative
        required = {'target_keys', 'source_keys', 'source_values', 'source_valid',
                    'target_query', 'target_valid', 'patch_grid'}
        if (not required <= transport.keys()
                or {'spatial_indices', 'semantic_indices', 'matching_logits', 'deformation'} & transport.keys()):
            raise ValueError('input RGB/Omega and camera addresses only; no GT or cached correspondence routes')
        extractor = getattr(self, '_image_cond_module', self.image_cond_model)
        matrices, _ = compute_multiview_projection_matrices(camera_info['transform_matrix'],
            camera_info['distance'], extractor.fixed_projection_transform)
        target, _ = self.encode_image_proj(images[:, 0],
            camera_angle_x=camera_info['camera_angle_x'][:, 0], distance=camera_info['distance'][:, 0],
            mesh_scale=camera_info['mesh_scale'], transform_matrix=matrices[:, 0], coords=None)
        membership = native_ray_membership(extractor.proj_grid,
            transform_matrix=camera_info['transform_matrix'], distance=camera_info['distance'],
            camera_angle_x=camera_info['camera_angle_x'], mesh_scale=camera_info['mesh_scale'], views=(1, 2, 3))
        patch = transport['patch_grid'].flatten(1).long()
        if (target['proj'].shape != (images.shape[0], 4096, 1024)
                or native['proj'].shape != target['proj'].shape
                or patch.shape != (images.shape[0], 4096)
                or transport['target_valid'].shape != (images.shape[0], 1024)
                or transport['target_valid'].dtype != torch.bool):
            raise ValueError('native SS16 grid and explicit input-alpha32patch query support required')
        gather = patch[..., None].expand(-1, -1, 1024)
        module = (self.training_models.get('deform_conditioner', self.deform_conditioner)
                  if hasattr(self, 'training_models') else self.deform_conditioner)
        kwargs = dict(native_voxels=target['proj'],
            target_omega_voxels=transport['target_keys'].gather(1, gather),
            target_native_voxels=transport['target_query'].gather(1, gather),
            ray_ids=membership, observed_query=transport['target_valid'].gather(1, patch),
            source_keys=transport['source_keys'], source_values=transport['source_values'],
            source_valid=transport['source_valid'], control=control)
        if self.deform_conditioner.correspondence_mode == 'native-spatial':
            keys, values = native_spatial_ray_candidates(extractor.proj_grid,
                transport['source_keys'], transport['source_values'], membership, camera_info)
            kwargs.update(spatial_keys=keys, spatial_values=values)
        output = module(**kwargs)
        correction = project_native_prior_source_delta(extractor.proj_grid,
            output.source_numerator, output.semantic_weight, transport['source_values'], camera_info)
        self._last_deform_condition = output.attention
        self._last_source_use_probability = output.use_probability.detach()
        return add_source_ray_correction(native, correction), negative
