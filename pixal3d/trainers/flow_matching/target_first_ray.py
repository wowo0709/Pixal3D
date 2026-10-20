"""Native SS target path plus retargeted source-camera ray residuals."""
import torch

from ...modules.target_first_ray import TargetFirstRayConditioner, add_source_ray_correction
from ...modules.ray_group_query import native_ray_membership
from .flow_matching import ImageConditionedProjFlowMatchingCFGTrainer
from .mixins.image_conditioned_proj import compute_multiview_projection_matrices
from .ray_group_matching import project_ray_delta


class TargetFirstRayImageConditionedProjFlowMatchingCFGTrainer(ImageConditionedProjFlowMatchingCFGTrainer):
    def init_models_and_more(self, **kwargs):
        if self.image_cond_model is None:
            self._init_image_cond_model()
        if self.deform_conditioner is None:
            if self.deform_conditioner_config is None:
                raise ValueError('target-first SS requires its explicit source ray conditioner')
            self.deform_conditioner = TargetFirstRayConditioner(**self.deform_conditioner_config).cuda()
            self.models['deform_conditioner'] = self.deform_conditioner
        super().init_models_and_more(**kwargs)

    def _encode_deform_views(self, images, camera_info, transport, *, control='normal'):
        if (images.ndim != 5 or images.shape[1] != 4 or camera_info.get('coords') is not None
                or control != 'normal' or self.image_attn_mode != 'proj'):
            raise ValueError('dense SS with edited RGB plus three source RGBs and native proj mode required')
        extractor = getattr(self, '_image_cond_module', self.image_cond_model)
        projection, _ = compute_multiview_projection_matrices(
            camera_info['transform_matrix'], camera_info['distance'], extractor.fixed_projection_transform)
        target, negative = self.encode_image_proj(images[:, 0],
            camera_angle_x=camera_info['camera_angle_x'][:, 0],
            distance=camera_info['distance'][:, 0], mesh_scale=camera_info['mesh_scale'],
            transform_matrix=projection[:, 0], coords=None)
        if transport is None:
            return target, negative
        required = {'target_keys', 'source_keys', 'source_values', 'source_valid',
                    'target_query', 'target_valid', 'patch_grid'}
        if not required <= transport.keys() or (
                {'spatial_indices', 'semantic_indices', 'matching_logits', 'deformation'} & transport.keys()):
            raise ValueError('edited RGB/Omega/native values and camera-only addresses;no GT or old anchor matches')
        membership = native_ray_membership(extractor.proj_grid,
            transform_matrix=camera_info['transform_matrix'], distance=camera_info['distance'],
            camera_angle_x=camera_info['camera_angle_x'], mesh_scale=camera_info['mesh_scale'], views=(1, 2, 3))
        # Keep all native camera/grid ray members, including target-background
        # rays. Alpha is not a per-voxel occlusion or target-surface oracle.
        patch = transport['patch_grid'].flatten(1).long()
        if target['proj'].shape != (images.shape[0], 4096, 1024) or patch.shape != (images.shape[0], 4096):
            raise ValueError('native SS16 grid and target32-patch addresses required')
        gather = patch[..., None].expand(-1, -1, 1024)
        module = (self.training_models.get('deform_conditioner', self.deform_conditioner)
                  if hasattr(self, 'training_models') else self.deform_conditioner)
        output = module(native_voxels=target['proj'],
            target_omega_voxels=transport['target_keys'].gather(1, gather),
            target_native_voxels=transport['target_query'].gather(1, gather),
            ray_ids=membership, source_keys=transport['source_keys'],
            source_values=transport['source_values'], source_valid=transport['source_valid'], control=control)
        projected = project_ray_delta(extractor.proj_grid, output.map_delta, camera_info)
        self._last_deform_condition = output.attention
        self._last_source_use_probability = output.use_probability.detach()
        return add_source_ray_correction(target, projected), negative
