"""One source feature per camera ray, consumed by the unchanged native ProjGrid."""
import torch

from ...modules.ray_group_matching import RayGroupConditioner
from ...modules.ray_group_query import native_ray_membership, pool_ray_queries
from .flow_matching import ImageConditionedProjFlowMatchingCFGTrainer
from .mixins.image_conditioned_proj import compute_multiview_projection_matrices


def project_ray_delta(grid, delta, camera):
    """Project three ray maps with their original cameras, retaining MV4's mean."""
    if delta.ndim != 4 or delta.shape[1:3] != (3, 1024):
        raise ValueError('three native32x32 source-camera delta maps required')
    matrices, _ = compute_multiview_projection_matrices(
        camera['transform_matrix'], camera['distance'], grid.front_view_transform_matrix)
    total = None
    for source in range(3):
        view = source + 1
        value = grid(delta[:, source].float().reshape(delta.shape[0], 32, 32, -1),
            camera['camera_angle_x'][:, view], camera['distance'][:, view],
            camera['mesh_scale'], matrices[:, view]).float()
        total = value if total is None else total + value
    return total / 4


class RayGroupImageConditionedProjFlowMatchingCFGTrainer(ImageConditionedProjFlowMatchingCFGTrainer):
    def init_models_and_more(self, **kwargs):
        # Retain extractor -> C1/R initialization -> original LoRA/DDP setup.
        if self.image_cond_model is None:
            self._init_image_cond_model()
        if self.deform_conditioner is None:
            if self.deform_conditioner_config is None:
                raise ValueError('ray-group trainer requires an explicit SS conditioner')
            self.deform_conditioner = RayGroupConditioner(**self.deform_conditioner_config).cuda()
            self.models['deform_conditioner'] = self.deform_conditioner
        super().init_models_and_more(**kwargs)

    def _inject_deform_transport(self, condition, transport, coords, *, source_views=3,
                                control='normal', anchor_condition=None, camera_info=None):
        if coords is not None or anchor_condition is None or camera_info is None or source_views != 3:
            raise ValueError('ray-group conditioner requires dense SS with one edited RGB and three source RGBs')
        module = (self.training_models.get('deform_conditioner', self.deform_conditioner)
                  if hasattr(self, 'training_models') else self.deform_conditioner)
        grid = self._image_cond_module.proj_grid
        key = 'proj_semantic' if 'proj_semantic' in condition else 'proj'
        native = condition[key]
        patch = transport['patch_grid'].flatten(1).long()
        if native.shape[:2] != patch.shape or native.shape[-1] != 1024:
            raise ValueError('native SS grid must align with original target anchor patch IDs')
        required = {'target_keys', 'source_keys', 'source_values', 'source_valid', 'target_query', 'target_valid'}
        if not required <= transport.keys():
            raise ValueError('explicit frozen RGB/Omega/DINO transport inputs required')
        if {'spatial_indices', 'semantic_indices', 'matching_logits'} & transport.keys():
            raise ValueError('ray queries cannot consume GT spatial addresses or old anchor-patch match caches')
        membership = native_ray_membership(grid,
            transform_matrix=camera_info['transform_matrix'], distance=camera_info['distance'],
            camera_angle_x=camera_info['camera_angle_x'], mesh_scale=camera_info['mesh_scale'], views=(1, 2, 3))
        # Input edited-RGBA support only; no target depth/surface is available.
        observed = transport['target_valid'].gather(1, patch)
        membership = torch.where(observed[:, None], membership, torch.full_like(membership, -1))
        gather = patch[..., None].expand(-1, -1, 1024)
        arguments = dict(native_voxels=native,
            target_omega_voxels=transport['target_keys'].gather(1, gather),
            target_native_voxels=transport['target_query'].gather(1, gather), ray_ids=membership,
            source_keys=transport['source_keys'], source_values=transport['source_values'],
            source_valid=transport['source_valid'], control=control)
        if self.deform_conditioner.correspondence_mode == 'native-spatial':
            matrices, _ = compute_multiview_projection_matrices(
                camera_info['transform_matrix'], camera_info['distance'], grid.front_view_transform_matrix)
            keys, values = [], []
            for source in range(3):
                view = source + 1
                def projected(name):
                    maps = transport[name][:, source].float().reshape(native.shape[0], 32, 32, 1024)
                    volume = grid(maps, camera_info['camera_angle_x'][:, view], camera_info['distance'][:, view],
                        camera_info['mesh_scale'], matrices[:, view])
                    return pool_ray_queries(volume, membership, num_rays=1024).values.flatten(1, 2)
                keys.append(projected('source_keys')); values.append(projected('source_values'))
            arguments.update(spatial_keys=torch.stack(keys, dim=2), spatial_values=torch.stack(values, dim=2))
        # New semantic support is retrieved for each pooled Omega ray query.
        output = module(**arguments)
        projected = project_ray_delta(grid, output.map_delta, camera_info)
        if projected.shape != native.shape:
            raise ValueError('native ray projection and denoiser condition shapes differ')
        condition[key] = native + projected.to(native.dtype)
        self._last_deform_condition = output.attention
        return condition
