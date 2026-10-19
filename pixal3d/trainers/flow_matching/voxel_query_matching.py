"""Dense SS semantic attention with a native query at each projection-grid site."""
from ...modules.voxel_query_matching import VoxelQueryConditioner, fuse_voxel_condition
from .flow_matching import ImageConditionedProjFlowMatchingCFGTrainer


class VoxelQueryImageConditionedProjFlowMatchingCFGTrainer(ImageConditionedProjFlowMatchingCFGTrainer):
    def init_models_and_more(self, **kwargs):
        # Preserve the reference initialization/RNG order: frozen extractor,
        # conditioner, then the shared LoRA/DDP/optimizer setup.
        if self.image_cond_model is None:
            self._init_image_cond_model()
        if self.deform_conditioner is None:
            if self.deform_conditioner_config is None:
                raise ValueError('voxel query trainer requires an explicit SS conditioner')
            self.deform_conditioner = VoxelQueryConditioner(**self.deform_conditioner_config).cuda()
            self.models['deform_conditioner'] = self.deform_conditioner
        super().init_models_and_more(**kwargs)

    def _inject_deform_transport(self, condition, transport, coords, *, source_views=3,
                                control='normal', anchor_condition=None, camera_info=None):
        if coords is not None or anchor_condition is None or camera_info is None:
            raise ValueError('voxel query requires dense SS and original camera/anchor metadata')
        module = (self.training_models.get('deform_conditioner', self.deform_conditioner)
                  if hasattr(self, 'training_models') else self.deform_conditioner)
        key = 'proj_semantic' if 'proj_semantic' in condition else 'proj'
        patch_grid = transport['patch_grid']
        if patch_grid.ndim != 4 or patch_grid.shape[0] != condition[key].shape[0]:
            raise ValueError('original dense target projection patch grid required')
        call = {k: v for k, v in transport.items() if k not in {'patch_grid', 'patch_ids'}}
        output = module(**call, native_query=condition[key], patch_index=patch_grid.flatten(1).long(), control=control)
        self._last_deform_condition = output
        return fuse_voxel_condition(condition, anchor_condition, output, source_views)
