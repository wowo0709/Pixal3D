"""Native MV4 SS carrier with member-preserving semantic source-ray updates."""
import torch

from ...modules.member_posterior_source_ray import MemberPosteriorSourceRayConditioner
from ...modules.native_prior_source_ray import project_native_prior_source_delta
from ...modules.ray_group_query import native_ray_membership
from ...modules.target_first_ray import add_source_ray_correction
from .native_prior_source_ray import NativePriorSourceRayImageConditionedProjFlowMatchingCFGTrainer
from .mixins.image_conditioned_proj import ImageConditionedProjMixin,compute_multiview_projection_matrices


class MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer(NativePriorSourceRayImageConditionedProjFlowMatchingCFGTrainer):
    def init_models_and_more(self,**kwargs):
        if self.image_cond_model is None:
            self._init_image_cond_model()
        if self.deform_conditioner is None:
            if self.deform_conditioner_config is None:
                raise ValueError('explicit member-posterior source-ray configuration required')
            self.deform_conditioner=MemberPosteriorSourceRayConditioner(**self.deform_conditioner_config).cuda()
            self.models['deform_conditioner']=self.deform_conditioner
        super().init_models_and_more(**kwargs)

    def _encode_deform_views(self,images,camera_info,transport,*,control='normal'):
        if (images.ndim!=5 or images.shape[1]!=4 or camera_info.get('coords') is not None
                or control!='normal' or self.image_attn_mode!='proj'):
            raise ValueError('native SS edited target RGB and three M0 source RGBs required')
        native,negative=ImageConditionedProjMixin._encode_deform_views(self,images,camera_info,None)
        if transport is None:
            return native,negative
        required={'target_keys','source_keys','source_values','source_valid','target_query','target_valid','patch_grid'}
        if (not required<=transport.keys()
                or {'spatial_indices','semantic_indices','matching_logits','deformation'}&transport.keys()):
            raise ValueError('input RGB/Omega and camera patch IDs only; no GT routes')
        extractor=getattr(self,'_image_cond_module',self.image_cond_model)
        matrices,_=compute_multiview_projection_matrices(camera_info['transform_matrix'],camera_info['distance'],extractor.fixed_projection_transform)
        target,_=self.encode_image_proj(images[:,0],camera_angle_x=camera_info['camera_angle_x'][:,0],
            distance=camera_info['distance'][:,0],mesh_scale=camera_info['mesh_scale'],transform_matrix=matrices[:,0],coords=None)
        rays=native_ray_membership(extractor.proj_grid,transform_matrix=camera_info['transform_matrix'],
            distance=camera_info['distance'],camera_angle_x=camera_info['camera_angle_x'],mesh_scale=camera_info['mesh_scale'],views=(1,2,3))
        patch=transport['patch_grid'].flatten(1).long()
        if (target['proj'].shape!=(images.shape[0],4096,1024) or native['proj'].shape!=target['proj'].shape
                or patch.shape!=(images.shape[0],4096) or transport['target_valid'].shape!=(images.shape[0],1024)
                or transport['target_valid'].dtype!=torch.bool):
            raise ValueError('nativeSS16 grid and input-alpha32patch observation support required')
        gather=patch[...,None].expand(-1,-1,1024)
        module=(self.training_models.get('deform_conditioner',self.deform_conditioner)
            if hasattr(self,'training_models') else self.deform_conditioner)
        output=module(native_voxels=target['proj'],target_omega_voxels=transport['target_keys'].gather(1,gather),
            target_native_voxels=transport['target_query'].gather(1,gather),frame_target_keys=transport['target_keys'],
            target_patch_ids=patch,ray_ids=rays,observed_query=transport['target_valid'].gather(1,patch),
            source_keys=transport['source_keys'],source_values=transport['source_values'],source_valid=transport['source_valid'],control=control)
        correction=project_native_prior_source_delta(extractor.proj_grid,output.source_numerator,
            output.semantic_weight,transport['source_values'],camera_info)
        self._last_deform_condition=output.attention
        self._last_source_use_probability=output.use_probability.detach()
        return add_source_ray_correction(native,correction),negative
