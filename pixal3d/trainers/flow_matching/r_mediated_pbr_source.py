"""Native sparse PBR flow consumer for explicit R-only source-camera rays."""
import torch

from ...modules.pbr_source_ray_geometry import (
    project_sparse_pbr_source_delta, sparse_source_ray_membership,
)
from ...modules.r_mediated_pbr_source import MODES, RMediatedPBRSourceConditioner
from ...modules.target_first_ray import add_source_ray_correction
from .member_feature_source_ray import verify_utility
from .mixins.image_conditioned_proj import (
    ImageConditionedProjMixin, compute_multiview_projection_matrices,
)
from .sparse_flow_matching import ImageConditionedProjSparseFlowMatchingCFGTrainer


def pbr_source_kwargs(spec):
    required = {'arm', 'implementation', 'key_dim', 'value_dim', 'query_dim', 'hidden_dim',
        'topk', 'g_max', 'q2c_checkpoint', 'freeze_q2c', 'candidate_aggregation',
        'projection_mode', 'score_temperature', 'ray_chunk_size', 'appearance_mode'}
    if (not required <= set(spec) or set(spec) - required - {'residual_logit_bound'}
            or (spec['arm'], spec['implementation']) != ('c1', 'r9-native-v5')
            or (spec['key_dim'], spec['query_dim'], spec['value_dim']) != (1024, 1024, 2048)
            or spec['hidden_dim'] != 256 or spec['topk'] != 64
            or spec['freeze_q2c'] is not True or not spec['q2c_checkpoint']
            or spec['candidate_aggregation'] != 'top1'
            or spec['projection_mode'] != 'native-bilinear' or spec['appearance_mode'] not in MODES
            or (spec['appearance_mode'] == 'top1-only' and 'residual_logit_bound' in spec)
            or (spec['appearance_mode'] != 'top1-only' and spec.get('residual_logit_bound') != 1.)):
        raise ValueError('explicit native PBR2048 source schema, real frozen matcher and physical NoDQ required')
    keys = ['appearance_mode', 'q2c_checkpoint', 'hidden_dim', 'score_temperature',
        'ray_chunk_size', 'g_max']
    if 'residual_logit_bound' in spec:
        keys.append('residual_logit_bound')
    return {key: spec[key] for key in keys}


class RMediatedPBRNativeTrainer(ImageConditionedProjSparseFlowMatchingCFGTrainer):
    """Keep the native sparse objective/LoRA/CFG/optimizer/save lifecycle."""
    def init_models_and_more(self, *, ray_feature_utility=None, **kwargs):
        spec = self.deform_conditioner_config
        if spec is None:
            raise ValueError('a source PBR trainer requires its explicit source configuration')
        source_kwargs = pbr_source_kwargs(spec)
        extractor_args = self.image_cond_model_config['args']
        if (self.multiview_stage != 'pbr1024' or self.image_attn_mode != 'proj'
                or extractor_args['grid_resolution'] != 64 or extractor_args['image_size'] != 1024
                or extractor_args['use_naf_upsample'] is not True
                or self.model_learning_rates != {'deform_conditioner': 1e-4}):
            raise ValueError('original native PBR64/DINO+NAF extractor and explicit source learning rate required')
        feature_head = spec['appearance_mode'] != 'top1-only'
        if feature_head:
            utility_sha = verify_utility(ray_feature_utility)
        elif ray_feature_utility is not None:
            raise ValueError('physical Top1 must not contain a score utility initializer')
        else:
            utility_sha = None
        if self.image_cond_model is None:
            self._init_image_cond_model()
        if self.deform_conditioner is not None:
            raise ValueError('construct one fresh own-schema PBR source module before native initialization')
        # Match the native constructor RNG for denoiser LoRA and data/noise.
        devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            module = RMediatedPBRSourceConditioner(**source_kwargs)
            if feature_head:
                module.load_utility(torch.load(ray_feature_utility['checkpoint'],
                    map_location='cpu', weights_only=True))
            self.deform_conditioner = module.cuda()
        self.models['deform_conditioner'] = self.deform_conditioner
        super().init_models_and_more(**kwargs)
        expected = 19 if feature_head else 10
        if (sum(p.requires_grad for p in self.models['denoiser'].parameters()) != 360
                or sum(p.requires_grad for p in module.parameters()) != expected
                or module.retargeter.input_dim != 3072 or self._proj_channels != 2048):
            raise ValueError('complete native360 LoRA plus physical PBR source10/19 and 3072-channel R required')
        self._PBR_source_initialization = dict(appearance_mode=spec['appearance_mode'],
            utility_SHA256=utility_sha, source_tensors=expected, value_channels=2048,
            retargeter_input_channels=3072, deformation_query_channels=0,
            constructor_RNG_preserved=True, R_only_delivery=True,
            actual_GPU_prediction_identity_not_proven_by_constructor=True)

    def _pop_deform_transport(self, kwargs):
        transport = super()._pop_deform_transport(kwargs)
        if transport is None:
            return None
        forbidden = {'deformation', 'spatial_indices', 'target_context', 'source_context'}
        if forbidden & set(transport):
            raise ValueError('PBR NoDQ source rays do not consume GT routes or deformation channels')
        # The existing feature dataset also reads legacy cache metadata. Its
        # target-patch support is NOT the new per-source-ray member posterior.
        for name in ['semantic_indices', 'matching_logits', 'candidate_valid', 'patch_grid']:
            transport.pop(name, None)
        return transport

    def _encode_deform_views(self, images, camera_info, transport, *, control='normal'):
        coords = camera_info.get('coords')
        if (images.ndim != 5 or images.shape[1:3] != (4, 3)
                or coords is None or coords.ndim != 2 or coords.shape[1] != 4
                or self.image_attn_mode != 'proj' or control != 'normal'):
            raise ValueError('native sparse PBR support and edited-target RGB plus three source RGBs required')
        native, negative = ImageConditionedProjMixin._encode_deform_views(self, images, camera_info, None)
        if transport is None:
            return native, negative
        required = {'target_keys', 'source_keys', 'source_values', 'source_valid',
            'target_query', 'target_valid', 'patch_ids'}
        if set(transport) != required:
            raise ValueError('PBR source rays require only image/Omega features and aligned native patch IDs')
        if (set(native) != {'global', 'proj'} or native['proj'].shape != (len(coords), 2048)
                or transport['patch_ids'].shape != (len(coords),)):
            raise ValueError('actual sparse native PBR2048 rows and matching patch-address count required')
        extractor = getattr(self, '_image_cond_module', self.image_cond_model)
        query = None
        if self.deform_conditioner.appearance_mode != 'top1-only':
            matrices, _ = compute_multiview_projection_matrices(camera_info['transform_matrix'],
                camera_info['distance'], extractor.fixed_projection_transform)
            target, _ = self.encode_image_proj(images[:, 0],
                camera_angle_x=camera_info['camera_angle_x'][:, 0], distance=camera_info['distance'][:, 0],
                mesh_scale=camera_info['mesh_scale'], transform_matrix=matrices[:, 0], coords=coords)
            if target['proj'].shape != native['proj'].shape:
                raise ValueError('edited target native projected DINO query must match packed PBR support')
            query = target['proj'][..., :1024]
        rays = sparse_source_ray_membership(extractor.proj_grid, coords, camera_info)
        module = (self.training_models.get('deform_conditioner', self.deform_conditioner)
            if hasattr(self, 'training_models') else self.deform_conditioner)
        output = module(native_query_rows=query, target_query=transport['target_query'],
            frame_target_keys=transport['target_keys'], target_patch_ids=transport['patch_ids'],
            target_valid=transport['target_valid'], ray_ids=rays, batch_ids=coords[:, 0],
            source_keys=transport['source_keys'], source_values=transport['source_values'],
            source_valid=transport['source_valid'])
        correction = project_sparse_pbr_source_delta(extractor.proj_grid, output.source_numerator,
            output.semantic_weight, transport['source_values'], coords, camera_info)
        self._last_deform_condition = output.attention
        self._last_source_use_probability = output.use_probability.detach()
        return add_source_ray_correction(native, correction), negative
