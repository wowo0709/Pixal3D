"""Matched RAW semantic and native spatial Top1 SS arms."""
import hashlib
import json
from pathlib import Path

from .r_mediated_native_recipe import common_contract, validate_config as validate_reference

NAME = 'RawCorrespondenceNativeFlowTrainer'
SCHEMA = 'unified-module-raw-correspondence-native-v1'
VARIANTS = {'C_RAW':'semantic','C_SPATIAL':'native-spatial','C_D1_COS':'semantic'}
MATCHING = {'C_RAW':'RAW-cosine','C_SPATIAL':'RAW-cosine','C_D1_COS':'D1-cosine'}


def validate_config(cfg, phase, reference=None):
    e,t=cfg['experiment'],cfg['trainer']['args']
    if reference is None:
        p=Path(e['reference_science_config'])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=e['reference_science_config_SHA256']:
            raise ValueError('bound full candidate recipe changed')
        reference=json.loads(p.read_text())
    if validate_reference(reference,'science')!='R_raw_topk':
        raise ValueError('shared raw-TopK-plus-feature R-only full candidate required')
    variant=e.get('native_variant')
    if (phase not in {'proof','capacity','science'} or variant not in VARIANTS
            or cfg['trainer']['name']!=NAME or e.get('conditioning_schema')!=SCHEMA
            or e.get('module_pipeline_schema')!='unified-module-pipeline-v2'
            or e.get('readout_selection')!='top1' or e.get('correspondence_mode')!=VARIANTS[variant]
            or e.get('appearance_mode')!='single-correspondence-Top1-feature' or e.get('matching_mode')!=MATCHING[variant] or e.get('residual_logit_bound') is not None
            or e.get('actual_source_tensors')!=10 or e.get('learned_projection_tensors')!=0 or e.get('matching_decoder_physically_absent') is not (MATCHING[variant]=='RAW-cosine') or e.get('matching_scalar_score_physically_absent') is not True
            or e.get('physical_deformation_query_channels')!=0
            or e.get('engineering_only') is not (phase!='science')
            or cfg['use_wandb'] is not (phase=='science')
            or 'ray_feature_utility' in t
            or t.get('model_learning_rates')!=reference['trainer']['args']['model_learning_rates']
            or t.get('deform_conditioner')!=reference['trainer']['args']['deform_conditioner']
            or any(e.get(k)!=reference['experiment'].get(k) for k in ['training_assets','epochs',
                'updates_per_epoch','target_modality','global_batch','world_size','microbatch_per_gpu',
                'validate_at_start','initial_function','additional_source_teacher_supervision',
                'source_teacher_updates','native_task_objective','initial_reference_evaluation',
                ])
            or common_contract(cfg)!=common_contract(reference)
            or cfg['dataset']!=reference['dataset'] or cfg['validation']!=reference['validation']
            or any(cfg.get(k)!=reference.get(k) for k in ['load_dir','load_ckpt','ckpt','auto_retry',
                'num_gpus','num_nodes','tryrun','profile'])):
        raise ValueError('common native SS recipe and physical matcher-free Top1 source10 required')
    return variant


def build_trainer(variant):
    if variant not in VARIANTS:raise ValueError('explicit RAW correspondence control arm required')
    import torch
    from pixal3d.modules.raw_correspondence_top1_source import RawCorrespondenceTop1SourceConditioner
    from pixal3d.modules.d1_cosine_top1_source import D1CosineTop1SourceConditioner
    from pixal3d.modules.r_mediated_top1_source import RMediatedTop1SourceConditioner
    from pixal3d.trainers.flow_matching.member_posterior_source_ray import MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer
    from pixal3d.trainers.flow_matching.member_feature_controls import MemberFeatureControlImageConditionedProjFlowMatchingCFGTrainer

    class RawTrainer(MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer):
        def init_models_and_more(self, **kwargs):
            if 'ray_feature_utility' in kwargs:raise ValueError('RAW Top1 has no learned feature utility')
            if self.image_cond_model is None:self._init_image_cond_model()
            if self.deform_conditioner is not None:raise ValueError('fresh source module required')
            devices=[torch.cuda.current_device()] if torch.cuda.is_initialized() else []
            with torch.random.fork_rng(devices=devices):
                constructor=D1CosineTop1SourceConditioner if MATCHING[variant]=='D1-cosine' else RawCorrespondenceTop1SourceConditioner
                module=constructor(correspondence=VARIANTS[variant],**self.deform_conditioner_config)
            with torch.random.fork_rng(devices=devices):
                reference=RMediatedTop1SourceConditioner(**self.deform_conditioner_config)
            common={n:p for n,p in module.named_parameters() if p.requires_grad}
            previous=dict(reference.named_parameters())
            if len(common)!=10 or any(not torch.equal(p,previous[n]) for n,p in common.items()):
                raise ValueError('actual RAW and decoder Top1 retargeting/use/fusion initial weights differ')
            del reference
            wrong_matcher=(module.q2c is not None if MATCHING[variant]=='RAW-cosine' else
                module.q2c is None or module.q2c.fusion_head is not None or any(p.requires_grad for p in module.q2c.parameters()))
            if wrong_matcher or hasattr(module,'ray_readout') or sum(p.requires_grad for p in module.parameters())!=10:
                raise ValueError('physical RAW correspondence source10 with no matcher or readout required')
            module._actual_initial_core10_equal_R19=True
            self.deform_conditioner=module.cuda();self.models['deform_conditioner']=self.deform_conditioner
            self.feature_control='RAW-native-spatial' if VARIANTS[variant]=='native-spatial' else 'RAW-semantic'
            super().init_models_and_more(**kwargs)

        def _encode_deform_views(self, images, camera_info, transport, *, control='normal'):
            if VARIANTS[variant]=='semantic':
                return super()._encode_deform_views(images,camera_info,transport,control=control)
            # Reuse the exact original camera projection and full-ray pooling;
            # only the selected source module and its RAW scorer differ.
            return MemberFeatureControlImageConditionedProjFlowMatchingCFGTrainer._encode_deform_views(
                self,images,camera_info,transport,control=control)
    return RawTrainer
