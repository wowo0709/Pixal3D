"""The last four design controls of the finite shared-pipeline protocol."""
import hashlib
import json
from pathlib import Path

from .r_mediated_native_recipe import common_contract, validate_config as validate_reference

NAME = 'UnifiedDesignNativeFlowTrainer'
SCHEMA = 'unified-module-design-native-v1'
SOURCE_COUNTS = dict(FULL_QK_ONLY=19,FULL_VOXEL_Q=19,FULL_MEAN_MATCH=19,C_DECODER_NO_R=5)
VARIANTS = tuple(SOURCE_COUNTS)


def validate_config(cfg, phase, reference=None):
    e,t=cfg['experiment'],cfg['trainer']['args']
    if reference is None:
        p=Path(e['reference_science_config'])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=e['reference_science_config_SHA256']:
            raise ValueError('bound full candidate recipe changed')
        reference=json.loads(p.read_text())
    if validate_reference(reference,'science')!='R_raw_topk':
        raise ValueError('common native full R_raw_topk reference required')
    variant=e.get('native_variant')
    no_R=variant=='C_DECODER_NO_R'
    if (phase not in {'proof','capacity','science'} or variant not in VARIANTS
            or cfg['trainer']['name']!=NAME or e.get('conditioning_schema')!=SCHEMA
            or e.get('module_pipeline_schema')!='unified-module-pipeline-v2'
            or e.get('design_control')!=variant or e.get('actual_source_tensors')!=SOURCE_COUNTS[variant]
            or e.get('retargeter_physically_absent') is not no_R
            or e.get('physical_deformation_query_channels')!=0
            or e.get('engineering_only') is not (phase!='science')
            or cfg['use_wandb'] is not (phase=='science')
            or ('ray_feature_utility' in t) is no_R
            or (not no_R and t['ray_feature_utility']!=reference['trainer']['args']['ray_feature_utility'])
            or t.get('model_learning_rates')!=reference['trainer']['args']['model_learning_rates']
            or t.get('deform_conditioner')!=reference['trainer']['args']['deform_conditioner']
            or any(e.get(k)!=reference['experiment'].get(k) for k in ['training_assets','epochs',
                'updates_per_epoch','target_modality','global_batch','world_size','microbatch_per_gpu',
                'validate_at_start','initial_function','additional_source_teacher_supervision',
                'source_teacher_updates','native_task_objective','initial_reference_evaluation'])
            or common_contract(cfg)!=common_contract(reference)
            or cfg['dataset']!=reference['dataset'] or cfg['validation']!=reference['validation']
            or any(cfg.get(k)!=reference.get(k) for k in ['load_dir','load_ckpt','ckpt','auto_retry',
                'num_gpus','num_nodes','tryrun','profile'])):
        raise ValueError('single design change on the same native SS96/480/RGB/LoRA recipe required')
    if (e.get('appearance_mode')!=('single-correspondence-Top1-feature' if no_R else 'raw-topk-plus-feature')
            or e.get('residual_logit_bound')!=(None if no_R else 1.)):
        raise ValueError('declared Top1/no-R or full learned attention configuration required')
    return variant


def build_trainer(variant):
    if variant not in VARIANTS:raise ValueError('explicit common-pipeline design arm required')
    import torch
    from pixal3d.modules.unified_design_controls import UnifiedFullDesignConditioner,UnifiedTop1NoRetargetingConditioner
    from pixal3d.modules.r_mediated_source_contrast import RMediatedSourceContrastConditioner
    from pixal3d.modules.r_mediated_top1_source import RMediatedTop1SourceConditioner
    from pixal3d.trainers.flow_matching.member_feature_source_ray import verify_utility
    from pixal3d.trainers.flow_matching.member_posterior_source_ray import MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer
    from .r_mediated_source_pilot import assert_R_only

    class DesignTrainer(MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer):
        def init_models_and_more(self, *, ray_feature_utility=None, **kwargs):
            no_R=variant=='C_DECODER_NO_R'
            if no_R and ray_feature_utility is not None:
                raise ValueError('no unused attention utility in actual Top1/no-R arm')
            if self.image_cond_model is None:self._init_image_cond_model()
            if self.deform_conditioner is not None:raise ValueError('fresh source module required')
            devices=[torch.cuda.current_device()] if torch.cuda.is_initialized() else []
            with torch.random.fork_rng(devices=devices):
                module=(UnifiedTop1NoRetargetingConditioner(**self.deform_conditioner_config) if no_R
                    else UnifiedFullDesignConditioner(design=variant,**self.deform_conditioner_config))
            with torch.random.fork_rng(devices=devices):
                reference=(RMediatedTop1SourceConditioner(**self.deform_conditioner_config) if no_R
                    else RMediatedSourceContrastConditioner(appearance_mode='raw-topk-plus-feature',
                        residual_logit_bound=1.,**self.deform_conditioner_config))
            if not no_R:
                verify_utility(ray_feature_utility)
                state=torch.load(ray_feature_utility['checkpoint'],map_location='cpu',weights_only=True)
                module.load_utility(state);reference.load_utility(state)
            common={n:p for n,p in module.named_parameters() if p.requires_grad}
            previous=dict(reference.named_parameters())
            if (len(common)!=SOURCE_COUNTS[variant]
                    or any(n not in previous or not torch.equal(p,previous[n]) for n,p in common.items())):
                raise ValueError('actual shared R/use/fusion/head initialization differs from original reference')
            if no_R and (module.retargeter is not None or hasattr(module,'ray_readout')):
                raise ValueError('no-R Top1 must physically have neither R nor readout')
            del reference
            module._actual_initial_common_parameters_equal_reference=True
            def check_delivery(source, inputs, output):
                if no_R:
                    if source.retargeter is not None or output.attention.retarget_residual.count_nonzero():
                        raise ValueError('actual no-R delivery contains a learned correction')
                else:
                    assert_R_only(source,output)
            module.register_forward_hook(check_delivery)
            self.deform_conditioner=module.cuda();self.models['deform_conditioner']=module
            super().init_models_and_more(**kwargs)
    return DesignTrainer
