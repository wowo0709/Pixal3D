"""Matched physical Top1-V/output and frozen-prior TopK native SS arms."""
import hashlib
import json
from pathlib import Path

from .r_mediated_native_recipe import common_contract, validate_config as validate_reference

NAME = 'ProjectedModuleNativeFlowTrainer'
SCHEMA = 'unified-module-projected-native-v1'
VARIANTS = {'C_TOP1_V':'top1', 'C_FIXED_TOPK':'fixed-topk'}


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
            or e.get('module_pipeline_schema')!='unified-module-pipeline-v1'
            or e.get('readout_selection')!=VARIANTS[variant]
            or e.get('actual_source_tensors')!=12 or e.get('learned_projection_tensors')!=2
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
                'appearance_mode','residual_logit_bound'])
            or common_contract(cfg)!=common_contract(reference)
            or cfg['dataset']!=reference['dataset'] or cfg['validation']!=reference['validation']
            or any(cfg.get(k)!=reference.get(k) for k in ['load_dir','load_ckpt','ckpt','auto_retry',
                'num_gpus','num_nodes','tryrun','profile'])):
        raise ValueError('common native SS recipe and physical two-projection source12 required')
    return variant


def build_trainer(variant):
    if variant not in VARIANTS:raise ValueError('explicit projected control arm required')
    import torch
    from pixal3d.modules.r_mediated_source_contrast import RMediatedSourceContrastConditioner
    from pixal3d.modules.matched_projected_readout import MatchedProjectedReadout
    from pixal3d.trainers.flow_matching.member_posterior_source_ray import MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer

    class ProjectedTrainer(MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer):
        def init_models_and_more(self, **kwargs):
            if 'ray_feature_utility' in kwargs:
                raise ValueError('projected controls have no learned utility to initialize')
            if self.image_cond_model is None:self._init_image_cond_model()
            if self.deform_conditioner is not None:raise ValueError('fresh source module required')
            devices=[torch.cuda.current_device()] if torch.cuda.is_initialized() else []
            with torch.random.fork_rng(devices=devices):
                module=RMediatedSourceContrastConditioner(appearance_mode='raw-topk-plus-feature',
                    residual_logit_bound=1.,**self.deform_conditioner_config)
                original={n:p.detach().clone() for n,p in module.named_parameters()
                          if p.requires_grad and not n.startswith('ray_readout.')}
                value,output=module.ray_readout.value,module.ray_readout.output
                module.ray_readout=MatchedProjectedReadout(module.ray_readout,mode=VARIANTS[variant])
            parameters=dict(module.named_parameters())
            if (len(original)!=10 or any(not torch.equal(p,parameters[n]) for n,p in original.items())
                    or module.ray_readout.value is not value or module.ray_readout.output is not output
                    or sum(p.requires_grad for p in module.parameters())!=12):
                raise ValueError('actual common core10 and shared V/output initialization changed')
            module._actual_initial_core10_equal_R19=True
            module._actual_initial_V_output_equal_full=True
            self.deform_conditioner=module.cuda();self.models['deform_conditioner']=self.deform_conditioner
            super().init_models_and_more(**kwargs)
    return ProjectedTrainer
