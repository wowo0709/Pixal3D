"""An additional physical Top1 control with the already fixed native-task contract."""
import hashlib
import json
from pathlib import Path

from .r_mediated_native_recipe import common_contract, validate_config as validate_reference


NAME = 'RTop1OnlyNativeFlowTrainer'
VARIANT = 'R_top1_only'
SCHEMA = 'r-mediated-native-top1-only-control-v1'


def validate_config(cfg, phase, reference=None):
    if phase not in {'proof', 'capacity', 'science'}:
        raise ValueError('explicit native Top1 phase required')
    e, t = cfg['experiment'], cfg['trainer']['args']
    if reference is None:
        path = Path(e['reference_science_config'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != e['reference_science_config_SHA256']:
            raise ValueError('closed original native R19 reference recipe changed')
        reference = json.loads(path.read_text())
    if validate_reference(reference, 'science') != 'R_feature':
        raise ValueError('actual current R_feature native task recipe is the matching reference')
    if (cfg['trainer']['name'] != NAME or e.get('native_variant') != VARIANT
            or e.get('conditioning_schema') != SCHEMA or e.get('appearance_mode') != 'single-semantic-Top1-feature'
            or e.get('actual_source_tensors') != 10 or e.get('learned_attention_head_tensors') != 0
            or e.get('physical_deformation_query_channels') != 0
            or e.get('additional_source_teacher_supervision') is not False or e.get('source_teacher_updates') != 0
            or e.get('native_task_objective') != 'SS_flow_mse'
            or e.get('engineering_only') is not (phase != 'science') or cfg.get('use_wandb') is not (phase == 'science')
            or e.get('training_assets') != 9149 or e.get('epochs') != 5 or e.get('updates_per_epoch') != 96
            or (e.get('global_batch'), e.get('world_size'), e.get('microbatch_per_gpu')) != (96, 1, 96)
            or e.get('target_modality') != 'rgb' or e.get('validate_at_start') is not True
            or any(cfg.get(k) != reference.get(k) for k in ['load_dir', 'load_ckpt', 'ckpt', 'auto_retry',
                'num_gpus', 'num_nodes', 'tryrun', 'profile'])
            or t.get('deform_conditioner') != reference['trainer']['args']['deform_conditioner']
            or 'ray_feature_utility' in t
            or t.get('model_learning_rates') != dict(deform_conditioner=1e-4)
            or common_contract(cfg) != common_contract(reference)
            or cfg['dataset'] != reference['dataset'] or cfg['validation'] != reference['validation']):
        raise ValueError('same native numerical/data contract and physical Top1-only source10 required')
    return VARIANT


def build_trainer():
    import torch
    from pixal3d.modules.r_mediated_source_contrast import RMediatedSourceContrastConditioner
    from pixal3d.modules.r_mediated_top1_source import RMediatedTop1SourceConditioner
    from pixal3d.trainers.flow_matching.member_posterior_source_ray import MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer

    class Top1Trainer(MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer):
        def init_models_and_more(self, **kwargs):
            if self.image_cond_model is None:
                self._init_image_cond_model()
            if self.deform_conditioner is not None:
                raise ValueError('fresh Top1 constructor required; no module reinterpretation')
            devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
            with torch.random.fork_rng(devices=devices):
                module = RMediatedTop1SourceConditioner(**self.deform_conditioner_config)
            with torch.random.fork_rng(devices=devices):
                reference = RMediatedSourceContrastConditioner(appearance_mode='top1-plus-feature',
                    residual_logit_bound=1., **self.deform_conditioner_config)
            common = {n: p for n, p in module.named_parameters() if p.requires_grad}
            other = dict(reference.named_parameters())
            if len(common) != 10 or not all(torch.equal(p, other[n]) for n, p in common.items()):
                raise ValueError('all ten actual cold R/use/fusion weights must equal current R19 initialization')
            del reference
            module._actual_initial_core10_equal_R19 = True
            self.deform_conditioner = module.cuda()
            self.models['deform_conditioner'] = self.deform_conditioner
            super().init_models_and_more(**kwargs)

    return Top1Trainer
