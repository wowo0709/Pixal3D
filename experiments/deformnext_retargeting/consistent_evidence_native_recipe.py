"""Two focused shared-evidence follow-ups, independent of the original 12 arms.

Both retain the native MV4 consumer, frozen member matcher and cold core10.
Only the second learns probabilities; V is the untransformed source feature.
"""
import copy
import hashlib
import json
from pathlib import Path

from .r_mediated_native_recipe import common_contract, validate_config as validate_reference

NAME = 'ConsistentEvidenceNativeFlowTrainer'
SCHEMA = 'consistent-raw-ray-evidence-native-v1'
STUDY_SCHEMA = 'consistent-raw-ray-evidence-followup-v1'
VARIANTS = {'E_FIXED_RAW': 'fixed-prior', 'E_LEARNED_RAW': 'learned-anchored'}
SOURCE_COUNTS = {'E_FIXED_RAW': 10, 'E_LEARNED_RAW': 17}


def validate_config(cfg, phase, reference=None):
    e, t = cfg['experiment'], cfg['trainer']['args']
    if reference is None:
        path = Path(e['reference_science_config'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != e['reference_science_config_SHA256']:
            raise ValueError('bound original native full recipe changed')
        reference = json.loads(path.read_text())
    if validate_reference(reference, 'science') != 'R_raw_topk':
        raise ValueError('original native full recipe required as numerical/data reference')
    variant = e.get('native_variant')
    expected = copy.deepcopy(reference['trainer']['args']['deform_conditioner'])
    expected.update(ray_mode='semantic-weighted', candidate_aggregation='weighted')
    unchanged = ['training_assets', 'epochs', 'updates_per_epoch', 'target_modality',
                 'global_batch', 'world_size', 'microbatch_per_gpu', 'validate_at_start',
                 'initial_function', 'additional_source_teacher_supervision',
                 'source_teacher_updates', 'native_task_objective', 'initial_reference_evaluation']
    if (phase not in {'proof', 'capacity', 'science'} or variant not in VARIANTS
            or cfg['trainer']['name'] != NAME or e.get('conditioning_schema') != SCHEMA
            or e.get('followup_study_schema') != STUDY_SCHEMA or 'module_pipeline_schema' in e
            or e.get('evidence_mode') != VARIANTS[variant]
            or e.get('actual_source_tensors') != SOURCE_COUNTS[variant]
            or e.get('learned_attention_head_tensors') != (7 if variant == 'E_LEARNED_RAW' else 0)
            or e.get('learned_projection_tensors') != 0
            or e.get('shared_carrier_use_retarget_evidence') is not True
            or e.get('physical_deformation_query_channels') != 0
            or e.get('appearance_mode') != 'consistent-raw-topk'
            or e.get('residual_logit_bound', 'missing') is not None
            or e.get('engineering_only') is not (phase != 'science')
            or cfg['use_wandb'] is not (phase == 'science') or 'ray_feature_utility' in t
            or t.get('model_learning_rates') != reference['trainer']['args']['model_learning_rates']
            or t.get('deform_conditioner') != expected
            or any(e.get(k) != reference['experiment'].get(k) for k in unchanged)
            or common_contract(cfg) != common_contract(reference)
            or cfg['dataset'] != reference['dataset'] or cfg['validation'] != reference['validation']
            or any(cfg.get(k) != reference.get(k) for k in ['load_dir', 'load_ckpt', 'ckpt',
                'auto_retry', 'num_gpus', 'num_nodes', 'tryrun', 'profile'])):
        raise ValueError('focused shared-evidence variant on the same native SS96/480/LoRA/data recipe required')
    return variant


def construct_conditioner(variant, kwargs):
    """Compare real cold core10 tensors without loading trained source weights."""
    if variant not in VARIANTS:
        raise ValueError('explicit shared-evidence follow-up required')
    import torch
    from pixal3d.modules.consistent_ray_evidence import ConsistentRayEvidenceConditioner
    from pixal3d.modules.r_mediated_top1_source import RMediatedTop1SourceConditioner
    reference_kwargs = dict(kwargs, ray_mode='semantic-top1', candidate_aggregation='top1')
    devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        reference = RMediatedTop1SourceConditioner(**reference_kwargs)
    with torch.random.fork_rng(devices=devices):
        module = ConsistentRayEvidenceConditioner(evidence_mode=VARIANTS[variant], **kwargs)
    actual = {n: p for n, p in module.named_parameters() if p.requires_grad}
    original = {n: p for n, p in reference.named_parameters() if p.requires_grad}
    if (len(original) != 10 or len(actual) != SOURCE_COUNTS[variant]
            or any(n not in actual or not torch.equal(p, actual[n]) for n, p in original.items())
            or any(p.requires_grad for p in module.q2c.parameters())):
        raise ValueError('actual cold core10 differs from the native Top1+R reference')
    module._actual_initial_core10_equal_Top1 = True
    return module


def build_trainer(variant):
    if variant not in VARIANTS:
        raise ValueError('explicit shared-evidence follow-up required')
    from pixal3d.trainers.flow_matching.member_posterior_source_ray import MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer

    class ConsistentTrainer(MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer):
        def init_models_and_more(self, **kwargs):
            if 'ray_feature_utility' in kwargs:
                raise ValueError('no value transform or feature-teacher initializer in shared raw evidence')
            if self.image_cond_model is None:
                self._init_image_cond_model()
            if self.deform_conditioner is not None:
                raise ValueError('fresh source module required')
            module = construct_conditioner(variant, self.deform_conditioner_config)
            self.deform_conditioner = module.cuda()
            self.models['deform_conditioner'] = self.deform_conditioner
            super().init_models_and_more(**kwargs)
    return ConsistentTrainer
