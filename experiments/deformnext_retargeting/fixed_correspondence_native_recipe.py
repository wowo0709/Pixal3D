"""Matched correspondence substitutions in the actual shared Fixed context."""
import copy
import hashlib
import json
from pathlib import Path
import re


NAME = 'FixedCorrespondenceNativeFlowTrainer'
SCHEMA = 'shared-fixed-correspondence-native-v1'
STUDY_SCHEMA = 'shared-fixed-correspondence-v1'
VARIANTS = dict(M_FIXED_SPATIAL='native-spatial', M_FIXED_RAW='RAW-cosine', M_FIXED_D1_COS='D1-cosine')
SOURCE_COUNTS = dict.fromkeys(VARIANTS, 10)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def required_sources(variant):
    if variant not in VARIANTS:
        raise ValueError('one explicit shared Fixed correspondence control required')
    from .pipeline_population_contract_v2 import required_sources as parent_sources
    directory = Path(__file__).resolve().parent
    paths = set(parent_sources('E_FIXED_RAW'))
    paths.update(str(directory/(name+'.py')) for name in (
        'fixed_correspondence_native_recipe', 'fixed_correspondence_native_runtime',
        'evaluate_fixed_correspondence_checkpoint', 'fixed_correspondence_pipeline_contract'))
    paths.update(str(directory.parents[1]/'pixal3d/modules'/(name+'.py')) for name in (
        'fixed_correspondence_evidence', 'raw_member_ray_posterior',
        'd1_cosine_member_posterior', 'raw_spatial_member_posterior',
        'target_first_observed_spatial'))
    paths.add(str(directory.parents[1]/'pixal3d/trainers/flow_matching/member_feature_controls.py'))
    return sorted(paths)


def make_config(parent, *, variant, phase, parent_config, output_dir, recipe_path,
                run_id, source_commit, assigned_GPUs, CPU_affinity_per_GPU, proof_json, capacity_json):
    from .consistent_evidence_native_recipe import validate_config as validate_parent
    if (variant not in VARIANTS or phase not in ('proof','capacity','science')
            or validate_parent(parent,'science') != 'E_FIXED_RAW'
            or not isinstance(run_id,str) or not run_id
            or not isinstance(source_commit,str) or re.fullmatch('[0-9a-f]{40}',source_commit) is None
            or not assigned_GPUs or len(set(assigned_GPUs)) != len(assigned_GPUs)
            or any(not isinstance(g,str) or not g.isdecimal() for g in assigned_GPUs)
            or set(CPU_affinity_per_GPU) != set(assigned_GPUs)
            or any(not v or any(type(c) is not int or c < 0 for c in v)
                   for v in CPU_affinity_per_GPU.values())):
        raise ValueError('same actual Fixed parent and explicit own phase/run/device assignment required')
    for path in (parent_config,output_dir,recipe_path,proof_json,capacity_json):
        if not Path(path).is_absolute():
            raise ValueError('explicit own absolute run and reference paths required')
    if (json.loads(Path(parent_config).read_text()) != parent
            or str(output_dir) == parent['output_dir']
            or str(recipe_path) == parent['experiment']['recipe_path']
            or str(proof_json) == parent['experiment']['cold_proof_json']
            or str(capacity_json) == parent['experiment']['micro96_capacity_json']):
        raise ValueError('actual same parent and distinct own proof/capacity/science paths required')
    matcher_hash = parent['dataset']['args']['expected_q2c_sha256']
    if (_sha(parent['trainer']['args']['deform_conditioner']['q2c_checkpoint']) != matcher_hash
            or parent['validation']['dataset_args']['expected_q2c_sha256'] != matcher_hash):
        raise ValueError('original input-feature cache provenance and frozen matcher initializer changed')
    value = copy.deepcopy(parent)
    value.update(output_dir=str(output_dir),name=run_id,wandb_name=run_id,wandb_id=run_id,
                 use_wandb=phase == 'science')
    value['trainer']['name'] = NAME
    e = value['experiment']; e.pop('followup_study_schema')
    e.update(native_variant=variant,conditioning_schema=SCHEMA,correspondence_study_schema=STUDY_SCHEMA,
        correspondence_parent_variant='E_FIXED_RAW',correspondence_mode=VARIANTS[variant],
        actual_candidate_budget=3 if variant=='M_FIXED_SPATIAL' else 64,
        matching_decoder_physically_absent=variant!='M_FIXED_D1_COS',
        matching_scalar_score_physically_absent=True,
        correspondence_parent_science_config=str(parent_config),
        correspondence_parent_science_config_SHA256=_sha(parent_config),
        parent_source_commit=parent['experiment'].get('source_commit'),source_commit=source_commit,
        engineering_only=phase!='science',recipe_path=str(recipe_path),
        assigned_GPUs=list(assigned_GPUs),CPU_affinity_per_GPU=copy.deepcopy(CPU_affinity_per_GPU),
        cold_proof_json=str(proof_json),micro96_capacity_json=str(capacity_json),
        correspondence_source_SHA256={p:_sha(p) for p in required_sources(variant)})
    return value


def validate_config(cfg,phase):
    e=cfg['experiment'];variant=e.get('native_variant')
    if (variant not in VARIANTS or phase not in ('proof','capacity','science')
            or type(cfg.get('training_seed')) is not int or cfg['training_seed'] not in (101,202)):
        raise ValueError('own explicit Fixed correspondence variant and integer seed required')
    path=Path(e['correspondence_parent_science_config'])
    if _sha(path)!=e['correspondence_parent_science_config_SHA256']:
        raise ValueError('actual Fixed parent recipe changed')
    expected=make_config(json.loads(path.read_text()),variant=variant,phase=phase,parent_config=path,
        output_dir=cfg['output_dir'],recipe_path=e['recipe_path'],run_id=cfg['wandb_id'],
        source_commit=e['source_commit'],assigned_GPUs=e['assigned_GPUs'],
        CPU_affinity_per_GPU=e['CPU_affinity_per_GPU'],proof_json=e['cold_proof_json'],
        capacity_json=e['micro96_capacity_json'])
    if json.dumps(cfg,sort_keys=True,allow_nan=False)!=json.dumps(expected,sort_keys=True,allow_nan=False):
        raise ValueError('only correspondence substitution and explicit operational metadata may change')
    return variant


def construct_conditioner(variant,kwargs):
    if variant not in VARIANTS:
        raise ValueError('own Fixed correspondence constructor required')
    import torch
    from .consistent_evidence_native_recipe import construct_conditioner as parent_constructor
    from pixal3d.modules.fixed_correspondence_evidence import FixedCorrespondenceEvidenceConditioner
    devices=[torch.cuda.current_device()] if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        reference=parent_constructor('E_FIXED_RAW',kwargs)
    with torch.random.fork_rng(devices=devices):
        module=FixedCorrespondenceEvidenceConditioner(correspondence=VARIANTS[variant],**kwargs)
    actual={n:p for n,p in module.named_parameters() if p.requires_grad}
    previous={n:p for n,p in reference.named_parameters() if p.requires_grad}
    if (len(actual)!=10 or set(actual)!=set(previous)
            or any(not torch.equal(p,actual[n]) for n,p in previous.items())
            or (module.q2c is None) != (variant!='M_FIXED_D1_COS')
            or module.q2c is not None and (module.q2c.fusion_head is not None
                or any(p.requires_grad for p in module.q2c.parameters()))):
        raise ValueError('same actual cold core10 with only physical matching substitution required')
    module._actual_initial_core10_equal_Fixed=True
    module._actual_initial_core10_equal_Top1=reference._actual_initial_core10_equal_Top1
    return module


def build_trainer(variant):
    if variant not in VARIANTS:
        raise ValueError('own Fixed correspondence trainer required')
    from pixal3d.trainers.flow_matching.member_posterior_source_ray import MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer
    from pixal3d.trainers.flow_matching.member_feature_controls import MemberFeatureControlImageConditionedProjFlowMatchingCFGTrainer

    class FixedCorrespondenceTrainer(MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer):
        def init_models_and_more(self,**kwargs):
            if 'ray_feature_utility' in kwargs:
                raise ValueError('no feature-teacher or value transform in Fixed correspondence controls')
            if self.image_cond_model is None:
                self._init_image_cond_model()
            if self.deform_conditioner is not None:
                raise ValueError('fresh source module required')
            module=construct_conditioner(variant,self.deform_conditioner_config)
            self.deform_conditioner=module.cuda();self.models['deform_conditioner']=self.deform_conditioner
            self.feature_control='Fixed-native-spatial' if variant=='M_FIXED_SPATIAL' else 'Fixed-semantic'
            super().init_models_and_more(**kwargs)

        def _encode_deform_views(self,images,camera_info,transport,*,control='normal'):
            if variant!='M_FIXED_SPATIAL':
                return super()._encode_deform_views(images,camera_info,transport,control=control)
            return MemberFeatureControlImageConditionedProjFlowMatchingCFGTrainer._encode_deform_views(
                self,images,camera_info,transport,control=control)
    return FixedCorrespondenceTrainer


def study_metadata(cfg,variant):
    return dict(conditioning_schema=SCHEMA,correspondence_study_schema=STUDY_SCHEMA,
        correspondence_parent_variant='E_FIXED_RAW',correspondence_mode=VARIANTS[variant],
        actual_candidate_budget=3 if variant=='M_FIXED_SPATIAL' else 64,
        matching_decoder_physically_absent=variant!='M_FIXED_D1_COS',
        matching_scalar_score_physically_absent=True,
        correspondence_source_SHA256=cfg['experiment']['correspondence_source_SHA256'])
