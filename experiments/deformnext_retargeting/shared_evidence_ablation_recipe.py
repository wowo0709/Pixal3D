"""Physical controls of the final shared-evidence candidates, without selection.

These constructors launch no training/evaluation. Native proof, capacity and
checkpoint/evaluator binding must precede actual science execution.
"""
import copy
import hashlib
import json
from pathlib import Path
import re


NAME = 'SharedEvidenceAblationNativeFlowTrainer'
SCHEMA = 'shared-raw-evidence-ablation-native-v1'
STUDY_SCHEMA = 'shared-raw-evidence-ablation-v1'
VARIANTS = {
    'A_FIXED_UNIFORM': ('E_FIXED_RAW', 'fixed-prior', 'uniform-k', 10),
    'A_FIXED_NO_R': ('E_FIXED_RAW', 'fixed-prior', 'without-retargeting', 5),
    'A_LEARNED_NO_R': ('E_LEARNED_RAW', 'learned-anchored', 'without-retargeting', 12),
    'A_LEARNED_NO_LOGIT_PRIOR': ('E_LEARNED_RAW', 'learned-anchored', 'without-logit-prior', 17),
}
SOURCE_COUNTS = {variant: fields[3] for variant, fields in VARIANTS.items()}


def required_sources(variant):
    if variant not in VARIANTS:
        raise ValueError('one explicit physical shared-evidence control required')
    from .pipeline_population_contract_v2 import required_sources as parent_sources
    directory = Path(__file__).resolve().parent
    return sorted(set(parent_sources(VARIANTS[variant][0])) | {
        str(directory/'shared_evidence_ablation_recipe.py'),
        str(directory/'shared_evidence_ablation_native_runtime.py'),
        str(directory/'evaluate_shared_evidence_ablation_checkpoint.py'),
        str(directory.parents[1]/'pixal3d/modules/shared_evidence_ablation.py'),
    })


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_config(parent, *, variant, phase, parent_config, output_dir, recipe_path,
                run_id, source_commit, assigned_GPUs, CPU_affinity_per_GPU, proof_json, capacity_json):
    """Bind the actual parent recipe, changing only source control/run metadata."""
    from .consistent_evidence_native_recipe import validate_config as validate_parent
    if (variant not in VARIANTS or phase not in ('proof', 'capacity', 'science')
            or validate_parent(parent, 'science') != VARIANTS[variant][0]
            or not isinstance(run_id, str) or not run_id
            or not isinstance(source_commit, str) or re.fullmatch('[0-9a-f]{40}', source_commit) is None
            or not assigned_GPUs or len(set(assigned_GPUs)) != len(assigned_GPUs)
            or any(not isinstance(g, str) or not g.isdecimal() for g in assigned_GPUs)
            or set(CPU_affinity_per_GPU) != set(assigned_GPUs)
            or any(not cpus or any(type(c) is not int or c < 0 for c in cpus)
                   for cpus in CPU_affinity_per_GPU.values())):
        raise ValueError('own parent, phase, run and explicit GPU/CPU assignment required')
    for path in (parent_config, output_dir, recipe_path, proof_json, capacity_json):
        if not Path(path).is_absolute():
            raise ValueError('explicit absolute own run/reference paths required')
    if (json.loads(Path(parent_config).read_text()) != parent
            or str(output_dir) == parent['output_dir']
            or str(recipe_path) == parent['experiment']['recipe_path']
            or str(proof_json) == parent['experiment']['cold_proof_json']
            or str(capacity_json) == parent['experiment']['micro96_capacity_json']):
        raise ValueError('actual parent bytes and distinct own run/proof/capacity paths required')
    matcher_hash = parent['dataset']['args']['expected_q2c_sha256']
    if (_sha(parent['trainer']['args']['deform_conditioner']['q2c_checkpoint']) != matcher_hash
            or parent['validation']['dataset_args']['expected_q2c_sha256'] != matcher_hash):
        raise ValueError('same actual parent frozen matcher bytes required')
    value = copy.deepcopy(parent)
    value.update(output_dir=str(output_dir), name=run_id, wandb_name=run_id,
                 wandb_id=run_id, use_wandb=phase == 'science')
    value['trainer']['name'] = NAME
    e = value['experiment']
    e.pop('followup_study_schema')
    e.pop('shared_carrier_use_retarget_evidence')
    parent_arm, mode, control, count = VARIANTS[variant]
    e.update(native_variant=variant, conditioning_schema=SCHEMA,
        ablation_study_schema=STUDY_SCHEMA, ablation_parent_variant=parent_arm,
        ablation_control=control, evidence_mode=mode, actual_source_tensors=count,
        shared_carrier_use_evidence=True, retargeter_present=control != 'without-retargeting',
        ablation_parent_science_config=str(parent_config),
        ablation_parent_science_config_SHA256=_sha(parent_config),
        parent_source_commit=parent['experiment'].get('source_commit'), source_commit=source_commit,
        engineering_only=phase != 'science', recipe_path=str(recipe_path),
        assigned_GPUs=list(assigned_GPUs), CPU_affinity_per_GPU=copy.deepcopy(CPU_affinity_per_GPU),
        cold_proof_json=str(proof_json), micro96_capacity_json=str(capacity_json),
        ablation_source_SHA256={path:_sha(path) for path in required_sources(variant)})
    return value


def validate_config(cfg, phase):
    e = cfg['experiment']
    variant = e.get('native_variant')
    if (variant not in VARIANTS or phase not in ('proof', 'capacity', 'science')
            or type(cfg.get('training_seed')) is not int or cfg['training_seed'] not in (101, 202)):
        raise ValueError('explicit own control family, native phase and integer training seed required')
    parent_path = Path(e['ablation_parent_science_config'])
    if _sha(parent_path) != e['ablation_parent_science_config_SHA256']:
        raise ValueError('bound actual parent recipe changed')
    parent = json.loads(parent_path.read_text())
    expected = make_config(parent, variant=variant, phase=phase, parent_config=parent_path,
        output_dir=cfg['output_dir'], recipe_path=e['recipe_path'], run_id=cfg['wandb_id'],
        source_commit=e['source_commit'],
        assigned_GPUs=e['assigned_GPUs'], CPU_affinity_per_GPU=e['CPU_affinity_per_GPU'],
        proof_json=e['cold_proof_json'], capacity_json=e['micro96_capacity_json'])
    if json.dumps(cfg, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError('only the declared physical control and own run metadata may differ from the parent')
    return variant


def construct_conditioner(variant, kwargs):
    if variant not in VARIANTS:
        raise ValueError('one explicit physical shared-evidence control required')
    import torch
    from .consistent_evidence_native_recipe import construct_conditioner as parent_constructor
    from pixal3d.modules.shared_evidence_ablation import SharedEvidenceAblationConditioner
    parent, evidence_mode, ablation, count = VARIANTS[variant]
    devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        reference = parent_constructor(parent, kwargs)
    with torch.random.fork_rng(devices=devices):
        module = SharedEvidenceAblationConditioner(evidence_mode=evidence_mode, ablation=ablation, **kwargs)
    actual = {n: p for n, p in module.named_parameters() if p.requires_grad}
    original = {n: p for n, p in reference.named_parameters() if p.requires_grad}
    retained = {n: p for n, p in original.items() if not (ablation == 'without-retargeting' and n.startswith('retargeter.'))}
    if (len(actual) != count or set(actual) != set(retained)
            or any(not torch.equal(p, actual[n]) for n, p in retained.items())
            or any(p.requires_grad for p in module.q2c.parameters())
            or (ablation == 'without-retargeting' and module.retargeter is not None)
            or (ablation == 'uniform-k' and module.ray_readout is not None)):
        raise ValueError('same actual retained cold weights and only the declared physical change required')
    module._actual_initial_retained_parameters_equal_parent = True
    module._parent_initial_core10_equal_Top1 = reference._actual_initial_core10_equal_Top1
    return module


def build_trainer(variant):
    """Use the existing native SS/LoRA trainer with only the declared source."""
    if variant not in VARIANTS:
        raise ValueError('one explicit physical shared-evidence control required')
    from pixal3d.trainers.flow_matching.member_posterior_source_ray import MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer

    class SharedControlTrainer(MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer):
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
    return SharedControlTrainer
