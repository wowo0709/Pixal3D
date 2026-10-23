"""Own physical checkpoint bindings for finite12 and the two explicit follow-ups.

The original twelve-arm protocol and its numerical entries remain unchanged.
Shared-evidence models have a distinct schema and retain their actual names.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import re

from . import native_population_contract as population
from . import unified_population_contract as original
from .consistent_evidence_native_recipe import SOURCE_COUNTS as FOLLOWUP_COUNTS
from .consistent_evidence_native_recipe import VARIANTS as FOLLOWUPS
from .module_pipeline_protocol_v2 import protocol
from .unified_module_summary import ALIASES


SCHEMA = 'consistent-raw-evidence-SS480-frozen-binding-v1'
ARMS = tuple(protocol()['arms'])+tuple(FOLLOWUPS)
SOURCE_COUNTS = dict(original.SOURCE_COUNTS, **FOLLOWUP_COUNTS)
MODE, ENGINEERING, FINAL = population.MODE, population.ENGINEERING, population.FINAL


def family(arm):
    return 'consistent_evidence' if arm in FOLLOWUPS else original.family(arm)


def schema_for(arm):
    family(arm)
    return SCHEMA if arm in FOLLOWUPS else original.SCHEMA


def bundle_validator(arm):
    if arm in FOLLOWUPS:
        from . import evaluate_consistent_evidence_checkpoint
        return evaluate_consistent_evidence_checkpoint
    return original.bundle_validator(arm)


@contextmanager
def evaluator_binding(arm):
    if arm in FOLLOWUPS:
        yield bundle_validator(arm)
    else:
        with original.evaluator_binding(arm) as core:
            yield core


def model_binding(bundle, cfg, hashes, *, arm):
    if arm not in FOLLOWUPS:
        return original.model_binding(bundle, cfg, hashes, arm=arm)
    actual = bundle_validator(arm).validate_bundle(bundle, cfg)
    root = Path(cfg['output_dir'])
    if (actual != arm or type(bundle['step']) is not int or bundle['step'] != 480
            or type(cfg['training_seed']) is not int or cfg['training_seed'] not in (101, 202)
            or not root.is_absolute() or bundle['native_config'] != str(root/'config.json')
            or bundle['checkpoint'] != str(root/'ckpts/denoiser_step0000480.pt')
            or bundle['conditioner_checkpoint'] != str(root/'ckpts/deform_conditioner_step0000480.pt')):
        raise ValueError('own complete shared-evidence480 NN/source/config and integer TRAIN seed required')
    def spec(path):
        digest = hashes.get(path)
        if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None:
            raise ValueError('actual lowercase SHA256 required for each own model file')
        return dict(path=path,SHA256=digest)
    return dict(schema=SCHEMA,module_arm=arm,variant=arm,runtime_family=family(arm),
        checkpoint_bundle_schema=bundle['schema'],training_seed=cfg['training_seed'],step=480,
        source_tensors=SOURCE_COUNTS[arm],evidence_mode=FOLLOWUPS[arm],
        shared_carrier_use_retarget_evidence=True,learned_projection_tensors=0,
        learned_attention_head_tensors=7 if arm=='E_LEARNED_RAW' else 0,
        native_config=spec(bundle['native_config']),checkpoint=spec(bundle['checkpoint']),
        conditioner_checkpoint=spec(bundle['conditioner_checkpoint']),conditioning_mode='edited-rgb',
        source_teacher_updates=0,native_task_objective='SS_flow_mse')


def validate_plan(plan, bundle, cfg, rows, panel, seeds, *, binding, **kwargs):
    arm = plan.get('module_arm')
    if arm not in FOLLOWUPS:
        return original.validate_plan(plan,bundle,cfg,rows,panel,seeds,binding=binding,**kwargs)
    hashes={binding[k]['path']:binding[k]['SHA256'] for k in
            ('native_config','checkpoint','conditioner_checkpoint')}
    own=model_binding(bundle,cfg,hashes,arm=arm)
    if (json.dumps(binding,sort_keys=True,allow_nan=False)!=json.dumps(own,sort_keys=True,allow_nan=False)
            or type(plan.get('generation_batch_size')) is not int
            or type(plan.get('expected_assets')) is not int):
        raise ValueError('actual shared-evidence physical architecture must equal frozen model binding')
    return population.validate_plan(plan,bundle,cfg,rows,panel,seeds,binding=binding,**kwargs)


def required_sources(arm):
    root=Path(__file__).resolve().parent
    paths={str(root/(name+'.py')) for name in
        ('pipeline_population_contract_v2','evaluate_pipeline_bound_population_v2')}
    if arm not in FOLLOWUPS:
        paths.update(original.required_sources(arm))
    else:
        paths.update(str(root/(name+'.py')) for name in (
            'native_population_contract','unified_population_contract','evaluate_native_bound_population',
            'evaluate_bound_population','cascade_view_roles','evaluate_ss_artifact_subset','quick_metrics',
            'consistent_evidence_native_recipe','consistent_evidence_native_runtime',
            'evaluate_consistent_evidence_checkpoint','r_mediated_native_recipe',
            'module_pipeline_protocol_v2','module_pipeline_protocol','unified_module_summary'))
        paths.add(str(root.parents[1]/'pixal3d/modules/consistent_ray_evidence.py'))
    return sorted(paths)
