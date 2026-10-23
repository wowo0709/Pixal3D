"""Own checkpoint binding for every physical family in the finite module study.

Population coverage, cache QA, held-out separation and two-load validation36
requirements are inherited unchanged. A module alias never reinterprets a
different family's source checkpoint or invents a production winner.
"""
from contextlib import contextmanager, nullcontext
import importlib
import json
from pathlib import Path
import re

from . import native_population_contract as population
from .module_pipeline_protocol_v2 import protocol
from .unified_module_summary import ALIASES, SOURCE_COUNTS


SCHEMA = 'unified-module-SS480-frozen-binding-v1'
MODE, ENGINEERING, FINAL = population.MODE, population.ENGINEERING, population.FINAL
EVALUATORS = dict(r='evaluate_r_native_checkpoint', r_top1='evaluate_r_top1_native_checkpoint',
    projected='evaluate_projected_native_checkpoint', raw_correspondence='evaluate_raw_correspondence_checkpoint',
    unified_design='evaluate_unified_design_checkpoint')


def family(arm):
    if arm not in protocol()['arms']:
        raise ValueError('one of the twelve prespecified physical module arms required')
    if arm in ('V1', 'FULL', 'C_DECODER'):
        return 'r_top1' if arm == 'C_DECODER' else 'r'
    if arm in ('C_TOP1_V', 'C_FIXED_TOPK'):
        return 'projected'
    if arm in ('C_RAW', 'C_SPATIAL', 'C_D1_COS'):
        return 'raw_correspondence'
    return 'unified_design'


def bundle_validator(arm):
    return importlib.import_module('.' + EVALUATORS[family(arm)], __package__)


@contextmanager
def evaluator_binding(arm):
    """Select the existing tested family, without replacing its forward math."""
    name = family(arm)
    core = importlib.import_module('.' + EVALUATORS['r' if name == 'r_top1' else name], __package__)
    if name == 'r_top1':
        from .evaluate_r_top1_native_checkpoint import top1_evaluation_binding
        context = top1_evaluation_binding()
    else:
        context = nullcontext()
    with context:
        yield core


def model_binding(bundle, cfg, hashes, *, arm):
    actual = bundle_validator(arm).validate_bundle(bundle, cfg)
    root = Path(cfg['output_dir'])
    expected_variant = ALIASES.get(arm, arm)
    source = None if arm == 'V1' else str(root/'ckpts/deform_conditioner_step0000480.pt')
    if (actual != expected_variant or bundle['variant'] != expected_variant
            or type(bundle['step']) is not int or bundle['step'] != 480
            or type(cfg['training_seed']) is not int or cfg['training_seed'] not in (101, 202)
            or not root.is_absolute()
            or bundle['checkpoint'] != str(root/'ckpts/denoiser_step0000480.pt')
            or bundle['conditioner_checkpoint'] != source
            or bundle['native_config'] != str(root/'config.json')):
        raise ValueError('own complete480 checkpoint, physical family and independent integer seed required')

    def spec(path):
        if path is None:
            return None
        digest = hashes.get(path)
        if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None:
            raise ValueError('actual lowercase SHA256 required for each own checkpoint/config')
        return dict(path=path, SHA256=digest)

    return dict(schema=SCHEMA, module_arm=arm, variant=actual, runtime_family=family(arm),
                checkpoint_bundle_schema=bundle['schema'], training_seed=cfg['training_seed'],
                step=480, source_tensors=SOURCE_COUNTS[arm],
                native_config=spec(bundle['native_config']), checkpoint=spec(bundle['checkpoint']),
                conditioner_checkpoint=spec(source), conditioning_mode='edited-rgb',
                source_teacher_updates=0, native_task_objective='SS_flow_mse')


def validate_plan(plan, bundle, cfg, rows, panel, seeds, *, binding, **kwargs):
    arm = plan.get('module_arm')
    hashes = {s['path']: s['SHA256'] for s in
              (binding[k] for k in ('native_config', 'checkpoint', 'conditioner_checkpoint'))
              if s is not None}
    own = model_binding(bundle, cfg, hashes, arm=arm)
    if (json.dumps(binding, sort_keys=True) != json.dumps(own, sort_keys=True)
            or type(plan.get('generation_batch_size')) is not int
            or type(plan.get('expected_assets')) is not int):
        raise ValueError('actual physical module family must equal the frozen model binding')
    return population.validate_plan(plan, bundle, cfg, rows, panel, seeds,
                                    binding=binding, **kwargs)


def required_sources(arm):
    """Critical entry/selection dependencies; retain the original runtime pins too."""
    root = Path(__file__).resolve().parent
    name = family(arm)
    modules = ['unified_population_contract', 'evaluate_unified_bound_population',
        'native_population_contract', 'evaluate_native_bound_population', 'evaluate_bound_population',
        'cascade_view_roles', 'evaluate_ss_artifact_subset', 'quick_metrics',
        'module_pipeline_protocol', 'module_pipeline_protocol_v2', 'unified_module_summary',
        'r_mediated_native_recipe']
    if name == 'r_top1':
        modules += ['evaluate_r_native_checkpoint', 'evaluate_r_top1_native_checkpoint',
                    'r_mediated_native_runtime', 'r_top1_native_runtime', 'r_top1_native_recipe']
    else:
        modules += [EVALUATORS[name]]
        prefix = 'r_mediated' if name == 'r' else name
        modules += [prefix + '_native_runtime', prefix + '_native_recipe']
    return sorted({str(root/(m + '.py')) for m in modules})
