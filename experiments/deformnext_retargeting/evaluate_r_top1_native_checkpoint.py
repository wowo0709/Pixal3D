"""Use the same native checkpoint evaluator with an explicit physical Top1 binding."""
from contextlib import contextmanager, nullcontext
from pathlib import Path

from . import evaluate_r_native_checkpoint as evaluation
from . import r_mediated_native_runtime as core
from .r_top1_native_recipe import VARIANT, validate_config
from .r_top1_native_runtime import parameter_groups, selected_trainer


def validate_bundle(bundle, cfg):
    variant = validate_config(cfg, 'science')
    step = bundle['step']
    root = Path(cfg['output_dir'])
    if (bundle.get('schema') != 'cold-native-task-Top1-SS-checkpoint-pair-v1'
            or bundle.get('variant') != variant or type(step) is not int or step not in (96, 192, 288, 384, 480)
            or bundle.get('checkpoint') != str(root / f'ckpts/denoiser_step{step:07d}.pt')
            or bundle.get('conditioner_checkpoint') != str(root / f'ckpts/deform_conditioner_step{step:07d}.pt')
            or bundle.get('native_config') != str(root / 'config.json')
            or bundle.get('source_teacher_updates') != 0 or bundle.get('native_task_objective') != 'SS_flow_mse'
            or bundle.get('native_validation', {}).get('step') != step):
        raise ValueError('actual same-step native Top1-only source10 checkpoint pair required')
    return VARIANT


@contextmanager
def top1_evaluation_binding():
    previous = evaluation.validate_bundle, evaluation.cold_source_binding, core.selected_trainer, core.parameter_groups
    evaluation.validate_bundle = validate_bundle
    evaluation.cold_source_binding = lambda variant: nullcontext()
    core.selected_trainer, core.parameter_groups = selected_trainer, parameter_groups
    try:
        yield
    finally:
        evaluation.validate_bundle, evaluation.cold_source_binding, core.selected_trainer, core.parameter_groups = previous


def main():
    with top1_evaluation_binding():
        evaluation.main()


if __name__ == '__main__':
    main()
