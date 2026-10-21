"""Evaluate a completed native SS source-input experiment, never a pilot.

The denoiser and source conditioner must come from the same completed480 run.
Teacher112 checkpoints are initialization artifacts, not joint SS results.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from .cascade_view_roles import file_sha256
from .source_input_joint_runtime import science_recipe_binding, COLD_INITIAL_FUNCTION, TEACHER_INITIAL_FUNCTION
from .source_teacher_initialization import CLASSES, source_input_binding


def validate_completed_joint_inputs(root, config_path, denoiser, conditioner, *, variant, step):
    """Bind saved files to an actually completed run before GPU initialization.

    Canonical file4 receipts are verified by the campaign wrapper. This guard
    additionally prevents combining different runs or relabeling teacher112.
    Model tensor layouts are checked by the native strict loaders at runtime.
    """
    root = Path(root).resolve()
    paths = dict(config=root/'config.json', progress=root/'progress.json',
        startup=root/'startup.json', runtime=root/'source-input-joint-runtime.json',
        denoiser=root/'ckpts/denoiser_step0000480.pt',
        conditioner=root/'ckpts/deform_conditioner_step0000480.pt')
    if (variant not in CLASSES or step != 480 or Path(config_path).resolve() != paths['config']
            or Path(denoiser).resolve() != paths['denoiser']
            or Path(conditioner).resolve() != paths['conditioner']):
        raise ValueError('same completed480 native run config/denoiser/source required; no teacher pilot substitution')
    cfg, progress, startup, runtime = [json.loads(paths[key].read_text())
                                      for key in ('config', 'progress', 'startup', 'runtime')]
    init = runtime.get('initialization', [])
    teacher = cfg['experiment'].get('additional_source_teacher_supervision')
    initial_function = TEACHER_INITIAL_FUNCTION if teacher else COLD_INITIAL_FUNCTION
    if (Path(cfg['output_dir']).resolve() != root or type(teacher) is not bool
            or cfg['experiment'].get('source_input_design') != variant
            or cfg['experiment'].get('initial_function') != initial_function
            or startup.get('initial_function') != initial_function
            or progress.get('stage') != 'completed' or progress.get('current') != 480
            or progress.get('total') != 480
            or startup.get('status') != 'PASS_ACTUAL_FRESH_MEMBER_FEATURE_SOURCE_RAY_SS_STARTUP'
            or (startup.get('global_batch'), startup.get('microbatch'), startup.get('accumulation'),
                startup.get('world_size'), startup.get('trainable_NN_LoRA_tensors'),
                startup.get('trainable_source_tensors'), startup.get('optimizer_states')) != (96,96,1,1,360,19,0)
            or runtime.get('status') != 'PASS_COMPLETED_NATIVE_SOURCE_INPUT_PHASE'
            or runtime.get('phase') != 'science' or runtime.get('source_input_design') != variant
            or len(init) != 1 or init[0].get('variant') != variant
            or init[0].get('source_tensors') != 19
            or init[0].get('additional_TRAIN_M1_source9_10_supervision') is not teacher):
        raise ValueError('actual480 native scientific completion and honest source initialization required')
    hashes = {str(path): file_sha256(path) for path in paths.values()}
    return cfg, hashes


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--source-input-design', choices=list(CLASSES), required=True)
    parser.add_argument('--completed-training-root', type=Path, required=True)
    local, remaining = parser.parse_known_args()
    if any(flag in remaining for flag in ('--source-teacher-init-spec', '--pilot-variant')):
        raise ValueError('joint checkpoint evaluation does not load teacher initialization')
    def value(flag): return remaining[remaining.index(flag)+1]
    if value('--stage') != 'ss64':
        raise ValueError('completed source-input joint experiments are native SS only')
    cfg, hashes = validate_completed_joint_inputs(local.completed_training_root,
        value('--arm-config'), value('--checkpoint'), value('--conditioner-checkpoint'),
        variant=local.source_input_design, step=int(value('--step')))
    teacher = cfg['experiment']['additional_source_teacher_supervision']
    from pixal3d.trainers.flow_matching import member_feature_source_ray as trainer_module
    from . import evaluate_member_feature_source_ray as evaluation
    # The original evaluator owns its deterministic sampler, native checkpoint
    # loading, RGB inputs, metrics and optional same-forward value intervention.
    previous_validator, argv = evaluation.validate_recipe, sys.argv
    holder = SimpleNamespace(validate_feature_recipe=previous_validator)
    try:
        with science_recipe_binding(holder, variant=local.source_input_design, teacher_enabled=teacher):
            evaluation.validate_recipe = holder.validate_feature_recipe
            with source_input_binding(trainer_module, variant=local.source_input_design) as instances:
                sys.argv = [sys.argv[0], *remaining]
                evaluation.main()
    finally:
        evaluation.validate_recipe, sys.argv = previous_validator, argv
    if (len(instances) != 1 or sum(p.requires_grad for p in instances[0].parameters()) != 19
            or any(file_sha256(path) != digest for path,digest in hashes.items())):
        raise ValueError('one original19-source model and unchanged completed-run files required')
    state = torch.load(value('--conditioner-checkpoint'), map_location='cpu', weights_only=True)
    actual = instances[0].state_dict()
    if set(state) != set(actual) or any(not torch.equal(v.detach().cpu(), state[k]) for k,v in actual.items()):
        raise ValueError('evaluated source state must equal the saved joint480 state exactly')
    output = Path(value('--output')); path = output/'evaluation.json'
    result = json.loads(path.read_text())
    if (result['step'] != 480 or result['optimizer_updates_during_evaluation'] != 0
            or result['native_prior_source_ray']['actual_optimizer_states'] != 0
            or result['input_states'] != ['M1','M0','M0','M0']
            or result['input_camera_ids'] != [8,9,10,11]):
        raise ValueError('actual read-only native SS480 evaluation with edited RGB input required')
    result['native_prior_source_ray']['native_MV4_global_projection_negative_initialization'] = not teacher
    result['source_input_joint'] = dict(variant=local.source_input_design,
        completed_native_joint_updates=480, original_multiview_base_step=4210,
        source_teacher_initialization_updates=112 if teacher else 0,
        additional_TRAIN_M1_source9_10_supervision=teacher,
        teacher_labels_and_teacher_checkpoint_not_loaded_during_evaluation=True,
        actual_joint_source_state_exact_after_evaluation=True,
        source_and_denoiser_same_completed_run=True, source_tensors=19,
        completed_run_input_SHA256=hashes,
        cold_native_identity_claim=False, actual_joint_SS_LoRA_and_source_finetuned=True,
        process_local_bindings_restored=True, model_adopted=False, full_goal_complete=False)
    path.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
