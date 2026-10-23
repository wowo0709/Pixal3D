"""Read-only replay of an actual cold native-task SS LoRA/source checkpoint pair.

This entry deliberately uses the native-task recipe. Frozen-denoiser feature
teacher checkpoints and legacy generic arm metadata are not interchangeable.
"""
import argparse
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path

from .cascade_view_roles import ClosedInputs, file_sha256
from .raw_correspondence_native_recipe import validate_config

def cold_source_binding(variant):
    return nullcontext()


BUNDLE_SCHEMA = 'unified-module-raw-correspondence-SS-checkpoint-pair-v1'


def validate_bundle(bundle, cfg):
    variant = validate_config(cfg, 'science')
    step = bundle['step']
    root = Path(cfg['output_dir'])
    expected = root / 'ckpts' / f'denoiser_step{step:07d}.pt'
    source = (None if variant == 'vanilla' else
              str(root / 'ckpts' / f'deform_conditioner_step{step:07d}.pt'))
    if (bundle.get('schema') != BUNDLE_SCHEMA or bundle.get('variant') != variant
            or type(step) is not int or step not in (96, 192, 288, 384, 480)
            or bundle.get('checkpoint') != str(expected)
            or bundle.get('conditioner_checkpoint') != source
            or bundle.get('native_config') != str(root / 'config.json')
            or bundle.get('source_teacher_updates') != 0
            or bundle.get('native_task_objective') != 'SS_flow_mse'
            or bundle.get('native_validation', {}).get('step') != step):
        raise ValueError('actual same-variant/same-step native-task checkpoint pair required')
    return variant


def validate_panel(plan, bundle, cfg):
    ids = json.loads(Path(plan['panel']).read_text())['sample_ids']
    seeds = json.loads(Path(plan['seeds']).read_text())
    if (plan['expected_assets'] not in (32, 128) or len(ids) != plan['expected_assets']
            or len(ids) != len(set(ids)) or any(sid not in seeds for sid in ids)
            or any(type(seeds[sid]) is not int for sid in ids)
            or plan['generation_batch_size'] != 8
            or type(plan['deterministic_algorithms']) is not bool):
        raise ValueError('unique fixed32/128 panel, explicit integer seeds and native batch8 required')
    if plan['mode'] == 'epoch-replay':
        original = cfg['validation']['quick_metrics']
        if (plan['expected_assets'] != 32 or plan['deterministic_algorithms']
                or file_sha256(plan['panel']) != file_sha256(original['panel_file'])
                or file_sha256(plan['seeds']) != file_sha256(original['seed_map_file'])):
            raise ValueError('epoch replay retains actual in-process panel, seeds and nondeterministic mode')
    elif plan['mode'] in ('final128', 'archive32'):
        if bundle['step'] != 480 or not plan['deterministic_algorithms']:
            raise ValueError('fixed final480 deterministic evaluation required')
        if plan['expected_assets'] != (128 if plan['mode'] == 'final128' else 32):
            raise ValueError('final evaluation/archive panel size differs')
    else:
        raise ValueError('explicit epoch-replay/final128/archive32 mode required')
    return ids, seeds


def check_epoch_replay(result, reference):
    if (result['step'] != reference['step'] or result['weights'] != reference['weights']
            or result['quick_panel_ids'] != reference['quick_panel_ids']
            or result['quick_metrics'] != reference['quick_metrics']
            or result['per_asset'] != reference['quick_per_asset']):
        raise ValueError('saved native LoRA/source pair did not exactly reproduce epoch records; preserve output')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    check = ClosedInputs(plan['canonical_local_root'])
    check.check(args.plan, args.plan.parent / 'file4-closure.json')
    for path, receipt in plan['closed_inputs'].items():
        check.check(path, receipt)
    bundle_path = Path(plan['bundle'])
    receipt = bundle_path.parent / 'file4-closure.json'
    check.check(bundle_path, receipt)
    bundle = json.loads(bundle_path.read_text())
    for key in ('native_config', 'checkpoint', 'conditioner_checkpoint'):
        if bundle.get(key):
            check.check(bundle[key], receipt)
    cfg = json.loads(Path(bundle['native_config']).read_text())
    variant = validate_bundle(bundle, cfg)
    ids, seeds = validate_panel(plan, bundle, cfg)
    bound = {**plan['source_SHA256'], **plan.get('input_SHA256', {})}
    if any(file_sha256(p) != digest for p, digest in bound.items()):
        raise ValueError('native evaluator or bound training runtime changed')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if gpu not in plan['assigned_GPUs'] or gpu not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(','):
        raise ValueError('one assigned physical GPU0-3 required')
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)

    import torch
    from PIL import Image
    from pixal3d import datasets, models
    from train import setup_rng
    from . import quick_metrics as quick
    from .evaluate_ss_artifact_subset import capture_ss_artifacts, validate_subset_reference
    from .raw_correspondence_native_runtime import parameter_groups, selected_trainer, state_sha
    setup_rng(0, cfg['training_seed'])
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(plan['deterministic_algorithms'])
    validation = copy.deepcopy(cfg['validation'])
    validation['quick_metrics'].update(panel_size=len(ids), panel_file=plan['panel'],
        seed_map_file=plan['seeds'], generation_batch_size=8, log_progress=True)
    dataset = getattr(datasets, validation['dataset_name'])(
        validation['roots_file'], **validation['dataset_args'])
    if len(dataset) != 508:
        raise ValueError('complete validation508 manifest required; explicit panel selects only DEV assets')
    with torch.device('meta'):
        md = {name: getattr(models, spec['name'])(**spec['args']) for name, spec in cfg['models'].items()}
    kw = copy.deepcopy(cfg['trainer']['args'])
    kw.update(continuation_ckpt={'denoiser': bundle['checkpoint']},
        batch_size_per_gpu=1, batch_split=1, num_workers=0, prefetch_data=False, max_steps=1)
    with cold_source_binding(variant):
        trainer = selected_trainer(variant)(md, dataset, **kw, output_dir=str(output / 'loader'),
            load_dir=None, step=None, wandb_run=None)
    if bundle['conditioner_checkpoint']:
        trainer.deform_conditioner.load_state_dict(torch.load(
            bundle['conditioner_checkpoint'], map_location='cpu', weights_only=True), strict=True)
    lora, source = parameter_groups(trainer, variant)
    before = state_sha(list(lora.items()) + list(source.items()))
    if trainer.optimizer.state or trainer.step != 0:
        raise ValueError('evaluation must create a fresh empty optimizer; no training/resume')
    for model in trainer.models.values():
        model.eval()
    trainer.p_uncond = 0.
    captured = []
    context = capture_ss_artifacts(quick, output, captured) if plan['mode'] in ('archive32','final128') else nullcontext()
    with context, torch.no_grad(), torch.autocast('cuda', dtype=trainer.mix_precision_dtype):
        metrics, panel, actual_ids = quick.evaluate_quick(trainer, dataset, validation)
    if (actual_ids != ids or trainer.optimizer.state
            or before != state_sha(list(lora.items()) + list(source.items()))):
        raise ValueError('evaluation asset order, optimizer or loaded trainable weights changed')
    rows = trainer._quick_per_asset
    if ([r['sample_id'] for r in rows] != ids
            or any(r['seed'] != seeds[r['sample_id']] for r in rows)):
        raise ValueError('actual generated asset/noise identities differ from fixed plan')
    result = dict(status='PASS', stage='ss64', step=bundle['step'], weights='raw',
        checkpoint=bundle['checkpoint'], conditioner_checkpoint=bundle['conditioner_checkpoint'],
        arm_config=bundle['native_config'], arm=variant, conditioning_mode='edited-rgb',
        target_modality='rgb', input_states=['M1', 'M0', 'M0', 'M0'], input_view_count=4,
        input_camera_ids=dataset.rows[0]['conditioning_camera_ids'], diagnostic_only=False,
        manifest_sha256=file_sha256(validation['roots_file']), generation_batch_size=8,
        sampling_steps=12, guidance=3., optimizer_updates=0, optimizer_updates_during_evaluation=0,
        quick_metrics=metrics, quick_panel_ids=actual_ids, per_asset=rows,
        panel_file=plan['panel'], seed_map_file=plan['seeds'], native_task_checkpoint_bundle=str(bundle_path),
        native_variant=variant, native_flow_training_updates=bundle['step'], source_teacher_updates=0,
        actual_LoRA_tensors=len(lora), actual_source_tensors=len(source),
        actual_loaded_trainable_SHA256=before, deterministic_algorithms=plan['deterministic_algorithms'],
        evaluation_mode=plan['mode'], quality_gain_proven=False, full_goal_complete=False)
    # Keep the actual measurements even if a later exact-replay assertion fails.
    path = output / 'evaluation.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    Image.fromarray(panel).save(output / 'panel.png')
    if plan['mode'] == 'epoch-replay':
        check_epoch_replay(result, bundle['native_validation'])
        result['exact_saved_pair_epoch_metric_replay'] = True
    elif plan['mode'] == 'archive32':
        reference_path = Path(plan['reference_evaluation'])
        reference = json.loads(reference_path.read_text())
        validate_subset_reference(result, reference, captured)
        result['native_SS_artifacts'] = dict(
            status='PASS_ACTUAL_NATIVE_SS_ARCHIVE_AND_EXACT_REFERENCE_SUBSET_REPLAY',
            original_prefix32_all_records_and_occupancy_counts_exact=True, artifacts=captured,
            reference_evaluation=str(reference_path), reference_SHA256=file_sha256(reference_path),
            source_SHA256=bound, reference_asset_count=128, archived_asset_count=32,
            predicted_SS_is_available_for_full_cascade=True, GT_SS_is_for_scoring_only=True,
            full_cascade_generated=False, optimizer_updates=0)
    if plan['mode']=='final128':
        if len(captured)!=128:
            raise ValueError('capture all original final128 predictions for the shared cascade')
        for artifact,row in zip(captured,rows):
            if (artifact['sample_id']!=row['sample_id'] or artifact['seed']!=row['seed']
                    or artifact['target_voxels_SHA256']!=row['target_voxels_sha256']
                    or any(artifact['occupancy_counts'][k]!=row[k] for k in ['tp','fp','fn'])):
                raise ValueError('original final128 archive and actual metrics differ')
        result['native_SS_artifacts']=dict(status='PASS_ACTUAL_ORIGINAL_CORRESPONDENCE_SS128_CAPTURE',
            archived_asset_count=128,same_forward_prediction_and_target_capture=True,
            artifacts=captured,predicted_SS_is_available_for_full_cascade=True,
            GT_SS_is_for_scoring_only=True,full_cascade_generated=False,optimizer_updates=0)
    if any(file_sha256(p) != digest for p, digest in {**bound, **check.inputs}.items()):
        raise ValueError('bound evaluator or closed actual checkpoint inputs changed during evaluation')
    result['verified_inputs'] = check.inputs
    path.write_text(json.dumps(result, indent=2) + '\n')
    (output / 'completed.json').write_text(json.dumps(dict(status='PASS',
        mode=plan['mode'], variant=variant, step=bundle['step'], assets=len(rows),
        optimizer_updates=0, full_goal_complete=False), indent=2) + '\n')
    print(json.dumps(dict(status='PASS', variant=variant, step=bundle['step'], mode=plan['mode'],
        assets=len(rows), quick_metrics=metrics)), flush=True)


if __name__ == '__main__':
    main()
