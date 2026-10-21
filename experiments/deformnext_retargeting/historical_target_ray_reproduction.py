"""Strict bindings for a faithful historical target-patch/R96 reproduction.

Reuse the original numerical entrypoints. Only owned output/logging paths and
the new, disposable capacity proof may change. The source-ray study is not a
substitute for this experiment, and this experiment is not an independent seed.
"""
from copy import deepcopy
import math


ARMS = {
    'retargeting': dict(trainer='AnchorJointTrainer', stage='retargeting-control',
        variant=None, module='train_anchor_joint', states=366),
    'direct-nodq': dict(trainer='RayFeatureJointTrainer', stage='ray-feature-attention',
        variant='direct-physical-NoDQ', module='train_direct_no_query_joint', states=375),
    'direct-dq': dict(trainer='DirectRayJointTrainer', stage='ray-feature-attention',
        variant='direct-DQ2048', module='train_direct_ray_joint', states=376),
}


def _leaves(value, prefix=()):
    if isinstance(value, dict):
        return {p: v for k, item in value.items() for p, v in _leaves(item, prefix+(k,)).items()}
    return {prefix: value}


def validate_source(config, arm):
    expected = ARMS[arm]
    args = config['trainer']['args']
    stage = args['anchor_stage']
    if (config['trainer']['name'] != expected['trainer']
            or stage['arm'] != expected['stage'] or stage.get('variant') != expected['variant']
            or stage['reference_step'] != 96 or stage['stage_rng_seed'] != 101
            or stage['diagnostic_profile'] or stage.get('capacity_profile', False)
            or args['batch_size_per_gpu'] != 48 or args['batch_split'] != 1
            or args['max_steps'] != 480 or config['training_seed'] != 101
            or config['auto_retry'] != 0 or config['ckpt'] != 'none'
            or config.get('load_ckpt') is not None or config['load_dir'] != ''
            or args['continuation_ckpt'] != {'denoiser': stage['denoiser_checkpoint']}
            or args['optimizer'] != dict(name='AdamW', args=dict(
                lr=1e-5, weight_decay=.01, betas=[.9, .95], eps=1e-8))
            or args['model_learning_rates'] != {'deform_conditioner': 1e-4}
            or args['lora'] != dict(rank=16, alpha=16, dropout=0.)):
        raise ValueError('original seed101 R96/DDP2 global96/micro48/accum1/Adam/LoRA contract required')
    module = args['deform_conditioner']
    query = arm == 'direct-dq'
    if (module['projection_mode'] != 'nearest' or module['candidate_aggregation'] != 'top1'
            or module['topk'] != 64 or module['implementation'] != 'r9-native-v5'
            or module['arm'] != ('c2' if query else 'c1')
            or (module.get('deformation_dim') != 2048 if query else 'deformation_dim' in module)
            or config['dataset']['name'] != ('DeformNeXtDeformationQuerySS64' if query else 'DeformNeXtSS64')
            or config['validation']['dataset_name'] != config['dataset']['name']):
        raise ValueError('original native target-patch Top1 correspondence and physical query structure required')


def bind_training(config, *, arm, phase, output, capacity_report, capacity_receipt, wandb_id):
    validate_source(config, arm)
    if phase not in {'capacity', 'science'} or output == config['output_dir']:
        raise ValueError('fresh owned capacity or scientific output required')
    if not all(isinstance(x, str) and x for x in [output, capacity_report, capacity_receipt, wandb_id]):
        raise ValueError('explicit output, own capacity binding and unique logging identity required')
    bound = deepcopy(config)
    bound['output_dir'] = output
    if 'default_output_dir' in bound:
        bound['default_output_dir'] = output
    bound['wandb_name'] = bound['wandb_id'] = wandb_id
    bound['use_wandb'] = config['use_wandb'] if phase == 'science' else False
    stage = bound['trainer']['args']['anchor_stage']
    stage.update(diagnostic_profile=phase == 'capacity', capacity_profile=phase == 'capacity',
                 capacity_proof=capacity_report, capacity_receipt=capacity_receipt)
    allowed = {('output_dir',), ('default_output_dir',), ('wandb_name',), ('wandb_id',), ('use_wandb',)}
    allowed |= {('trainer', 'args', 'anchor_stage', key) for key in
                ['diagnostic_profile', 'capacity_profile', 'capacity_proof', 'capacity_receipt']}
    left, right = _leaves(config), _leaves(bound)
    changed = {p for p in left.keys() | right.keys() if left.get(p) != right.get(p)}
    if not changed <= allowed:
        raise ValueError('numerical source configuration changed')
    return bound, dict(changed_paths=sorted('.'.join(p) for p in changed),
        native_module='experiments.deformnext_retargeting.'+ARMS[arm]['module'],
        original_model_data_camera_latent_LoRA_optimizer_and_sampler_settings_preserved=True,
        world_size=2, global_batch=96, microbatch=48, accumulation=1,
        restored_R_LoRA_Adam_and_sampler_step=96, final_step=480, additional_updates=384,
        independent_training_seed=False, actual_GPU_reproduction_proven=False)


def verify_capacity(reports, goldens, *, arm):
    """Require each actual rank's common step97 loss, not a rank0 proxy."""
    if set(reports) != {0, 1} or set(goldens) != {0, 1}:
        raise ValueError('two actual rank reports and two original golden rank reports required')
    expected = ARMS[arm]['states']
    steps = dict(default=[99], deform_conditioner=[99])
    changed = dict(default=360, deform_conditioner=6)
    if arm != 'retargeting':
        steps['ray_readout'] = [3]
        changed['ray_readout'] = 9
    if arm == 'direct-dq':
        steps['deformation_query'] = [3]
        changed['deformation_query'] = 1
    for rank, row in reports.items():
        golden = goldens[rank]
        if (row['status'] != 'PASS_ACTUAL_MATCHED_ANCHOR_JOINT_SS_PROFILE'
                or (row['world_size'], row['microbatch'], row['accumulation']) != (2, 48, 1)
                or not row['post_update_DDP_equal'] or not row['NN_base_and_matching_frozen_exact']
                or row['restored_common_Adam_states'] != 366 or row['actual_optimizer_states'] != expected
                or row['stage_initial_step'] != 96 or row['scientific_training_updates'] != 0
                or not row['profile_model_discarded'] or row['temporary_updates'] != 3
                or row['changed_trainable_tensors_by_group'] != changed
                or row['final_optimizer_steps_by_group'] != steps
                or [x['step'] for x in row['updates']] != [97, 98, 99]
                or any(not math.isfinite(x['loss']['mse']) for x in row['updates'])):
            raise ValueError('actual disposable matched 366/375/376-state capacity proof differs')
        if (row['updates'][0]['loss'] != golden['updates'][0]['loss']
                or row['frozen_state_SHA256']['denoiser'] != golden['frozen_state_SHA256']['denoiser']):
            raise ValueError('historical same-rank first-loss or frozen native base reproduction differs')
    return dict(status='PASS_ACTUAL_TWO_RANK_R96_TARGET_PATCH_CAPACITY_REPRODUCTION',
        arm=arm, actual_optimizer_states=expected, both_rank97_loss_dictionaries_exact=True,
        native_base_exact_to_historical=True, common_Adam96_and_fresh_head_steps_verified=True,
        all_capacity_updates_discarded=True, scientific_updates=0, independent_training_seed=False)


def reproduction_needed(selection):
    if selection.get('heldout_test_predictions_read') or selection.get('model_adopted'):
        raise ValueError('development-only selection without model adoption required')
    selected = selection['selected_candidate']
    status = selection['status']
    if ((selected is None and status != 'NO_PROMISING_NATIVE_SOURCE_COMBINATION_YET')
            or (selected is not None and status != 'PROMISING_NATIVE_SOURCE_COMBINATION_FOR_SEED_CONFIRMATION')):
        raise ValueError('explicit consistent completed development decision required')
    return selected is None


def stage_jobs(plan, selection, index):
    """A supported source-ray candidate keeps its existing seed202 branch."""
    if not reproduction_needed(selection):
        return []
    if set(index['models']) != set(ARMS):
        raise ValueError('all three matched historical models required')
    result = []
    for arm, model in index['models'].items():
        pair = model['GPU_pair']
        if (len(pair) != 2 or len(set(pair)) != 2
                or not set(pair) <= set(plan['assigned_GPUs'])):
            raise ValueError('explicit assigned pair for each historical arm required')
        prefix = 'faithful-target-ray-'+arm
        for phase in ['capacity', 'science', 'noise0', 'noise1', 'noise2']:
            item = model['phases'][phase]
            parents = [] if phase == 'capacity' else ['capacity' if phase == 'science' else 'science']
            command = ['env', '-C', plan['repo'], 'PYTHONPATH='+plan['repo'], plan['python'], '-u', '-m',
                'experiments.deformnext_retargeting.historical_target_ray_stage',
                '--plan', item['plan'], '--plan-sha256', item['plan_SHA256']]
            job = dict(name=prefix+'-'+phase+'-v1', resource='exclusive', priority=2,
                depends_on=[plan['activation_job']]+[prefix+'-'+p+'-v1' for p in parents],
                requires=[plan['templates_receipt'], plan['activation_receipt'], item['receipt']],
                outputs=[item['output_receipt']], command=command)
            if phase in {'capacity', 'science'}:
                job['gpus'] = list(pair)
            result.append(job)
    result.append(dict(name='summarize-faithful-target-ray-reproduction-v1', resource='cpu', priority=2,
        depends_on=['faithful-target-ray-'+arm+'-noise'+str(n)+'-v1' for arm in ARMS for n in range(3)],
        requires=[plan['templates_receipt'], plan['activation_receipt']],
        outputs=[plan['summary_receipt']],
        command=['env', '-C', plan['repo'], 'PYTHONPATH='+plan['repo'], plan['python'], '-u', '-m',
            'experiments.deformnext_retargeting.historical_target_ray_summary',
            '--plan', plan['summary_plan'], '--plan-sha256', plan['summary_plan_SHA256']]))
    return result
