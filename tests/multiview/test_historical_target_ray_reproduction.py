from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.historical_target_ray_reproduction import (
    ARMS, bind_training, reproduction_needed, stage_jobs, validate_source, verify_capacity)


def source(arm):
    spec = ARMS[arm]
    query = arm == 'direct-dq'
    stage = dict(arm=spec['stage'], variant=spec['variant'], reference_step=96,
        stage_rng_seed=101, diagnostic_profile=False, capacity_profile=False,
        denoiser_checkpoint='/reference/denoiser96.pt', conditioner_checkpoint='/reference/source96.pt',
        misc_checkpoint='/reference/misc96.pt', capacity_proof='/old/capacity.json',
        capacity_receipt='/old/closure.json')
    conditioner = dict(projection_mode='nearest', candidate_aggregation='top1', topk=64,
        implementation='r9-native-v5', arm='c2' if query else 'c1')
    if query:
        conditioner['deformation_dim'] = 2048
    dataset = 'DeformNeXtDeformationQuerySS64' if query else 'DeformNeXtSS64'
    return dict(training_seed=101, auto_retry=0, ckpt='none', load_ckpt=None, load_dir='',
        output_dir='/old/run', default_output_dir='/old/run', use_wandb=True,
        wandb_name='old', wandb_id='old', dataset=dict(name=dataset, args=dict(image_size=512)),
        validation=dict(dataset_name=dataset, every_steps=96),
        trainer=dict(name=spec['trainer'], args=dict(anchor_stage=stage,
            batch_size_per_gpu=48, batch_split=1, max_steps=480,
            continuation_ckpt={'denoiser': '/reference/denoiser96.pt'},
            optimizer=dict(name='AdamW', args=dict(lr=1e-5, weight_decay=.01, betas=[.9, .95], eps=1e-8)),
            model_learning_rates={'deform_conditioner': 1e-4},
            lora=dict(rank=16, alpha=16, dropout=0.), deform_conditioner=conditioner)))


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('phase', ['capacity', 'science'])
def test_preserves_numerical_settings_and_owned_capacity_binding(arm, phase):
    original = source(arm); snapshot = deepcopy(original)
    bound, audit = bind_training(original, arm=arm, phase=phase, output='/new/'+phase,
        capacity_report='/new/capacity/profile.json', capacity_receipt='/new/capacity/closure.json',
        wandb_id='reproduction-'+arm)
    assert original == snapshot
    assert bound['dataset'] == original['dataset'] and bound['validation'] == original['validation']
    assert bound['trainer']['args']['continuation_ckpt'] == original['trainer']['args']['continuation_ckpt']
    assert bound['trainer']['args']['anchor_stage']['misc_checkpoint'] == '/reference/misc96.pt'
    assert bound['trainer']['args']['anchor_stage']['capacity_proof'] == '/new/capacity/profile.json'
    assert bound['use_wandb'] == (phase == 'science')
    assert audit['additional_updates'] == 384 and audit['world_size'] == 2
    assert not audit['actual_GPU_reproduction_proven']


@pytest.mark.parametrize('change', ['micro', 'seed', 'optimizer', 'warm-step', 'projection', 'zero-DQ'])
def test_rejects_unfaithful_training_or_structural_substitution(change):
    config = source('direct-nodq'); args = config['trainer']['args']
    if change == 'micro': args['batch_size_per_gpu'] = 96
    if change == 'seed': config['training_seed'] = 202
    if change == 'optimizer': args['optimizer']['args']['lr'] = 2e-5
    if change == 'warm-step': args['anchor_stage']['reference_step'] = 0
    if change == 'projection': args['deform_conditioner']['projection_mode'] = 'native-bilinear'
    if change == 'zero-DQ': args['deform_conditioner']['deformation_dim'] = 0
    with pytest.raises(ValueError): validate_source(config, 'direct-nodq')


def reports(arm):
    steps = dict(default=[99], deform_conditioner=[99]); changed = dict(default=360, deform_conditioner=6)
    if arm != 'retargeting': steps['ray_readout'] = [3]; changed['ray_readout'] = 9
    if arm == 'direct-dq': steps['deformation_query'] = [3]; changed['deformation_query'] = 1
    return {rank: dict(status='PASS_ACTUAL_MATCHED_ANCHOR_JOINT_SS_PROFILE',
        world_size=2, microbatch=48, accumulation=1, post_update_DDP_equal=True,
        NN_base_and_matching_frozen_exact=True, restored_common_Adam_states=366,
        actual_optimizer_states=ARMS[arm]['states'], stage_initial_step=96,
        scientific_training_updates=0, profile_model_discarded=True, temporary_updates=3,
        changed_trainable_tensors_by_group=deepcopy(changed), final_optimizer_steps_by_group=deepcopy(steps),
        updates=[dict(step=s, loss=dict(mse=.1+rank, loss=.1+rank)) for s in [97, 98, 99]],
        frozen_state_SHA256={'denoiser': 'same-native-base'}) for rank in [0, 1]}


@pytest.mark.parametrize('arm', ARMS)
def test_checks_real_rank1_not_only_rank0_and_discards_profile(arm):
    actual = reports(arm); golden = deepcopy(actual)
    assert verify_capacity(actual, golden, arm=arm)['both_rank97_loss_dictionaries_exact']
    actual[1]['updates'][0]['loss']['mse'] += .00001
    with pytest.raises(ValueError, match='same-rank first-loss'):
        verify_capacity(actual, golden, arm=arm)


@pytest.mark.parametrize('failure', ['cold-Adam', 'wrong-state-count', 'warm-head-Adam', 'changed-base', 'one-rank'])
def test_capacity_does_not_accept_a_compatible_but_different_initialization(failure):
    actual = reports('direct-dq'); golden = deepcopy(actual)
    if failure == 'cold-Adam': actual[0]['restored_common_Adam_states'] = 0
    if failure == 'wrong-state-count': actual[0]['actual_optimizer_states'] = 375
    if failure == 'warm-head-Adam': actual[1]['final_optimizer_steps_by_group']['ray_readout'] = [99]
    if failure == 'changed-base': actual[1]['frozen_state_SHA256']['denoiser'] = 'different'
    if failure == 'one-rank': del actual[1]
    with pytest.raises(ValueError): verify_capacity(actual, golden, arm='direct-dq')


@pytest.mark.parametrize('selected,status,needed', [
    (None, 'NO_PROMISING_NATIVE_SOURCE_COMBINATION_YET', True),
    ('member-qkv', 'PROMISING_NATIVE_SOURCE_COMBINATION_FOR_SEED_CONFIRMATION', False)])
def test_conditional_reproduction_does_not_replace_promising_candidate_confirmation(selected, status, needed):
    selection = dict(selected_candidate=selected, status=status, heldout_test_predictions_read=False, model_adopted=False)
    assert reproduction_needed(selection) == needed
    selection['heldout_test_predictions_read'] = True
    with pytest.raises(ValueError): reproduction_needed(selection)


def test_complete_negative_graph_preserves_each_capacity_science_and_three_final_noises():
    plan = dict(repo='/source', python='/python', assigned_GPUs=[0, 1, 2, 3],
        activation_job='activate-historical', templates_receipt='/closed/templates.json',
        activation_receipt='/closed/activation.json', summary_receipt='/closed/summary.json',
        summary_plan='/plans/summary.json', summary_plan_SHA256='summary-hash')
    index = dict(models={arm: dict(GPU_pair=[0, 1] if arm != 'direct-nodq' else [2, 3],
        phases={phase: dict(plan='/plans/'+arm+'/'+phase+'.json', plan_SHA256=arm+phase,
            receipt='/plans/closure.json', output_receipt='/closed/'+arm+'/'+phase+'.json')
            for phase in ['capacity', 'science', 'noise0', 'noise1', 'noise2']}) for arm in ARMS})
    selection = dict(selected_candidate=None, status='NO_PROMISING_NATIVE_SOURCE_COMBINATION_YET')
    jobs = stage_jobs(plan, selection, index)
    assert len(jobs) == 16 and len({j['name'] for j in jobs}) == 16
    names = {j['name'] for j in jobs}
    for job in jobs:
        assert set(job['depends_on']) <= names | {'activate-historical'}
        if job['resource'] != 'cpu':
            assert job['resource'] == 'exclusive'
            assert '--plan-sha256' in job['command']
    assert len(jobs[-1]['depends_on']) == 9
    assert all('noise' in name for name in jobs[-1]['depends_on'])
    selection.update(selected_candidate='member-qkv', status='PROMISING_NATIVE_SOURCE_COMBINATION_FOR_SEED_CONFIRMATION')
    assert stage_jobs(plan, selection, index) == []
    selection.update(selected_candidate=None, status='NO_PROMISING_NATIVE_SOURCE_COMBINATION_YET')
    index['models']['direct-nodq']['GPU_pair'] = [6, 7]
    with pytest.raises(ValueError, match='assigned pair'):
        stage_jobs(plan, selection, index)
