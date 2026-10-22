from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.native_final_freeze import make_freeze, pbr_binding
from experiments.deformnext_retargeting.native_fullcascade_contract import PAIRS
from experiments.deformnext_retargeting.native_population_contract import SCHEMA


def spec(path, number=1):
    return dict(path=path, SHA256=f'{number:064x}')


def pbr_inputs(mode='top1-only'):
    root='/native/'+mode
    cfg=dict(output_dir=root, training_seed=101,
        experiment=dict(PBR_source_mode=mode, native_task_objective='PBR_sparse_flow_mse',
                        source_teacher_updates=0, engineering_only=False),
        trainer=dict(args=dict(deform_conditioner=None if mode=='vanilla' else dict(appearance_mode=mode))))
    completed=dict(status='PASS_ACTUAL_ORIGINAL_PBR16850_FRESH_NATIVE1900_STAGE', mode=mode,
        actual_optimizer_updates=1900, training_seed=101, source_multiview_step=16850,
        source_tensors=0 if mode=='vanilla' else 10 if mode=='top1-only' else 19,
        stage_training_complete=True, frozen_base_matcher_and_anchors_exact=True,
        post_training_DDP_parameters_exact=True, world_size=4, global_batch=24, microbatch=6, accumulation=1)
    hashes={root+'/'+name:f'{i+20:064x}' for i,name in enumerate([
        'config.json','ckpts/denoiser_step0001900.pt','ckpts/deform_conditioner_step0001900.pt'])}
    return cfg,completed,hashes


def inputs():
    def rows(split, n):
        return [dict(sample_id=split+str(i), split=split, source_uid=split+'UID'+str(i),
                     source_cluster_id=split+'cluster'+str(i%134)) for i in range(n)]
    data=rows('test',508)
    ss={}; pbr={}; parities={}; evaluations={}
    for i,(variant,mode) in enumerate(PAIRS.items()):
        root='/SS/'+variant
        binding=dict(schema=SCHEMA, variant=variant, step=480, training_seed=101,
            native_config=spec(root+'/config.json',i+1),
            checkpoint=spec(root+'/ckpts/denoiser_step0000480.pt',i+1),
            conditioner_checkpoint=None if variant=='vanilla' else
                spec(root+'/ckpts/deform_conditioner_step0000480.pt',i+1))
        ss[variant]=binding
        c,s,h=pbr_inputs(mode);h[c['output_dir']+'/ckpts/denoiser_step0001900.pt']=f'{i+30:064x}'
        pbr[mode]=pbr_binding(c,s,h)
        parities[variant]=dict(status='PASS_ACTUAL_NATIVE_BOUND_VALIDATION36_REPEATABILITY',
            model_binding=binding, independent_model_loads=2, reference_prefix32_records_exact=True,
            all36_records_and_archived_occupancy_exact=True, tail_generation_batch=4, optimizer_updates=0)
        evaluations[variant]=dict(status='PASS', native_variant=variant, evaluation_mode='final128',step=480,
            per_asset=[{}]*128, checkpoint=binding['checkpoint']['path'], arm_config=binding['native_config']['path'],
            conditioner_checkpoint=None if variant=='vanilla' else binding['conditioner_checkpoint']['path'],
            verified_inputs={v['path']:v['SHA256'] for k,v in binding.items()
                             if k in ('checkpoint','native_config','conditioner_checkpoint') and v is not None})
    stages={name:dict(step=step,config=spec('/original/'+name+'/config.json'),
        checkpoint=spec('/original/'+name+f'/ckpts/denoiser_step{step:07d}.pt'))
        for name,step in [('shape512',4210),('shape1024',16850),('pbr1024',16850)]}
    development=dict(
        native_SS128=dict(status='PASS_PAIRED_NATIVE_FOUR_MODEL_DEV128_THREE_NOISE_ANALYSIS',
                         assets=128,source_clusters=93,held_out_test508_evaluated=False),
        native_cascade32=dict(status='PASS_ACTUAL_NATIVE_SS480_PBR1900_EIGHT_MODEL32_EIGHT_METRICS',
            assets=32,views=768,fixed_original_Shape_models=True,
            source_LPIPS_DreamSim_CLIP_and_target_PSNR_SSIM_MIoU_CD_FScore_measured=True),
        native_two_seeds=dict(status='PASS_ACTUAL_NATIVE_TWO_TRAINING_SEEDS_PAIRED128_THREE_GENERATION_NOISES',
            training_seeds=[101,202],actual_complete480_input_noise_timestep_global_traces_equal_within_seed=True,
            actual_training_streams_differ_between_101_and_202=True,
            actual_original_frozen_NN_base_tensors_exact_in_all_four_final_checkpoints=True,
            Top1_R_IoU_direction_positive_in_both_training_seeds=False))
    return dict(ss_models=ss,pbr_models=pbr,original_stages=stages,rows=data,
        train_rows=rows('train',9149),validation_rows=rows('validation',508),
        seeds={r['sample_id']:100+i for i,r in enumerate(data)},parities=parities,
        development=development,development_evidence=[spec('/dev/'+str(i)) for i in range(3)],
        ss_evaluations=evaluations,cascade_statistics=dict(
            original_stage_checkpoint_SHA256={k:v['checkpoint']['SHA256'] for k,v in stages.items()},
            trained_PBR_checkpoint_SHA256={k:v['checkpoint']['SHA256'] for k,v in pbr.items()}),
        heldout_quality_read_before_freeze=False)


def test_all_models_and_negative_confirmation_are_preserved_without_forced_adoption():
    args=inputs();before=deepcopy(args);freeze=make_freeze(**args)
    assert args==before and set(freeze['models'])==set(PAIRS)
    assert not freeze['automatic_model_adoption'] and not freeze['forced_attention_success']
    assert sum(map(len,freeze['primary_metric_families'].values()))==10
    assert freeze['population']['rendered_cameras']==[8,9,10]
    assert len(freeze['original_zero_controls'])==2 and freeze['input_cameras']==[8,9,10,11]
    assert not freeze['all_ten_metrics_measured'] and not freeze['full_goal_complete']


@pytest.mark.parametrize('damage',[
    'test-read','missing-arm','mismatched-PBR','mismatched-SS','old-step','source-removed','no-parity',
    'old-parity-model','seed-not-independent','missing-full-cascade','changed-original-shape',
    'development-weight-swap','training-overlap','missing-test','nonhash-evidence',
])
def test_cannot_freeze_with_leakage_partial_evidence_or_different_actual_weights(damage):
    a=inputs()
    if damage=='test-read':a['heldout_quality_read_before_freeze']=True
    elif damage=='missing-arm':a['ss_models'].pop('R_raw_topk')
    elif damage=='mismatched-PBR':a['pbr_models']['top1-only']['mode']='vanilla'
    elif damage=='mismatched-SS':a['ss_models']['R_top1_only']['variant']='vanilla'
    elif damage=='old-step':a['ss_models']['R_top1_only']['step']=96
    elif damage=='source-removed':a['ss_models']['R_top1_only']['conditioner_checkpoint']=None
    elif damage=='no-parity':a['parities']['vanilla']['independent_model_loads']=1
    elif damage=='old-parity-model':a['parities']['vanilla']['model_binding']=dict(a['ss_models']['vanilla'],step=96)
    elif damage=='seed-not-independent':a['development']['native_two_seeds']['actual_training_streams_differ_between_101_and_202']=False
    elif damage=='missing-full-cascade':a['development']['native_cascade32']['views']=0
    elif damage=='changed-original-shape':a['original_stages']['shape512']['checkpoint']['SHA256']='f'*64
    elif damage=='development-weight-swap':a['ss_evaluations']['vanilla']['verified_inputs'][a['ss_models']['vanilla']['checkpoint']['path']]='f'*64
    elif damage=='training-overlap':a['rows'][0]['source_cluster_id']=a['train_rows'][0]['source_cluster_id']
    elif damage=='missing-test':a['rows'].pop()
    elif damage=='nonhash-evidence':a['development_evidence'][0]['SHA256']='not a hash'
    with pytest.raises(ValueError):make_freeze(**a)


@pytest.mark.parametrize('damage',['incomplete','pilot','old-step','source-count','seed','DDP','wrong-mode'])
def test_pbr_binding_requires_actual_scientific_completion(damage):
    cfg,completion,hashes=pbr_inputs()
    if damage=='incomplete':completion['stage_training_complete']=False
    elif damage=='pilot':cfg['experiment']['engineering_only']=True
    elif damage=='old-step':completion['actual_optimizer_updates']=3
    elif damage=='source-count':completion['source_tensors']=19
    elif damage=='seed':completion['training_seed']=202
    elif damage=='DDP':completion['world_size']=2
    elif damage=='wrong-mode':cfg['trainer']['args']['deform_conditioner']['appearance_mode']='raw-topk-plus-feature'
    with pytest.raises(ValueError):pbr_binding(cfg,completion,hashes)


def test_vanilla_pbr_physically_omits_source_weight():
    binding=pbr_binding(*pbr_inputs('vanilla'))
    assert binding['conditioner_checkpoint'] is None and binding['source_tensors']==0
