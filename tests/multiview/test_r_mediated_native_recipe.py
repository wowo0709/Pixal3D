import copy

import pytest
import torch

from experiments.deformnext_retargeting.r_mediated_native_recipe import (
    NAME,SCHEMA,INITIAL,VARIANTS,cold_source_binding,common_contract,validate_config,
)


def recipe(variant='R_raw_topk',phase='science'):
    cfg=dict(trainer=dict(name=NAME,args=dict(batch_size_per_gpu=96,batch_split=1,max_steps=480,
        ffn_chunk_assets=24,prefetch_data=False,microbatch_to_device=True,num_workers=2,
        mix_precision_mode='amp',mix_precision_dtype='bfloat16',ema_rate=[],
        lora=dict(rank=16,alpha=16,dropout=0.),continuation_ckpt=dict(denoiser='SS4210'),finetune_ckpt=None,
        optimizer=dict(name='AdamW',args=dict(lr=1e-5,weight_decay=.01,betas=[.9,.95],eps=1e-8)),
        lr_scheduler=dict(name='LinearWarmupLRScheduler',args=dict(warmup_steps=25)),p_uncond=.1,i_save=96,
        model_learning_rates=dict(deform_conditioner=1e-4),
        ray_feature_utility=dict(checkpoint='utility',SHA256='hash',completed='completed'),
        deform_conditioner=dict(arm='c1',ray_mode='semantic-top1',candidate_aggregation='top1',
            projection_mode='native-bilinear',freeze_q2c=True,topk=64,score_temperature=.05,
            ray_chunk_size=64,key_dim=1024,query_dim=1024,value_dim=1024,g_max=.25,task_attention=None))),
        experiment=dict(native_variant=variant,conditioning_schema=SCHEMA,initial_function=INITIAL,
            additional_source_teacher_supervision=False,source_teacher_updates=0,native_task_objective='SS_flow_mse',
            engineering_only=phase!='science',training_assets=9149,epochs=5,updates_per_epoch=96,target_modality='rgb',
            global_batch=96,world_size=1,microbatch_per_gpu=96,validate_at_start=True,
            appearance_mode=VARIANTS[variant],residual_logit_bound=1.),
        dataset=dict(name='DeformNeXtSS64',args=dict(expected_split='train',image_size=512,
            expected_ss_schema='deformnext-ss64-m1-anchor-v3',transport_arm='c1')),
        validation=dict(dataset_name='DeformNeXtSS64',dataset_args=dict(expected_split='validation',transport_arm='c1'),
            max_assets=32,every_steps=96,quick_metrics=dict(panel_size=32,generation_batch_size=8,
                sampling_steps=12,panel_file='panel32',seed_map_file='seeds')),
        models=dict(denoiser={'original':'architecture'}),data_dir='TRAIN9149',use_wandb=phase=='science',
        training_seed=101,load_dir='',load_ckpt=None,ckpt='none',auto_retry=0,num_gpus=1,num_nodes=1,tryrun=False,profile=False)
    if variant=='vanilla':
        t=cfg['trainer']['args'];t['deform_conditioner']=None;t.pop('model_learning_rates');t.pop('ray_feature_utility')
        cfg['dataset']['args']['transport_arm']='v1';cfg['validation']['dataset_args']['transport_arm']='v1'
    return cfg


@pytest.mark.parametrize('variant',VARIANTS)
@pytest.mark.parametrize('phase',['proof','capacity','science'])
def test_declared_native_phases_and_common_control_contract(variant,phase):
    cfg=recipe(variant,phase)
    assert validate_config(cfg,phase)==variant
    assert common_contract(cfg)==common_contract(recipe('vanilla',phase))


def test_smaller_internal_ffn_chunk_keeps_actual_microbatch_and_all_arm_contracts():
    configs=[recipe(v,'capacity')for v in VARIANTS]
    for cfg in configs:
        cfg['trainer']['args']['ffn_chunk_assets']=12
        validate_config(cfg,'capacity')
        assert cfg['trainer']['args']['batch_size_per_gpu']==96
        assert cfg['trainer']['args']['batch_split']==1
    assert all(common_contract(c)==common_contract(configs[0])for c in configs)


@pytest.mark.parametrize('damage',['teacher','world','batch','resume','proxy','uniform','fake-DQ','vanilla-source'])
def test_rejects_training_or_module_confound(damage):
    cfg=recipe('vanilla'if damage=='vanilla-source'else'R_raw_topk');t=cfg['trainer']['args'];e=cfg['experiment']
    if damage=='teacher':e['source_teacher_updates']=112
    elif damage=='world':e['world_size']=2
    elif damage=='batch':t['batch_size_per_gpu']=48
    elif damage=='resume':cfg['load_dir']='old-trained-run'
    elif damage=='proxy':e['native_task_objective']='source_feature_MSE'
    elif damage=='uniform':t['deform_conditioner']['candidate_aggregation']='uniform'
    elif damage=='fake-DQ':t['deform_conditioner']['deformation_dim']=2048
    else:t['deform_conditioner']=dict(dq=0)
    with pytest.raises(ValueError):validate_config(cfg,'science')


def test_cold_constructor_preserves_rng_common19_and_restores_registry():
    from pixal3d.trainers.flow_matching import member_feature_source_ray as trainer
    from pixal3d.modules.member_feature_source_ray import MemberFeatureSourceRayConditioner
    kwargs=dict(arm='c1',implementation='r9-native-v5',key_dim=1024,value_dim=1024,
        query_dim=1024,hidden_dim=256,topk=64,score_temperature=.05,projection_mode='native-bilinear',
        ray_mode='semantic-top1',ray_chunk_size=4,candidate_aggregation='top1')
    values=[]
    for variant in ['R_feature','R_raw_topk']:
        state=torch.random.get_rng_state()
        with cold_source_binding(variant)as instances:
            module=trainer.MemberFeatureSourceRayConditioner(**kwargs)
            assert instances==[module]
            assert torch.equal(state,torch.random.get_rng_state())
            module.train()
            assert module.training and not module.q2c.training
            module.load_utility(module.ray_readout.score_selector.state_dict())
            assert torch.equal(state,torch.random.get_rng_state())
            values.append({n:p.detach().clone()for n,p in module.named_parameters()if p.requires_grad})
        assert trainer.MemberFeatureSourceRayConditioner is MemberFeatureSourceRayConditioner
    assert len(values[0])==19 and values[0].keys()==values[1].keys()
    assert all(torch.equal(v,values[1][n])for n,v in values[0].items())
    with pytest.raises(RuntimeError,match='owned'):
        with cold_source_binding('R_feature'):raise RuntimeError('owned')
    assert trainer.MemberFeatureSourceRayConditioner is MemberFeatureSourceRayConditioner
