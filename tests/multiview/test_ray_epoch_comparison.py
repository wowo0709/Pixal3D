"""Intermediate contrasts reject execution drift while allowing declared modules."""
import copy

import pytest

from experiments.deformnext_retargeting.compare_ray_group_epochs import verify_epoch_audit,verify_matched_recipes


def test_static_and_learned_attention_need_their_actual_optimizer_counts():
    audit=dict(status='PASS_ACTUAL_EPOCH_OPTIMIZER_LOGGING_AND_BATCH',epoch=1,step=96,
        training_seed=101,global_batch=96,world_size=2,microbatch_per_rank=48,accumulation=1,
        all_optimizer_steps=96,actual_optimizer_states=366,
        actual_remote_four_numeric_fields_verified=True,actual_remote_PNG_byte_exact=True)
    verify_epoch_audit(audit,'ray-weighted',1)
    with pytest.raises(ValueError,match='optimizer/logging'):
        verify_epoch_audit(audit,'ray-full',1)
    full={**audit,'actual_optimizer_states':371}
    verify_epoch_audit(full,'ray-full',1)
    with pytest.raises(ValueError,match='optimizer/logging'):
        verify_epoch_audit({**full,'actual_remote_PNG_byte_exact':False},'ray-full',1)


def test_module_contrast_cannot_hide_a_different_LR_checkpoint_or_dataset():
    common=dict(models={'name':'SS'},dataset={'manifest':'train9149'},validation={'manifest':'val508'},
        data_dir='/cache',training_seed=101,num_gpus=2,
        trainer={'name':'RayTrainer','args':{'deform_conditioner':{'task_attention':True},
            'continuation_ckpt':{'denoiser':'SS4210'},'optimizer':{'lr':1e-5},'batch_split':1}})
    static=copy.deepcopy(common)
    static['trainer']['args']['deform_conditioner']['task_attention']=None
    verify_matched_recipes(common,static)
    for key,value in [('optimizer',{'lr':2e-5}),('continuation_ckpt',{'denoiser':'different'}),('batch_split',2)]:
        changed=copy.deepcopy(static)
        changed['trainer']['args'][key]=value
        with pytest.raises(ValueError,match='trainer/continuation/learning rate'):
            verify_matched_recipes(common,changed)
    changed=copy.deepcopy(static)
    changed['dataset']={'manifest':'different'}
    with pytest.raises(ValueError,match='dataset'):
        verify_matched_recipes(common,changed)
