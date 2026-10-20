import copy
import hashlib
import json
import sys

import pytest
import torch

from experiments.deformnext_retargeting import member_feature_runtime as runtime
from experiments.deformnext_retargeting.member_feature_recipe import NAME, SCHEMA, ARM, READOUT, validate_feature_recipe
from experiments.deformnext_retargeting.member_posterior_runtime import PROPOSAL
from experiments.deformnext_retargeting.member_feature_value_control import feature_value_control
from pixal3d.trainers.flow_matching.member_feature_source_ray import verify_utility
from pixal3d.trainers.flow_matching.member_feature_source_ray import MemberFeatureSourceRayImageConditionedProjFlowMatchingCFGTrainer as Trainer
from test_member_feature_source_ray import pair, delta


def recipe():
    return dict(trainer=dict(name=NAME,args=dict(
        ffn_chunk_assets=24,batch_size_per_gpu=96,batch_split=1,max_steps=480,prefetch_data=False,
        mix_precision_mode='amp',mix_precision_dtype='bfloat16',ema_rate=[],microbatch_to_device=True,
        lora=dict(rank=16,alpha=16,dropout=0.),continuation_ckpt={'denoiser':'original.pt'},
        optimizer=dict(name='AdamW',args=dict(lr=1e-5)),model_learning_rates={'deform_conditioner':1e-4},
        lr_scheduler=dict(name='LinearWarmupLRScheduler',args=dict(warmup_steps=25)),p_uncond=.1,i_save=96,
        ray_feature_utility=dict(checkpoint='utility.pt',SHA256='0'*64,completed='utility-completed.json'),
        deform_conditioner=dict(arm='c1',ray_mode='semantic-top1',correspondence_mode='semantic',
            candidate_aggregation='top1',task_attention=None,freeze_q2c=True,topk=64,
            score_temperature=.05,ray_chunk_size=64))),
        experiment=dict(conditioning_schema=SCHEMA,readout_arm=ARM,feature_readout=READOUT,
            matching_proposals=PROPOSAL,no_query_fallback='Native pretrained MV4 prior, zero source correction',
            initial_function='Exact native MV4 global/projection/negative; zero source correction',
            capacity_probe=False,physical_deformation_query_channels=0,target_modality='rgb',
            validate_at_start=True,training_assets=9149,updates_per_epoch=96,
            global_batch=96,world_size=1,microbatch_per_gpu=96),
        training_seed=101,auto_retry=0,load_ckpt=None,load_dir='',ckpt='none',use_wandb=True,
        wandb_entity='youngwoojeon-unist',wandb_project='pixal3d-deformnext-retargeting',wandb_id='new-run',
        num_gpus=1,num_nodes=1,tryrun=False,profile=False,
        dataset=dict(name='DeformNeXtSS64',args=dict(expected_split='train')),
        validation=dict(dataset_name='DeformNeXtSS64',dataset_args=dict(expected_split='validation'),
            every_steps=96,max_assets=32,quick_metrics=dict(enabled=True,panel_size=32,sampling_steps=12,seed=42)))


def test_exact_science_contract_is_checked_without_mutating_the_resolved_recipe():
    cfg=recipe();before=copy.deepcopy(cfg)
    assert validate_feature_recipe(cfg)==ARM and cfg==before
    cfg['training_seed']=202
    assert validate_feature_recipe(cfg)==ARM


@pytest.mark.parametrize('path,value',[
    (['trainer','args','batch_split'],2),(['trainer','args','batch_size_per_gpu'],48),
    (['trainer','args','max_steps'],960),(['trainer','args','ffn_chunk_assets'],96),
    (['trainer','args','deform_conditioner','task_attention'],{'decoder':True}),
    (['trainer','args','deform_conditioner','deformation_dim'],2048),
    (['trainer','args','deform_conditioner','ray_mode'],'semantic-attention'),
    (['trainer','args','ray_feature_utility','SHA256'],'bad'),
    (['experiment','readout_arm'],'top1'),(['experiment','physical_deformation_query_channels'],2048),
    (['experiment','initial_function'],'target-only'),(['validation','quick_metrics','sampling_steps'],24),
    (['trainer','args','optimizer','args','lr'],1e-4),(['dataset','args','expected_split'],'validation')])
def test_recipe_rejects_changed_native_protocol_or_hidden_query_channels(path,value):
    cfg=recipe();parent=cfg
    for key in path[:-1]:parent=parent[key]
    parent[path[-1]]=value
    with pytest.raises(ValueError):validate_feature_recipe(cfg)


def test_same_forward_feature_control_checks_actual_weights_and_original_use():
    _,model,data=pair();model.eval()
    torch.nn.init.normal_(model.ray_readout.output.weight,std=.01)
    events=[]
    with torch.no_grad():
        normal,_=delta(model,data)
        with feature_value_control(model,events):wrong,_=delta(model,data)
        restored,_=delta(model,data)
    assert not torch.equal(normal,wrong) and torch.equal(normal,restored)
    assert events and events[0]['queries']>0
    assert all(chunk['all_four_head_weights_exact_same_forward'] for chunk in events[0]['chunks'])
    assert events[0]['source_use_probability_exact_same_forward']
    assert events[0]['original_Top1_feature_carrier_exact']
    assert 'forward' not in model.__dict__
    assert not model.ray_readout.query._forward_hooks and not model.ray_readout.value._forward_hooks
    assert not model.ray_readout._forward_hooks and not model.evidence_use._forward_pre_hooks


def test_control_restores_all_hooks_and_forward_after_failed_head(monkeypatch):
    _,model,data=pair();model.eval();previous=model.forward
    def fail(*args):raise RuntimeError('expected head failure')
    monkeypatch.setattr(model.ray_readout,'forward',fail)
    with torch.no_grad(),pytest.raises(RuntimeError,match='expected head failure'):
        with feature_value_control(model,[]):model(**data)
    assert model.forward==previous and 'forward' not in model.__dict__
    for layer in [model.ray_readout,model.ray_readout.query,model.ray_readout.key,
                  model.ray_readout.score_selector,model.ray_readout.value]:
        assert not layer._forward_hooks
    assert not model.evidence_use._forward_pre_hooks


def test_historical_utility_provenance_requires_matching_hash_and_all_seven_epochs(tmp_path):
    checkpoint=tmp_path/'utility.pt';checkpoint.write_bytes(b'complete verified utility test')
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    completed=tmp_path/'completed.json'
    record=dict(status='PASS_FIXED_SEVEN_EPOCH_TARGET_PATCH_UTILITY_TRAINING',epochs=7,
        optimizer_updates=112,pure_utility_export_SHA256=digest)
    completed.write_text(json.dumps(record))
    spec=dict(checkpoint=str(checkpoint),SHA256=digest,completed=str(completed))
    assert verify_utility(spec)==digest
    record['epochs']=3;completed.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='seven-epoch'):verify_utility(spec)
    completed.write_text(json.dumps(dict(record,epochs=7)))
    checkpoint.write_bytes(b'changed utility')
    with pytest.raises(ValueError,match='seven-epoch'):verify_utility(spec)


def test_trainer_installs_actual_feature_class_and_complete_utility_before_native_lifecycle(tmp_path,monkeypatch):
    from pixal3d.trainers.flow_matching.member_posterior_source_ray import MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer as Parent
    baseline,model,_=pair()
    checkpoint=tmp_path/'utility.pt'
    torch.save(model.ray_readout.score_selector.state_dict(),checkpoint)
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    completed=tmp_path/'completed.json'
    completed.write_text(json.dumps(dict(status='PASS_FIXED_SEVEN_EPOCH_TARGET_PATCH_UTILITY_TRAINING',
        epochs=7,optimizer_updates=112,pure_utility_export_SHA256=digest)))
    settings=dict(arm='c1',implementation='r9-native-v5',key_dim=1024,value_dim=1024,
        query_dim=1024,hidden_dim=256,topk=64,score_temperature=.5,projection_mode='native-bilinear',
        ray_mode='semantic-top1',ray_chunk_size=4,candidate_aggregation='top1')
    observed=[]
    def native_lifecycle(self,**kwargs):
        observed.append((type(self.deform_conditioner),kwargs))
        assert self.models['deform_conditioner'] is self.deform_conditioner
    monkeypatch.setattr(Parent,'init_models_and_more',native_lifecycle)
    monkeypatch.setattr(torch.nn.Module,'cuda',lambda self:self)
    trainer=object.__new__(Trainer);trainer.image_cond_model=object();trainer.deform_conditioner=None
    trainer.models={};trainer.deform_conditioner_config=settings
    trainer.init_models_and_more(ray_feature_utility=dict(checkpoint=str(checkpoint),SHA256=digest,
        completed=str(completed)),ffn_chunk_assets=24)
    assert observed==[(type(model),dict(ffn_chunk_assets=24))]
    assert trainer._ray_feature_utility_initialization['SHA256']==digest
    assert trainer._ray_feature_utility_initialization['joint_feature_head_trainable']
    assert sum(p.requires_grad for p in trainer.deform_conditioner.parameters())==19


def test_phase_arguments_are_restored_when_the_actual_phase_fails(monkeypatch):
    from experiments.deformnext_retargeting import probe_member_feature_source_ray as proof
    def fail():
        assert '--phase' not in sys.argv
        raise RuntimeError('expected phase failure')
    monkeypatch.setattr(proof,'main',fail)
    monkeypatch.setattr(sys,'argv',['feature-runtime','--phase','native','--config','unused.json'])
    previous=sys.argv
    with pytest.raises(RuntimeError,match='expected phase failure'):runtime.main()
    assert sys.argv is previous
