import copy
import hashlib
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from experiments.deformnext_retargeting import member_feature_deformation_runtime as runtime
from experiments.deformnext_retargeting.member_feature_deformation_recipe import NAME,DATASET,SCHEMA,ARM,validate_feature_deformation_recipe
from pixal3d.trainers.flow_matching.member_feature_deformation_source_ray import MemberFeatureDeformationSourceRayImageConditionedProjFlowMatchingCFGTrainer as Trainer
from test_member_feature_runtime import recipe as NoDQ_recipe
from test_member_feature_deformation_source_ray import matched


def recipe():
    cfg=NoDQ_recipe();cfg['trainer']['name']=NAME;cfg['dataset']['name']=DATASET
    cfg['validation']['dataset_name']=DATASET
    cfg['experiment'].update(conditioning_schema=SCHEMA,readout_arm=ARM,physical_deformation_query_channels=2048)
    cfg['trainer']['args']['deform_conditioner']['deformation_dim']=2048
    for args in [cfg['dataset']['args'],cfg['validation']['dataset_args']]:
        args.update(deformation_query_channels=2048,match_root=None,task_matching_features=False,
            transport_arm='c1',transport_implementation='r9-native-v5')
    return cfg


def test_matches_the_entire_original_optimization_recipe_without_mutation():
    cfg=recipe();before=copy.deepcopy(cfg)
    assert validate_feature_deformation_recipe(cfg)==ARM and cfg==before


@pytest.mark.parametrize('path,value',[
    (['trainer','args','batch_split'],2),(['trainer','args','batch_size_per_gpu'],48),
    (['trainer','args','deform_conditioner','deformation_dim'],0),
    (['dataset','args','deformation_query_channels'],0),(['validation','dataset_name'],'DeformNeXtSS64'),
    (['dataset','args','match_root'],'old-cache'),(['dataset','args','task_matching_features'],True),
    (['validation','dataset_args','deformation_root'],'GT'),(['experiment','physical_deformation_query_channels'],0)])
def test_rejects_hidden_zero_channels_old_matches_or_changed_native_batch(path,value):
    cfg=recipe();parent=cfg
    for key in path[:-1]:parent=parent[key]
    parent[path[-1]]=value
    with pytest.raises(ValueError):validate_feature_deformation_recipe(cfg)


def test_scoped_RAW_dataset_registration_and_arguments_restore_after_failed_phase(monkeypatch):
    from pixal3d import datasets
    from pixal3d.datasets.deformnext_member_raw import DeformNeXtMemberRawSS64
    from experiments.deformnext_retargeting import probe_member_feature_deformation_source_ray as phase
    before=datasets.__dict__.get(DATASET)
    def fail():
        assert datasets.__dict__[DATASET] is DeformNeXtMemberRawSS64 and '--phase' not in sys.argv
        raise RuntimeError('expected phase failure')
    monkeypatch.setattr(phase,'main',fail)
    monkeypatch.setattr(sys,'argv',['runtime','--phase','native','--config','unused.json'])
    previous=sys.argv
    with pytest.raises(RuntimeError,match='expected phase failure'):runtime.main()
    assert sys.argv is previous and datasets.__dict__.get(DATASET) is before


def test_actual_DQ_class_and_utility_are_installed_before_native_initialization(tmp_path,monkeypatch):
    from pixal3d.trainers.flow_matching.member_posterior_source_ray import MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer as Parent
    _,model,_,_=matched();checkpoint=tmp_path/'utility.pt'
    torch.save(model.ray_readout.score_selector.state_dict(),checkpoint)
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest();completed=tmp_path/'completed.json'
    completed.write_text(json.dumps(dict(status='PASS_FIXED_SEVEN_EPOCH_TARGET_PATCH_UTILITY_TRAINING',
        epochs=7,optimizer_updates=112,pure_utility_export_SHA256=digest)))
    seen=[]
    monkeypatch.setattr(Parent,'init_models_and_more',lambda self,**kw:seen.append((type(self.deform_conditioner),kw)))
    monkeypatch.setattr(torch.nn.Module,'cuda',lambda self:self)
    trainer=object.__new__(Trainer);trainer.image_cond_model=object();trainer.deform_conditioner=None;trainer.models={}
    trainer.deform_conditioner_config=dict(arm='c1',implementation='r9-native-v5',key_dim=1024,value_dim=1024,
        query_dim=1024,hidden_dim=256,topk=64,score_temperature=.5,projection_mode='native-bilinear',
        ray_mode='semantic-top1',ray_chunk_size=4,candidate_aggregation='top1',deformation_dim=2048)
    trainer.init_models_and_more(ray_feature_utility=dict(checkpoint=str(checkpoint),SHA256=digest,
        completed=str(completed)),ffn_chunk_assets=24)
    assert seen==[(type(model),dict(ffn_chunk_assets=24))]
    assert trainer._ray_feature_utility_initialization['physical_deformation_query_channels']==2048
    assert trainer.deform_conditioner.retargeter.input_dim==4096
    assert sum(p.requires_grad for p in trainer.deform_conditioner.parameters())==20


def test_trainer_routes_actual_RAW_contexts_and_keeps_native_MV4_global_negative():
    from pixal3d.modules.target_first_ray_v2 import TargetFirstRayEvidence
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin,ProjGrid
    grid=ProjGrid(16,512);transforms=torch.eye(4).repeat(1,4,1,1);transforms[:,:,2,3]=2
    camera=dict(transform_matrix=transforms,distance=torch.full((1,4),2.),camera_angle_x=torch.ones(1,4),mesh_scale=torch.ones(1))
    native=dict(proj=torch.randn(1,4096,1024),**{'global':torch.randn(1,1,1024)})
    negative={key:torch.randn_like(value) for key,value in native.items()};target={key:torch.randn_like(value) for key,value in native.items()}
    seen=[]
    class Conditioner:
        def __call__(self,**kwargs):
            seen.append(kwargs);mass=torch.zeros(1,3,1024,1)
            return TargetFirstRayEvidence(torch.zeros_like(kwargs['source_values']),mass,SimpleNamespace(),torch.zeros(1,3072))
    trainer=object.__new__(Trainer);trainer.image_attn_mode='proj';trainer.deform_conditioner=Conditioner()
    trainer.image_cond_model=SimpleNamespace(proj_grid=grid,fixed_projection_transform=grid.front_view_transform_matrix)
    trainer.encode_image_proj=lambda images,**kwargs:(target,negative)
    transport=dict(target_keys=torch.randn(1,1024,1024),source_keys=torch.randn(1,3,1024,1024),
        target_query=torch.randn(1,1024,1024),source_values=torch.randn(1,3,1024,1024),source_valid=torch.ones(1,3,1024,dtype=torch.bool),
        target_valid=torch.ones(1,1024,dtype=torch.bool),patch_grid=(torch.arange(4096)%1024).reshape(1,16,16,16),
        target_context=torch.randn(1,1024,2048),source_context=torch.randn(1,3,1024,2048))
    with patch.object(ImageConditionedProjMixin,'_encode_deform_views',return_value=(native,negative)):
        condition,neg=trainer._encode_deform_views(torch.rand(1,4,3,8,8),camera,transport)
    assert seen[0]['target_context'] is transport['target_context'] and seen[0]['source_context'] is transport['source_context']
    assert all(torch.equal(condition[key],value) for key,value in native.items()) and neg is negative
    assert condition['global'] is native['global']


def test_activated_NoDQ_parity_helper_restores_source_mappings_and_determinism_on_failure(monkeypatch):
    from experiments.deformnext_retargeting.member_feature_deformation_parity import verify_activated_NoDQ_identity
    _,model,_,_=matched();settings=dict(arm='c1',implementation='r9-native-v5',key_dim=1024,value_dim=1024,
        query_dim=1024,hidden_dim=256,topk=64,score_temperature=.5,projection_mode='native-bilinear',
        ray_mode='semantic-top1',ray_chunk_size=4,candidate_aggregation='top1',deformation_dim=2048)
    trainer=SimpleNamespace(deform_conditioner=model,models={'deform_conditioner':model},
        training_models={'deform_conditioner':model},device=torch.device('cpu'))
    def fail(*args):raise RuntimeError('expected native condition failure')
    trainer._encode_deform_views=fail
    monkeypatch.setattr(torch.nn.Module,'cuda',lambda self:self)
    monkeypatch.setattr(torch.random,'fork_rng',lambda **kwargs:nullcontext())
    before=torch.are_deterministic_algorithms_enabled()
    with pytest.raises(RuntimeError,match='expected native condition failure'):
        verify_activated_NoDQ_identity(trainer,dict(trainer=dict(args=dict(deform_conditioner=settings))),
            dict(cond=None),None,None,None)
    assert trainer.deform_conditioner is model
    assert trainer.models['deform_conditioner'] is model and trainer.training_models['deform_conditioner'] is model
    assert torch.are_deterministic_algorithms_enabled()==before
