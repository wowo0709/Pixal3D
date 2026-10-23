"""Physical structure and feature algebra; no CPU neural forwards."""
import copy

import pytest
import torch

from experiments.deformnext_retargeting.unified_design_native_recipe import NAME,SCHEMA,SOURCE_COUNTS,validate_config
from pixal3d.modules.unified_design_controls import (
    UnifiedFullDesignConditioner,UnifiedTop1NoRetargetingConditioner,
    distinct_ray_patch_pairs,voxel_members_by_ray,gather_ray_voxel_members,voxel_readout_chunk,
    NoPriorBoundedReadout,
)
from test_r_mediated_native_recipe import recipe


def control(arm='FULL_QK_ONLY',phase='science'):
    ref=recipe('R_raw_topk');cfg=copy.deepcopy(ref);cfg['trainer']['name']=NAME
    no_R=arm=='C_DECODER_NO_R'
    if no_R:cfg['trainer']['args'].pop('ray_feature_utility')
    cfg['use_wandb']=phase=='science'
    cfg['experiment'].update(native_variant=arm,design_control=arm,conditioning_schema=SCHEMA,
        module_pipeline_schema='unified-module-pipeline-v2',actual_source_tensors=SOURCE_COUNTS[arm],
        retargeter_physically_absent=no_R,physical_deformation_query_channels=0,
        appearance_mode='single-correspondence-Top1-feature' if no_R else 'raw-topk-plus-feature',
        residual_logit_bound=None if no_R else 1.,engineering_only=phase!='science')
    return cfg,ref


@pytest.mark.parametrize('arm',SOURCE_COUNTS)
@pytest.mark.parametrize('phase',['proof','capacity','science'])
def test_only_declared_design_changes_on_common_native_training_contract(arm,phase):
    cfg,ref=control(arm,phase)
    assert validate_config(cfg,phase,ref)==arm


@pytest.mark.parametrize('damage',['R-presence','source-count','other-design','batch','data','utility','teacher','extra-DQ'])
def test_rejects_confounding_another_module_or_training_setting(damage):
    cfg,ref=control();e=cfg['experiment'];t=cfg['trainer']['args']
    if damage=='R-presence':e['retargeter_physically_absent']=True
    elif damage=='source-count':e['actual_source_tensors']=5
    elif damage=='other-design':e['design_control']='FULL_MEAN_MATCH'
    elif damage=='batch':t['batch_size_per_gpu']=48
    elif damage=='data':cfg['data_dir']='other assets'
    elif damage=='utility':t.pop('ray_feature_utility')
    elif damage=='teacher':e['source_teacher_updates']=112
    else:e['physical_deformation_query_channels']=2048
    with pytest.raises(ValueError):validate_config(cfg,'science',ref)


def kwargs():
    return dict(arm='c1',implementation='r9-native-v5',key_dim=1024,value_dim=1024,
        query_dim=1024,hidden_dim=256,topk=64,score_temperature=.05,
        projection_mode='native-bilinear',ray_mode='semantic-top1',ray_chunk_size=64,
        candidate_aggregation='top1')


def test_physical_module_removal_and_shared_initial_parameters_without_NN_forward():
    from pixal3d.modules.r_mediated_source_contrast import RMediatedSourceContrastConditioner
    with torch.random.fork_rng():
        torch.manual_seed(101)
        reference=RMediatedSourceContrastConditioner(appearance_mode='raw-topk-plus-feature',residual_logit_bound=1.,**kwargs())
    previous={n:p for n,p in reference.named_parameters() if p.requires_grad}
    for arm in SOURCE_COUNTS:
        with torch.random.fork_rng():
            torch.manual_seed(101)
            module=(UnifiedTop1NoRetargetingConditioner(**kwargs()) if arm=='C_DECODER_NO_R'
                else UnifiedFullDesignConditioner(design=arm,**kwargs()))
        common={n:p for n,p in module.named_parameters() if p.requires_grad}
        assert len(common)==SOURCE_COUNTS[arm]
        assert all(torch.equal(p,previous[n]) for n,p in common.items())
        module.train();assert not module.q2c.training
        if arm=='C_DECODER_NO_R':
            assert module.retargeter is None and not hasattr(module,'ray_readout')
            assert not any(n.startswith('retargeter.') for n in module.state_dict())
        elif arm=='FULL_QK_ONLY':assert isinstance(module.ray_readout,NoPriorBoundedReadout)
        else:assert not isinstance(module.ray_readout,NoPriorBoundedReadout)
        with pytest.raises(RuntimeError):module.load_state_dict(reference.state_dict(),strict=True)


def test_distinct_patch_membership_is_identical_and_abstains_for_missing_observations():
    rays=torch.tensor([[0,0,0,-1],[2,2,-1,3],[-1,-1,-1,-1]])
    patches=torch.tensor([1,1,2,9]);seen=torch.tensor([True,True,True,False])
    ray,patch=distinct_ray_patch_pairs(rays,patches,seen)
    assert ray.tolist()==[0,0,1026] and patch.tolist()==[1,2,1]
    ray,patch=distinct_ray_patch_pairs(rays,patches,torch.zeros_like(seen))
    assert ray.numel()==patch.numel()==0


def test_per_voxel_readouts_are_pooled_before_R_and_keep_the_same_ray_candidates():
    # Pure feature-polynomial callable, not a NN or a learned CPU forward.
    def algebra_head(native,context,keys,values,valid,prior):
        logits=prior[:,None].expand(-1,4,-1)+native[:,0,None,None]*keys[:,None,:,0]
        weights=logits.masked_fill(~valid[:,None],-torch.inf).softmax(-1)
        output=native.square()+(weights.mean(1)[...,None]*values).sum(1)
        return output,weights
    native=torch.zeros(1,4,1024);native[0,:,0]=torch.tensor([1.,3.,5.,7.])
    rays=torch.tensor([[[0,0,1,-1],[-1,-1,-1,-1],[-1,-1,-1,-1]]])
    take=torch.tensor([0,1]);members=voxel_members_by_ray(rays)
    local,batch,voxel,counts=gather_ray_voxel_members(take,members)
    assert local.tolist()==[0,0,1] and voxel.tolist()==[0,1,2] and counts.tolist()==[2,1]
    indices=torch.zeros(3072,64,dtype=torch.long);indices[:,:2]=torch.tensor([0,1])
    valid=torch.zeros(3072,64,dtype=torch.bool);valid[:,:2]=True
    prior=torch.full((3072,64),-torch.inf);prior[:,:2]=0.
    values=torch.zeros(1,3072,1024);values[0,1,0]=2.
    keys=torch.zeros_like(values);keys[0,1,0]=.1
    context=torch.zeros(3072,1024)
    outputs=voxel_readout_chunk(algebra_head,native,context,values,keys,take,indices,valid,prior,
        local,batch,voxel,counts,2)
    changes=[];aggregates=[];probs=[]
    for point in [0,1,2]:
        query=native[:,point];ids=indices[take[local[point]]][None]
        v=values[:,ids[0]];k=keys[:,ids[0]]
        delta,p=algebra_head(query,context[:1],k,v,valid[:1],prior[:1])
        changes.append(delta[0]);aggregates.append((p.mean(1)[...,None]*v).sum(1)[0]);probs.append(p[0])
    for actual,rows in zip(outputs,[changes,aggregates,probs]):
        expected=torch.stack((torch.stack(rows[:2]).mean(0),rows[2]))
        torch.testing.assert_close(actual,expected)
    pooled_query=native[:,:2].mean(1)
    pooled_delta,_=algebra_head(pooled_query,context[:1],keys[:,:64],values[:,:64],valid[:1],prior[:1])
    assert not torch.allclose(outputs[0][0],pooled_delta[0])
    # A different streaming chunk gives the same mathematical control.
    again=voxel_readout_chunk(algebra_head,native,context,values,keys,take,indices,valid,prior,
        local,batch,voxel,counts,1)
    for a,b in zip(outputs,again):torch.testing.assert_close(a,b)
