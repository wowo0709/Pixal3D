from types import SimpleNamespace
from unittest.mock import patch

import torch

from pixal3d.modules.target_first_ray_v2 import TargetFirstRayConditionerV2, project_target_first_evidence
from pixal3d.trainers.flow_matching.ray_group_matching import project_ray_delta
from experiments.deformnext_retargeting.historical_reproduction import target_first_static_projection
from pixal3d.trainers.flow_matching.target_first_ray_v2 import TargetFirstRayImageConditionedProjFlowMatchingCFGTrainerV2 as TrainerV2


class LinearGrid:
    front_view_transform_matrix = torch.eye(4)

    def __call__(self,values,*camera):
        return values.mean((1,2))[:,None].expand(-1,4096,-1)


def camera():
    matrix=torch.eye(4).repeat(1,4,1,1);matrix[:,:,2,3]=2
    return dict(transform_matrix=matrix,distance=torch.ones(1,4)*2,
                camera_angle_x=torch.ones(1,4),mesh_scale=torch.ones(1))


def fixture():
    torch.manual_seed(71)
    model=TargetFirstRayConditionerV2(arm='c1',implementation='r9-native-v5',
        key_dim=1024,value_dim=1024,query_dim=1024,topk=64,hidden_dim=256,
        score_temperature=.5,projection_mode='native-bilinear',ray_mode='semantic-top1',
        ray_chunk_size=4,candidate_aggregation='top1')
    data=dict(native_voxels=torch.randn(1,5,1024),target_omega_voxels=torch.randn(1,5,1024),
        target_native_voxels=torch.randn(1,5,1024),
        ray_ids=torch.tensor([[[0,0,1,1,-1],[2,2,1,1,-1],[3,3,4,4,-1]]]),
        source_keys=torch.randn(1,3,32,1024),source_values=torch.randn(1,3,32,1024),
        source_valid=torch.ones(1,3,32,dtype=torch.bool))
    return model,data


def test_equal_target_evidence_cancels_instead_of_subtracting_an_absent_source():
    grid=LinearGrid();target=torch.full((1,4096,1024),2.)
    mass=torch.full((1,3,1024,1),.25)
    numerator=torch.full((1,3,1024,1024),.5)  # mass*semantic;semantic equals target2
    correction=project_target_first_evidence(grid,numerator,mass,camera(),target)
    assert torch.count_nonzero(correction)==0
    # v1 reused a replacement delta for a source feature that was not in T.
    wrong_delta=torch.full_like(numerator,.5*.5*(2-(-7)))
    old=target+project_ray_delta(grid,wrong_delta,camera())
    assert torch.equal(old,torch.full_like(target,3.6875))


def test_partial_native_write_reproduces_the_positive_historical_target_first_formula():
    grid=LinearGrid();target=torch.full((1,4096,1024),2.)
    mass=torch.zeros(1,3,1024,1);mass[:,:,::2]=.5
    numerator=mass.expand(-1,-1,-1,1024)*6
    projected=torch.stack([grid(numerator[:,v].reshape(1,32,32,1024)) for v in range(3)],1)
    valid=torch.stack([grid(mass[:,v].reshape(1,32,32,1)) for v in range(3)],1)
    historical=target_first_static_projection(target,projected,valid)
    actual=target+project_target_first_evidence(grid,numerator,mass,camera(),target)
    torch.testing.assert_close(actual,historical,rtol=0,atol=0)


def test_retargeting_can_learn_outside_the_initial_zero_semantic_gate():
    model,data=fixture();out=model(**data)
    assert not out.source_numerator.any() and not out.semantic_weight.any()
    loss=(out.source_numerator*torch.randn_like(out.source_numerator)).sum()
    loss.backward()
    assert model.fusion_gate.grad.abs()>0
    assert model.retargeter.net[-1].weight.grad.abs().sum()>0
    with torch.no_grad():
        model.fusion_gate.fill_(.2)
        model.retargeter.net[-1].weight.normal_(std=.001)
        out=model(**data);g=torch.tanh(model.fusion_gate)
        expected=(out.use_probability[...,None]*g).reshape_as(out.semantic_weight)
        assert torch.equal(out.semantic_weight,expected)
        expected=(out.use_probability[...,None]*(g*out.attention.base_evidence+out.attention.retarget_residual)).reshape_as(out.source_numerator)
        torch.testing.assert_close(out.source_numerator,expected)
        empty=model(**{**data,'source_valid':torch.zeros_like(data['source_valid'])})
        assert not empty.source_numerator.any() and not empty.semantic_weight.any()


def test_zero_semantic_weight_does_not_turn_off_the_retargeting_residual():
    target=torch.full((1,4096,1024),2.);mass=torch.zeros(1,3,1024,1)
    residual=torch.full((1,3,1024,1024),.125)
    delta=project_target_first_evidence(LinearGrid(),residual,mass,camera(),target)
    torch.testing.assert_close(delta,torch.full_like(target,3*.125/4),rtol=0,atol=0)


def test_trainer_uses_anchor_cancellation_with_native_target_global_and_negative_path():
    grid=LinearGrid();extractor=SimpleNamespace(proj_grid=grid,fixed_projection_transform=torch.eye(4))
    positive={'global':torch.full((1,5,1024),11.),'proj':torch.full((1,4096,1024),2.)}
    negative={key:torch.zeros_like(value) for key,value in positive.items()}
    calls=[]
    def encode(images,**camera):
        calls.append(images)
        return positive,negative
    def source(**kwargs):
        return SimpleNamespace(source_numerator=torch.full((1,3,1024,1024),.5),
            semantic_weight=torch.full((1,3,1024,1),.25),attention=object(),
            use_probability=torch.ones(1,3072))
    facade=SimpleNamespace(image_attn_mode='proj',image_cond_model=extractor,_image_cond_module=extractor,
        encode_image_proj=encode,deform_conditioner=source,training_models={'deform_conditioner':source})
    images=torch.randn(1,4,3,4,4)
    transport=dict(target_keys=torch.zeros(1,1024,1024),target_query=torch.zeros(1,1024,1024),
        source_keys=torch.zeros(1,3,1024,1024),source_values=torch.full((1,3,1024,1024),-7.),
        source_valid=torch.ones(1,3,1024,dtype=torch.bool),target_valid=torch.zeros(1,1024,dtype=torch.bool),
        patch_grid=(torch.arange(4096)%1024).reshape(1,16,16,16))
    membership=(torch.arange(4096)%1024).reshape(1,1,4096).expand(-1,3,-1)
    with patch('pixal3d.trainers.flow_matching.target_first_ray_v2.native_ray_membership',return_value=membership):
        actual,neg=TrainerV2._encode_deform_views(facade,images,camera(),transport)
    assert len(calls)==1 and torch.equal(calls[0],images[:,0])
    assert actual['global'] is positive['global'] and neg is negative
    assert torch.equal(actual['proj'],positive['proj'])
