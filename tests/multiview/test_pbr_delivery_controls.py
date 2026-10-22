from copy import deepcopy

import pytest
import torch

from experiments.deformnext_retargeting.pbr_delivery_controls import delivery_control
from pixal3d.modules.bounded_member_feature_source_ray import masked_probabilities
from pixal3d.modules.direct_ray_retargeting import DirectRayRetargetingConditioner


def fixture():
    torch.manual_seed(718)
    model=DirectRayRetargetingConditioner(arm='c2',deformation_dim=2048,key_dim=1024,value_dim=2048,
        query_dim=1024,topk=64,implementation='r9-native-v5',candidate_aggregation='top1',projection_mode='nearest',score_temperature=.05).eval()
    with torch.no_grad():
        model.ray_readout.output.weight.normal_(std=.02)
        model.retargeter.net[-1].weight.normal_(std=.01)
        model.retargeter.deformation_projection.weight.normal_(std=.01)
    source=torch.randn(1,3,1024,2048);target=torch.randn(1,3,2048)
    data=dict(source_values=torch.randn(1,3,1024,2048),source_valid=torch.ones(1,3,1024,dtype=torch.bool),
        target_query=torch.randn(1,3,1024),semantic_indices=torch.arange(64).reshape(1,1,64).expand(1,3,-1),
        matching_logits=.03*torch.randn(1,3,64),candidate_valid=torch.ones(1,3,64,dtype=torch.bool),
        target_valid=torch.tensor([[True,True,False]]),target_keys=target[...,:1024],source_keys=source[...,:1024],target_context=target,source_context=source)
    return model,data


def test_normal_is_bit_exact_and_all_hooks_restore_after_success():
    model,data=fixture();before=deepcopy(model.state_dict());events=[]
    with torch.no_grad():
        expected=model(**data)
        with delivery_control(model,'normal',events):actual=model(**data)
    assert torch.equal(expected.evidence,actual.evidence)and torch.equal(expected.retarget_residual,actual.retarget_residual)
    assert len(events)==1 and events[0]['physical_R_channels']==5120
    assert not model._forward_hooks and not model._forward_pre_hooks
    assert all(torch.equal(v,model.state_dict()[k])for k,v in before.items())


def test_prior_replacement_recomputes_actual_feature_and_returned_RAW_probabilities():
    model,data=fixture();seen=[];check=[]
    def observe(head,args,output):
        native,context,keys,values,valid,prior=args;feature,probability=output
        expected=masked_probabilities(prior[:,None].expand(-1,4,-1),valid)
        vv=head.value(torch.cat([torch.nn.functional.layer_norm(x,(1024,))for x in values.split(1024,-1)],-1))
        vv=vv.reshape(len(native),64,4,64).transpose(1,2)
        reference=head.output((expected[...,None]*vv.float()).sum(-2).reshape(len(native),256)).float()
        check.append(torch.equal(probability,expected)and torch.equal(feature,reference))
        seen.append(probability.clone())
    with torch.no_grad(),delivery_control(model,'matching-prior',[]):
        h=model.ray_readout.register_forward_hook(observe)
        try:out=model(**data)
        finally:h.remove()
    assert check and all(check)and seen[0].shape==(2,4,64)
    assert torch.isfinite(out.evidence).all()and not out.evidence[0,2].any()
    assert not model.ray_readout._forward_hooks and not model.ray_readout.value._forward_hooks


@pytest.mark.parametrize('mode',['no-direct-delivery','matching-prior-no-direct-delivery'])
def test_R_delivery_uses_actual_R_output_while_physical_DQ_is_present(mode):
    model,data=fixture();R=[];events=[]
    with torch.no_grad(),delivery_control(model,mode,events):
        h=model.retargeter.register_forward_hook(lambda m,a,o:R.append(o.clone()))
        try:out=model(**data)
        finally:h.remove()
    expected=R[0]*out.query_valid[...,None]
    assert torch.equal(out.retarget_residual,expected)
    # Check the actual matching-selected carrier rather than assume address0.
    address=out.candidate_indices.gather(2,out.candidate_weights.argmax(-1,keepdim=True)).squeeze(2)
    top1=data['source_values'].flatten(1,2).gather(1,address[...,None].expand(-1,-1,2048))*out.query_valid[...,None]
    assert torch.equal(out.evidence,top1+expected)
    assert model.retargeter.input_dim==5120 and model.retargeter.deformation_projection.in_features==2048
    assert events[0]['removed_direct_correction_RMS']>0


def test_hooks_restore_after_exception_and_training_is_rejected():
    model,data=fixture()
    with torch.no_grad():
        with pytest.raises(RuntimeError):
            with delivery_control(model,'matching-prior-no-direct-delivery',[]):raise RuntimeError('probe')
    assert not model._forward_hooks and not model._forward_pre_hooks and not model.ray_readout._forward_hooks
    with pytest.raises(ValueError):
        with delivery_control(model,'matching-prior',[]):pass


@pytest.mark.parametrize('mode', ['matching-prior', 'no-direct-delivery',
                                  'matching-prior-no-direct-delivery'])
def test_interventions_leave_weights_and_subsequent_native_forward_exact(mode):
    model, data = fixture()
    before = deepcopy(model.state_dict())
    with torch.no_grad():
        expected = model(**data)
        with delivery_control(model, mode, []):
            model(**data)
        actual = model(**data)
    assert all(torch.equal(value, model.state_dict()[name])
               for name, value in before.items())
    assert torch.equal(actual.evidence, expected.evidence)
    assert torch.equal(actual.retarget_residual, expected.retarget_residual)
    assert all(not module._forward_hooks and not module._forward_pre_hooks
               for module in model.modules())
