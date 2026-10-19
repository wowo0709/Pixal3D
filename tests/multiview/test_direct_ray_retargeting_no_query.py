import pytest
import torch

from pixal3d.modules.direct_ray_retargeting import DirectRayRetargetingConditioner
from pixal3d.modules.direct_ray_retargeting_no_query import DirectRayRetargetingNoQueryConditioner
from pixal3d.modules.ray_feature_attention import RayFeatureAttentionConditioner


def args(width):
    return dict(arm='c1',key_dim=1024,value_dim=width,query_dim=1024,topk=64,
        implementation='r9-native-v5',candidate_aggregation='top1',
        projection_mode='nearest',score_temperature=.05,feature_chunk_size=1)


def inputs(width):
    source = torch.randn(1,3,1024,2048)
    target = torch.randn(1,2,2048)
    basic = dict(source_values=torch.randn(1,3,1024,width),
        source_valid=torch.ones(1,3,1024,dtype=torch.bool),target_query=torch.randn(1,2,1024),
        semantic_indices=torch.arange(64).reshape(1,1,64).expand(1,2,-1),
        matching_logits=torch.randn(1,2,64),candidate_valid=torch.ones(1,2,64,dtype=torch.bool),
        target_valid=torch.ones(1,2,dtype=torch.bool),target_keys=target[...,:1024],source_keys=source[...,:1024])
    return basic,dict(basic,target_context=target,source_context=source)


@pytest.mark.parametrize('width',[1024,2048])
def test_absent_channels_match_full_direct_when_only_added_projection_is_zero(width):
    torch.manual_seed(121)
    absent = DirectRayRetargetingNoQueryConditioner(**args(width))
    full = DirectRayRetargetingConditioner(**dict(args(width),arm='c2',deformation_dim=2048))
    # Exercise learned feature delivery and a nonzero existing R. An all-zero
    # output comparison would not verify that both direct paths are the same.
    torch.nn.init.normal_(full.ray_readout.output.weight,std=.01)
    torch.nn.init.normal_(full.retargeter.net[-1].weight,std=.01)
    full.retargeter.gate_logit.data.fill_(.2)
    common = {name:value for name,value in full.state_dict().items()
              if name!='retargeter.deformation_projection.weight'}
    absent.load_state_dict(common,strict=True)
    assert not hasattr(absent.retargeter,'deformation_projection')
    assert not any('deformation_projection' in name for name in absent.state_dict())
    assert absent.retargeter.input_dim==width+1024
    assert full.retargeter.input_dim==width+1024+2048
    basic,physical = inputs(width)
    with torch.no_grad():
        left,right = absent(**basic),full(**physical)
    assert torch.equal(left.evidence,right.evidence)
    assert torch.equal(left.retarget_residual,right.retarget_residual)
    assert torch.equal(left.candidate_weights,right.candidate_weights)
    torch.nn.init.normal_(full.retargeter.deformation_projection.weight,std=.01)
    with torch.no_grad():
        changed = full(**physical)
    assert not torch.allclose(left.retarget_residual,changed.retarget_residual)
    # Supplying fake-zero RAW fields is rejected, not used as an absence arm.
    with pytest.raises(ValueError):
        absent(**physical)
    assert not absent.ray_readout._forward_hooks


def test_initial_identity_and_checkpoint_backward_without_semantic_gate():
    torch.manual_seed(122)
    baseline = RayFeatureAttentionConditioner(**args(1024))
    direct = DirectRayRetargetingNoQueryConditioner(**args(1024))
    direct.load_state_dict(baseline.state_dict(),strict=True)
    pack,_ = inputs(1024)
    before,after = baseline(**pack),direct(**pack)
    assert torch.equal(before.evidence,after.evidence)
    assert float(direct.fusion_gate)==0
    after.evidence.square().sum().backward()
    assert direct.ray_readout.output.weight.grad.abs().sum()>0
    assert not direct.ray_readout._forward_hooks
    assert direct.last_feature_attention_stats['deformation_query_module_present'] is False


def test_no_active_rays_retains_finite_zero_outputs_and_releases_hook():
    direct = DirectRayRetargetingNoQueryConditioner(**args(1024))
    pack,_ = inputs(1024)
    pack['target_valid'].zero_()
    output = direct(**pack)
    assert torch.equal(output.evidence,torch.zeros_like(output.evidence))
    assert torch.equal(output.retarget_residual,torch.zeros_like(output.retarget_residual))
    assert not direct.ray_readout._forward_hooks
