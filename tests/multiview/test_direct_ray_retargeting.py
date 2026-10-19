import torch

from pixal3d.modules.direct_ray_retargeting import DirectRayRetargetingConditioner
from pixal3d.modules.ray_deformation_query import RayDeformationQueryConditioner


def test_direct_delivery_preserves_initial_function_and_true_checkpointed_gradients():
    torch.manual_seed(91)
    args = dict(arm='c2', deformation_dim=2048, key_dim=1024, value_dim=1024,
                query_dim=1024, topk=64, implementation='r9-native-v5',
                candidate_aggregation='top1', projection_mode='nearest', score_temperature=.05)
    reference = RayDeformationQueryConditioner(**args)
    direct = DirectRayRetargetingConditioner(**args)
    torch.nn.init.normal_(reference.retargeter.net[-1].weight, std=.01)
    reference.retargeter.gate_logit.data.fill_(.2)
    reference.fusion_gate.data.fill_(.015)
    direct.load_state_dict(reference.state_dict())
    source = torch.randn(1,3,1024,2048)
    target = torch.randn(1,2,2048)
    pack = dict(source_values=torch.randn(1,3,1024,1024),
        source_valid=torch.ones(1,3,1024,dtype=torch.bool), target_query=torch.randn(1,2,1024),
        semantic_indices=torch.arange(64).reshape(1,1,64).expand(1,2,-1),
        matching_logits=torch.randn(1,2,64), candidate_valid=torch.ones(1,2,64,dtype=torch.bool),
        target_valid=torch.ones(1,2,dtype=torch.bool), target_keys=target[...,:1024],
        source_keys=source[...,:1024], target_context=target, source_context=source)
    before = reference(**pack)
    after = direct(**pack)
    assert torch.equal(before.evidence, after.evidence)
    assert torch.equal(before.retarget_residual, after.retarget_residual)
    # Backward recomputes the checkpointed readout after the capture hook has
    # gone. This must still propagate genuine gradients to the new projection.
    after.evidence.square().sum().backward()
    assert direct.ray_readout.output.weight.grad.abs().sum() > 0
    assert direct.retargeter.deformation_projection.weight.grad.abs().sum() > 0
    assert not direct.ray_readout._forward_hooks
    torch.nn.init.normal_(direct.ray_readout.output.weight, std=1.)
    with torch.no_grad():
        result = direct(**pack)
    assert torch.isfinite(result.evidence).all()
    assert direct.last_feature_attention_stats['direct_feature_bound'] == .25
    assert direct.last_feature_attention_stats['direct_retargeted_feature_RMS'] <= .25
