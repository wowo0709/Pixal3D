import pytest
import torch

from pixal3d.modules.anchor_ray_composition import AnchorRayCompositionConditioner
from pixal3d.modules.deform_conditioner import DeformationAwareConditioner


def pair():
    torch.manual_seed(318)
    config=dict(arm='c1', key_dim=1024, value_dim=1024, query_dim=1024,
        implementation='r9-native-v5', candidate_aggregation='top1', score_temperature=.05)
    reference=DeformationAwareConditioner(**config).eval()
    with torch.no_grad():
        reference.retargeter.net[-1].weight.normal_(0, .01)
        reference.retargeter.net[-1].bias.normal_(0, .01)
        reference.fusion_gate.fill_(.04)
    model=AnchorRayCompositionConditioner(**config, composer_chunk_size=2).eval()
    model.load_reference(reference.state_dict())
    data=dict(source_values=torch.randn(1,3,1024,1024), source_valid=torch.ones(1,3,1024,dtype=torch.bool),
        source_keys=torch.randn(1,3,1024,1024), target_query=torch.randn(1,3,1024),
        target_keys=torch.randn(1,3,1024), semantic_indices=torch.arange(64).expand(1,3,64),
        matching_logits=torch.randn(1,3,64), candidate_valid=torch.ones(1,3,64,dtype=torch.bool),
        target_valid=torch.tensor([[True,False,True]]))
    return reference,model,data


def test_zero_gate_preserves_every_original_field_with_nonzero_R():
    reference,model,data=pair()
    with torch.no_grad():left,right=reference(**data),model(**data)
    for key in vars(left):
        a,b=getattr(left,key),getattr(right,key)
        assert (a is None and b is None) or torch.equal(a,b),key
    assert len([p for p in model.parameters() if p.requires_grad])==6
    assert all(not p.requires_grad for name,p in model.named_parameters()
        if name!='ray_mix' and not name.startswith('ray_composer.'))


def test_projected_gate_boundary_has_gradient_and_cannot_stick_below_zero():
    _,model,_=pair();optimizer=torch.optim.SGD([model.ray_mix],lr=1.)
    handle=model.install_gate_projection(optimizer)
    model.ray_mix.grad=torch.ones_like(model.ray_mix);optimizer.step()
    assert model.ray_mix.item()==0
    optimizer.zero_grad();(-model.ray_mix).backward();optimizer.step()
    assert model.ray_mix.item()==1
    handle.remove()


def test_positive_gate_is_a_normalized_mixture_and_trains_only_composer():
    _,model,data=pair();model.ray_mix.data.fill_(.2)
    result=model(**data);result.evidence.square().mean().backward()
    assert torch.all(result.candidate_weights>=0)
    assert torch.allclose(result.candidate_weights.sum(-1),result.query_valid.float(),atol=1e-6)
    assert model.ray_mix.grad is not None and model.ray_mix.grad.abs()>0
    assert model.ray_composer.content_output.weight.grad.abs().sum()>0
    assert all(p.grad is None for name,p in model.named_parameters()
        if name!='ray_mix' and not name.startswith('ray_composer.'))


def test_no_source_support_is_inert_and_keeps_all_new_parameter_edges():
    _,model,data=pair();data['source_valid'].zero_();out=model(**data)
    assert not out.query_valid.any() and not out.evidence.any() and not out.candidate_weights.any()
    out.evidence.sum().backward()
    assert all(p.grad is not None and not p.grad.any() for p in model.parameters() if p.requires_grad)


def test_unprojected_gate_and_teacher_arguments_are_rejected():
    _,model,data=pair();model.ray_mix.data.fill_(-.1)
    with pytest.raises(ValueError,match='project'):model(**data)
    model.project_mix_()
    with pytest.raises(ValueError,match='declared'):model(**data,positive_indices=torch.ones(1))
