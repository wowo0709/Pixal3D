import pytest
import torch

from pixal3d.modules.member_feature_source_ray import MemberFeatureSourceRayConditioner
from pixal3d.modules.source_contrast_member_feature import (
    SourceContrastMemberFeatureConditioner, SourceContrastRetargeter,
)
from pixal3d.modules.native_prior_source_ray import source_ray_delta
from test_source_referenced_member_feature import pair


def fixture():
    old, _, data = pair()
    kwargs = dict(arm='c1', implementation='r9-native-v5', key_dim=1024, value_dim=1024,
        query_dim=1024, hidden_dim=256, topk=64, score_temperature=.5,
        projection_mode='native-bilinear', ray_mode='semantic-top1', ray_chunk_size=4,
        candidate_aggregation='top1')
    torch.manual_seed(1317); new = SourceContrastMemberFeatureConditioner(**kwargs)
    return old, new, data


def test_cold_function_and_nineteen_parameter_initialization_preserved():
    old, new, data = fixture()
    assert sum(p.requires_grad for p in new.parameters()) == 19
    assert set(new.state_dict())-set(old.state_dict()) == {'retargeter.source_contrast_schema'}
    assert all(torch.equal(value,new.state_dict()[key]) for key,value in old.state_dict().items())
    a, b = old(**data), new(**data)
    assert torch.equal(a.source_numerator,b.source_numerator)
    assert torch.equal(a.attention.candidate_indices,b.attention.candidate_indices)
    assert torch.equal(a.use_probability,b.use_probability)
    assert not source_ray_delta(b.source_numerator,b.semantic_weight,data['source_values']).any()


def test_real_source_changes_the_first_output_gradient_without_an_extra_parameter():
    _, new, _ = fixture(); r = new.retargeter
    x, source = torch.randn(2,2048), torch.randn(2,1024)
    with r.native_source_context(source): a = r(x,.25)
    a.sum().backward(); first = r.net[-1].weight.grad.clone(); r.zero_grad(set_to_none=True)
    with r.native_source_context(source+1): b = r(x,.25)
    b.sum().backward()
    assert not a.any() and not b.any()  # Identical cold outputs.
    assert not torch.equal(first,r.net[-1].weight.grad)
    assert r._source_context is None


def test_retargeter_uses_exact_matched_minus_native_and_preserves_target_context():
    _, new, _ = fixture(); r = new.retargeter
    x, source = torch.randn(2,2048), torch.randn(2,1024)
    torch.nn.init.normal_(r.net[-1].weight,std=.03)
    expected=.25*torch.tanh(r.gate_logit)*torch.tanh(r.net(torch.cat((x[:,:1024]-source,x[:,1024:]),-1)))
    with r.native_source_context(source): actual=r(x,.25)
    assert torch.equal(actual,expected)


def test_checkpoint_marker_rejects_accidental_parent_loading_and_warm_migration():
    old,new,_=fixture()
    with pytest.raises(RuntimeError,match='source_contrast_schema'):
        old.load_state_dict(new.state_dict(),strict=True)
    torch.nn.init.normal_(old.retargeter.net[-1].weight,std=.03)
    with pytest.raises(ValueError,match='cold zero-output'):
        SourceContrastRetargeter(old.retargeter)


def test_empty_support_remains_native_with_all_nineteen_gradients_connected():
    _,new,data=fixture();data['observed_query'].zero_();out=new(**data)
    delta=source_ray_delta(out.source_numerator,out.semantic_weight,data['source_values'])
    assert not delta.any();delta.sum().backward()
    assert all(p.grad is not None and not p.grad.any() for p in new.parameters() if p.requires_grad)
