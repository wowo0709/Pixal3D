"""Actual tiny CPU module control, not native SS training or a quality result."""
import pytest
import torch

from experiments.deformnext_retargeting.member_feature_value_control import feature_value_control
from test_source_contrast_member_feature import fixture
from test_member_posterior_source_ray import activate


def prepared():
    _,model,data=fixture();activate(model)
    torch.nn.init.normal_(model.ray_readout.output.weight,std=.1)
    model.eval()
    return model,data


def test_same_forward_V_control_preserves_source_relative_context_weights_and_state():
    model,data=prepared();state={k:v.clone()for k,v in model.state_dict().items()};events=[]
    with torch.no_grad():
        normal=model(**data)
        with feature_value_control(model,events):wrong=model(**data)
        restored=model(**data)
    assert events and all(e['projected_V_only']and e['original_Top1_feature_carrier_exact']
        and e['source_use_probability_exact_same_forward']and e['original_matching_logits_and_attention_weights_exact']for e in events)
    assert not torch.equal(normal.source_numerator,wrong.source_numerator)
    assert torch.equal(normal.source_numerator,restored.source_numerator)
    assert torch.equal(normal.use_probability,wrong.use_probability)
    assert torch.equal(normal.attention.base_evidence,wrong.attention.base_evidence)
    assert model.retargeter._source_context is None
    assert all(torch.equal(v,state[k])for k,v in model.state_dict().items())
    assert 'forward'not in model.__dict__ and not model.ray_readout.value._forward_hooks


def test_failed_head_restores_source_context_and_all_temporary_control_hooks(monkeypatch):
    model,data=prepared();events=[]
    def fail(*args):raise RuntimeError('unit head failure')
    monkeypatch.setattr(model.ray_readout,'forward',fail)
    with torch.no_grad(),pytest.raises(RuntimeError,match='unit head failure'):
        with feature_value_control(model,events):model(**data)
    assert model.retargeter._source_context is None and 'forward'not in model.__dict__
    for layer in [model.ray_readout,model.ray_readout.query,model.ray_readout.key,
                  model.ray_readout.score_selector,model.ray_readout.value,model.evidence_use]:
        assert not layer._forward_hooks and not layer._forward_pre_hooks
    assert not events
