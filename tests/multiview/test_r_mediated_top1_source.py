from copy import deepcopy

import pytest
import torch

from pixal3d.modules.native_prior_source_ray import source_ray_delta
from pixal3d.modules.r_mediated_top1_source import RMediatedTop1SourceConditioner
from pixal3d.modules.r_mediated_source_contrast import RMediatedSourceContrastConditioner
from test_member_posterior_source_ray import fixture, activate


def models():
    _, data = fixture('semantic-top1')
    config = dict(arm='c1', implementation='r9-native-v5', key_dim=1024, value_dim=1024,
        query_dim=1024, hidden_dim=256, topk=64, score_temperature=.5,
        projection_mode='native-bilinear', ray_mode='semantic-top1', ray_chunk_size=4,
        candidate_aggregation='top1')
    torch.manual_seed(1317)
    top1 = RMediatedTop1SourceConditioner(**config)
    torch.manual_seed(1317)
    feature = RMediatedSourceContrastConditioner(appearance_mode='top1-plus-feature',
        residual_logit_bound=1., **config)
    return top1, feature, data


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_source_core_initial_values_and_native_cold_function_match_full_R19(dtype):
    top1, feature, data = models()
    full = dict(feature.named_parameters())
    own = {n: p for n, p in top1.named_parameters() if p.requires_grad}
    assert len(own) == 10
    assert all(torch.equal(v, full[n]) for n, v in own.items())
    assert not hasattr(top1, 'ray_readout') and top1.task_attention is None
    assert not any('ray_readout' in n or 'deformation_projection' in n for n in top1.state_dict())
    with torch.no_grad(), torch.autocast('cpu', dtype=dtype, enabled=dtype != torch.float32):
        a, b = top1(**data), feature(**data)
    assert torch.equal(a.source_numerator, b.source_numerator)
    assert torch.equal(a.semantic_weight, b.semantic_weight)
    assert torch.equal(a.attention.base_evidence, b.attention.base_evidence)
    assert not source_ray_delta(a.source_numerator, a.semantic_weight, data['source_values']).any()
    selected = a.attention.query_valid
    assert (a.attention.candidate_weights[selected].count_nonzero(-1) == 1).all()


def test_nonzero_R_still_consumes_the_actual_single_winning_feature_and_real_source_reference():
    top1, feature, data = models()
    activate(feature)
    full = dict(feature.named_parameters())
    with torch.no_grad():
        for n, p in top1.named_parameters():
            if p.requires_grad: p.copy_(full[n])
    source_reference = []
    h = top1.retargeter.register_forward_pre_hook(lambda m, a: source_reference.append(m._source_context.clone()))
    try:
        a, b = top1(**data), feature(**data)
    finally:
        h.remove()
    selected = top1.last_source_reference_rows
    assert torch.equal(source_reference[0], data['source_values'].reshape(-1, 1024)[selected])
    winner = a.attention.candidate_indices.reshape(-1, 64)[selected].gather(
        1, a.attention.matching_logits.reshape(-1, 64)[selected].argmax(-1)[:, None])[:, 0]
    expected = data['source_values'].flatten(1, 2)[selected // 3072, winner]
    assert torch.equal(top1.last_R_input_appearance, expected)
    assert torch.equal(a.source_numerator, b.source_numerator)
    assert torch.equal(a.attention.retarget_residual, b.attention.retarget_residual)
    assert top1.retargeter._source_context is None


def test_R_only_top1_backward_reaches_all_ten_weights_and_matcher_stays_frozen():
    top1, _, data = models()
    activate(top1)
    out = top1(**data)
    delta = source_ray_delta(out.source_numerator, out.semantic_weight, data['source_values'])
    (delta * torch.randn_like(delta)).sum().backward()
    for name, p in top1.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
        else:
            assert p.grad is None
    top1.train()
    assert not top1.q2c.training


def test_missing_correspondence_keeps_native_prior_and_all_ten_finite_zero_gradient_edges():
    top1, _, data = models()
    data['observed_query'].zero_()
    out = top1(**data)
    delta = source_ray_delta(out.source_numerator, out.semantic_weight, data['source_values'])
    assert not delta.any()
    delta.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and not p.grad.any()
        for p in top1.parameters() if p.requires_grad)


def test_exact_own_schema_roundtrip_rejects_R19_and_legacy_top1_schemas():
    top1, feature, data = models()
    with pytest.raises(RuntimeError): top1.load_state_dict(feature.state_dict(), strict=True)
    top1, _, data = models()
    with pytest.raises(RuntimeError): feature.load_state_dict(top1.state_dict(), strict=True)
    activate(top1)
    restored, _, _ = models()
    restored.load_state_dict(deepcopy(top1.state_dict()), strict=True)
    with torch.no_grad(): assert torch.equal(top1(**data).source_numerator, restored(**data).source_numerator)
