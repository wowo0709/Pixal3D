from copy import deepcopy

import pytest
import torch

from pixal3d.modules.native_prior_source_ray import source_ray_delta
from pixal3d.modules.r_mediated_source_contrast import (
    APPEARANCE_MODES, RMediatedSourceContrastConditioner,
)
from test_bounded_source_contrast_member_feature import pair
from test_member_posterior_source_ray import activate


def model_pair(mode):
    _, parent, data = pair()
    torch.manual_seed(1317)
    new = RMediatedSourceContrastConditioner(appearance_mode=mode, residual_logit_bound=1.,
        arm='c1', implementation='r9-native-v5', key_dim=1024, value_dim=1024,
        query_dim=1024, hidden_dim=256, topk=64, score_temperature=.5,
        projection_mode='native-bilinear', ray_mode='semantic-top1', ray_chunk_size=4,
        candidate_aggregation='top1')
    new.load_utility(deepcopy(parent.ray_readout.score_selector.state_dict()))
    return parent, new, data


@pytest.mark.parametrize('mode', APPEARANCE_MODES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_native_cold_function_same_nineteen_weights_and_real_prior(mode, dtype):
    parent, new, data = model_pair(mode)
    parameters = dict(parent.named_parameters())
    assert sum(p.requires_grad for p in new.parameters()) == 19
    assert all(torch.equal(v, parameters[k]) for k, v in new.named_parameters())
    with torch.no_grad(), torch.autocast('cpu', dtype=dtype, enabled=dtype != torch.float32):
        a, b = parent(**data), new(**data)
    assert torch.equal(a.source_numerator, b.source_numerator)
    assert torch.equal(a.semantic_weight, b.semantic_weight)
    assert torch.equal(a.attention.candidate_indices, b.attention.candidate_indices)
    assert torch.equal(parent.last_feature_probabilities[1], new.last_feature_probabilities[1])
    assert not source_ray_delta(b.source_numerator, b.semantic_weight, data['source_values']).any()
    assert not new.last_direct_feature_correction.any()
    assert new.retargeter._source_context is None
    assert not any('deformation_projection' in k for k in new.state_dict())


@pytest.mark.parametrize('mode', APPEARANCE_MODES)
def test_head_cannot_bypass_zero_R_even_when_it_produces_nonzero_features(mode):
    _, model, data = model_pair(mode)
    seen = []
    with torch.no_grad():
        model.ray_readout.output.weight.normal_(std=.03)
        model.fusion_gate.zero_()
    handle = model.ray_readout.register_forward_hook(lambda m, args, out: seen.append(out[0]))
    try:
        out = model(**data)
    finally:
        handle.remove()
    assert torch.cat(seen).abs().sum() > 0
    assert not out.attention.retarget_residual.any()
    assert not source_ray_delta(out.source_numerator, out.semantic_weight, data['source_values']).any()


@pytest.mark.parametrize('mode', APPEARANCE_MODES)
def test_nonreentrant_backward_reaches_all_nineteen_parameters_through_R(mode):
    _, model, data = model_pair(mode)
    activate(model)
    with torch.no_grad():
        model.ray_readout.output.weight.normal_(std=.01)
    actual_R = []
    h = model.retargeter.register_forward_hook(lambda m, args, out: actual_R.append(out))
    try:
        out = model(**data)
    finally:
        h.remove()
    selected = model.last_source_reference_rows
    assert torch.equal(out.attention.retarget_residual.reshape(-1, 1024)[selected], actual_R[0])
    (out.source_numerator*torch.randn_like(out.source_numerator)).sum().backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    assert all(p.grad is None for p in model.q2c.parameters())
    assert model.retargeter._source_context is None


def test_raw_appearance_uses_actual_posterior_values_without_changing_Top1_carrier():
    _, a, data = model_pair('top1-plus-feature')
    _, b, _ = model_pair('raw-topk-plus-feature')
    with torch.no_grad():
        x, y = a(**data), b(**data)
    selected, probability = b.last_feature_probabilities
    indices = y.attention.candidate_indices.reshape(-1, 64)[selected]
    raw = data['source_values'].flatten(1, 2)[(selected//3072)[:, None], indices]
    expected = (probability.mean(1)[..., None]*raw).sum(1)
    assert torch.equal(b.last_R_input_appearance, expected)
    assert torch.equal(x.attention.base_evidence, y.attention.base_evidence)
    assert torch.equal(x.use_probability, y.use_probability)
    assert not torch.equal(a.last_R_input_appearance, b.last_R_input_appearance)


def test_modes_reject_each_other_and_parent_checkpoints_but_roundtrip_exactly():
    parent, a, data = model_pair('top1-plus-feature')
    _, b, _ = model_pair('raw-topk-plus-feature')
    with pytest.raises(RuntimeError):
        a.load_state_dict(parent.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        b.load_state_dict(a.state_dict(), strict=True)
    activate(a)
    with torch.no_grad():
        a.ray_readout.output.weight.normal_(std=.02)
    _, restored, _ = model_pair('top1-plus-feature')
    restored.load_state_dict(a.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.equal(a(**data).source_numerator, restored(**data).source_numerator)


def test_wrong_retrieved_values_changes_R_input_with_keys_scores_probability_use_fixed():
    _, model, data = model_pair('raw-topk-plus-feature')
    activate(model)
    with torch.no_grad():
        model.ray_readout.output.weight.normal_(std=.02)
    model.eval()
    with torch.no_grad():
        normal = model(**data)
        appearance = model.last_R_input_appearance.clone()
        probability = model.last_feature_probabilities[1].clone()
        wrong = model(**data, control='wrong-retrieved-values-fixed-use')
    assert torch.equal(normal.attention.candidate_indices, wrong.attention.candidate_indices)
    assert torch.equal(normal.attention.matching_logits, wrong.attention.matching_logits)
    assert torch.equal(normal.attention.base_evidence, wrong.attention.base_evidence)
    assert torch.equal(normal.use_probability, wrong.use_probability)
    assert torch.equal(probability, model.last_feature_probabilities[1])
    assert not torch.equal(appearance, model.last_R_input_appearance)
    assert not torch.equal(normal.attention.retarget_residual, wrong.attention.retarget_residual)


@pytest.mark.parametrize('mode', APPEARANCE_MODES)
def test_cold_head_gradient_is_delayed_but_opens_after_real_R_updates(mode):
    _, model, data = model_pair(mode)
    parameters = {n:p for n,p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(parameters.values(), lr=1e-4, betas=(.9,.95))
    target = .02*torch.randn_like(data['source_values'])
    gradients = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        out = model(**data)
        delta = source_ray_delta(out.source_numerator, out.semantic_weight, data['source_values'])
        (delta-target).square().mean().backward()
        gradients.append({n:float(p.grad.abs().sum()) for n,p in parameters.items()})
        optimizer.step()
    assert gradients[0]['ray_readout.output.weight'] == 0
    assert all(value > 0 for value in gradients[2].values())


@pytest.mark.parametrize('mode', APPEARANCE_MODES)
def test_missing_matching_preserves_native_with_finite_DDP_edges(mode):
    _, model, data = model_pair(mode)
    data['observed_query'].zero_()
    out = model(**data)
    delta = source_ray_delta(out.source_numerator, out.semantic_weight, data['source_values'])
    assert not delta.any()
    delta.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and not p.grad.any()
               for p in model.parameters() if p.requires_grad)
