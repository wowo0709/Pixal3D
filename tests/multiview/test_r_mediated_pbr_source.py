from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from pixal3d.modules.r_mediated_pbr_source import (
    MODES, RMediatedPBRSourceConditioner, packed_member_posteriors,
)


def fixture(mode):
    torch.manual_seed(8137)
    model = RMediatedPBRSourceConditioner(appearance_mode=mode, score_temperature=.5,
        ray_chunk_size=4, residual_logit_bound=None if mode == 'top1-only' else 1.)
    torch.manual_seed(1261)
    valid = torch.zeros(2, 3, 1024, dtype=torch.bool)
    valid[:, 0, :128] = True
    data = dict(native_query_rows=None if mode == 'top1-only' else torch.randn(5, 1024),
        target_query=torch.randn(2, 1024, 1024), frame_target_keys=torch.randn(2, 1024, 1024),
        target_patch_ids=torch.tensor([7, 9, 7, 11, 12]),
        target_valid=torch.ones(2, 1024, dtype=torch.bool),
        ray_ids=torch.tensor([[0, 4, -1], [1, -1, 5], [0, 4, -1], [1, 3, -1], [0, 3, 2]]),
        batch_ids=torch.tensor([1, 0, 1, 0, 0], dtype=torch.int32),
        source_keys=torch.randn(2, 3, 1024, 1024),
        source_values=torch.randn(2, 3, 1024, 2048), source_valid=valid)
    return model, data


def delta(output, data):
    return output.source_numerator - output.semantic_weight * data['source_values'].float()


def activate(model):
    with torch.no_grad():
        model.fusion_gate.fill_(.2)
        model.retargeter.net[-1].weight.normal_(std=.01)
        model.evidence_use.network[-1].weight.normal_(std=.01)
        if hasattr(model, 'ray_readout'):
            model.ray_readout.output.weight.normal_(std=.01)
            model.ray_readout.score_selector.content_output.weight.normal_(std=.01)


@pytest.mark.parametrize('mode', list(MODES))
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_PBR_cold_function_zero_actual_DINO_NAF_values_and_physical_parameter_schema(mode, dtype):
    model, data = fixture(mode)
    with torch.no_grad(), torch.autocast('cpu', dtype=dtype, enabled=dtype != torch.float32):
        out = model(**data)
    assert out.source_numerator.shape == (2, 3, 1024, 2048)
    assert not delta(out, data).any() and not out.semantic_weight.any()
    assert model.retargeter.input_dim == 3072 and model.retargeter.source_width == 2048
    assert sum(p.requires_grad for p in model.parameters()) == (10 if mode == 'top1-only' else 19)
    assert not any('deformation' in name for name in model.state_dict())
    assert hasattr(model, 'ray_readout') is (mode != 'top1-only')
    selected = model.last_source_reference_rows
    weights = out.attention.candidate_weights.reshape(-1, 64)[selected]
    assert (weights.count_nonzero(-1) == 1).all()
    winner = out.attention.candidate_indices.reshape(-1, 64)[selected].gather(1, weights.argmax(-1)[:, None])[:, 0]
    expected = data['source_values'].flatten(1, 2)[selected // 3072, winner]
    assert torch.equal(out.attention.base_evidence.reshape(-1, 2048)[selected], expected)
    assert model.task_attention is None


def test_all_PBR_modes_share_actual_ten_R_core_initial_values_and_strict_own_schema():
    modules = [fixture(mode)[0] for mode in MODES]
    core = {n: p for n, p in modules[0].named_parameters() if p.requires_grad}
    assert len(core) == 10
    for model in modules:
        parameters = dict(model.named_parameters())
        assert all(torch.equal(value, parameters[name]) for name, value in core.items())
        with pytest.raises(ValueError):
            if model.appearance_mode == 'top1-only':
                model.load_utility({})
            else:
                model.load_utility(dict(list(model.ray_readout.score_selector.state_dict().items())[:-1]))
    for a in modules:
        for b in modules:
            if a is not b:
                with pytest.raises(RuntimeError):
                    a.load_state_dict(b.state_dict(), strict=True)


@pytest.mark.parametrize('mode', list(MODES))
def test_PBR_R_sees_real_own_source_and_delivers_both_NAF_and_DINO_with_active_gradients(mode):
    model, data = fixture(mode)
    activate(model)
    references = []
    hook = model.retargeter.register_forward_pre_hook(lambda m, a: references.append(m._source_context.clone()))
    try:
        out = model(**data)
    finally:
        hook.remove()
    selected = model.last_source_reference_rows
    assert torch.equal(references[0], data['source_values'].reshape(-1, 2048)[selected])
    assert model.retargeter._source_context is None
    residual = out.attention.retarget_residual.reshape(-1, 2048)[selected]
    assert torch.equal(residual, model.last_R_only_output)
    assert residual[:, :1024].abs().sum() > 0 and residual[:, 1024:].abs().sum() > 0
    (delta(out, data) * torch.randn_like(out.source_numerator)).sum().backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
        else:
            assert parameter.grad is None
    model.train()
    assert not model.q2c.training


@pytest.mark.parametrize('mode', list(MODES))
def test_PBR_absent_matching_abstains_even_after_learning_and_keeps_all_DDP_edges(mode):
    model, data = fixture(mode)
    activate(model)
    data['target_valid'].zero_()
    out = model(**data)
    assert not delta(out, data).any() and not out.use_probability.any()
    delta(out, data).sum().backward()
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and not parameter.grad.any()


@pytest.mark.parametrize('mode', ['top1-plus-feature', 'raw-topk-plus-feature'])
def test_nonzero_PBR_feature_head_has_no_direct_consumer_bypass_and_initial_prior_matches(mode):
    model, data = fixture(mode)
    utility = deepcopy(model.ray_readout.score_selector.state_dict())
    model.load_utility(utility)
    with torch.no_grad():
        model.ray_readout.output.weight.normal_(std=.02)
        out = model(**data)
    assert model.ray_readout.first_forward_prior_exact
    assert not delta(out, data).any()
    assert not model.last_R_only_output.any()
    assert model.last_R_input_appearance.shape[-1] == 2048


def test_packed_member_construction_keeps_real_unequal_batch_addresses_without_phantom_votes():
    model, data = fixture('top1-only')
    captured = {}
    def posterior(matcher, **kwargs):
        captured.update(kwargs)
        return 'captured'
    with patch('pixal3d.modules.r_mediated_pbr_source.member_ray_posteriors', side_effect=posterior):
        value = packed_member_posteriors(model.q2c, target_keys=data['frame_target_keys'],
            source_keys=data['source_keys'], source_valid=data['source_valid'],
            ray_ids=data['ray_ids'], patch_ids=data['target_patch_ids'],
            observed_query=torch.ones(5, dtype=torch.bool), batch_ids=data['batch_ids'],
            temperature=.5, chunk_size=4)
    assert value == 'captured'
    assert captured['ray_ids'].shape == (2, 3, 3)
    assert captured['observed_query'].sum() == 5
    assert not captured['observed_query'][1, 2]
    assert (captured['ray_ids'][1, :, 2] == -1).all()
    assert torch.equal(captured['patch_ids'][0], data['target_patch_ids'][torch.tensor([1, 3, 4])])


def test_PBR_wrong_width_and_fabricated_query_for_physical_Top1_are_rejected():
    model, data = fixture('top1-only')
    with pytest.raises(ValueError):
        model(**dict(data, source_values=data['source_values'][..., :1024]))
    with pytest.raises(ValueError):
        model(**dict(data, native_query_rows=torch.zeros(5, 1024)))
    head, data = fixture('top1-plus-feature')
    with pytest.raises(ValueError):
        head(**dict(data, native_query_rows=None))
