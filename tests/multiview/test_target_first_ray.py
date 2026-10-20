from contextlib import nullcontext

import pytest
import torch

from pixal3d.modules.target_first_ray import (
    TargetFirstRayConditioner, SourceEvidenceUse, add_source_ray_correction)


def fixture(mode='semantic-attention'):
    torch.manual_seed(813)
    config = dict(arm='c1', implementation='r9-native-v5', key_dim=1024, value_dim=1024,
        query_dim=1024, topk=64, hidden_dim=256, score_temperature=.5,
        projection_mode='native-bilinear', ray_mode=mode, ray_chunk_size=4)
    if mode == 'semantic-attention':
        config['task_attention'] = dict(decoder=True, score_prior=True, train_decoder=False,
            train_score=False, content_mode='residual-zero', initial_content_scale=16., chunk_size=4)
    else:
        config['candidate_aggregation'] = 'top1'
    model = TargetFirstRayConditioner(**config)
    data = dict(native_voxels=torch.randn(1, 5, 1024), target_omega_voxels=torch.randn(1, 5, 1024),
        target_native_voxels=torch.randn(1, 5, 1024),
        ray_ids=torch.tensor([[[0, 0, 1, 1, -1], [2, 2, 1, 1, -1], [3, 3, 4, 4, -1]]]),
        source_keys=torch.randn(1, 3, 32, 1024), source_values=torch.randn(1, 3, 32, 1024),
        source_valid=torch.ones(1, 3, 32, dtype=torch.bool))
    return model, data


def activate(model):
    with torch.no_grad():
        model.fusion_gate.fill_(.2)
        model.retargeter.net[-1].weight.normal_(std=.01)
        if model.task_attention is not None:
            model.task_attention.content_output.weight.normal_(std=.02)
        model.evidence_use.network[-1].weight.normal_(std=.01)


@pytest.mark.parametrize('mode', ['semantic-top1', 'semantic-attention'])
def test_zero_start_matches_target_carrier_and_has_initial_source_learning_signals(mode):
    model, data = fixture(mode)
    output = model(**data)
    assert torch.count_nonzero(output.map_delta) == 0
    condition = {'global': torch.randn(1, 5, 1024), 'proj': data['native_voxels'].clone()}
    actual = add_source_ray_correction(condition, torch.zeros_like(condition['proj']))
    assert torch.equal(actual['proj'], condition['proj'])
    assert actual['global'] is condition['global']
    (output.map_delta * torch.randn_like(output.map_delta)).sum().backward()
    assert model.fusion_gate.grad.abs() > 0
    assert model.retargeter.net[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.q2c.parameters())


@pytest.mark.parametrize('amp', [False, True])
def test_no_candidate_or_no_ray_reverts_to_target_after_nonzero_learning(amp):
    model, data = fixture()
    activate(model)
    original = {k:v.clone() for k,v in data.items()}
    ctx = torch.autocast('cpu', dtype=torch.bfloat16) if amp else nullcontext()
    with ctx:
        active = model(**data)
        assert active.map_delta.abs().sum() > 0
        no_source = model(**{**data, 'source_valid': torch.zeros_like(data['source_valid'])})
        no_ray = model(**{**data, 'ray_ids': torch.full_like(data['ray_ids'], -1)})
    for result in [no_source, no_ray]:
        assert torch.count_nonzero(result.map_delta) == 0
        assert torch.count_nonzero(result.use_probability) == 0
    for key in data:
        assert torch.equal(data[key], original[key])
    assert torch.count_nonzero(active.map_delta.flatten(1,2)[~active.attention.query_valid]) == 0


def test_source_use_gate_can_abstain_without_altering_correspondence_weights():
    model, data = fixture()
    activate(model)
    with torch.no_grad():
        normal = model(**data)
        model.evidence_use.network[-1].weight.zero_()
        model.evidence_use.network[-1].bias.fill_(-30)
        abstained = model(**data)
    assert torch.equal(normal.attention.candidate_indices, abstained.attention.candidate_indices)
    assert torch.equal(normal.attention.candidate_weights, abstained.attention.candidate_weights)
    assert abstained.map_delta.abs().max() < 1e-12
    assert normal.map_delta.abs().max() > 1e-4


def test_only_source_path_receives_trainable_corrections_and_all_heads_have_gradients():
    model, data = fixture()
    activate(model)
    result = model(**data)
    result.map_delta.square().mean().backward()
    for module in [model.retargeter, model.task_attention, model.evidence_use]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in module.parameters())
    assert all(p.grad is None for p in model.q2c.parameters())
    anchor = {'global': torch.randn(1, 5, 1024), 'proj': data['native_voxels'].clone()}
    saved = {key:value.clone() for key,value in anchor.items()}
    delta = torch.randn_like(anchor['proj'], requires_grad=True)
    final = add_source_ray_correction(anchor, delta)
    assert final['global'] is anchor['global']
    assert all(torch.equal(anchor[key], saved[key]) for key in anchor)
    final['proj'].square().mean().backward()
    assert delta.grad.abs().sum() > 0


def test_top1_control_really_uses_one_native_feature_and_has_no_attention_weights():
    model, data = fixture('semantic-top1')
    result = model(**data).attention
    weights = result.candidate_weights[result.query_valid]
    assert model.task_attention is None
    assert ((weights > 0).sum(-1) == 1).all()
    chosen = result.candidate_indices[result.query_valid].gather(1, weights.argmax(-1,keepdim=True))[:, 0]
    assert torch.equal(result.base_evidence[result.query_valid], data['source_values'][0].flatten(0,1)[chosen])


def test_missing_slots_cannot_poison_or_change_the_evidence_use_gate():
    torch.manual_seed(3)
    gate = SourceEvidenceUse()
    a, b = torch.randn(1, 3, 1024), torch.randn(1, 3, 1024)
    valid = torch.tensor([[True, False, True]])
    expected = gate(a, b, valid)
    a[:, 1] = float('nan'); b[:, 1] = float('inf')
    actual = gate(a, b, valid)
    assert torch.equal(expected, actual)
    assert actual[0, 1] == 0
