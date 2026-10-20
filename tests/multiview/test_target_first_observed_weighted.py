import pytest
import torch

from experiments.deformnext_retargeting.target_first_value_control import fixed_use_value_control
from pixal3d.modules.target_first_observed_query import ObservedQueryTargetFirstRayConditioner
from pixal3d.modules.target_first_observed_weighted import ObservedQueryWeightedTargetFirstRayConditioner


def fixture():
    base = dict(arm='c1', implementation='r9-native-v5', key_dim=1024, value_dim=1024,
        query_dim=1024, topk=64, hidden_dim=256, score_temperature=.5,
        projection_mode='native-bilinear', ray_chunk_size=4)
    torch.manual_seed(914)
    weighted = ObservedQueryWeightedTargetFirstRayConditioner(**base, ray_mode='semantic-weighted')
    torch.manual_seed(914)
    attention = ObservedQueryTargetFirstRayConditioner(**base, ray_mode='semantic-attention',
        task_attention=dict(decoder=True, score_prior=True, train_decoder=False,
            train_score=False, content_mode='residual-zero', initial_content_scale=16., chunk_size=4))
    data = dict(native_voxels=torch.randn(1, 5, 1024), target_omega_voxels=torch.randn(1, 5, 1024),
        target_native_voxels=torch.randn(1, 5, 1024), observed_query=torch.tensor([[True, False, True, False, False]]),
        ray_ids=torch.tensor([[[0, 0, 1, 1, -1], [2, 2, 1, 1, -1], [3, 3, 4, 4, -1]]]),
        source_keys=torch.randn(1, 3, 32, 1024), source_values=torch.randn(1, 3, 32, 1024),
        source_valid=torch.ones(1, 3, 32, dtype=torch.bool))
    return weighted, attention, data


def test_common_state_and_cold_top64_readout_exactly_match_learned_attention():
    weighted, attention, data = fixture()
    reference = attention.state_dict()
    assert all(torch.equal(value, reference[key]) for key, value in weighted.state_dict().items())
    assert not any('task_attention' in key for key in weighted.state_dict())
    with torch.no_grad():
        for model in [weighted, attention]:
            model.fusion_gate.fill_(.2)
            model.retargeter.net[-1].weight.fill_(.001)
            model.evidence_use.network[-1].weight.fill_(.01)
        fixed, learned = weighted(**data), attention(**data)
    for field in ['candidate_indices', 'matching_logits', 'candidate_weights', 'base_evidence', 'retarget_residual']:
        assert torch.equal(getattr(fixed.attention, field), getattr(learned.attention, field))
    assert torch.equal(fixed.use_probability, learned.use_probability)
    assert torch.equal(fixed.source_numerator, learned.source_numerator)
    assert torch.equal(fixed.semantic_weight, learned.semantic_weight)
    active = fixed.attention.query_valid
    assert (fixed.attention.candidate_weights[active] > 0).sum(-1).min() == 64
    assert not torch.allclose(fixed.attention.candidate_weights[active],
        torch.full_like(fixed.attention.candidate_weights[active], 1/64))


def test_weighted_control_has_identical_initial_gradient_paths_and_learnable_R_and_use():
    weighted, _, data = fixture()
    initial = weighted(**data)
    (initial.source_numerator * torch.randn_like(initial.source_numerator)).sum().backward()
    assert weighted.fusion_gate.grad.abs() > 0
    assert weighted.retargeter.net[-1].weight.grad.abs().sum() > 0
    weighted.zero_grad(set_to_none=True)
    with torch.no_grad():
        weighted.fusion_gate.fill_(.2)
        weighted.retargeter.net[-1].weight.normal_(std=.01)
        weighted.evidence_use.network[-1].weight.normal_(std=.02)
    weighted(**data).source_numerator.square().mean().backward()
    for name, value in weighted.named_parameters():
        if value.requires_grad:
            assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0, name
    assert all(value.grad is None for value in weighted.q2c.parameters())


def test_empty_query_fallback_and_fixed_use_wrong_values_keep_frozen_scores():
    weighted, _, data = fixture()
    weighted.eval()
    events = []
    with torch.no_grad():
        weighted.fusion_gate.fill_(.2)
        weighted.retargeter.net[-1].weight.normal_(std=.01)
        weighted.evidence_use.network[-1].weight.normal_(std=.02)
        normal = weighted(**data)
        with fixed_use_value_control(weighted, events):
            wrong = weighted(**data)
            empty = weighted(**{**data, 'observed_query': torch.zeros_like(data['observed_query'])})
        restored = weighted(**data)
    for field in ['candidate_indices', 'matching_logits', 'candidate_weights']:
        assert torch.equal(getattr(normal.attention, field), getattr(wrong.attention, field))
    assert torch.equal(normal.use_probability, wrong.use_probability)
    assert not torch.equal(normal.source_numerator, wrong.source_numerator)
    assert not empty.source_numerator.any() and not empty.semantic_weight.any()
    assert torch.equal(normal.source_numerator, restored.source_numerator)
    assert all(event['source_use_probability_exact_same_forward'] for event in events)


@pytest.mark.parametrize('override', [dict(ray_mode='semantic-top1'), dict(candidate_aggregation='top1'),
    dict(task_attention={'decoder': True}), dict(correspondence_mode='native-spatial')])
def test_misdeclared_weighted_or_spatial_arms_are_rejected(override):
    with pytest.raises(ValueError, match='fixed matching-score weighted'):
        ObservedQueryWeightedTargetFirstRayConditioner(**{'ray_mode': 'semantic-weighted', **override})
