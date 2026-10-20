import copy

import pytest
import torch

from experiments.deformnext_retargeting.target_first_value_control import fixed_use_value_control
from pixal3d.modules.target_first_ray_v2 import TargetFirstRayConditionerV2


def fixture(mode):
    torch.manual_seed(813)
    config = dict(arm='c1', implementation='r9-native-v5', key_dim=1024, value_dim=1024,
        query_dim=1024, topk=64, hidden_dim=256, score_temperature=.5,
        projection_mode='native-bilinear', ray_mode=mode, ray_chunk_size=2)
    if mode == 'semantic-attention':
        config['task_attention'] = dict(decoder=True, score_prior=True, train_decoder=False,
            train_score=False, content_mode='residual-zero', initial_content_scale=16., chunk_size=2)
    else:
        config['candidate_aggregation'] = 'top1'
    model = TargetFirstRayConditionerV2(**config).eval()
    with torch.no_grad():
        model.fusion_gate.fill_(.2)
        model.retargeter.net[-1].weight.normal_(std=.01)
        model.evidence_use.network[-1].weight.normal_(std=.2)
        if model.task_attention is not None:
            model.task_attention.content_output.weight.normal_(std=.02)
    data = dict(native_voxels=torch.randn(2, 5, 1024), target_omega_voxels=torch.randn(2, 5, 1024),
        target_native_voxels=torch.randn(2, 5, 1024),
        ray_ids=torch.tensor([[[0, 0, 1, 1, -1], [2, 2, 1, 1, -1], [3, 3, 4, 4, -1]],
                              [[-1, 0, 0, 1, 1], [-1, 2, 2, 1, 1], [-1, 3, 3, 4, 4]]]),
        source_keys=torch.randn(2, 3, 32, 1024), source_values=torch.randn(2, 3, 32, 1024),
        source_valid=torch.ones(2, 3, 32, dtype=torch.bool))
    return model, data


@pytest.mark.parametrize('mode', ['semantic-top1', 'semantic-attention'])
def test_wrong_values_preserve_gate_scores_weights_and_restore_every_hook(mode):
    model, data = fixture(mode)
    inputs = {key: value.clone() for key, value in data.items()}
    state = copy.deepcopy(model.state_dict())
    events = []
    with torch.no_grad():
        normal = model(**data)
        with fixed_use_value_control(model, events):
            wrong = model(**data)
            # A missing-source call inside the same context cannot consume
            # the preceding asset's stored normal evidence.
            empty = model(**{**data, 'source_valid': torch.zeros_like(data['source_valid'])})
            repeated = model(**data)
        restored = model(**data)
    for field in ['candidate_indices', 'candidate_weights', 'matching_logits', 'query_valid']:
        assert torch.equal(getattr(normal.attention, field), getattr(wrong.attention, field))
    assert torch.equal(normal.use_probability, wrong.use_probability)
    assert torch.equal(normal.semantic_weight, wrong.semantic_weight)
    assert not torch.equal(normal.attention.base_evidence, wrong.attention.base_evidence)
    assert not torch.equal(normal.attention.retarget_residual, wrong.attention.retarget_residual)
    assert not torch.equal(normal.source_numerator, wrong.source_numerator)
    assert torch.equal(wrong.source_numerator, repeated.source_numerator)
    assert not empty.source_numerator.any() and not empty.semantic_weight.any()
    assert torch.equal(normal.source_numerator, restored.source_numerator)
    assert len(events) == 3
    assert events[0]['gate_evidence_change_prevented_RMS'] > 0
    assert events[1]['active_rays'] == 0
    assert all(event['same_forward_reference'] for event in events)
    assert all(event['source_use_probability_exact_same_forward'] for event in events)
    assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())
    assert all(torch.equal(value, inputs[key]) for key, value in data.items())
    assert 'forward' not in model.__dict__ and '_attention' not in model.__dict__
    assert not model.evidence_use._forward_pre_hooks
    if model.task_attention is not None:
        assert not model.task_attention._forward_hooks


@pytest.mark.parametrize('mode', ['semantic-top1', 'semantic-attention'])
def test_exception_cleans_up_and_training_is_rejected(mode):
    model, data = fixture(mode)
    with pytest.raises(RuntimeError, match='test abort'):
        with fixed_use_value_control(model, []):
            with torch.no_grad():
                model(**data)
            raise RuntimeError('test abort')
    assert 'forward' not in model.__dict__ and '_attention' not in model.__dict__
    assert not model.evidence_use._forward_pre_hooks
    if model.task_attention is not None:
        assert not model.task_attention._forward_hooks
    with fixed_use_value_control(model, []):
        with pytest.raises(ValueError, match='eval and no_grad'):
            model(**data)
        model.train()
        with torch.no_grad(), pytest.raises(ValueError, match='eval and no_grad'):
            model(**data)
