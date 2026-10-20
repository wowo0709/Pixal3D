from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from pixal3d.modules.ray_group_matching import RayGroupConditioner
from pixal3d.modules.target_first_observed_query import ObservedQueryTargetFirstRayConditioner
from pixal3d.trainers.flow_matching.target_first_observed_query import ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer as Trainer


def fixture(mode):
    torch.manual_seed(813)
    config = dict(arm='c1', implementation='r9-native-v5', key_dim=1024, value_dim=1024,
        query_dim=1024, topk=64, hidden_dim=256, score_temperature=.05,
        projection_mode='native-bilinear', ray_mode=mode, ray_chunk_size=4)
    if mode == 'semantic-attention':
        config['task_attention'] = dict(decoder=True, score_prior=True, train_decoder=False,
            train_score=False, content_mode='residual-zero', initial_content_scale=16., chunk_size=4)
    else:
        config['candidate_aggregation'] = 'top1'
    model = ObservedQueryTargetFirstRayConditioner(**config)
    data = dict(native_voxels=torch.randn(1, 5, 1024), target_omega_voxels=torch.randn(1, 5, 1024),
        target_native_voxels=torch.randn(1, 5, 1024),
        ray_ids=torch.tensor([[[0, 0, 1, 1, -1], [2, 2, 1, 1, -1], [3, 3, 4, 4, -1]]]),
        source_keys=torch.randn(1, 3, 32, 1024), source_values=torch.randn(1, 3, 32, 1024),
        source_valid=torch.ones(1, 3, 32, dtype=torch.bool))
    observed = torch.tensor([[True, False, True, False, False]])
    return model, data, observed


def activate(model):
    with torch.no_grad():
        model.fusion_gate.fill_(.2)
        model.retargeter.net[-1].weight.normal_(std=.01)
        model.evidence_use.network[-1].weight.normal_(std=.01)
        if model.task_attention is not None:
            model.task_attention.content_output.weight.normal_(std=.01)


@pytest.mark.parametrize('mode', ['semantic-top1', 'semantic-attention'])
def test_matching_proposals_scores_and_initial_weights_reproduce_clean_reference(mode):
    model, data, observed = fixture(mode)
    masked = torch.where(observed[:, None], data['ray_ids'], torch.full_like(data['ray_ids'], -1))
    with torch.no_grad():
        reference = RayGroupConditioner.forward(model, **{**data, 'ray_ids': masked}).attention
        result = model(**data, observed_query=observed)
    for name in ['candidate_indices', 'matching_logits', 'candidate_weights', 'query_valid', 'base_evidence']:
        assert torch.equal(getattr(reference, name), getattr(result.attention, name))
    assert not result.source_numerator.any() and not result.semantic_weight.any()
    assert model.retargeter.net[0].in_features == 2048  # no physical extra DQ channels


@pytest.mark.parametrize('mode', ['semantic-top1', 'semantic-attention'])
def test_unobserved_omega_does_not_contaminate_matching_but_native_context_remains(mode):
    model, data, observed = fixture(mode)
    activate(model)
    altered = data['target_omega_voxels'].clone()
    altered[~observed] = torch.randn_like(altered[~observed]) * 100
    with torch.no_grad():
        original = model(**data, observed_query=observed)
        clean = model(**{**data, 'target_omega_voxels': altered}, observed_query=observed)
        assert torch.equal(original.source_numerator, clean.source_numerator)
        native = data['target_native_voxels'].clone()
        native[~observed] += torch.randn_like(native[~observed]) * 3
        context = model(**{**data, 'target_native_voxels': native}, observed_query=observed)
    assert torch.equal(original.attention.candidate_indices, context.attention.candidate_indices)
    assert torch.equal(original.attention.matching_logits, context.attention.matching_logits)
    assert not torch.equal(original.attention.retarget_residual, context.attention.retarget_residual)
    assert not torch.equal(original.source_numerator, context.source_numerator)


@pytest.mark.parametrize('mode', ['semantic-top1', 'semantic-attention'])
def test_initial_gate_and_R_learn_then_all_new_tensors_connect_to_loss(mode):
    model, data, observed = fixture(mode)
    result = model(**data, observed_query=observed)
    (result.source_numerator * torch.randn_like(result.source_numerator)).sum().backward()
    assert model.fusion_gate.grad.abs() > 0
    assert model.retargeter.net[-1].weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    activate(model)
    learned = model(**data, observed_query=observed)
    learned.source_numerator.square().mean().backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0, name
    assert all(parameter.grad is None for parameter in model.q2c.parameters())


@pytest.mark.parametrize('mode', ['semantic-top1', 'semantic-attention'])
def test_missing_observed_query_retains_target_and_has_zero_DDP_edges(mode):
    model, data, observed = fixture(mode)
    activate(model)
    result = model(**data, observed_query=torch.zeros_like(observed))
    assert not result.source_numerator.any() and not result.semantic_weight.any()
    assert not result.use_probability.any() and not result.attention.query_valid.any()
    result.source_numerator.sum().backward()
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    with torch.no_grad():
        empty = model(**{**data, 'source_valid': torch.zeros_like(data['source_valid'])}, observed_query=observed)
    assert not empty.source_numerator.any() and not empty.semantic_weight.any()


@pytest.mark.parametrize('mode', ['semantic-top1', 'semantic-attention'])
def test_saved_weight_wrong_values_keep_observed_query_routing_and_original_use_gate(mode):
    from experiments.deformnext_retargeting.target_first_value_control import fixed_use_value_control

    model, data, observed = fixture(mode)
    model.eval()
    activate(model)
    events = []
    with torch.no_grad():
        normal = model(**data, observed_query=observed)
        with fixed_use_value_control(model, events):
            wrong = model(**data, observed_query=observed)
            empty = model(**data, observed_query=torch.zeros_like(observed))
            repeated = model(**data, observed_query=observed)
        restored = model(**data, observed_query=observed)
    for name in ['candidate_indices', 'matching_logits', 'candidate_weights', 'query_valid']:
        assert torch.equal(getattr(normal.attention, name), getattr(wrong.attention, name))
    assert torch.equal(normal.use_probability, wrong.use_probability)
    assert torch.equal(normal.semantic_weight, wrong.semantic_weight)
    assert not torch.equal(normal.source_numerator, wrong.source_numerator)
    assert torch.equal(wrong.source_numerator, repeated.source_numerator)
    assert not empty.source_numerator.any() and not empty.use_probability.any()
    assert torch.equal(normal.source_numerator, restored.source_numerator)
    assert all(event['source_use_probability_exact_same_forward'] for event in events)
    assert not model.evidence_use._forward_pre_hooks and 'forward' not in model.__dict__


def test_trainer_keeps_native_membership_and_delivers_to_all_voxels_including_unobserved_ones():
    class Grid:
        front_view_transform_matrix = torch.eye(4)

        def __call__(self, values, *camera):
            return values.mean((1, 2))[:, None].expand(-1, 4096, -1)

    extractor = SimpleNamespace(proj_grid=Grid(), fixed_projection_transform=torch.eye(4))
    positive = {'global': torch.full((1, 5, 1024), 11.), 'proj': torch.full((1, 4096, 1024), 2.)}
    negative = {key: torch.zeros_like(value) for key, value in positive.items()}
    seen = []
    membership = (torch.arange(4096) % 1024).reshape(1, 1, 4096).expand(-1, 3, -1)
    target_valid = torch.zeros(1, 1024, dtype=torch.bool)
    target_valid[:, :4] = True

    def encode(images, **camera):
        seen.append(images)
        return positive, negative

    def module(**kwargs):
        assert torch.equal(kwargs['ray_ids'], membership)
        assert kwargs['observed_query'].sum() == 16
        assert (kwargs['ray_ids'][:, :, ~kwargs['observed_query'][0]] >= 0).all()
        return SimpleNamespace(source_numerator=torch.full((1, 3, 1024, 1024), .5),
            semantic_weight=torch.zeros(1, 3, 1024, 1), attention=object(),
            use_probability=torch.ones(1, 3072))

    facade = SimpleNamespace(image_attn_mode='proj', image_cond_model=extractor, _image_cond_module=extractor,
        encode_image_proj=encode, deform_conditioner=module, training_models={'deform_conditioner': module})
    images = torch.randn(1, 4, 3, 4, 4)
    transport = dict(target_keys=torch.zeros(1, 1024, 1024), target_query=torch.zeros(1, 1024, 1024),
        source_keys=torch.zeros(1, 3, 1024, 1024), source_values=torch.zeros(1, 3, 1024, 1024),
        source_valid=torch.ones(1, 3, 1024, dtype=torch.bool), target_valid=target_valid,
        patch_grid=(torch.arange(4096) % 1024).reshape(1, 16, 16, 16))
    transforms = torch.eye(4).repeat(1, 4, 1, 1)
    transforms[:, :, 2, 3] = 2
    camera = dict(transform_matrix=transforms, distance=torch.ones(1, 4) * 2,
                  camera_angle_x=torch.ones(1, 4), mesh_scale=torch.ones(1))
    with patch('pixal3d.trainers.flow_matching.target_first_observed_query.native_ray_membership', return_value=membership):
        actual, neg = Trainer._encode_deform_views(facade, images, camera, transport)
    assert len(seen) == 1 and torch.equal(seen[0], images[:, 0])
    assert actual['global'] is positive['global'] and neg is negative
    torch.testing.assert_close(actual['proj'], torch.full_like(actual['proj'], 2.375), rtol=0, atol=0)
    assert torch.equal(positive['proj'], torch.full_like(positive['proj'], 2.))
