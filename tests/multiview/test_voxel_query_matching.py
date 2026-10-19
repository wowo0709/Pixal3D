"""A new query must preserve the candidate/head contract and native zero start."""
from contextlib import nullcontext

import pytest
import torch

from experiments.deformnext_retargeting.task_attention_controls import task_value_context
from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
from pixal3d.modules.voxel_query_matching import VoxelQueryConditioner, fuse_voxel_condition


def fixture():
    # Random tiny CPU inputs test algebra/gradients, never model quality. The
    # loose score temperature prevents random prior saturation in that test.
    torch.manual_seed(7123)
    config = dict(implementation='r9-native-v5', arm='c1', score_temperature=.5,
        key_dim=1024, value_dim=1024, query_dim=1024, hidden_dim=256, topk=64,
        task_attention=dict(decoder=True, score_prior=True, train_decoder=False,
            train_score=False, content_mode='residual-zero', initial_content_scale=16., chunk_size=2))
    model = VoxelQueryConditioner(voxel_chunk_size=2, **config)
    reference = DeformationAwareConditioner(**config)
    reference.load_state_dict(model.state_dict(), strict=True)
    inputs = dict(target_keys=torch.randn(1, 4, 1024), source_keys=torch.randn(1, 3, 32, 1024),
        source_values=torch.randn(1, 3, 32, 1024), source_valid=torch.ones(1, 3, 32, dtype=torch.bool),
        target_query=torch.randn(1, 4, 1024), target_valid=torch.tensor([[True, True, False, True]]),
        semantic_indices=torch.stack([torch.arange(64).roll(i) for i in range(4)])[None],
        candidate_valid=torch.ones(1, 4, 64, dtype=torch.bool))
    patch_index = torch.tensor([[0, 0, 1, 2, 3]])
    native = inputs['target_query'].gather(1, patch_index[..., None].expand(-1, -1, 1024))
    return model, reference, inputs, patch_index, native


@pytest.mark.parametrize('learned_utility', [False, True])
@pytest.mark.parametrize('amp', [False, True])
def test_repeated_patch_queries_reproduce_reference_candidates_scores_and_evidence(learned_utility, amp):
    model, reference, inputs, patch_index, native = fixture()
    if learned_utility:
        with torch.no_grad():
            model.task_attention.content_output.weight.normal_(std=.02)
            model.retargeter.net[-1].weight.normal_(std=.001)
        reference.load_state_dict(model.state_dict(), strict=True)
    ctx = torch.autocast('cpu', dtype=torch.bfloat16) if amp else nullcontext()
    with torch.no_grad(), ctx:
        expected = reference(**inputs)
        actual = model(**inputs, native_query=native, patch_index=patch_index)
    for key in ['evidence', 'base_evidence', 'candidate_weights', 'matching_logits', 'retarget_residual']:
        value = getattr(expected, key)
        gathered = value.gather(1, patch_index[..., None].expand(-1, -1, value.shape[-1]))
        torch.testing.assert_close(getattr(actual, key), gathered, rtol=2e-5, atol=2e-6)
    assert torch.equal(actual.query_valid, inputs['target_valid'].gather(1, patch_index))
    assert torch.equal(actual.candidate_indices, inputs['semantic_indices'].gather(
        1, patch_index[..., None].expand(-1, -1, 64)))


def test_distinct_native_voxel_queries_change_weights_receive_gradients_and_keep_frozen_head():
    model, _, inputs, patch_index, native = fixture()
    with torch.no_grad():
        model.task_attention.content_output.weight.normal_(std=.03)
    native = native.clone()
    native[:, 1] = torch.randn_like(native[:, 1])
    native.requires_grad_(True)
    frozen = {n: p.detach().clone() for n, p in model.q2c.named_parameters()}
    actual = model(**inputs, native_query=native, patch_index=patch_index)
    assert not torch.equal(actual.candidate_weights[:, 0], actual.candidate_weights[:, 1])
    actual.evidence.square().mean().backward()
    assert native.grad.isfinite().all() and native.grad.abs().sum() > 0
    for name in ['query', 'context', 'key', 'content_output']:
        grad = getattr(model.task_attention, name).weight.grad
        assert grad is not None and grad.isfinite().all() and grad.abs().sum() > 0
    for name, parameter in model.q2c.named_parameters():
        assert not parameter.requires_grad and parameter.grad is None
        assert torch.equal(parameter, frozen[name])


def test_native_initial_fusion_is_exact_and_invalid_voxels_stay_native_after_learning():
    model, _, inputs, patch_index, native = fixture()
    actual = model(**inputs, native_query=native, patch_index=patch_index)
    anchor = {'proj': torch.randn_like(native)}
    torch.testing.assert_close(fuse_voxel_condition({'proj': native.clone()}, anchor, actual, 3)['proj'], native,
                               rtol=0, atol=0)
    with torch.no_grad():
        model.fusion_gate.fill_(.5)
        model.retargeter.net[-1].weight.normal_(std=.001)
    trained = model(**inputs, native_query=native, patch_index=patch_index)
    fused = fuse_voxel_condition({'proj': native.clone()}, anchor, trained, 3)['proj']
    assert torch.equal(fused[:, 3], native[:, 3])
    assert not torch.equal(fused[:, 0], native[:, 0])
    assert model.retargeter.input_dim == 2048


def test_frozen_native_inputs_still_backpropagate_through_checkpoint_and_zero_start():
    model, _, inputs, patch_index, native = fixture()
    assert not native.requires_grad and not inputs['source_values'].requires_grad
    output = model(**inputs, native_query=native, patch_index=patch_index)
    fused = fuse_voxel_condition({'proj': native.clone()}, {'proj': native * .5}, output, 3)['proj']
    (fused * torch.randn_like(fused)).sum().backward()
    assert model.fusion_gate.grad.abs() > 0
    assert model.retargeter.net[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert model.task_attention.content_output.weight.grad.abs().sum() == 0
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.task_attention.content_output.weight.normal_(std=.02)
    learned = model(**inputs, native_query=native, patch_index=patch_index)
    learned.evidence.square().mean().backward()
    assert model.task_attention.query.weight.grad.abs().sum() > 0
    assert model.task_attention.key.weight.grad.abs().sum() > 0


def test_value_hook_still_changes_only_weight_to_value_association_and_cached_logits_are_ignored():
    model, _, inputs, patch_index, native = fixture()
    with torch.no_grad():
        model.task_attention.content_output.weight.normal_(std=.02)
        normal = model(**inputs, native_query=native, patch_index=patch_index)
        events = []
        with task_value_context(model, 'wrong-address', events):
            altered = model(**inputs, native_query=native, patch_index=patch_index)
        replay = model(**inputs, native_query=native, patch_index=patch_index,
                       matching_logits=torch.randn(1, 4, 64) * 1e4)
    assert torch.equal(normal.candidate_weights, altered.candidate_weights)
    assert torch.equal(normal.matching_logits, altered.matching_logits)
    assert not torch.equal(normal.base_evidence, altered.base_evidence)
    assert torch.equal(normal.evidence, replay.evidence)
    assert events and all(e['matching_logits_and_weights_unchanged'] for e in events)


def test_no_active_voxels_has_finite_zero_evidence_and_all_trainable_gradient_edges():
    model, _, inputs, patch_index, native = fixture()
    inputs['target_valid'].fill_(False)
    actual = model(**inputs, native_query=native, patch_index=patch_index)
    assert not actual.query_valid.any() and not actual.evidence.any()
    assert actual.evidence.isfinite().all()
    actual.evidence.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_registered_trainer_uses_new_module_without_replacing_reference_class():
    from pixal3d import trainers
    original = trainers.ImageConditionedProjFlowMatchingCFGTrainer
    new = trainers.VoxelQueryImageConditionedProjFlowMatchingCFGTrainer
    assert original.__module__ == 'pixal3d.trainers.flow_matching.flow_matching'
    assert new.__module__ == 'pixal3d.trainers.flow_matching.voxel_query_matching'
    assert issubclass(new, original)
