import pytest
import torch

from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin


def inputs():
    torch.manual_seed(4)
    return dict(
        target_keys=torch.randn(2, 5, 8),
        source_keys=torch.randn(2, 3, 24, 8),
        source_values=torch.randn(2, 3, 24, 6),
        source_valid=torch.ones(2, 3, 24, dtype=torch.bool),
    )


def make(arm, **kwargs):
    return DeformationAwareConditioner(
        arm=arm, key_dim=8, value_dim=6, query_dim=4, topk=64, **kwargs
    )


def test_v1_uses_only_supplied_spatial_correspondence():
    data = inputs()
    index = torch.zeros(2, 3, 5, dtype=torch.long)
    out = make("v1")(**data, spatial_indices=index)
    expected = data["source_values"][:, :, 0].mean(dim=1)[:, None].expand(-1, 5, -1)
    torch.testing.assert_close(out.evidence, expected)
    assert out.matching_logits is None


def test_v1_does_not_require_semantic_key_tensors():
    data = inputs()
    out = make("v1")(
        source_values=data["source_values"],
        source_valid=data["source_valid"],
        spatial_indices=torch.zeros(2, 3, 5, dtype=torch.long),
    )
    assert out.evidence.shape == (2, 5, 6)


def test_b1_is_raw_cosine_top1_and_b2_exposes_no_score_consumer_weights():
    data = inputs()
    b1 = make("b1")(**data)
    b2 = make("b2")(**data)
    assert b1.candidate_indices.shape == (2, 5, 1)
    assert b2.matching_logits.shape == (2, 5, 64)
    assert b2.candidate_weights is None and b2.view_weights is None


def test_b3_matching_scores_control_candidate_and_view_aggregation():
    data = inputs()
    model = make("b3")
    out = model(**data)
    assert out.candidate_weights.shape == (2, 5, 64)
    assert out.view_weights.shape == (2, 5, 3)
    torch.testing.assert_close(out.view_weights.sum(-1), torch.ones(2, 5))
    assert torch.isfinite(out.evidence).all()


def test_q2c_factorized_first_layer_matches_concatenated_mlp():
    torch.manual_seed(14)
    model = make("b2")
    target = torch.randn(2, 5, 8)
    candidates = torch.randn(2, 5, 7, 8)
    cosine = torch.randn(2, 5, 7)
    expanded = target[:, :, None].expand_as(candidates)
    expected = model.q2c(
        torch.cat((expanded, candidates, cosine[..., None]), dim=-1)
    ).squeeze(-1)
    actual = model._q2c_logits(target, candidates, cosine)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_c1_has_no_deformation_channel_and_rejects_one():
    model = make("c1")
    assert model.retargeter.input_dim == 6 + 4
    data = inputs()
    query = torch.randn(2, 5, 4)
    with pytest.raises(ValueError, match="no deformation input channel"):
        model(**data, target_query=query, deformation=torch.zeros(2, 5, 7))


def test_c2_has_explicit_deformation_schema_and_zero_init_parity():
    data = inputs()
    base_model = make("b3")
    model = make("c2", deformation_dim=7)
    model.q2c.load_state_dict(base_model.q2c.state_dict())
    base = base_model(**data)
    out = model(
        **data, target_query=torch.randn(2, 5, 4), deformation=torch.randn(2, 5, 7)
    )
    assert model.retargeter.input_dim == 6 + 4 + 7
    torch.testing.assert_close(out.retarget_residual, torch.zeros_like(out.retarget_residual))
    torch.testing.assert_close(out.evidence, base.evidence)


def test_c2_derives_inference_safe_state_difference_when_not_cached():
    data = inputs()
    model = make("c2", deformation_dim=8)
    out = model(**data, target_query=torch.randn(2, 5, 4))
    assert out.retarget_residual.shape == (2, 5, 6)


@pytest.mark.parametrize("arm", ["v1", "b1", "b2", "b3", "c1", "c2"])
def test_source_off_is_exact_zero_for_every_arm(arm):
    model = make(arm, **({"deformation_dim": 7} if arm == "c2" else {}))
    data = inputs()
    if arm == "v1":
        data["spatial_indices"] = torch.zeros(2, 3, 5, dtype=torch.long)
    out = model(**data, control="source-off")
    assert torch.count_nonzero(out.evidence) == 0


def test_trainer_bridge_maps_patch_evidence_to_dense_projection_grid():
    bridge = object.__new__(ImageConditionedProjMixin)
    bridge.deform_conditioner = DeformationAwareConditioner(
        arm="b1", key_dim=2, value_dim=3, query_dim=1, topk=2
    )
    bridge.training_models = {}
    transport = dict(
        target_keys=torch.tensor([[[1., 0.], [0., 1.]]]),
        source_keys=torch.tensor([[[[1., 0.], [0., 1.]]]]),
        source_values=torch.tensor([[[[1., 2., 3.], [4., 5., 6.]]]]),
        source_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        patch_grid=torch.tensor([[[[0, 1], [1, 0]], [[1, 0], [0, 1]]]]),
    )
    condition = {"global": torch.zeros(1, 1, 2), "proj": torch.zeros(1, 8, 3)}
    actual = bridge._inject_deform_transport(condition, transport, coords=None)["proj"]
    expected = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [4., 5., 6.], [1., 2., 3.],
                              [4., 5., 6.], [1., 2., 3.], [1., 2., 3.], [4., 5., 6.]]])
    torch.testing.assert_close(actual, expected)


def test_trainer_bridge_maps_patch_evidence_to_sparse_coordinates():
    bridge = object.__new__(ImageConditionedProjMixin)
    bridge.deform_conditioner = DeformationAwareConditioner(
        arm="b1", key_dim=2, value_dim=3, query_dim=1, topk=2
    )
    bridge.training_models = {}
    transport = dict(
        target_keys=torch.tensor([[[1., 0.], [0., 1.]]]),
        source_keys=torch.tensor([[[[1., 0.], [0., 1.]]]]),
        source_values=torch.tensor([[[[1., 2., 3.], [4., 5., 6.]]]]),
        source_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        patch_grid=torch.tensor([[[[0, 1], [1, 0]], [[1, 0], [0, 1]]]]),
    )
    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1]])
    condition = {"global": torch.zeros(1, 1, 2), "proj": torch.zeros(2, 3)}
    actual = bridge._inject_deform_transport(condition, transport, coords=coords)["proj"]
    torch.testing.assert_close(actual, torch.tensor([[1., 2., 3.], [4., 5., 6.]]))


def test_shared_q2c_checkpoint_loads_strictly_and_freezes(tmp_path):
    source = DeformationAwareConditioner(
        arm="b2", key_dim=3, value_dim=4, query_dim=2, hidden_dim=5, topk=2
    )
    with torch.no_grad():
        for parameter in source.q2c.parameters():
            parameter.fill_(0.125)
    path = tmp_path / "q2c.pt"
    torch.save({"q2c": source.q2c.state_dict()}, path)
    loaded = DeformationAwareConditioner(
        arm="c1", key_dim=3, value_dim=4, query_dim=2, hidden_dim=5, topk=2,
        q2c_checkpoint=str(path), freeze_q2c=True,
    )
    assert all(not parameter.requires_grad for parameter in loaded.q2c.parameters())
    for expected, actual in zip(source.q2c.parameters(), loaded.q2c.parameters()):
        assert torch.equal(expected, actual)


def test_freezing_q2c_without_checkpoint_is_rejected():
    with pytest.raises(ValueError, match="requires q2c_checkpoint"):
        DeformationAwareConditioner(
            arm="b2", key_dim=3, value_dim=4, query_dim=2, topk=2,
            freeze_q2c=True,
        )


def test_sparse_transport_accepts_compact_patch_ids():
    bridge = object.__new__(ImageConditionedProjMixin)
    bridge.deform_conditioner = DeformationAwareConditioner(
        arm="b1", key_dim=2, value_dim=3, query_dim=1, topk=2
    )
    bridge.training_models = {}
    transport = dict(
        target_keys=torch.tensor([[[1., 0.], [0., 1.]]]),
        source_keys=torch.tensor([[[[1., 0.], [0., 1.]]]]),
        source_values=torch.tensor([[[[1., 2., 3.], [4., 5., 6.]]]]),
        source_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        patch_ids=torch.tensor([1, 0]),
    )
    coords = torch.tensor([[0, 0, 0, 0], [0, 63, 63, 63]])
    condition = {"global": torch.zeros(1, 1, 2), "proj": torch.zeros(2, 3)}
    actual = bridge._inject_deform_transport(condition, transport, coords=coords)["proj"]
    torch.testing.assert_close(actual, torch.tensor([[4., 5., 6.], [1., 2., 3.]]))
