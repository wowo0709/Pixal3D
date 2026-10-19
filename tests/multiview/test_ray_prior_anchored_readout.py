import torch

from pixal3d.modules.ray_prior_anchored_readout import (
    PriorAnchoredRayFeatureReadout, correspondence_probabilities)
from experiments.deformnext_retargeting.ray_set_head_training import ray_set_logit_loss


def fixture():
    torch.manual_seed(12)
    model = PriorAnchoredRayFeatureReadout()
    model.requires_grad_(False)
    model.query.requires_grad_(True)
    model.key.requires_grad_(True)
    # A learned utility prior must survive initialization as well.
    torch.nn.init.normal_(model.score_selector.content_output.weight, std=.1)
    args = (torch.randn(3, 1024), torch.randn(3, 1024), torch.randn(3, 5, 1024),
            torch.randn(3, 5, 1024), torch.ones(3, 5, dtype=torch.bool), torch.randn(3, 5))
    return model, args


def test_anchored_probabilities_and_output_preserve_actual_prior():
    model, (native, context, keys, values, valid, prior) = fixture()
    correction, actual = model(native, context, keys, values, valid, prior)
    expected = model.score_selector(native, context, values, valid, prior)[1]
    assert torch.count_nonzero(model.query.weight) == 0
    assert torch.count_nonzero(correction) == 0
    assert torch.equal(actual, expected[:, None].expand_as(actual))
    assert torch.equal(actual, correspondence_probabilities(
        model, native, context, keys, values, valid, prior))


def test_two_stage_qk_learning_matches_native_forward_and_preserves_frozen_weights():
    model, args = fixture()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW([model.query.weight, model.key.weight], lr=1e-3)
    positive = torch.tensor([[True, False, False, False, False]] * 3)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        probability = correspondence_probabilities(model, *args)
        assert torch.equal(probability, model(*args)[1])
        loss = ray_set_logit_loss(probability.mean(1).clamp_min(1e-30).log(),
                                  torch.arange(3), positive)[0]
        loss.backward()
        assert model.query.weight.grad.abs().sum() > 0
        assert bool(model.key.weight.grad.abs().sum() > 0) == bool(step)
        optimizer.step()
    assert len(optimizer.state) == 2
    changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
    assert changed == {'query.weight', 'key.weight'}
    assert torch.count_nonzero(model(*args)[0]) == 0


def test_masked_and_empty_rays_use_identical_native_probability_math():
    model, args = fixture()
    torch.nn.init.normal_(model.query.weight, std=.01)
    args[-2][0, 1:] = False
    args[-2][1] = False
    actual = model(*args)[1]
    assert torch.equal(actual, correspondence_probabilities(model, *args))
    assert torch.count_nonzero(actual[1]) == 0
    assert torch.equal(actual[0, :, 0], torch.ones(4))
