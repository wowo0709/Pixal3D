import copy
import hashlib

import numpy as np
import pytest
import torch

from experiments.deformnext_retargeting.ray_correspondence_reliability import inference_features
from pixal3d.modules.ray_correspondence_reliability import (
    RayCorrespondenceReliability, reliability_features)
from pixal3d.modules.ray_consistent_retargeting import CandidateGeometryRetargeter, paired_correction


def test_deployed_reliability_matches_training_features_and_frozen_checkpoint(tmp_path):
    rng = np.random.default_rng(101)
    native, context = [rng.normal(size=(5, 1024)).astype(np.float32) for _ in range(2)]
    prior = rng.normal(size=(5, 64)).astype(np.float32)*10
    valid = rng.random((5, 64)) > .2
    valid[-1] = False
    ids = rng.integers(0, 3072, (5, 64))
    arrays = native, context, prior, valid, ids
    tensors = tuple(torch.from_numpy(a) for a in arrays)
    expected = inference_features(*arrays)
    actual = reliability_features(*tensors)
    assert np.allclose(actual.numpy(), expected, atol=3e-6, rtol=3e-6)
    model = RayCorrespondenceReliability()
    with pytest.raises(ValueError, match='load the actual'):
        model(*tensors)
    state = dict(head=copy.deepcopy(model.network.state_dict()),
        TRAIN_summary_mean=torch.randn(12), TRAIN_summary_scale=torch.rand(12)+.1, input_width=2060)
    path = tmp_path/'head.pt'
    torch.save(state, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    model.load_pilot(path, digest)
    x = torch.from_numpy(expected)
    x[:, 2048:] = (x[:, 2048:]-state['TRAIN_summary_mean'])/state['TRAIN_summary_scale']
    with torch.no_grad():
        probability = model.network(x)[:, 0].sigmoid()*tensors[3].any(-1)
    assert torch.allclose(model(*tensors), probability, atol=1e-6, rtol=1e-6)
    assert model(*tensors)[-1] == 0
    assert not any(p.requires_grad for p in model.parameters())
    empty = reliability_features(*(v[:0] for v in tensors))
    assert empty.shape == (0, 2060)


@pytest.mark.parametrize('width', [1024, 2048])
def test_zero_geometry_preserves_reference_and_common_gradients_with_immediate_new_gradient(width):
    torch.manual_seed(105)
    module = CandidateGeometryRetargeter(width+1024, width, 16)
    torch.nn.init.normal_(module.net[2].weight, std=.05)
    target = torch.randn(2, 1024)
    raw_target, raw = torch.randn(2, 2048), torch.randn(1, 128, 2048)
    values = torch.randn(1, 128, width)
    ids = torch.randint(0, 128, (2, 64))
    valid = torch.ones(2, 64, dtype=torch.bool)
    valid[1, -15:] = False
    p = torch.randn(2, 64).masked_fill(~valid, -torch.inf).softmax(-1)
    batch = torch.zeros(2, dtype=torch.long)
    appearance = torch.cat([values[0, ids[:, 0]], target], -1)
    base = module.appearance_forward(appearance, .25)
    correction = paired_correction(module, target, raw_target, values, raw,
        batch, ids, valid, p, .25, 8)
    assert torch.count_nonzero(correction) == 0
    assert torch.equal(base+correction, base)
    common = [v for n, v in module.named_parameters() if n != 'deformation_projection.weight']
    reference_grad = torch.autograd.grad(base.square().mean(), common, retain_graph=True)
    new_grad = torch.autograd.grad((base+correction).square().mean(), common, retain_graph=True)
    assert all(torch.equal(a, b) for a, b in zip(reference_grad, new_grad))
    gradient = torch.autograd.grad((base+correction).square().mean(), module.deformation_projection.weight)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_candidate_pairing_is_address_sensitive_and_joint_permutation_invariant():
    torch.manual_seed(107)
    module = CandidateGeometryRetargeter(2048, 1024, 16)
    torch.nn.init.normal_(module.net[2].weight, std=.05)
    torch.nn.init.normal_(module.deformation_projection.weight, std=.02)
    target, raw_target = torch.randn(2, 1024), torch.randn(2, 2048)
    values, raw = torch.randn(1, 128, 1024), torch.randn(1, 128, 2048)
    ids = torch.randint(0, 128, (2, 64))
    valid, batch = torch.ones(2, 64, dtype=torch.bool), torch.zeros(2, dtype=torch.long)
    probability = torch.randn(2, 64).softmax(-1)
    args = module, target, raw_target, values, raw, batch
    expected = paired_correction(*args, ids, valid, probability, .25, 8)
    perm = torch.randperm(64)
    shuffled = paired_correction(*args, ids[:, perm], valid[:, perm], probability[:, perm], .25, 16)
    assert torch.allclose(shuffled, expected, atol=2e-7, rtol=1e-5)
    wrong = paired_correction(module, target, raw_target, values, raw.roll(17, 1),
        batch, ids, valid, probability, .25, 8)
    assert not torch.allclose(wrong, expected, atol=1e-5)
    mean_difference = raw_target-(probability[..., None]*raw[batch[:, None], ids]).sum(1)
    mean = module.correction(torch.cat([values[0, ids[:, 0]], target], -1), mean_difference, .25)
    assert not torch.allclose(mean, expected, atol=1e-5)
