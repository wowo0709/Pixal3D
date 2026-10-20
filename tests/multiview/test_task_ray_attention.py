"""The new task attention must retain the prior and learn without unbounding it."""
import torch

from pixal3d.modules.task_ray_retargeting import BoundedTaskRayAttention


def inputs():
    torch.manual_seed(177)
    native,context = torch.randn(3,1024),torch.randn(3,1024)
    deformation,keys = torch.randn(3,2048),torch.randn(3,64,1024)
    valid = torch.ones(3,64,dtype=torch.bool)
    valid[1,50:] = False
    valid[2] = False
    prior = torch.randn(3,4,64).masked_fill(~valid[:,None],-1e9).softmax(-1)*valid[:,None]
    return native,context,deformation,keys,valid,prior


def test_zero_query_preserves_semantic_prior_and_handles_empty_support():
    model = BoundedTaskRayAttention(torch.randn(256,1024)*.03)
    args = inputs()
    probability,kl = model(*args)
    assert model.query.in_features == 4096
    assert torch.allclose(probability,args[-1],atol=2e-7,rtol=1e-6)
    assert probability[2].count_nonzero() == 0 and torch.isfinite(kl).all()
    assert kl.max() < 1e-6
    cost = torch.randn_like(probability)
    (probability*cost).sum().backward()
    assert model.query.weight.grad.abs().sum() > 0
    assert model.key.weight.grad.abs().sum() == 0


def test_deformation_channels_affect_attention_and_prior_change_is_bounded():
    model = BoundedTaskRayAttention(torch.randn(256,1024)*.03)
    torch.nn.init.normal_(model.query.weight,std=.01)
    args = inputs()
    probability,kl = model(*args)
    changed = list(args)
    changed[2] = -args[2]
    alternative,_ = model(*changed)
    assert not torch.allclose(probability,alternative,atol=1e-6)
    assert (kl[:2] > 0).all()
    prior = args[-1]
    valid = args[-2][:,None].expand_as(probability)
    log_ratio = (probability.clamp_min(1e-30)/prior.clamp_min(1e-30)).log()
    # Normalization can move the log ratio by at most another bound.
    assert log_ratio[valid].abs().max() <= 2*model.logit_bound+1e-5
    (probability*torch.randn_like(probability)).sum().backward()
    assert model.key.weight.grad.abs().sum() > 0
