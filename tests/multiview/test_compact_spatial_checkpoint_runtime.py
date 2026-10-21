from contextlib import nullcontext
import gc
import weakref

import pytest
import torch

from experiments.deformnext_retargeting.compact_spatial_checkpoint_runtime import compact_spatial_checkpoint_inputs
from pixal3d.modules import member_feature_controls as module
from pixal3d.modules.ray_feature_attention import RayFeatureReadout


def inputs():
    torch.manual_seed(8221)
    head = RayFeatureReadout()
    torch.nn.init.normal_(head.output.weight, std=.01)
    native = torch.randn(2, 1024, requires_grad=True)
    context = torch.randn(2, 1024)
    keys, values = torch.randn(32, 3, 1024), torch.randn(32, 3, 1024)
    indices = torch.tensor([7, 23])
    valid = torch.ones(2, 3, dtype=torch.bool)
    prior = torch.randn(2, 3)
    return head, native, context, keys, values, indices, valid, prior, 'normal'


def execute(compacted):
    args = inputs()
    with compact_spatial_checkpoint_inputs() if compacted else nullcontext() as statistics:
        out = module.checkpoint(module._spatial_feature_chunk, *args, use_reentrant=False)
        (out[0].square().sum()+out[1].square().sum()).backward()
    return out, args[1].grad, {k: p.grad for k, p in args[0].named_parameters()}, statistics


def test_exact_same_native_and_head_gradients_and_outputs():
    original = module.checkpoint
    a, input_a, parameters_a, _ = execute(False)
    b, input_b, parameters_b, statistics = execute(True)
    assert all(torch.equal(left, right) for left, right in zip(a, b))
    assert torch.equal(input_a, input_b)
    assert parameters_a.keys() == parameters_b.keys()
    assert all(torch.equal(parameters_a[k], parameters_b[k]) for k in parameters_a)
    assert statistics['checkpoints'] == 1 and statistics['selected_rows'] == 2
    assert statistics['cumulative_compact_input_bytes'] < statistics['maximum_full_frozen_input_bytes']
    assert module.checkpoint is original


@pytest.mark.parametrize('compacted', [False, True])
def test_actual_full_frozen_maps_lifetime_until_backward(compacted):
    args = inputs()
    key, value = weakref.ref(args[3]), weakref.ref(args[4])
    context = compact_spatial_checkpoint_inputs() if compacted else nullcontext()
    with context:
        out = module.checkpoint(module._spatial_feature_chunk, *args, use_reentrant=False)
    del args
    gc.collect()
    assert (key() is None) == compacted and (value() is None) == compacted
    out[0].sum().backward()


def test_rejects_trainable_feature_maps_and_restores_original_dispatch():
    args = list(inputs()); args[3].requires_grad_(True)
    original = module.checkpoint
    with pytest.raises(ValueError, match='frozen original'):
        with compact_spatial_checkpoint_inputs():
            module.checkpoint(module._spatial_feature_chunk, *args, use_reentrant=False)
    assert module.checkpoint is original


def test_unrelated_checkpoint_functions_use_original_implementation():
    value = torch.ones(3, requires_grad=True)
    with compact_spatial_checkpoint_inputs() as stats:
        result = module.checkpoint(lambda x: x*x, value, use_reentrant=False)
        result.sum().backward()
    assert torch.equal(value.grad, torch.full_like(value, 2))
    assert stats['checkpoints'] == 0
