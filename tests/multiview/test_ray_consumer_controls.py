import pytest
import torch

from experiments.deformnext_retargeting.ray_consumer_controls import restrict_consumer


def test_all_mask_preserves_inside_bits_even_when_subtract_add_would_cancel():
    native=torch.full((1,2,2),10000.,dtype=torch.bfloat16)
    updated=torch.tensor([[[.125,.25],[.5,.75]]],dtype=torch.bfloat16)
    observed=torch.tensor([[True,False]])
    before_native=native.clone();before_updated=updated.clone()
    result=restrict_consumer(native,updated,observed,'mask-all')
    assert torch.equal(result[observed],updated[observed])
    assert torch.equal(result[~observed],native[~observed])
    assert torch.equal(native,before_native) and torch.equal(updated,before_updated)


def test_r_only_mask_retains_other_correction_and_does_not_touch_inside():
    native=torch.zeros(1,2,3);updated=torch.full((1,2,3),3.)
    observed=torch.tensor([[True,False]]);residual=torch.ones_like(native)
    result=restrict_consumer(native,updated,observed,'mask-R',residual)
    assert torch.equal(result[observed],updated[observed])
    assert torch.equal(result[~observed],torch.full((1,3),2.))
    assert restrict_consumer(native,updated,observed,'measure-only') is updated


def test_projection_control_rejects_missing_r_or_misaligned_mask():
    native=torch.zeros(1,2,3);observed=torch.tensor([[True,False]])
    with pytest.raises(ValueError,match='projected R'):
        restrict_consumer(native,native,observed,'mask-R')
    with pytest.raises(ValueError,match='aligned'):
        restrict_consumer(native,native,torch.ones(1,3,dtype=torch.bool),'mask-all')
