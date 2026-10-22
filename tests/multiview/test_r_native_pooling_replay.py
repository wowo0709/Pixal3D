import pytest
import torch

from experiments.deformnext_retargeting.probe_r_native_pooling_replay import compare_tensors


def test_exactness_is_not_replaced_by_a_tolerance():
    value = torch.tensor([1., 2.])
    changed = value.clone()
    changed[0] = torch.nextafter(changed[0], torch.tensor(2.))
    assert compare_tensors(value, value)['exact']
    actual = compare_tensors(value, changed)
    assert not actual['exact'] and actual['different_elements'] == 1
    assert actual['max_absolute'] > 0


def test_reject_invalid_repeat_contract():
    with pytest.raises(ValueError, match='shapes and dtypes'):
        compare_tensors(torch.ones(1), torch.ones(2))
    with pytest.raises(ValueError, match='nonfinite'):
        compare_tensors(torch.tensor([float('nan')]), torch.ones(1))
