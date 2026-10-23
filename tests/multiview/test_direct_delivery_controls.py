"""Consumer arithmetic and cancellation guards; no neural model on CPU."""
import pytest
import torch

from experiments.deformnext_retargeting.direct_delivery_controls import (
    controlled_residual, nearest_dense_consumer,
)


def test_controls_change_only_direct_coefficient_and_keep_original_normal_object():
    R = torch.tensor([[[.02, -.01]]])
    C = torch.tensor([[[.1, .04]]])
    gate = torch.tensor(.03)
    residual = R + C
    assert controlled_residual(residual, R, C, gate, 'normal') is residual
    assert controlled_residual(residual, R, C, gate, 'direct-C-zero') is R
    torch.testing.assert_close(controlled_residual(residual, R, C, gate, 'direct-C-fusion-gated'), R + gate*C)
    assert torch.equal(R, torch.tensor([[[.02, -.01]]]))


@pytest.mark.parametrize('control', ['normal', 'direct-C-zero', 'direct-C-fusion-gated'])
def test_false_R_decomposition_is_rejected_even_for_normal(control):
    R, C = torch.ones(1, 2, 3), torch.ones(1, 2, 3)
    with pytest.raises(ValueError, match='exactly reconstruct'):
        controlled_residual(R + C + .01, R, C, torch.tensor(.03), control)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_nearest_delivery_shares_a_patch_feature_across_ray_and_preserves_invalid_native(dtype):
    carrier = torch.tensor([[[1., 2.], [4., 8.]]])
    residual = torch.tensor([[[.1, -.2], [.7, .9]]])
    native = torch.zeros(1, 8, 2, dtype=dtype)
    anchor = torch.ones_like(native)
    grid = torch.tensor([[[[0, 1], [0, 1]], [[0, 1], [0, 1]]]])
    valid = torch.tensor([[True, False]])
    gate = torch.tensor(.03)
    actual = nearest_dense_consumer(native, anchor, carrier, residual, valid, gate, grid)
    expected_ray = gate.to(dtype)*((anchor[:, :1] + 3*carrier[:, :1].to(dtype))/4) + .75*residual[:, :1].to(dtype)
    assert torch.equal(actual[:, ::2], expected_ray.expand(1, 4, 2))
    assert torch.equal(actual[:, 1::2], native[:, 1::2])


def test_preserve_original_carrier_before_intervention_avoids_cancellation_change():
    # Actual floating-point subtraction is part of the historical contract.
    evidence = torch.tensor([[[1e8, 1.]]])
    original_residual = torch.tensor([[[1., 1e-7]]])
    carrier = evidence - original_residual
    R = torch.zeros_like(carrier)
    grid = torch.zeros(1, 1, 1, 1, dtype=torch.long)
    native = torch.zeros_like(carrier)
    actual = nearest_dense_consumer(native, native, carrier, R, torch.ones(1, 1, dtype=torch.bool), torch.tensor(1.), grid)
    assert torch.equal(actual, 3*carrier/4)
    assert not torch.equal((carrier + original_residual) - original_residual, evidence)


@pytest.mark.parametrize('mutation', ['shape', 'index', 'control', 'gate'])
def test_invalid_consumer_contract_rejected(mutation):
    z = torch.zeros(1, 2, 3)
    if mutation in {'control', 'gate'}:
        with pytest.raises(ValueError):
            controlled_residual(z, z, z, torch.tensor(float('nan') if mutation == 'gate' else .02),
                                'uniform' if mutation == 'control' else 'normal')
    else:
        with pytest.raises(ValueError):
            nearest_dense_consumer(z if mutation == 'shape' else z[:, :1], z[:, :1], z, z,
                torch.ones(1, 2, dtype=torch.bool), torch.tensor(.02),
                torch.full((1, 1, 1, 1), 2 if mutation == 'index' else 0))
