from types import SimpleNamespace

import pytest
import torch

from experiments.deformnext_retargeting.probe_generated_r_pbr_support import output_bias_intervention


@pytest.mark.parametrize('fail', [False, True])
def test_engineering_R_bias_intervention_restores_exact_original_after_forward_error(fail):
    layer = torch.nn.Linear(3, 4)
    model = SimpleNamespace(retargeter=SimpleNamespace(net=[layer]))
    before = layer.bias.detach().clone()
    try:
        with output_bias_intervention(model):
            assert torch.equal(layer.bias, torch.ones_like(layer.bias))
            if fail:
                raise RuntimeError('synthetic proof forward error')
    except RuntimeError:
        assert fail
    assert torch.equal(layer.bias, before)
