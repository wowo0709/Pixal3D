"""Do not silently reset or misassign the shared epoch1 Adam moments."""
import copy

import pytest
import torch

from experiments.deformnext_retargeting.anchor_joint_runtime import transplant_adam_prefix


def example():
    parameters = [torch.nn.Parameter(torch.zeros(2)) for _ in range(4)]
    optimizer = torch.optim.AdamW([
        dict(params=parameters[:2], model_name='default', lr=1e-5),
        dict(params=parameters[2:3], model_name='deform_conditioner', lr=1e-4),
        dict(params=parameters[3:], model_name='ray_composer', lr=1e-4)])
    source = copy.deepcopy(optimizer.state_dict())
    source['param_groups'] = source['param_groups'][:2]
    source['state'] = {i: dict(step=torch.tensor(96.), exp_avg=torch.full((2,), float(i+1)),
                               exp_avg_sq=torch.full((2,), float(i+2))) for i in range(3)}
    return parameters, optimizer, source


def test_common_moments_continue_and_new_group_starts_fresh():
    parameters, optimizer, source = example()
    assert transplant_adam_prefix(optimizer, source, expected_step=96) == 3
    assert parameters[3] not in optimizer.state
    for i, parameter in enumerate(parameters[:3]):
        assert torch.equal(optimizer.state[parameter]['exp_avg'], source['state'][i]['exp_avg'])
        assert optimizer.state[parameter]['step'] == 96
    assert [group['lr'] for group in optimizer.param_groups] == [1e-5, 1e-4, 1e-4]


@pytest.mark.parametrize('corruption', ['step', 'shape', 'group_order', 'unmapped'])
def test_rejects_incompatible_common_optimizer(corruption):
    _, optimizer, source = example()
    if corruption == 'step':
        source['state'][1]['step'] = torch.tensor(95.)
    elif corruption == 'shape':
        source['state'][1]['exp_avg'] = torch.zeros(3)
    elif corruption == 'group_order':
        source['param_groups'].reverse()
    else:
        source['state'][99] = copy.deepcopy(source['state'][0])
    with pytest.raises(ValueError):
        transplant_adam_prefix(optimizer, source, expected_step=96)
