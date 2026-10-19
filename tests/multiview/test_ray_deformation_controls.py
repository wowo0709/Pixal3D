import pytest
import torch

from pixal3d.modules import ray_deformation_query as module
from experiments.deformnext_retargeting.ray_deformation_controls import deformation_control


def conditioner():
    return module.RayDeformationQueryConditioner(arm='c2', deformation_dim=2048,
        key_dim=1024, value_dim=1024, query_dim=1024, hidden_dim=256, topk=64,
        implementation='r9-native-v5', score_temperature=.05,
        candidate_aggregation='top1', projection_mode='nearest')


def test_correspondence_control_moves_both_value_routes_with_exact_probabilities():
    torch.manual_seed(81)
    model = conditioner()
    torch.nn.init.normal_(model.ray_readout.output.weight, std=.01)
    valid = torch.tensor([[1,0,1,1,0,1,0], [0,1,0,0,0,0,0], [0,0,0,0,0,0,0]], dtype=torch.bool)
    args = (model.ray_readout, torch.randn(3,1024), torch.randn(3,1024),
            torch.randn(1,7,1024), torch.randn(1,7,1024), torch.randn(1,7,2048),
            torch.randn(3,2048), torch.zeros(3,dtype=torch.long),
            torch.arange(7).expand(3,-1), valid, torch.randn(3,7))
    before = module.geometry_readout(*args)
    events = []
    with deformation_control(model, 'wrong-value-address', events):
        changed = module.geometry_readout(*args)
    restored = module.geometry_readout(*args)
    assert torch.equal(before[2], changed[2])
    assert not torch.equal(before[0][0], changed[0][0])
    assert not torch.equal(before[1][0], changed[1][0])
    for a,b in zip(before,restored):
        assert torch.equal(a,b)
    for a,b in zip(before,changed):
        assert torch.equal(a[1:],b[1:])
    assert events[-1]['raw_Omega_values_reassigned'] == 4
    assert events[-1]['appearance_and_RAW_use_same_valid_slot_permutation']


def test_projection_intervention_retains_real_observations_and_restores_on_failure():
    torch.manual_seed(83)
    model = conditioner()
    projection = model.retargeter.deformation_projection
    torch.nn.init.normal_(projection.weight, std=.01)
    actual = torch.randn(3,2048)
    before = projection(actual)
    events = []
    with pytest.raises(RuntimeError):
        with deformation_control(model, 'DQ-projection-off', events):
            assert torch.count_nonzero(projection(actual)) == 0
            assert torch.count_nonzero(actual) > 0
            raise RuntimeError('restore hook on failed evaluation')
    assert torch.equal(before, projection(actual))
    assert events[0]['real_DQ_input_RMS'] > 0
    assert not events[0]['fake_zero_input_observations']


def test_decoder_control_changes_only_added_query_and_restores_function():
    model = conditioner()
    original = module.contextual_query
    query = torch.randn(2,1024)
    events = []
    with deformation_control(model, 'decoder-context-bypass', events):
        result = module.contextual_query(model.q2c.decoder, query,
                                        torch.randn(2,4,1024), torch.ones(2,4,dtype=torch.bool))
        assert result is query
    assert module.contextual_query is original
    assert events[0]['cached_D1_proposal_and_score_bank_retained']
