"""Do not attribute weight changes to a value-address intervention."""
import pytest
import torch

from pixal3d.modules.ray_feature_attention import RayFeatureReadout
from experiments.deformnext_retargeting.ray_feature_controls import readout_control, valid_value_permutation


@pytest.mark.parametrize('width',[1024,2048])
def test_value_permutation_preserves_attention_and_changes_only_transformed_values(width):
    torch.manual_seed(61)
    head = RayFeatureReadout(width)
    torch.nn.init.normal_(head.output.weight,std=.01)
    native,context = torch.randn(3,1024),torch.randn(3,1024)
    keys,values = torch.randn(3,7,1024),torch.randn(3,7,width)
    valid = torch.tensor([[1,0,1,1,0,1,0],[0,1,0,0,0,0,0],[0,0,0,0,0,0,0]],dtype=torch.bool)
    prior = torch.randn(3,7)
    arguments = native,context,keys,values,valid,prior
    baseline,weights = head(*arguments)
    events = []
    with readout_control(head,'wrong-value-address',events):
        changed,controlled_weights = head(*arguments)
    assert torch.equal(weights,controlled_weights)
    assert not torch.equal(baseline[0],changed[0])
    assert torch.equal(baseline[1:],changed[1:])
    assert events[0]['reassigned_valid_slots'] == 4
    after,after_weights = head(*arguments)
    assert torch.equal(after,baseline) and torch.equal(after_weights,weights)


def test_only_valid_value_multiset_moves_and_singletons_remain_fixed():
    values = torch.arange(6.).reshape(2,3,1)
    valid = torch.tensor([[1,0,1],[0,1,0]],dtype=torch.bool)
    moved,count = valid_value_permutation(values,valid)
    assert count == 2
    assert moved[:,:,0].tolist() == [[2.,0.,0.],[0.,4.,0.]]


def test_prior_control_retains_content_attention_and_hooks_restore_on_error():
    torch.manual_seed(17)
    head = RayFeatureReadout()
    args = (torch.randn(2,1024),torch.randn(2,1024),torch.randn(2,4,1024),
            torch.randn(2,4,1024),torch.ones(2,4,dtype=torch.bool),torch.tensor([[30.,0.,0.,0.]]).expand(2,-1))
    _,baseline = head(*args)
    events = []
    with pytest.raises(RuntimeError):
        with readout_control(head,'score-prior-off',events):
            _,changed = head(*args)
            assert not torch.equal(changed,baseline)
            raise RuntimeError('leave the intervention')
    _,restored = head(*args)
    assert torch.equal(restored,baseline)
    assert events[0]['content_attention_retained']
