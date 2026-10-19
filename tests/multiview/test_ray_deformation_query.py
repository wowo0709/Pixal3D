"""Physical query channels and a learnable geometry route beyond appearance means."""
import pytest
import torch

from pixal3d.modules.deform_conditioner import _Retargeter
from pixal3d.modules.ray_feature_attention import RayFeatureReadout
from pixal3d.modules.ray_deformation_query import DeformationQueryRetargeter,geometry_readout


@pytest.mark.parametrize('value_width',[1024,2048])
def test_real_deformation_channels_preserve_old_R_and_learn_without_fake_zero_input(value_width):
    torch.manual_seed(67)
    appearance_width=value_width+1024
    original=_Retargeter(appearance_width,value_width,256)
    torch.nn.init.normal_(original.net[-1].weight,std=.01)
    extended=DeformationQueryRetargeter(appearance_width,value_width,256)
    state=extended.state_dict();state.update(original.state_dict());extended.load_state_dict(state)
    appearance=torch.randn(3,appearance_width);deformation=torch.randn(3,2048)
    assert original.input_dim==appearance_width and extended.input_dim==appearance_width+2048
    assert not hasattr(original,'deformation_projection')
    assert torch.count_nonzero(deformation)>0
    before=original(appearance,.25)
    after=extended(torch.cat([appearance,deformation],-1),.25)
    assert torch.equal(before,after)
    after.square().mean().backward()
    assert extended.deformation_projection.weight.grad.abs().sum()>0
    for name,value in original.state_dict().items():assert torch.equal(value,extended.state_dict()[name])
    with pytest.raises(ValueError):extended(appearance,.25)


def test_geometry_retains_correspondence_information_when_appearance_candidates_are_identical():
    torch.manual_seed(73)
    head=RayFeatureReadout()
    native=torch.randn(1,1024);context=torch.randn(1,1024)
    keys=torch.randn(1,2,1024);values=torch.randn(1,1,1024).expand(1,2,-1)
    raw=torch.cat([keys,torch.randn(1,2,1024)],-1);target=torch.randn(1,2048)
    batch=torch.zeros(1,dtype=torch.long);indices=torch.tensor([[0,1]]);valid=torch.ones(1,2,dtype=torch.bool)
    args=(head,native,context,values,keys,raw,target,batch,indices,valid)
    delta_a,geometry_a,_=geometry_readout(*args,torch.tensor([[30.,-30.]]))
    delta_b,geometry_b,_=geometry_readout(*args,torch.tensor([[-30.,30.]]))
    assert torch.equal(delta_a,delta_b)
    assert not torch.allclose(geometry_a,geometry_b)
