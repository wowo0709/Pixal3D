import pytest
import torch

from experiments.deformnext_retargeting.predicted_support_guides import predicted_support_guides


def expand(coords, mask):
    row, bit = mask.nonzero(as_tuple=True)
    offset = torch.stack([bit % 2, bit // 2 % 2, bit // 4 % 2], dim=1)
    return coords[row, 1:]*2+offset


def test_preserves_unsorted_roots_dead_roots_and_exact_native_child_bit_order():
    roots=torch.tensor([[0,1,0,0],[0,0,0,0],[0,2,0,0]],dtype=torch.int32)
    leaves=torch.tensor([[31,15,15],[0,0,0],[16,0,0]],dtype=torch.int32)
    guides=predicted_support_guides(roots,leaves)
    assert torch.equal(guides[0][0],roots)
    assert torch.equal(guides[0][1],torch.tensor([[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]],dtype=torch.bool))
    assert torch.equal(expand(*guides[0]),torch.tensor([[2,0,0],[3,1,1],[0,0,0]],dtype=torch.int32))
    assert torch.equal(expand(*guides[-1]),torch.tensor([[16,0,0],[31,15,15],[0,0,0]],dtype=torch.int32))
    for (a,m),(b,n) in zip(guides,guides[1:]):
        assert torch.equal(expand(a,m),b[:,1:])


def test_upper_grid_corner_and_xyz_child_axes():
    roots=torch.tensor([[0,63,63,63]],dtype=torch.int64)
    leaves=torch.tensor([[1023,1023,1023],[1022,1023,1023],[1023,1022,1023],[1023,1023,1022]],dtype=torch.int64)
    guide=predicted_support_guides(roots,leaves)
    assert torch.equal(guide[-1][1],torch.tensor([[0,0,0,1,0,1,1,1]],dtype=torch.bool))


@pytest.mark.parametrize('roots,leaves',[
    ([[0,0,0,0],[0,0,0,0]],[[0,0,0]]),
    ([[0,0,0,0]],[[0,0,0],[0,0,0]]),
    ([[0,0,0,0]],[[16,0,0]]),
    ([[1,0,0,0]],[[0,0,0]]),
    ([[0,0,0,0]],[[-1,0,0]]),
    ([[0,0,0,0]],[[1024,0,0]]),
])
def test_rejects_invalid_or_unreachable_predictions(roots,leaves):
    with pytest.raises(ValueError):
        predicted_support_guides(torch.tensor(roots,dtype=torch.int32),torch.tensor(leaves,dtype=torch.int32))


def test_rejects_noninteger_support():
    with pytest.raises(ValueError):
        predicted_support_guides(torch.zeros((1,4)),torch.zeros((1,3)))
