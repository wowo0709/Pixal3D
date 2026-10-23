"""Selection arithmetic and module structure; no CPU neural forward."""
import pytest
import torch
from torch import nn
from types import SimpleNamespace

from pixal3d.modules.matched_projected_readout import MatchedProjectedReadout,selection_probabilities


def test_top1_uses_actual_winner_and_fixed_topk_keeps_nonuniform_matching_scores():
    prior=torch.tensor([[0.,2.,-torch.inf]])
    valid=torch.tensor([[True,True,False]])
    hard=selection_probabilities(prior,valid,'top1')
    weighted=selection_probabilities(prior,valid,'fixed-topk')
    assert torch.equal(hard,torch.tensor([[[0.,1.,0.]]]).expand(-1,4,-1))
    torch.testing.assert_close(weighted[0,0,:2],prior[0,:2].softmax(-1))
    assert not weighted[:,:,2].any() and weighted[0,0,1]>weighted[0,0,0]


@pytest.mark.parametrize('mode',['top1','fixed-topk'])
def test_projection_weights_are_shared_exactly_and_QK_utility_physically_absent(mode):
    with torch.device('meta'):
        full=SimpleNamespace(value=nn.Linear(1024,256,bias=False),
                             output=nn.Linear(256,1024,bias=False),value_width=1024)
        readout=MatchedProjectedReadout(full,mode=mode)
    assert readout.value is full.value and readout.output is full.output
    assert set(dict(readout.named_parameters()))=={'value.weight','output.weight'}
    assert not hasattr(readout,'query') and not hasattr(readout,'key') and not hasattr(readout,'score_selector')


def test_empty_candidate_row_is_rejected_instead_of_silent_uniform_fallback():
    with pytest.raises(ValueError):
        selection_probabilities(torch.zeros(1,3),torch.zeros(1,3,dtype=torch.bool),'fixed-topk')
