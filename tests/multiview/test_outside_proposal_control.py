import torch

from experiments.deformnext_retargeting.outside_proposal_control import outside_addresses,outside_proposal_control
from pixal3d.modules.direct_ray_retargeting import DirectRayRetargetingConditioner


def test_donors_are_valid_outside_candidates_and_missing_donors_do_not_fake_intervention():
    source = torch.tensor([[1,0,1,1,1,0],[1,1,0,0,0,0]],dtype=torch.bool)
    indices = torch.tensor([[0,2],[0,1]])
    slots = torch.ones(2,2,dtype=torch.bool)
    donor,changed,available = outside_addresses(source,torch.tensor([0,1]),indices,slots)
    assert donor[0].tolist()==[3,4]
    assert changed.tolist()==[[True,True],[False,False]]
    assert torch.equal(donor[1],indices[1])
    assert available.tolist()==[2,0]


def test_full_direct_readout_changes_payload_with_exact_attention_and_restores_hooks():
    torch.manual_seed(109)
    model = DirectRayRetargetingConditioner(arm='c2',deformation_dim=2048,key_dim=1024,
        value_dim=1024,query_dim=1024,topk=64,implementation='r9-native-v5',
        score_temperature=.05,candidate_aggregation='top1',projection_mode='nearest')
    torch.nn.init.normal_(model.ray_readout.output.weight,std=.01)
    source = torch.randn(1,3,1024,2048)
    target = torch.randn(1,2,2048)
    pack = dict(source_values=torch.randn(1,3,1024,1024),
        source_valid=torch.ones(1,3,1024,dtype=torch.bool),target_query=torch.randn(1,2,1024),
        semantic_indices=torch.arange(64).reshape(1,1,64).expand(1,2,-1),
        matching_logits=torch.randn(1,2,64),candidate_valid=torch.ones(1,2,64,dtype=torch.bool),
        target_valid=torch.ones(1,2,dtype=torch.bool),target_keys=target[...,:1024],
        source_keys=source[...,:1024],target_context=target,source_context=source)
    probabilities=[]
    observer=model.ray_readout.register_forward_hook(lambda m,a,o:probabilities.append(o[1].clone()))
    events=[]
    with torch.no_grad():
        original=model(**pack)
        with outside_proposal_control(model,events):changed=model(**pack)
        restored=model(**pack)
    observer.remove()
    assert torch.equal(probabilities[0],probabilities[1])
    assert not torch.equal(original.evidence,changed.evidence)
    assert torch.equal(original.evidence,restored.evidence)
    assert events[0]['reassigned_outside_valid_slots']==128
    assert events[0]['queries_without_outside_source_patches']==0
    assert not model._forward_pre_hooks and not model.ray_readout.value._forward_pre_hooks
