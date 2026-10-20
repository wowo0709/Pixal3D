import torch
import pytest

from pixal3d.modules.r9_matching import R9Scorer
from pixal3d.modules.member_ray_posterior import member_ray_posteriors
from experiments.deformnext_retargeting.ray_member_proposals import (
    member_table,observed_point_routes,mixture_member_top64,
)


def frozen_matcher():
    torch.manual_seed(6243)
    matcher=R9Scorer(decoder_arm='D1-Q2C')
    with torch.no_grad():
        matcher.decoder.blocks[0].attention_gate.fill_(.4)
        matcher.decoder.blocks[0].ffn_gate.fill_(.2)
        matcher.fusion_head.output.weight.normal_(std=.01)
    return matcher.eval().requires_grad_(False)


def inputs():
    torch.manual_seed(9451)
    target=torch.randn(2,1024,1024);source=torch.randn(2,3,1024,1024)
    valid=torch.zeros(2,3,1024,dtype=torch.bool)
    valid[0].flatten()[:128]=True;valid[1].flatten()[:63]=True
    patch=torch.tensor([[7,7,9,11,11,12]]).repeat(2,1)
    ray=torch.tensor([[[0,0,0,1,1,2],[4,4,4,-1,-1,5],[-1,-1,-1,-1,-1,-1]]]).repeat(2,1,1)
    observed=torch.tensor([[True,True,True,False,False,True]]).repeat(2,1)
    return dict(target_keys=target,source_keys=source,source_valid=valid,
                ray_ids=ray,patch_ids=patch,observed_query=observed)


def test_production_posterior_matches_independent_CPU_prototype_and_abstains():
    matcher=frozen_matcher();x=inputs()
    before={k:v.clone() for k,v in matcher.state_dict().items()}
    calls=[]
    hook=matcher.decoder.register_forward_pre_hook(lambda _,args:calls.append(len(args[0])))
    actual=member_ray_posteriors(matcher,**x,chunk_size=2)
    hook.remove()
    assert calls==[2,1]
    alpha=torch.zeros(1024,dtype=torch.bool);alpha[[7,9,12]]=True
    table=member_table(x['ray_ids'][0],x['patch_ids'][0],alpha)
    routes=observed_point_routes(matcher,target_keys=x['target_keys'][0],source_keys=x['source_keys'][0],
        source_valid=x['source_valid'][0],members=table,chunk_size=2)
    expected=mixture_member_top64(table,routes,x['source_valid'][0])
    assert torch.equal(actual.candidate_indices[0],expected['candidates'])
    assert torch.equal(actual.candidate_valid[0],expected['valid'])
    assert torch.equal(actual.probabilities[0],expected['weights'])
    assert torch.equal(actual.contextual_query[0],expected['context'])
    assert torch.equal(actual.retained_mass[0],expected['retained_mass'])
    assert actual.distinct_member_count[0,0]==2
    assert actual.distinct_member_count[0,1]==0
    assert not actual.query_valid[1].any() and not actual.probabilities[1].any()
    assert not actual.probabilities.requires_grad
    assert all(torch.equal(v,before[k]) for k,v in matcher.state_dict().items())


def test_no_observed_queries_never_calls_matching_decoder():
    matcher=frozen_matcher();x=inputs();x['observed_query'].zero_()
    def forbidden(*_):raise AssertionError('empty observations reached matcher')
    hook=matcher.decoder.register_forward_pre_hook(forbidden)
    actual=member_ray_posteriors(matcher,**x)
    hook.remove()
    assert not actual.query_valid.any() and not actual.probabilities.any()
    assert not actual.contextual_query.any() and not actual.distinct_member_count.any()


def test_rejects_trainable_matching_head_in_frozen_proposal_study():
    matcher=frozen_matcher();matcher.fusion_head.requires_grad_(True)
    with pytest.raises(ValueError,match='must be frozen'):
        member_ray_posteriors(matcher,**inputs())
