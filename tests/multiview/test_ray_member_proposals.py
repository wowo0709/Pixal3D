from types import SimpleNamespace

import torch
import pytest

from experiments.deformnext_retargeting.ray_member_proposals import (
    member_table,score_candidate_rows,observed_point_routes,maximum_member_raw64,mixture_member_top64,
)


def members():
    # Ray0 observes patches0/1. Repeated voxels for patch0 count once.
    rays=torch.tensor([[0,0,0,1],[0,0,0,-1],[-1,-1,-1,-1]])
    patches=torch.tensor([0,0,1,1]);valid=torch.zeros(1024,dtype=torch.bool);valid[:2]=True
    return member_table(rays,patches,valid)


def test_unique_members_and_input_alpha_exclusion():
    result=members()
    assert result['counts'][0]==2 and result['counts'][1]==1
    assert result['counts'][1024]==2 and result['counts'].sum()==5
    observed=torch.zeros(1024,dtype=torch.bool);observed[0]=True
    filtered=member_table(torch.tensor([[0,0,0,1],[0,0,0,-1],[-1,-1,-1,-1]]),torch.tensor([0,0,1,1]),observed)
    assert filtered['point_ids'].tolist()==[0] and filtered['counts'][0]==1 and filtered['counts'][1]==0


def test_mixture_weights_members_equally_and_retains_one_ray_distribution():
    result=members();source_valid=torch.zeros(3,1024,dtype=torch.bool);source_valid[0,:64]=True
    weights=torch.full((2,64),1e-12);weights[0,10]=.8;weights[0,20]=.2;weights[1,10]=.2;weights[1,30]=.8
    weights/=weights.sum(-1,keepdim=True)
    routes=dict(complete=True,candidates=torch.arange(64).repeat(2,1),weights=weights,context=torch.stack([torch.ones(1024),torch.ones(1024)*3]))
    mixed=mixture_member_top64(result,routes,source_valid)
    dense=torch.zeros(3072).scatter(0,mixed['candidates'][0],mixed['weights'][0])
    torch.testing.assert_close(dense[[10,20,30]],torch.tensor([.5,.1,.4]))
    torch.testing.assert_close(mixed['context'][0],torch.ones(1024)*2)
    assert mixed['active'].sum()==3 and mixed['weights'].shape==(3072,64)
    assert torch.equal(mixed['valid'][mixed['active']],torch.ones(3,64,dtype=torch.bool))
    assert not mixed['weights'][~mixed['active']].any()


def test_raw_union_respects_member_support_and_source_alpha():
    result=members();raw=torch.arange(3072).float().repeat(2,1)
    source_valid=torch.zeros(3,1024,dtype=torch.bool);source_valid[0,:100]=True
    raw[:,100:]=-torch.inf;raw[0,0]=1e5;raw[1,1]=2e5
    ids,valid,active=maximum_member_raw64(result,dict(complete=True,raw=raw),source_valid)
    assert ids[0,0]==1 and 0 in ids[0]
    assert ids[1,0]==1 and 0 not in ids[1]
    assert valid[active].all() and source_valid.flatten()[ids[active]].all()


def test_single_member_matches_original_point_posterior_support():
    torch.manual_seed(517)
    rays=torch.full((3,1),-1,dtype=torch.long);rays[0,0]=7
    alpha=torch.zeros(1024,dtype=torch.bool);alpha[3]=True
    table=member_table(rays,torch.tensor([3]),alpha)
    matcher=SimpleNamespace(decoder=lambda query,keys:(query,keys),fusion_head=lambda query,keys,scalars,base:base)
    source=torch.randn(3,1024,1024);target=torch.randn(1024,1024);valid=torch.ones(3,1024,dtype=torch.bool)
    routes=observed_point_routes(matcher,target_keys=target,source_keys=source,source_valid=valid,members=table)
    mixed=mixture_member_top64(table,routes,valid)
    assert torch.equal(mixed['candidates'][7],routes['candidates'][0])
    torch.testing.assert_close(mixed['weights'][7],routes['weights'][0],atol=1e-7,rtol=1e-6)


def test_incomplete_source_support_abstains_and_never_calls_decoder():
    table=members();valid=torch.zeros(3,1024,dtype=torch.bool);valid.flatten()[:63]=True
    def forbidden(*args):raise AssertionError('matcher called without64 available source patches')
    matcher=SimpleNamespace(decoder=forbidden)
    routes=observed_point_routes(matcher,target_keys=torch.randn(1024,1024),source_keys=torch.randn(3,1024,1024),source_valid=valid,members=table)
    mixed=mixture_member_top64(table,routes,valid)
    assert not mixed['active'].any() and not mixed['weights'].any()


def test_non_normalized_point_evidence_is_rejected():
    table=members();routes=dict(complete=True,candidates=torch.arange(64).repeat(2,1),weights=torch.ones(2,64),context=torch.zeros(2,1024))
    with pytest.raises(ValueError,match='normalized'):
        mixture_member_top64(table,routes,torch.ones(3,1024,dtype=torch.bool))


def test_decoder_batches_only_active_queries_before_chunking():
    calls=[]
    def decoder(query,keys):
        calls.append(query.clone())
        return query,keys
    query=torch.full((17,1024),torch.nan)
    active=torch.tensor([1,8,14]);query[active]=torch.tensor([1.,2.,3.])[:,None]
    valid=torch.zeros(17,64,dtype=torch.bool);valid[active]=True
    matcher=SimpleNamespace(decoder=decoder,fusion_head=lambda q,k,s,b:b)
    probability,context,_=score_candidate_rows(matcher,query,torch.ones(3072,1024),
        torch.arange(64).repeat(17,1),valid,chunk_size=2)
    assert [len(part) for part in calls]==[2,1]
    torch.testing.assert_close(torch.cat(calls),query[active])
    torch.testing.assert_close(context[active],query[active])
    assert torch.isfinite(probability).all() and not probability[~valid.any(-1)].any()
