import torch
import pytest
import numpy as np
from pathlib import Path

from experiments.deformnext_retargeting.audit_native_transport_projection import normalized_evidence, DiagnosticDataset
from experiments.deformnext_retargeting.probe_source_target_balance import blend_projected
from experiments.deformnext_retargeting.compare_ss_noise_panels import compare
from pixal3d.modules.transport_projection import support_weighted_projection
from pixal3d.modules.matching_v2_conditioner import MatchingV2Conditioner
from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
from experiments.deformnext_retargeting.matching_v2_queue import choose_gpu,can_run,dependencies_ready


def test_partial_support_does_not_dilute_transport_value_and_empty_support_is_finite():
    # A sample halfway between a valid constant feature and an invalid slot.
    def project(x):
        return (x[:, :1] + x[:, 1:2]) / 2
    values=torch.tensor([[[2.,4.],[100.,200.]],[[5.,7.],[8.,9.]]],requires_grad=True)
    valid=torch.tensor([[True,False],[False,False]])
    result,coverage=normalized_evidence(project,values,valid)
    torch.testing.assert_close(result[0,0],torch.tensor([2.,4.]))
    torch.testing.assert_close(coverage[:,0,0],torch.tensor([.5,0.]))
    assert torch.equal(result[1],torch.zeros_like(result[1]))
    result.sum().backward()
    assert torch.equal(values.grad[:,1],torch.zeros_like(values.grad[:,1]))
    assert torch.isfinite(values.grad).all()


def test_role_probe_keeps_source_views_and_zero_strength_is_native():
    anchor=torch.tensor([2.]);sources=torch.tensor([3.,5.,7.])
    native=(anchor+sources.sum())/4
    assert torch.equal(blend_projected(native,anchor,0),native)
    expected=.625*anchor+.125*sources.sum()
    torch.testing.assert_close(blend_projected(native,anchor,.5),expected)


def test_noise_repeats_stay_in_source_clusters_and_noise_is_paired():
    baseline={'per_asset':[{'sample_id':str(i),'source_cluster_id':'c'+str(i//2),
        'seed':i,'tp':10,'fp':5,'fn':5} for i in range(8)]}
    candidate={'per_asset':[{**r,'tp':r['tp']+int(r['sample_id'])%3} for r in baseline['per_asset']]}
    once=compare([('0',baseline,candidate)])
    repeated=compare([(str(i),baseline,candidate) for i in range(3)])
    assert repeated['source_clusters']==4
    assert repeated['sampling_noise_seeds']==3
    assert repeated['paired_cluster_bootstrap_ci95']==once['paired_cluster_bootstrap_ci95']
    candidate['per_asset'][0]['seed']=999
    with pytest.raises(ValueError,match='noise seed'):
        compare([('bad',baseline,candidate)])


def test_matched_teacher_support_preserves_learned_addresses(tmp_path):
    np.savez(tmp_path/'asset.npz',sample_id='asset',split='train',
        positive_indices=np.array([[3,4,5],[0,1,2]]),
        positive_visible=np.array([[True,False,False],[False,False,False]]))
    cache=tmp_path/'features.npz'
    np.savez(cache,conditioning_input_sha256='input',target_query=np.ones((2,1024),np.float16))
    indices=torch.tensor([[1,4],[2,5]])
    class Raw:
        rows=[dict(sample_id='asset',split='train',transport_cache=str(cache),conditioning_input_sha256='input')]
        def _resolve(self,path):return Path(path)
        def __getitem__(self,index):return {'transport_semantic_indices':indices.clone(),
            'transport_target_valid':torch.ones(2,dtype=torch.bool),'transport_source_valid':torch.zeros(1,6,dtype=torch.bool)}
    item=DiagnosticDataset(Raw(),tmp_path,oracle=False,teacher_support=True)[0]
    assert torch.equal(item['transport_semantic_indices'],indices)
    assert item['transport_target_valid'].tolist()==[True,False]
    assert item['transport_source_valid'].flatten().nonzero().flatten().tolist()==[3]


def test_bilinear_rejection_applies_to_entire_update_including_adapter():
    # The second patch is rejected despite a very large adapter residual.
    values=torch.tensor([[[2.],[1000.]]]);residual=torch.tensor([[[.4],[5000.]]])
    usage=torch.tensor([[.8,0.]])
    project=lambda x:x.mean(1,keepdim=True)
    source,adapter,valid,mass=support_weighted_projection(project,values,residual,usage)
    native=torch.tensor([[[1.]]]);anchor=torch.tensor([[[3.]]]);gate=.2
    update=(gate*((anchor+3*source)/4-native)+.75*adapter)*mass
    expected=.5*.8*(gate*((anchor+3*2)/4-native)+.75*.4)
    torch.testing.assert_close(update,expected)
    assert valid.all()
    _,_,valid,mass=support_weighted_projection(project,values,residual,usage*0)
    assert not valid.any() and not mass.any()


def test_reference_preaggregation_preserves_retargeter_initialization_rng():
    torch.manual_seed(92017)
    original=DeformationAwareConditioner(arm='c1',key_dim=1024,value_dim=1024,query_dim=1024)
    original_rng=torch.get_rng_state()
    torch.manual_seed(92017)
    cached=MatchingV2Conditioner(arm='c1',initialization_rng='r9-compatible')
    for k,v in original.retargeter.state_dict().items():
        assert torch.equal(v,cached.retargeter.state_dict()[k])
    assert torch.equal(original_rng,torch.get_rng_state())


def test_separate_projection_preserves_packed_value_and_gradient_math():
    values=torch.randn(2,4,5,requires_grad=True)
    residual=torch.randn(2,4,5,requires_grad=True)
    usage=torch.tensor([[1.,0.,.2,.5],[0.,0.,0.,0.]])
    project=lambda x:(x[:,:2]+x[:,2:])/2
    packed=project(torch.cat((values*usage[...,None],residual*usage[...,None],usage[...,None]),-1))
    mass=packed[...,-1:].clamp(0,1)
    reference=(packed[...,:5]+packed[...,5:10])/mass.clamp_min(1e-8)*mass
    source,adapter,_,actual_mass=support_weighted_projection(project,values,residual,usage)
    actual=(source+adapter)*actual_mass
    torch.testing.assert_close(actual,reference,rtol=0,atol=0)
    expected_grad=torch.autograd.grad(reference.sum(),(values,residual),retain_graph=True)
    actual_grad=torch.autograd.grad(actual.sum(),(values,residual))
    for a,b in zip(actual_grad,expected_grad):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_ddp_queue_reserves_all_assigned_devices_atomically():
    allowed=['8','9','10','11']
    job={'gpus':['10','11'],'resource':'exclusive'}
    assert choose_gpu(job,allowed,{})=='10,11'
    state={'ddp':{'status':'running','gpu':'10,11','resource':'exclusive'}}
    assert not can_run('10','evaluation',state) and not can_run('11','light',state)
    assert choose_gpu({'gpus':['9','10'],'resource':'exclusive'},allowed,state) is None
    assert choose_gpu({'gpus':['8','9'],'resource':'exclusive'},allowed,state)=='8,9'
    with pytest.raises(ValueError,match='authorized'):
        choose_gpu({'gpus':['7','8'],'resource':'exclusive'},allowed,{})


def test_dependencies_wait_for_completion_even_without_explicit_required_files():
    job={'depends_on':['fixed-final-head']}
    assert not dependencies_ready(job,{})
    assert not dependencies_ready(job,{'fixed-final-head':{'status':'running'}})
    assert dependencies_ready(job,{'fixed-final-head':{'status':'completed'}})
    assert not dependencies_ready(job,{'fixed-final-head':{'status':'HELD'}})
