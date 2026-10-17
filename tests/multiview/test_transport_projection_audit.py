import torch
import pytest
import numpy as np
from pathlib import Path

from experiments.deformnext_retargeting.audit_native_transport_projection import normalized_evidence, DiagnosticDataset
from experiments.deformnext_retargeting.probe_source_target_balance import blend_projected
from experiments.deformnext_retargeting.compare_ss_noise_panels import compare


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
