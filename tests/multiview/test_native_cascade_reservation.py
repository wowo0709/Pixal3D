from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.native_cascade_reservation import INNER,validate_envelope


def inputs(tmp_path,kind='case'):
    base=tmp_path/'canonical';inner_out=base/'cases'/'asset'
    worker=tmp_path/INNER[kind][0];inner_plan=base/'prelaunch'/'case-plan.json'
    document=dict(output=str(inner_out),phase='perceptual' if kind=='official' else 'case',
                  frozen_model='original NN',sampling_seed=20261020)
    hashes={str(worker):'a'*64,str(inner_plan):'b'*64}
    plan=dict(inner_kind=kind,python='/python',inner_command=['/python','-u',str(worker),'--plan',str(inner_plan),'--expected-self-sha','a'*64],
        source_SHA256={str(worker):'a'*64},inner_plan=str(inner_plan),inner_plan_SHA256='b'*64,
        inner_read_contract=deepcopy(document),canonical_local_root=str(base),owned_guard_root=str(base/'guards'),
        output=str(base/'guards'/'asset'),expected_inner_artifacts=[dict(path=str(inner_out/'audit.json'),
            receipt=str(inner_out/'file4-closure.json'),status=INNER[kind][1])])
    return plan,document,dict(physical_gpu='2',allowed_gpus=['0','1','2','3'],sha=lambda p:hashes[p])


@pytest.mark.parametrize('kind',['case','official'])
def test_wrapper_keeps_original_argument_vector_and_read_contract(tmp_path,kind):
    p,d,k=inputs(tmp_path,kind);before=deepcopy((p,d));assert validate_envelope(p,d,**k)==p['expected_inner_artifacts'];assert(p,d)==before


@pytest.mark.parametrize('error',['GPU','unauthorized','command','extra-argument','source','inner-plan','consumer-contract','status','overlap','outside'])
def test_rejects_changed_inputs_and_false_reservation_ownership(tmp_path,error):
    p,d,k=inputs(tmp_path)
    if error=='GPU':k['physical_gpu']='6'
    elif error=='unauthorized':k['allowed_gpus']=['0','1']
    elif error=='command':p['inner_command'][0]='/different/python'
    elif error=='extra-argument':p['inner_command'].append('--different-seed')
    elif error=='source':p['source_SHA256'][p['inner_command'][2]]='c'*64
    elif error=='inner-plan':p['inner_plan_SHA256']='c'*64
    elif error=='consumer-contract':p['inner_read_contract']['sampling_seed']=42
    elif error=='status':p['expected_inner_artifacts'][0]['status']='PASS'
    elif error=='overlap':p['output']=d['output']
    else:p['output']=str(tmp_path/'unowned')
    with pytest.raises(ValueError):validate_envelope(p,d,**k)


def test_CPU_summary_cannot_reserve_GPU_as_official_metric(tmp_path):
    p,d,k=inputs(tmp_path,'official');d['phase']='summary';p['inner_read_contract']=deepcopy(d)
    with pytest.raises(ValueError):validate_envelope(p,d,**k)
