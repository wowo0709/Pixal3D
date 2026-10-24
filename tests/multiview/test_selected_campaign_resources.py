"""Actual CPU process ownership and disk transfer; no NN or final TEST work."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.deformnext_retargeting.native_run_disk_budget import DiskBudget
from experiments.deformnext_retargeting.selected_campaign_resources import (CampaignCaseBudget,
    geometry_owners,live,process_identity,ready_aliases,reserve_asset_group,token_processes,verify_queue_owner)


def models():
    return {'zero-source4':{'geometry_group':'zero-source4'},'zero-edited4':{'geometry_group':'zero-edited4'},
        'V1':{'geometry_group':'V1'},'C_DECODER':{'geometry_group':'C_DECODER'},'SELECTED':{'geometry_group':'V1'}}


def test_whole_asset_credit_keeps_fifth_shared_geometry_consumer_admissible(tmp_path):
    budget=DiskBudget(tmp_path/'reservations.json',workspace_bytes=10,free_bytes=lambda:1010)
    cases={name:dict(output=str(tmp_path/name),bytes=100) for name in models()}
    actual=process_identity();admission=reserve_asset_group(budget,'asset',cases=cases,output=tmp_path/'group',owner=actual)
    assert admission['admitted'] and admission['required_free_bytes']==1010
    value=json.loads(budget.path.read_text())['runs']['asset']
    assert value['checkpoint_budget_bytes']==1000 and len(value['cases'])==5
    # No producer is admitted alone with a consumer left outside the window.
    assert not reserve_asset_group(budget,'next',cases=cases,output=tmp_path/'next',owner=actual)['admitted']
    assert set(json.loads(budget.path.read_text())['runs'])=={'asset'}
    assert ready_aliases(models(),{},set())==['zero-source4']
    assert ready_aliases(models(),{}, {'zero-source4'})==[]
    assert ready_aliases(models(),{'zero-source4':True},{'zero-edited4'})==['V1','C_DECODER']
    assert 'SELECTED' in ready_aliases(models(),{'zero-source4':True,'V1':True},{'C_DECODER'})
    assert geometry_owners(models())['SELECTED']=='V1'


def actual_transfer(tmp_path,damage=None):
    path=tmp_path/'reservations.json';owner=process_identity();output=tmp_path/'case'
    budget=DiskBudget(path,workspace_bytes=10,free_bytes=lambda:1000)
    reserve_asset_group(budget,'asset',cases={'case':dict(output=str(output),bytes=100)},output=tmp_path/'group',owner=owner)
    credit=dict(case_name='case',group_name='asset',coordinator=owner)
    if damage=='wrong-owner':credit['coordinator']['birth']-=1
    if damage=='stale-group':
        value=json.loads(path.read_text());value['runs']['asset']['released']=True;path.write_text(json.dumps(value))
    if damage=='wrong-bytes':
        value=json.loads(path.read_text());value['runs']['asset']['cases']['case']['bytes']=101;path.write_text(json.dumps(value))
    script='''import json,os,sys
from experiments.deformnext_retargeting.selected_campaign_resources import CampaignCaseBudget,process_identity
path,output,credit=sys.argv[1:];b=CampaignCaseBudget(path,workspace_bytes=10,free_bytes=lambda:1000,credit=json.loads(credit))
r=b.reserve('case',checkpoint_budget_bytes=100,native_output=output,wrapper_output=output,owner=process_identity())
print(json.dumps(r))
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    child=subprocess.run([sys.executable,'-c',script,str(path),str(output),json.dumps(credit)],env=env,capture_output=True,text=True)
    return child,json.loads(path.read_text())


def test_real_direct_child_atomically_transfers_future_credit_without_double_reserving(tmp_path):
    child,value=actual_transfer(tmp_path)
    assert child.returncode==0,child.stderr
    runs=value['runs'];assert runs['asset']['checkpoint_budget_bytes']==0
    assert runs['asset']['cases']['case']['claimed'] is True
    assert runs['case']['checkpoint_budget_bytes']==100 and runs['case']['released'] is False
    assert runs['case']['transferred_from_campaign_asset']=='asset'
    assert json.loads(child.stdout)['actual_atomic_campaign_credit_transfer'] is True


@pytest.mark.parametrize('damage',['wrong-owner','stale-group','wrong-bytes'])
def test_mismatched_campaign_credit_cannot_transfer_or_create_a_case(tmp_path,damage):
    child,value=actual_transfer(tmp_path,damage)
    assert child.returncode!=0 and 'case' not in value['runs']
    assert value['runs']['asset']['checkpoint_budget_bytes']==200


def test_later_consumer_does_not_need_new_disk_admission_after_prior_case_bytes_write(tmp_path):
    # The two-byte group allowance compensates both the actual physical write
    # and the unchanged legacy case entry, which retains its entire raw budget.
    free=[610];b=DiskBudget(tmp_path/'r.json',workspace_bytes=10,free_bytes=lambda:free[0]);owner=process_identity()
    reserve_asset_group(b,'asset',cases={n:dict(output=str(tmp_path/n),bytes=100) for n in ['a','b','c']},output=tmp_path/'group',owner=owner)
    for name in ['a','b','c']:
        credit=dict(case_name=name,group_name='asset',coordinator=owner)
        # Use a real direct child. It reads the current measured free value.
        script='''import json,sys
from experiments.deformnext_retargeting.selected_campaign_resources import CampaignCaseBudget,process_identity
p,o,c,f=sys.argv[1:];b=CampaignCaseBudget(p,workspace_bytes=10,free_bytes=lambda:int(f),credit=json.loads(c))
b.reserve(o.split('/')[-1],checkpoint_budget_bytes=100,native_output=o,wrapper_output=o,owner=process_identity())
'''
        result=subprocess.run([sys.executable,'-c',script,str(b.path),str(tmp_path/name),json.dumps(credit),str(free[0])],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        (tmp_path/name).write_bytes(b'x'*100);free[0]-=100
    entries=json.loads(b.path.read_text())['runs']
    assert entries['asset']['checkpoint_budget_bytes']==0 and sum(v['checkpoint_budget_bytes'] for v in entries.values())==300


def bound_child(tmp_path,assigned):
    command=[sys.executable,'-c','import time;time.sleep(30)'];env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(assigned),MATCHING_V2_ALLOWED_GPUS=','.join(assigned))
    child=subprocess.Popen(command,env=env);identity=process_identity(child.pid)
    group=dict(dispatcher=process_identity(),jobs=str(tmp_path/'jobs.json'),states=str(tmp_path/'states.json'),
        assigned_GPUs=assigned,job_name='actual-reservation')
    Path(group['jobs']).write_text(json.dumps([dict(name=group['job_name'],command=command,gpus=assigned,resource='exclusive' if assigned else 'cpu')]))
    Path(group['states']).write_text(json.dumps({group['job_name']:dict(status='running',pid=child.pid,command=command,gpu=','.join(assigned),resource='exclusive' if assigned else 'cpu')}))
    return child,identity,group,command


@pytest.mark.parametrize('assigned',[['0','1','2','3'],['6','7'],[]])
def test_actual_dispatcher_child_command_environment_and_full_group_ownership(tmp_path,assigned):
    child,identity,group,command=bound_child(tmp_path,assigned)
    try:assert verify_queue_owner(group,identity=identity,command=command)['pid']==child.pid
    finally:child.terminate();child.wait()
    assert not live(identity)
    with pytest.raises(ValueError):verify_queue_owner(group,identity=identity,command=command)


@pytest.mark.parametrize('damage',['birth','state','command','descriptor','parent','GPU'])
def test_stale_files_cannot_substitute_for_actual_exclusive_dispatcher_ownership(tmp_path,damage):
    child,identity,group,command=bound_child(tmp_path,['6','7'])
    try:
        if damage=='birth':identity['birth']-=1
        elif damage=='parent':group['dispatcher']['birth']-=1
        elif damage=='descriptor':Path(group['jobs']).write_text('[]')
        elif damage=='GPU':group['assigned_GPUs']=['7']
        else:
            value=json.loads(Path(group['states']).read_text());value[group['job_name']]['status' if damage=='state' else 'command']='completed'
            Path(group['states']).write_text(json.dumps(value))
        with pytest.raises(ValueError):verify_queue_owner(group,identity=identity,command=command)
    finally:child.terminate();child.wait()


def test_actual_token_detects_a_descendant_that_started_a_different_session(tmp_path):
    token='test-owned-campaign-'+tmp_path.name;env=dict(os.environ,SELECTED_FINAL_CAMPAIGN_TOKEN=token,SELECTED_FINAL_WORKER_TOKEN=token+':case')
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],env=env,start_new_session=True)
    try:
        rows=token_processes(token);assert any(r['PID']==child.pid and r['session_ID']==child.pid for r in rows)
        assert any(r['PID']==child.pid for r in token_processes(token+':case',field='SELECTED_FINAL_WORKER_TOKEN'))
        assert not token_processes(token+'-different')
    finally:child.terminate();child.wait()
    assert not token_processes(token)
