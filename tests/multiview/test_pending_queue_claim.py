"""Actual dummy process/atomic-queue checks. No neural model is run."""
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil
import pytest

from experiments.deformnext_retargeting import pending_queue_claim as claim


def proposals():
    main=[dict(name='existing',resource='exclusive',command=['unchanged'])]
    aux=[dict(name='phase',resource='exclusive',depends_on=[],command=['original']),
         dict(name='unrelated',resource='exclusive',command=['preserve'])]
    after=deepcopy(aux);after[0].update(resource='cpu',command=['actual-completion-bridge'])
    return main,aux,main+[dict(name='promoted',resource='exclusive')],after


@pytest.mark.parametrize('damage',[None,'active','unrelated','old-main','new-name','duplicate','dependency'])
def test_only_pending_execution_descriptors_are_claimed(damage):
    m,a,n,b=proposals();states={'phase':dict(status='pending')}
    if damage=='active':states['phase']['status']='running'
    elif damage=='unrelated':b[1]['command']=['changed']
    elif damage=='old-main':n[0]=dict(n[0],command=['changed'])
    elif damage=='new-name':n[-1]['name']='existing'
    elif damage=='duplicate':b.append(deepcopy(b[0]))
    elif damage=='dependency':b[0]['depends_on']=['fabricated-success']
    if damage:
        with pytest.raises(ValueError):claim.validate_proposal(m,a,n,b,changed_aux=['phase'],new_main=['promoted'],aux_state=states)
    else:claim.validate_proposal(m,a,n,b,changed_aux=['phase'],new_main=['promoted'],aux_state=states)


DUMMY='''import json,pathlib,subprocess,sys,time
state=pathlib.Path(sys.argv[1]);value=json.loads(sys.argv[2])
while True:
 state.write_text(json.dumps(value))
 time.sleep(10)
'''


@pytest.fixture
def dispatchers(tmp_path):
    source=tmp_path/'dispatcher.py';source.write_text(DUMMY)
    processes=[];specs=[];states=[]
    for name,value in [('main',{}),('aux',{'phase':dict(status='pending')})]:
        state=tmp_path/(name+'-state.json');process=subprocess.Popen([sys.executable,str(source),str(state),json.dumps(value)])
        processes.append(process)
        deadline=time.monotonic()+3
        while not state.exists():
            assert time.monotonic()<deadline;time.sleep(.01)
        p=psutil.Process(process.pid)
        specs.append(dict(PID=p.pid,birth=p.create_time(),command=p.cmdline(),source=str(source)))
        states.append(state)
    yield specs,states,processes
    for process in processes:
        if process.poll() is None:
            os.kill(process.pid,signal.SIGCONT);process.terminate()
        process.wait()


def payloads(tmp_path):
    m,a,n,b=proposals();encode=lambda v:(json.dumps(v,indent=2)+'\n').encode()
    main=tmp_path/'main-jobs.json';aux=tmp_path/'aux-jobs.json'
    before_main,before_aux,after_main,after_aux=map(encode,[m,a,n,b])
    main.write_bytes(before_main);aux.write_bytes(before_aux)
    return dict(main_path=main,aux_path=aux,main_before=before_main,aux_before=before_aux,
        main_after=after_main,aux_after=after_aux,changed_aux=['phase'],new_main=['promoted'],
        old_outputs=[str(tmp_path/'never-started')],record_folder=tmp_path/'install')


def test_actual_processes_keep_pids_and_state_files_while_pending_jobs_are_installed(tmp_path,dispatchers):
    specs,states,processes=dispatchers;kw=payloads(tmp_path)
    before=[p.read_bytes() for p in states]
    result=claim.install(**kw,main_dispatcher=specs[0],aux_dispatcher=specs[1],main_state=states[0],aux_state=states[1])
    assert result['dispatcher_PIDs_preserved'] and result['queue_state_files_not_written']
    assert kw['main_path'].read_bytes()==kw['main_after'] and kw['aux_path'].read_bytes()==kw['aux_after']
    assert [p.read_bytes() for p in states]==before
    assert all(p.poll() is None and psutil.Process(p.pid).status()!=psutil.STATUS_STOPPED for p in processes)


@pytest.mark.parametrize('damage',['wrong-birth','became-running','already-output','queue-edited'])
def test_race_or_wrong_owner_aborts_and_resumes_dispatchers(tmp_path,dispatchers,damage):
    specs,states,processes=dispatchers;kw=payloads(tmp_path)
    if damage=='wrong-birth':specs[1]['birth']+=1
    elif damage=='became-running':states[1].write_text(json.dumps({'phase':dict(status='running')}))
    elif damage=='already-output':Path(kw['old_outputs'][0]).mkdir()
    else:kw['main_path'].write_bytes(b'[]')
    main,aux=kw['main_path'].read_bytes(),kw['aux_path'].read_bytes()
    with pytest.raises(ValueError):claim.install(**kw,main_dispatcher=specs[0],aux_dispatcher=specs[1],main_state=states[0],aux_state=states[1])
    assert kw['main_path'].read_bytes()==main and kw['aux_path'].read_bytes()==aux
    assert all(p.poll() is None and psutil.Process(p.pid).status()!=psutil.STATUS_STOPPED for p in processes)


def test_failure_between_file_switches_restores_both_queues(tmp_path,dispatchers,monkeypatch):
    specs,states,processes=dispatchers;kw=payloads(tmp_path);original=claim.atomic
    def fault(path,data,tag):
        if tag=='pending-claim-main':raise OSError('injected-main-switch-failure')
        return original(path,data,tag)
    monkeypatch.setattr(claim,'atomic',fault)
    with pytest.raises(OSError):claim.install(**kw,main_dispatcher=specs[0],aux_dispatcher=specs[1],main_state=states[0],aux_state=states[1])
    assert kw['main_path'].read_bytes()==kw['main_before'] and kw['aux_path'].read_bytes()==kw['aux_before']
    assert all(psutil.Process(p.pid).status()!=psutil.STATUS_STOPPED for p in processes)


def test_only_dispatcher_is_parked_and_another_process_keeps_running(tmp_path,dispatchers):
    specs,states,_=dispatchers;folder=tmp_path/'park';folder.mkdir()
    heartbeat=tmp_path/'heartbeat'
    code='import pathlib,sys,time\np=pathlib.Path(sys.argv[1])\nwhile True:\n p.write_text(str(time.monotonic()))\n time.sleep(.01)\n'
    child=subprocess.Popen([sys.executable,'-c',code,str(heartbeat)])
    try:
        deadline=time.monotonic()+2
        while not heartbeat.exists():
            assert time.monotonic()<deadline;time.sleep(.01)
        with pytest.raises(RuntimeError):
            with claim.parked_dispatcher(specs[0],states[0],folder):
                assert psutil.Process(specs[0]['PID']).status()==psutil.STATUS_STOPPED
                before=float(heartbeat.read_text());time.sleep(.05)
                assert float(heartbeat.read_text())>before
                raise RuntimeError('ensure-finally-resumes')
        assert psutil.Process(specs[0]['PID']).status()!=psutil.STATUS_STOPPED
    finally:child.terminate();child.wait()


def test_rejects_unverified_sleep_or_dispatch_after_sleep(tmp_path):
    source=tmp_path/'bad.py'
    for body in ['while True:\n time.sleep(1)\n','while True:\n time.sleep(10)\n dispatch()\n']:
        source.write_text(body)
        with pytest.raises(ValueError):claim.verify_dispatcher_source(source)
