"""Install pending-only cross-queue descriptors without interrupting NN children.

The two single-thread dispatchers are briefly parked at their end-of-loop
sleep, so neither can launch a stale in-memory descriptor during installation.
Their PIDs and child handles survive. No queue-state record is ever written.
"""
import ast
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


def require(value,message):
    if not value:raise ValueError(message)


def exact(a,b):
    return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)


def validate_proposal(main_before,aux_before,main_after,aux_after,*,changed_aux,new_main,aux_state):
    """No active, unrelated or numerical descriptor can be silently replaced."""
    require(len(set(changed_aux))==len(changed_aux)>0 and len(set(new_main))==len(new_main)>0,
        'explicit distinct pending replacements and new main stages required')
    for rows in [main_before,aux_before,main_after,aux_after]:
        require(len({r['name'] for r in rows})==len(rows),'unique job IDs required')
    require(len(main_after)==len(main_before)+len(new_main)
        and exact(main_after[:len(main_before)],main_before)
        and {r['name'] for r in main_after[len(main_before):]}==set(new_main)
        and not {r['name'] for r in main_before}&set(new_main),
        'main queue must be append-only with every previous descriptor exact')
    require([r['name'] for r in aux_before]==[r['name'] for r in aux_after]
        and set(changed_aux)<={r['name'] for r in aux_before},'same ordered auxiliary IDs required')
    for before,after in zip(aux_before,aux_after):
        if before['name'] not in changed_aux:
            require(exact(before,after),'unrelated auxiliary descriptor changed')
        else:
            require(aux_state.get(before['name'],{}).get('status')=='pending',
                'every replaced auxiliary phase must still be pending')
            require(before.get('depends_on',[])==after.get('depends_on',[]) and after['resource']=='cpu',
                'original dependency names must survive through actual CPU completion bridges')


def verify_dispatcher_source(path):
    tree=ast.parse(Path(path).read_text())
    sleeps=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and isinstance(n.func.value,ast.Name) and n.func.value.id=='time' and n.func.attr=='sleep']
    require(len(sleeps)==1 and len(sleeps[0].args)==1
        and isinstance(sleeps[0].args[0],ast.Constant) and sleeps[0].args[0].value==10,
        'verified dispatcher with one ten-second end-of-loop sleep required')
    require(any(isinstance(n,ast.While) and isinstance(n.body[-1],ast.Expr)
        and n.body[-1].value is sleeps[0] for n in ast.walk(tree)),
        'sleep must end the dispatch loop before fresh descriptors are read')


def atomic(path,data,tag):
    path=Path(path);temp=path.with_name(path.name+'.'+tag+'.tmp')
    with temp.open('xb') as stream:stream.write(data)
    os.replace(temp,path)


def owner(spec):
    command=spec['command'];source=Path(spec['source'])
    require(str(source) in command or '-m' in command
        and command[command.index('-m')+1].split('.')[-1]==source.stem,
        'declared dispatcher source must match its actual script/module command')
    p=psutil.Process(spec['PID'])
    require(p.create_time()==spec['birth'] and p.cmdline()==spec['command']
        and p.num_threads()==1 and p.status() not in [psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD],
        'exact single-thread dispatcher PID/birth/command required')
    return p


WATCHDOG=r'''
import json,os,pathlib,signal,sys,time,psutil
marker=pathlib.Path(sys.argv[1]);time.sleep(30)
value=json.loads(marker.read_text())
if value.get('needs_resume'):
 try:
  p=psutil.Process(value['PID'])
  if p.create_time()==value['birth'] and p.cmdline()==value['command'] and p.status()==psutil.STATUS_STOPPED:
   os.kill(p.pid,signal.SIGCONT)
   marker.with_suffix('.watchdog-resumed.json').write_text(json.dumps(dict(PID=p.pid,birth=p.create_time(),watchdog_resumed=True)))
 except psutil.NoSuchProcess:pass
'''


@contextmanager
def parked_dispatcher(spec,state_path,record_folder):
    verify_dispatcher_source(spec['source'])
    p=owner(spec);require(p.status()!=psutil.STATUS_STOPPED,'dispatcher was already stopped by another owner')
    marker=Path(record_folder)/(str(p.pid)+'-resume.json')
    marker.write_text(json.dumps(dict(spec,needs_resume=True)))
    watchdog=subprocess.Popen([sys.executable,'-c',WATCHDOG,str(marker)],start_new_session=True,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    stopped=False;started=None
    try:
        deadline=time.monotonic()+15
        while True:
            p=owner(spec)
            age=time.time()-Path(state_path).stat().st_mtime
            # This source has exactly one sleep, immediately after atomically
            # publishing state. A <3s age leaves >7s before stale tasks can run.
            if 0<=age<3 and Path('/proc',str(p.pid),'wchan').read_text().strip()=='hrtimer_nanosleep':
                started=time.monotonic();os.kill(p.pid,signal.SIGSTOP);stopped=True
                end=time.monotonic()+1
                while owner(spec).status()!=psutil.STATUS_STOPPED:
                    require(time.monotonic()<end,'dispatcher did not park promptly');time.sleep(.01)
                require(time.time()-Path(state_path).stat().st_mtime<4,
                    'dispatcher was not parked at the verified fresh-state sleep boundary')
                break
            require(time.monotonic()<deadline,'no verified dispatcher sleep boundary observed')
            time.sleep(.02)
        yield dict(PID=p.pid,birth=p.create_time(),only_dispatcher_signalled=True)
    finally:
        if stopped:
            try:
                p=owner(spec)
                if p.status()==psutil.STATUS_STOPPED:os.kill(p.pid,signal.SIGCONT)
                end=time.monotonic()+1
                while owner(spec).status()==psutil.STATUS_STOPPED:
                    require(time.monotonic()<end,'dispatcher did not resume promptly');time.sleep(.01)
            except psutil.NoSuchProcess:pass
        marker.write_text(json.dumps(dict(spec,needs_resume=False,
            parked_seconds=None if started is None else time.monotonic()-started)))
        if watchdog.poll() is None:watchdog.terminate()
        watchdog.wait()


def install(*,main_path,aux_path,main_before,aux_before,main_after,aux_after,
            main_dispatcher,aux_dispatcher,main_state,aux_state,changed_aux,new_main,
            old_outputs,record_folder):
    record_folder=Path(record_folder);record_folder.mkdir(parents=True,exist_ok=False)
    decode=lambda raw:json.loads(raw)
    validate_proposal(*map(decode,[main_before,aux_before,main_after,aux_after]),
        changed_aux=changed_aux,new_main=new_main,aux_state=decode(Path(aux_state).read_bytes()))
    # Only small metadata validation/file switches occur while parked: no SSH,
    # NN-file hashing, model construction or neural execution.
    observations=[];installed=False
    with ExitStack() as stack:
        observations.append(stack.enter_context(parked_dispatcher(main_dispatcher,main_state,record_folder)))
        observations.append(stack.enter_context(parked_dispatcher(aux_dispatcher,aux_state,record_folder)))
        require(Path(main_path).read_bytes()==main_before and Path(aux_path).read_bytes()==aux_before,
            'queues changed after the closed proposal was prepared')
        validate_proposal(*map(decode,[main_before,aux_before,main_after,aux_after]),
            changed_aux=changed_aux,new_main=new_main,aux_state=decode(Path(aux_state).read_bytes()))
        require(not any(Path(p).exists() for p in old_outputs),'pending source phase already has outputs')
        require(not set(new_main)&set(decode(Path(main_state).read_bytes())),
            'new stage IDs already exist in main state')
        try:
            atomic(aux_path,aux_after,'pending-claim-aux')
            atomic(main_path,main_after,'pending-claim-main')
            require(Path(main_path).read_bytes()==main_after and Path(aux_path).read_bytes()==aux_after,
                'actual installed descriptors differ from the closed proposal')
            installed=True
        except BaseException:
            if Path(main_path).read_bytes()==main_after:atomic(main_path,main_before,'pending-claim-main-rollback')
            if Path(aux_path).read_bytes()==aux_after:atomic(aux_path,aux_before,'pending-claim-aux-rollback')
            raise
    require(installed,'pending-only descriptors were not installed')
    result=dict(status='PASS_PENDING_ONLY_CROSS_QUEUE_INSTALL',dispatchers=observations,
        dispatcher_PIDs_preserved=True,NN_children_not_signalled=True,queue_state_files_not_written=True,
        previous_main_descriptors_exact=True,only_pending_auxiliary_descriptors_replaced=True,
        full_goal_complete=False)
    (record_folder/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
