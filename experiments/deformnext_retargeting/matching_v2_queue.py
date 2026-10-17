"""Persistent resource queue restricted to explicitly assigned GPU IDs.

Small matcher jobs may share with bounded evaluation. SS training is exclusive.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil


def gpu_ids(value):
    values = value.split(',')
    if len(values) != len(set(values)) or any(not x.isdigit() for x in values):
        raise ValueError('explicit distinct nonnegative physical GPU IDs required')
    return values


def alive(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def can_run(gpu, resource, state):
    occupants=[r.get('resource','exclusive') for r in state.values() if r['status']=='running' and r.get('gpu')==gpu]
    if resource=='cpu':return True
    if resource=='exclusive':return not occupants
    if resource=='light':return not any(r in {'exclusive','light'} for r in occupants)
    if resource=='evaluation':return not any(r in {'exclusive','evaluation'} for r in occupants)
    raise ValueError('unknown queue resource class')


def choose_gpu(job, allowed, state):
    resource=job.get('resource','exclusive')
    if resource=='cpu':return ''
    if 'gpu' in job and str(job['gpu']) not in allowed:raise ValueError('job GPU not authorized')
    free=[g for g in allowed if can_run(g,resource,state)]
    candidate=str(job.get('gpu',free[0] if free else ''))
    return candidate if candidate in free else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-root',type=Path,required=True);p.add_argument('--gpus',required=True)
    a=p.parse_args();allowed=gpu_ids(a.gpus);repo=Path(__file__).resolve().parents[2]
    state_path=a.work_root/'queue-state.json';state=json.loads(state_path.read_text()) if state_path.exists() else {}
    own={}
    while not (a.work_root/'QUEUE_STOP').exists():
        tasks=json.loads((a.work_root/'queue-jobs.json').read_text())
        for job in tasks:
            name=job['name'];record=state.setdefault(name,{'status':'pending'})
            record['resource']=job.get('resource','exclusive')
            if job.get('attached_pid') and record['status']=='pending':
                if str(job['gpu']) not in allowed:raise ValueError('attached job GPU not authorized')
                record.update(status='running',pid=job['attached_pid'],gpu=str(job['gpu']),started=time.time())
            if record['status']=='running' and not alive(record['pid']):
                process=own.pop(name,None);code=process.poll() if process is not None else None
                complete=all(Path(f).exists() for f in job.get('outputs',[]))
                record.update(status='completed' if complete and code in (None,0) else 'FAILED',returncode=code,finished=time.time())
                print(json.dumps({'job':name,**record}),flush=True)
        for job in sorted(tasks,key=lambda j:j.get('priority',10)):
            name=job['name'];record=state[name]
            if record['status']!='pending':continue
            unavailable=[dep for dep in job.get('depends_on',[]) if state.get(dep,{}).get('status') in {'FAILED','HELD'}]
            if unavailable:
                record.update(status='HELD',reason={'unavailable_dependencies':unavailable});continue
            gate=job.get('gate_file')
            if gate:
                if not Path(gate).exists():continue
                decision=json.loads(Path(gate).read_text())
                if decision.get('status')!='PASS':
                    record.update(status='HELD',reason=decision);continue
            if any(not Path(f).exists() for f in job.get('requires',[])):continue
            if all(Path(f).exists() for f in job.get('outputs',[])):
                record.update(status='completed',reused=True);continue
            resource=job.get('resource','exclusive')
            gpu=choose_gpu(job,allowed,state)
            if gpu is None:continue
            environment=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',
                PYTHONPATH=str(repo),ATTN_BACKEND='sdpa',NUMPY_MADVISE_HUGEPAGE='0',
                OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',WANDB__DISABLE_STATS='true',
                PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',MATCHING_V2_ALLOWED_GPUS=','.join(allowed))
            environment.pop('SSHPASS',None)
            log=a.work_root/(name+'.log')
            command=job['command']
            with log.open('ab',buffering=0) as stream:
                child=subprocess.Popen(command,cwd=repo,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            own[name]=child;record.update(status='running',pid=child.pid,gpu=gpu,started=time.time(),command=command,log=str(log))
            print(json.dumps({'job':name,**record}),flush=True)
        tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');os.replace(tmp,state_path)
        time.sleep(10)


if __name__=='__main__':main()
