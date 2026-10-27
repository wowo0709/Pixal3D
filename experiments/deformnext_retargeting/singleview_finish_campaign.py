"""Wait for the 508 original single-view cases, then run official table metrics."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def count(root, name):
    return len(list((root/'cases').glob('*/'+name)))


def alive(pid):
    p=Path(f'/proc/{pid}')
    if not p.exists():return False
    try:return (p/'stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:return False


def run(command, *, env, log):
    with Path(log).open('w') as f:
        done=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
    if done.returncode:
        raise RuntimeError(f'exit {done.returncode}: {command}; inspect {log}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();root=args.root
    repo=Path(__file__).resolve().parents[2]
    workers=[]
    for n in ('generation-workers.json','render-workers.json'):
        workers += json.loads((root/n).read_text())
    tail=[]
    for name in ('tail-generation-workers.json','tail-render-workers.json'):
        if (root/name).exists():
            tail += json.loads((root/name).read_text())
    assignment_all=json.loads((root/'assignment-all.json').read_text())
    expected={case['row']['sample_id'] for case in assignment_all}
    if len(expected)!=508:raise ValueError('all 508 explicit assignment IDs required')
    while True:
        generated=count(root,'geometry/generation.json')
        rendered=count(root,'case-audit.json')
        failures=count(root,'generation-failure.json')+count(root,'render-failure.json')
        dead=[]
        for worker in workers:
            if alive(worker['pid']):continue
            gpu=worker['gpu']
            assignment=Path(worker['assignment']) if gpu in (0,1,2,3) else root/f'render-assignment-gpu{gpu}.json'
            rows=json.loads(assignment.read_text())
            if gpu in (6,7):
                incomplete=any(not (root/'cases'/case['sample_id']/'case-audit.json').is_file()
                               for case in rows)
            else:
                incomplete=any(not (root/'cases'/case['row']['sample_id']/'geometry/generation.json').is_file()
                               for case in rows)
            if incomplete:dead.append((gpu,worker['pid']))
        tail_incomplete=[]
        for worker in tail:
            if alive(worker['pid']):continue
            assignment=json.loads(Path(worker['assignment']).read_text())
            if worker['gpu'] in (6,7):
                incomplete=any(not (root/'cases'/case['sample_id']/'case-audit.json').is_file()
                               for case in assignment)
            else:
                incomplete=any(not (root/'cases'/case['row']['sample_id']/'geometry/generation.json').is_file()
                               for case in assignment)
            if incomplete:tail_incomplete.append((worker['gpu'],worker['pid']))
        print(json.dumps(dict(generated=generated,rendered=rendered,failures=failures,
                              dead=dead,tail_incomplete=tail_incomplete)),flush=True)
        if failures:raise RuntimeError('case failure; inspect case failure reports')
        if generated==rendered==508:
            seen={p.parent.name for p in (root/'cases').glob('*/case-audit.json')}
            if seen!=expected:raise ValueError('complete case set differs from assignment')
            break
        if dead or tail_incomplete:raise RuntimeError('worker exited before its cases completed')
        time.sleep(30)
    deadline=time.monotonic()+120
    all_workers=workers+tail
    while any(alive(x['pid']) for x in all_workers) and time.monotonic()<deadline:time.sleep(2)
    if any(alive(x['pid']) for x in all_workers):raise RuntimeError('source workers have not drained')
    base=[sys.executable,'-u','-m','experiments.deformnext_retargeting.singleview_table_metrics']
    env=os.environ.copy();env['PYTHONPATH']=str(repo);env['CUDA_VISIBLE_DEVICES']=''
    run(base+['prepare','--root',str(root)],env=env,log=root/'prepare-metrics.log')
    metrics=root/'metrics';metric_jobs=[]
    for phase,gpu in (('perceptual','0'),('inception','1')):
        cmd=[sys.executable,'-u','-m','experiments.deformnext_retargeting.singleview_adapted_eval_entry',
             '--phase',phase,'--plan',str(metrics/'plans'/f'{phase}.json')]
        if phase=='inception':cmd+=['--output',str(metrics/'inception')]
        e=os.environ.copy();e.update(PYTHONPATH=str(repo),CUDA_VISIBLE_DEVICES=gpu,
                                     MATCHING_V2_ALLOWED_GPUS='0,1,2,3,6,7')
        log=(root/f'{phase}-metrics.log').open('w')
        metric_jobs.append((phase,subprocess.Popen(cmd,cwd=repo,env=e,stdout=log,stderr=subprocess.STDOUT),log))
    failed=[]
    for phase,proc,log in metric_jobs:
        code=proc.wait();log.close()
        if code:failed.append((phase,code))
    if failed:raise RuntimeError(f'official metric jobs failed: {failed}')
    run(base+['collect','--root',str(root)],env=env,log=root/'collect-metrics.log')
    print((metrics/'table-row.json').read_text(),flush=True)


if __name__=='__main__':main()
