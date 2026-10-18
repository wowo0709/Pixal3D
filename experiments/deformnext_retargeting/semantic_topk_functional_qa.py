"""Queue actual saved-weight RGB/cache/BF16/CFG checks on assigned free GPUs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .semantic_topk_search import write_new
from .summarize_semantic_topk_search import MODELS


def job(root,base,name,seed,panel):
    label=f's{seed}-'+name;output=root/'functional-QA-v1'/label
    config=base/'configs'/(name+'.json');run=base/'runs_ss64'/name/'ckpts'
    checkpoint=run/'denoiser_step0000480.pt';conditioner=run/'deform_conditioner_step0000480.pt'
    return {'name':'functional-QA-'+label,'resource':'evaluation','priority':0,
        'depends_on':[(f'confirm-seed{seed}-' if seed!=101 else '')+'train-'+name],
        'requires':[str(p) for p in [config,checkpoint,conditioner,panel]],'outputs':[str(output/'summary.json')],
        'command':[sys.executable,'-u','-m','experiments.deformnext_retargeting.verify_ss_conditioning_contract',
            '--config',str(config),'--checkpoint',str(checkpoint),'--conditioner-checkpoint',str(conditioner),
            '--panel-file',str(panel),'--output',str(output)]}


def attention_probe_jobs(root,panel):
    """Complete missing final attention statistics on two already-trained controls."""
    jobs=[]
    for name in ['static-c1','task-full']:
        item=job(root,root,name,101,panel);output=root/'attention-statistics-v1'/name
        item.update(name='attention-statistics-s101-'+name,priority=2,
            depends_on=['train-task-no-decoder','train-'+name],outputs=[str(output/'summary.json')])
        item['command'][item['command'].index('--output')+1]=str(output)
        jobs.append(item)
    return jobs


def install(a):
    path=a.work_root/'queue-jobs.json';old=path.read_bytes();existing=json.loads(old)
    jobs=[job(a.root,a.root,name,101,a.panel_file) for name in [*MODELS,'static-top1-c1']]
    jobs.append(job(a.root,a.root/'seed-confirmation-v1','top1-reference',202,a.panel_file))
    if {j['name'] for j in jobs}&{j['name'] for j in existing}:raise FileExistsError('functionalQA already queued')
    all_names={j['name'] for j in jobs+existing}
    if any(d not in all_names for j in jobs for d in j['depends_on']):raise ValueError('functionalQA training dependency missing')
    for item in existing:
        if item['name']=='summarize-seven-arm-semantic-topk-search':
            item['depends_on'] += [j['name'] for j in jobs if j['name'].startswith('functional-QA-s101-')]
        if item['name']=='train-static-c1':item['depends_on'].append('functional-QA-s101-top1-reference')
        if item['name']=='train-task-full':item['depends_on'].append('functional-QA-s101-soft-topk')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for j in jobs:j['source_commit']=source
    write_new(a.root/'functional-QA-contract-v1.json',{'status':'QUEUED_NOT_EXECUTED','source_commit':source,
        'initial_jobs':[j['name'] for j in jobs],'scope':'Two actual active fixed-panel cases per saved model;fresh FP32 DINO native-token check plus cached/online current matcher,BF16,train/eval modes andCFG p0/p1/mixed. No SGD or model-quality claim.',
        'task_frozen_state_policy':'Allow trainable current matcher;frozen initializer state must remain exact. Cached old logits must not override current task head.',
        'remaining':'Selected candidate202 proof is added only after fixed-final model selection.'})
    with (a.work_root/'queue-jobs-before-functional-QA.json').open('xb') as f:f.write(old)
    temp=path.with_suffix('.functional.tmp');temp.write_text(json.dumps(existing+jobs,indent=2)+'\n');os.replace(temp,path)
    print(json.dumps({'status':'QUEUED_NOT_EXECUTED','new_jobs':len(jobs),'total':len(existing+jobs)}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--work-root',type=Path,required=True);p.add_argument('--panel-file',type=Path,required=True)
    install(p.parse_args())


if __name__=='__main__':main()
