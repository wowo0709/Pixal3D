"""Hold the retargeter fixed when testing Top1 against soft Top-K evidence."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from .compare_ss_noise_panels import compare
from .semantic_topk_confirmation import make_jobs
from .semantic_topk_search import write_new
from .summarize_semantic_topk_search import aggregate,validate_evaluation
from .summarize_ss_module_study import audit_training,audit_final_replay


NAME='static-top1-c1'


def configuration(source,root):
    cfg=copy.deepcopy(source);path=root/'configs'/(NAME+'.json');run=root/'runs_ss64'/NAME
    spec=cfg['trainer']['args']['deform_conditioner']
    if spec['arm']!='c1' or spec.get('task_attention') is not None:
        raise ValueError('matched control must clone the static C1 recipe')
    spec.update(candidate_aggregation='top1',freeze_q2c=True)
    cfg.update(config=str(path),output_dir=str(run),default_output_dir=str(run),master_port='30455',
        wandb_id='ss-semantic-topk-v1-'+NAME,wandb_name='SS semantic TopK '+NAME)
    cfg['experiment'].update(arm=NAME,recipe_path=str(path),retargeter=True,
        candidate_aggregation='matching-score argmax only;no soft aggregation',
        matching_head_frozen=True,release_status='pending-native-and-two-rank-memory-proof',
        purpose='Isolate soft evidence versus Top1 while retaining identical C1 retargeter architecture,initialization and budget.')
    return cfg


def summarize(root,previous):
    seven=json.loads((root/'seven-arm-final-report.json').read_text())
    if seven['status']!='PASS_SEVEN_ARM_FIXED_FINAL_DEVELOPMENT_SEARCH':raise ValueError('seven-arm report incomplete')
    audit=audit_training(root,NAME,seven['training_audits']['static-c1']['initial_iou'])
    names=[NAME,'static-c1','task-full','task-no-decoder','task-no-prior','top1-reference']
    records={name:[json.loads((root/'evaluations'/name/f'noise-{n}/evaluation.json').read_text()) for n in range(3)] for name in names}
    records['previous-c1']=[json.loads((previous/'ss-seed-confirmation-v1/evaluations/c1'/f'noise-{n}/evaluation.json').read_text()) for n in range(3)]
    for rows in records.values():
        for e in rows:validate_evaluation(e)
    audit['final128_replays_epoch32']=audit_final_replay(audit,records[NAME][0])
    pairs=[(name,NAME) for name in ['static-c1','task-full','task-no-decoder','task-no-prior','previous-c1']]+[(NAME,'top1-reference')]
    contrasts={a+'-vs-'+b:compare([(str(n),records[b][n],records[a][n]) for n in range(3)]) for a,b in pairs}
    result={'status':'PASS_MATCHED_RETARGETER_TOP1_FIXED_FINAL','models':{k:aggregate(v) for k,v in records.items()},
        'contrasts':contrasts,'training_audit':audit,'full_scope_complete':False,
        'scope':'Static-C1 versus static-Top1-C1 holds retargeter and frozen matcher fixed;task-full comparison additionally changes learned attention/head paths. Conditional source-cluster CI on development seed101,not population training-seed CI.',
        'candidate_selection_and_seed_confirmation_required':True}
    write_new(root/'matched-retargeter-top1-report.json',result)
    print(json.dumps({'status':result['status'],'models':result['models'],'contrasts':contrasts}),flush=True)


def prepare(a):
    devices=a.gpus.split(',')
    if len(devices)!=2 or len(set(devices))!=2 or any(not g.isdigit() for g in devices):raise ValueError('explicit assigned GPU pair required')
    path=a.work_root/'queue-jobs.json';old=path.read_bytes();existing=json.loads(old)
    state=json.loads((a.work_root/'queue-state.json').read_text())
    baseline='confirm-seed202-train-top1-reference'
    if state[baseline]['status']!='pending':raise ValueError('cannot change an active baseline dependency')
    cfg=configuration(json.loads((a.root/'configs/static-c1.json').read_text()),a.root)
    jobs=make_jobs(a.root,a.previous_work,NAME,devices,[f'final-task-no-prior-noise{n}' for n in range(3)],101,prefix='')
    jobs.append({'name':'summarize-matched-retargeter-top1','resource':'cpu','priority':1,
        'depends_on':['summarize-seven-arm-semantic-topk-search',*[f'final-{NAME}-noise{n}' for n in range(3)]],
        'outputs':[str(a.root/'matched-retargeter-top1-report.json')],
        'command':[sys.executable,'-u','-m','experiments.deformnext_retargeting.matched_top1_retargeter',
            '--mode','summarize','--root',str(a.root),'--previous-root',str(a.previous_root)]})
    if {j['name'] for j in jobs}&{j['name'] for j in existing}:raise FileExistsError('matched Top1-C1 already queued')
    for job in existing:
        if job['name']==baseline:job['depends_on']=[f'final-{NAME}-noise{n}' for n in range(3)]
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for job in jobs:job['source_commit']=source
    write_new(a.root/'configs'/(NAME+'.json'),cfg)
    write_new(a.root/'matched-top1-retargeter-contract.json',{'status':'CONFIGURED_NOT_TRAINED','name':NAME,'source_commit':source,
        'full_goal_complete':False,'same_retargeter_as_static_C1':True,'same_initial_RNG_and_weights_required':True,
        'score_behavior':'Frozen matching-score Top1,not uniform;full C1retargeter remains trainable.',
        'execution':'Overlap native-spatial development arm;move still-pending seed202 baseline after this control.',
        'selection_requires':'Both seven-arm and matched-retargeter-Top1 fixed-final reports.'})
    with (a.work_root/'queue-jobs-before-matched-top1-retargeter.json').open('xb') as f:f.write(old)
    temp=path.with_suffix('.matched-top1.tmp');temp.write_text(json.dumps(existing+jobs,indent=2)+'\n');os.replace(temp,path)
    print(json.dumps({'status':'QUEUED_NOT_TRAINED','new_jobs':len(jobs),'total':len(existing+jobs)}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['prepare','summarize'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--previous-root',type=Path,required=True)
    p.add_argument('--work-root',type=Path);p.add_argument('--previous-work',type=Path);p.add_argument('--gpus')
    a=p.parse_args()
    if a.mode=='prepare':
        if a.work_root is None or a.previous_work is None or a.gpus is None:p.error('prepare requires work-root,previous-work,gpus')
        prepare(a)
    else:summarize(a.root,a.previous_root)


if __name__=='__main__':main()
