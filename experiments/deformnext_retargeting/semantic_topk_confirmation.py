"""Overlap a fresh second-seed Top1 baseline with the last development arm."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from .semantic_topk_search import write_new


def seed_config(source,folder,name,seed):
    cfg=copy.deepcopy(source);recipe=folder/'configs'/(name+'.json');run=folder/'runs_ss64'/name
    cfg.update(config=str(recipe),output_dir=str(run),default_output_dir=str(run),training_seed=seed,
        master_port='30440',wandb_id=f'ss-semantic-topk-v1-seed{seed}-'+name,
        wandb_name=f'SS semantic TopK seed{seed} '+name)
    cfg['experiment'].update(training_seed=seed,recipe_path=str(recipe),arm=name,
        comparison_scope=f'Fresh seed{seed} confirmation; raw480/global96/world2/micro48/accumulation1',
        release_status='pending-native-and-two-rank-memory-proof')
    return cfg


def make_jobs(folder,work,name,devices,dependencies,seed,*,prefix=None):
    cfg=folder/'configs'/(name+'.json');run=folder/'runs_ss64'/name
    if prefix is None:prefix=f'confirm-seed{seed}-'
    training=prefix+'train-'+name
    jobs=[{'name':training,'resource':'exclusive','gpus':devices,'priority':2,'depends_on':dependencies,
        'requires':[str(cfg)],'outputs':[str(run/'completed.json')],
        'command':[sys.executable,'-u','-m','experiments.deformnext_retargeting.matching_v2_ss',
                   '--mode','train','--root',str(folder),'--config',str(cfg)]}]
    for n,gpu in [(0,devices[0]),(1,devices[1]),(2,devices[0])]:
        output=folder/'evaluations'/name/f'noise-{n}'
        seed_map=work/'panels'/('seed-map.json' if n==0 else f'seed-map-extra-noise-{n}.json')
        jobs.append({'name':prefix+f'final-{name}-noise{n}','resource':'evaluation','gpu':gpu,'priority':1,
            'depends_on':[training]+([prefix+f'final-{name}-noise0'] if n==2 else []),
            'requires':[str(seed_map)],'outputs':[str(output/'evaluation.json'),str(output/'panel.png')],
            'command':[sys.executable,'-u','-m','experiments.deformnext_retargeting.evaluate_native_checkpoint',
                '--stage','ss64','--config-root',str(cfg.parent),'--arm-config',str(cfg),
                '--checkpoint',str(run/'ckpts/denoiser_step0000480.pt'),'--conditioner-checkpoint',str(run/'ckpts/deform_conditioner_step0000480.pt'),
                '--step','480','--output',str(output),'--panel-file',str(work/'panels/panel-128.json'),
                '--seed-map-file',str(seed_map),'--generation-batch','8']})
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--work-root',type=Path,required=True)
    p.add_argument('--previous-work',type=Path,required=True);p.add_argument('--gpus',required=True)
    a=p.parse_args();devices=a.gpus.split(',')
    if len(devices)!=2 or len(set(devices))!=2 or any(not g.isdigit() for g in devices):raise ValueError('explicit assigned GPU pair required')
    folder=a.root/'seed-confirmation-v1';cfg=seed_config(json.loads((a.root/'configs/top1-reference.json').read_text()),folder,'top1-reference',202)
    path=a.work_root/'queue-jobs.json';old=path.read_bytes();existing=json.loads(old)
    jobs=make_jobs(folder,a.previous_work,'top1-reference',devices,[f'final-task-no-prior-noise{n}' for n in range(3)],202)
    if {j['name'] for j in jobs}&{j['name'] for j in existing}:raise FileExistsError('seed202 baseline already queued')
    write_new(folder/'configs/top1-reference.json',cfg)
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for j in jobs:j['source_commit']=source
    write_new(folder/'baseline-contract.json',{'status':'CONFIGURED_NOT_TRAINED','training_seed':202,
        'source_commit':source,'purpose':'Overlap future matched Top1 reference with native-spatial development run; candidate selected only after fixed-final seven-arm report.',
        'candidate_preselected':False,'same_noises_as_seed101':True,'training_seed_CI_claim':False,
        'candidate_and_final_selection_still_required':True})
    with (a.work_root/'queue-jobs-before-seed202-baseline.json').open('xb') as f:f.write(old)
    temp=path.with_suffix('.confirm.tmp');temp.write_text(json.dumps(existing+jobs,indent=2)+'\n');os.replace(temp,path)
    print(json.dumps({'status':'QUEUED_NOT_TRAINED','jobs':len(jobs),'total':len(existing+jobs),'seed':202}),flush=True)


if __name__=='__main__':main()
