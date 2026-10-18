"""Add static C1, task model and separated decoder/score/native-spatial controls."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .semantic_topk_search import write_new


ARMS=['static-c1','task-full','task-no-decoder','task-no-prior','task-native-spatial']


def configurations(base,root,temperature):
    recipes={}
    for index,name in enumerate(ARMS):
        c=copy.deepcopy(base);run=root/'runs_ss64'/name;path=root/'configs'/(name+'.json')
        c.update(config=str(path),output_dir=str(run),default_output_dir=str(run),master_port=str(30420+index),
                 wandb_id='ss-semantic-topk-v1-'+name,wandb_name='SS semantic TopK '+name)
        spec=c['trainer']['args']['deform_conditioner']
        spec.update(arm='c1',score_temperature=temperature)
        if name!='static-c1':
            spec.update(task_attention={'decoder':name!='task-no-decoder','score_prior':name!='task-no-prior','chunk_size':64},
                        correspondence_mode='native-spatial' if name=='task-native-spatial' else 'semantic')
            for d in [c['dataset']['args'],c['validation']['dataset_args']]:d['task_matching_features']=True
        c['experiment'].update(arm=name,recipe_path=str(path),release_status='pending-native-memory-and-task-gradient-proof',
            matching_temperature=temperature,rejection=False,retargeter=True,
            physical_deformation_query_channels=0,inference_gt_inputs=False,
            matching_decoder='trainable D1-Q2C' if name in ['task-full','task-no-prior','task-native-spatial'] else 'physical D0' if name=='task-no-decoder' else 'frozen D1-Q2C',
            matching_score_prior=name!='task-no-prior',
            score_prior_OFF_is_not_user_Top1_no_score_arm=True,
            correspondence='original native ProjGrid3 source views' if name=='task-native-spatial' else 'same frozen Omega semantic Top64 indices',
            native_spatial_scope='3 geometric view candidates versus64 semantic candidates; native module comparison,not fixed-K address-only intervention',
            task_matching_score_initial_head=c['experiment']['head_sha256'],
            matching_head_frozen=name=='static-c1')
        recipes[name]=c
    return recipes


def jobs_for(recipes,root,previous_work,pairs):
    prerequisites={
        'static-c1':[f'final-top1-reference-noise{n}' for n in range(3)],
        'task-full':[f'final-soft-topk-noise{n}' for n in range(3)],
        'task-no-decoder':[f'final-static-c1-noise{n}' for n in range(3)],
        'task-no-prior':[f'final-task-full-noise{n}' for n in range(3)],
        'task-native-spatial':[f'final-task-no-decoder-noise{n}' for n in range(3)]}
    jobs=[]
    for index,(name,c) in enumerate(recipes.items()):
        devices=pairs[index%2];run=root/'runs_ss64'/name;path=root/'configs'/(name+'.json');train='train-'+name
        jobs.append({'name':train,'resource':'exclusive','gpus':devices,'priority':2,
            'depends_on':prerequisites[name],'requires':[str(path)],'outputs':[str(run/'completed.json')],
            'command':[sys.executable,'-u','-m','experiments.deformnext_retargeting.matching_v2_ss',
                       '--mode','train','--root',str(root),'--config',str(path)]})
        for n,gpu in [(0,devices[0]),(1,devices[1]),(2,devices[0])]:
            output=root/'evaluations'/name/f'noise-{n}'
            seed=previous_work/'panels'/('seed-map.json' if n==0 else f'seed-map-extra-noise-{n}.json')
            jobs.append({'name':f'final-{name}-noise{n}','resource':'evaluation','gpu':gpu,'priority':1,
                'depends_on':[train]+([f'final-{name}-noise0'] if n==2 else []),
                'requires':[str(seed)],'outputs':[str(output/'evaluation.json'),str(output/'panel.png')],
                'command':[sys.executable,'-u','-m','experiments.deformnext_retargeting.evaluate_native_checkpoint',
                    '--stage','ss64','--config-root',str(path.parent),'--arm-config',str(path),
                    '--checkpoint',str(run/'ckpts/denoiser_step0000480.pt'),'--conditioner-checkpoint',str(run/'ckpts/deform_conditioner_step0000480.pt'),
                    '--step','480','--output',str(output),'--panel-file',str(previous_work/'panels/panel-128.json'),
                    '--seed-map-file',str(seed),'--generation-batch','8']})
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','work-root','previous-root','previous-work']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpu-pairs',nargs=2,required=True);a=p.parse_args()
    pairs=[x.split(',') for x in a.gpu_pairs];flat=[g for pair in pairs for g in pair]
    if any(len(x)!=2 for x in pairs) or len(set(flat))!=4 or any(not g.isdigit() for g in flat):raise ValueError('two disjoint assigned GPU pairs required')
    base=json.loads((a.previous_root/'ss-seed-confirmation-v1/configs/c1.json').read_text())
    cal=json.loads((a.root/'attention-calibration.json').read_text())
    # A second preregistered broad-evidence candidate; no GT/val score fitting.
    temperature=min(cal['records'],key=lambda r:abs(__import__('math').log(r['mean_effective_candidates_entropy']/8)))['temperature']
    recipes=configurations(base,a.root,temperature);jobs=jobs_for(recipes,a.root,a.previous_work,pairs)
    path=a.work_root/'queue-jobs.json';old=path.read_bytes();existing=json.loads(old)
    names={j['name'] for j in existing}
    if names&{j['name'] for j in jobs}:raise FileExistsError('task model search already queued')
    all_names=names|{j['name'] for j in jobs}
    if any(d not in all_names for j in jobs for d in j.get('depends_on',[])):raise ValueError('task search dependency missing')
    if any((a.root/'configs'/(name+'.json')).exists() for name in recipes):raise FileExistsError('preserve task model recipes')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for name,c in recipes.items():write_new(a.root/'configs'/(name+'.json'),c)
    for j in jobs:j['source_commit']=source
    write_new(a.root/'task-models-contract.json',{'status':'CONFIGURED_NOT_TRAINED','arms':list(recipes),'source_commit':source,
        'temperature':temperature,'selection':'closest calibration entropy-effectiveK to8,not SS quality selection',
        'actual_final_checkpoint_steps':480,'matching_decoder_prior_independent':True,
        'query_channels_physically_absent':True,'GT_depth_spatial_indices_prohibited':True,
        'candidate_shared_only_for_semantic_models':True,'native_spatial_candidates':3,
        'task_matching_head_initial_SHA':base['experiment']['head_sha256'],'task_matching_head_is_trainable':True,
        'remaining_after_seven_runs':'strict effect analysis and additional training-seed confirmation of any selected improvement; do not declare complete at queue insertion'})
    backup=a.work_root/'queue-jobs-before-task-models.json'
    with backup.open('xb') as f:f.write(old)
    tmp=path.with_suffix('.task.tmp');tmp.write_text(json.dumps(existing+jobs,indent=2)+'\n');os.replace(tmp,path)
    print(json.dumps({'status':'QUEUED_NOT_TRAINED','new_jobs':len(jobs),'new_total':len(existing+jobs),'temperature':temperature}),flush=True)


if __name__=='__main__':main()
