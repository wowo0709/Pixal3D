"""Replay legacy zero-shot noise0 with explicit RGB protocol and actual GT hashes."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from .compare_ss_noise_panels import compare
from .semantic_topk_search import write_new


MODES={'zero-source-only':'source-only','zero-edited-rgb':'edited-rgb'}
SUMMARY_JOB='summarize-zero-shot-replay-for-topk'


def zero_paths(root,previous,name):
    return [root/'evaluations/zero-shot-replay-v1'/MODES[name]/'evaluation.json',
            *[previous/f'baseline-noise-confirmation/noise-{n}/{name}/evaluation.json' for n in [1,2]]]


def validate_zero(e,mode):
    expected={'status':'PASS','stage':'ss64','step':0,'weights':'raw','sampling_steps':12,
        'guidance':3.,'generation_batch_size':8,'optimizer_updates_during_evaluation':0,
        'conditioning_mode':mode,'target_modality':'rgb'}
    if any(e.get(k)!=v for k,v in expected.items()) or e.get('diagnostic_only',False):
        raise ValueError('zero-shot protocol differs')
    states=['M0']*4 if mode=='source-only' else ['M1','M0','M0','M0']
    if e.get('input_camera_ids')!=[8,9,10,11] or e.get('input_states')!=states:
        raise ValueError('zero-shot requires four distinct original RGB views including camera08')
    if len(e['per_asset'])!=128 or any(not r.get('target_voxels_sha256') for r in e['per_asset']):
        raise ValueError('zero-shot actual128 target hashes missing')


def exact_legacy_replay(fresh,legacy):
    if legacy.get('status')!='PASS' or legacy.get('optimizer_updates_during_evaluation')!=0:
        raise ValueError('legacy zero-shot record incomplete')
    keys=['sample_id','source_cluster_id','seed','tp','fp','fn']
    rows=[{r['sample_id']:tuple(r[k] for k in keys) for r in e['per_asset']} for e in [fresh,legacy]]
    if len(rows[0])!=128 or len(rows[1])!=128 or rows[0]!=rows[1]:
        raise ValueError('fresh zero-shot does not replay legacy noise0 counts')
    return True


def summarize(root,previous):
    result={}
    reference=[json.loads((root/'evaluations/top1-reference'/f'noise-{n}/evaluation.json').read_text()) for n in range(3)]
    for name,mode in MODES.items():
        paths=zero_paths(root,previous,name);rows=[json.loads(p.read_text()) for p in paths]
        for e in rows:validate_zero(e,mode)
        old=json.loads((previous/'evaluations/initializer'/mode/'evaluation.json').read_text())
        replay=exact_legacy_replay(rows[0],old)
        paired=compare([(str(n),reference[n],rows[n]) for n in range(3)])
        if paired['paired_target_hashes_verified']!=384:raise ValueError('zero-shot desired-state GT hashes incomplete')
        result[name]={'evaluations':[str(p) for p in paths],'new_noise0_exact_legacy_counts':replay,
            'all384_actual_target_hashes_equal_current_final_reference':True,
            'versus_current_top1_reference':paired,'legacy_noise0_record_unchanged':True}
    write_new(root/'zero-shot-replay-summary-v1.json',{'status':'PASS_ACTUAL_ZERO_SHOT_REPLAYS_AND_GT_HASHES','models':result,
        'scope':'Both originalSS4210 zero-shot RGB protocols;new noise0 metadata/hash record exactly replays legacy counts. Existing noises1/2 retained. All use desired M1 GT;source-only does not measure M0 reconstruction.'})
    print('PASS_ACTUAL_ZERO_SHOT_REPLAYS_AND_GT_HASHES',flush=True)


def prepare(a):
    devices=a.gpus.split(',')
    if len(devices)!=2 or len(set(devices))!=2 or any(not x.isdigit() for x in devices):raise ValueError('explicit assigned evaluation GPUs required')
    source=json.loads((a.previous_root/'ss-seed-confirmation-v1/configs/v1.json').read_text())
    if source['trainer']['args']['deform_conditioner'] is not None or source['dataset']['args']['transport_arm']!='v1':
        raise ValueError('zero-shot replay must use original native V1 structure')
    cfg=copy.deepcopy(source);recipe=a.root/'configs/zero-shot-reference.json'
    cfg.update(config=str(recipe));cfg['experiment'].update(arm='zero-shot-reference',recipe_path=str(recipe),deformation_optimizer_updates=0)
    path=a.work_root/'queue-jobs.json';old=path.read_bytes();existing=json.loads(old);jobs=[]
    for (name,mode),gpu,parent in zip(MODES.items(),devices,['top1-reference','soft-topk']):
        output=a.root/'evaluations/zero-shot-replay-v1'/mode
        jobs.append({'name':'zero-shot-replay-'+mode,'resource':'evaluation','gpu':gpu,'priority':0,
            'depends_on':['final-'+parent+'-noise1'],'requires':[str(recipe)],
            'outputs':[str(output/'evaluation.json'),str(output/'panel.png')],
            'command':[sys.executable,'-u','-m','experiments.deformnext_retargeting.evaluate_native_checkpoint',
                '--stage','ss64','--config-root',str(recipe.parent),'--arm-config',str(recipe),
                '--checkpoint',cfg['trainer']['args']['continuation_ckpt']['denoiser'],
                '--step','0','--conditioning-mode',mode,'--output',str(output),
                '--panel-file',str(a.previous_work/'panels/panel-128.json'),
                '--seed-map-file',str(a.previous_work/'panels/seed-map.json'),'--generation-batch','8']})
    jobs.append({'name':SUMMARY_JOB,'resource':'cpu','priority':1,
        'depends_on':[j['name'] for j in jobs]+[f'final-top1-reference-noise{n}' for n in range(3)],
        'outputs':[str(a.root/'zero-shot-replay-summary-v1.json')],
        'command':[sys.executable,'-u','-m','experiments.deformnext_retargeting.semantic_topk_zero_replay',
            '--mode','summarize','--root',str(a.root),'--previous-root',str(a.previous_root)]})
    if {j['name'] for j in jobs}&{j['name'] for j in existing}:raise FileExistsError('zero-shot replay already queued')
    for job in existing:
        if job['name']=='summarize-seven-arm-semantic-topk-search':job['depends_on'].append(SUMMARY_JOB)
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for job in jobs:job['source_commit']=source_commit
    write_new(recipe,cfg)
    write_new(a.root/'zero-shot-replay-contract-v1.json',{'status':'QUEUED_NOT_EVALUATED','source_commit':source_commit,
        'views':{'source-only':['M0_08','M0_09','M0_10','M0_11'],'edited-rgb':['M1_08','M0_09','M0_10','M0_11']},
        'GT':'same desired M1 target for both','old_noise0_records_unchanged':True,
        'schedule':'Use other half of each assigned pair whilefinalnoise2 runs;originalSS4210,no conditioner,no SGD.'})
    with (a.work_root/'queue-jobs-before-zero-shot-replay.json').open('xb') as f:f.write(old)
    temp=path.with_suffix('.zero-replay.tmp');temp.write_text(json.dumps(existing+jobs,indent=2)+'\n');os.replace(temp,path)
    print(json.dumps({'status':'QUEUED_NOT_EVALUATED','jobs':len(jobs),'total':len(existing+jobs)}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['prepare','summarize'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--previous-root',type=Path,required=True)
    p.add_argument('--work-root',type=Path);p.add_argument('--previous-work',type=Path);p.add_argument('--gpus');a=p.parse_args()
    if a.mode=='summarize':summarize(a.root,a.previous_root)
    else:
        if a.work_root is None or a.previous_work is None or a.gpus is None:p.error('prepare requires work-root,previous-work,gpus')
        prepare(a)


if __name__=='__main__':main()
