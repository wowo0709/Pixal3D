"""Prepare matched seed101 V1/B3/C1 to confirm the original positive SS path.

Native identity/backward and per-rank memory gates precede each fresh run.
This follows the completed primary matrix; no historical run is resumed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ARMS = ('v1', 'b3', 'c1')
TRANSPORT = {'transport_arm','transport_implementation','match_root','projection_root','expected_q2c_sha256'}


def read(path):
    return json.loads(path.read_text())


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def make_config(saved, arm, root, work):
    cfg = copy.deepcopy(saved)
    output = root/'runs_ss64'/arm
    recipe = root/'configs'/(arm+'.json')
    cfg.update(config=str(recipe), default_output_dir=str(output), output_dir=str(output),
        load_dir='', ckpt='none', load_ckpt=None, num_gpus=2, num_nodes=1, node_rank=0,
        master_addr='127.0.0.1', master_port=str(30110+ARMS.index(arm)),
        training_seed=101, auto_retry=0, tryrun=False, profile=False, smoke_steps=None,
        use_wandb=True, wandb_project='pixal3d-deformnext-retargeting', wandb_entity='youngwoojeon-unist',
        wandb_id='ss-original-seed101-'+arm+'-v1', wandb_name='SS original '+arm+' matched seed101')
    cfg['trainer']['args'].update(max_steps=480, i_save=96, max_checkpoints=5,
        batch_size_per_gpu=48, batch_split=1)
    cfg['validation'].update(every_steps=96, max_assets=32)
    cfg['validation']['quick_metrics'].update(enabled=True, panel_size=32,
        panel_file=str(work/'panels/panel-32.json'), seed_map_file=str(work/'panels/seed-map.json'),
        sampling_steps=12, generation_batch_size=8)
    spec = cfg['trainer']['args'].get('deform_conditioner')
    if (arm == 'v1') != (spec is None) or cfg['dataset']['args']['transport_arm'] != arm:
        raise ValueError('source recipe is not the specified original arm')
    head = None
    if spec:
        if spec['implementation'] != 'r9-native-v5' or spec['arm'] != arm or spec.get('projection_mode','nearest') != 'nearest':
            raise ValueError('use the actual original R9 candidate, without changing its operator')
        head = cfg['dataset']['args']['expected_q2c_sha256']
    for key in ['pilot_steps','planned_total_steps','pilot_baseline_file']:
        cfg['experiment'].pop(key, None)
    cfg['experiment'].update(arm=arm, epochs=5, updates_per_epoch=96, global_batch=96, world_size=2,
        microbatch_per_gpu=48, head_sha256=head, validate_at_start=True, auto_continue_after_pilot=False,
        required_microbatch_per_gpu=48,
        release_status='matched-seed-confirmation-pending-native-memory-proof', recipe_path=str(recipe),
        training_seed=101, source_run_execution_changed=True,
        comparison_scope='same seed101/world2/micro48/accumulation1, V1 versus original B3 versus original C1')
    return cfg


def check_matched(configs):
    def core(cfg):
        result = {k: copy.deepcopy(cfg[k]) for k in ['models','dataset','trainer','validation','data_dir','training_seed','num_gpus']}
        for args in [result['dataset']['args'], result['validation']['dataset_args']]:
            for key in TRANSPORT: args.pop(key, None)
        result['trainer']['args'].pop('deform_conditioner', None)
        result['trainer']['args'].pop('model_learning_rates', None)
        return result
    reference = core(configs['v1'])
    if any(core(cfg) != reference for cfg in configs.values()):
        raise ValueError('matched data/model/optimizer/validation or execution settings differ')
    for arm,cfg in configs.items():
        args = cfg['trainer']['args']
        if (cfg['training_seed'],cfg['num_gpus'],args['batch_size_per_gpu'],args['batch_split'],args['max_steps']) != (101,2,48,1,480):
            raise ValueError('matched seed/world/global96/micro48/accumulation1 budget changed')
        if cfg['experiment'].get('required_microbatch_per_gpu')!=48:
            raise ValueError('matched micro48 may not silently fall back')
        rates = args.get('model_learning_rates', {})
        if rates != ({} if arm == 'v1' else {'deform_conditioner':1e-4}):
            raise ValueError('original conditioner learning rate changed')
        if cfg['experiment']['training_assets'] != 9149 or math.ceil(9149/96) != 96:
            raise ValueError('fixed DeformNeXt training population/budget differs')
    b3 = copy.deepcopy(configs['b3']['trainer']['args']['deform_conditioner'])
    c1 = copy.deepcopy(configs['c1']['trainer']['args']['deform_conditioner'])
    b3.pop('arm'); c1.pop('arm')
    if b3 != c1 or configs['b3']['experiment']['head_sha256'] != configs['c1']['experiment']['head_sha256']:
        raise ValueError('B3/C1 must use the same frozen matcher/operator')
    for key in ['dataset','validation']:
        field='args' if key=='dataset' else 'dataset_args'
        left=copy.deepcopy(configs['b3'][key][field]);right=copy.deepcopy(configs['c1'][key][field])
        left.pop('transport_arm');right.pop('transport_arm')
        if left!=right:raise ValueError('B3/C1 correspondence/projection caches differ')


def prepare(root, legacy, work):
    folder = root/'ss-seed-confirmation-v1'
    sources = {arm:legacy/('runs_world_v4' if arm == 'v1' else 'runs_r9_v5')/'ss64'/arm/'config_resolved.json' for arm in ARMS}
    configs = {arm:make_config(read(path),arm,folder,work) for arm,path in sources.items()}
    check_matched(configs)
    ready = folder/'configs-ready.json'
    if ready.exists() or any((folder/'configs'/(arm+'.json')).exists() for arm in ARMS):
        raise FileExistsError('preserve previous seed-confirmation recipes')
    for arm,cfg in configs.items(): write_new(folder/'configs'/(arm+'.json'),cfg)
    write_new(ready,{'status':'PASS_CONFIGS_ONLY','arms':list(ARMS),'seed':101,'global_batch':96,
        'world_size':2,'microbatch_per_rank':48,'accumulation':1,'updates':480,'epochs':5,
        'source_recipes':{arm:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for arm,path in sources.items()},
        'scope':'Original positive R9 path; fresh matched execution, native/backward/memory proof still required',
        'claim_training_or_performance_PASS':False})
    print(json.dumps({'status':'PASS_CONFIGS_ONLY','root':str(folder),'arms':list(ARMS)}),flush=True)


def build_jobs(root, work):
    folder=root/'ss-seed-confirmation-v1'
    configs={arm:read(folder/'configs'/(arm+'.json')) for arm in ARMS}
    check_matched(configs)
    jobs=[]
    for arm,devices,dependencies in [
        ('c1',[0,1],['diagnose32-original-c1-r9-wrong-object','diagnose32-c2-query-r-shuffled-deform']),
        ('v1',[2,3],['current-geometry-endpoint-adapter-only-s303-v2','current-geometry-endpoint-adapter-query-s303-v2']),
        ('b3',[0,1],['final128-seed101-c1-noise-1','final128-seed101-c1-noise-2']),
    ]:
        cfg=folder/'configs'/(arm+'.json');run=folder/'runs_ss64'/arm;train_name='train-ss-original-seed101-'+arm
        jobs.append({'name':train_name,'resource':'exclusive','gpus':devices,'priority':-3,
            'depends_on':dependencies,'requires':[str(cfg),str(folder/'configs-ready.json')],
            'outputs':[str(run/'completed.json')],
            'command':[sys.executable,'-u','-m','experiments.deformnext_retargeting.matching_v2_ss',
                '--mode','train','--root',str(folder),'--config',str(cfg)],
            'scope':'Fresh original R9 seed101 matched global96/DDP2/micro48/accumulation1. Native/backward and real2-update memory proof mandatory; never resume historical runs.'})
        for noise,device in [(0,devices[0]),(1,devices[1]),(2,devices[0])]:
            output=folder/'evaluations'/arm/f'noise-{noise}'
            seed_map=work/'panels'/('seed-map.json' if noise==0 else f'seed-map-extra-noise-{noise}.json')
            command=[sys.executable,'-u','-m','experiments.deformnext_retargeting.evaluate_native_checkpoint',
                '--stage','ss64','--config-root',str(cfg.parent),'--arm-config',str(cfg),
                '--checkpoint',str(run/'ckpts/denoiser_step0000480.pt'),'--step','480','--output',str(output),
                '--panel-file',str(work/'panels/panel-128.json'),'--seed-map-file',str(seed_map),'--generation-batch','8']
            if arm!='v1':command+=['--conditioner-checkpoint',str(run/'ckpts/deform_conditioner_step0000480.pt')]
            jobs.append({'name':f'final128-seed101-{arm}-noise-{noise}','resource':'evaluation','gpu':str(device),
                'priority':-3,'depends_on':[train_name]+([f'final128-seed101-{arm}-noise-0'] if noise==2 else []),
                'requires':[str(seed_map),str(work/'panels/panel-128.json')],
                'outputs':[str(output/name) for name in ['evaluation.json','panel.png','actual_inputs.png']],
                'command':command,'scope':'Same fixed-final128/three-noise sampler12 CFG3; no optimizer update or diagnostic perturbation.'})
    summary=folder/'final-seed101-report.json'
    jobs.append({'name':'summarize-original-SS-seed101-confirmation','resource':'cpu','priority':-3,
        'depends_on':[job['name'] for job in jobs if job['name'].startswith('final128-')],
        'outputs':[str(summary),str(summary.with_suffix('.md'))],
        'command':[sys.executable,'-u','-m','experiments.deformnext_retargeting.prepare_ss_seed_confirmation',
            '--mode','summarize','--root',str(root),'--work-root',str(work)]})
    return jobs


def append_queue(root, work):
    path=work/'queue-jobs.json';original=path.read_bytes();existing=json.loads(original);jobs=build_jobs(root,work)
    old={row['name'] for row in existing};new={row['name'] for row in jobs}
    if old&new or len(new)!=len(jobs):raise ValueError('seed confirmation was already queued')
    if any(dep not in old|new for job in jobs for dep in job.get('depends_on',[])):
        raise ValueError('seed confirmation dependency is not in this campaign')
    if any(not Path(p).exists() for job in jobs for p in job.get('requires',[]) if '/ckpts/' not in p):
        raise ValueError('fixed panels, seed maps or recipes are missing')
    backup=work/'queue-jobs-before-original-seed101-confirmation.json'
    if backup.exists():raise FileExistsError('preserve the previous seed-confirmation queue change')
    backup.write_bytes(original)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for job in jobs:job['source_commit']=commit
    tmp=path.with_suffix('.seed101.tmp');tmp.write_text(json.dumps(existing+jobs,indent=2)+'\n');os.replace(tmp,path)
    write_new(root/'provenance/original-SS-seed101-confirmation-queue.json',{'status':'QUEUED',
        'jobs':[job['name'] for job in jobs],'source_commit':commit,'previous_jobs':len(existing),
        'new_total_jobs':len(existing+jobs),'training_interrupted':False,'physical_GPU_allowlist':[0,1,2,3],
        'scope':'Three matched fresh seed101 runs to separate original C1 gain from execution/RNG changes; full model/PBR adoption remains separate.'})
    print(json.dumps({'status':'QUEUED','new_jobs':len(jobs),'new_total':len(existing+jobs)}),flush=True)


def summarize(root, work):
    import numpy as np
    from .compare_ss_noise_panels import compare
    from .summarize_ss_module_study import audit_training,audit_final_replay
    folder=root/'ss-seed-confirmation-v1';output=folder/'final-seed101-report.json'
    if output.exists() or output.with_suffix('.md').exists():raise FileExistsError('preserve the earlier seed101 report')
    configs={arm:read(folder/'configs'/(arm+'.json')) for arm in ARMS};check_matched(configs)
    primary=read(root/'module-study-native-v4/final-primary-report.json')
    initial=primary['training_audit']['c1-retarget-r']['initial_iou']
    audits={arm:audit_training(folder,arm,initial) for arm in ARMS}
    records={arm:[read(folder/'evaluations'/arm/f'noise-{noise}/evaluation.json') for noise in range(3)] for arm in ARMS}
    for arm,values in records.items():
        if any(row.get('status')!='PASS' or row.get('diagnostic_only') or row.get('r9_path_control','normal')!='normal'
               or row.get('transport_control','normal')!='normal' or row.get('conditioning_mode')!='edited-rgb'
               or row.get('optimizer_updates_during_evaluation')!=0 or row.get('sampling_steps')!=12 or row.get('guidance')!=3.
               or len(row['per_asset'])!=128 for row in values):
            raise ValueError('seed101 final generation protocol differs')
        for row in values:
            if any(not item.get('target_voxels_sha256') for item in row['per_asset']):
                raise ValueError('new seed101 results must carry actual target occupancy hashes')
        audits[arm]['final128_replays_epoch32']=audit_final_replay(audits[arm],values[0])
    contrasts={f'{candidate}-vs-{baseline}':compare([(str(noise),records[baseline][noise],records[candidate][noise])
        for noise in range(3)]) for candidate,baseline in [('c1','v1'),('b3','v1'),('c1','b3')]}
    models={}
    for arm,values in records.items():
        tp,fp,fn=np.array([[item[k] for k in ['tp','fp','fn']] for row in values for item in row['per_asset']]).sum(0)
        models[arm]={'micro_voxel_iou':float(tp/(tp+fp+fn)),'micro_voxel_f1':float(2*tp/(2*tp+fp+fn))}
    result={'status':'PASS_MATCHED_SS_SEED101_CONFIRMATION','training_seed':101,'arms':list(ARMS),
        'models':models,'contrasts':contrasts,'training_audits':audits,'initial_native_iou':initial,
        'scope':'Matched world2/micro48/accumulation1 confirmation. Source-cluster CIs conditional on this training seed. Historical seed0 used different execution; no pooled seed-population CI or production promotion.',
        'production_promotion':False,'PBR_or_Full_selection_complete':False}
    lines=['# Matched original SS seed101 confirmation','','Three fresh480-update runs; global96/DDP2/micro48/accumulation1. Fixed final128, three sampling noises,12steps/CFG3.','',
        '| Arm | Micro IoU | Micro F1 |','| --- | ---: | ---: |']
    for arm,row in models.items():lines.append(f"| {arm} | {row['micro_voxel_iou']:.6f} | {row['micro_voxel_f1']:.6f} |")
    lines+=['','| Contrast | Delta IoU, pp | 95% source-cluster CI, pp |','| --- | ---: | --- |']
    for name,row in contrasts.items():
        low,high=row['paired_cluster_bootstrap_ci95'];lines.append(f"| {name} | {100*row['gain']:+.3f} | [{100*low:+.3f}, {100*high:+.3f}] |")
    lines+=['',result['scope'],'','PBR/Full adoption remains an explicit review of these results and source-use controls.','']
    with output.with_suffix('.md').open('x') as stream:stream.write('\n'.join(lines))
    write_new(output,result);print(json.dumps({'status':result['status'],'models':models}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['prepare','queue','summarize'],default='prepare')
    for key in ['root','work-root']: parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--legacy-root',type=Path)
    args=parser.parse_args()
    if args.mode=='prepare':
        if args.legacy_root is None:parser.error('--legacy-root is required for prepare')
        prepare(args.root,args.legacy_root,args.work_root)
    elif args.mode=='queue':append_queue(args.root,args.work_root)
    else:summarize(args.root,args.work_root)


if __name__ == '__main__': main()
