"""Separate SS search for useful semantic Top-K evidence, without exact-match rejection."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
from PIL import Image


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def temperature_calibration(rows, match_root, resolve):
    """Descriptor-score concentration only; no SS GT, labels or validation assets."""
    temperatures = [.005, .01, .02, .05, .1, .2]
    values = {t: [] for t in temperatures}
    used = []
    for row in rows[:64]:
        with np.load(match_root / (row['sample_id']+'.npz')) as z:
            logits = z['matching_logits'].astype(np.float64)
            valid = z['candidate_valid'].astype(bool)
        with Image.open(resolve(row['target_anchor'])) as im:
            alpha = np.asarray(im.convert('RGBA').resize((512,512),Image.Resampling.NEAREST))[8::16,8::16,3].reshape(-1)>127
        logits = logits[alpha & valid.any(-1)]
        valid = valid[alpha & valid.any(-1)]
        if not len(logits):
            continue
        used.append(row['sample_id'])
        for t in temperatures:
            score = np.where(valid, logits, -np.inf)
            p = np.exp((score-score.max(-1,keepdims=True))/t)
            p /= p.sum(-1,keepdims=True)
            entropy = -(p*np.log(np.clip(p,1e-300,1))).sum(-1)
            values[t].extend(zip(p.max(-1).tolist(), np.exp(entropy).tolist()))
    records = []
    for t, rows_at_t in values.items():
        a = np.asarray(rows_at_t)
        if not len(a):
            raise ValueError('calibration has no inference-valid queries')
        records.append({'temperature':t,'queries':len(a),
                        'mean_max_weight':float(a[:,0].mean()),
                        'mean_effective_candidates_entropy':float(a[:,1].mean())})
    selected = min(records,key=lambda r:abs(np.log(r['mean_effective_candidates_entropy']/4)))
    return {'status':'PASS_CONCENTRATION_CALIBRATION_ONLY','assets':used,'records':records,
            'selected_temperature':selected['temperature'],
            'selection':'closest mean entropy effective-candidate count to4 among fixed grid',
            'GT_or_exact_match_labels_used':False,'SS_quality_selection':False,
            'scope':'train-internal matching calibration, not evidence that softer attention improves generation'}


def prepare(a):
    source = a.previous_root/'ss-seed-confirmation-v1/configs/b3.json'
    base = json.loads(source.read_text())
    if a.root.exists() or a.work_root.exists():
        raise FileExistsError('use a new owned search namespace')
    a.root.mkdir(parents=True);a.work_root.mkdir(parents=True)
    cal_path = a.previous_work/'panels/head_calibration.jsonl'
    rows = [json.loads(s) for s in cal_path.read_text().splitlines() if s.strip()]
    args = base['dataset']['args']
    def resolve(value):
        for prefix, replacement in args['path_map'].items():
            if value==prefix or value.startswith(prefix+'/'):
                return Path(replacement+value[len(prefix):])
        return Path(value)
    cal = temperature_calibration(rows, Path(args['match_root']), resolve)
    cal.update(manifest=str(cal_path),manifest_sha256=hashlib.sha256(cal_path.read_bytes()).hexdigest())
    write_new(a.root/'attention-calibration.json',cal)
    specs = [('top1-reference','b2',.005),('soft-topk','b3',cal['selected_temperature'])]
    jobs = []
    for index,(name,arm,tau) in enumerate(specs):
        cfg = copy.deepcopy(base);run = a.root/'runs_ss64'/name;recipe = a.root/'configs'/(name+'.json')
        cfg.update(config=str(recipe),output_dir=str(run),default_output_dir=str(run),master_port=str(30410+index),
                   wandb_id='ss-semantic-topk-v1-'+name,wandb_name='SS semantic TopK '+name)
        for d in [cfg['dataset']['args'],cfg['validation']['dataset_args']]:
            d['transport_arm'] = arm
        cfg['trainer']['args']['deform_conditioner'].update(arm=arm,score_temperature=tau)
        cfg['experiment'].update(arm=name,recipe_path=str(recipe),release_status='pending-native-memory-proof',
                                 parent_campaign=str(a.previous_root),matching_topk=64,
                                 matching_temperature=tau,rejection=False,retargeter=False,
                                 comparison_scope='fresh originalSS4210, seed101 world2 micro48 accumulation1; Top1 versus calibrated softTop64',
                                 inference_gt_inputs=False)
        write_new(recipe,cfg)
        train_name='train-'+name
        jobs.append({'name':train_name,'resource':'exclusive','gpus':list(a.gpu_pairs[index]),'priority':0,
                     'requires':[str(recipe)],'outputs':[str(run/'completed.json')],
                     'command':[sys.executable,'-u','-m','experiments.deformnext_retargeting.matching_v2_ss',
                                '--mode','train','--root',str(a.root),'--config',str(recipe)]})
        for noise,gpu in [(0,a.gpu_pairs[index][0]),(1,a.gpu_pairs[index][1]),(2,a.gpu_pairs[index][0])]:
            output=a.root/'evaluations'/name/f'noise-{noise}'
            seed_map=a.previous_work/'panels'/('seed-map.json' if noise==0 else f'seed-map-extra-noise-{noise}.json')
            jobs.append({'name':f'final-{name}-noise{noise}','resource':'evaluation','gpu':gpu,'priority':1,
                'depends_on':[train_name]+([f'final-{name}-noise0'] if noise==2 else []),
                'requires':[str(seed_map)],'outputs':[str(output/'evaluation.json'),str(output/'panel.png')],
                'command':[sys.executable,'-u','-m','experiments.deformnext_retargeting.evaluate_native_checkpoint',
                    '--stage','ss64','--config-root',str(recipe.parent),'--arm-config',str(recipe),
                    '--checkpoint',str(run/'ckpts/denoiser_step0000480.pt'),
                    '--conditioner-checkpoint',str(run/'ckpts/deform_conditioner_step0000480.pt'),
                    '--step','480','--output',str(output),'--panel-file',str(a.previous_work/'panels/panel-128.json'),
                    '--seed-map-file',str(seed_map),'--generation-batch','8']})
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for job in jobs:job['source_commit']=commit
    write_new(a.work_root/'queue-jobs.json',jobs)
    write_new(a.root/'initial-search-contract.json',{
        'status':'CONFIGURED_NOT_TRAINED','source_commit':commit,'first_wave':[x[0] for x in specs],
        'full_scope':'semantic TopK model search and matching-decoder/score/correspondence contribution checks; first wave does not complete this scope',
        'initializer':base['trainer']['args']['continuation_ckpt'],'seed':101,'epochs':5,'updates':480,
        'global_batch':96,'world_size':2,'microbatch_per_rank':48,'accumulation':1,
        'training_population':'DeformNeXt world-v4 train9149,validation508; clean80k not reused',
        'evaluation':'fixed128,93source clusters,3distinct noise seeds,12steps CFG3 raw480; development comparison',
        'native_spatial_control':'native Pixal3D projection only; cached spatial_indices use GT depth and are prohibited',
        'full_model_not_preselected':True,'success_not_guaranteed':True,'old_results_unchanged':True})
    print(json.dumps({'status':'CONFIGURED_NOT_TRAINED','selected_temperature':cal['selected_temperature'],
                      'root':str(a.root),'jobs':len(jobs)}),flush=True)


def storage(a):
    ssh=['ssh','-S',a.ssh_control,'-p',a.ssh_port,'-o','BatchMode=yes','-o','ConnectTimeout=8']
    subprocess.run([*ssh,a.remote_host,'mkdir -p '+shlex.quote(a.remote_root)],check=True)
    while not (a.work_root/'STORAGE_STOP').exists():
        state={'time':time.time()}
        try:
            for source,target,extra in [(a.root,a.remote_root,[]),(a.work_root,a.remote_root+'/runtime',['--exclude=wandb/'])]:
                subprocess.run(['rsync','-a','--partial','--timeout=60','--exclude=*.tmp','--exclude=.*tmp*',
                    *extra,'-e',shlex.join(ssh),str(source)+'/',a.remote_host+':'+target+'/'],check=True)
            state['status']='synced'
        except subprocess.CalledProcessError as e:
            state.update(status='retrying',returncode=e.returncode)
        (a.work_root/'storage-state.json').write_text(json.dumps(state)+'\n')
        print(json.dumps(state),flush=True)
        for _ in range(24):
            if (a.work_root/'STORAGE_STOP').exists():break
            time.sleep(5)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['prepare','storage'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--work-root',type=Path,required=True)
    p.add_argument('--previous-root',type=Path);p.add_argument('--previous-work',type=Path)
    p.add_argument('--gpu-pairs',nargs=2,help='Two explicitly assigned disjoint physical-GPU pairs')
    for name in ['remote-host','remote-root','ssh-control','ssh-port']:p.add_argument('--'+name)
    a=p.parse_args()
    if a.mode=='prepare':
        if not a.gpu_pairs:raise ValueError('GPU pairs required')
        a.gpu_pairs=[x.split(',') for x in a.gpu_pairs]
        flat=[g for pair in a.gpu_pairs for g in pair]
        if any(len(pair)!=2 for pair in a.gpu_pairs) or len(set(flat))!=4 or any(not g.isdigit() for g in flat):
            raise ValueError('two disjoint explicit GPU pairs required')
        prepare(a)
    else:
        if any(getattr(a,n.replace('-','_')) is None for n in ['remote-host','remote-root','ssh-control','ssh-port']):
            raise ValueError('explicit remote storage arguments required')
        storage(a)


if __name__=='__main__':main()
