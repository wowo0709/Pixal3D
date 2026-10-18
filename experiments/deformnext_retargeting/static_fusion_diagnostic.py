"""Same-weight static SS aggregation/gate interventions on a common32 panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from .compare_ss_noise_panels import compare
from .semantic_topk_search import write_new


CONTROLS={'soft-normal':'normal','top1-normal':'top1-feature',
    'soft-gate4':'fusion-x4','top1-gate4':'top1-feature-fusion-x4'}


def jobs(root,work):
    folder=root/'firstwave-fusion-dose-v1';config=root/'configs/soft-topk.json';ckpts=root/'runs_ss64/soft-topk/ckpts'
    tasks=[]
    for name,control in CONTROLS.items():
        output=folder/name
        tasks.append({'name':'firstwave-fusion-dose-'+name,'resource':'evaluation','priority':2,
            'depends_on':['train-static-c1','train-soft-topk'],
            'requires':[str(config),str(ckpts/'denoiser_step0000480.pt'),str(ckpts/'deform_conditioner_step0000480.pt'),
                str(work/'panels/panel-32.json'),str(work/'panels/seed-map.json')],
            'outputs':[str(output/'evaluation.json'),str(output/'panel.png')],
            'command':[sys.executable,'-u','-m','experiments.deformnext_retargeting.evaluate_native_checkpoint',
                '--stage','ss64','--config-root',str(config.parent),'--arm-config',str(config),
                '--checkpoint',str(ckpts/'denoiser_step0000480.pt'),'--conditioner-checkpoint',str(ckpts/'deform_conditioner_step0000480.pt'),
                '--step','480','--output',str(output),'--panel-file',str(work/'panels/panel-32.json'),
                '--seed-map-file',str(work/'panels/seed-map.json'),'--generation-batch','8','--r9-path-control',control],
            'scope':'Same soft raw480 saved weights/common32/noise0,no training. Run only on a free authorized GPU;does not replace main model selection.'})
    tasks.append({'name':'summarize-firstwave-fusion-dose','resource':'cpu','priority':2,
        'depends_on':[j['name'] for j in tasks],'outputs':[str(folder/'summary.json')],
        'command':[sys.executable,'-u','-m','experiments.deformnext_retargeting.static_fusion_diagnostic',
            '--mode','summarize','--root',str(root)]})
    return tasks


def summarize(records,reference):
    if len(reference['per_asset'])!=128 or set(records)!=set(CONTROLS):raise ValueError('four diagnostics and final128 reference required')
    actual={r['sample_id']:r for r in reference['per_asset']}
    if len(actual)!=128:raise ValueError('duplicate final-reference sample')
    gates={}
    for name,control in CONTROLS.items():
        e=records[name]
        expected={'status':'PASS','stage':'ss64','step':480,'weights':'raw','sampling_steps':12,'guidance':3.,
            'generation_batch_size':8,'optimizer_updates_during_evaluation':0,'target_modality':'rgb',
            'conditioning_mode':'edited-rgb','r9_path_control':control,'transport_control':'normal','task_value_control':'normal',
            'diagnostic_only':control!='normal'}
        if any(e.get(k)!=v for k,v in expected.items()):raise ValueError('fusion diagnostic protocol/control differs')
        for field in ['checkpoint','conditioner_checkpoint','arm_config','manifest_sha256']:
            if not e.get(field) or e[field]!=reference.get(field):raise ValueError('same saved model/data required')
        if len(e['per_asset'])!=32 or e.get('input_camera_ids')!=[8,9,10,11] or e.get('input_states')!=['M1','M0','M0','M0']:
            raise ValueError('common32 original four RGB input required')
        if control!='normal':
            events=e.get('r9_path_control_events',[])
            top1=control.startswith('top1-feature');dose=4 if control.endswith('x4') else 1
            if (not events or not all(x['mode']==control and x['optimizer_updates']==0 and x['pre_tanh_fusion_multiplier']==dose
                and x['actual_module_arm']==('b2' if top1 else 'b3') and x['selected_feature_count']==(1 if top1 else 64) for x in events)
                or sum(x['inference_valid_queries'] for x in events)<=0):
                raise ValueError('actual inference intervention events missing or wrong')
            before={x['original_fusion_gate'] for x in events};after={x['actual_fusion_gate'] for x in events}
            if len(before)!=1 or len(after)!=1:raise ValueError('fusion gate changed inside an evaluation')
            gates[name]={'original':next(iter(before)),'actual':next(iter(after)),
                'pre_tanh_multiplier':dose,'scope':'Multiply pre-tanh parameter,not exactly fourfold the bounded tanh gate.'}
            x=gates[name]
            if (not math.isfinite(x['original']) or not 0<abs(x['original'])<1
                or not math.isfinite(x['actual']) or abs(x['actual']-math.tanh(dose*math.atanh(x['original'])))>1e-6):
                raise ValueError('actual bounded gate does not match declared dose')
    if len({g['original'] for g in gates.values()})!=1:raise ValueError('different trained fusion gates across controls')
    for row in records['soft-normal']['per_asset']:
        if row['sample_id'] not in actual or any(row[k]!=actual[row['sample_id']][k] for k in ['source_cluster_id','seed','tp','fp','fn','target_voxels_sha256','target_voxel_shape']):
            raise ValueError('normal common32 must exactly replay the same-weight final128 subset')
    pairs=[('soft-normal','top1-normal'),('soft-gate4','top1-gate4'),('soft-gate4','soft-normal'),('top1-gate4','top1-normal')]
    contrasts={}
    for a,b in pairs:
        c=compare([('0',records[b],records[a])])
        if c['paired_target_hashes_verified']!=32:raise ValueError('actual paired32 GT hashes missing')
        contrasts[a+'-vs-'+b]=c
    models={}
    for name,e in records.items():
        total={k:sum(r[k] for r in e['per_asset']) for k in ['tp','fp','fn']};tp,fp,fn=[total[k] for k in ['tp','fp','fn']]
        models[name]={'micro_IoU':tp/(tp+fp+fn),'micro_F1':2*tp/(2*tp+fp+fn),'counts':total}
    return {'status':'PASS_COMPLETED_SAME_WEIGHT_AGGREGATION_FUSION_DIAGNOSTIC','models':models,'contrasts':contrasts,
        'actual_intervention_gates':gates,'normal_exact_final128_subset':True,'optimizer_updates':0,
        'scope':'One saved soft raw480 model/common32/one sampling noise. Isolates inference aggregation and gate dose,not a fresh-training ablation. Post-hoc exploratory source-cluster intervals;no candidate re-selection or population training-seed claim.',
        'if_dose_improves':'Headroom hypothesis only;train and fixed-final128x3 validate a revised model before adoption.',
        'if_dose_fails':'A larger fixed inference gate failed here;does not establish matching/learned-retargeter utility impossible.',
        'full_goal_complete':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['install','summarize'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--work-root',type=Path);p.add_argument('--previous-work',type=Path);a=p.parse_args()
    folder=a.root/'firstwave-fusion-dose-v1'
    if a.mode=='summarize':
        paths={name:folder/name/'evaluation.json' for name in CONTROLS};ref=a.root/'evaluations/soft-topk/noise-0/evaluation.json'
        result=summarize({name:json.loads(path.read_text()) for name,path in paths.items()},json.loads(ref.read_text()))
        result['evaluation_SHA256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [*paths.values(),ref]}
        write_new(folder/'summary.json',result);print(result['status'],flush=True);return
    if a.work_root is None or a.previous_work is None:p.error('install needs work-root and previous-work')
    path=a.work_root/'queue-jobs.json';old=path.read_bytes();existing=json.loads(old);new=jobs(a.root,a.previous_work)
    if {j['name'] for j in new}&{j['name'] for j in existing}:raise FileExistsError('fusion diagnosis already installed')
    all_names={j['name'] for j in existing+new}
    if any(d not in all_names for j in new for d in j['depends_on']):raise ValueError('missing training dependency')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for j in new:j['source_commit']=source
    write_new(folder/'contract.json',{'status':'QUEUED_NOT_EVALUATED','source_commit':source,'controls':CONTROLS,
        'saved_model':'soft-topk raw480','assets':32,'sampling_noise':0,'SGD_updates':0,
        'reason':'Fixed128 soft valuecosine nearTop1 and weak analytic gated delta motivate same-weight aggregation/dose checks. No causal amplitude claim before real generation results.',
        'scheduling':'priority2,evaluation-resource,any free authorized GPU after static-C1 completes;never overlaps an exclusive training allocation.'})
    with (a.work_root/'queue-jobs-before-fusion-dose-diagnostic.json').open('xb') as f:f.write(old)
    temp=path.with_suffix('.fusion.tmp');temp.write_text(json.dumps(existing+new,indent=2)+'\n');os.replace(temp,path)
    print(json.dumps({'status':'QUEUED_NOT_EVALUATED','new_jobs':len(new),'total':len(existing+new),'source':source}),flush=True)


if __name__=='__main__':main()
