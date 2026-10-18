"""Matched quick32 epoch trends, including same-case zero-shot references.

These interim comparisons never replace fixed last-epoch128x3 selection.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

from .compare_ss_noise_panels import compare
from .semantic_topk_search import write_new
from .semantic_topk_zero_replay import MODES,zero_paths,validate_zero


def validate_epoch(row,step):
    records=row['quick_per_asset'];ids=row['quick_panel_ids']
    if row.get('step')!=step or row.get('weights')!='raw':raise ValueError('requested raw epoch differs')
    if len(ids)!=32 or len(set(ids))!=32 or [r['sample_id'] for r in records]!=ids:
        raise ValueError('epoch requires original ordered distinct32')
    if any(not r.get('target_voxels_sha256') or r.get('target_voxel_shape')!=[64,64,64] for r in records):
        raise ValueError('actual32 SS64 target hashes required')
    return records


def aggregate(records):
    counts={k:sum(r[k] for r in records) for k in ['tp','fp','fn']};tp,fp,fn=[counts[k] for k in ['tp','fp','fn']]
    return {'micro_voxel_iou':tp/(tp+fp+fn),'micro_voxel_f1':2*tp/(2*tp+fp+fn),'counts':counts}


def summarize(root,epoch):
    if epoch not in range(1,6):raise ValueError('planned SS epoch must be1..5')
    step=epoch*96;old=root.parent/'ss-seed-confirmation-v1';models={};records={};inputs={};initials={};previous={}
    specs=[('previous-'+arm,old,arm) for arm in ['v1','b3','c1']]
    specs += [(name,root,name) for name in ['top1-reference','soft-topk','static-c1','task-full']]
    for label,base,name in specs:
        path=base/'runs_ss64'/name/'validation.jsonl';history=[json.loads(s) for s in path.read_text().splitlines()]
        selected=[r for r in history if r['step']==step]
        if len(selected)!=1:raise ValueError('requested epoch is not complete/unique for '+label)
        row=selected[0];records[label]=validate_epoch(row,step);models[label]=aggregate(records[label])
        for output,key in [('micro_voxel_iou','voxel_iou'),('micro_voxel_f1','voxel_f1')]:
            if abs(models[label][output]-row['quick_metrics'][key])>1e-12:raise ValueError('logged voxel metric differs from actual counts')
        initials[label]=validate_epoch(next(r for r in history if r['step']==0),0)
        if label in ['static-c1','task-full']:
            before=next(r for r in history if r['step']==step-96);previous[label]=validate_epoch(before,step-96)
        cfgpath=base/'configs'/(name+'.json');cfg=json.loads(cfgpath.read_text())
        if cfg['validation']['quick_metrics']['sampling_steps']!=12:raise ValueError('quick sampler differs')
        inputs[label]={'validation_path':str(path),'selected_record':row,
            'selected_record_SHA256':hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest(),
            'config_path':str(cfgpath),'config_SHA256':hashlib.sha256(cfgpath.read_bytes()).hexdigest()}
    ids=[r['sample_id'] for r in records['static-c1']]
    if any([r['sample_id'] for r in rs]!=ids for rs in records.values()):raise ValueError('epoch model panels differ')
    for name,mode in MODES.items():
        path=zero_paths(root,root.parent,name)[0];e=json.loads(path.read_text());validate_zero(e,mode)
        byid={r['sample_id']:r for r in e['per_asset']};subset=[byid[sid] for sid in ids]
        records[name]=subset;models[name]=aggregate(subset)
        inputs[name]={'source_evaluation':str(path),'source_SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'scope':'Exact ordered32 subset of existing fixed128/noise0;no new inference. Same desired M1 GT.'}
    for rows in initials.values():
        for left,right in zip(rows,records['zero-edited-rgb']):
            if any(left[k]!=right[k] for k in ['sample_id','source_cluster_id','seed','tp','fp','fn','target_voxels_sha256','target_voxel_shape']):
                raise ValueError('native initial32 does not replay actual edited-RGB zero-shot')
    def paired(a,b):
        c=compare([('quick-noise0',{'per_asset':b,'sampling_steps':12,'guidance':3.},
            {'per_asset':a,'sampling_steps':12,'guidance':3.})])
        if c['assets']!=32 or c['paired_target_hashes_verified']!=32:raise ValueError('actual common32 GT pair incomplete')
        return c
    baselines=['previous-v1','previous-b3','previous-c1','top1-reference','soft-topk',*MODES]
    pairs=[(name,baseline) for name in ['static-c1','task-full'] for baseline in baselines]
    pairs += [('task-full','static-c1'),('zero-edited-rgb','zero-source-only')]
    contrasts={a+'-vs-'+b:paired(records[a],records[b]) for a,b in pairs}
    temporal={name:paired(records[name],before) for name,before in previous.items()}
    result={'status':'PASS_MATCHED_QUICK32_EPOCH_WITH_ZERO_REFERENCES','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'epoch':epoch,'step':step,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'models':models,'contrasts':contrasts,'within_run_previous_epoch':temporal,'inputs':inputs,'sample_ids':ids,
        'zero_edited_exact_all_native_initial_records':True,'new_inference_or_training':False,'full_goal_complete':False,
        'scope':'Same32/one fixed sampling noise,matched raw epoch. Conditional exploratory source-cluster intervals,no multiple-comparison adjustment. Not final128x3 or population training-seed evidence.',
        'zero_scope':'Source-only uses4M0 RGB including camera08;edited uses1M1+3M0 RGB. Both use desiredM1 GT. Exact subset of actual completed128,not a new32generation run.',
        'temporal_scope':'Same run consecutive checkpoints;changes reflect ongoing joint optimization,not isolated module effects.',
        'selection_rule_unchanged':'Keep fixed last raw480/final128x3 and all predeclared candidate gates;do not select a best epoch from these tables.'}
    folder=root/f'wave2-epoch{epoch}-comparison-v2';write_new(folder/'report.json',result)
    lines=[f'# SS semantic matching epoch{epoch} quick32 comparison','',result['scope'],'',
        '| Model | Micro IoU | Micro F1 |','| --- | ---: | ---: |']
    for name,m in models.items():lines.append(f"| {name} | {m['micro_voxel_iou']:.6f} | {m['micro_voxel_f1']:.6f} |")
    lines+=['','| Contrast | IoU gain,pp | 95% source-cluster CI,pp |','| --- | ---: | --- |']
    for name,c in contrasts.items():
        lo,hi=c['paired_cluster_bootstrap_ci95'];lines.append(f"| {name} | {100*c['gain']:+.3f} | [{100*lo:+.3f},{100*hi:+.3f}] |")
    lines+=['',result['zero_scope'],'',result['temporal_scope'],'',result['selection_rule_unchanged']]
    with (folder/'report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'status':result['status'],'epoch':epoch,'models':models,
        'full_minus_static':contrasts['task-full-vs-static-c1'],'within_run_epoch_changes':temporal}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--epoch',type=int,choices=range(1,6),required=True);a=p.parse_args();summarize(a.root,a.epoch)


if __name__=='__main__':main()
