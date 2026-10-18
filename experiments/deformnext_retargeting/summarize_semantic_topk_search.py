"""Strict fixed-final semantic Top-K model/effect report, before seed confirmation."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

from .compare_ss_noise_panels import compare
from .extend_task_attention_search import ARMS
from .semantic_topk_search import write_new
from .summarize_ss_module_study import audit_training,audit_final_replay
from .semantic_topk_zero_replay import MODES,zero_paths,validate_zero


MODELS=['top1-reference','soft-topk',*ARMS]


def validate_evaluation(e):
    expected={'status':'PASS','stage':'ss64','step':480,'weights':'raw','sampling_steps':12,
              'guidance':3.,'generation_batch_size':8,'optimizer_updates_during_evaluation':0,
              'target_modality':'rgb','conditioning_mode':'edited-rgb','diagnostic_only':False}
    if any(e.get(k)!=v for k,v in expected.items()):raise ValueError('final SS protocol differs')
    if e.get('input_camera_ids')!=[8,9,10,11] or e.get('input_states')!=['M1','M0','M0','M0']:
        raise ValueError('four-view RGB conditioning differs')
    if len(e['per_asset'])!=128 or any(not x.get('target_voxels_sha256') for x in e['per_asset']):
        raise ValueError('fixed128 actual GT hashes missing')


def aggregate(records):
    sums={k:sum(x[k] for e in records for x in e['per_asset']) for k in ['tp','fp','fn']}
    tp,fp,fn=[sums[k] for k in ['tp','fp','fn']]
    return {'micro_voxel_iou':tp/(tp+fp+fn),'micro_voxel_f1':2*tp/(2*tp+fp+fn),'counts':sums}


def task_gradient_audit(root,name):
    cfg=json.loads((root/'configs'/(name+'.json')).read_text())
    task=cfg['trainer']['args']['deform_conditioner'].get('task_attention')
    if task is None:return {'scope':'static frozen matcher; no task head gradient expected'}
    ranks=[]
    for rank in range(2):
        p=root/'runs_ss64'/name/('runtime_batch.jsonl' if rank==0 else 'runtime_batch.rank1.jsonl')
        rows=[json.loads(s) for s in p.read_text().splitlines()]
        seen={}
        for branch in ['matching_decoder','matching_score','task_attention']:
            values=[r['conditioner'][branch+'_gradient_L1'] for r in rows if branch+'_gradient_L1' in r.get('conditioner',{})]
            if not values or not all(math.isfinite(x) and x>=0 for x in values):
                raise ValueError('task runtime gradient evidence missing/nonfinite: '+name+' '+branch)
            required={'matching_decoder':task['decoder'],'matching_score':task['score_prior'],'task_attention':True}[branch]
            if (max(values)>0)!=required:raise ValueError('actual task branch usage differs: '+name+' '+branch)
            seen[branch]={'records':len(values),'maximum_gradient_L1':max(values),'expected_active':required}
        ranks.append({'rank':rank,'actual_SS_loss_gradient':seen})
    return {'status':'PASS_ACTUAL_TWO_RANK_TASK_GRADIENTS','ranks':ranks,
            'scope':'Observed nonzero usage,not proof of quality contribution'}


def summarize(root,previous):
    records={};training={}
    initial=json.loads((previous/'ss-seed-confirmation-v1/final-seed101-report.json').read_text())['initial_native_iou']
    for name in MODELS:
        training[name]=audit_training(root,name,initial)
        records[name]=[json.loads((root/'evaluations'/name/f'noise-{n}/evaluation.json').read_text()) for n in range(3)]
        for e in records[name]:validate_evaluation(e)
        training[name]['final128_replays_epoch32']=audit_final_replay(training[name],records[name][0])
        training[name]['task_branch_gradients']=task_gradient_audit(root,name)
    for name in ['v1','c1']:
        label='previous-'+name
        records[label]=[json.loads((previous/'ss-seed-confirmation-v1/evaluations'/name/f'noise-{n}/evaluation.json').read_text()) for n in range(3)]
        for e in records[label]:validate_evaluation(e)
    training['previous-c1']=audit_training(previous/'ss-seed-confirmation-v1','c1',initial)
    training['previous-c1']['final128_replays_epoch32']=audit_final_replay(training['previous-c1'],records['previous-c1'][0])
    zero=json.loads((root/'zero-shot-replay-summary-v1.json').read_text())
    if zero['status']!='PASS_ACTUAL_ZERO_SHOT_REPLAYS_AND_GT_HASHES':raise ValueError('zero-shot replay incomplete')
    for name,mode in MODES.items():
        records[name]=[json.loads(p.read_text()) for p in zero_paths(root,previous,name)]
        for e in records[name]:validate_zero(e,mode)
    contrasts={}
    pairs=[(candidate,baseline) for candidate in MODELS for baseline in ['previous-v1','top1-reference'] if candidate!=baseline]
    pairs += [('task-full','static-c1'),('task-full','task-no-decoder'),('task-full','task-no-prior'),
              ('task-full','task-native-spatial'),('task-full','previous-c1'),('static-c1','previous-c1')]
    pairs += [('previous-c1','previous-v1'),('previous-c1','top1-reference')]
    pairs += [(name,baseline) for name in MODELS for baseline in MODES]
    for candidate,baseline in pairs:
        contrasts[candidate+'-vs-'+baseline]=compare([(str(n),records[baseline][n],records[candidate][n]) for n in range(3)])
    report={'status':'PASS_SEVEN_ARM_FIXED_FINAL_DEVELOPMENT_SEARCH','models':{k:aggregate(v) for k,v in records.items()},
        'contrasts':contrasts,'training_audits':training,'training_seed':101,
        'full_scope_complete':False,'required_remaining':['selected-model additional training seed','actual final tensor/cache/AMP/CFG audit','actual remote logging/storage readback','final interpretation and model selection'],
        'scope':'Fixed raw480,development128/93source clusters/3noises;source-cluster CI conditional on one training seed;exploratory model search without multiplicity adjustment.',
        'native_spatial_control_scope':'Natural3camera spatial candidates versus64semantic candidates. Not fixed-K address-only causality.',
        'zero_shot_scope':zero['scope'],
        'original_results_unchanged':True,'positive_result_not_assumed':True}
    lines=['# Semantic Top-K SS fixed-final search','','| Model | Micro IoU | Micro F1 |','| --- | ---: | ---: |']
    for name,m in report['models'].items():lines.append(f"| {name} | {m['micro_voxel_iou']:.6f} | {m['micro_voxel_f1']:.6f} |")
    lines+=['','| Contrast | Gain IoU,pp | 95% source-cluster CI,pp |','| --- | ---: | --- |']
    for name,c in contrasts.items():
        lo,hi=c['paired_cluster_bootstrap_ci95'];lines.append(f"| {name} | {100*c['gain']:+.3f} | [{100*lo:+.3f},{100*hi:+.3f}] |")
    lines+=['',report['scope'],'',report['native_spatial_control_scope'],'',report['zero_shot_scope'],'','This is not completion of seed confirmation or model selection.']
    path=root/'seven-arm-final-report.json'
    if path.exists() or path.with_suffix('.md').exists():raise FileExistsError('preserve fixed-final report')
    write_new(path,report)
    with path.with_suffix('.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'status':report['status'],'models':report['models'],'scope_complete':False}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--previous-root',type=Path,required=True)
    a=p.parse_args();summarize(a.root,a.previous_root)


if __name__=='__main__':main()
