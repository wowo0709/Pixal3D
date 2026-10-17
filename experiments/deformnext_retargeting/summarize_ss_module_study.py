"""Compile the complete fixed-final SS study; refuse partial runs or panels."""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from .compare_ss_noise_panels import compare
from .native_module_study import ARMS


def read(path):
    return json.loads(path.read_text())


def evaluation_paths(root,study,arm):
    return [study/'evaluations'/arm/'evaluation.json',
        *[study/f'noise-confirmation/noise-{n}/{arm}/evaluation.json' for n in [1,2]]]


def audit_training(study,arm,initial_iou):
    run=study/'runs_ss64'/arm;done=read(run/'completed.json');cfg=read(study/'configs'/(arm+'.json'))
    if any(done.get(k)!=v for k,v in {'status':'PASS','updates':480,'epochs':5,'global_batch':96,'world_size':2}.items()):
        raise ValueError('incomplete or different training budget: '+arm)
    progress=read(run/'progress.json')
    if progress['stage']!='completed' or progress['current']!=480:raise ValueError('nonterminal SS run: '+arm)
    proof=read(study/'verification'/arm/'native-proof.json')
    if proof['status']!='PASS' or any(x!=0 for x in proof['condition_max_abs_errors'].values()):
        raise ValueError('native initialization differs: '+arm)
    if not proof['fusion_gate_gradient']>0:raise ValueError('disconnected conditioner: '+arm)
    spec=cfg['trainer']['args']['deform_conditioner']
    if spec['arm'] in {'c1','c2'} and not proof.get('adapter_output_gradient_l1',0)>0:
        raise ValueError('adapter backward proof missing: '+arm)
    if spec['arm']=='c2' and (not proof.get('query_channels') or proof['query_channels']['extra_channels']!=2048
        or not proof['query_channels']['query_weight_gradient_l1']>0):raise ValueError('physical query proof missing: '+arm)
    validation=[json.loads(x) for x in (run/'validation.jsonl').read_text().splitlines()]
    if [r['step'] for r in validation]!=[0,96,192,288,384,480]:raise ValueError('epoch validations incomplete: '+arm)
    if validation[0]['quick_metrics']['voxel_iou']!=initial_iou:raise ValueError('initial generation quality differs: '+arm)
    rank_records=[]
    for rank in [0,1]:
        file=run/('runtime_batch.jsonl' if rank==0 else 'runtime_batch.rank1.jsonl')
        records=[json.loads(x) for x in file.read_text().splitlines()]
        if [r['step'] for r in records]!=list(range(1,481)):raise ValueError('missing/duplicate optimizer updates: '+arm)
        if any(r['effective_batch_size']!=96 or r['world_size']!=2 or r['accumulation_packs']!=1 or r['microbatch_assets']!=[48] for r in records):
            raise ValueError('unmatched execution nuisance: '+arm)
        if any(not math.isfinite(r['loss']['loss']) for r in records):raise ValueError('nonfinite training loss: '+arm)
        rank_records.append({'rank':rank,'updates':len(records),
            'steady_compute_seconds_median':float(np.median([r['compute_seconds'] for r in records[1:]])),
            'peak_allocated_gib':max(r['peak_allocated_gib'] for r in records)})
    for prefix in ['denoiser','deform_conditioner','misc']:
        path=run/'ckpts'/f'{prefix}_step0000480.pt'
        if not path.is_file() or not path.stat().st_size:raise ValueError('final checkpoint missing: '+arm)
    return {'completed':done,'initial_iou':initial_iou,'validation':validation,'rank_runtime':rank_records,
        'head_sha256':cfg['experiment']['head_sha256'],'omega_checkpoint':cfg['experiment']['omega_checkpoint'],
        'physical_query_channels':2048 if spec['arm']=='c2' else 0,'config':str(study/'configs'/(arm+'.json'))}


def result_contrasts(results):
    contrasts={}
    for candidate in results:
        if candidate in {'v1','original-c1'}:continue
        for baseline in ['v1','original-c1']:
            contrasts[candidate+'-vs-'+baseline]=compare([(str(n),results[baseline]['records'][n],results[candidate]['records'][n]) for n in range(3)])
    for candidate,baseline in [('p2-matcher-grid','p1-native-grid'),('b2-top1-r','b2-top1'),
        ('b3-attention-r','b2-top1-r'),('c1-retarget-r','b3-attention-r'),('c2-query-r','c1-retarget-r')]:
        contrasts[candidate+'-vs-'+baseline]=compare([(str(n),results[baseline]['records'][n],results[candidate]['records'][n]) for n in range(3)])
    return contrasts


def render(report):
    rows=['# SS complete fixed-final primary comparison','',
        'All new runs reached 480 updates. Shared development128, three distinct per-asset sampling noises, sampler12/CFG3. ',
        'Intervals bootstrap source clusters with noise repeats retained inside clusters. One training seed; exploratory contrasts without multiplicity correction.',
        '', '| Arm | Aggregate IoU | Gain vs V1, pp | 95% CI, pp |', '| --- | ---: | ---: | --- |']
    for name,value in report['models'].items():
        if name=='v1':rows.append(f'| {name} | {value["voxel_iou"]:.6f} | baseline | |');continue
        comp=report['original_c1_vs_v1'] if name=='original-c1' else report['contrasts'][name+'-vs-v1']
        lo,hi=comp['paired_cluster_bootstrap_ci95']
        rows.append(f'| {name} | {value["voxel_iou"]:.6f} | {100*comp["gain"]:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] |')
    rows+=['','## Module increments','', '| Contrast | Gain, pp | 95% CI, pp | Evidence |','| --- | ---: | --- | --- |']
    for name in ['b2-top1-vs-v1','b2-top1-r-vs-b2-top1','b3-attention-r-vs-b2-top1-r',
        'c1-retarget-r-vs-b3-attention-r','c2-query-r-vs-c1-retarget-r']:
        comp=report['contrasts'][name];lo,hi=comp['paired_cluster_bootstrap_ci95']
        evidence='positive at this training seed' if lo>0 else 'negative at this training seed' if hi<0 else 'unconfirmed'
        rows.append(f'| {name} | {100*comp["gain"]:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] | {evidence} |')
    rows+=['','## Scope and follow-up','',
        'GT is decoded aligned-target SS latent occupancy. Paired target foreground counts are checked; full occupancy hashes are compared wherever both records contain them.',
        'Both zero-shot baselines use four distinct RGB views and the same desired-state M1 target: source-only uses four M0 views; edited-RGB uses one M1 and three M0 views. Source-only scores here do not measure M0 reconstruction.',
        'This report completes the primary matrix and does not establish unseen-test performance, training-seed replication, source-utilization causality or PBR rendering quality.',
        'The full model is not automatically C2. Review matched increments, uncertainty, learning curves and counterfactual diagnostics before adoption. Small gains need training-seed confirmation.',
        '', 'Matching-head calibration/audits, initialization/physical-query proofs, per-rank runtime, checkpoint locations and evaluation inputs are retained in the JSON companion.']
    return '\n'.join(rows)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;study=root/'module-study-native-v4';reference=root/'retarget-native-v3'
    initial=json.loads((reference/'runs_ss64/p1-native-grid/validation.jsonl').read_text().splitlines()[0])['quick_metrics']['voxel_iou']
    training={name:audit_training(study,name,initial) for name,_,_ in ARMS}
    training.update({name:audit_training(reference,name,initial) for name in ['p1-native-grid','p2-matcher-grid']})
    head=read(root/'selected-head.json')
    if any(v['head_sha256']!=head['head_sha256'] for k,v in training.items() if k!='p1-native-grid'):
        raise ValueError('shared matcher changed inside the module ladder')
    paths={name:[root/f'evaluations/initializer/{mode}/evaluation.json',*[root/f'baseline-noise-confirmation/noise-{n}/{name}/evaluation.json' for n in [1,2]]]
        for name,mode in [('zero-source-only','source-only'),('zero-edited-rgb','edited-rgb')]}
    paths.update({'v1':[root/'evaluations/v1/native/evaluation.json',*[root/f'noise-confirmation/noise-{n}/v1/evaluation.json' for n in [1,2]]],
        'original-c1':[root/'evaluations/c1/normal/evaluation.json',*[root/f'noise-confirmation/noise-{n}/c1/evaluation.json' for n in [1,2]]]}
    )
    paths.update({name:[root/f'evaluations/{name}/normal/evaluation.json',*[root/f'baseline-noise-confirmation/noise-{n}/{name}/evaluation.json' for n in [1,2]]]
        for name in ['b1','b2','b3']})
    paths.update({name:evaluation_paths(root,reference,name) for name in ['p1-native-grid','p2-matcher-grid']})
    paths.update({name:evaluation_paths(root,study,name) for name,_,_ in ARMS})
    results={}
    for name,files in paths.items():
        records=[read(f) for f in files]
        if any(r.get('status')!='PASS' or len(r['per_asset'])!=128 or r.get('sampling_steps')!=12 or r.get('guidance')!=3.
            or r.get('optimizer_updates_during_evaluation')!=0 for r in records):
            raise ValueError('incomplete or nonmatching final generation evaluation: '+name)
        counts=np.array([[r['tp'],r['fp'],r['fn']] for result in records for r in result['per_asset']],dtype=np.int64).sum(0)
        tp,fp,fn=counts;results[name]={'records':records,'evaluations':[str(f) for f in files],
            'voxel_iou':float(tp/(tp+fp+fn)),'voxel_f1':float(2*tp/(2*tp+fp+fn)),
            'voxel_precision':float(tp/(tp+fp)),'voxel_recall':float(tp/(tp+fn))}
    contrasts=result_contrasts(results)
    baseline=compare([(str(n),results['v1']['records'][n],results['original-c1']['records'][n]) for n in range(3)])
    models={name:{k:v for k,v in record.items() if k!='records'} for name,record in results.items()}
    report={'status':'primary_matrix_complete','training_seed_replication':False,'unseen_test_evaluation':False,
        'models':models,'contrasts':contrasts,'original_c1_vs_v1':baseline,'training_audit':training,
        'head_contract':head,'claim_scope':'fixed-final exploratory development comparison; final adoption and any required seed/causal follow-up remain separate'}
    for suffix,content in [('md',render(report)),('json',json.dumps(report,indent=2)+'\n')]:
        path=study/('final-primary-report.'+suffix);tmp=path.with_suffix('.'+suffix+'.tmp')
        tmp.write_text(content);os.replace(tmp,path)
    print(json.dumps({'status':report['status'],'models':len(models),'contrasts':len(contrasts),'training_seed_replication':False}))


if __name__=='__main__':main()
