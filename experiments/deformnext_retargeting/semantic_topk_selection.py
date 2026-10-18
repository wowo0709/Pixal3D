"""Select an observed Top-K combination and queue fresh seed confirmation."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .compare_ss_noise_panels import compare
from .semantic_topk_confirmation import make_jobs,seed_config
from .semantic_topk_search import write_new
from .summarize_semantic_topk_search import aggregate,task_gradient_audit,validate_evaluation
from .summarize_ss_module_study import audit_training,audit_final_replay


CANDIDATES=['soft-topk','static-c1','task-full','task-no-decoder','task-no-prior']
SELECT_JOB='select-and-queue-semantic-topk-seed202'


def supported(contrast):
    return (contrast['gain']>0 and contrast['paired_cluster_bootstrap_ci95'][0]>0
            and len(contrast['per_noise'])==3 and all(x['gain']>0 for x in contrast['per_noise']))


def selection(seven,matched):
    if seven['status']!='PASS_SEVEN_ARM_FIXED_FINAL_DEVELOPMENT_SEARCH' or matched['status']!='PASS_MATCHED_RETARGETER_TOP1_FIXED_FINAL':
        raise ValueError('selection requires both complete fixed-final reports')
    if any(seven['models'][name]!=matched['models'][name] for name in ['static-c1','task-full','task-no-decoder','task-no-prior','top1-reference','previous-c1']):
        raise ValueError('fixed-final reports disagree')
    candidates={}
    scopes={'soft-topk':'Frozen matcher,no retargeter:weighting only',
        'static-c1':'Frozen matcher,same C1 retargeter:weighting only',
        'task-full':'Learned D1/scalar-score/content attention plus C1:whole-path comparison',
        'task-no-decoder':'Learned pairwise scalar-score/content attention plus C1;physical D0',
        'task-no-prior':'Learned D1-context/content attention plus C1;scalar-score prior OFF'}
    for name in CANDIDATES:
        versus_v1=seven['contrasts'][name+'-vs-previous-v1']
        versus_top1=seven['contrasts'][name+'-vs-top1-reference']
        candidates[name]={'micro_voxel_iou':seven['models'][name]['micro_voxel_iou'],
            'supported_vs_V1_and_Top1_in_development':supported(versus_v1) and supported(versus_top1),
            'versus_V1':versus_v1,'versus_Top1':versus_top1,
            'same_retargeter_Top1_control':seven['contrasts']['soft-topk-vs-top1-reference'] if name=='soft-topk' else matched['contrasts'][name+'-vs-static-top1-c1'],
            'TopK_effect_scope':scopes[name]}
    eligible=[name for name in CANDIDATES if candidates[name]['supported_vs_V1_and_Top1_in_development']]
    chosen=max(eligible,key=lambda n:candidates[n]['micro_voxel_iou']) if eligible else None
    overall={**seven['models'],**matched['models']}
    best_overall=max(overall,key=lambda n:overall[n]['micro_voxel_iou'])
    return {'status':'PROMISING_COMBINATION_FOR_SEED_CONFIRMATION' if chosen else 'NO_PROMISING_COMBINATION_YET',
        'selected_candidate':chosen,'candidate_evidence':candidates,
        'best_overall_development_model':best_overall,
        'selected_exceeds_previous_C1_point_estimate':None if chosen is None else overall[chosen]['micro_voxel_iou']>overall['previous-c1']['micro_voxel_iou'],
        'selection_rule':'Among semanticTopK+weighted-attention models with positive source-cluster CI against V1 and Top1 andpositive direction at all3noises,pick highest microIoU. Decoder/prior may be removed when their ablations win;do not presume task-full wins. Native-spatial/Top1 controls cannot establish semanticTopK utility.',
        'same_retargeter_frozen_score_TopK_effect':matched['contrasts']['static-c1-vs-static-top1-c1'],
        'scope':'Exploratory development model selection. Additional training seed uses the same128assets/noises;not independent held-out test or a population training-seed CI.',
        'full_goal_complete':False,'if_no_candidate':'Retain all negative evidence and continue model diagnosis/design;do not mark goal complete.'}


def queue_candidate(a):
    paths=[a.root/'seven-arm-final-report.json',a.root/'matched-retargeter-top1-report.json']
    result=selection(*[json.loads(p.read_text()) for p in paths]);name=result['selected_candidate']
    result['actual_fixed_final_report_SHA256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    output=a.root/'candidate-selection-v1.json'
    if output.exists():raise FileExistsError('preserve actual selection')
    if name is None:
        write_new(output,result);print(json.dumps(result),flush=True);return
    devices=a.gpus.split(',')
    if len(devices)!=2 or len(set(devices))!=2 or any(not x.isdigit() for x in devices):raise ValueError('explicit assigned GPU pair required')
    folder=a.root/'seed-confirmation-v1'
    cfg=seed_config(json.loads((a.root/'configs'/(name+'.json')).read_text()),folder,name,202)
    cfg['master_port']='30441'  # Concurrent Top1 baseline owns30440 on this host.
    path=a.work_root/'queue-jobs.json';old=path.read_bytes();existing=json.loads(old)
    jobs=make_jobs(folder,a.previous_work,name,devices,[SELECT_JOB],202)
    summary='summarize-semantic-topk-seed202'
    from .semantic_topk_functional_qa import job as functional_job
    from .semantic_topk_selected_correspondence import jobs_and_contract as correspondence_jobs,SUMMARY_JOB
    qa=functional_job(a.root,folder,name,202,a.previous_work/'panels/panel-128.json')
    diagnostic_jobs,diagnostic_contract=correspondence_jobs(a.root,a.previous_work,name,SELECT_JOB)
    jobs.append({'name':summary,'resource':'cpu','priority':2,
        'depends_on':[f'confirm-seed202-final-{arm}-noise{n}' for arm in [name,'top1-reference'] for n in range(3)]+[qa['name'],'functional-QA-s202-top1-reference',SUMMARY_JOB],
        'outputs':[str(folder/'final-confirmation-report.json'),str(folder/'final-confirmation-report.md')],
        'command':[sys.executable,'-u','-m','experiments.deformnext_retargeting.semantic_topk_selection',
                   '--mode','summarize','--root',str(a.root)]})
    jobs.append(qa)
    jobs.extend(diagnostic_jobs)
    from .semantic_topk_additional_validation import jobs_and_contract
    reservation_path=a.root/'additional-validation-v1/panel-contract.json'
    extra_jobs,extra_contract=jobs_and_contract(a.root,a.root.parent,name,json.loads(reservation_path.read_text()),SELECT_JOB)
    extra_contract['reservation_SHA256']=hashlib.sha256(reservation_path.read_bytes()).hexdigest()
    jobs.extend(extra_jobs)
    if {j['name'] for j in jobs}&{j['name'] for j in existing}:raise FileExistsError('candidate seed202 already queued')
    if not any(j['name']==SELECT_JOB for j in existing):raise ValueError('selection job missing from queue')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for job in jobs:job['source_commit']=source
    write_new(folder/'configs'/(name+'.json'),cfg)
    write_new(folder/'candidate-contract.json',{'status':'CONFIGURED_NOT_TRAINED','candidate':name,'training_seed':202,
        'source_commit':source,'master_port':30441,'fresh_original_SS4210':True,'fixed_final_report_SHA256':result['actual_fixed_final_report_SHA256'],
        'scope':'Additional optimization-seed confirmation on same development panel,no test-generalization claim.'})
    write_new(a.root/'additional-validation-v1/evaluation-contract.json',extra_contract)
    write_new(a.root/'selected-correspondence-v1/contract.json',diagnostic_contract)
    with (a.work_root/'queue-jobs-before-selected-candidate202.json').open('xb') as f:f.write(old)
    temp=path.with_suffix('.selected-candidate.tmp');temp.write_text(json.dumps(existing+jobs,indent=2)+'\n');os.replace(temp,path)
    result.update(queued_jobs=len(jobs),source_commit=source)
    write_new(output,result);print(json.dumps({'status':result['status'],'candidate':name,'queued_jobs':len(jobs)}),flush=True)


def summarize(root):
    decision=json.loads((root/'candidate-selection-v1.json').read_text());name=decision['selected_candidate']
    if not name or decision['status']!='PROMISING_COMBINATION_FOR_SEED_CONFIRMATION':raise ValueError('no selected combination')
    folder=root/'seed-confirmation-v1';seven=json.loads((root/'seven-arm-final-report.json').read_text())
    models={};training={};records={}
    for arm in [name,'top1-reference']:
        training[arm]=audit_training(folder,arm,seven['training_audits'][arm]['initial_iou'])
        records[arm]=[json.loads((folder/'evaluations'/arm/f'noise-{n}/evaluation.json').read_text()) for n in range(3)]
        for e in records[arm]:validate_evaluation(e)
        training[arm]['final128_replays_epoch32']=audit_final_replay(training[arm],records[arm][0])
        training[arm]['task_branch_gradients']=task_gradient_audit(folder,arm)
        models[arm]=aggregate(records[arm])
        for n in range(3):
            previous=json.loads((root/'evaluations'/arm/f'noise-{n}/evaluation.json').read_text())
            keys=['sample_id','source_cluster_id','seed','target_voxels_sha256','target_voxel_shape']
            byid={r['sample_id']:r for r in previous['per_asset']}
            if any(any(r[k]!=byid[r['sample_id']][k] for k in keys) for r in records[arm][n]['per_asset']):
                raise ValueError('confirmation changed the fixed evaluation data/noise')
        cfg=json.loads((folder/'configs'/(arm+'.json')).read_text())
        if cfg['training_seed']!=202:raise ValueError('confirmation did not use a fresh training seed202')
    second=compare([(str(n),records['top1-reference'][n],records[name][n]) for n in range(3)])
    first=seven['contrasts'][name+'-vs-top1-reference']
    report={'status':'PASS_FIXED_FINAL_SEED202_CONFIRMATION','selected_candidate':name,'models':models,
        'training_audits':training,'per_training_seed_contrast':{'101':first,'202':second},
        'positive_CI_and_all_noise_direction_at_both_seeds':supported(first) and supported(second),
        'descriptive_mean_gain_over_training_seeds':(first['gain']+second['gain'])/2,
        'training_seed_population_CI':None,'full_goal_complete':False,
        'required_remaining':['reserved source-disjoint additional121 evaluation','trained cache/BF16/CFG functional audit','actual final logging/storage/source closure','module-effect interpretation and final adoption'],
        'scope':'Two training seeds evaluated on the same fixed development128/93source clusters/3noises. CIs remain separate conditional source-cluster intervals;no held-out-test or training-seed-population CI claim.'}
    lines=['# Semantic Top-K additional training-seed confirmation','',f'Candidate: {name}','','| Training seed | Gain IoU vs matched Top1,pp | 95% source-cluster CI,pp |','| --- | ---: | --- |']
    for seed,c in report['per_training_seed_contrast'].items():
        lo,hi=c['paired_cluster_bootstrap_ci95'];lines.append(f"| {seed} | {100*c['gain']:+.3f} | [{100*lo:+.3f},{100*hi:+.3f}] |")
    lines+=['',report['scope'],'','Functional QA,logging/storage closure and adoption are not completed by this report.']
    write_new(folder/'final-confirmation-report.json',report)
    with (folder/'final-confirmation-report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'status':report['status'],'candidate':name,'confirmed':report['positive_CI_and_all_noise_direction_at_both_seeds']}),flush=True)


def install(a):
    devices=a.gpus.split(',')
    if len(devices)!=2 or len(set(devices))!=2 or any(not x.isdigit() for x in devices):raise ValueError('explicit assigned GPU pair required')
    path=a.work_root/'queue-jobs.json';old=path.read_bytes();jobs=json.loads(old)
    if any(j['name']==SELECT_JOB for j in jobs):raise FileExistsError('selection job already installed')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    job={'name':SELECT_JOB,'resource':'cpu','priority':2,'source_commit':source,
        'depends_on':['summarize-seven-arm-semantic-topk-search','summarize-matched-retargeter-top1','summarize-task-value-address-diagnostic'],
        'outputs':[str(a.root/'candidate-selection-v1.json')],
        'command':[sys.executable,'-u','-m','experiments.deformnext_retargeting.semantic_topk_selection','--mode','queue',
            '--root',str(a.root),'--work-root',str(a.work_root),'--previous-work',str(a.previous_work),'--gpus',a.gpus]}
    if any(dep not in {j['name'] for j in jobs} for dep in job['depends_on']):raise ValueError('required comparison missing')
    with (a.work_root/'queue-jobs-before-candidate-selection.json').open('xb') as f:f.write(old)
    temp=path.with_suffix('.selection.tmp');temp.write_text(json.dumps(jobs+[job],indent=2)+'\n');os.replace(temp,path)
    print(json.dumps({'status':'INSTALLED_NOT_SELECTED','jobs':len(jobs)+1,'source_commit':source}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['install','queue','summarize'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--work-root',type=Path)
    p.add_argument('--previous-work',type=Path);p.add_argument('--gpus');a=p.parse_args()
    if a.mode=='summarize':summarize(a.root)
    else:
        if a.work_root is None or a.previous_work is None or a.gpus is None:p.error('install/queue requires work-root,previous-work,gpus')
        (install if a.mode=='install' else queue_candidate)(a)


if __name__=='__main__':main()
