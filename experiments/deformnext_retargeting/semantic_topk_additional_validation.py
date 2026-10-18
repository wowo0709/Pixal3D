"""Post-selection source-disjoint SS generation validation on the reserved121."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

from .compare_ss_noise_panels import compare
from .semantic_topk_search import write_new
from .summarize_semantic_topk_search import MODELS,aggregate


def jobs_and_contract(root,previous,candidate,reservation,dependency):
    if reservation['status']!='RESERVED_NOT_EVALUATED' or reservation['additional_assets']!=121 or reservation['additional_source_clusters']!=39:
        raise ValueError('additional validation requires the original reserved121/39 panel')
    if candidate not in set(MODELS)-{'top1-reference','task-native-spatial'}:
        raise ValueError('additional validation requires a selected semanticTopK model')
    folder=root/'additional-validation-v1';models={}
    def trained(label,base,name,seed,parent):
        models[label]={'config':str(base/'configs'/(name+'.json')),
            'checkpoint':str(base/'runs_ss64'/name/'ckpts/denoiser_step0000480.pt'),
            'conditioner_checkpoint':str(base/'runs_ss64'/name/'ckpts/deform_conditioner_step0000480.pt') if name!='v1' else None,
            'step':480,'training_seed':seed,'conditioning_mode':'edited-rgb','depends_on':parent}
    for name in [*MODELS,'static-top1-c1']:trained('s101-'+name,root,name,101,[dependency])
    for name in ['v1','c1']:trained('previous-'+name,previous/'ss-seed-confirmation-v1',name,101,[dependency])
    for name in [candidate,'top1-reference']:
        trained('s202-'+name,root/'seed-confirmation-v1',name,202,[dependency,'confirm-seed202-train-'+name])
    initializer=json.loads((root/'configs/zero-shot-reference.json').read_text())['trainer']['args']['continuation_ckpt']['denoiser']
    for label,mode in [('zero-source-only','source-only'),('zero-edited-rgb','edited-rgb')]:
        models[label]={'config':str(root/'configs/zero-shot-reference.json'),'checkpoint':initializer,
            'conditioner_checkpoint':None,'step':0,'training_seed':None,'conditioning_mode':mode,'depends_on':[dependency]}
    jobs=[];panel=folder/'panel-source-disjoint-121.json'
    for label,spec in models.items():
        for n in range(3):
            output=folder/'evaluations'/label/f'noise-{n}';seedmap=reservation['seed_maps'][str(n)]['path']
            command=[sys.executable,'-u','-m','experiments.deformnext_retargeting.evaluate_native_checkpoint',
                '--stage','ss64','--config-root',str(Path(spec['config']).parent),'--arm-config',spec['config'],
                '--checkpoint',spec['checkpoint'],'--step',str(spec['step']),'--conditioning-mode',spec['conditioning_mode'],
                '--output',str(output),'--panel-file',str(panel),'--seed-map-file',seedmap,'--generation-batch','8']
            if spec['conditioner_checkpoint']:command+=['--conditioner-checkpoint',spec['conditioner_checkpoint']]
            jobs.append({'name':f'additional121-{label}-noise{n}','resource':'evaluation','priority':4,
                'depends_on':spec['depends_on'],'requires':[str(panel),seedmap,spec['config'],spec['checkpoint'],
                    *([spec['conditioner_checkpoint']] if spec['conditioner_checkpoint'] else [])],
                'outputs':[str(output/'evaluation.json'),str(output/'panel.png')],'command':command})
    jobs.append({'name':'summarize-additional121-semantic-topk','resource':'cpu','priority':3,
        'depends_on':[j['name'] for j in jobs]+['summarize-semantic-topk-seed202'],
        'outputs':[str(folder/'final-report.json'),str(folder/'final-report.md')],
        'command':[sys.executable,'-u','-m','experiments.deformnext_retargeting.semantic_topk_additional_validation','--root',str(root)]})
    contract={'status':'QUEUED_NOT_EVALUATED','candidate':candidate,'models':models,
        'assets':121,'source_clusters':39,'sampling_noises':3,'panel_SHA256':reservation['panel_SHA256'],
        'generation_batch':8,'sampling_steps':12,'guidance':3.,'optimizer_updates_during_evaluation':0,
        'scope':'After fixed128 selection;source-disjoint additional generation validation within the existing validation split,not an independent test dataset. Selection must remain fixed before these metrics.'}
    return jobs,contract


def validate(e,spec,ids,rows,seeds):
    expected={'status':'PASS','stage':'ss64','step':spec['step'],'weights':'raw','sampling_steps':12,
        'guidance':3.,'generation_batch_size':8,'optimizer_updates_during_evaluation':0,
        'conditioning_mode':spec['conditioning_mode'],'target_modality':'rgb','diagnostic_only':False,
        'checkpoint':spec['checkpoint'],'conditioner_checkpoint':spec['conditioner_checkpoint'],'arm_config':spec['config']}
    if any(e.get(k)!=v for k,v in expected.items()):raise ValueError('additional validation protocol/checkpoint differs')
    states=['M0']*4 if spec['conditioning_mode']=='source-only' else ['M1','M0','M0','M0']
    if e.get('input_camera_ids')!=[8,9,10,11] or e.get('input_states')!=states:
        raise ValueError('additional validation must use original four RGB views')
    if e['quick_panel_ids']!=ids or [r['sample_id'] for r in e['per_asset']]!=ids:
        raise ValueError('additional validation differs from reserved121 panel')
    for r in e['per_asset']:
        sid=r['sample_id']
        if (r['source_cluster_id']!=rows[sid]['source_cluster_id'] or r['seed']!=seeds[sid]
            or not r.get('target_voxels_sha256') or r.get('target_voxel_shape')!=[64,64,64]):
            raise ValueError('additional validation source/noise/GT identity differs')


def summarize(root):
    from .semantic_topk_selection import supported
    folder=root/'additional-validation-v1';contract=json.loads((folder/'evaluation-contract.json').read_text())
    reservation=json.loads((folder/'panel-contract.json').read_text());panel=folder/'panel-source-disjoint-121.json'
    if hashlib.sha256((folder/'panel-contract.json').read_bytes()).hexdigest()!=contract['reservation_SHA256']:
        raise ValueError('additional panel reservation changed after selection')
    decision=json.loads((root/'candidate-selection-v1.json').read_text());candidate=contract['candidate']
    if candidate!=decision['selected_candidate'] or decision['status']!='PROMISING_COMBINATION_FOR_SEED_CONFIRMATION':
        raise ValueError('additional evaluation did not retain fixed128 candidate selection')
    if hashlib.sha256(panel.read_bytes()).hexdigest()!=contract['panel_SHA256'] or contract['panel_SHA256']!=reservation['panel_SHA256']:
        raise ValueError('reserved panel changed after model selection')
    ids=json.loads(panel.read_text())['sample_ids'];cfg=json.loads((root/'configs/top1-reference.json').read_text())
    path=Path(cfg['validation']['roots_file']);train=Path(cfg['data_dir'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=reservation['manifest_SHA256']['validation'] or hashlib.sha256(train.read_bytes()).hexdigest()!=reservation['manifest_SHA256']['train']:
        raise ValueError('reserved manifest changed')
    rows={r['sample_id']:r for r in [json.loads(s) for s in path.read_text().splitlines() if s.strip()]}
    clusters={rows[s]['source_cluster_id'] for s in ids}
    development=json.loads((root/'evaluations/top1-reference/noise-0/evaluation.json').read_text())['per_asset']
    if len(ids)!=121 or len(clusters)!=39 or clusters&{r['source_cluster_id'] for r in development}:
        raise ValueError('additional source groups overlap the development selection panel')
    records={};seedmaps={}
    for n in range(3):
        p=Path(reservation['seed_maps'][str(n)]['path'])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=reservation['seed_maps'][str(n)]['sha256']:raise ValueError('sampling seed map changed')
        seedmaps[n]=json.loads(p.read_text())
    for label,spec in contract['models'].items():
        records[label]=[json.loads((folder/'evaluations'/label/f'noise-{n}/evaluation.json').read_text()) for n in range(3)]
        for n,e in enumerate(records[label]):validate(e,spec,ids,rows,seedmaps[n])
    pairs=[(label,'previous-v1') for label in records if label!='previous-v1']
    pairs += [('s101-task-full','s101-task-no-decoder'),('s101-task-full','s101-task-no-prior'),
        ('s101-task-full','s101-task-native-spatial'),('s101-task-full','s101-static-c1'),
        ('s101-static-c1','s101-static-top1-c1'),('s101-static-top1-c1','s101-top1-reference'),
        ('s101-soft-topk','s101-top1-reference'),('s101-'+candidate,'previous-c1'),
        ('s101-'+candidate,'s101-task-native-spatial'),('zero-edited-rgb','zero-source-only')]
    pairs += [('s'+str(seed)+'-'+candidate,'s'+str(seed)+'-top1-reference') for seed in [101,202]]
    contrasts={}
    for a,b in pairs:
        name=a+'-vs-'+b
        if name in contrasts:continue
        c=compare([(str(n),records[b][n],records[a][n]) for n in range(3)])
        if c['assets']!=121 or c['source_clusters']!=39 or c['paired_target_hashes_verified']!=363:raise ValueError('additional paired GT/source population differs')
        contrasts[name]=c
    models={name:aggregate(es) for name,es in records.items()}
    for name,es in records.items():
        models[name]['secondary_macro_case_iou']=sum(r['tp']/max(1,r['tp']+r['fp']+r['fn']) for e in es for r in e['per_asset'])/363
    perseed={str(seed):contrasts['s'+str(seed)+'-'+candidate+'-vs-s'+str(seed)+'-top1-reference'] for seed in [101,202]}
    result={'status':'PASS_ADDITIONAL121_FIXED_SELECTED_MODEL_VALIDATION','selected_candidate':candidate,
        'models':models,'contrasts':contrasts,'per_training_seed_matched_Top1':perseed,
        'positive_CI_and_all_noises_on_both_additional_training_seeds':all(supported(c) for c in perseed.values()),
        'source_disjoint_from_fixed128':True,'actual_GT_hashes_verified_for_every_pair':True,
        'candidate_selection_SHA256':hashlib.sha256((root/'candidate-selection-v1.json').read_bytes()).hexdigest(),
        'full_goal_complete':False,'scope':contract['scope'],
        'CI_scope':'Separate conditional source-cluster intervals for each pair;not population training-seed CI. Controls on this additional panel do not reselect the candidate.',
        'native_spatial_scope':'Natural3projected-view candidates versus64semantic candidates,not fixed-K address-only causality.'}
    lines=['# Reserved source-disjoint additional121 SS validation','',f'Selected candidate: {candidate}','',
        '| Model | Micro IoU | Micro F1 | Secondary macro case IoU |','| --- | ---: | ---: | ---: |']
    for name,m in models.items():lines.append(f"| {name} | {m['micro_voxel_iou']:.6f} | {m['micro_voxel_f1']:.6f} | {m['secondary_macro_case_iou']:.6f} |")
    lines+=['','| Contrast | Gain IoU,pp | 95% source-cluster CI,pp |','| --- | ---: | --- |']
    for name,c in contrasts.items():
        lo,hi=c['paired_cluster_bootstrap_ci95'];lines.append(f"| {name} | {100*c['gain']:+.3f} | [{100*lo:+.3f},{100*hi:+.3f}] |")
    lines+=['',result['scope'],'',result['CI_scope'],'',result['native_spatial_scope'],'','Functional QA,actual logging/storage closure and model adoption remain required.']
    write_new(folder/'final-report.json',result)
    with (folder/'final-report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'status':result['status'],'candidate':candidate,'confirmed':result['positive_CI_and_all_noises_on_both_additional_training_seeds']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    summarize(p.parse_args().root)


if __name__=='__main__':main()
