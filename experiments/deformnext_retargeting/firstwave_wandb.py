"""Publish completed SS first-wave evidence, without declaring search complete."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from .semantic_topk_search import write_new
from .semantic_topk_zero_replay import MODES,zero_paths,validate_zero
from .summarize_semantic_topk_search import validate_evaluation


PROJECT='youngwoojeon-unist/pixal3d-deformnext-retargeting'
RUN_ID='ss-semantic-topk-firstwave-v1'
ORDER=['zero-source-only','zero-edited-rgb','previous-v1','top1-reference','soft-topk','previous-b3','previous-c1']
LABELS=['Zero source4','Zero edited+source3','V1','Top1','Soft T0.02','B3 T0.005','C1 T0.005']


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def near(left,right):
    if not math.isclose(left,right,rel_tol=0,abs_tol=1e-12):raise ValueError('actual remote numeric differs')


def validate_visual_sources(records,images):
    if not records or set(records)!=set(images):raise ValueError('projection source identities differ')
    first=next(iter(records));reference=records[first];pixels=images[first]
    ids=[r['sample_id'] for r in reference]
    if len(ids)!=128 or len(set(ids))!=128:raise ValueError('complete distinct fixed128 required')
    for name,rows in records.items():
        if images[name].shape!=(128*64,3*64,3):raise ValueError('expected saved SS64 GT/Pred/Diff128 image')
        if [r['sample_id'] for r in rows]!=ids:raise ValueError('fixed128 order differs')
        for row,base in zip(rows,reference):
            keys=['source_cluster_id','seed','target_voxels_sha256','target_voxel_shape']
            if (not row.get('target_voxels_sha256') or row.get('target_voxel_shape')!=[64,64,64]
                or any(row[k]!=base[k] for k in keys)
                or row['tp']+row['fn']!=base['tp']+base['fn']):raise ValueError('source/noise/actual3D GT differs')
        if not np.array_equal(images[name][:,:64],pixels[:,:64]):raise ValueError('actual GT projection pixels differ')
    return ids


def sources(root):
    report=read(root/'firstwave-fixed-final-v1/report.json')
    if report['status']!='PASS_FIXED_FINAL_FIRST_WAVE_ONLY' or report['full_goal_complete'] is not False:
        raise ValueError('completed firstwave report with incomplete full-search scope required')
    old=root.parent/'ss-seed-confirmation-v1'
    paths={name:root/'evaluations'/name/'noise-0/evaluation.json' for name in ['top1-reference','soft-topk']}
    paths.update({'previous-'+name:old/'evaluations'/name/'noise-0/evaluation.json' for name in ['v1','b3','c1']})
    paths.update({name:zero_paths(root,root.parent,name)[0] for name in MODES})
    records={};images={};files=[]
    for path,digest in report['input_SHA256'].items():
        p=Path(path)
        if sha(p)!=digest:raise ValueError('completed firstwave input changed')
        files.append(p)
    for name in ORDER:
        e=read(paths[name])
        if name in MODES:validate_zero(e,MODES[name])
        else:validate_evaluation(e)
        records[name]=e['per_asset'];images[name]=np.asarray(Image.open(paths[name].parent/'panel.png').convert('RGB'))
        files.append(paths[name].parent/'panel.png')
    ids=validate_visual_sources(records,images)
    return report,records,images,ids,files


def render(root):
    folder=root/'firstwave-wandb-v1/presentation'
    if folder.exists():raise FileExistsError('preserve completed or incomplete prior presentation')
    report,records,images,ids,files=sources(root);folder.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(folder/'all128-fixed-order.pdf') as pdf:
        for start in range(0,128,8):
            figure,axes=plt.subplots(8,8,figsize=(13.5,12),layout='constrained')
            for row,case in enumerate(range(start,start+8)):
                axes[row,0].imshow(images[ORDER[0]][case*64:(case+1)*64,:64],interpolation='nearest')
                axes[row,0].set_ylabel(f'{case+1}: '+ids[case].split('_')[1][:8],fontsize=7)
                for column,name in enumerate(ORDER,1):
                    axes[row,column].imshow(images[name][case*64:(case+1)*64,64:128],interpolation='nearest')
                    item=records[name][case];iou=item['tp']/(item['tp']+item['fp']+item['fn'])
                    axes[row,column].set_xlabel(f'IoU64 {iou:.3f}',fontsize=7)
                for column in range(8):
                    axes[row,column].set_xticks([]);axes[row,column].set_yticks([])
                    if row==0:axes[row,column].set_title((['GT']+LABELS)[column],fontsize=8)
            figure.suptitle('Same fixed cases, noise0: SS occupancy max-projections\nNumbers: actual 3D IoU. First-wave comparison; full search ongoing.',fontsize=10)
            pdf.savefig(figure)
            if start==0:figure.savefig(folder/'first8-fixed-order.png',dpi=160)
            plt.close(figure)
    write_new(folder/'summary.json',{'status':'PASS_EXISTING_FIXED128_FIRSTWAVE_PRESENTATION','models':ORDER,
        'sample_ids':ids,'cases':128,'PDF_pages':16,'all_actual_GT_pixels_and_3D_hashes_exact':True,
        'display_scope':'All128 in original evaluation order;PNG first8,not chosen by gain. 2D max-projections are not new3D metrics.',
        'input_fingerprints':{str(p):sha(p) for p in files},'optimizer_updates':0,'new_inference':False,
        'full_goal_complete':False})
    print('PASS_EXISTING_FIXED128_FIRSTWAVE_PRESENTATION',flush=True)


def payload(root):
    report,_,_,_,files=sources(root);folder=root/'firstwave-wandb-v1/presentation'
    if read(folder/'summary.json')['status']!='PASS_EXISTING_FIXED128_FIRSTWAVE_PRESENTATION':raise ValueError('actual presentation required')
    model_rows=[[name,report['models'][name]['micro_voxel_iou'],report['models'][name]['micro_voxel_f1']] for name in ORDER]
    contrast_rows=[[name,c['gain'],*c['paired_cluster_bootstrap_ci95']] for name,c in report['contrasts'].items()]
    metrics={f'final128/{name}/{key}':report['models'][name][key] for name in ORDER for key in ['micro_voxel_iou','micro_voxel_f1']}
    for name,c in report['contrasts'].items():
        metrics.update({f'contrasts/{name}/gain':c['gain'],f'contrasts/{name}/ci95_lower':c['paired_cluster_bootstrap_ci95'][0],f'contrasts/{name}/ci95_upper':c['paired_cluster_bootstrap_ci95'][1]})
    tables={'final128/models':{'columns':['method','micro_IoU','micro_F1'],'data':model_rows},
        'final128/contrasts':{'columns':['contrast','gain_IoU','ci95_lower','ci95_upper'],'data':contrast_rows}}
    images={'final128/first8_fixed':folder/'first8-fixed-order.png',
        'diagnosis/static_value_distribution':root/'firstwave-static-value-diagnosis-v1/diagnosis.png'}
    files += [root/'firstwave-fixed-final-v1'/f for f in ['report.json','report.md']]
    files += [root/'firstwave-static-value-diagnosis-v1'/f for f in ['report.json','diagnosis.png','diagnosis.pdf']]
    files += [folder/f for f in ['summary.json','first8-fixed-order.png','all128-fixed-order.pdf']]
    files += [root/'provenance'/f for f in ['firstwave-trained-QA-v1-file4.json','firstwave-final-and-wave2-startup-v1-file4.json']]
    files=list(dict.fromkeys(files))
    return {'metrics':metrics,'tables':tables,'images':{k:str(p) for k,p in images.items()},
        'artifact_files':{str(p.relative_to(root.parent)):str(p) for p in files},
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'scope':report['scope'],'full_goal_complete':False}


def publish(root):
    import wandb
    folder=root/'firstwave-wandb-v1'
    if (folder/'publication.json').exists():raise FileExistsError('already uploaded;use readback,do not publish again')
    spec=payload(root);write_new(folder/'publication-spec.json',spec)
    run=wandb.init(entity=PROJECT.split('/')[0],project=PROJECT.split('/')[1],id=RUN_ID,
        name='Semantic TopK first-wave fixed-final comparison',resume='never',dir=str(folder),
        settings=wandb.Settings(x_disable_stats=True,x_disable_meta=True),
        config={'source_commit':spec['source_commit'],'scope':spec['scope'],'training_in_this_run':False,
            'full_campaign_complete':False,'primary_metric':'SS64 micro voxelIoU/F1,decoded aligned target latent',
            'canonical_results':'/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1'})
    run.log(spec['metrics'])
    run.log({**{key:wandb.Table(**table) for key,table in spec['tables'].items()},
        **{key:wandb.Image(path) for key,path in spec['images'].items()}})
    artifact=wandb.Artifact(RUN_ID,type='evaluation-report')
    for name,path in spec['artifact_files'].items():artifact.add_file(path,name=name)
    logged=run.log_artifact(artifact);logged.wait();version=logged.version
    run.summary.update({'status':'PASS_COMPLETED_FIRSTWAVE_PUBLICATION_ONLY','firstwave_complete':True,
        'full_campaign_complete':False,'selected_model':'not_selected','model_weights_uploaded':False})
    url=run.url;run.finish()
    write_new(folder/'publication.json',{'status':'PASS_WANDB_FINISHED_FIRSTWAVE_ONLY','run_url':url,
        'run_id':RUN_ID,'artifact_version':version,'spec_SHA256':sha(folder/'publication-spec.json'),
        'independent_readback_required':True,'full_goal_complete':False})
    print(json.dumps({'status':'PASS_WANDB_FINISHED_FIRSTWAVE_ONLY','run_url':url,'version':version}),flush=True)


def readback(root):
    import wandb
    folder=root/'firstwave-wandb-v1';receipt=read(folder/'publication.json');spec=read(folder/'publication-spec.json')
    if sha(folder/'publication-spec.json')!=receipt['spec_SHA256']:raise ValueError('actual publication intent changed')
    if (folder/'remote-readback/audit.json').exists():raise FileExistsError('preserve previous successful readback')
    api=wandb.Api(timeout=30)
    for attempt in range(7):
        api.flush();actual=api.run(PROJECT+'/'+receipt['run_id'])
        if actual.state=='finished' and actual.summary.get('status')=='PASS_COMPLETED_FIRSTWAVE_PUBLICATION_ONLY':break
        if attempt==6:raise RuntimeError('remote upload still pending;poll same run without republishing')
        time.sleep(10)
    if actual.summary.get('full_campaign_complete') is not False or actual.summary.get('selected_model')!='not_selected':
        raise ValueError('actual published scope differs')
    for key,value in spec['metrics'].items():near(actual.summary[key],value)
    index=0
    while (folder/f'remote-readback/attempt-{index}').exists():index+=1
    output=folder/f'remote-readback/attempt-{index}';output.mkdir(parents=True)
    tables={}
    for key,expected in spec['tables'].items():
        meta=actual.summary[key];path=Path(actual.file(meta['path']).download(root=str(output/key.replace('/','_')),replace=False).name)
        table=read(path)
        if table['columns']!=expected['columns'] or len(table['data'])!=len(expected['data']):raise ValueError('actual table shape/columns differ')
        for left,right in zip(table['data'],expected['data']):
            if left[0]!=right[0]:raise ValueError('actual table identifiers differ')
            for a,b in zip(left[1:],right[1:]):near(a,b)
        tables[key]=len(table['data'])
    for key,local in spec['images'].items():
        meta=actual.summary[key];path=Path(actual.file(meta['path']).download(root=str(output/key.replace('/','_')),replace=False).name)
        if sha(path)!=sha(Path(local)) or sha(path)!=meta['sha256']:raise ValueError('actual uploaded PNG bytes differ')
    artifact=api.artifact(PROJECT+'/'+RUN_ID+':'+receipt['artifact_version'])
    downloaded=Path(artifact.download(root=str(output/'artifact')))
    for name,local in spec['artifact_files'].items():
        if sha(downloaded/name)!=sha(Path(local)):raise ValueError('actual artifact file differs: '+name)
    write_new(folder/'remote-readback/audit.json',{'status':'PASS_ACTUAL_FIRSTWAVE_WANDB_READBACK',
        'run_url':actual.url,'actual_remote_state':actual.state,'numeric_fields':len(spec['metrics']),
        'tables':tables,'downloaded_PNGs_byte_exact':len(spec['images']),
        'downloaded_artifact_files_byte_exact':len(spec['artifact_files']),
        'source_commit':spec['source_commit'],'spec_SHA256':receipt['spec_SHA256'],
        'full_goal_complete':False,'scope':'Actual finished first-wave-only run. Full model search/adoption remains active.'})
    print(json.dumps({'status':'PASS_ACTUAL_FIRSTWAVE_WANDB_READBACK','run_url':actual.url,
        'numeric_fields':len(spec['metrics']),'artifact_files':len(spec['artifact_files'])}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['render','publish','readback'],required=True)
    p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    {'render':render,'publish':publish,'readback':readback}[a.mode](a.root)


if __name__=='__main__':main()
