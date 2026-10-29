"""Orchestrate all ten render metrics for the six geometry ablations."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np

GPUS=('0','1','2','3','6','7')
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--render-root',type=Path,required=True);p.add_argument('--template-plan-root',type=Path,required=True);p.add_argument('--references',type=Path,required=True);p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--registry',type=Path,required=True);p.add_argument('--benchmark-root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--metric-python',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--workers-per-gpu',type=int,default=2);a=p.parse_args()
    if a.output_root.exists() or json.loads((a.render_root/'complete.json').read_text()).get('jobs')!=9018:raise ValueError('fresh score root and complete renders required')
    refs={r['sample_id']:r for r in rows(a.references)};target={}
    for path in a.template_plan_root.glob('chunk*/jobs.jsonl'):
        for job in rows(path):
            if job.get('method')=='gt_target8':target[job['sample_id']]=str(path.parent/'stage'/job['output_rgba'])
    tasks=[]
    for chunk in a.render_root.glob('chunk*'):
        if not (chunk/'jobs.jsonl').is_file():continue
        for job in rows(chunk/'jobs.jsonl'):
            arm=job['method'][len('geometry_'):].upper();sid=job['sample_id'];camera=job['camera_id']
            if camera==8:reference=target[sid]
            else:reference=str(a.reference_root/{v['camera_id']:v for v in refs[sid]['views']}[camera]['rgba'])
            tasks.append({'arm':arm,'sample_id':sid,'camera_id':camera,'prediction':str(chunk/'stage'/job['output_rgba']),'reference':reference})
    if len(tasks)!=9018:raise ValueError('render task closure differs')
    a.output_root.mkdir(parents=True);task_path=a.output_root/'tasks.jsonl';task_path.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in tasks));shards=a.output_root/'shards';logs=a.output_root/'logs';shards.mkdir();logs.mkdir()
    children=[];gpus=[g for g in GPUS for _ in range(a.workers_per_gpu)];env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[2]),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for rank,gpu in enumerate(gpus):
        out=shards/f'rank{rank:02d}.jsonl';stream=(logs/f'rank{rank:02d}-gpu{gpu}.log').open('w');cmd=[str(a.metric_python),'-u','-m','experiments.deformnext_retargeting.score_geometry_full_render_worker','--tasks',str(task_path),'--registry',str(a.registry),'--benchmark-root',str(a.benchmark_root),'--physical-gpu',gpu,'--rank',str(rank),'--world-size',str(len(gpus)),'--output',str(out)]
        proc=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT);children.append((proc,stream,out))
    for proc,stream,out in children:
        code=proc.wait();stream.close()
        if code or not out.is_file():raise RuntimeError('metric worker failed')
    scored=[r for _,_,out in children for r in rows(out)];protocol=json.loads(a.protocol.read_text());sys.path.insert(0,str(a.benchmark_root));from deformnext_benchmark.metrics import frechet_distance,kernel_distance
    expected={(task['arm'],task['sample_id'],task['camera_id']) for task in tasks}
    actual=[(row['arm'],row['sample_id'],row['camera_id']) for row in scored]
    if len(expected)!=9018 or len(actual)!=9018 or len(set(actual))!=9018 or set(actual)!=expected:
        raise ValueError('scored render identities do not match the complete 6 x 501 x 3 population')
    result={};arms=sorted(set(r['arm'] for r in scored));settings=protocol['distribution']
    for arm in arms:
        own=[r for r in scored if r['arm']==arm];target_rows=[r for r in own if r['camera_id']==8];source=[r for r in own if r['camera_id'] in (9,10)]
        summary={k:float(np.mean([r['metrics'][k] for r in source])) for k in ('LPIPS','DreamSim','CLIP')};summary.update({k:float(np.mean([r['metrics'][k] for r in target_rows])) for k in ('PSNR','SSIM','M_IoU')})
        features={}
        for _,_,out in children:
            with np.load(out.with_suffix('.npz')) as z:
                keys, predictions, references = z['keys'], z['prediction'], z['reference']
                for i,key in enumerate(keys):
                    name,sid,camera=json.loads(key)
                    if name==arm:
                        if (sid,camera) in features:raise ValueError('duplicate inception feature')
                        features[sid,camera]=(predictions[i].copy(),references[i].copy())
        required={(sid,camera) for sid in refs for camera in (9,10)}
        if set(features)!=required:raise ValueError('incomplete 1002-image inception population')
        ordered=sorted(required);pf=np.stack([features[k][0] for k in ordered]);rf=np.stack([features[k][1] for k in ordered])
        summary['FID']=frechet_distance(pf,rf);summary.update(kernel_distance(pf,rf,subset_size=settings['kid_subset_size'],subsets=settings['kid_subsets'],seed=settings['seed']));result[arm]=summary
    payload={'status':'PASS_COMMON501_GEOMETRY_SIX_ARM_RENDER_METRICS','samples':501,'rows':result};(a.output_root/'metrics.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload),flush=True)
if __name__=='__main__':main()
