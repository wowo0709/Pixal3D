"""Fit source-only camera hypotheses for the frozen public BenchUp view list."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, subprocess, sys, time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,required=True);p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if not 1<=a.workers<=4:raise ValueError('bounded CPU worker count required')
    rows=[json.loads(line) for line in a.index.read_text().splitlines()];ids={r['sample_id'] for r in rows}
    if len(rows)!=109 or len(ids)!=109 or len({r['object_id'] for r in rows})!=23:raise ValueError('expected exact published population')
    objects={r['object_id'] for r in rows};views={(r['object_id'],r['same_index_source_candidate']) for r in rows}
    for obj in objects:
        for index in (0,8,17,25):views.add((obj,f'{obj}/source/mv/render_{index:04}.webp'))
    if len(views)!=180 or any(view is None for _,view in views):raise ValueError('frozen view inventory differs')
    tasks=[]
    for obj,view in sorted(views):
        for seed in range(4):
            ident=Path(view).stem
            tasks.append((obj,view,seed,a.output/'fits'/obj/f'{ident}-seed{seed}.json'))
    a.output.mkdir(parents=True,exist_ok=True);logs=a.output/'logs';logs.mkdir(exist_ok=True)
    fingerprint=hashlib.sha256(a.index.read_bytes()).hexdigest()
    plan={'schema':'benchup-source-camera-campaign-v1','index_sha256':fingerprint,'objects':23,'views':180,'seeds':[0,1,2,3], 'tasks':len(tasks),'fixed_source_candidate_view_ids':[0,8,17,25],'camera_is_estimated':True,'requires_render_QA':True}
    plan_path=a.output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise ValueError('pre-existing camera plan differs')
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    script=Path(__file__).with_name('fit_source_orthographic_pilot.py').resolve();env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def run(task):
        obj,view,seed,out=task;mesh=a.input_root/obj/'source/src.glb';image=a.input_root/view
        if not mesh.is_file() or not image.is_file():raise FileNotFoundError(f'{mesh}, {image}')
        if out.is_file():
            d=json.loads(out.read_text())
            if d.get('status')!='diagnostic_unvalidated_orthographic' or d['seed']!=seed or not out.with_suffix('.png').is_file():raise ValueError('invalid preexisting camera result '+str(out))
            return 'reused'
        out.parent.mkdir(parents=True,exist_ok=True);log=logs/f'{obj}-{Path(view).stem}-{seed}.log'
        with log.open('w') as stream:
            subprocess.run([sys.executable,str(script),'--mesh',str(mesh),'--image',str(image),'--seed',str(seed),'--output',str(out)],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        return 'created'
    counts={'created':0,'reused':0};start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,t) for t in tasks]
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            counts[f.result()]+=1
            if i%10==0 or i==len(tasks):
                progress={'status':'running','complete':i,'total':len(tasks),'counts':counts,'elapsed_seconds':round(time.monotonic()-start,2)}
                (a.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n');print(json.dumps(progress),flush=True)
    complete={'status':'PASS_SOURCE_CAMERA_HYPOTHESES_NOT_RENDER_VALIDATED','tasks':len(tasks),'views':len(views),'index_sha256':fingerprint,'counts':counts}
    (a.output/'complete.json').write_text(json.dumps(complete,indent=2)+'\n');print(json.dumps(complete),flush=True)


if __name__=='__main__':main()
