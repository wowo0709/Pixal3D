"""Refit low-QA source views using Blender-evaluated geometry."""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--qa',type=Path,required=True);p.add_argument('--input-root',type=Path,required=True);p.add_argument('--surfaces',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--threshold',type=float,default=.7);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    rows=[json.loads(x) for x in a.qa.read_text().splitlines()]
    if len(rows)!=180 or len({(r['object_id'],r['view_name']) for r in rows})!=180:raise ValueError('complete source view QA required')
    if not 1<=a.workers<=4 or not 0<a.threshold<1:raise ValueError('invalid configuration')
    selected=sorted((r['object_id'],r['view_name']) for r in rows if r['best']['mask_iou']<a.threshold)
    if not selected:raise ValueError('no source views below threshold')
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'logs').mkdir(exist_ok=True)
    plan={'schema':'benchup-evaluated-source-camera-recovery-v1','source_qa_sha256':hashlib.sha256(a.qa.read_bytes()).hexdigest(),'threshold':a.threshold,'views':len(selected),'selected':selected,'seeds':[0,1,2,3],'tasks':len(selected)*4,'requires_render_QA':True,'target_images_used':False}
    path=a.output/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('existing recovery plan differs')
    path.write_text(json.dumps(plan,indent=2)+'\n');tasks=[(o,v,s) for o,v in selected for s in range(4)]
    script=Path(__file__).with_name('fit_source_orthographic_pilot.py').resolve();env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def run(t):
        obj,view,seed=t;out=a.output/'fits'/obj/f'{view}-seed{seed}.json';out.parent.mkdir(parents=True,exist_ok=True)
        if out.is_file():
            r=json.loads(out.read_text());assert r['seed']==seed and r['evaluated_surface'] and out.with_suffix('.png').is_file();return 'reused'
        log=a.output/'logs'/f'{obj}-{view}-{seed}.log'
        with log.open('w') as stream:subprocess.run([sys.executable,str(script),'--mesh',str(a.input_root/obj/'source/src.glb'),'--image',str(a.input_root/obj/'source/mv'/f'{view}.webp'),'--evaluated-surface',str(a.surfaces/f'{obj}.npz'),'--output',str(out),'--seed',str(seed)],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        return 'created'
    counts={'created':0,'reused':0};start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,t) for t in tasks]
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            counts[f.result()]+=1
            if i%10==0 or i==len(tasks):
                progress={'status':'running','complete':i,'total':len(tasks),'counts':counts,'seconds':round(time.monotonic()-start,2)};(a.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n');print(json.dumps(progress),flush=True)
    (a.output/'complete.json').write_text(json.dumps({'status':'PASS_RECOVERY_FITS_NOT_RENDER_VALIDATED','views':len(selected),'tasks':len(tasks),'counts':counts},indent=2)+'\n')

if __name__=='__main__':main()
