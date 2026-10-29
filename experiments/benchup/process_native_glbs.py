"""Stream already-textured ShapeUP/SpaceControl outputs to source-only registration."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from .run_v1_cascade import sha

def live(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for x in ('manifest','run-root','surfaces','output','benchmark-root'):p.add_argument('--'+x,type=Path,required=True)
    p.add_argument('--method',required=True,choices=['shapeup','spacecontrol']);p.add_argument('--producer-pid',required=True,type=int);p.add_argument('--rank',type=int,required=True);p.add_argument('--world-size',type=int,default=4);p.add_argument('--remote-root',required=True);p.add_argument('--host',default='rvi@10.0.12.170');p.add_argument('--socket',default='/tmp/file4-ctl-v2');a=p.parse_args()
    if not 0<=a.rank<a.world_size:raise ValueError('invalid shard')
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x]
    if len(rows)!=109:raise ValueError('frozen population required')
    rows=rows[a.rank::a.world_size];a.output.mkdir(parents=True,exist_ok=True);(a.output/'logs').mkdir(exist_ok=True)
    seen=set();failed={};ssh=['ssh','-o','BatchMode=yes','-S',a.socket,'-p','55555',a.host]
    while True:
        for row in rows:
            sid=row['sample_id'];src=a.run_root/'cases'/sid;rec=src/'receipt.json'
            if sid in seen or sid in failed or not rec.is_file():continue
            out=a.output/'cases'/sid
            try:
                if not (out/'source-registration.json').is_file():
                    cmd=[sys.executable,'-u','-m','experiments.benchup.register_native_glb','--method',a.method,'--manifest',str(a.manifest),'--glb',str(src/'textured.glb'),'--receipt',str(rec),'--source-surface',str(a.surfaces/(row['source_uid']+'.npz')),'--benchmark-root',str(a.benchmark_root),'--output',str(out)]
                    with (a.output/'logs'/f'{sid}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
                reg=json.loads((out/'source-registration.json').read_text());cert=json.loads((out/'native-glb.json').read_text());glb=src/'textured.glb'
                if cert['status']!='NATIVE_TEXTURED_GLB_VALIDATED_NEEDS_RENDER_QA' or cert['glb_sha256']!=sha(glb) or reg['prediction_sha256']!=cert['glb_sha256']:raise ValueError('native result mismatch')
                if not (out/'benchmark.glb').exists():os.link(glb,out/'benchmark.glb')
                if not (out/'benchmark.json').exists():(out/'benchmark.json').write_text(json.dumps(cert,indent=2)+'\n')
                dest=a.remote_root+'/'+sid;subprocess.run([*ssh,'mkdir','-p',dest],check=True)
                subprocess.run(['rsync','-a','--checksum','-e',f'ssh -o BatchMode=yes -S {a.socket} -p 55555',str(out/'benchmark.glb'),str(out/'benchmark.json'),str(out/'source-registration.json'),a.host+':'+dest+'/'],check=True)
                if subprocess.check_output([*ssh,'sha256sum',dest+'/benchmark.glb'],text=True).split()[0]!=cert['glb_sha256']:raise ValueError('file4 hash mismatch')
                (out/'archive.json').write_text(json.dumps({'status':'PASS_FILE4_SHA256','sha256':cert['glb_sha256'],'remote':dest})+'\n');seen.add(sid)
                print(json.dumps({'status':'PROCESSED_NATIVE_GLB','sample_id':sid,'count':len(seen)}),flush=True)
            except Exception as e:
                failed[sid]=f'{type(e).__name__}: {e}';print(json.dumps({'status':'CASE_FAILED','sample_id':sid,'error':failed[sid]}),flush=True)
        remaining={r['sample_id'] for r in rows}-seen-failed.keys()
        if not live(a.producer_pid):
            if any((a.run_root/'cases'/sid/'receipt.json').is_file() for sid in remaining):continue
            if remaining:failed.update({sid:'producer stopped before receipt' for sid in remaining})
            break
        time.sleep(10)
    (a.output/'complete.json').write_text(json.dumps({'status':'NATIVE_GLB_SHARD_FINISHED','processed':len(seen),'failed':failed},indent=2)+'\n')
    if failed:raise SystemExit(1)

if __name__=='__main__':main()
