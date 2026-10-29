"""Render and score each successfully baked V1 case as soon as it is ready."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def live(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for x in ('manifest','post-root','output','benchmark-root','repo','blender'):p.add_argument('--'+x,type=Path,required=True)
    p.add_argument('--producer-pids',required=True);p.add_argument('--remote-root',required=True)
    p.add_argument('--host',default='rvi@10.0.12.170');p.add_argument('--socket',default='/tmp/file4-ctl-v2')
    a=p.parse_args();pids=[int(x) for x in a.producer_pids.split(',')]
    if len(pids)!=4:raise ValueError('four explicit postprocessor handles required')
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'logs').mkdir(exist_ok=True)
    seen=set();failed={};ssh=['ssh','-o','BatchMode=yes','-S',a.socket,'-p','55555',a.host]
    while True:
        for arc in sorted(a.post_root.glob('v1-postprocess-shard-*/cases/*/archive.json')):
            sid=arc.parent.name
            if sid in seen or sid in failed:continue
            folder=arc.parent;chunk=a.output/'cases'/sid;score=chunk/'diagnostic-scores.json'
            commands=[]
            if not chunk.exists():commands.append([sys.executable,'-u','-m','experiments.benchup.prepare_render_case','--manifest',str(a.manifest),'--registration',str(folder/'source-registration.json'),'--glb',str(folder/'benchmark.glb'),'--bake-receipt',str(folder/'benchmark.json'),'--benchmark-root',str(a.benchmark_root),'--output',str(chunk)])
            if not (chunk/'render-summary.json').exists():commands.append([sys.executable,'-u','-m','experiments.deformnext_retargeting.render_source_sim3_chunk','--chunk',str(chunk),'--stage-root',str(chunk/'stage'),'--blender',str(a.blender),'--batch-script',str(a.repo/'experiments/deformnext_retargeting/render_source_sim3_blender_batch.py'),'--benchmark-root',str(a.benchmark_root),'--protocol',str(a.benchmark_root/'protocol.json'),'--gpus','0,1,2,3','--threads-per-gpu','2'])
            if not score.exists():commands.append([sys.executable,'-u','-m','experiments.benchup.score_render_case','--manifest',str(a.manifest),'--chunk',str(chunk),'--benchmark-root',str(a.benchmark_root),'--output',str(score)])
            try:
                for n,cmd in enumerate(commands):
                    with (a.output/'logs'/f'{sid}-{n}.log').open('w') as log:subprocess.run(cmd,cwd=a.repo,stdout=log,stderr=subprocess.STDOUT,check=True)
                if json.loads(score.read_text())['status']!='IMAGE_DIAGNOSTIC_PROVISIONAL_NOT_FINAL_BENCHMARK':raise ValueError('score contract mismatch')
                dest=a.remote_root+'/'+sid;subprocess.run([*ssh,'mkdir','-p',dest],check=True)
                subprocess.run(['rsync','-a','--checksum','-e',f'ssh -o BatchMode=yes -S {a.socket} -p 55555',str(score),str(chunk/'stage/raw'),str(chunk/'stage/source_m0'),a.host+':'+dest+'/'],check=True)
                seen.add(sid);print(json.dumps({'status':'RENDERED_PROVISIONAL_V1','sample_id':sid,'count':len(seen)}),flush=True)
            except Exception as e:
                failed[sid]=f'{type(e).__name__}: {e}';print(json.dumps({'status':'RENDER_CASE_FAILED','sample_id':sid,'error':failed[sid]}),flush=True)
        pending={x.parent.name for x in a.post_root.glob('v1-postprocess-shard-*/cases/*/archive.json')}-seen-failed.keys()
        if not any(live(pid) for pid in pids) and not pending:break
        time.sleep(5)
    result={'status':'V1_PROVISIONAL_IMAGE_DIAGNOSTICS_FINISHED','rendered':len(seen),'failed':failed}
    (a.output/'complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if failed:raise SystemExit(1)

if __name__=='__main__':main()
