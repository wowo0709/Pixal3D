"""Re-register archived TRELLIS-frame predictions after front-axis correction."""
from __future__ import annotations
import argparse,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
from .run_v1_cascade import sha


def live(pid):
    try: os.kill(pid,0); return True
    except ProcessLookupError: return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','runtime','raw-root','output-root','surfaces','benchmark-root'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--method',choices=['trellis2','recon','mvadapter-recon','corradapter-recon'],required=True)
    p.add_argument('--rank',type=int,required=True);p.add_argument('--world-size',type=int,default=4)
    p.add_argument('--producer-pids',required=True);p.add_argument('--remote-root',required=True)
    p.add_argument('--socket',default='/tmp/file4-ctl-v2');p.add_argument('--host',default='rvi@10.0.12.170')
    a=p.parse_args();pids=[int(x) for x in a.producer_pids.split(',')]
    if not 0<=a.rank<a.world_size or not pids or any(x<=0 for x in pids):raise ValueError('explicit rank and producer handles required')
    if not re.fullmatch(r'/[a-zA-Z0-9_./-]+',a.remote_root):raise ValueError('unsafe remote path')
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x]
    if len(rows)!=109 or len({x['sample_id'] for x in rows})!=109:raise ValueError('frozen population required')
    byid={r['sample_id']:r for r in rows[a.rank::a.world_size]}; output=a.output_root/f'{a.method}-postprocess-shard-{a.rank}'
    output.mkdir(parents=True,exist_ok=True);(output/'logs').mkdir(exist_ok=True)
    ssh=['ssh','-o','BatchMode=yes','-S',a.socket,'-p','55555',a.host];rsync_ssh=f'ssh -o BatchMode=yes -S {a.socket} -p 55555'
    done=set();failed={}
    while True:
        for sid,row in byid.items():
            if sid in done or sid in failed:continue
            arcs=sorted(a.runtime.glob(f'{a.method}-postprocess-shard-*/cases/{sid}/archive.json'))
            if not arcs:continue
            src=arcs[0].parent;out=output/'cases'/sid;out.mkdir(parents=True,exist_ok=True);temp=out/'restored-prediction.npz'
            try:
                archive=json.loads(arcs[0].read_text());bake=json.loads((src/'benchmark.json').read_text());receipt=a.raw_root/'cases'/sid/'receipt.json'
                raw=json.loads(receipt.read_text());glb=src/'benchmark.glb'
                if archive['status']!='PASS_FILE4_SHA256' or archive['sha256']!=bake['glb_sha256'] or sha(glb)!=bake['glb_sha256'] or raw['prediction_sha256']!=bake['prediction_sha256']:raise ValueError('source evidence mismatch')
                marker=out/'archive.json';reg=out/'source-registration.json'
                if marker.exists():
                    m=json.loads(marker.read_text());v=json.loads(reg.read_text())
                    if m['status']!='PASS_FILE4_SHA256' or m['sha256']!=bake['glb_sha256'] or v['native_frame']!='trellis_minus_y_front_plus_z_up' or v['prediction_sha256']!=raw['prediction_sha256']:raise ValueError('corrected output mismatch')
                    done.add(sid);continue
                prediction=a.raw_root/'cases'/sid/'prediction.npz'
                if not reg.exists():
                    if not prediction.exists():
                        off=json.loads((a.raw_root/'cases'/sid/'offload.json').read_text())
                        if off['status']!='LOCAL_DUPLICATE_RETIRED_VERIFIED_FILE4_RETAINED' or off['sha256']!=raw['prediction_sha256']:raise ValueError('invalid offload record')
                        subprocess.run(['rsync','-a','-e',rsync_ssh,a.host+':'+off['remote'],str(temp)],check=True);prediction=temp
                    cmd=[sys.executable,'-u','-m','experiments.benchup.register_source_m0','--manifest',str(a.manifest),'--prediction',str(prediction),'--receipt',str(receipt),'--source-surface',str(a.surfaces/(row['source_uid']+'.npz')),'--benchmark-root',str(a.benchmark_root),'--output',str(reg),'--native-trellis-frame']
                    with (output/'logs'/f'{sid}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
                value=json.loads(reg.read_text())
                if value['native_frame']!='trellis_minus_y_front_plus_z_up' or value['prediction_sha256']!=raw['prediction_sha256']:raise ValueError('corrected registration mismatch')
                if not (out/'benchmark.glb').exists():os.link(glb,out/'benchmark.glb')
                if not (out/'benchmark.json').exists():shutil.copyfile(src/'benchmark.json',out/'benchmark.json')
                dest=a.remote_root+'/'+sid;subprocess.run([*ssh,'mkdir','-p',dest],check=True)
                remote_glb=dest+'/benchmark.glb'
                link=subprocess.run([*ssh,'ln',archive['remote']+'/benchmark.glb',remote_glb],capture_output=True,text=True)
                if link.returncode and 'File exists' not in link.stderr:raise RuntimeError('file4 hardlink failed: '+link.stderr)
                subprocess.run(['rsync','-a','--checksum','-e',rsync_ssh,str(out/'benchmark.json'),str(reg),a.host+':'+dest+'/'],check=True)
                if subprocess.check_output([*ssh,'sha256sum',remote_glb],text=True).split()[0]!=bake['glb_sha256']:raise ValueError('file4 mesh hash mismatch')
                marker.write_text(json.dumps({'status':'PASS_FILE4_SHA256','sha256':bake['glb_sha256'],'remote':dest,'orientation':'trellis_minus_y_front_plus_z_up'})+'\n')
                done.add(sid);print(json.dumps({'status':'CORRECTED_NATIVE_PROCESSED','sample_id':sid,'count':len(done)}),flush=True)
            except Exception as e:
                failed[sid]=f'{type(e).__name__}: {e}';print(json.dumps({'status':'CASE_FAILED','sample_id':sid,'error':failed[sid]}),flush=True)
            finally:temp.unlink(missing_ok=True)
        pending={sid for sid in byid if sid not in done and sid not in failed}
        if not any(live(pid) for pid in pids):
            if any(list(a.runtime.glob(f'{a.method}-postprocess-shard-*/cases/{sid}/archive.json')) for sid in pending):continue
            failed.update({sid:'upstream postprocessor stopped before archived mesh' for sid in pending});break
        time.sleep(10)
    (output/'complete.json').write_text(json.dumps({'status':'CORRECTED_NATIVE_SHARD_FINISHED','processed':len(done),'failed':failed},indent=2)+'\n')
    if failed:raise SystemExit(1)

if __name__=='__main__':main()
