"""Stream V1 case registration and texture baking while generation proceeds."""
from __future__ import annotations
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from .run_v1_cascade import sha

def live(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False

def export_slot():
    """Limit simultaneous new native exports across all streaming workers."""
    while True:
        for index in range(4):
            handle=open(f'/tmp/benchup-textured-export-slot-{index}.lock','a')
            try:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                return handle
            except BlockingIOError:handle.close()
        time.sleep(10)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('manifest','shard-root','surfaces','output','benchmark-root'):p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--gpu',required=True,choices=['0','1','2','3']);p.add_argument('--producer-pid',type=int,required=True)
    p.add_argument('--native-trellis-frame',action='store_true')
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1)
    p.add_argument('--host',default='rvi@10.0.12.170');p.add_argument('--socket',default='/tmp/file4-ctl-v2');p.add_argument('--remote-root',required=True)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu:raise ValueError('explicit allowed GPU required')
    if not 0<=a.rank<a.world_size:raise ValueError('invalid rank')
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x];byid={r['sample_id']:r for r in rows[a.rank::a.world_size]}
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'logs').mkdir(exist_ok=True)
    seen=set();failed={};ssh=['ssh','-o','BatchMode=yes','-S',a.socket,'-p','55555',a.host]
    while True:
        for rec in sorted((a.shard_root/'cases').glob('*/receipt.json')):
            sid=rec.parent.name
            if sid not in byid or sid in seen or sid in failed:continue
            row=byid[sid];folder=a.output/'cases'/sid;folder.mkdir(parents=True,exist_ok=True)
            prediction=rec.parent/'prediction.npz';reg=folder/'source-registration.json';glb=folder/'benchmark.glb';bake=folder/'benchmark.json'
            commands=[]
            if not reg.exists():
                cmd=[sys.executable,'-u','-m','experiments.benchup.register_source_m0','--manifest',str(a.manifest),'--prediction',str(prediction),'--receipt',str(rec),'--source-surface',str(a.surfaces/(row['source_uid']+'.npz')),'--benchmark-root',str(a.benchmark_root),'--output',str(reg)]
                if a.native_trellis_frame:cmd.append('--native-trellis-frame')
                commands.append(cmd)
            if not glb.exists() and not bake.exists():commands.append([sys.executable,'-u','-m','experiments.benchup.export_textured_mesh','--gpu',a.gpu,'--prediction',str(prediction),'--receipt',str(rec),'--output',str(glb),'--benchmark-root',str(a.benchmark_root)])
            try:
                for n,cmd in enumerate(commands):
                    slot=None
                    try:
                        if 'experiments.benchup.export_textured_mesh' in cmd:
                            while True:
                                gpu_free=int(subprocess.check_output(['nvidia-smi','--id='+a.gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                                memory_lines=Path('/proc/meminfo').read_text().splitlines()
                                mem_available=int(next(x.split()[1] for x in memory_lines if x.startswith('MemAvailable:')))
                                # Native TRELLIS meshes can have tens of millions
                                # of faces; concurrent exports must not drive
                                # the host back into swap thrashing.
                                host_floor=160000000 if a.native_trellis_frame else 120000000
                                if gpu_free>=30000 and mem_available>=host_floor:
                                    slot=export_slot()
                                    # Another worker may have started while
                                    # this process was waiting for the slot.
                                    refreshed=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
                                    if refreshed>=host_floor:break
                                    fcntl.flock(slot,fcntl.LOCK_UN);slot.close();slot=None
                                time.sleep(10)
                        with (a.output/'logs'/f'{sid}-{n}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
                    finally:
                        if slot is not None:fcntl.flock(slot,fcntl.LOCK_UN);slot.close()
                b=json.loads(bake.read_text());r=json.loads(reg.read_text())
                if b['status']!='TEXTURED_GLB_BAKED_NEEDS_RENDER_QA' or b['glb_sha256']!=sha(glb) or b['prediction_sha256']!=r['prediction_sha256']:raise ValueError('output contract mismatch')
                dest=a.remote_root+'/'+sid;subprocess.run([*ssh,'mkdir','-p',dest],check=True)
                subprocess.run(['rsync','-a','-e',f'ssh -o BatchMode=yes -S {a.socket} -p 55555',str(glb),str(bake),str(reg),a.host+':'+dest+'/'],check=True)
                if subprocess.check_output([*ssh,'sha256sum',dest+'/benchmark.glb'],text=True).split()[0]!=b['glb_sha256']:raise ValueError('file4 hash mismatch')
                (folder/'archive.json').write_text(json.dumps({'status':'PASS_FILE4_SHA256','sha256':b['glb_sha256'],'remote':dest})+'\n')
                seen.add(sid);print(json.dumps({'status':'PROCESSED_V1_CASE','sample_id':sid,'count':len(seen)}),flush=True)
            except Exception as e:
                failed[sid]=f'{type(e).__name__}: {e}';print(json.dumps({'status':'CASE_FAILED','sample_id':sid,'error':failed[sid]}),flush=True)
        pending=({x.parent.name for x in (a.shard_root/'cases').glob('*/receipt.json')}&byid.keys())-seen-failed.keys()
        if not live(a.producer_pid) and not pending:break
        time.sleep(5)
    # One last scan is guaranteed because the live check happens after scanning.
    result={'status':'FINISHED_V1_BAKING_SHARD','processed':len(seen),'failed':failed,'gpu':a.gpu}
    (a.output/'complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if failed:raise SystemExit(1)

if __name__=='__main__':main()
