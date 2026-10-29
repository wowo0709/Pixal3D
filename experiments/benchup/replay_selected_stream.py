"""Apply pinned selected PBR to completed geometry, and archive each result."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from .run_v1_cascade import sha

def live(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for x in ('manifest','shape-root','template','output'):p.add_argument('--'+x,type=Path,required=True)
    p.add_argument('--gpu',required=True,choices=['0','1','2','3']);p.add_argument('--producer-pid',type=int,required=True);p.add_argument('--remote-root',required=True)
    p.add_argument('--host',default='rvi@10.0.12.170');p.add_argument('--socket',default='/tmp/file4-ctl-v2')
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu:raise ValueError('explicit GPU assignment required')
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'logs').mkdir(exist_ok=True)
    seen=set();failed={};ssh=['ssh','-o','BatchMode=yes','-S',a.socket,'-p','55555',a.host]
    while True:
        for rec in sorted((a.shape_root/'cases').glob('*/receipt.json')):
            sid=rec.parent.name
            if sid in seen or sid in failed:continue
            out=a.output/'cases'/sid
            try:
                free=int(subprocess.check_output(['nvidia-smi','--id='+a.gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free<38000:break
                if not out.exists():
                    cmd=[sys.executable,'-u','-m','experiments.benchup.replay_selected_pbr','--gpu',a.gpu,'--manifest',str(a.manifest),'--shape-case',str(rec.parent),'--template',str(a.template),'--output',str(out)]
                    with (a.output/'logs'/f'{sid}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
                receipt=json.loads((out/'receipt.json').read_text())
                if receipt['status']!='SELECTED_FULL_PBR_GENERATED_NO_TARGET_3D_GT' or receipt['prediction_sha256']!=sha(out/'prediction.npz'):raise ValueError('generated PBR provenance mismatch')
                dest=a.remote_root+'/'+sid;subprocess.run([*ssh,'mkdir','-p',dest],check=True)
                subprocess.run(['rsync','-a','-e',f'ssh -o BatchMode=yes -S {a.socket} -p 55555',str(out/'prediction.npz'),str(out/'receipt.json'),str(rec.parent/'native-shape-guides.pt'),a.host+':'+dest+'/'],check=True)
                remote=subprocess.check_output([*ssh,'sha256sum',dest+'/prediction.npz'],text=True).split()[0]
                if remote!=receipt['prediction_sha256']:raise ValueError('file4 checksum mismatch')
                shape_record=json.loads(rec.read_text())
                subprocess.run(['rsync','-a','-e',f'ssh -o BatchMode=yes -S {a.socket} -p 55555',str(rec.parent/'prediction.npz'),a.host+':'+dest+'/shape-prediction.npz'],check=True)
                remote_shape=subprocess.check_output([*ssh,'sha256sum',dest+'/shape-prediction.npz'],text=True).split()[0]
                if remote_shape!=shape_record['prediction_sha256']:raise ValueError('file4 original shape checksum mismatch')
                (out/'archive.json').write_text(json.dumps({'status':'PASS_FILE4_SHA256','sha256':remote,'remote':dest})+'\n');seen.add(sid)
                print(json.dumps({'status':'SELECTED_FULL_ARCHIVED','sample_id':sid,'count':len(seen)}),flush=True)
            except Exception as e:
                failed[sid]=f'{type(e).__name__}: {e}';print(json.dumps({'status':'SELECTED_PBR_FAILED','sample_id':sid,'error':failed[sid]}),flush=True)
        pending={x.parent.name for x in (a.shape_root/'cases').glob('*/receipt.json')}-seen-failed.keys()
        if not live(a.producer_pid) and not pending:break
        time.sleep(5)
    result={'status':'SELECTED_PBR_SHARD_FINISHED','samples':len(seen),'failed':failed}
    (a.output/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
    if failed:raise SystemExit(1)

if __name__=='__main__':main()
