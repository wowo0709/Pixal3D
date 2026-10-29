"""Archive completed GT-free V1 cascade cases and verify file4 hashes."""
from __future__ import annotations
import argparse,json,re,subprocess,time
from pathlib import Path
from .run_v1_cascade import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--remote-root',required=True);p.add_argument('--receipts',type=Path,required=True)
    p.add_argument('--host',default='rvi@10.0.12.170');p.add_argument('--socket',default='/tmp/file4-ctl-v2');p.add_argument('--port',type=int,default=55555)
    a=p.parse_args()
    if not re.fullmatch(r'/[a-zA-Z0-9_./-]+',a.remote_root) or not re.fullmatch(r'[a-zA-Z0-9_@.-]+',a.host):raise ValueError('invalid remote')
    a.receipts.mkdir(parents=True,exist_ok=True)
    ssh=['ssh','-o','BatchMode=yes','-S',a.socket,'-p',str(a.port),a.host]
    while True:
        for receipt in sorted(a.root.glob('v1-cascade-shard-*/cases/*/receipt.json')):
            sid=receipt.parent.name
            if not re.fullmatch(r'[a-z0-9_]+',sid):raise ValueError('invalid sample id')
            marker=a.receipts/(sid+'.json')
            if marker.exists():continue
            record=json.loads(receipt.read_text());prediction=receipt.parent/'prediction.npz';digest=sha(prediction)
            if record['status']!='V1_CASCADE_GENERATED_NO_TARGET_3D_GT' or record['prediction_sha256']!=digest:raise ValueError('local case mismatch')
            dest=a.remote_root+'/'+sid
            subprocess.run([*ssh,'mkdir','-p',dest],check=True)
            subprocess.run(['rsync','-a','-e',f'ssh -o BatchMode=yes -S {a.socket} -p {a.port}',str(prediction),str(receipt),a.host+':'+dest+'/'],check=True)
            if subprocess.check_output([*ssh,'sha256sum',dest+'/prediction.npz'],text=True).split()[0]!=digest:raise ValueError('remote hash mismatch')
            marker.write_text(json.dumps({'status':'PASS_FILE4_SHA256','sample_id':sid,'sha256':digest,'remote':dest})+'\n')
            print(json.dumps({'archived':sid}),flush=True)
        count=len(list(a.receipts.glob('*.json')))
        if count==109:
            (a.receipts/'complete').write_text('PASS_ALL109_FILE4_ARCHIVED\n');return
        time.sleep(10)

if __name__=='__main__':main()
