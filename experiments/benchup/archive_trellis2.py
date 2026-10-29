"""Mirror completed TRELLIS.2 raw predictions to file4 and verify hashes."""
from __future__ import annotations
import argparse,hashlib,json,re,subprocess,time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--receipt-root',type=Path,required=True);p.add_argument('--remote-root',required=True);p.add_argument('--ssh-socket',type=Path,required=True);p.add_argument('--host',required=True);p.add_argument('--port',type=int,default=55555);p.add_argument('--kind',choices=['trellis2','recon'],default='trellis2');a=p.parse_args()
    if not re.fullmatch(r'/[a-zA-Z0-9_./-]+',a.remote_root) or not re.fullmatch(r'[a-zA-Z0-9_@.-]+',a.host):raise ValueError('explicit safe remote endpoint required')
    a.receipt_root.mkdir(parents=True,exist_ok=True);ssh=['ssh','-o','BatchMode=yes','-S',str(a.ssh_socket),'-p',str(a.port),a.host]
    while True:
        count=0
        for receipt in sorted((a.run_root/'cases').glob('*/receipt.json')):
            sid=receipt.parent.name
            if not re.fullmatch(r'[a-z0-9_]+',sid):raise ValueError('invalid sample id')
            marker=a.receipt_root/(sid+'.json')
            if marker.is_file():count+=1;continue
            r=json.loads(receipt.read_text());prediction=receipt.parent/'prediction.npz';digest=sha(prediction)
            expected='PASS_TRELLIS2_GENERATED_RAW' if a.kind=='trellis2' else 'RECONVIAGEN_V05_GENERATED_RAW'
            if r['status']!=expected or digest!=r['prediction_sha256']:raise ValueError('local output checksum differs')
            dest=a.remote_root+'/'+sid
            subprocess.run([*ssh,'mkdir','-p',dest],check=True)
            subprocess.run(['rsync','-a','-e',f'ssh -o BatchMode=yes -S {a.ssh_socket} -p {a.port}',str(prediction),str(receipt),a.host+':'+dest+'/'],check=True)
            output=subprocess.check_output([*ssh,'sha256sum',dest+'/prediction.npz'],text=True).split()[0]
            if output!=digest:raise ValueError('file4 checksum mismatch')
            marker.write_text(json.dumps({'status':'PASS_FILE4_SHA256','sample_id':sid,'sha256':digest,'remote':dest},indent=2)+'\n');count+=1
            print(json.dumps({'archived':count,'sample_id':sid}),flush=True)
        if count==109:
            (a.receipt_root/'complete.json').write_text(json.dumps({'status':'PASS_ALL109_FILE4_ARCHIVED','samples':109},indent=2)+'\n');return
        time.sleep(10)

if __name__=='__main__':main()
