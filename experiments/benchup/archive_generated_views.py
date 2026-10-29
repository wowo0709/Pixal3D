"""Verify and mirror generated multiview inputs to file4 as they complete."""
from __future__ import annotations
import argparse,json,re,subprocess,time
from pathlib import Path
from .run_mvadapter_views import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--remote-root',required=True);p.add_argument('--producer-pid',type=int,required=True);p.add_argument('--host',default='rvi@10.0.12.170');p.add_argument('--socket',default='/tmp/file4-ctl-v2');a=p.parse_args()
    if not re.fullmatch(r'/[a-zA-Z0-9_./-]+',a.remote_root) or not re.fullmatch(r'[a-zA-Z0-9_@.-]+',a.host):raise ValueError('safe endpoint required')
    ssh=['ssh','-o','BatchMode=yes','-S',a.socket,'-p','55555',a.host];receipts=a.root/'archive-receipts';receipts.mkdir(parents=True,exist_ok=True);seen=set()
    while True:
        for rec in sorted((a.root/'cases').glob('*/receipt.json')):
            sid=rec.parent.name
            if sid in seen:continue
            if not re.fullmatch(r'[a-z0-9_]+',sid):raise ValueError('invalid sample id')
            item=json.loads(rec.read_text())
            if item.get('status') not in {'MVADAPTER_TARGET_ONLY_SIX_GENERATED_VIEWS','CORRADAPTER_TARGET_ONLY_SIX_GENERATED_VIEWS'} or len(item['views'])!=6:raise ValueError('invalid view receipt')
            for v in item['views']:
                if not re.fullmatch(r'view_\d\d\.png',v['name']) or sha(rec.parent/v['name'])!=v['sha256']:raise ValueError('view hash mismatch')
            if sha(rec.parent/'reference.png')!=item['reference_sha256']:raise ValueError('reference mismatch')
            marker=receipts/(sid+'.json')
            if not marker.exists():
                dest=a.remote_root+'/'+sid;subprocess.run([*ssh,'mkdir','-p',dest],check=True)
                subprocess.run(['rsync','-a','--checksum','-e',f'ssh -o BatchMode=yes -S {a.socket} -p 55555',str(rec.parent)+'/',a.host+':'+dest+'/'],check=True)
                remote=subprocess.check_output([*ssh,'sha256sum',*[dest+'/'+v['name'] for v in item['views']]],text=True).splitlines()
                if [x.split()[0] for x in remote]!=[v['sha256'] for v in item['views']]:raise ValueError('remote view hash mismatch')
                marker.write_text(json.dumps({'status':'PASS_FILE4_VIEW_SHA256','sample_id':sid,'remote':dest})+'\n')
            seen.add(sid);print(json.dumps({'status':'archived','sample_id':sid,'count':len(seen)}),flush=True)
        try: __import__('os').kill(a.producer_pid,0);live=True
        except ProcessLookupError:live=False
        if not live:
            print(json.dumps({'status':'FINISHED_VIEW_ARCHIVE','count':len(seen),'expected':109}),flush=True)
            if len(seen)!=109:raise SystemExit(1)
            return
        time.sleep(10)

if __name__=='__main__':main()
