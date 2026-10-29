"""Remove only postprocessed V1 local raw duplicates after verified file4 archive."""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
from .run_v1_cascade import sha

def live(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--producer-pids',required=True);a=p.parse_args()
    pids=[int(x) for x in a.producer_pids.split(',')];done=set()
    while True:
        for marker in sorted(a.root.glob('v1-postprocess-shard-*/cases/*/archive.json')):
            sid=marker.parent.name
            if sid in done:continue
            shard=marker.parents[2].name.rsplit('-',1)[-1];case=a.root/f'v1-cascade-shard-{shard}'/'cases'/sid
            source=case/'prediction.npz';local_receipt=case/'receipt.json';archive=a.root/'v1-cascade-archive-receipts'/f'{sid}.json'
            if not source.exists() and (case/'offload.json').exists():done.add(sid);continue
            if not source.exists() or not archive.exists() or not local_receipt.exists():continue
            ar=json.loads(archive.read_text());raw=json.loads(local_receipt.read_text());post=json.loads(marker.read_text())
            if (ar['status']!='PASS_FILE4_SHA256' or post['status']!='PASS_FILE4_SHA256' or ar['sha256']!=raw['prediction_sha256'] or sha(source)!=ar['sha256']):raise ValueError('archive and local raw checksum must match before offload')
            payload={'status':'LOCAL_DUPLICATE_RETIRED_VERIFIED_FILE4_RETAINED','sample_id':sid,'sha256':ar['sha256'],'remote':ar['remote']+'/prediction.npz','restore':'rsync from remote path over the existing file4 SSH connection'}
            temp=case/'offload.tmp';temp.write_text(json.dumps(payload,indent=2)+'\n');os.replace(temp,case/'offload.json');source.unlink();done.add(sid)
            print(json.dumps({'status':'V1_RAW_OFFLOADED','sample_id':sid,'count':len(done)}),flush=True)
        pending={x.parent.name for x in a.root.glob('v1-postprocess-shard-*/cases/*/archive.json')}-done
        if not any(live(pid) for pid in pids) and not pending:break
        time.sleep(15)

if __name__=='__main__':main()
