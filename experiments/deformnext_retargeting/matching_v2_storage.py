"""Mirror owned matching-v2 outputs and retain gracefully stopped RGB runs."""
import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--work-root',type=Path,required=True)
    p.add_argument('--legacy-root',type=Path,required=True);p.add_argument('--remote-host',required=True)
    p.add_argument('--remote-root',required=True);p.add_argument('--ssh-control',required=True);p.add_argument('--ssh-port',required=True)
    a=p.parse_args();ssh=['ssh','-S',a.ssh_control,'-p',a.ssh_port,'-o','BatchMode=yes','-o','ConnectTimeout=8']
    def transfer(source,target,extra=()):
        subprocess.run([*ssh,a.remote_host,'mkdir -p '+shlex.quote(target)],check=True)
        subprocess.run(['rsync','-a','--partial','--timeout=60','--exclude=.*tmp*',*extra,'-e',shlex.join(ssh),str(source).rstrip('/')+'/',a.remote_host+':'+target.rstrip('/')+'/'],check=True)
    preserved=False
    while True:
        try:
            if not preserved:
                for stage,arm in [('ss64','c2'),('pbr1024','b3')]:
                    source=a.legacy_root/'runs_r9_v5'/stage/arm
                    record=json.loads((source/'progress.json').read_text())
                    if record['stage']!='stopped':raise RuntimeError('old trainer has not finished saving')
                    target=a.remote_root.rsplit('/',1)[0]+'/rgb_v1/runs_r9_v5/'+stage+'/'+arm
                    transfer(source,target)
                preserved=True
            transfer(a.data_root,a.remote_root)
            transfer(a.work_root,a.remote_root+'/runtime',('--exclude=evaluation-checkpoints/','--exclude=*.pt','--exclude=wandb/'))
            state={'status':'synced','time':time.time(),'old_runs_preserved':preserved}
        except Exception as e:
            state={'status':'retrying','time':time.time(),'error':str(e),'old_runs_preserved':preserved}
        (a.work_root/'storage-state.json').write_text(json.dumps(state)+'\n');print(json.dumps(state),flush=True)
        time.sleep(120)


if __name__=='__main__':main()
