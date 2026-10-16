"""Build image-dependent RGB caches and a fresh fixed-seven-epoch D1 head.

Runs only on an explicitly assigned preparation GPU. Geometry latents remain
the audited world-v4 data. Local legacy caches may be evicted only after a
checksum-verified file4 mirror and an atomically completed RGB replacement.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work-root','data-root','legacy-data-root','file4-mirror-root','vggt-root','base-checkpoint','epoch7-checkpoint'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',required=True);p.add_argument('--dino',required=True)
    p.add_argument('--remote',required=True);p.add_argument('--legacy-remote',required=True)
    p.add_argument('--ssh-control',required=True);p.add_argument('--ssh-port',type=int,required=True)
    a=p.parse_args();w=a.work_root;b=a.data_root
    if not a.gpu.isdigit():raise ValueError('one explicit GPU is required')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',
        WANDB__DISABLE_STATS='true',PYTHONPATH=str(REPO),ATTN_BACKEND='sdpa',OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1',NUMPY_MADVISE_HUGEPAGE='0',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env.pop('SSHPASS',None)
    ssh=['ssh','-S',a.ssh_control,'-p',str(a.ssh_port),'-o','BatchMode=yes','-o','ConnectTimeout=8']
    def state(name,**extra):
        record={'state':name,'time':time.time(),'gpu':a.gpu,**extra}
        (w/'rgb-matching-state.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    def run(module,args,name):
        path=w/(name+'.log')
        with path.open('ab',buffering=0) as log:
            child=subprocess.Popen([sys.executable,'-u','-m',module,*args],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            began=time.time()
            while child.poll() is None:
                if time.time()-max(began,path.stat().st_mtime)>300:
                    child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    raise RuntimeError(f'{name} produced no progress for 300 seconds; stopped owned preparation worker')
                time.sleep(2)
            if child.returncode:raise RuntimeError(f'{name} exited {child.returncode}: {path}')
    def copy(root,relative):
        host,remote=a.remote.split(':',1);target=remote.rstrip('/')+'/'+relative.strip('/')+'/'
        subprocess.run([*ssh,host,'mkdir -p '+shlex.quote(target)],check=True)
        subprocess.run(['rsync','-a','--partial','--timeout=60','--exclude=*.tmp*','-e',shlex.join(ssh),str(root)+'/',host+':'+target],check=True)
    try:
        state('waiting_for_audited_rgb_inputs')
        while not (w/'rgb-training-data-ready.json').exists():
            pp=w/'rgb-pipeline-state.json'
            if pp.exists() and json.loads(pp.read_text()).get('data',{}).get('state')=='FAILED':
                raise RuntimeError('RGB input preparation failed')
            time.sleep(5)
        rows=[]
        for split in ('validation','train'):
            rows.extend(json.loads(x) for x in (b/'manifests_world_v4'/f'{split}.jsonl').read_text().splitlines())
        population=w/'population.jsonl';population.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        state('verifying_legacy_cache_mirrors_before_local_eviction')
        receipt_path=w/'legacy-cache-mirror-receipt.json'
        if not receipt_path.exists():
            inventory={}
            for folder in ('transport','transport_pbr'):
                root=a.legacy_data_root/folder;destination=a.legacy_remote.rstrip('/')+'/'+folder+'/'
                check=subprocess.run(['rsync','-rnc','--timeout=60','--out-format=%i %n','-e',shlex.join(ssh),
                    str(root)+'/',destination],capture_output=True,text=True,check=True,timeout=7200)
                if check.stdout.strip():
                    # Preserve differing local historical artifacts in a new
                    # owned archive instead of replacing authoritative files.
                    copy(root,'history_depth_cache/'+folder)
                    destination=a.remote.rstrip('/')+'/history_depth_cache/'+folder+'/'
                    check=subprocess.run(['rsync','-rnc','--timeout=60','--out-format=%i %n','-e',shlex.join(ssh),
                        str(root)+'/',destination],capture_output=True,text=True,check=True,timeout=7200)
                    if check.stdout.strip():raise RuntimeError('legacy archive checksum differs; refusing eviction')
                for file in root.glob('*.npz'):
                    if file.is_symlink():raise ValueError('legacy cache symlinks cannot be evicted')
                    stat=file.stat();inventory[str(file.resolve())]={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'remote':destination+file.name}
            receipt_path.write_text(json.dumps({'verified_by':'rsync-rnc-no-differences','files':inventory,
                'owned_roots':[str((a.legacy_data_root/f).resolve()) for f in ('transport','transport_pbr')]},indent=2)+'\n')
        mapping='/file4='+str(a.file4_mirror_root)
        for stage in ('ss64','pbr1024'):
            state('building_rgb_'+stage+'_transport',assets=len(rows))
            run('experiments.deformnext_retargeting.precompute_transport',[
                '--manifest',str(population),'--vggt-root',str(a.vggt_root),'--base-checkpoint',str(a.base_checkpoint),
                '--epoch7-checkpoint',str(a.epoch7_checkpoint),'--dino',a.dino,'--stage',stage,
                '--path-map',mapping,'--legacy-mirror-receipt',str(receipt_path)],'rgb-transport-'+stage)
        state('building_rgb_topology_teacher')
        run('experiments.deformnext_retargeting.topology_labels',['--manifest',str(population),
            '--output-root',str(b/'topology_labels_r9_v5'),'--path-map',mapping,'--continue-on-validation-error'],
            'rgb-topology-teacher')
        certified=[r for r in rows if (b/'topology_labels_r9_v5'/f"{r['sample_id']}.npz").exists()]
        for split in ('train','validation'):
            subset=[r for r in certified if r['split']==split]
            if not subset:raise RuntimeError('no certified topology teacher for '+split)
            (w/f'matcher-{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in subset))
        certified_ids={r['sample_id'] for r in certified}
        (w/'rgb-teacher-coverage.json').write_text(json.dumps({'total':len(rows),'certified':len(certified),
            'rejected':[{k:r[k] for k in ('sample_id','split')} for r in rows if r['sample_id'] not in certified_ids]},indent=2)+'\n')
        state('training_rgb_head_fixed_seven_epochs',teacher_assets=len(certified))
        head=b/'q2c_r9_v5/d1-topology-final.pt';head.parent.mkdir(parents=True,exist_ok=True)
        run('experiments.deformnext_retargeting.train_q2c',['--manifest',str(w/'matcher-train.jsonl'),
            '--validation-manifest',str(w/'matcher-validation.jsonl'),'--label-root',str(b/'topology_labels_r9_v5'),
            '--output',str(head),'--path-map',mapping,'--epochs','7','--device','cuda',
            '--wandb-project','pixal3d-deformnext-retargeting','--wandb-entity','youngwoojeon-unist',
            '--wandb-id','deformnext-rgb-v1-d1-head-7ep','--wandb-name','rgb-v1-d1-head-7epoch'],'rgb-head')
        state('building_rgb_raw_and_learned_matches')
        run('experiments.deformnext_retargeting.precompute_raw_matches',['--manifest',str(population),
            '--output-root',str(b/'matches_raw_top1_v5'),'--path-map',mapping],'rgb-raw-matches')
        run('experiments.deformnext_retargeting.precompute_matches',['--manifest',str(population),
            '--q2c-checkpoint',str(head),'--output-root',str(b/'matches_r9_v5'),'--path-map',mapping],'rgb-learned-matches')
        from .run_r9_pipeline import matching_cache_checks
        from .precompute_matches import sha256
        if not all(matching_cache_checks(rows,b/'matches_r9_v5',sha256(head))):raise RuntimeError('RGB full cache integrity failed')
        categories={}
        for r in rows:
            if r['split']=='validation':categories.setdefault(r['category'],r)
        panel=w/'rgb-cache-verification-panel.jsonl';panel.write_text(''.join(json.dumps(categories[k])+'\n' for k in sorted(categories)[:8]))
        state('verifying_rgb_cache_online_parity')
        run('experiments.deformnext_retargeting.verify_match_cache',['--config-root',str(w/'configs'),
            '--manifest',str(panel),'--match-root',str(b/'matches_r9_v5'),'--q2c-checkpoint',str(head),
            '--output',str(w/'rgb-cache-parity.json')],'rgb-cache-parity')
        copy(b/'q2c_r9_v5','q2c_r9_v5');copy(b/'matches_r9_v5','matches_r9_v5');copy(b/'matches_raw_top1_v5','matches_raw_top1_v5')
        report={'status':'PASS','target_modality':'rgb','assets':len(rows),'q2c_sha256':sha256(head),'head_epochs':7}
        tmp=w/'rgb-matching-ready.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');os.replace(tmp,w/'rgb-matching-ready.json')
        state('ready',**report)
    except Exception:
        report={'status':'FAILED','error':traceback.format_exc(),'time':time.time()}
        (w/'rgb-matching-FAILED.json').write_text(json.dumps(report,indent=2)+'\n');state('FAILED',error=report['error']);raise


if __name__=='__main__':main()
