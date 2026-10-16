"""Run RGB zero-shot controls, fresh vanilla LoRA, then validated RGB arms.

This queue never resumes depth-conditioned experiment weights. Matching
preparation is an independent producer; missing/failed readiness blocks arms.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
import traceback

REPO=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work-root','data-root','legacy-data-root','file4-mirror-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--ss-gpus',required=True);p.add_argument('--pbr-gpus',required=True)
    p.add_argument('--remote',required=True);p.add_argument('--ssh-control',required=True)
    p.add_argument('--ssh-port',type=int,required=True)
    a=p.parse_args();w=a.work_root;b=a.data_root
    ss=a.ss_gpus.split(',');pg=a.pbr_gpus.split(',')
    if len(ss)!=1 or len(pg)!=3 or len(set(ss+pg))!=4:raise ValueError('explicit disjoint 1/3 GPU assignment required')
    states={};lock=threading.Lock()
    ssh=['ssh','-S',a.ssh_control,'-p',str(a.ssh_port),'-o','BatchMode=yes','-o','ConnectTimeout=8']
    def state(key,status,**extra):
        with lock:
            states[key]={'state':status,'updated_at':time.time(),**extra}
            tmp=w/'rgb-pipeline-state.tmp';tmp.write_text(json.dumps(states,indent=2)+'\n');os.replace(tmp,w/'rgb-pipeline-state.json')
        print(key,status,flush=True)
    def env(gpus):
        e=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(gpus),CUDA_DEVICE_ORDER='PCI_BUS_ID',
            PYTHONPATH=str(REPO),ATTN_BACKEND='sdpa',NUMPY_MADVISE_HUGEPAGE='0',
            OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',WANDB__DISABLE_STATS='true',
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True');e.pop('SSHPASS',None);return e
    def run(cmd,name,gpus):
        with (w/(name+'.log')).open('ab',buffering=0) as log:
            subprocess.run([sys.executable,'-u',*cmd],cwd=REPO,env=env(gpus),stdout=log,stderr=subprocess.STDOUT,check=True)
    def wait_pid(path):
        import psutil
        if not path.exists():return
        pid=int(path.read_text())
        while psutil.pid_exists(pid):
            try:
                if psutil.Process(pid).status()==psutil.STATUS_ZOMBIE:break
            except psutil.NoSuchProcess:break
            time.sleep(5)
    def sync():
        host,root=a.remote.split(':',1)
        subprocess.run([*ssh,host,'mkdir -p '+shlex.quote(root)],check=True)
        subprocess.run(['rsync','-a','--partial','--timeout=60','--exclude=*.tmp*','-e',shlex.join(ssh),str(b)+'/',a.remote.rstrip('/')+'/'],check=True)
        subprocess.run(['rsync','-a','--partial','--timeout=60','--exclude=wandb/','--exclude=*.pt','-e',shlex.join(ssh),str(w)+'/',a.remote.rstrip('/')+'/runtime/'],check=True)
    def archive(root,relative):
        if not (root.resolve().is_relative_to(b.resolve()) or root.resolve().is_relative_to(w.resolve())):
            raise ValueError('archive must belong to this RGB experiment')
        host,remote=a.remote.split(':',1);target=remote.rstrip('/')+'/'+relative.strip('/')+'/'
        subprocess.run([*ssh,host,'mkdir -p '+shlex.quote(target)],check=True)
        subprocess.run(['rsync','-a','--partial','--timeout=60','-e',shlex.join(ssh),str(root)+'/',host+':'+target],check=True)
        files=list((root/'ckpts').glob('*.pt'))
        check=subprocess.run(['rsync','-rnc','--timeout=60','--out-format=%i %n','-e',shlex.join(ssh),
            str(root/'ckpts')+'/',host+':'+target+'ckpts/'],check=True,capture_output=True,text=True)
        if check.stdout.strip():raise RuntimeError('RGB checkpoint mirror checksum differs')
        receipt=root/'checkpoint_archive.json';receipt.write_text(json.dumps({'remote':host+':'+target+'ckpts/',
            'verified_by':'rsync checksum comparison','files':{f.name:f.stat().st_size for f in files}},indent=2)+'\n')
        subprocess.run(['rsync','-a','-e',shlex.join(ssh),str(receipt),host+':'+target],check=True)
        for file in files:file.unlink()
    def storage():
        while True:
            try:sync();(w/'rgb-storage-state.json').write_text(json.dumps({'status':'synced','time':time.time()})+'\n')
            except Exception:(w/'rgb-storage-state.json').write_text(json.dumps({'status':'retrying','error':traceback.format_exc(),'time':time.time()})+'\n')
            time.sleep(120)
    def manifest():
        try:
            state('data','waiting_for_rgb_render_copy');wait_pid(w/'fetch-target-rgb.pid');wait_pid(w/'fetch-four-camera-source.pid')
            run(['-m','experiments.deformnext_retargeting.prepare_rgb_manifest',
                '--manifest-root',str(a.legacy_data_root/'manifests_world_v4'),'--output-root',str(b/'manifests_world_v4'),
                '--published-root',a.remote.split(':',1)[1],'--path-map','/file4='+str(a.file4_mirror_root)],'prepare-rgb-manifests',[])
            (w/'rgb-training-data-ready.json').write_text(json.dumps({'status':'PASS','target_modality':'rgb'})+'\n')
            state('data','rgb_manifest_audit_pass')
        except Exception:state('data','FAILED',error=traceback.format_exc())
    def stage(stage,gpus):
        try:
            cfgpath=w/'configs'/f'{stage}-v1.json';cfg=json.loads(cfgpath.read_text())
            original=cfg['trainer']['args']['continuation_ckpt']['denoiser']
            for mode in ('edited-rgb','source-only'):
                result=w/'zero-shot'/stage/mode/'evaluation.json'
                state(stage,'zero_shot_'+mode)
                if mode=='edited-rgb':wait_pid(w/f'zero-shot-{stage}.pid')
                wait_pid(w/f'zero-shot-{stage}-{mode}.pid')
                if not result.exists():
                    run(['-m','experiments.deformnext_retargeting.evaluate_native_checkpoint','--stage',stage,
                        '--config-root',str(w/'configs'),'--checkpoint',original,'--output',str(result.parent),
                        '--step','0','--conditioning-mode',mode,'--panel-size','32' if stage=='ss64' else '16'],
                        f'zero-shot-{stage}-{mode}',gpus[:1])
                record=json.loads(result.read_text())
                if record['optimizer_updates']!=0 or record['conditioning_mode']!=mode:raise RuntimeError('zero-shot provenance differs')
            state(stage,'zero_shot_complete_waiting_for_train_rgb')
            while not (w/'rgb-training-data-ready.json').exists():
                if states.get('data',{}).get('state')=='FAILED':raise RuntimeError('RGB manifest preparation failed')
                time.sleep(5)
            if stage=='pbr1024':
                # The initial paired SS control may temporarily use one of
                # the three PBR devices; finish it before allocating DDP.
                wait_pid(w/'zero-shot-ss64-source-only.pid')
            state(stage,'rgb_v1_training')
            run(['train.py','--config',str(cfgpath),'--num_gpus',str(len(gpus)),
                '--master_port','29810' if stage=='ss64' else '29820','--multiview_runtime','--auto_retry','0'],
                f'train-rgb-{stage}-v1',gpus)
            progress=json.loads((Path(cfg['default_output_dir'])/'progress.json').read_text())
            if progress['stage']!='completed' or progress['current']!=cfg['trainer']['args']['max_steps']:
                raise RuntimeError('RGB V1 did not complete')
            archive(Path(cfg['default_output_dir']),f'runs_world_v4/{stage}/v1')
            state(stage,'rgb_v1_complete_waiting_matching')
            while not (w/'rgb-matching-ready.json').exists():
                failure=w/'rgb-matching-FAILED.json'
                if failure.exists():raise RuntimeError(f'RGB matching preparation failed: {failure}')
                time.sleep(10)
            from .configure import arm_conditioner
            from .precompute_matches import sha256
            head=b/'q2c_r9_v5/d1-topology-final.pt';ready=json.loads((w/'rgb-matching-ready.json').read_text())
            if ready.get('status')!='PASS' or ready.get('q2c_sha256')!=sha256(head):raise RuntimeError('RGB matching readiness differs')
            proof=w/'verification-production'
            state(stage,'verifying_rgb_native_start')
            run(['-m','experiments.deformnext_retargeting.verify_native_start','--stage',stage,'--config-root',str(w/'configs'),
                '--output',str(proof/stage),'--samples','32' if stage=='ss64' else '16',
                '--q2c-checkpoint',str(head),'--raw-match-root',str(b/'matches_raw_top1_v5')],f'verify-rgb-{stage}',gpus[:1])
            ep=cfg['experiment']['updates_per_epoch']
            for arm in ('b1','b2','b3','c1','c2'):
                c=copy.deepcopy(cfg);c['default_output_dir']=str(b/'runs_r9_v5'/stage/arm)
                for d in (c['dataset']['args'],c['validation']['dataset_args']):
                    d.update(transport_arm=arm,transport_implementation='r9-native-v5',
                        match_root=str(b/('matches_raw_top1_v5' if arm=='b1' else 'matches_r9_v5')),
                        projection_root=str(a.legacy_data_root/'projection_maps_world_v4'))
                    if arm!='b1':d['expected_q2c_sha256']=sha256(head)
                c['trainer']['args']['deform_conditioner']=arm_conditioner(arm,1024 if stage=='ss64' else 2048,str(head))
                c['trainer']['args']['model_learning_rates']={'deform_conditioner':1e-4}
                c['trainer']['args'].update(max_steps=ep,i_save=ep,max_checkpoints=5)
                path=w/'configs'/f'{stage}-{arm}.json'
                c['experiment'].update(arm=arm,conditioning_schema='r9-native-residual-v5',release_status='pilot-ready',
                    planned_total_steps=cfg['trainer']['args']['max_steps'],pilot_steps=ep,auto_continue_after_pilot=True,
                    recipe_path=str(path),pilot_baseline_file=str(Path(cfg['default_output_dir'])/'validation.jsonl'),q2c_checkpoint=str(head))
                c.update(wandb_id=f'deformnext-rgb-v1-{stage}-{arm}',wandb_name=f'rgb-v1-{stage}-{arm}')
                path.write_text(json.dumps(c,indent=2)+'\n')
                if stage=='pbr1024':
                    state(stage,arm+'_ddp_preflight')
                    probe=copy.deepcopy(c);root=w/'ddp-stress'/arm;root.mkdir(parents=True,exist_ok=True)
                    probe.update(default_output_dir=str(root),use_wandb=False,master_port=29830)
                    probe['trainer']['args'].update(max_steps=2,i_save=2,i_log=1,num_workers=0,max_checkpoints=1)
                    probe['experiment'].update(validate_at_start=False,auto_continue_after_pilot=False)
                    probe['validation']['every_steps']=3
                    # Largest-token assets retained by this RGB population.
                    rows=[json.loads(x) for x in Path(c['data_dir']).read_text().splitlines()]
                    rows=sorted((r for r in rows if int(r['pbr_tokens'])<=32768),key=lambda r:r['pbr_tokens'],reverse=True)[:24]
                    panel=root/'manifest.jsonl';panel.write_text(''.join(json.dumps(r)+'\n' for r in rows));probe['data_dir']=str(panel)
                    pp=root/'launch.json';pp.write_text(json.dumps(probe,indent=2)+'\n')
                    run(['train.py','--config',str(pp),'--num_gpus','3','--master_port','29830','--multiview_runtime','--auto_retry','0'],f'preflight-rgb-pbr-{arm}',gpus)
                    for rank in range(3):
                        file=root/('runtime_batch.jsonl' if rank==0 else f'runtime_batch.rank{rank}.jsonl')
                        records=[json.loads(x) for x in file.read_text().splitlines()]
                        if len(records)!=2 or max(r['peak_allocated_gib'] for r in records)>91:raise RuntimeError('PBR RGB memory preflight failed')
                    archive(root,'runtime/ddp-stress/'+arm)
                state(stage,arm+'_training')
                run(['-m','experiments.deformnext_retargeting.run_campaign','--stage',stage,'--gpus',','.join(gpus),
                    '--config-root',str(w/'configs'),'--verification-root',str(proof),'--log-root',str(w/'logs'),
                    '--master-port','29810' if stage=='ss64' else '29820','--label','rgb-v1','--arms',arm,
                    '--remote',a.remote.rstrip('/')+'/runs_r9_v5','--ssh-control',a.ssh_control,'--ssh-port',str(a.ssh_port),
                    '--archive-checkpoints'],f'campaign-rgb-{stage}-{arm}',gpus)
                if stage=='ss64':
                    for probe in (w/'logs/memory-bench-rgb-v1').glob(arm+'-split*'):
                        if any((probe/'ckpts').glob('*.pt')):archive(probe,'runtime/memory-bench/'+probe.name)
            state(stage,'completed_rgb_arms')
        except Exception:state(stage,'FAILED',error=traceback.format_exc())
    (w/'logs').mkdir(exist_ok=True)
    threading.Thread(target=storage,daemon=True).start()
    threads=[threading.Thread(target=manifest),threading.Thread(target=stage,args=('ss64',ss)),threading.Thread(target=stage,args=('pbr1024',pg))]
    for t in threads:t.start()
    for t in threads:t.join()
    if any(v['state']=='FAILED' for v in states.values()):raise SystemExit(1)


if __name__=='__main__':main()
