"""Durable R9 cache and five-arm training queue on explicitly assigned GPUs.

Attach to the already-started SS B1 pilot and D1 warm-up. Raw matching and
certified teacher populations must already be complete. Failed quality or
execution gates stop the affected queue and preserve its checkpoint.
"""
from __future__ import annotations
import argparse,copy,json,os,shlex,subprocess,sys,threading,time,traceback
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--baseline-config-root',type=Path,required=True)
    p.add_argument('--file4-mirror-root',type=Path,required=True)
    p.add_argument('--ss-gpus',required=True)
    p.add_argument('--pbr-gpus',required=True)
    p.add_argument('--remote',required=True)
    p.add_argument('--ssh-control',required=True)
    p.add_argument('--ssh-port',type=int,required=True)
    args=p.parse_args();w=args.work_root;b=args.data_root
    ss=args.ss_gpus.split(',');pg=args.pbr_gpus.split(',')
    if len(ss)!=1 or len(pg)!=3 or len(set(ss+pg))!=4:raise ValueError('requires disjoint explicit 1/3 GPU assignment')
    ssh=['ssh','-S',args.ssh_control,'-p',str(args.ssh_port),'-o','BatchMode=yes','-o','ConnectTimeout=8']
    host,remote_base=args.remote.split(':',1)
    ready=threading.Event();prep_failed=threading.Event();states={};lock=threading.Lock()
    def state(key,value,**data):
        with lock:
            states[key]={'state':value,'updated_at':time.time(),**data}
            tmp=w/'pipeline-state.tmp';tmp.write_text(json.dumps(states,indent=2)+'\n');os.replace(tmp,w/'pipeline-state.json')
        print(json.dumps({key:states[key]}),flush=True)
    def env(gpus):
        e=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(gpus),PYTHONPATH=str(REPO),ATTN_BACKEND='sdpa',OMP_NUM_THREADS='8',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        e.pop('SSHPASS',None);return e
    def run(command,name,gpus):
        with (w/(name+'.log')).open('ab',buffering=0) as log:
            subprocess.run([sys.executable,*command],cwd=REPO,env=env(gpus),stdout=log,stderr=subprocess.STDOUT,check=True)
    def live(pid):
        try:
            import psutil
            process=psutil.Process(pid)
            return process.is_running() and process.status()!=psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:return False
    def wait_pid(name):
        pid=int((w/name).read_text())
        while live(pid):time.sleep(10)
    def remote_copy(root,relative,evict=False):
        if evict and not (root.resolve().is_relative_to(b.resolve()) or root.resolve().is_relative_to(w.resolve())):
            raise ValueError('only owned cache checkpoints may be evicted')
        target=remote_base.rstrip('/')+'/'+relative.strip('/')+'/'
        subprocess.run([*ssh,host,'mkdir -p '+shlex.quote(target)],check=True)
        subprocess.run(['rsync','-a','--partial','--exclude=*.tmp*','-e',shlex.join(ssh),str(root)+'/',host+':'+target],check=True)
        if evict:
            ckpts=root/'ckpts'
            check=subprocess.run(['rsync','-rnc','--out-format=%i %n','-e',shlex.join(ssh),str(ckpts)+'/',host+':'+target+'ckpts/'],capture_output=True,text=True,check=True)
            if check.stdout.strip():raise RuntimeError('checkpoint archive checksum differs')
            files=[x for x in ckpts.glob('*.pt') if x.is_file() and not x.is_symlink()]
            receipt=root/'checkpoint_archive.json'
            receipt.write_text(json.dumps({'remote':host+':'+target+'ckpts/','verified_by':'rsync checksum comparison',
                'files':{x.name:x.stat().st_size for x in files}},indent=2)+'\n')
            subprocess.run(['rsync','-a','-e',shlex.join(ssh),str(receipt),host+':'+target],check=True)
            for file in files:file.unlink()
    def baseline_file(stage):
        return str(b/'runs_world_v4/ss64/v1/validation.jsonl') if stage=='ss64' else str(w/'native-baseline/pbr-epoch1-eval/evaluation.json')
    def write_arm(stage,arm):
        from .configure import arm_conditioner
        from .precompute_matches import sha256
        cfg=json.loads((args.baseline_config_root/f'{stage}-v1.json').read_text())
        cfg['_comment']=f'R9-native-v5 {stage}/{arm}: native-zero residual and shared frozen D1 matching'
        checkpoint=b/'q2c_r9_v5/d1-topology-final.pt'
        width=1024 if stage=='ss64' else 2048
        cfg['default_output_dir']=str(b/'runs_r9_v5'/stage/arm)
        for d in (cfg['dataset']['args'],cfg['validation']['dataset_args']):
            d.update(transport_arm=arm,transport_implementation='r9-native-v5',match_root=str(b/'matches_r9_v5'),
                projection_root=str(b/'projection_maps_world_v4'),expected_q2c_sha256=sha256(checkpoint))
        cfg['trainer']['args']['deform_conditioner']=arm_conditioner(arm,width,str(checkpoint))
        cfg['trainer']['args']['model_learning_rates']={'deform_conditioner':1e-4}
        ep=cfg['experiment']['updates_per_epoch'];total=cfg['trainer']['args']['max_steps']
        path=w/'configs'/f'{stage}-{arm}.json'
        if path.exists() and any((b/'runs_r9_v5'/stage/arm/'ckpts').glob('misc_step*.pt')):
            previous=json.loads(path.read_text())
            if previous['dataset']['args'].get('expected_q2c_sha256')!=sha256(checkpoint):
                raise ValueError('cannot resume a semantic arm with a different frozen matcher')
            if (b/'runs_r9_v5'/stage/arm/'PILOT_HOLD.json').exists():
                raise ValueError('semantic pilot remains held for analysis')
            return path
        cfg['experiment'].update(arm=arm,conditioning_schema='r9-native-residual-v5',release_status='pilot-ready',
            planned_total_steps=total,pilot_steps=ep,auto_continue_after_pilot=True,recipe_path=str(path),
            pilot_baseline_file=baseline_file(stage),q2c_checkpoint=str(checkpoint))
        cfg['trainer']['args'].update(max_steps=ep,i_save=ep,max_checkpoints=5)
        if stage=='pbr1024':
            selected=json.loads((w/'pbr-memory-profile/profile.json').read_text())['selected']
            cfg['trainer']['args']['batch_split']=selected['batch_split']
            cfg['experiment']['microbatch_per_gpu']=selected['microbatch']
        cfg['wandb_name']=f'r9-v5-{stage}-{arm}';cfg['wandb_id']=f'deformnext-r9-v5-{stage}-{arm}'
        path.write_text(json.dumps(cfg,indent=2)+'\n')
        return path
    def campaign(stage,arm,gpus,proof):
        run(['-m','experiments.deformnext_retargeting.run_campaign','--stage',stage,'--gpus',','.join(gpus),
            '--config-root',str(w/'configs'),'--verification-root',str(proof),'--log-root',str(w/'logs'),
            '--master-port','29760' if stage=='ss64' else '29770','--label','r9-v5','--arms',arm,
            '--remote',args.remote.rstrip('/')+'/runs_r9_v5','--ssh-control',args.ssh_control,'--ssh-port',str(args.ssh_port),
            '--archive-checkpoints'],f'campaign-{stage}-{arm}',gpus)
    def confirm_complete(stage,arm):
        root=b/'runs_r9_v5'/stage/arm;progress=json.loads((root/'progress.json').read_text())
        expected=480 if stage=='ss64' else 1905
        if progress['stage']!='completed' or progress['current']!=expected:raise RuntimeError(f'incomplete {stage}/{arm}: {progress}')
        if (root/'PILOT_HOLD.json').exists():raise RuntimeError(f'pilot held {stage}/{arm}')
    def pbr_stress(path,arm):
        cfg=json.loads(path.read_text());root=w/'ddp-stress'/arm;root.mkdir(parents=True,exist_ok=True)
        cfg.update(default_output_dir=str(root),num_gpus=3,use_wandb=False,master_port='29790')
        cfg['data_dir']=str(w/'pbr-stress-top24.jsonl')
        cfg['trainer']['args'].update(max_steps=2,i_save=2,i_log=1,num_workers=0,max_checkpoints=1)
        cfg['experiment'].update(validate_at_start=False,auto_continue_after_pilot=False,release_status='memory-profile-only')
        cfg['validation']['every_steps']=3
        cp=root/'config.json';cp.write_text(json.dumps(cfg,indent=2)+'\n')
        run(['train.py','--config',str(cp),'--num_gpus','3','--master_port','29790','--multiview_runtime','--auto_retry','0'],f'ddp-stress-{arm}',pg)
        for rank in range(3):
            file=root/('runtime_batch.jsonl' if rank==0 else f'runtime_batch.rank{rank}.jsonl')
            rows=[json.loads(x) for x in file.read_text().splitlines()]
            assert len(rows)==2 and max(r['peak_allocated_gib'] for r in rows)<=91
            cap=cfg['trainer']['args']['batch_size_per_gpu']//cfg['trainer']['args']['batch_split']
            assert all(max(r['microbatch_assets'])<=cap for r in rows)
        remote_copy(root,'runtime_diagnostics_r9_v5/ddp-stress-'+arm,evict=True)
    def proof(stage,gpus):
        root=w/'verification-production'
        run(['-m','experiments.deformnext_retargeting.verify_native_start','--stage',stage,
            '--config-root',str(args.baseline_config_root),'--output',str(root/stage),'--samples','32' if stage=='ss64' else '16',
            '--q2c-checkpoint',str(b/'q2c_r9_v5/d1-topology-final.pt'),'--raw-match-root',str(b/'matches_raw_top1_v5')],f'verify-production-{stage}',gpus[:1])
        return root
    def prep():
        children=[]
        try:
            state('matching','warmup_running');wait_pid('d1-warmup.pid')
            checkpoint=b/'q2c_r9_v5/d1-topology-final.pt'
            if not checkpoint.exists():raise RuntimeError('D1 warm-up exited without final checkpoint')
            wait_pid('pbr-memory-profile.pid')
            if not (w/'pbr-memory-profile/profile.json').exists():raise RuntimeError('PBR per-rank profile failed')
            wait_pid('baseline-pbr.pid')
            state('matching','building_frozen_cache')
            for rank,gpu in enumerate(pg):
                log=(w/f'match-rank{rank}.log').open('ab',buffering=0)
                cmd=[sys.executable,'-m','experiments.deformnext_retargeting.precompute_matches','--manifest',str(w/'population.jsonl'),
                    '--q2c-checkpoint',str(checkpoint),'--output-root',str(b/'matches_r9_v5'),'--path-map','/file4='+str(args.file4_mirror_root),
                    '--rank',str(rank),'--world-size',str(len(pg))]
                child=subprocess.Popen(cmd,cwd=REPO,env=env([gpu]),stdout=log,stderr=subprocess.STDOUT);children.append((child,log))
            for child,log in children:
                code=child.wait();log.close()
                if code:raise RuntimeError(f'match worker failed: {code}')
            from .precompute_matches import sha256,valid_cache
            digest=sha256(checkpoint)
            rows=[json.loads(x) for x in (w/'population.jsonl').read_text().splitlines()]
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as pool:
                checks=list(pool.map(lambda row:valid_cache(b/'matches_r9_v5'/f"{row['sample_id']}.npz",digest,64),rows))
            if not all(checks):raise RuntimeError('full matching cache integrity failed')
            categories={}
            for row in rows:
                if row['split']=='validation':categories.setdefault(row['category'],row)
            panel=w/'verification-cache-validation.jsonl'
            panel.write_text(''.join(json.dumps(categories[k])+'\n' for k in sorted(categories)[:8]))
            run(['-m','experiments.deformnext_retargeting.verify_match_cache','--config-root',str(args.baseline_config_root),
                '--manifest',str(panel),'--match-root',str(b/'matches_r9_v5'),'--q2c-checkpoint',str(checkpoint),
                '--output',str(w/'production-cache-parity.json')],'verify-production-cache',pg[:1])
            remote_copy(b/'matches_r9_v5','matches_r9_v5');remote_copy(b/'q2c_r9_v5','q2c_r9_v5')
            (w/'matching-ready.json').write_text(json.dumps({'status':'PASS','assets':len(rows),'q2c_sha256':digest},indent=2)+'\n')
            state('matching','ready',assets=len(rows),q2c_sha256=digest);ready.set()
        except Exception:
            detail=traceback.format_exc()
            for child,log in children:
                if child.poll() is None:child.terminate();child.wait(timeout=30)
                if not log.closed:log.close()
            state('matching','FAILED',error=detail);prep_failed.set();ready.set()
    def stage_worker(stage,gpus):
        try:
            if stage=='ss64':
                state(stage,'b1_running');wait_pid('campaign-ss64.pid')
                root=b/'runs_r9_v5'/stage/'b1';progress=json.loads((root/'progress.json').read_text())
                if progress['current']==96:
                    from .pilot import review
                    path=w/'configs/ss64-b1.json';cfg=json.loads(path.read_text())
                    rows=[json.loads(x) for x in (root/'validation.jsonl').read_text().splitlines()]
                    if not any(row['step']==96 for row in rows):
                        output=w/'recovered-validation-ss64-b1'
                        run(['-m','experiments.deformnext_retargeting.evaluate_native_checkpoint','--stage','ss64',
                            '--config-root',str(args.baseline_config_root),'--arm-config',str(path),
                            '--checkpoint',str(root/'ckpts/denoiser_step0000096.pt'),
                            '--conditioner-checkpoint',str(root/'ckpts/deform_conditioner_step0000096.pt'),
                            '--step','96','--output',str(output)],'recover-ss64-b1-validation',gpus)
                        recovered=json.loads((output/'evaluation.json').read_text());rows.append(recovered)
                        with (root/'validation.jsonl').open('a') as stream:stream.write(json.dumps(recovered)+'\n')
                    native=next(json.loads(x) for x in Path(baseline_file(stage)).read_text().splitlines() if json.loads(x)['step']==96)
                    verdict=review(cfg,rows,native)
                    (root/'pilot_review.json').write_text(json.dumps(verdict,indent=2)+'\n')
                    if not verdict['passed']:raise RuntimeError('SS B1 epoch-one pilot held')
                    cfg['trainer']['args']['max_steps']=480
                    cfg['experiment'].update(release_status='pilot-validated',auto_continue_after_pilot=False,pilot_transition_mode='native_checkpoint_resume')
                    path.write_text(json.dumps(cfg,indent=2)+'\n')
                    campaign(stage,'b1',gpus,w/'verification-b1')
                confirm_complete(stage,'b1')
                if not (root/'checkpoint_archive.json').exists():remote_copy(root,'runs_r9_v5/ss64/b1',evict=True)
                ready.wait()
            else:
                state(stage,'validating_ddp_with_lightweight_matching_job')
                wait_pid('pbr-memory-profile.pid')
                pbr_stress(w/'configs/pbr1024-b1.json','b1')
                state(stage,'b1_running');campaign(stage,'b1',gpus,w/'verification-b1');confirm_complete(stage,'b1')
                ready.wait()
            if prep_failed.is_set():raise RuntimeError('matching preparation failed')
            verification=proof(stage,gpus)
            for arm in ['b2','b3','c1','c2']:
                existing=b/'runs_r9_v5'/stage/arm/'progress.json'
                if existing.exists():
                    progress=json.loads(existing.read_text())
                    if progress['stage']=='completed' and progress['current']==(480 if stage=='ss64' else 1905):
                        confirm_complete(stage,arm);continue
                path=write_arm(stage,arm)
                if stage=='pbr1024':pbr_stress(path,arm)
                state(stage,arm+'_running');campaign(stage,arm,gpus,verification);confirm_complete(stage,arm)
                if stage=='ss64':
                    # Production arms are archived by run_campaign. Its
                    # two-update memory probes are separate owned artifacts.
                    for probe in (w/'logs/memory-bench-r9-v5').glob(f'{arm}-split*'):
                        if any((probe/'ckpts').glob('*.pt')):
                            remote_copy(probe,'runtime_diagnostics_r9_v5/ss-'+probe.name,evict=True)
            state(stage,'completed_all_five_arms')
        except Exception:state(stage,'FAILED',error=traceback.format_exc())
    threads=[threading.Thread(target=prep),threading.Thread(target=stage_worker,args=('ss64',ss)),
             threading.Thread(target=stage_worker,args=('pbr1024',pg))]
    for thread in threads:thread.start()
    for thread in threads:thread.join()
    if any(v['state']=='FAILED' for v in states.values()):raise SystemExit(1)


if __name__=='__main__':main()
