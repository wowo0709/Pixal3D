"""Choose the largest measured PBR microbatch with the production pack cap."""
from __future__ import annotations
import argparse,copy,json,os,subprocess,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--stress-manifest',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--master-port',type=int,required=True)
    args=p.parse_args();cfg=json.loads(args.config.read_text())
    args.output_root.mkdir(parents=True,exist_ok=True)
    results=[];local_batch=cfg['trainer']['args']['batch_size_per_gpu']
    for split in (1,2,4,8):
        if local_batch%split:continue
        micro=local_batch//split;root=args.output_root/f'micro{micro}';root.mkdir(exist_ok=True)
        bench=copy.deepcopy(cfg);bench.update(default_output_dir=str(root),num_gpus=1,use_wandb=False,master_port=str(args.master_port))
        bench['data_dir']=str(args.stress_manifest)
        bench['trainer']['args'].update(max_steps=2,batch_split=split,i_save=2,i_log=1,num_workers=0)
        bench['experiment'].update(world_size=1,global_batch=local_batch,validate_at_start=False,
                                   auto_continue_after_pilot=False,release_status='memory-profile-only')
        bench['validation']['every_steps']=3
        path=root/'config.json';path.write_text(json.dumps(bench,indent=2)+'\n')
        cmd=[sys.executable,'train.py','--config',str(path),'--num_gpus','1','--master_port',str(args.master_port),'--multiview_runtime','--auto_retry','0']
        with (root/'launch.log').open('ab',buffering=0) as log:ret=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if ret.returncode:
            detail=(root/'launch.log').read_text(errors='replace')
            if 'out of memory' not in detail.lower():raise RuntimeError(f'Non-OOM PBR profile failure: {root}')
            results.append({'microbatch':micro,'status':'OOM'});print(json.dumps(results[-1]),flush=True);continue
        rows=[json.loads(x) for x in (root/'runtime_batch.jsonl').read_text().splitlines()]
        if len(rows)!=2:raise ValueError('profile did not perform two updates')
        peak=max(r['peak_allocated_gib'] for r in rows)
        assert all(max(r['microbatch_assets'])<=micro for r in rows)
        row={'microbatch':micro,'batch_split':split,'peak_allocated_gib':peak,'status':'PASS' if peak<=91 else 'INSUFFICIENT_HEADROOM'}
        results.append(row);print(json.dumps(row),flush=True)
        if peak>91:continue
        cfg['trainer']['args']['batch_split']=split;cfg['experiment']['microbatch_per_gpu']=micro
        cfg['experiment']['microbatch_profile']=str(args.output_root/'profile.json')
        args.config.write_text(json.dumps(cfg,indent=2)+'\n')
        (args.output_root/'profile.json').write_text(json.dumps({'status':'PASS','selected':row,'probes':results},indent=2)+'\n')
        return
    raise RuntimeError('no PBR microbatch fits the verified memory budget')


if __name__=='__main__':main()
