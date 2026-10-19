"""Gate, configure, verify and run independent SS matching-v2 arms."""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PUBLIC = {'b2-v2':('b2',False),'b2-v2-r':('b2',True),'b3-v2':('b3',True),'c1-v2':('c1',True),'c2-v2':('c2',True)}


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def select_oracle(root):
    summary=json.loads((root/'calibration-oracle/summary.json').read_text())
    cases=[k for k in summary if k.startswith('oracle-')]
    best=max(cases,key=lambda k:summary[k]['voxel_iou'])
    gain=summary[best]['voxel_iou']-summary['native']['voxel_iou']
    result={'status':'PASS' if gain>0 else 'HOLD','scale':float(best.removeprefix('oracle-')),
            'calibration_gain':gain,'selection_scope':'train-internal calibration32; GT diagnostic only',
            'metrics':summary}
    write(root/'oracle-scale.json',result);print(json.dumps(result),flush=True)


def assess_oracle(a):
    root=a.root;decision=json.loads((root/'oracle-scale.json').read_text())
    if decision['status']=='PASS':
        verification=root/'oracle-verified/summary.json'
        while not verification.exists():
            state=json.loads((a.work_root/'queue-state.json').read_text())
            if state.get('verify-calibrated-oracle',{}).get('status') in {'FAILED','HELD'}:
                decision={'status':'HOLD','reason':'calibrated oracle verification failed'};break
            time.sleep(5)
        if verification.exists():
            summary=json.loads(verification.read_text());case=next(k for k in summary if k.startswith('oracle-'))
            gain=summary[case]['voxel_iou']-summary['native']['voxel_iou']
            decision={**decision,'status':'PASS' if gain>0 else 'HOLD','validation_oracle_gain':gain,
                      'reason':'nonzero calibrated GT transport has SS headroom' if gain>0 else 'GT transport did not improve SS; revisit injection before main training',
                      'scope':'32-asset screening; permits experiment, not a performance promotion'}
    write(root/'transport-gate.json',decision);print(json.dumps(decision),flush=True)


def configure(a):
    root=a.root;decision=json.loads((root/'transport-gate.json').read_text())
    audit=json.loads((root/'cache-audit.json').read_text())
    if audit.get('status')!='PASS':raise ValueError('cache parity has not passed')
    write(root/'ss-decision.json',decision)
    if decision['status']!='PASS':
        write(root/'configs-ready.json',{'status':'HOLD','reason':decision});return
    contract=json.loads((root/'matches/contract.json').read_text())
    base=json.loads((a.config_root/'ss64-v1.json').read_text())
    for name,(arm,reject) in PUBLIC.items():
        cfg=copy.deepcopy(base);cfg['default_output_dir']=str(root/'runs_ss64'/name)
        for d in (cfg['dataset']['args'],cfg['validation']['dataset_args']):
            d.update(transport_arm=arm,transport_implementation='matching-v2',match_root=str(root/'matches'),
                     expected_q2c_sha256=contract['head_sha256'],projection_root=str(a.legacy_root/'projection_maps_world_v4'))
        cfg['trainer']['args'].update(max_steps=480,i_save=96,max_checkpoints=5,batch_size_per_gpu=96,batch_split=1,
            deform_conditioner={'implementation':'matching-v2','arm':arm,'value_dim':1024,'query_dim':1024,
                'hidden_dim':256,'g_max':.25,'reject':reject,'reject_threshold':contract['reliability']['threshold'],
                **({'deformation_dim':2048} if arm=='c2' else {})},model_learning_rates={'deform_conditioner':1e-4})
        cfg['validation']['quick_metrics'].update(panel_size=32,panel_file=str(a.work_root/'panels/panel-32.json'),
            seed_map_file=str(a.work_root/'panels/seed-map.json'),generation_batch_size=8)
        cfg['experiment'].update(arm=name,conditioning_schema='matching-v2',auto_continue_after_pilot=False,
            release_status='oracle-screened-pending-native-proof',validate_at_start=True,epochs=5,
            q2c_checkpoint=contract['head'],head_sha256=contract['head_sha256'],global_batch=96,world_size=1,
            microbatch_per_gpu=96,updates_per_epoch=96,oracle_screening=decision)
        cfg.update(num_gpus=1,master_port=str(29900+list(PUBLIC).index(name)),use_wandb=True,
            wandb_project='pixal3d-deformnext-retargeting',wandb_entity='youngwoojeon-unist',
            wandb_id='ss-matching-v2-'+name,wandb_name='SS matching-v2 '+name)
        write(root/'configs'/(name+'.json'),cfg)
    write(root/'configs-ready.json',{'status':'PASS','arms':list(PUBLIC),'head_sha256':contract['head_sha256']})


def native_proof(cfg,output):
    repo=Path(__file__).resolve().parents[2]
    code_paths=[Path(__file__).resolve(),repo/'pixal3d/modules/deform_conditioner.py',
        repo/'pixal3d/modules/task_matching_attention.py',repo/'pixal3d/datasets/deformnext.py',
        repo/'pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py',
        repo/'experiments/multiview_finetune/runtime.py']
    code_hashes={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}
    proof_source={'commit_at_entry':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                  'actual_component_SHA256':code_hashes}
    import torch
    from pixal3d import datasets,models,trainers
    from pixal3d.utils.data_utils import recursive_to_device
    torch.set_num_threads(4);torch.cuda.set_device(0)
    from train import setup_rng
    setup_rng(0,cfg.get('training_seed',0))
    ds=getattr(datasets,cfg['dataset']['name'])(cfg['validation']['roots_file'],**cfg['validation']['dataset_args'])
    # Select an actually accepted inference query; do not fabricate confidence
    # to make a disconnected/reject-all module pass the gradient check.
    index=None
    for i,row in enumerate(ds.rows):
        spec=cfg['trainer']['args']['deform_conditioner']
        if spec is None:
            index=i;break
        if spec['implementation']=='matching-v2':
            with np.load(ds.match_root/(row['sample_id']+'.npz')) as z:
                use=z['cached_valid']
                if spec['reject']:use=use&(z['p_use']>=spec['reject_threshold'])
                if use.any():index=i;break
        else:
            item=ds[i]
            if item['transport_target_valid'].any() and item['transport_source_valid'].sum()>=64:
                index=i;break
    if index is None:raise ValueError('matching-v2 accepts no validation queries')
    with torch.device('meta'):
        md={name:getattr(models,spec['name'])(**spec['args']) for name,spec in cfg['models'].items()}
    kw=copy.deepcopy(cfg['trainer']['args']);kw.update(batch_size_per_gpu=1,batch_split=1,num_workers=0,max_steps=1,prefetch_data=False)
    trainer=getattr(trainers,cfg['trainer']['name'])(md,ds,**kw,output_dir=str(output),load_dir=None,step=None,wandb_run=None)
    trainer.p_uncond=0;trainer.models['denoiser'].eval()
    pack=recursive_to_device(ds.collate_fn([ds[index]]),'cuda');pack.pop('_dataset_name',None);pack.pop('_sha256',None)
    target=pack.pop('x_0')
    with torch.no_grad(),torch.autocast('cuda',dtype=trainer.mix_precision_dtype):
        actual=trainer.get_inference_cond(**pack)['cond']
        saved=trainer.deform_conditioner_config;trainer.deform_conditioner_config=None
        native=trainer.get_inference_cond(**{k:v for k,v in pack.items() if not k.startswith('transport_')})['cond']
        trainer.deform_conditioner_config=saved
    errors={k:float((actual[k]-native[k]).abs().max()) for k in actual}
    if max(errors.values())!=0:raise ValueError('matching-v2 initial native condition differs')
    with torch.autocast('cuda',dtype=trainer.mix_precision_dtype):loss,_=trainer.training_losses(x_0=target,**pack)
    loss['loss'].backward()
    if code_hashes!={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}:
        raise RuntimeError('native proof source changed while the contract was executing')
    lora_gradient=sum(float(parameter.grad.float().abs().sum()) for name,parameter in
        trainer.models['denoiser'].named_parameters() if 'lora_' in name and parameter.grad is not None)
    if not np.isfinite(lora_gradient) or lora_gradient<=0:
        raise ValueError('native-start proof has no finite LoRA gradient')
    if trainer.deform_conditioner is None:
        write(output/'native-proof.json',{'status':'PASS','sample_id':ds.rows[index]['sample_id'],
            'condition_max_abs_errors':errors,'lora_gradient_l1':lora_gradient,
            'fusion_gate_gradient':None,'adapter_output_gradient_l1':None,
            'head_sha256':None,'optimizer_updates':0,'query_channels':None,'actual_source':proof_source})
        return
    gate=float(trainer.deform_conditioner.fusion_gate.grad.abs())
    adapter=trainer.deform_conditioner.retargeter
    gradient=float(adapter.net[-1].weight.grad.abs().sum()) if adapter is not None else None
    if not gate>0 or (gradient is not None and not gradient>0):raise ValueError('matching-v2 has no first-step gradient')
    query_proof=None
    if trainer.deform_conditioner.arm=='c2':
        if adapter.input_dim!=4096:raise ValueError('C2 must physically add RAW2048 channels')
        deformation=pack['transport_deformation'].float()
        valid_query=pack['transport_target_valid']&pack.get('transport_cached_valid',torch.ones_like(pack['transport_target_valid']))
        last=adapter.net[-1];saved_weight=last.weight.detach().clone();direction=last.weight.grad.detach().clone()
        # A zero output layer intentionally blocks first-layer gradients at
        # initialization. A temporary, restored output perturbation verifies
        # the actual query channels without applying any optimizer update.
        try:
            with torch.no_grad():last.weight.add_(1e-3*direction)
            for model in trainer.models.values():model.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=trainer.mix_precision_dtype):probe,_=trainer.training_losses(x_0=target,**pack)
            probe['loss'].backward();query_gradient=float(adapter.net[0].weight.grad[:,-2048:].abs().sum())
            if not query_gradient>0 or not np.isfinite(query_gradient):raise ValueError('C2 query channels are disconnected')
            query_proof={'input_dim':adapter.input_dim,'extra_channels':2048,'deformation_rms':float(deformation.square().mean().sqrt()),
                'valid_deformation_rms':float(deformation[valid_query].square().mean().sqrt()),
                'query_weight_gradient_l1':query_gradient,'temporary_output_probe_restored':True,'optimizer_updates':0}
        finally:
            with torch.no_grad():last.weight.copy_(saved_weight)
    task_proof=None
    task=getattr(trainer.deform_conditioner,'task_attention',None)
    if task is not None:
        last=adapter.net[-1];saved_weight=last.weight.detach().clone();direction=last.weight.grad.detach().clone()
        try:
            with torch.no_grad():last.weight.add_(1e-3*direction.sign())
            for model in trainer.models.values():model.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=trainer.mix_precision_dtype):probe,_=trainer.training_losses(x_0=target,**pack)
            probe['loss'].backward()
            parts={'decoder':trainer.deform_conditioner.q2c.decoder,
                   'score':trainer.deform_conditioner.q2c.fusion_head,'task_attention':task}
            gradients={name:sum(float(p.grad.detach().float().abs().sum()) for p in module.parameters() if p.grad is not None)
                       for name,module in parts.items()}
            task_config=trainer.deform_conditioner.task_attention_config
            required={'decoder':task_config.get('train_decoder',task_config['decoder']),
                      'score':task_config.get('train_score',task_config['score_prior']),'task_attention':True}
            if any(not np.isfinite(v) or (v>0)!=required[k] for k,v in gradients.items()):
                raise ValueError('actual SS-loss task branch gradient differs from decoder/score switches')
            task_proof={'actual_SS_loss_gradient_L1':gradients,'required_active':required,
                        'physical_decoder_blocks':len(parts['decoder'].blocks),
                        'temporary_output_probe_not_optimizer_update':True,'optimizer_updates':0}
        finally:
            with torch.no_grad():last.weight.copy_(saved_weight)
        if not torch.equal(last.weight,saved_weight):raise ValueError('task gradient probe did not restore its output weight')
        task_proof['temporary_output_weight_restored_exact']=True
    if code_hashes!={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}:
        raise RuntimeError('native proof source changed during branch probes')
    write(output/'native-proof.json',{'status':'PASS','sample_id':ds.rows[index]['sample_id'],
        'condition_max_abs_errors':errors,'fusion_gate_gradient':gate,'adapter_output_gradient_l1':gradient,
        'lora_gradient_l1':lora_gradient,
        'head_sha256':cfg['experiment']['head_sha256'],'optimizer_updates':0,'query_channels':query_proof,
        'actual_source':proof_source,'task_branch_probe':task_proof})


def audit_memory_probe(byrank,per_rank,world,split):
    if len(byrank)!=world:raise ValueError('memory probe lacks a rank')
    for records in byrank:
        if [r['step'] for r in records]!=[1,2]:raise ValueError('memory probe requires two distinct updates per rank')
        if any(r['world_size']!=world or r['effective_batch_size']!=per_rank*world
            or r['microbatch_assets']!=[per_rank//split]*split or r['accumulation_packs']!=split for r in records):
            raise ValueError('memory probe execution differs from the declared micro/global batch')
        if any(not np.isfinite(r['loss']['loss']) for r in records):raise ValueError('memory probe has nonfinite loss')
    return max(r['peak_allocated_gib'] for records in byrank for r in records)<=91


def train(a):
    allowed=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    authorized=os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(',')
    if not allowed or len(allowed)!=len(set(allowed)) or any(not g.isdigit() or g not in authorized for g in allowed):
        raise ValueError('distinct physical GPUs explicitly assigned by this campaign are required')
    cfg=json.loads(a.config.read_text());name=cfg['experiment']['arm'];repo=Path(__file__).resolve().parents[2]
    world=len(allowed);per_rank=cfg['trainer']['args']['batch_size_per_gpu']
    if cfg['num_gpus']!=world or per_rank*world!=96:
        raise ValueError('SS launch GPU count and per-rank batch must preserve global96')
    proof=a.root/'verification'/name;proof.mkdir(parents=True,exist_ok=True)
    # Separate process releases all proof allocations before the maximum-batch probe.
    command=[sys.executable,'-u','-m','experiments.deformnext_retargeting.matching_v2_ss',
             '--mode','proof','--root',str(a.root),'--config',str(a.config)]
    with (proof/'proof.log').open('ab') as log:subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
    selected=None
    required=cfg['experiment'].get('required_microbatch_per_gpu')
    if required is not None and (required<=0 or per_rank%required):
        raise ValueError('required matched microbatch must divide the per-rank batch')
    splits=[per_rank//required] if required is not None else [d for d in [1,2,3,4,6,8,12,16,24,48,96] if per_rank%d==0]
    for split in splits:
        bench=copy.deepcopy(cfg);directory=a.root/'memory-bench'/name/f'split{split}';directory.mkdir(parents=True,exist_ok=True)
        bench.update(default_output_dir=str(directory),use_wandb=False)
        bench['trainer']['args'].update(max_steps=2,batch_split=split,num_workers=0,i_save=2,max_checkpoints=1,i_log=1)
        bench['experiment'].update(validate_at_start=False,auto_continue_after_pilot=False)
        bench['validation']['every_steps']=3
        path=directory/'config.json';write(path,bench);logpath=directory/'launch.log'
        with logpath.open('ab') as log:
            done=subprocess.run([sys.executable,'-u','train.py','--config',str(path),'--num_gpus',str(world),'--master_port',str(cfg['master_port']),'--multiview_runtime','--auto_retry','0'],cwd=repo,stdout=log,stderr=subprocess.STDOUT)
        if done.returncode:
            text=logpath.read_text(errors='replace')
            if 'out of memory' in text.lower():continue
            raise RuntimeError('SS memory probe failed; see '+str(logpath))
        byrank=[[json.loads(l) for l in (directory/('runtime_batch.jsonl' if rank==0 else f'runtime_batch.rank{rank}.jsonl')).read_text().splitlines()] for rank in range(world)]
        if not audit_memory_probe(byrank,per_rank,world,split):continue
        selected=split;break
    if selected is None:raise RuntimeError('no verified SS microbatch')
    cfg['trainer']['args']['batch_split']=selected;cfg['experiment']['microbatch_per_gpu']=per_rank//selected
    cfg['experiment']['release_status']='native-and-memory-verified';write(a.config,cfg)
    subprocess.run([sys.executable,'-u','train.py','--config',str(a.config),'--num_gpus',str(world),
        '--master_port',str(cfg['master_port']),'--multiview_runtime','--auto_retry','0'],cwd=repo,check=True)
    progress=json.loads((Path(cfg['default_output_dir'])/'progress.json').read_text())
    if progress['stage']!='completed' or progress['current']!=480:raise RuntimeError('SS run ended before five epochs')
    write(Path(cfg['default_output_dir'])/'completed.json',{'status':'PASS','updates':480,'epochs':5,'microbatch':per_rank//selected,'world_size':world,'global_batch':96,'head_sha256':cfg['experiment']['head_sha256']})


def resume_spec(cfg, root):
    """Validate a stopped, complete run before removing its stop sentinel."""
    import torch
    output=Path(cfg['default_output_dir'])
    if output.resolve()!=(root/'runs_ss64'/cfg['experiment']['arm']).resolve():
        raise ValueError('resume output must belong to the selected arm')
    saved=json.loads((output/'config.json').read_text())
    if cfg.get('training_seed',0)!=saved.get('training_seed',0):
        raise ValueError('resume recipe changed: training_seed')
    for key in ['models','dataset','trainer','validation','data_dir','num_gpus','wandb_id']:
        if cfg.get(key)!=saved.get(key):raise ValueError('resume recipe changed: '+key)
    progress=json.loads((output/'progress.json').read_text());step=progress['current']
    if progress['stage']!='stopped' or not 0<step<cfg['trainer']['args']['max_steps']:
        raise ValueError('resume requires a stopped, incomplete run')
    if progress['last_checkpoint_step']!=step or not (output/'STOP').is_file():
        raise ValueError('resume requires a complete stop checkpoint and STOP sentinel')
    paths={name:output/'ckpts'/f'{name}_step{step:07d}.pt' for name in ['denoiser','deform_conditioner','misc']}
    if any(not path.is_file() or not path.stat().st_size for path in paths.values()):
        raise ValueError('resume checkpoint set is incomplete')
    misc=torch.load(paths['misc'],map_location='cpu',weights_only=False)
    args=cfg['trainer']['args']
    required={'step','optimizer','data_sampler','lora','lr_scheduler','grad_clip'}
    if not required.issubset(misc) or misc['step']!=step or misc['lora']!=args['lora']:
        raise ValueError('resume optimizer/sampler/LoRA state is incomplete or incompatible')
    if not misc['optimizer']['state']:raise ValueError('resume optimizer moments are empty')
    pinned=copy.deepcopy(cfg)
    pinned.update(ckpt=str(step),load_dir=str(output),output_dir=str(output),auto_retry=0)
    receipt={'status':'validated','resume_step':step,'restores':['model','conditioner','optimizer','scheduler','sampler','gradient_clip'],
        'sampler':misc['data_sampler'],'optimizer_learning_rates':[g['lr'] for g in misc['optimizer']['param_groups']],
        'rng_limitation':'native checkpoint does not store per-rank RNG; this is stateful resume, not bitwise replay',
        'world_size':cfg['num_gpus'],'global_batch':96,'head_sha256':cfg['experiment']['head_sha256']}
    return output,pinned,receipt


def resume(a):
    cfg=json.loads(a.config.read_text())
    allowed=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    authorized=os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(',')
    if not allowed or len(set(allowed))!=len(allowed) or any(not g.isdigit() or g not in authorized for g in allowed):
        raise ValueError('distinct physical GPUs explicitly assigned by this campaign are required')
    args=cfg['trainer']['args'];world=len(allowed)
    if world!=cfg['num_gpus'] or args['batch_size_per_gpu']*world!=96:
        raise ValueError('resume must preserve the verified GPU count and global96')
    output,pinned,receipt=resume_spec(cfg,a.root)
    spec=output/'resume-config.json';write(spec,pinned)
    write(output/'resume-request.json',receipt)
    (output/'STOP').rename(output/f'STOP.before-resume-step{receipt["resume_step"]:07d}')
    repo=Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable,'-u','train.py','--config',str(spec),'--num_gpus',str(world),
        '--ckpt',str(receipt['resume_step']),'--load_dir',str(output),'--master_port',str(cfg['master_port']),
        '--multiview_runtime','--auto_retry','0'],cwd=repo,check=True)
    progress=json.loads((output/'progress.json').read_text())
    if progress['stage']!='completed' or progress['current']!=480:raise RuntimeError('resumed SS run ended before five epochs')
    write(output/'completed.json',{'status':'PASS','updates':480,'epochs':5,'microbatch':args['batch_size_per_gpu']//args['batch_split'],
        'world_size':world,'global_batch':96,'head_sha256':cfg['experiment']['head_sha256'],'resumed_from_step':receipt['resume_step']})


def promote(root):
    baseline=json.loads((root/'evaluations/v1/native/evaluation.json').read_text())
    byid={r['sample_id']:r for r in baseline['per_asset']}
    records={}
    for name in ['b2-v2','b2-v2-r','b3-v2']:
        row=json.loads((root/'evaluations'/name/'evaluation.json').read_text())
        if set(byid)!={r['sample_id'] for r in row['per_asset']}:raise ValueError('promotion panels differ')
        groups={}
        for r in row['per_asset']:
            base=byid[r['sample_id']];key=r['source_cluster_id']
            counts=np.array([r['tp'],r['fp'],r['fn'],base['tp'],base['fp'],base['fn']],dtype=float)
            groups[key]=groups.get(key,np.zeros(6))+counts
        arrays=np.stack(list(groups.values()));rng=np.random.default_rng(42017)
        sampled=arrays[rng.integers(0,len(arrays),size=(2000,len(arrays)))].sum(1)
        differences=sampled[:,0]/sampled[:,:3].sum(1)-sampled[:,3]/sampled[:,3:].sum(1)
        records[name]={'iou':row['quick_metrics']['voxel_iou'],'gain_vs_v1':row['quick_metrics']['voxel_iou']-baseline['quick_metrics']['voxel_iou'],
                       'paired_cluster_bootstrap_ci95':np.quantile(differences,[.025,.975]).tolist()}
    best=max(records,key=lambda n:records[n]['iou'])
    passed=best=='b3-v2' and records['b3-v2']['paired_cluster_bootstrap_ci95'][0]>0
    decision={'status':'PASS' if passed else 'HOLD','best_matching_arm':best,'comparisons':records,
              'reason':'B3 improves V1 with positive paired cluster CI' if passed else 'C arms held: matching gain/parent needs review; no full model promoted',
              'extra_seed_replication':'required before claiming a small final gain; this one-seed gate only permits C experiments'}
    write(root/'c-promotion.json',decision);print(json.dumps(decision),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['oracle-select','oracle-assess','configure','proof','train','resume','promote'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--work-root',type=Path)
    p.add_argument('--config-root',type=Path);p.add_argument('--legacy-root',type=Path);p.add_argument('--config',type=Path)
    a=p.parse_args()
    if a.mode=='oracle-select':select_oracle(a.root)
    elif a.mode=='oracle-assess':assess_oracle(a)
    elif a.mode=='configure':configure(a)
    elif a.mode=='proof':
        cfg=json.loads(a.config.read_text());out=a.root/'verification'/cfg['experiment']['arm'];out.mkdir(parents=True,exist_ok=True);native_proof(cfg,out)
    elif a.mode=='train':train(a)
    elif a.mode=='resume':resume(a)
    else:promote(a.root)


if __name__=='__main__':main()
