"""Dedicated-GPU version of the native R-only PBR probe; only allocator limit differs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from easydict import EasyDict
import torch
import torch.multiprocessing as mp


def worker(rank,config):
    from pixal3d import trainers
    from experiments.multiview_finetune.runtime import install_hooks,evaluate
    from .retarget_only_pbr_runtime import RetargetOnlyPBRTrainer
    from .retarget_only_pbr_contract import ARM,validate_profile
    from .task_ray_science_contract import closed_json
    from .pbr_stage_metrics import install_pbr_stage_metrics
    from .evaluate_pbr_streaming import tensor_sha
    from .frozen_projection_streaming import install_frozen_projection_streaming
    import train as native

    torch.cuda.set_device(rank);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if config._R_only_execution_mode=='profile':
        torch.cuda.set_per_process_memory_fraction(.40,rank)
    install_frozen_projection_streaming(max_assets=1,
        audit_directory=Path(config.output_dir)/'projection-streaming-audit')
    install_hooks();install_pbr_stage_metrics()
    selected=RetargetOnlyPBRTrainer
    if config._R_only_execution_mode!='train':
        class ProfileTrainer(RetargetOnlyPBRTrainer):
            def run(self):
                def frozen_hash(module):
                    trainable={name for name,p in module.named_parameters() if p.requires_grad}
                    digest=hashlib.sha256()
                    for name,value in module.state_dict().items():
                        if name not in trainable:
                            digest.update(name.encode())
                            digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
                    return digest.hexdigest()
                before={name:frozen_hash(module) for name,module in self.models.items()}
                initial=len(self.optimizer.state)
                snapshots={id(p):p.detach().cpu().clone() for g in self.optimizer.param_groups for p in g['params']}
                spec=self.pbr_stage;capacity=spec['capacity_profile']
                sid=spec['capacity_sample_id'] if capacity else spec['diagnostic_sample_id']
                indices=[i for i,row in enumerate(self.dataset.rows) if row['sample_id']==sid]
                if len(indices)!=1 or self.dataset.rows[indices[0]]['split']!='train':
                    raise ValueError('one declared real TRAIN asset is required')
                item=self.dataset[indices[0]]
                if any(k in item for k in ['transport_deformation','transport_target_context','transport_source_context','transport_spatial_indices']):
                    raise ValueError('R-only must physically omit RAW/GT deformation and spatial teacher')
                tokens=len(item['coords'])
                if capacity and tokens!=31900:
                    raise ValueError('same native maximum31900-token asset is required')
                updates=[];inputs={};predictions=[];start=time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                def update():
                    began=time.monotonic();data=self.dataset.collate_fn([item]*self.batch_size_per_gpu,split_size=1)
                    self._load_seconds=time.monotonic()-began
                    if not updates:
                        inputs.update({key:tensor_sha(data[0][key]) for key in ['cond','x_0','concat_cond']})
                        handle=self.models['denoiser'].register_forward_hook(lambda module,args,output:predictions.append(tensor_sha(output)))
                    try:result=self.run_step(data)
                    finally:
                        if not updates:handle.remove()
                    self.step+=1
                    gradients={name:float(p.grad.detach().float().abs().sum()) if p.grad is not None else None
                        for name,p in self.deform_conditioner.named_parameters() if p.requires_grad}
                    if len(gradients)!=6 or any(v is None or not math.isfinite(v) for v in gradients.values()):
                        raise ValueError('all6 R gradients must be present and finite')
                    updates.append(dict(step=self.step,loss=result['loss'],seconds=time.monotonic()-began,new_gradient_L1=gradients))
                for _ in range(3):update()
                validated=False
                if capacity:
                    torch.distributed.barrier()
                    if self.is_master:
                        evaluate(self)
                        row=json.loads((Path(self.output_dir)/'validation.jsonl').read_text().splitlines()[-1])
                        if not row['quick_metrics'] or len(row['quick_per_asset'])!=4:
                            raise ValueError('native four-case rendering validation is required')
                    torch.distributed.barrier();update();validated=True
                after={name:frozen_hash(module) for name,module in self.models.items()}
                changed={g['model_name']:sum(not torch.equal(p.detach().cpu(),snapshots[id(p)]) for p in g['params']) for g in self.optimizer.param_groups}
                steps={g['model_name']:sorted({int(self.optimizer.state[p]['step']) for p in g['params']}) for g in self.optimizer.param_groups}
                if before!=after or initial!=360 or len(self.optimizer.state)!=366 or len(predictions)!=1 or any(v<=0 for v in updates[2]['new_gradient_L1'].values()):
                    raise ValueError('own native NN360/R6 learnability and frozen-state proof failed')
                if self.world_size>1:self.check_ddp()
                identity,loss_exact=False,False
                if capacity:
                    root=Path(spec['V1_capacity_root'])
                    reference=closed_json(root/f'PBR-profile.rank{self.rank}.json',root/'file4-closure.json',Path(spec['campaign_root']))
                    if reference['world_size']!=4 or reference['microbatch']!=6 or reference['actual_asset']!=sid:
                        raise ValueError('actual matched V1 world4 capacity reference differs')
                    identity=reference['first_prediction_SHA256']==predictions[0] and reference['input_SHA256']==inputs
                    loss_exact=reference['updates'][0]['loss']==updates[0]['loss']
                report=dict(status='PASS_ACTUAL_RETARGET_ONLY_PBR_DDP4_CAPACITY' if capacity else 'PASS_ACTUAL_RETARGET_ONLY_PBR_MICRO1_PROFILE',
                    arm=ARM,source_step=762,total_budget=1905,initial_optimizer_states=initial,
                    actual_optimizer_states=len(self.optimizer.state),changed_trainable_tensors_by_group=changed,
                    final_optimizer_steps=steps,all_frozen_NN_and_matcher_states_exact=before==after,
                    common_R_exact_to_Full_constructor=self._R_only_initializer['common_R_exact_to_Full_constructor'],
                    common_R_SHA256=self._R_only_initializer['common_R_SHA256'],physical_R_input_channels=3072,
                    attention_and_DQ_physically_absent=True,world_size=self.world_size,microbatch=self.batch_size_per_gpu,
                    global_batch=self.batch_size,accumulation=self.batch_split,post_update_DDP_equal=self.world_size>1,
                    post_validation_full_micro_update_passed=validated,initial_native_prediction_exact_to_V1_capacity=identity,
                    first_loss_fields_exact_to_V1_capacity=loss_exact,actual_asset=sid,actual_asset_tokens=tokens,
                    input_SHA256=inputs,first_prediction_SHA256=predictions[0],temporary_updates=len(updates),
                    scientific_training_updates=0,prototype_discarded=True,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-start,updates=updates,
                    allocator_fraction_request=.40 if not capacity else 1.)
                validate_profile(report,capacity=capacity)
                out=Path(self.output_dir)
                (out/f'R-only-PBR-profile.rank{self.rank}.json').write_text(json.dumps(report,indent=2)+'\n')
                if self.world_size>1:torch.distributed.barrier()
                if self.is_master:(out/'R-only-PBR-profile.json').write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(dict(status=report['status'],peak_GiB=report['peak_allocated_GiB'])),flush=True)
        selected=ProfileTrainer
    trainers.RetargetOnlyPBRTrainer=selected
    native.main(rank,config)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--mode',choices=['profile','capacity','train'],required=True)
    p.add_argument('--num-gpus',type=int,choices=[1,4],required=True);p.add_argument('--master-port',type=int,required=True)
    p.add_argument('--resume-step',type=int)
    a=p.parse_args();visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',');allowed=set(os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','))
    if len(visible)!=a.num_gpus or len(set(visible))!=len(visible) or not set(visible)<={'0','1','2','3'}&allowed:
        raise ValueError('exact physical GPU0-3 allocation is required')
    cfg=EasyDict(json.loads(a.config.read_text()));spec=cfg.trainer.args.pbr_stage
    if (cfg.trainer.name!='RetargetOnlyPBRTrainer' or cfg.training_seed!=101 or cfg.auto_retry!=0
            or cfg.load_dir!='' or cfg.ckpt!='none' or cfg.get('load_ckpt') is not None
            or spec.R_only_mode!=a.mode or spec.diagnostic_profile!=(a.mode!='train')
            or (a.mode=='profile' and a.num_gpus!=1) or (a.mode!='profile' and a.num_gpus!=4)
            or cfg.dataset.name!='DeformNeXtPBR1024' or cfg.validation.dataset_name!='DeformNeXtPBR1024'
            or cfg.dataset.args.transport_arm!='c1' or (a.mode!='train' and cfg.use_wandb)):
        raise ValueError('explicit same-budget R-only native schema and execution mode required')
    cfg.update(config=str(a.config),num_gpus=a.num_gpus,num_nodes=1,node_rank=0,
        master_addr='127.0.0.1',master_port=str(a.master_port),multiview_runtime=True,
        tryrun=False,profile=False,_R_only_execution_mode=a.mode)
    out=Path(cfg.output_dir)
    if a.resume_step is None:
        out.mkdir(parents=True,exist_ok=False)
        (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    else:
        saved=EasyDict(json.loads((out/'config.json').read_text()))
        if (a.mode!='train' or not 762<a.resume_step<1905 or saved.trainer!=cfg.trainer
                or saved.num_gpus!=a.num_gpus or not (out/f'ckpts/misc_step{a.resume_step:07d}.pt').is_file()):
            raise ValueError('own intermediate R-only checkpoint and exact saved world/config required')
        cfg.update(load_dir=str(out),load_ckpt=a.resume_step)
    if a.num_gpus==1:worker(0,cfg)
    else:mp.spawn(worker,args=(cfg,),nprocs=a.num_gpus,join=True)
    if a.mode=='train':
        progress=json.loads((out/'progress.json').read_text())
        if progress['stage']!='completed' or progress['current']!=1905:
            raise ValueError('actual fixed1905 native R-only training must complete')
        for name in ['denoiser','deform_conditioner','misc']:
            if not (out/f'ckpts/{name}_step0001905.pt').is_file():raise ValueError('complete final native checkpoint trio required')
        misc=torch.load(out/'ckpts/misc_step0001905.pt',map_location='cpu',weights_only=False,mmap=True)
        optimizer=misc['optimizer']
        steps={g['model_name']:sorted({int(optimizer['state'][i]['step']) for i in g['params']}) for g in optimizer['param_groups']}
        if len(optimizer['state'])!=366 or steps!=dict(default=[1905],retargeter=[1143]) or misc['lr_scheduler']['last_epoch']!=1143:
            raise ValueError('NN360 continuation and fresh R6 final optimizer steps differ')
        (out/'stage-completed.json').write_text(json.dumps(dict(status='PASS_ACTUAL_RETARGET_ONLY_PBR1905',arm=spec.arm,
            source_step=762,additional_updates=1143,total_updates=1905,optimizer_steps=steps,
            global_batch=24,world_size=4,microbatch=6,accumulation=1,model_adopted=False,full_goal_complete=False),indent=2)+'\n')


if __name__=='__main__':main()
