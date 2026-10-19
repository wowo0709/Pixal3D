"""Explicit matched PBR V1/direct continuation on the user's GPUs0-3."""
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


def worker(local_rank, config):
    from pixal3d import datasets,trainers
    from pixal3d.datasets.deformnext_deformation_query_pbr import DeformNeXtDeformationQueryPBR1024
    from experiments.multiview_finetune.runtime import install_hooks, evaluate
    from .pbr_joint_runtime import PBRJointTrainer
    from .pbr_stage_metrics import install_pbr_stage_metrics
    from .evaluate_pbr_streaming import tensor_sha
    import train as native

    torch.cuda.set_device(local_rank)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if config._PBR_execution_mode == 'profile':
        torch.cuda.set_per_process_memory_fraction(.30,local_rank)
    install_hooks()
    install_pbr_stage_metrics()
    datasets.DeformNeXtDeformationQueryPBR1024 = DeformNeXtDeformationQueryPBR1024
    selected = PBRJointTrainer
    if config._PBR_execution_mode != 'train':
        class ProfileTrainer(PBRJointTrainer):
            def run(self):
                def frozen_hash(module):
                    digest = hashlib.sha256()
                    for name,p in module.named_parameters():
                        if not p.requires_grad:
                            digest.update(name.encode())
                            digest.update(p.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
                    return digest.hexdigest()
                before = {name:frozen_hash(module) for name,module in self.models.items()}
                snapshots = {id(p):p.detach().cpu().clone()
                    for g in self.optimizer.param_groups for p in g['params']}
                initial = len(self.optimizer.state)
                spec = self.pbr_stage
                sid = spec['capacity_sample_id'] if spec.get('capacity_profile') else spec['diagnostic_sample_id']
                indices = [i for i,row in enumerate(self.dataset.rows) if row['sample_id']==sid]
                if len(indices) != 1:
                    raise ValueError('declared actual TRAIN profile asset missing or duplicated')
                item = self.dataset[indices[0]]
                if self.dataset.rows[indices[0]]['split'] != 'train':
                    raise ValueError('profile may use only the declared TRAIN asset')
                tokens = len(item['coords'])
                if spec.get('capacity_profile') and tokens != 31900:
                    raise ValueError('capacity profile must repeat the audited actual maximum31900-token asset')
                inputs = {}
                predictions = []
                updates = []
                torch.cuda.reset_peak_memory_stats()
                began = time.monotonic()

                def update():
                    started = time.monotonic()
                    data = self.dataset.collate_fn([item]*self.batch_size_per_gpu,split_size=1)
                    self._load_seconds = time.monotonic()-started
                    if not updates:
                        inputs.update({key:tensor_sha(data[0][key]) for key in ['cond','x_0','concat_cond']})
                        def capture(module,args,output):
                            predictions.append(tensor_sha(output))
                        handle = self.models['denoiser'].register_forward_hook(capture)
                    try:
                        result = self.run_step(data)
                    finally:
                        if not updates:
                            handle.remove()
                    self.step += 1
                    grads = {name:float(p.grad.detach().float().abs().sum()) if p.grad is not None else None
                        for name,p in self.models.get('deform_conditioner',torch.nn.Module()).named_parameters()
                        if p.requires_grad}
                    if any(v is None or not math.isfinite(v) for v in grads.values()):
                        raise ValueError('present finite native conditioner gradients required')
                    updates.append(dict(step=self.step,loss=result['loss'],seconds=time.monotonic()-started,
                        new_gradient_L1=grads,
                        group_gradient_L1={g['model_name']:sum(float(p.grad.detach().float().abs().sum())
                            for p in g['params'] if p.grad is not None) for g in self.optimizer.param_groups}))

                for _ in range(3):
                    update()
                validation = False
                if spec.get('profile_validation',True):
                    if self.world_size > 1:
                        torch.distributed.barrier()
                    if self.is_master:
                        evaluate(self)
                        record = json.loads((Path(self.output_dir)/'validation.jsonl').read_text().splitlines()[-1])
                        if (not record['quick_metrics'] or len(record['quick_per_asset']) != 4
                                or len(record['quick_panel_ids']) != 4):
                            raise ValueError('actual four-case rendering validation and per-case audits required')
                    if self.world_size > 1:
                        torch.distributed.barrier()
                    update()
                    validation = True
                after = {name:frozen_hash(module) for name,module in self.models.items()}
                changed = {g['model_name']:sum(not torch.equal(p.detach().cpu(),snapshots[id(p)])
                    for p in g['params']) for g in self.optimizer.param_groups}
                steps = {g['model_name']:sorted({int(self.optimizer.state[p]['step']) for p in g['params']})
                    for g in self.optimizer.param_groups}
                direct = spec['arm']=='direct-retargeting-dq'
                count = 376 if direct else 360
                expected_changed = dict(default=360,retargeter=6,ray_readout=9,deformation_query=1) if direct else dict(default=360)
                expected_steps = dict(default=[762+len(updates)])
                if direct:
                    expected_steps.update(retargeter=[len(updates)],ray_readout=[len(updates)],deformation_query=[len(updates)])
                if (before != after or initial != 360 or len(self.optimizer.state) != count
                        or changed != expected_changed or steps != expected_steps
                        or (direct and any(v<=0 for v in updates[2]['new_gradient_L1'].values()))
                        or len(predictions) != 1):
                    raise ValueError('actual matched PBR NN/common-Adam/new16 proof failed')
                if self.world_size > 1:
                    self.check_ddp()
                report = dict(status='PASS_ACTUAL_PBR_JOINT_CAPACITY' if self.world_size>1 else 'PASS_ACTUAL_PBR_JOINT_MICRO1_CORRECTNESS',
                    arm=spec['arm'],source_step=762,total_training_budget=1905,
                    world_size=self.world_size,global_batch=self.batch_size,microbatch=self.batch_size_per_gpu,
                    accumulation=self.batch_split,actual_optimizer_states=count,restored_common_NN_states=initial,
                    changed_trainable_tensors_by_group=changed,final_optimizer_steps=steps,
                    frozen_NN_base_and_matching_exact=True,frozen_SHA256=before,
                    post_update_DDP_equal=self.world_size>1,
                    post_validation_full_micro_update_passed=validation,
                    actual_asset=sid,actual_asset_tokens=tokens,microbatch_assets=self.batch_size_per_gpu,
                    sparse_tokens_per_rank=tokens*self.batch_size_per_gpu,
                    fixed_repeated_TRAIN_batch_not_scientific_sampler_consumption=True,
                    input_SHA256=inputs,first_prediction_SHA256=predictions[0],
                    temporary_updates=len(updates),scientific_training_updates=0,prototype_discarded=True,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                    seconds=time.monotonic()-began,updates=updates,
                    scope='Actual max-token fullbatch plus rendering validation and next update;not real-data throughput'
                          if self.world_size>1 else 'Actual fixed TRAIN micro1 warmstart correctness;not fullbatch capacity or quality.')
                with (Path(self.output_dir)/f'PBR-profile.rank{self.rank}.json').open('x') as stream:
                    json.dump(report,stream,indent=2)
                    stream.write('\n')
                if self.world_size>1:
                    torch.distributed.barrier()
                if self.is_master:
                    with (Path(self.output_dir)/'PBR-profile.json').open('x') as stream:
                        json.dump(report,stream,indent=2)
                        stream.write('\n')
                print(json.dumps(dict(status=report['status'],arm=spec['arm'],peak_GiB=report['peak_allocated_GiB'])),flush=True)
        selected = ProfileTrainer
    trainers.PBRJointTrainer = selected
    native.main(local_rank,config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--mode',choices=['profile','capacity','train'],required=True)
    parser.add_argument('--num-gpus',type=int,choices=[1,2,4],required=True)
    parser.add_argument('--master-port',type=int,required=True)
    parser.add_argument('--resume-step',type=int)
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    allowed = set(os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','))
    if (len(visible)!=args.num_gpus or len(set(visible))!=len(visible)
            or not set(visible).issubset({'0','1','2','3'}&allowed)):
        raise ValueError('exact assigned physical GPU0-3 allocation required')
    cfg = EasyDict(json.loads(args.config.read_text()))
    spec = cfg.trainer.args.pbr_stage
    if (cfg.trainer.name!='PBRJointTrainer' or cfg.get('training_seed')!=101
            or cfg.get('auto_retry')!=0 or cfg.get('load_dir')!='' or cfg.get('ckpt')!='none'
            or cfg.get('load_ckpt') is not None or spec.stage_rng_seed!=101
            or spec.diagnostic_profile!=(args.mode!='train')
            or (args.mode=='profile' and args.num_gpus!=1)
            or (args.mode in {'train','capacity'} and args.num_gpus not in {2,4})
            or (args.mode=='capacity' and (not spec.capacity_profile or not spec.profile_validation))
            or (args.mode!='train' and cfg.use_wandb)):
        raise ValueError('explicit seed101 native PBR stage,mode and no automatic retry required')
    expected_dataset = 'DeformNeXtDeformationQueryPBR1024' if spec.arm=='direct-retargeting-dq' else 'DeformNeXtPBR1024'
    if cfg.dataset.name!=expected_dataset or cfg.validation.dataset_name!=expected_dataset:
        raise ValueError('physical presence/absence of the conditioner dataset must match the arm')
    cfg.update(config=str(args.config),num_gpus=args.num_gpus,num_nodes=1,node_rank=0,
        master_addr='127.0.0.1',master_port=str(args.master_port),multiview_runtime=True,
        tryrun=False,profile=False,_PBR_execution_mode=args.mode)
    output = Path(cfg.output_dir)
    if args.resume_step is None:
        output.mkdir(parents=True,exist_ok=False)
        with (output/'config.json').open('x') as stream:
            json.dump(cfg,stream,indent=2)
            stream.write('\n')
    else:
        if args.mode!='train' or not 762<args.resume_step<1905:
            raise ValueError('only this PBR stage intermediate checkpoint can resume')
        saved = EasyDict(json.loads((output/'config.json').read_text()))
        if (saved.trainer!=cfg.trainer or saved.num_gpus!=args.num_gpus
                or not (output/'ckpts'/f'misc_step{args.resume_step:07d}.pt').is_file()):
            raise ValueError('saved PBR recipe/world/checkpoint differ')
        cfg.update(load_dir=str(output),load_ckpt=args.resume_step)
    if args.num_gpus==1:
        worker(0,cfg)
    else:
        mp.spawn(worker,args=(cfg,),nprocs=args.num_gpus,join=True)
    if args.mode=='train':
        progress = json.loads((output/'progress.json').read_text())
        if progress['stage']=='stopped':
            (output/'stage-stopped.json').write_text(json.dumps(progress,indent=2)+'\n')
            return
        names = ['denoiser','misc']+(['deform_conditioner'] if spec.arm=='direct-retargeting-dq' else [])
        checkpoints = [output/'ckpts'/f'{name}_step0001905.pt' for name in names]
        if progress['stage']!='completed' or progress['current']!=1905 or not all(p.is_file() for p in checkpoints):
            raise ValueError('actual fixed1905 final native PBR checkpoints required')
        (output/'stage-completed.json').write_text(json.dumps(dict(status='PASS_MATCHED_PBR_CONTINUATION1905',
            arm=spec.arm,source_step=762,additional_updates=1143,total_updates=1905,
            global_batch=24,world_size=args.num_gpus,microbatch=24//args.num_gpus,accumulation=1,
            final_checkpoints=[str(p) for p in checkpoints],model_adopted=False,full_goal_complete=False),indent=2)+'\n')


if __name__=='__main__':
    main()
