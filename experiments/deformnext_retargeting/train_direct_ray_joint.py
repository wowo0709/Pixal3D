"""Launch the direct-DQ2048 matched SS arm on assigned GPUs0-3."""
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
    from pixal3d import trainers
    from experiments.multiview_finetune.runtime import install_hooks
    from .direct_ray_joint_runtime import DirectRayJointTrainer
    from .deformation_query_metrics import install_metric_dispatch
    install_metric_dispatch()
    from pixal3d import datasets
    from pixal3d.datasets.deformnext_deformation_query import DeformNeXtDeformationQuerySS64
    datasets.DeformNeXtDeformationQuerySS64 = DeformNeXtDeformationQuerySS64
    import train as native

    torch.cuda.set_device(local_rank)
    torch.set_num_threads(2)
    if config._DQ_execution_mode == 'profile':
        torch.cuda.set_per_process_memory_fraction(.2, local_rank)
    install_hooks()
    selected = DirectRayJointTrainer
    if config._DQ_execution_mode != 'train':
        class ProfileTrainer(DirectRayJointTrainer):
            def run(self):
                def frozen_hash(module):
                    digest = hashlib.sha256()
                    for name,tensor in module.named_parameters():
                        if not tensor.requires_grad:
                            digest.update(name.encode())
                            digest.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
                    return digest.hexdigest()
                before = {name:frozen_hash(module) for name,module in self.models.items()}
                snapshots = {id(p):p.detach().cpu().clone()
                    for group in self.optimizer.param_groups for p in group['params']}
                initial = len(self.optimizer.state)
                torch.cuda.reset_peak_memory_stats()
                began = time.monotonic()
                updates = []
                for _ in range(3):
                    started = time.monotonic()
                    self._load_seconds = 0.
                    result = self.run_step(self.load_data())
                    self.step += 1
                    feature_grads = {name:float(p.grad.detach().float().abs().sum())
                        if p.grad is not None else None
                        for name,p in self.deform_conditioner.named_parameters()
                        if name.startswith('ray_readout.') or name == 'retargeter.deformation_projection.weight'}
                    updates.append(dict(step=self.step, loss=result['loss'],
                        seconds=time.monotonic()-started, feature_gradient_L1=feature_grads,
                        group_gradient_L1={g['model_name']:sum(float(p.grad.detach().float().abs().sum())
                            for p in g['params'] if p.grad is not None) for g in self.optimizer.param_groups}))
                after = {name:frozen_hash(module) for name,module in self.models.items()}
                changed = {g['model_name']:sum(not torch.equal(p.detach().cpu(), snapshots[id(p)])
                    for p in g['params']) for g in self.optimizer.param_groups}
                steps = {g['model_name']:sorted({int(self.optimizer.state[p]['step']) for p in g['params']})
                    for g in self.optimizer.param_groups}
                feature_learns = all(value is not None and math.isfinite(value) and value > 0
                    for value in updates[-1]['feature_gradient_L1'].values())
                if (before != after or initial != 366 or len(self.optimizer.state) != 376
                        or changed['default'] != 360 or changed['deform_conditioner'] != 6
                        or changed['ray_readout'] != 9 or changed['deformation_query'] != 1 or not feature_learns
                        or len(updates[-1]['feature_gradient_L1']) != 10
                        or steps != {'default':[99], 'deform_conditioner':[99], 'ray_readout':[3], 'deformation_query':[3]}
                        or any(not math.isfinite(value) for row in updates
                               for value in row['group_gradient_L1'].values())):
                    raise ValueError('actual376-state NN/R/attention/DQ joint update proof failed')
                if self.world_size > 1:
                    self.check_ddp()
                report = dict(status='PASS_ACTUAL_MATCHED_ANCHOR_JOINT_SS_PROFILE',
                    arm='direct-ray-retargeting-dq', world_size=self.world_size,
                    microbatch=self.batch_size_per_gpu, accumulation=self.batch_split,
                    allocator_fraction=.2 if self.world_size == 1 else 1.,
                    post_update_DDP_equal=self.world_size > 1,
                    restored_common_Adam_states=initial, actual_optimizer_states=len(self.optimizer.state),
                    changed_trainable_tensors_by_group=changed, final_optimizer_steps_by_group=steps,
                    all_ten_new_tensors_learn=feature_learns,
                    physical_retargeter_input_channels=self.deform_conditioner.retargeter.input_dim,
                    actual_SS_and_R_gradients_and_updates=True, GT_in_model_inputs=False,
                    NN_base_and_matching_frozen_exact=before == after, frozen_state_SHA256=before,
                    profile_model_discarded=True, scientific_training_updates=0,
                    stage_initial_step=96, temporary_updates=3,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                    seconds=time.monotonic()-began, updates=updates,
                    scope='ExclusiveDDP2/micro48 capacity' if self.world_size > 1
                          else 'Disposable actual micro2 correctness;not dedicated-pair capacity')
                with (Path(self.output_dir)/f'joint-profile.rank{self.rank}.json').open('x') as stream:
                    json.dump(report, stream, indent=2);stream.write('\n')
                if self.world_size > 1:
                    torch.distributed.barrier()
                if self.is_master:
                    with (Path(self.output_dir)/'joint-profile.json').open('x') as stream:
                        json.dump(report, stream, indent=2);stream.write('\n')
                print(json.dumps(report), flush=True)
        selected = ProfileTrainer
    trainers.DirectRayJointTrainer = selected
    native.main(local_rank, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=['train','profile','capacity'], required=True)
    parser.add_argument('--num-gpus', type=int, required=True)
    parser.add_argument('--master-port', type=int, required=True)
    parser.add_argument('--resume-step', type=int)
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    allowed = set(os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','))
    if (len(visible) != args.num_gpus or len(set(visible)) != len(visible)
            or not set(visible).issubset({'0','1','2','3'} & allowed)):
        raise ValueError('exact assigned GPUs0-3 and worker count required')
    config = EasyDict(json.loads(args.config.read_text()))
    if (config.dataset.name != 'DeformNeXtDeformationQuerySS64'
            or config.validation.dataset_name != 'DeformNeXtDeformationQuerySS64'
            or config.experiment.arm != 'direct-ray-retargeting-dq'
            or config.trainer.name != 'DirectRayJointTrainer' or config.get('auto_retry') != 0
            or config.get('training_seed') != 101 or config.get('ckpt') != 'none'
            or config.get('load_ckpt') is not None or config.get('load_dir') != ''):
        raise ValueError('explicit seed101 fresh continuation stage without automatic retry required')
    spec = config.trainer.args.anchor_stage
    if ((spec.arm != 'ray-feature-attention' or spec.get('variant') != 'direct-DQ2048') or spec.diagnostic_profile != (args.mode != 'train')
            or (args.mode in {'train','capacity'} and args.num_gpus != 2)
            or (args.mode == 'profile' and args.num_gpus != 1)
            or (args.mode != 'train' and config.use_wandb)
            or (args.mode == 'capacity' and not spec.get('capacity_profile',False))):
        raise ValueError('mode, assigned allocation or profile declaration differs')
    config.update(config=str(args.config),num_gpus=args.num_gpus,num_nodes=1,node_rank=0,
        master_addr='127.0.0.1',master_port=str(args.master_port),multiview_runtime=True,
        tryrun=False,profile=False,_DQ_execution_mode=args.mode)
    output = Path(config.output_dir)
    if args.resume_step is None:
        output.mkdir(parents=True,exist_ok=False)
        with (output/'config.json').open('x') as stream:
            json.dump(config,stream,indent=2);stream.write('\n')
    else:
        if args.mode != 'train' or not 96 < args.resume_step < 480:
            raise ValueError('only this stage intermediate checkpoint can resume')
        saved = EasyDict(json.loads((output/'config.json').read_text()))
        if (saved.trainer != config.trainer or saved.training_seed != config.training_seed
                or saved.num_gpus != args.num_gpus
                or not (output/'ckpts'/f'misc_step{args.resume_step:07d}.pt').is_file()):
            raise ValueError('declared resume checkpoint or recipe differs')
        config.update(load_dir=str(output),load_ckpt=args.resume_step)
    if args.num_gpus == 1:
        worker(0,config)
    else:
        mp.spawn(worker,args=(config,),nprocs=args.num_gpus,join=True)
    if args.mode == 'train':
        progress = json.loads((output/'progress.json').read_text())
        if progress['stage'] == 'stopped':
            (output/'stage-stopped.json').write_text(json.dumps(progress,indent=2)+'\n')
            return
        checkpoints = [output/'ckpts'/f'{name}_step0000480.pt'
            for name in ['denoiser','deform_conditioner','misc']]
        if (progress['stage'] != 'completed' or progress['current'] != 480
                or not all(path.is_file() for path in checkpoints)):
            raise ValueError('declared total480 and native final checkpoint trio required')
        (output/'stage-completed.json').write_text(json.dumps(dict(
            status='PASS_MATCHED_DIRECT_RAY_JOINT480',arm='direct-ray-retargeting-dq',source_step=96,
            additional_updates=384,total_updates=480,global_batch=96,
            microbatch_per_rank=48,accumulation=1,world_size=2,
            final_checkpoint_trio=[str(p) for p in checkpoints],model_adopted=False,
            full_goal_complete=False),indent=2)+'\n')


if __name__ == '__main__':
    main()
