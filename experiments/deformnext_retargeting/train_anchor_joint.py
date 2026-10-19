"""Explicit GPU launcher for the matched target-ray SS continuation stage."""
from __future__ import annotations

import argparse
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
    from .anchor_joint_runtime import AnchorJointTrainer
    import train as native

    torch.cuda.set_device(local_rank)
    if config._anchor_execution_mode == 'profile':
        torch.cuda.set_per_process_memory_fraction(.2, local_rank)
    install_hooks()
    torch.set_num_threads(2)
    selected = AnchorJointTrainer
    if config._anchor_execution_mode in {'profile', 'capacity'}:
        class ProfileTrainer(AnchorJointTrainer):
            def run(self):
                import hashlib

                def frozen_hash(module):
                    digest = hashlib.sha256()
                    for name, tensor in module.named_parameters():
                        if not tensor.requires_grad:
                            digest.update(name.encode())
                            digest.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
                    return digest.hexdigest()

                before = {name: frozen_hash(module) for name, module in self.models.items()}
                trainable_before = {id(p): p.detach().cpu().clone()
                    for group in self.optimizer.param_groups for p in group['params']}
                optimizer_states_initial = len(self.optimizer.state)
                torch.cuda.reset_peak_memory_stats()
                began = time.monotonic()
                updates = []
                for number in range(3):
                    # The second temporary update explicitly probes the active
                    # utility branch. Its weights and gate are discarded.
                    if number == 1 and hasattr(self.deform_conditioner, 'ray_mix'):
                        with torch.no_grad():
                            self.deform_conditioner.ray_mix.fill_(.2)
                    started = time.monotonic()
                    self._load_seconds = 0.
                    result = self.run_step(self.load_data())
                    self.step += 1
                    updates.append(dict(step=self.step, loss=result['loss'],
                        seconds=time.monotonic()-started,
                        group_gradient_L1={group['model_name']: sum(float(p.grad.detach().float().abs().sum())
                            for p in group['params'] if p.grad is not None) for group in self.optimizer.param_groups},
                        gate=float(self.deform_conditioner.ray_mix.detach())
                            if hasattr(self.deform_conditioner, 'ray_mix') else None))
                after = {name: frozen_hash(module) for name, module in self.models.items()}
                if before != after or optimizer_states_initial != 366:
                    raise ValueError('frozen weights or common Adam initialization differs')
                expected = 372 if hasattr(self.deform_conditioner, 'ray_mix') else 366
                if len(self.optimizer.state) != expected:
                    raise ValueError('actual joint optimizer state count differs')
                changed = {group['model_name']: sum(not torch.equal(p.detach().cpu(), trainable_before[id(p)])
                    for p in group['params']) for group in self.optimizer.param_groups}
                state_steps = {group['model_name']: sorted({int(self.optimizer.state[p]['step'])
                    for p in group['params']}) for group in self.optimizer.param_groups}
                if (changed['default'] == 0 or changed['deform_conditioner'] == 0
                        or any(not math.isfinite(value) for row in updates
                               for value in row['group_gradient_L1'].values())):
                    raise ValueError('actual NN LoRA/R update or gradient proof failed')
                if self.world_size > 1:
                    self.check_ddp()
                result = dict(status='PASS_ACTUAL_MATCHED_ANCHOR_JOINT_SS_PROFILE',
                    arm=self.anchor_stage['arm'], world_size=self.world_size,
                    microbatch=self.batch_size_per_gpu, accumulation=self.batch_split,
                    allocator_fraction=.2 if self.world_size == 1 else 1.,
                    post_update_DDP_equal=self.world_size > 1,
                    restored_common_Adam_states=optimizer_states_initial,
                    actual_optimizer_states=len(self.optimizer.state), temporary_updates=3,
                    changed_trainable_tensors_by_group=changed, final_optimizer_steps_by_group=state_steps,
                    actual_SS_and_R_gradients_and_updates=True, GT_in_model_inputs=False,
                    NN_base_and_matching_frozen_exact=before == after,
                    frozen_state_SHA256=before, profile_model_discarded=True,
                    scientific_training_updates=0, stage_initial_step=96,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                    seconds=time.monotonic()-began, updates=updates,
                    scope='Disposable actual joint updates;exclusive DDP2/micro48 capacity' if self.world_size > 1
                          else 'Disposable micro2 cohost correctness proof;not exclusive DDP2/micro48 capacity.')
                with (Path(self.output_dir)/f'joint-profile.rank{self.rank}.json').open('x') as stream:
                    json.dump(result, stream, indent=2)
                    stream.write('\n')
                if self.world_size > 1:
                    torch.distributed.barrier()
                if self.is_master:
                    with (Path(self.output_dir)/'joint-profile.json').open('x') as stream:
                        json.dump(result, stream, indent=2)
                        stream.write('\n')
                print(json.dumps(result), flush=True)
        selected = ProfileTrainer
    # Every spawned worker installs its own constructor. No frozen registry edit.
    trainers.AnchorJointTrainer = selected
    native.main(local_rank, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=['train', 'profile', 'capacity'], required=True)
    parser.add_argument('--num-gpus', type=int, required=True)
    parser.add_argument('--master-port', type=int, required=True)
    parser.add_argument('--resume-step', type=int)
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    allowed = set(os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(','))
    if (len(visible) != args.num_gpus or len(set(visible)) != len(visible)
            or not set(visible).issubset({'0', '1', '2', '3'} & allowed)):
        raise ValueError('explicitly assigned physical GPU0-3 and exact worker count required')
    config = EasyDict(json.loads(args.config.read_text()))
    if (config.trainer.name != 'AnchorJointTrainer' or config.get('auto_retry') != 0
            or config.get('training_seed') != 101 or config.get('ckpt') != 'none'
            or config.get('load_ckpt') is not None or config.get('load_dir') != ''):
        raise ValueError('explicit fresh stage declaration with seed101 and no automatic retry required')
    spec = config.trainer.args.anchor_stage
    if spec.diagnostic_profile != (args.mode != 'train'):
        raise ValueError('profile and scientific configurations must remain distinct')
    if args.mode in {'train', 'capacity'} and args.num_gpus != 2:
        raise ValueError('scientific stage requires assigned DDP pair')
    if args.mode == 'profile' and (args.num_gpus != 1 or config.use_wandb):
        raise ValueError('disposable profile requires single GPU and no scientific W&B run')
    if args.mode == 'capacity' and (not spec.get('capacity_profile', False) or config.use_wandb):
        raise ValueError('dedicated capacity declaration and no scientific W&B run required')
    config.update(config=str(args.config), num_gpus=args.num_gpus, num_nodes=1, node_rank=0,
        master_addr='127.0.0.1', master_port=str(args.master_port), multiview_runtime=True,
        tryrun=False, profile=False, _anchor_execution_mode=args.mode)
    output = Path(config.output_dir)
    if args.resume_step is None:
        output.mkdir(parents=True, exist_ok=False)
        with (output/'config.json').open('x') as stream:
            json.dump(config, stream, indent=2)
            stream.write('\n')
    else:
        if args.mode != 'train' or not 96 < args.resume_step < 480:
            raise ValueError('only an intermediate checkpoint of this scientific stage may resume')
        saved = EasyDict(json.loads((output/'config.json').read_text()))
        if (saved.trainer != config.trainer or saved.training_seed != config.training_seed
                or saved.num_gpus != args.num_gpus):
            raise ValueError('resume recipe or allocation differs from this run')
        if not (output/'ckpts'/f'misc_step{args.resume_step:07d}.pt').is_file():
            raise ValueError('the declared native stage checkpoint is missing')
        config.update(load_dir=str(output), load_ckpt=args.resume_step)
    if args.num_gpus == 1:
        worker(0, config)
    else:
        mp.spawn(worker, args=(config,), nprocs=args.num_gpus, join=True)
    if args.mode == 'train':
        progress = json.loads((output/'progress.json').read_text())
        if progress['stage'] == 'stopped':
            (output/'stage-stopped.json').write_text(json.dumps(progress, indent=2)+'\n')
            return
        if progress['stage'] != 'completed' or progress['current'] != 480:
            raise ValueError('native stage did not complete the declared total480 budget')
        paths = [output/'ckpts'/f'{name}_step0000480.pt' for name in ['denoiser','deform_conditioner','misc']]
        if not all(path.is_file() for path in paths):
            raise ValueError('complete native final checkpoint trio required')
        (output/'stage-completed.json').write_text(json.dumps(dict(status='PASS_MATCHED_ANCHOR_JOINT480',
            arm=spec.arm, source_step=96, additional_updates=384, total_updates=480,
            global_batch=96, microbatch_per_rank=48, accumulation=1, world_size=2,
            final_checkpoint_trio=[str(path) for path in paths], model_adopted=False,
            full_goal_complete=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
