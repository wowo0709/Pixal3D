"""Matched gated direct-C continuation on explicitly assigned physical GPUs.

Reuse the original per-rank native375-state worker; keep its frozen source
unchanged. The launcher preserves its science configuration and lifecycle,
with explicit GPU assignment and gated-delivery capacity/initial parity proof.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from easydict import EasyDict
import torch.multiprocessing as mp

from .gated_direct_delivery_runtime import POLICY


def worker(local_rank, config):
    physical = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[local_rank]
    os.sched_setaffinity(0, set(config.trainer.args.anchor_stage.CPU_affinity_per_GPU[physical]))
    from pixal3d.modules.direct_ray_retargeting_no_query import DirectRayRetargetingNoQueryConditioner
    from .gated_direct_delivery_runtime import GatedDirectNoQueryJointTrainer
    from . import ray_feature_joint_runtime as runtime
    from .train_ray_feature_joint import worker as native_worker
    original_trainer, original_module = runtime.RayFeatureJointTrainer, runtime.RayFeatureAttentionConditioner
    runtime.RayFeatureJointTrainer = GatedDirectNoQueryJointTrainer
    runtime.RayFeatureAttentionConditioner = DirectRayRetargetingNoQueryConditioner
    try:
        return native_worker(local_rank, config)
    finally:
        runtime.RayFeatureJointTrainer, runtime.RayFeatureAttentionConditioner = original_trainer, original_module


def finalize_gated_profile(config, mode, world):
    if mode == 'train':
        return
    from .historical_target_ray_reproduction import verify_capacity
    out = Path(config.output_dir)
    reports = {rank: json.loads((out/f'joint-profile.rank{rank}.json').read_text()) for rank in range(world)}
    spec = config.trainer.args.anchor_stage
    if mode == 'capacity':
        goldens = {int(rank):json.loads(Path(path).read_text())
                   for rank,path in spec.reference_capacity_profiles.items()}
        verify_capacity(reports, goldens, arm='direct-nodq')
    for rank, report in reports.items():
        events = [json.loads(line) for line in (out/f'gated-delivery-forward-events.rank{rank}.jsonl').read_text().splitlines()]
        first = events[0]
        if (not first['actual_C_is_zero'] or first['original_consumer_exact_when_C_zero'] is not True
                or first['step_before_update'] != 96 or not any(e['original_C_RMS'] > 0 for e in events)
                or any(e['policy'] != POLICY or e['original_R_plus_C_decomposition_exact'] is not True for e in events)):
            raise ValueError('actual gated consumption, initial exact identity and learned feature delivery required')
        report.update(variant='direct-physical-NoDQ', physical_deformation_channels=0,
            deformation_projection_present=False, retargeter_input_channels=2048,
            direct_delivery_policy=POLICY, new_delivery_parameters=0,
            actual_initial97_loss_matches_R96_reference=mode == 'capacity',
            actual_zero_C_consumer_identity=True, actual_gated_delivery_events=events)
        (out/f'joint-profile.rank{rank}.json').write_text(json.dumps(report, indent=2)+'\n')
    (out/'joint-profile.json').write_text(json.dumps(reports[0], indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--assigned-gpus', required=True)
    parser.add_argument('--mode', choices=['train','profile','capacity'], required=True)
    parser.add_argument('--num-gpus', type=int, required=True)
    parser.add_argument('--master-port', type=int, required=True)
    parser.add_argument('--resume-step', type=int)
    args = parser.parse_args()
    assigned = args.assigned_gpus.split(',')
    if len(set(assigned)) != len(assigned) or any(not g.isdigit() for g in assigned):
        raise ValueError('explicit distinct assigned physical GPUs required')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    allowed = set(os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','))
    if (len(visible) != args.num_gpus or len(set(visible)) != len(visible)
            or set(visible) != set(assigned) or not set(visible).issubset(allowed)):
        raise ValueError('exact assigned physical GPUs and worker count required')
    config = EasyDict(json.loads(args.config.read_text()))
    if (config.trainer.name != 'RayFeatureJointTrainer' or config.get('auto_retry') != 0
            or config.get('training_seed') != 101 or config.get('ckpt') != 'none'
            or config.get('load_ckpt') is not None or config.get('load_dir') != ''):
        raise ValueError('explicit seed101 fresh continuation stage without automatic retry required')
    spec = config.trainer.args.anchor_stage
    if spec.get('assigned_GPUs') != assigned or spec.get('direct_delivery_policy') != POLICY:
        raise ValueError('same explicit GPU assignment and gated delivery policy required')
    if (spec.arm != 'ray-feature-attention' or spec.diagnostic_profile != (args.mode != 'train')
            or (args.mode in {'train','capacity'} and args.num_gpus != 2)
            or (args.mode == 'profile' and args.num_gpus != 1)
            or (args.mode != 'train' and config.use_wandb)
            or (args.mode == 'capacity' and not spec.get('capacity_profile',False))):
        raise ValueError('mode, assigned allocation or profile declaration differs')
    config.update(config=str(args.config),num_gpus=args.num_gpus,num_nodes=1,node_rank=0,
        master_addr='127.0.0.1',master_port=str(args.master_port),multiview_runtime=True,
        tryrun=False,profile=False,_feature_execution_mode=args.mode)
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
    finalize_gated_profile(config, args.mode, args.num_gpus)
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
            status='PASS_MATCHED_RAY_FEATURE_JOINT480',arm=spec.arm,source_step=96,
            additional_updates=384,total_updates=480,global_batch=96,
            microbatch_per_rank=48,accumulation=1,world_size=2,
            final_checkpoint_trio=[str(p) for p in checkpoints],model_adopted=False,
            full_goal_complete=False),indent=2)+'\n')


if __name__ == '__main__':
    main()
