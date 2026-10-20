"""Run the first96 additional updates; preserve the declared480-step trajectory.

Only the loop boundary changes. Model, data, objective, optimizer and LR
schedule use the already verified full480 recipe. Native192 checkpoints
retain Adam/RNG/sampler/scheduler for a later full-stage continuation.
"""
import argparse
import json
import os
from pathlib import Path

from easydict import EasyDict
import torch
import torch.multiprocessing as mp


def worker(rank, config):
    from pixal3d import trainers, datasets
    from experiments.multiview_finetune.runtime import install_hooks
    from .deformation_query_metrics import install_metric_dispatch
    from .consistent_ray_joint_runtime import ConsistentRayJointTrainer
    from pixal3d.trainers.flow_matching.flow_matching import ImageConditionedProjFlowMatchingCFGTrainer
    from pixal3d.datasets.deformnext_deformation_query import DeformNeXtDeformationQuerySS64
    import train as native

    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    install_hooks()
    install_metric_dispatch()
    datasets.DeformNeXtDeformationQuerySS64 = DeformNeXtDeformationQuerySS64

    class BoundedTrainer(ConsistentRayJointTrainer):
        def run(self):
            if self.step != 96 or self.max_steps != 480:
                raise ValueError('fresh verified R96-to480 stage required')
            self.max_steps = 192
            try:
                # Initial32 and both-rank initial97 parity are already closed.
                # Bypass only AnchorJointTrainer's duplicate initial generation.
                return ImageConditionedProjFlowMatchingCFGTrainer.run(self)
            finally:
                self.max_steps = 480

    trainers.ConsistentRayJointTrainer = BoundedTrainer
    native.main(rank, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--master-port', type=int, required=True)
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if visible != ['2', '3'] or not set(visible).issubset(os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(',')):
        raise ValueError('this bounded segment is assigned physical GPUs2/3')
    cfg = EasyDict(json.loads(args.config.read_text()))
    if (cfg.trainer.name != 'ConsistentRayJointTrainer'
            or cfg.trainer.args.max_steps != 480 or cfg.training_seed != 101
            or cfg.get('load_dir') != '' or cfg.get('load_ckpt') is not None
            or cfg.get('ckpt') != 'none' or cfg.auto_retry != 0
            or cfg.experiment.get('auto_continue_after_pilot') is not False
            or cfg.segment != dict(source_step=96, end_step=192, planned_final_step=480)):
        raise ValueError('explicit first96-update segment of the verified480 recipe required')
    cfg.update(config=str(args.config), num_gpus=2, num_nodes=1, node_rank=0,
        master_addr='127.0.0.1', master_port=str(args.master_port), multiview_runtime=True,
        tryrun=False, profile=False, _DQ_execution_mode='train')
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    mp.spawn(worker, args=(cfg,), nprocs=2, join=True)
    progress = json.loads((output/'progress.json').read_text())
    paths = [output/'ckpts'/f'{name}_step0000192.pt'
             for name in ['denoiser', 'deform_conditioner', 'misc']]
    if progress['stage'] != 'completed' or progress['current'] != 192 or not all(p.is_file() for p in paths):
        raise ValueError('native segment192 completion and checkpoint trio required')
    misc = torch.load(paths[-1], map_location='cpu', weights_only=False)
    optimizer = misc['optimizer']
    if len(optimizer['state']) != 367 or misc['lr_scheduler']['last_epoch'] != 96:
        raise ValueError('all367 Adam states and full-trajectory scheduler position required')
    expected_steps = dict(default=192, deform_conditioner=192, deformation_query=96)
    for group in optimizer['param_groups']:
        steps = {int(optimizer['state'][p]['step']) for p in group['params']}
        if steps != {expected_steps[group['model_name']]}:
            raise ValueError('native Adam continuation counts differ')
    rng = [output/'stage_rng'/f'rank{rank}'/'step0000192.pt' for rank in range(2)]
    if not all(p.exists() for p in rng):
        raise ValueError('both rank RNG/sampler states required for later continuation')
    (output/'segment-completed.json').write_text(json.dumps(dict(
        status='PASS_CONSISTENT_FIRST96_UPDATE_SEGMENT', source_step=96,
        completed_step=192, planned_final_step=480, additional_updates=96,
        full480_completed=False, optimizer_states=367, scheduler_last_epoch=96,
        initial32_reused_from_closed_native_proof=True,
        checkpoint_trio=[str(p) for p in paths], rank_RNG=[str(p) for p in rng],
        global_batch=96, world_size=2, micro_per_rank=48, accumulation=1,
        quality_gain_claimed=False, model_adopted=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
