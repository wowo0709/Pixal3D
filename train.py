import os
import sys
import json
import glob
import argparse
from easydict import EasyDict as edict

import torch
import torch.multiprocessing as mp
import numpy as np
import random

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb' to enable wandb logging.")

from pixal3d import models, datasets, trainers
from pixal3d.utils.dist_utils import setup_dist
from data_toolkit.pipeline import training_manifest


def find_ckpt(cfg):
    # Load checkpoint
    cfg['load_ckpt'] = None
    if cfg.load_dir != '':
        if cfg.ckpt == 'latest':
            files = glob.glob(os.path.join(cfg.load_dir, 'ckpts', 'misc_*.pt'))
            if len(files) != 0:
                cfg.load_ckpt = max([
                    int(os.path.basename(f).split('step')[-1].split('.')[0])
                    for f in files
                ])
        elif cfg.ckpt == 'none':
            cfg.load_ckpt = None
        else:
            cfg.load_ckpt = int(cfg.ckpt)
    return cfg


def setup_rng(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    np.random.seed(rank)
    random.seed(rank)


def get_model_summary(model):
    num_params = 0
    num_trainable_params = 0
    for name, param in model.named_parameters():
        num_params += param.numel()
        if param.requires_grad:
            num_trainable_params += param.numel()
    model_summary = f'Number of parameters: {num_params:,}\n'
    model_summary += f'Number of trainable parameters: {num_trainable_params:,}\n'
    return model_summary


def apply_smoke_overrides(cfg, smoke_steps):
    if smoke_steps is None:
        return cfg
    if smoke_steps not in (1, 10):
        raise ValueError("smoke_steps must be 1 or 10")
    cfg.trainer.args.max_steps = smoke_steps
    cfg.trainer.args.i_log = 1
    cfg.trainer.args.i_sample = 1 if smoke_steps == 1 else 5
    cfg.trainer.args.i_save = smoke_steps
    cfg.trainer.args.snapshot_batch_size = 1
    cfg.trainer.args.snapshot_num_samples = 1
    cfg.trainer.args.num_workers = 0
    cfg.trainer.args.prefetch_data = False
    return cfg


def resolve_output_dirs(config, cli_output_dir=None, cli_load_dir=""):
    default_output_dir = config.get("default_output_dir")
    output_dir = cli_output_dir or default_output_dir
    if not output_dir:
        raise ValueError(
            "output_dir is required: pass --output_dir or set default_output_dir in the config"
        )
    load_dir = cli_load_dir or output_dir
    return output_dir, load_dir


def main(local_rank, cfg):
    # Set up distributed training
    rank = cfg.node_rank * cfg.num_gpus + local_rank
    world_size = cfg.num_nodes * cfg.num_gpus
    if world_size > 1:
        setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)
    
    # Multi-GPU training verification
    print(f'[Rank {rank}/{world_size}] Process started on GPU {local_rank} (cuda:{torch.cuda.current_device()})')
    if rank == 0:
        print(f'\n{"="*60}')
        print(f'Multi-GPU Training Verification:')
        print(f'  - Total GPUs (world_size): {world_size}')
        print(f'  - num_gpus per node: {cfg.num_gpus}')
        print(f'  - num_nodes: {cfg.num_nodes}')
        print(f'{"="*60}\n')

    # Initialize wandb (only on rank 0)
    wandb_run = None
    if rank == 0 and cfg.use_wandb and WANDB_AVAILABLE:
        # Use WANDB_DIR env var (local SSD) to avoid S3 FUSE rename/append issues
        wandb_dir = os.environ.get('WANDB_DIR', cfg.output_dir)
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_name if cfg.wandb_name else os.path.basename(cfg.output_dir),
            config=cfg.__dict__,
            dir=wandb_dir,
            resume="allow",
            id=cfg.wandb_id if cfg.wandb_id else None,
        )
        print(f'Wandb initialized: {wandb_run.url}')

        # Upload config JSON file as wandb artifact
        config_file = cfg.get('config', None)
        if config_file and os.path.isfile(config_file):
            config_artifact = wandb.Artifact(
                name=f"config-{wandb_run.id}",
                type="config",
                description=f"Training config for {wandb_run.name}",
            )
            config_artifact.add_file(config_file, name=os.path.basename(config_file))
            # Also save the resolved full config (with command-line args merged)
            resolved_config_path = os.path.join(cfg.output_dir, 'config_resolved.json')
            with open(resolved_config_path, 'w') as f:
                json.dump(cfg.__dict__, f, indent=4, default=str)
            config_artifact.add_file(resolved_config_path, name='config_resolved.json')
            wandb_run.log_artifact(config_artifact)
            print(f'Uploaded config artifact to wandb: {config_file}')

    # Seed rngs
    setup_rng(rank)

    # Load data
    dataset_kwargs = dict(cfg.dataset.args)
    dataset = getattr(datasets, cfg.dataset.name)(cfg.data_dir, **dataset_kwargs)

    # Print dataset info (only on rank 0)
    if rank == 0:
        print(f'\nDataset: {cfg.dataset.name}, Number of samples: {len(dataset):,}\n')

    # Build model
    model_dict = {
        name: getattr(models, model.name)(**model.args).cuda()
        for name, model in cfg.models.items()
    }

    # Model summary
    if rank == 0:
        for name, backbone in model_dict.items():
            model_summary = get_model_summary(backbone)
            print(f'\n\nBackbone: {name}\n' + model_summary)
            with open(os.path.join(cfg.output_dir, f'{name}_model_summary.txt'), 'w') as fp:
                print(model_summary, file=fp)

    # Build trainer
    trainer = getattr(trainers, cfg.trainer.name)(
        model_dict, dataset, 
        **cfg.trainer.args, 
        output_dir=cfg.output_dir, 
        load_dir=cfg.load_dir, 
        step=cfg.load_ckpt,
        wandb_run=wandb_run,  # Pass wandb run to trainer
    )

    # Train
    if not cfg.tryrun:
        if cfg.profile:
            trainer.profile()
        else:
            trainer.run()
    
    # Close wandb
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    # Arguments and config
    parser = argparse.ArgumentParser()
    ## config
    parser.add_argument('--config', type=str, required=True, help='Experiment config file')
    ## io and resume
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory; defaults to config default_output_dir',
    )
    parser.add_argument('--load_dir', type=str, default='', help='Load directory, default to output_dir')
    parser.add_argument('--ckpt', type=str, default='latest', help='Checkpoint step to resume training, default to latest')
    parser.add_argument(
        '--training_data',
        type=str,
        default=None,
        help='Verified combined training manifest; mutually exclusive with --data_dir',
    )
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
    parser.add_argument('--auto_retry', type=int, default=3, help='Number of retries on error')
    ## dubug
    parser.add_argument('--tryrun', action='store_true', help='Try run without training')
    parser.add_argument('--profile', action='store_true', help='Profile training')
    parser.add_argument('--smoke_steps', type=int, choices=(1, 10))
    ## multi-node and multi-gpu
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
    parser.add_argument('--num_gpus', type=int, default=-1, help='Number of GPUs per node, default to all')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master address for distributed training')
    parser.add_argument('--master_port', type=str, default='12666', help='Port for distributed training')
    ## wandb
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='pixal3d-training', help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default='', help='Wandb run name, default to output_dir basename')
    parser.add_argument('--wandb_id', type=str, default='', help='Wandb run id for resuming')
    opt = parser.parse_args()
    ## Load config
    config = json.load(open(opt.config, 'r'))
    try:
        resolved_data_dir, training_evidence = (
            training_manifest.resolve_training_input(
                config,
                cli_data_dir=opt.data_dir,
                cli_training_data=opt.training_data,
            )
        )
        resolved_output_dir, resolved_load_dir = resolve_output_dirs(
            config,
            cli_output_dir=opt.output_dir,
            cli_load_dir=opt.load_dir,
        )
    except ValueError as exc:
        parser.error(str(exc))

    opt.data_dir = resolved_data_dir
    opt.output_dir = resolved_output_dir
    opt.load_dir = resolved_load_dir
    opt.num_gpus = torch.cuda.device_count() if opt.num_gpus == -1 else opt.num_gpus
    ## Combine arguments and config
    cfg = edict()
    cfg.update(opt.__dict__)
    cfg.update(config)
    cfg.data_dir = resolved_data_dir
    cfg.output_dir = resolved_output_dir
    cfg.load_dir = resolved_load_dir
    if training_evidence is not None:
        cfg.training_evidence = training_evidence
    apply_smoke_overrides(cfg, opt.smoke_steps)
    print('\n\nConfig:')
    print('=' * 80)
    print(json.dumps(cfg.__dict__, indent=4))

    # Prepare output directory
    if cfg.node_rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        ## Save command and config
        with open(os.path.join(cfg.output_dir, 'command.txt'), 'w') as fp:
            print(' '.join(['python'] + sys.argv), file=fp)
        with open(os.path.join(cfg.output_dir, 'config.json'), 'w') as fp:
            json.dump(config, fp, indent=4)

    # Run
    if cfg.auto_retry == 0:
        cfg = find_ckpt(cfg)
        if cfg.num_gpus > 1:
            mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
        else:
            main(0, cfg)
    else:
        for rty in range(cfg.auto_retry):
            try:
                cfg = find_ckpt(cfg)
                if cfg.num_gpus > 1:
                    mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
                else:
                    main(0, cfg)
                break
            except Exception as e:
                import traceback
                print(f'\n{"="*60}')
                print(f'Error: {e}')
                print(f'{"="*60}')
                print('Full traceback:')
                traceback.print_exc()
                print(f'{"="*60}')
                print(f'Retrying ({rty + 1}/{cfg.auto_retry})...')
