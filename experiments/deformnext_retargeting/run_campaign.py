"""Run verified arms on explicitly assigned GPUs; fail closed on train errors."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ARMS = ('v1', 'b1', 'b2', 'b3', 'c1', 'c2')
REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('ss64', 'pbr1024'), required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--config-root', type=Path, required=True)
    parser.add_argument('--verification-root', type=Path, required=True)
    parser.add_argument('--log-root', type=Path, required=True)
    parser.add_argument('--master-port', type=int, required=True)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    parser.add_argument('--remote', help='user@host:/absolute/results/root')
    parser.add_argument('--ssh-control')
    parser.add_argument('--ssh-port', type=int, default=22)
    args = parser.parse_args()
    verification = json.loads((args.verification_root / args.stage / 'verification.json').read_text())
    if verification['status'] != 'PASS' or verification['native_denoiser_max_abs_error'] != 0:
        raise ValueError('native checkpoint GPU verification has not passed')
    if {row['arm'] for row in verification['arms']} != set(ARMS):
        raise ValueError('six-arm GPU verification is incomplete')
    args.log_root.mkdir(parents=True, exist_ok=True)
    gpus = args.gpus.split(',')
    if len(set(gpus)) != len(gpus) or any(not x.isdigit() for x in gpus):
        raise ValueError('GPU IDs must be explicit and unique')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, ATTN_BACKEND='sdpa',
               OMP_NUM_THREADS='8', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env.pop('SSHPASS', None)
    for arm in args.arms:
        path = args.config_root / f'{args.stage}-{arm}.json'
        cfg = json.loads(path.read_text())
        if cfg['experiment']['conditioning_schema'] != 'native-mv-anchor-frame-v2':
            raise ValueError('legacy conditioning config is not allowed')
        if cfg['experiment']['world_size'] != len(gpus):
            raise ValueError('assigned GPU count and global batch recipe differ')
        if arm == 'v1' and cfg['trainer']['args']['deform_conditioner'] is not None:
            raise ValueError('V1 must use the unmodified native MV conditioner')
        output = Path(cfg['default_output_dir'])
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'STOP').exists():
            raise ValueError(f'explicit STOP present: {output}')
        # Dense semantic/adapter arms can have a different peak from V1.
        # Select the largest tested microbatch instead of extrapolating V1 VRAM.
        if args.stage == 'ss64' and arm != 'v1':
            import copy
            selected = None
            for split in (1, 2, 4, 8):
                bench = copy.deepcopy(cfg)
                bench_root = args.log_root / 'memory-bench-native-v2' / f'{arm}-split{split}'
                bench_root.mkdir(parents=True, exist_ok=True)
                bench['default_output_dir'] = str(bench_root)
                bench['use_wandb'] = False
                bench['experiment']['validate_at_start'] = False
                bench['trainer']['args'].update(max_steps=1, i_save=1, i_log=1, batch_split=split)
                bench['validation']['every_steps'] = 2
                bench_path = bench_root / 'launch.json'
                bench_path.write_text(json.dumps(bench, indent=2) + '\n')
                bench_command = [sys.executable, str(REPO / 'train.py'), '--config', str(bench_path),
                                 '--num_gpus', str(len(gpus)), '--master_port', str(args.master_port),
                                 '--multiview_runtime', '--auto_retry', '0']
                bench_log = bench_root / 'launch.log'
                with bench_log.open('ab', buffering=0) as log:
                    result = subprocess.run(bench_command, env=env, cwd=REPO, stdout=log, stderr=log)
                if result.returncode:
                    text = bench_log.read_text(errors='replace')
                    if 'out of memory' not in text.lower():
                        raise RuntimeError(f'arm preflight failed for a non-memory reason: {bench_log}')
                    continue
                row = json.loads((bench_root / 'runtime_batch.jsonl').read_text().splitlines()[-1])
                # Reserve decoder/validation headroom; this is allocated peak,
                # not allocator reserved memory.
                if row['peak_allocated_gib'] > 91.0:
                    continue
                selected = split
                break
            if selected is None:
                raise RuntimeError(f'no verified microbatch for {arm}')
            cfg['trainer']['args']['batch_split'] = selected
            cfg['experiment']['microbatch_per_gpu'] = cfg['trainer']['args']['batch_size_per_gpu'] // selected
            path.write_text(json.dumps(cfg, indent=2) + '\n')
            print(f"Selected {arm} microbatch {cfg['experiment']['microbatch_per_gpu']}", flush=True)
        command = [sys.executable, str(REPO / 'train.py'), '--config', str(path),
                   '--num_gpus', str(len(gpus)), '--master_port', str(args.master_port),
                   '--multiview_runtime', '--auto_retry', '0']
        print(f'Starting {args.stage}/{arm}', flush=True)
        with (args.log_root / f'train-native-v2-{args.stage}-{arm}.log').open('ab', buffering=0) as log:
            subprocess.run(command, env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        progress = json.loads((output / 'progress.json').read_text())
        if progress['stage'] != 'completed' or progress['current'] != progress['total']:
            raise RuntimeError(f'arm exited without completing: {progress}')
        if args.remote:
            ssh = ['ssh', '-p', str(args.ssh_port), '-o', 'ConnectTimeout=5',
                   '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2']
            if args.ssh_control:
                ssh += ['-S', args.ssh_control]
            import shlex
            destination = f'{args.remote.rstrip("/")}/{args.stage}/{arm}/'
            host, directory = destination.split(':', 1)
            try:
                subprocess.run([*ssh, host, 'mkdir -p ' + shlex.quote(directory)], check=True, timeout=15)
                with (args.log_root / f'sync-native-v2-{args.stage}-{arm}.log').open('ab') as log:
                    subprocess.run(['rsync', '-a', '--partial', '--timeout=30', '-e', shlex.join(ssh),
                                    str(output) + '/', destination], stdout=log, stderr=log,
                                   check=True, timeout=900)
            except (subprocess.SubprocessError, OSError) as error:
                with (args.log_root / 'sync-native-v2-deferred.log').open('a') as log:
                    log.write(f'{args.stage}/{arm}: {type(error).__name__}\n')
                print(f'Deferred file-server sync for {args.stage}/{arm}; local checkpoints retained', flush=True)
    (args.log_root / f'campaign-native-v2-{args.stage}.done').touch()


if __name__ == '__main__':
    main()
