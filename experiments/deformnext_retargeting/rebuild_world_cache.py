"""Build and audit versioned world-frame caches; never starts training."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--config-root', type=Path, required=True)
    parser.add_argument('--log-root', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--remote')
    parser.add_argument('--ssh-control')
    parser.add_argument('--ssh-port', type=int, default=22)
    args = parser.parse_args()
    base = args.data_root
    args.log_root.mkdir(parents=True, exist_ok=True)
    pending = base / 'manifests_world_v4_pending'
    combined = pending / 'all.jsonl'
    combined.write_text((pending / 'validation.jsonl').read_text() + (pending / 'train.jsonl').read_text())
    mappings = ['/file4=/root/dev/.cache/deformnext-file4/file4']
    common = ['--dump-root', str(base), '--path-map', mappings[0]]
    gpus = args.gpus.split(',')
    if len(set(gpus)) != len(gpus) or any(not gpu.isdigit() for gpu in gpus):
        raise ValueError('explicit unique GPU IDs required')
    env = dict(os.environ, PYTHONPATH='.', ATTN_BACKEND='sdpa', OMP_NUM_THREADS='4')
    processes = []
    for rank, gpu in enumerate(gpus):
        command = [sys.executable, '-u', '-m', 'experiments.deformnext_retargeting.build_pbr_latents',
                   '--manifest', str(combined), *common, '--rank', str(rank), '--world-size', str(len(gpus))]
        with (args.log_root / f'world-v4-all-rank{rank}.log').open('ab', buffering=0) as log:
            proc = subprocess.Popen(command, env=dict(env, CUDA_VISIBLE_DEVICES=gpu), stdout=log, stderr=subprocess.STDOUT)
        processes.append(proc)
    codes = [proc.wait() for proc in processes]
    if any(codes):
        raise RuntimeError(f'World-frame encoding failed: {codes}')
    env['CUDA_VISIBLE_DEVICES'] = gpus[0]
    def run(module, arguments):
        subprocess.run([sys.executable, '-u', '-m', module, *arguments], env=env, check=True)
    ready = base / 'manifests_world_v4'
    run('experiments.deformnext_retargeting.build_ss_anchor_latents', [
        '--manifest', str(pending / 'train.jsonl'), '--manifest', str(pending / 'validation.jsonl'),
        '--output-root', str(base / 'ss_latents_world_v4'), '--manifest-root', str(ready),
        '--published-root', '/file4/youngwoo/deformation/experiments/retargeting-v1/ss_latents_world_v4',
        '--path-map', mappings[0]])
    for split in ('train', 'validation'):
        run('experiments.deformnext_retargeting.projection_maps', [
            '--manifest', str(ready / f'{split}.jsonl'), '--output-root', str(base / 'projection_maps_world_v4'),
            '--path-map', mappings[0]])
    configs = base / 'world_v4_audit_configs'
    configs.mkdir(exist_ok=True)
    for stage in ('ss64', 'pbr1024'):
        cfg = json.loads((args.config_root / f'{stage}-v1.json').read_text())
        cfg['data_dir'] = str(ready / 'train.jsonl')
        cfg['validation']['roots_file'] = str(ready / 'validation.jsonl')
        (configs / f'{stage}-v1.json').write_text(json.dumps(cfg, indent=2))
    audit = base / 'audit_registration_world_v4'
    run('experiments.deformnext_retargeting.audit_registration', [
        '--ss-config', str(configs / 'ss64-v1.json'), '--pbr-config', str(configs / 'pbr1024-v1.json'),
        '--output', str(audit), '--samples', '32'])
    report = json.loads((audit / 'report.json').read_text())
    renders = report['renders']
    if len(renders) != 32 or min(row['shape_gt_mask_iou'] for row in renders) < .95:
        raise RuntimeError('Decoded Shape registration acceptance failed')
    if min(row['ss_vae_roundtrip_voxel_iou'] for row in renders) < .99:
        raise RuntimeError('SS VAE support reconstruction acceptance failed')
    if args.remote:
        ssh = ['ssh', '-p', str(args.ssh_port), '-o', 'ConnectTimeout=10']
        if args.ssh_control:
            ssh.extend(['-S', args.ssh_control])
        for name in ('shape1024_latents_world_v4', 'pbr_latents_world_v4', 'ss_latents_world_v4',
                     'projection_maps_world_v4', 'manifests_world_v4', 'audit_registration_world_v4'):
            subprocess.run(['rsync', '-a', '--partial', '--timeout=30', '-e', shlex.join(ssh),
                            str(base / name), args.remote.rstrip('/') + '/'], check=True)
    status = {'status': 'CACHE_AUDIT_PASS_TRAINING_PREFLIGHT_PENDING',
              'assets': report['assets_checked'], 'report': str(audit / 'report.json'),
              'remaining': 'Fresh six-arm GPU parity, full-batch smoke and step-zero baseline before training.'}
    (base / 'world_v4_cache_status.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
