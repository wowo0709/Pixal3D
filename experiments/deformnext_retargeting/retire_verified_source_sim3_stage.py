"""Reclaim only a staged render chunk after file4 checksum verification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunk', type=Path, required=True)
    parser.add_argument('--remote-renders', required=True,
                        help='rsync destination, e.g. user@host:/file4/.../chunk000/renders/')
    parser.add_argument('--ssh-port', type=int, default=55555)
    parser.add_argument('--ssh-control-socket', type=Path,
                        help='Direct file4 SSH master socket, when used by the renderer')
    args = parser.parse_args()
    chunk = args.chunk.resolve()
    stage = chunk / 'stage'
    if not chunk.name.startswith('chunk') or not stage.is_dir() or not (chunk / 'jobs.jsonl').is_file():
        raise ValueError('an explicitly planned owned chunk stage is required')
    if ': /' in args.remote_renders or ':/file4/youngwoo/deformation/experiments/retargeting-v1/source-sim3-test508-20261028/' not in args.remote_renders:
        raise ValueError('remote must be the declared source-Sim3 file4 publication root')
    if args.ssh_control_socket and (not args.ssh_control_socket.is_socket()
            or not args.remote_renders.startswith('rvi@10.0.12.170:')):
        raise ValueError('direct file4 control socket requires the direct file4 host')
    jobs = [json.loads(line) for line in (chunk / 'jobs.jsonl').read_text().splitlines()]
    for job in jobs:
        png = stage / job['output_rgba']
        if not png.is_file() or not png.with_suffix('.receipt.json').is_file():
            raise ValueError('incomplete local output: ' + str(png))
    rsync_ssh = ('ssh ' + ('-S ' + str(args.ssh_control_socket) + ' '
                            if args.ssh_control_socket else '')
                 + f'-p {args.ssh_port} -o BatchMode=yes')
    command = ['rsync', '-a', '--checksum', '--dry-run', '--itemize-changes',
               '-e', rsync_ssh,
               str(stage / 'renders') + '/', args.remote_renders]
    check = subprocess.run(command, text=True, capture_output=True, check=True)
    if check.stdout.strip() or check.stderr.strip():
        raise ValueError('remote checksum copy differs; preserve stage:\n' + check.stdout + check.stderr)
    count = len(jobs)
    shutil.rmtree(stage)
    if stage.exists():
        raise RuntimeError('verified stage not reclaimed')
    print(json.dumps({'status': 'PASS_VERIFIED_FILE4_AND_LOCAL_STAGE_RECLAIMED',
                      'chunk': str(chunk), 'jobs': count, 'remote': args.remote_renders}))


if __name__ == '__main__':
    main()
