"""Stage common benchmark assets in bounded chunks, render, archive on file4."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-root', type=Path, required=True)
    parser.add_argument('--remote-host', required=True)
    parser.add_argument('--remote-output-root', required=True)
    parser.add_argument('--blender', type=Path, required=True)
    parser.add_argument('--benchmark-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--ssh-port', type=int, default=55555)
    parser.add_argument('--campaign-kind', choices=('external-seven', 'official-rgb1', 'ours-two'),
                        default='external-seven')
    parser.add_argument('--expected-samples', type=int, default=502)
    args = parser.parse_args()
    if (not args.remote_output_root.startswith('/file4/youngwoo/deformation/experiments/retargeting-v1/source-sim3-test508-20261028/')
            or not args.blender.is_file() or not args.protocol.is_file()):
        raise ValueError('declared file4 archive, Blender and protocol required')
    chunks = sorted(args.plan_root.glob('chunk[0-9][0-9][0-9]'))
    if (args.campaign_kind == 'external-seven' and args.expected_samples != 502
            or args.campaign_kind in {'official-rgb1', 'ours-two'}
            and not 1 <= args.expected_samples <= 502):
        raise ValueError('declared frozen population size differs')
    arms = {'external-seven': 7, 'official-rgb1': 2, 'ours-two': 4}[args.campaign_kind]
    expected_jobs = args.expected_samples * arms * 3
    plans = [json.loads((chunk / 'plan.json').read_text()) for chunk in chunks]
    if not chunks or sum(plan['jobs'] for plan in plans) != expected_jobs:
        raise ValueError('complete common502 render plan required')
    if args.campaign_kind in {'official-rgb1', 'ours-two'}:
        expected_status = ('PREPARED_OFFICIAL_RGB1_SCALE_UNGIVEN_RAW_AND_M0_RENDER'
                           if args.campaign_kind == 'official-rgb1' else
                           'PREPARED_OURS_V1_SELECTED_GT_FREE_RAW_AND_M0_RENDER')
        jobs_per_sample = 6 if args.campaign_kind == 'official-rgb1' else 12
        if (sum(len(plan['sample_ids']) for plan in plans) != args.expected_samples
                or any(plan.get('status') != expected_status
                       or plan['jobs'] != len(plan['sample_ids']) * jobs_per_sample
                       or plan['conditions'] != ['raw', 'source_m0']
                       or plan['cameras'] != [8, 9, 10] for plan in plans)):
            raise ValueError('paired raw/+M0 × camera8/9/10 plan required')
        ids = [sid for plan in plans for sid in plan['sample_ids']]
        if len(set(ids)) != args.expected_samples:
            raise ValueError('common-success population must be unique')
    # Ours' input and output artifacts both live on file4.  Bypassing the
    # node16 /file4 mount avoids relaying tens of GB of GLBs through node16.
    # Existing official/external campaigns keep their declared transport.
    socket = Path('/tmp/file4-ctl-v2')
    direct_file4 = args.campaign_kind == 'ours-two' and socket.is_socket()
    if direct_file4 and any(not path.startswith('file4/')
                            for chunk in chunks
                            for path in (chunk / 'input-files.txt').read_text().splitlines()):
        raise ValueError('direct file4 transport requires file4-only render inputs')
    host = 'rvi@10.0.12.170' if direct_file4 else args.remote_host
    ssh = ['ssh']
    if direct_file4:
        ssh += ['-S', str(socket)]
    ssh += ['-o', 'BatchMode=yes', '-p', str(args.ssh_port), host]
    remote_source = host + ':/'
    remote_prefix = host + ':'
    rsync_ssh = ('ssh ' + ('-S ' + str(socket) + ' ' if direct_file4 else '')
                 + '-p ' + str(args.ssh_port) + ' -o BatchMode=yes')
    print(json.dumps({'phase': 'RENDER_FILE4_TRANSPORT',
                      'campaign_kind': args.campaign_kind,
                      'host': host, 'direct_control_socket': direct_file4}), flush=True)
    scripts = Path(__file__).parent
    for position, chunk in enumerate(chunks, 1):
        complete = chunk / 'completed.json'
        if complete.is_file():
            record = json.loads(complete.read_text())
            if record['status'] != 'PASS_RENDER_CHUNK_ARCHIVED_ON_FILE4' or record['chunk'] != chunk.name:
                raise ValueError('invalid completed chunk receipt')
            print(json.dumps({'chunk': chunk.name, 'status': 'REUSED', 'position': position}), flush=True)
            continue
        stage = chunk / 'stage'
        stage.mkdir(exist_ok=True)
        paths = (chunk / 'input-files.txt').read_text().splitlines()
        if not paths or any(Path(path).is_absolute() or '..' in Path(path).parts for path in paths):
            raise ValueError('staged inputs must be portable paths below remote root')
        started = time.monotonic()
        print(json.dumps({'chunk': chunk.name, 'phase': 'STAGE', 'position': position,
                          'chunks': len(chunks), 'inputs': len(paths)}), flush=True)
        run(['rsync', '-a', '--partial', '--files-from=' + str(chunk / 'input-files.txt'),
             '-e', rsync_ssh, remote_source, str(stage) + '/'])
        run([sys.executable, str(scripts / 'render_source_sim3_chunk.py'),
             '--chunk', str(chunk), '--stage-root', str(stage),
             '--blender', str(args.blender),
             '--batch-script', str(scripts / 'render_source_sim3_blender_batch.py'),
             '--benchmark-root', str(args.benchmark_root), '--protocol', str(args.protocol),
             '--gpus', args.gpus, '--threads-per-gpu', '4'])
        remote_chunk = args.remote_output_root.rstrip('/') + '/' + chunk.name
        remote_renders = remote_chunk + '/renders/'
        run([*ssh, 'mkdir', '-p', remote_renders])
        run(['rsync', '-a', '-e', rsync_ssh, str(stage / 'renders') + '/',
             remote_prefix + remote_renders])
        retire = [sys.executable, str(scripts / 'retire_verified_source_sim3_stage.py'),
                  '--chunk', str(chunk), '--remote-renders', remote_prefix + remote_renders,
                  '--ssh-port', str(args.ssh_port)]
        if direct_file4:
            retire += ['--ssh-control-socket', str(socket)]
        run(retire)
        record = {'status': 'PASS_RENDER_CHUNK_ARCHIVED_ON_FILE4', 'chunk': chunk.name,
                  'samples': len(json.loads((chunk / 'plan.json').read_text())['sample_ids']),
                  'jobs': json.loads((chunk / 'plan.json').read_text())['jobs'],
                  'remote': remote_chunk, 'seconds': round(time.monotonic() - started, 2)}
        complete.write_text(json.dumps(record, indent=2) + '\n')
        run(['rsync', '-a', '-e', rsync_ssh, str(chunk / 'jobs.jsonl'), str(chunk / 'plan.json'),
             str(chunk / 'render-summary.json'), str(complete), remote_prefix + remote_chunk + '/'])
        print(json.dumps(record), flush=True)
    status = {'external-seven': 'PASS_SOURCE_SIM3_RENDER_CAMPAIGN',
              'official-rgb1': 'PASS_OFFICIAL_RGB1_RAW_AND_M0_RENDER_CAMPAIGN',
              'ours-two': 'PASS_OURS_V1_SELECTED_RAW_AND_M0_RENDER_CAMPAIGN'}[args.campaign_kind]
    print(json.dumps({'status': status,
                      'chunks': len(chunks), 'jobs': expected_jobs}), flush=True)


if __name__ == '__main__':
    main()
