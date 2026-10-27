"""Render frozen V1/Selected common502 raw/+M0 after both GPU campaigns release."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


REMOTE_WORK = '/home/youngwoo/retargeting-unified-test508'
REMOTE_SCORES = REMOTE_WORK + '/ours-v1-selected-source-sim3-final-v1'
REMOTE_PLAN = REMOTE_WORK + '/ours-v1-selected-render-plan-v1'
REMOTE_ARCHIVE = ('/file4/youngwoo/deformation/experiments/retargeting-v1/'
                  'scale-ungiven-test508-20261028/cascade-v1')
REMOTE_RENDERS = ('/file4/youngwoo/deformation/experiments/retargeting-v1/'
                  'source-sim3-test508-20261028/ours-v1-selected-scale-ungiven-common502-v1')


def live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def remote(host: str, port: int, parts: list[str]) -> str:
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-p', str(port), host,
                             ' '.join(shlex.quote(part) for part in parts)],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-1200:] + result.stdout[-1200:])
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--surface-complete', type=Path, required=True)
    parser.add_argument('--surface-watcher-pid', type=int, required=True)
    parser.add_argument('--official-render-complete', type=Path, required=True)
    parser.add_argument('--official-render-watcher-pid', type=int, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--benchmark-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--blender', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3,6,7')
    parser.add_argument('--remote-host', default='youngwoo@n16.unist.info')
    parser.add_argument('--ssh-port', type=int, default=55555)
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    if (args.output_root.exists() or args.poll_seconds < 10
            or not args.blender.is_file() or not args.protocol.is_file()):
        raise ValueError('fresh output, Blender and frozen common protocol required')
    args.output_root.mkdir(parents=True)
    while True:
        ready = [args.surface_complete.is_file(), args.official_render_complete.is_file()]
        print(json.dumps({'phase': 'WAIT_OURS_SURFACE_AND_OFFICIAL_RENDER_GPU_RELEASE',
                          'complete': ready,
                          'watchers_live': [live(args.surface_watcher_pid),
                                            live(args.official_render_watcher_pid)]}), flush=True)
        if all(ready):
            break
        for done, pid in zip(ready, (args.surface_watcher_pid, args.official_render_watcher_pid)):
            if not done and not live(pid):
                raise RuntimeError('required frozen surface/render watcher ended before completion')
        time.sleep(args.poll_seconds)
    scores = json.loads(args.surface_complete.read_text())
    official = json.loads(args.official_render_complete.read_text())
    if (scores['status'] != 'PASS_FROZEN_OURS_V1_SELECTED_COMMON502_SURFACE'
            or scores['samples'] != 502 or scores['rows'] != 1004
            or official['status'] != 'PASS_OFFICIAL_RGB1_COMMON502_RENDER_ARCHIVED'
            or not 1 <= official['samples'] <= 502):
        raise ValueError('surface and official render closures differ')
    script = Path(__file__).parent / 'build_minimal_test_ours_render_jobs.py'
    subprocess.run(['rsync', '-a', '-e', f'ssh -o BatchMode=yes -p {args.ssh_port}',
                    str(script), f'{args.remote_host}:{REMOTE_WORK}/{script.name}'], check=True)
    metric_python = '/home/seunghyun/editing_reproduction/setup/envs/metrics/bin/python'
    command = [metric_python, REMOTE_WORK + '/' + script.name,
               '--scores', *(f'{REMOTE_SCORES}/shard{i}.jsonl' for i in range(4)),
               '--common-subset', REMOTE_WORK + '/common-success-502.json',
               '--archive-root', REMOTE_ARCHIVE,
               '--target8-jobs', REMOTE_WORK + '/target8-baseline-jobs.jsonl',
               '--references', REMOTE_WORK + '/references.jsonl',
               '--output-root', REMOTE_PLAN, '--expected-samples', '502']
    output = remote(args.remote_host, args.ssh_port, command)
    print(output, flush=True)
    built = json.loads(output.splitlines()[-1])
    if (built['status'] != 'PASS_OURS_V1_SELECTED_RAW_AND_M0_RENDER_PLAN'
            or built['samples'] != 502 or built['jobs'] != 6024):
        raise ValueError('complete two-arm raw/+M0 render plan required')
    local_plan = args.output_root / 'render-plan'
    transport = f'ssh -o BatchMode=yes -p {args.ssh_port}'
    source = f'{args.remote_host}:{REMOTE_PLAN}/'
    subprocess.run(['rsync', '-a', '-e', transport, source, str(local_plan) + '/'], check=True)
    check = subprocess.run(['rsync', '-a', '--checksum', '--dry-run', '--itemize-changes',
                            '-e', transport, source, str(local_plan) + '/'],
                           capture_output=True, text=True, check=True)
    if check.stdout.strip() or check.stderr.strip():
        raise ValueError('node16/local Ours render plan differs')
    campaign = [sys.executable,
                str(Path(__file__).parent / 'run_source_sim3_render_campaign.py'),
                '--plan-root', str(local_plan), '--remote-host', args.remote_host,
                '--remote-output-root', REMOTE_RENDERS,
                '--blender', str(args.blender), '--benchmark-root', str(args.benchmark_root),
                '--protocol', str(args.protocol), '--gpus', args.gpus,
                '--ssh-port', str(args.ssh_port), '--campaign-kind', 'ours-two',
                '--expected-samples', '502']
    subprocess.run(campaign, check=True)
    result = {'status': 'PASS_OURS_V1_SELECTED_RAW_AND_M0_COMMON502_RENDER_ARCHIVED',
              'samples': 502, 'jobs': 6024, 'remote': REMOTE_RENDERS,
              'training_updates': 0, 'gpus': args.gpus.split(',')}
    (args.output_root / 'complete.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
