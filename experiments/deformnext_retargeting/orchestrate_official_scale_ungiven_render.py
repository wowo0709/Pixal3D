"""Wait for complete official RGB1 artifacts and the baseline GPU campaign, then render common502."""
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
REMOTE_PLAN = REMOTE_WORK + '/official-rgb1-render-plan-v2'
REMOTE_SCORES = REMOTE_WORK + '/official-scale-ungiven-source-sim3-common-success-v2'
REMOTE_EXPORTS = REMOTE_WORK + '/official-scale-ungiven-export-common-success-v2'
REMOTE_OFFICIAL = REMOTE_WORK + '/no-scale-official1024'
REMOTE_ARCHIVE = ('/file4/youngwoo/deformation/experiments/retargeting-v1/'
                  'source-sim3-test508-20261028/official-rgb1-scale-ungiven-common-success-v2')


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, text=True, **kwargs)


def remote(host: str, port: int, command: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                             '-p', str(port), host, ' '.join(shlex.quote(part) for part in command)],
                            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f'remote command failed ({result.returncode}): '
                           f'{result.stdout[-2000:]} {result.stderr[-2000:]}')
    return result


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def complete_baseline_chunks(plan: Path) -> int:
    expected = [plan / f'chunk{i:03d}' for i in range(32)]
    if any(not (chunk / 'plan.json').is_file() for chunk in expected):
        raise ValueError('frozen 32-chunk baseline plan required')
    return sum((chunk / 'completed.json').is_file()
               and json.loads((chunk / 'completed.json').read_text()).get('status')
               == 'PASS_RENDER_CHUNK_ARCHIVED_ON_FILE4' for chunk in expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-host', default='youngwoo@n16.unist.info')
    parser.add_argument('--ssh-port', type=int, default=55555)
    parser.add_argument('--baseline-plan', type=Path, required=True)
    parser.add_argument('--baseline-render-pid', type=int, required=True)
    parser.add_argument('--local-prerequisite-complete', type=Path,
                        help='Finish planned local frozen SS generation before using its GPUs')
    parser.add_argument('--local-prerequisite-pid', type=int)
    parser.add_argument('--remote-score-watcher-pid', type=int, required=True)
    parser.add_argument('--remote-export-watcher-pid', type=int, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--benchmark-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--blender', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3,6,7')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if not args.blender.is_file() or not args.protocol.is_file():
        raise FileNotFoundError('declared benchmark Blender/protocol required')
    if args.output_root.exists() and not args.check_only:
        raise FileExistsError('fresh official campaign output required')
    if bool(args.local_prerequisite_complete) != bool(args.local_prerequisite_pid):
        raise ValueError('local prerequisite receipt and process PID must be supplied together')
    score_done = REMOTE_SCORES + '/complete.json'
    export_done = REMOTE_EXPORTS + '/complete.json'
    while True:
        baseline_count = complete_baseline_chunks(args.baseline_plan)
        query = remote(args.remote_host, args.ssh_port, [
            'python3', '-c',
            ('import json,os; from pathlib import Path; '
             f'paths={[score_done, export_done]!r}; '
             'print(json.dumps({"complete": [Path(p).is_file() for p in paths], '
             f'"watchers_live": [os.path.exists("/proc/{args.remote_score_watcher_pid}"), '
             f'os.path.exists("/proc/{args.remote_export_watcher_pid}")' + ']}))')])
        status = json.loads(query.stdout)
        prerequisite_done = (args.local_prerequisite_complete is None
                             or args.local_prerequisite_complete.is_file())
        print(json.dumps({'phase': 'WAIT_COMMON_RENDER_INPUTS', 'baseline_chunks': baseline_count,
                          'local_prerequisite_done': prerequisite_done, **status}), flush=True)
        if baseline_count == 32 and all(status['complete']) and prerequisite_done:
            break
        if args.check_only:
            return
        if baseline_count < 32 and not process_exists(args.baseline_render_pid):
            raise RuntimeError('baseline render process ended before all 32 archived chunks')
        for done, live in zip(status['complete'], status['watchers_live']):
            if not done and not live:
                raise RuntimeError('official score/export watcher stopped before completion')
        if (not prerequisite_done and args.local_prerequisite_pid is not None
                and not process_exists(args.local_prerequisite_pid)):
            raise RuntimeError('local frozen SS generation stopped before completion')
        time.sleep(60)
    if args.check_only:
        return
    if args.local_prerequisite_complete is not None:
        local = json.loads(args.local_prerequisite_complete.read_text())
        if local['status'] != 'PASS_FROZEN_TEST508_V1_SELECTED_SS_IMAGE_SCALE' or local['samples'] != 508:
            raise ValueError('local frozen SS prerequisite receipt differs')
    receipts = []
    for path, expected in ((score_done, 'PASS_OFFICIAL_RGB1_SCALE_UNGIVEN_COMMON_SUCCESS_SURFACE'),
                           (export_done, 'PASS_OFFICIAL_RGB1_SCALE_UNGIVEN_COMMON_SUCCESS_EXPORT')):
        row = json.loads(remote(args.remote_host, args.ssh_port, ['cat', path]).stdout)
        if row['status'] != expected or not 1 <= row['samples'] <= 502:
            raise ValueError('official common-success completion receipt differs: ' + path)
        receipts.append(row)
    expected_samples = receipts[0]['samples']
    if (receipts[1]['samples'] != expected_samples
            or receipts[0]['excluded_model_failures'] != receipts[1]['excluded_model_failures']):
        raise ValueError('official surface and texture success populations differ')
    builder = [
        '/home/seunghyun/editing_reproduction/setup/envs/metrics/bin/python',
        REMOTE_WORK + '/build_official_scale_ungiven_render_jobs.py',
        '--scores', *(f'{REMOTE_SCORES}/shard{i}.jsonl' for i in range(4)),
        '--exports', REMOTE_EXPORTS, '--official-root', REMOTE_OFFICIAL,
        '--target8-jobs', REMOTE_WORK + '/target8-baseline-jobs.jsonl',
        '--references', REMOTE_WORK + '/references.jsonl', '--output-root', REMOTE_PLAN,
        '--expected-samples', str(expected_samples)]
    result = remote(args.remote_host, args.ssh_port, builder)
    print(result.stdout.strip(), flush=True)
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    if (receipt['status'] != 'PASS_OFFICIAL_RGB1_RAW_AND_M0_RENDER_PLAN'
            or receipt['jobs'] != expected_samples * 6):
        raise ValueError('official full render plan failed')
    args.output_root.mkdir(parents=True)
    local_plan = args.output_root / 'render-plan'
    rsync_ssh = f'ssh -p {args.ssh_port} -o BatchMode=yes'
    remote_source = args.remote_host + ':' + REMOTE_PLAN + '/'
    run(['rsync', '-a', '-e', rsync_ssh, remote_source, str(local_plan) + '/'])
    verify = run(['rsync', '-a', '--checksum', '--dry-run', '--itemize-changes', '-e', rsync_ssh,
                  remote_source, str(local_plan) + '/'], capture_output=True)
    if verify.stdout.strip() or verify.stderr.strip():
        raise ValueError('official render plan copy differs from node16')
    campaign = [sys.executable, str(Path(__file__).parent / 'run_source_sim3_render_campaign.py'),
                '--plan-root', str(local_plan), '--remote-host', args.remote_host,
                '--remote-output-root', REMOTE_ARCHIVE, '--blender', str(args.blender),
                '--benchmark-root', str(args.benchmark_root), '--protocol', str(args.protocol),
                '--gpus', args.gpus, '--ssh-port', str(args.ssh_port),
                '--campaign-kind', 'official-rgb1', '--expected-samples', str(expected_samples)]
    run(campaign)
    result = {'status': 'PASS_OFFICIAL_RGB1_COMMON502_RENDER_ARCHIVED',
              'samples': expected_samples, 'jobs': expected_samples * 6,
              'chunks': (expected_samples + 15) // 16, 'remote': REMOTE_ARCHIVE}
    (args.output_root / 'complete.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
