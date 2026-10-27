"""Score archived frozen V1/Selected TEST surfaces after cascade closure."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


REMOTE_WORK = '/home/youngwoo/retargeting-unified-test508'
REMOTE_ARCHIVE = ('/file4/youngwoo/deformation/experiments/retargeting-v1/'
                  'scale-ungiven-test508-20261028/cascade-v1')
REMOTE_MANIFEST = ('/file4/youngwoo/deformation/experiments/retargeting-v1/'
                   'rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/'
                   'anchor-ray-composition-v1/feature-readout-v1/test508-preparation-v2/'
                   'rgb_v1/manifests_world_v4_pending/test.jsonl')


def run_remote(host: str, port: int, parts: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-p', str(port), host,
                             ' '.join(shlex.quote(part) for part in parts)],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-1200:] + result.stdout[-1200:])
    return result


def live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cascade-complete', type=Path, required=True)
    parser.add_argument('--cascade-watcher-pid', type=int, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--remote-host', default='youngwoo@n16.unist.info')
    parser.add_argument('--ssh-port', type=int, default=55555)
    parser.add_argument('--remote-output-root', default=REMOTE_WORK + '/ours-v1-selected-source-sim3-final-v1')
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    if (args.output_root.exists() or args.poll_seconds < 10
            or args.remote_output_root != REMOTE_WORK + '/ours-v1-selected-source-sim3-final-v1'):
        raise ValueError('fresh bounded frozen Ours surface campaign required')
    args.output_root.mkdir(parents=True)
    while not args.cascade_complete.is_file():
        print(json.dumps({'phase': 'WAIT_FROZEN_OURS_TWO_CASCADES',
                          'cascade_watcher_live': live(args.cascade_watcher_pid)}), flush=True)
        if not live(args.cascade_watcher_pid):
            raise RuntimeError('frozen Ours cascade ended before file4 closure')
        time.sleep(args.poll_seconds)
    cascade = json.loads(args.cascade_complete.read_text())
    if (cascade['status'] != 'PASS_FROZEN_V1_SELECTED_TEST508_CASCADES_FILE4_ARCHIVED'
            or cascade['samples'] != 508 or cascade['shape_cases'] != 1016
            or cascade['selected_top1_pbr_cases'] != 508):
        raise ValueError('complete frozen two-arm file4 archive required')
    scripts = Path(__file__).parent
    remote_script = REMOTE_WORK + '/score_minimal_test_ours.py'
    for filename in ('score_minimal_test_ours.py', 'score_source_sim3_baselines.py'):
        subprocess.run(['rsync', '-a', '-e', f'ssh -o BatchMode=yes -p {args.ssh_port}',
                        str(scripts / filename),
                        f'{args.remote_host}:{REMOTE_WORK}/{filename}'], check=True)
    remote_exists = run_remote(args.remote_host, args.ssh_port,
                               ['test', '!', '-e', args.remote_output_root])
    run_remote(args.remote_host, args.ssh_port, ['mkdir', '-p', args.remote_output_root])
    metric_python = '/home/seunghyun/editing_reproduction/setup/envs/metrics/bin/python'
    processes = []
    for rank in range(4):
        output = f'{args.remote_output_root}/shard{rank}.jsonl'
        command = ['env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
                   metric_python, '-u', remote_script,
                   '--archive-root', REMOTE_ARCHIVE, '--manifest', REMOTE_MANIFEST,
                   '--common-subset', REMOTE_WORK + '/common-success-502.json',
                   '--benchmark-repo', '/home/seunghyun/editing_reproduction',
                   '--output', output, '--expected-samples', '502',
                   '--shard-count', '4', '--shard-index', str(rank)]
        log = (args.output_root / f'shard{rank}.log').open('w')
        process = subprocess.Popen(
            ['ssh', '-o', 'BatchMode=yes', '-p', str(args.ssh_port), args.remote_host,
             ' '.join(shlex.quote(part) for part in command)],
            stdout=log, stderr=subprocess.STDOUT)
        processes.append((rank, process, log))
    print(json.dumps({'phase': 'SCORE_FROZEN_OURS_TWO_SURFACES',
                      'remote': args.remote_output_root,
                      'local_ssh_pids': {rank: proc.pid for rank, proc, _ in processes}}), flush=True)
    results = [(rank, process.wait()) for rank, process, _ in processes]
    for _, _, log in processes:
        log.close()
    if any(code for _, code in results):
        raise RuntimeError('frozen Ours surface shard failed: ' + repr(results))
    all_rows = []
    for rank in range(4):
        content = run_remote(args.remote_host, args.ssh_port,
                             ['cat', f'{args.remote_output_root}/shard{rank}.jsonl']).stdout
        all_rows.extend(json.loads(line) for line in content.splitlines() if line)
    keys = {(row['sample_id'], row['method']) for row in all_rows}
    if (len(all_rows) != 1004 or len(keys) != 1004
            or {row['method'] for row in all_rows}
            != {'ours_v1_image_scale_1024', 'ours_selected_image_scale_1024'}
            or len({row['sample_id'] for row in all_rows}) != 502):
        raise ValueError('frozen common502 Ours raw/+M0 surface closure differs')
    result = {'status': 'PASS_FROZEN_OURS_V1_SELECTED_COMMON502_SURFACE',
              'samples': 502, 'rows': 1004, 'remote': args.remote_output_root,
              'training_updates': 0}
    (args.output_root / 'complete.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
