"""Process disjoint TEST tail cases on GPUs 6/7 until official rendering needs them."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


ROOT = Path('/root/dev/.cache/deformnext-retargeting/scale-ungiven-test508-20261028')
REMOTE_EXPORT = ('/home/youngwoo/retargeting-unified-test508/'
                 'official-scale-ungiven-export-common-success-v2/complete.json')


def state(pid: int) -> str:
    path = Path(f'/proc/{pid}/stat')
    return path.read_text().split(') ', 1)[1][0] if path.is_file() else 'missing'


def official_export_complete() -> bool:
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                             '-p', '55555', 'youngwoo@n16.unist.info',
                             'test', '-f', REMOTE_EXPORT], timeout=12)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cascade-pid', type=int, required=True)
    parser.add_argument('--official-render-pid', type=int, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--min-index', type=int, default=400)
    parser.add_argument('--first-index6', type=int, default=507)
    parser.add_argument('--first-index7', type=int, default=506)
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    if (args.output_root.exists() or not 300 <= args.min_index < 506
            or not args.min_index <= args.first_index6 <= 507
            or not args.min_index <= args.first_index7 <= 507
            or args.first_index6 % 2 == args.first_index7 % 2
            or args.poll_seconds < 10
            or state(args.cascade_pid) in ('missing', 'Z')
            or state(args.official_render_pid) in ('missing', 'Z')
            or not Path('/tmp/file4-ctl-v2').is_socket()):
        raise ValueError('live cascade, official renderer, and direct file4 socket required')
    if official_export_complete():
        raise RuntimeError('official renderer already has its export input')
    args.output_root.mkdir(parents=True)
    stop = threading.Event()
    results = {'6': [], '7': []}
    released = False
    os.kill(args.official_render_pid, signal.SIGSTOP)
    print(json.dumps({'phase': 'RESERVE_GPUS_6_7_FOR_DISJOINT_TEST_TAIL',
                      'official_render_pid': args.official_render_pid,
                      'min_index': args.min_index,
                      'first_index6': args.first_index6,
                      'first_index7': args.first_index7}), flush=True)

    def worker(gpu: int, initial: int) -> None:
        env = dict(os.environ, PIXAL3D_FILE4_CONTROL_SOCKET='/tmp/file4-ctl-v2')
        for index in range(initial, args.min_index - 1, -2):
            # Recheck at the case boundary as well as in the supervisor loop:
            # a case can finish between its 30-second remote polls.
            if (stop.is_set() or official_export_complete()
                    or (ROOT / 'cascade-two-arm-execution-v2-direct-file4/complete.json').is_file()):
                break
            command = [sys.executable, '-u', '-m',
                       'experiments.deformnext_retargeting.run_minimal_test_cascade_worker',
                       '--campaign', str(ROOT / 'cascade-two-arm-plans-v1/campaign.json'),
                       '--ss-complete', str(ROOT / 'ss-two-arms-v1/complete.json'),
                       '--runtime-python', '/opt/conda/envs/pixal3d/bin/python',
                       '--remote-host', 'rvi@10.0.12.170',
                       '--remote-base', '/file4/youngwoo/deformation/experiments/'
                                        'retargeting-v1/scale-ungiven-test508-20261028/cascade-v1',
                       '--physical-gpu', str(gpu), '--allowed-gpus', '0,1,2,3,6,7',
                       '--rank', str(index), '--world-size', '508']
            log = args.output_root / f'gpu{gpu}-index{index:03d}.log'
            with log.open('w') as stream:
                completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                                           env=env, stdin=subprocess.DEVNULL,
                                           stdout=stream, stderr=subprocess.STDOUT)
            if completed.returncode:
                stop.set()
                raise RuntimeError(f'tail case {index} on GPU{gpu} failed: {log}')
            results[str(gpu)].append(index)
            print(json.dumps({'phase': 'PASS_DISJOINT_TEST_TAIL_CASE',
                              'gpu': gpu, 'index': index}), flush=True)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = (pool.submit(worker, 6, args.first_index6),
                       pool.submit(worker, 7, args.first_index7))
            while not stop.is_set():
                if official_export_complete():
                    stop.set()
                    print(json.dumps({'phase': 'OFFICIAL_EXPORT_READY_RELEASE_GPUS_6_7'}),
                          flush=True)
                    break
                if (ROOT / 'cascade-two-arm-execution-v2-direct-file4/complete.json').is_file():
                    stop.set()
                    break
                if any(future.done() for future in futures):
                    if any(future.done() and future.exception() for future in futures):
                        stop.set()
                        break
                    if all(future.done() for future in futures):
                        stop.set()
                        break
                if state(args.official_render_pid) != 'T':
                    stop.set()
                    raise RuntimeError('official renderer resumed before tail release')
                time.sleep(args.poll_seconds)
            for future in futures:
                future.result()
        receipt = {'status': 'PASS_DISJOINT_TEST_TAIL_RELEASED_FOR_OFFICIAL_RENDER',
                   'completed_indices': results,
                   'cases': sum(map(len, results.values())),
                   'official_export_ready': official_export_complete()}
        (args.output_root / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt), flush=True)
        released = True
    finally:
        # Keep the official watcher paused after a failed tail case if its
        # verified export has not been handed over yet.  Resuming it here used
        # to race a recovery attempt on the same GPUs.
        if state(args.official_render_pid) == 'T' and (released or official_export_complete()):
            os.kill(args.official_render_pid, signal.SIGCONT)
            print(json.dumps({'phase': 'RESUME_OFFICIAL_RENDER_WATCHER',
                              'pid': args.official_render_pid}), flush=True)


if __name__ == '__main__':
    main()
