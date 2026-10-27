"""Continue the disjoint GPU 6/7 TEST tail after a verified earlier segment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .use_idle_gpus_for_test_tail import ROOT, official_export_complete, state


def completed_segment(path: Path, last6: int, last7: int) -> dict | None:
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text())
    if receipt.get('status') != 'PASS_DISJOINT_TEST_TAIL_RELEASED_FOR_OFFICIAL_RENDER':
        raise ValueError('predecessor tail completion differs')
    for gpu, last in (('6', last6), ('7', last7)):
        indices = receipt['completed_indices'][gpu]
        if not indices or indices[-1] != last:
            raise ValueError(f'predecessor GPU{gpu} stopped before index {last}')
        if indices != list(range(indices[0], last - 1, -2)):
            raise ValueError(f'predecessor GPU{gpu} has a gap or duplicate')
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predecessor-root', type=Path, required=True)
    parser.add_argument('--predecessor-pid', type=int, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--cascade-pid', type=int, required=True)
    parser.add_argument('--official-render-pid', type=int, required=True)
    parser.add_argument('--predecessor-last6', type=int, required=True)
    parser.add_argument('--predecessor-last7', type=int, required=True)
    parser.add_argument('--min-index', type=int, required=True)
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    next6 = args.predecessor_last6 - 2
    next7 = args.predecessor_last7 - 2
    if (args.output_root.exists() or args.poll_seconds < 10
            or next6 % 2 == next7 % 2
            or not 300 <= args.min_index <= min(next6, next7)):
        raise ValueError('fresh, disjoint continuation segment required')
    previous = args.predecessor_root / 'complete.json'
    cascade_complete = ROOT / 'cascade-two-arm-execution-v2-direct-file4/complete.json'
    while True:
        if official_export_complete() or cascade_complete.is_file():
            print(json.dumps({'phase': 'SKIP_TAIL_EXTENSION_INPUT_READY'}), flush=True)
            return
        receipt = completed_segment(previous, args.predecessor_last6,
                                    args.predecessor_last7)
        if receipt is not None and state(args.predecessor_pid) in {'missing', 'Z', 'X'}:
            if receipt['official_export_ready']:
                print(json.dumps({'phase': 'SKIP_TAIL_EXTENSION_OFFICIAL_READY'}), flush=True)
                return
            break
        if state(args.predecessor_pid) in {'missing', 'Z', 'X'} and receipt is None:
            raise RuntimeError('predecessor exited without a completion receipt')
        print(json.dumps({'phase': 'WAIT_PREDECESSOR_TAIL',
                          'pid': args.predecessor_pid,
                          'receipt': receipt is not None}), flush=True)
        time.sleep(args.poll_seconds)
    command = [sys.executable, '-u', '-m',
               'experiments.deformnext_retargeting.use_idle_gpus_for_test_tail',
               '--cascade-pid', str(args.cascade_pid),
               '--official-render-pid', str(args.official_render_pid),
               '--output-root', str(args.output_root),
               '--min-index', str(args.min_index),
               '--first-index6', str(next6),
               '--first-index7', str(next7),
               '--poll-seconds', str(args.poll_seconds)]
    print(json.dumps({'phase': 'START_DISJOINT_TAIL_EXTENSION',
                      'gpu6_first': next6, 'gpu7_first': next7,
                      'min_index': args.min_index}), flush=True)
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
