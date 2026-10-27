"""Resume a frozen render shard if its initial export wait expires."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--initial-pid', type=int, required=True)
    parser.add_argument('--prep-root', type=Path, required=True)
    parser.add_argument('--benchmark-repo', type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.rank < 4:
        raise ValueError('only four frozen renderer shards')
    original = args.root / f'ours-render-rank{args.rank}.log'
    while alive(args.initial_pid):
        time.sleep(60)
    tail = original.read_text()[-4000:]
    if 'PASS_COMMON_OURS_RENDER_SHARD' in tail:
        status = 'PASS_INITIAL_RENDER_WORKER'
    elif 'TimeoutError: frozen export workers did not complete all render inputs' in tail:
        command = [sys.executable, str(args.root / 'run_common_benchmark_ours_render_worker.py'),
                   '--manifest', str(args.root / 'export-manifest-v2.jsonl'),
                   '--root', str(args.root), '--prep-root', str(args.prep_root),
                   '--benchmark-repo', str(args.benchmark_repo),
                   '--rank', str(args.rank), '--world-size', '4',
                   '--max-wait-seconds', '259200']
        with (args.root / f'ours-render-restart-rank{args.rank}.log').open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        status = 'PASS_RESUMED_RENDER_WORKER' if result.returncode == 0 else 'FAIL_RESUMED_RENDER_WORKER'
    else:
        status = 'FAIL_INITIAL_RENDER_WORKER'
    path = args.root / f'ours-render-supervisor-rank{args.rank}.json'
    path.write_text(json.dumps({'rank': args.rank, 'initial_pid': args.initial_pid,
                                'status': status}, indent=2) + '\n')
    if status.startswith('FAIL'):
        raise RuntimeError(status + ': inspect rank ' + str(args.rank))


if __name__ == '__main__':
    main()
