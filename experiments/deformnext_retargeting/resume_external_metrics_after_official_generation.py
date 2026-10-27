"""Keep node16 GPU7 generation and learned metrics from overlapping."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time


def state(pid: int) -> str:
    path = Path(f'/proc/{pid}/stat')
    return path.read_text().split(') ', 1)[1][0] if path.is_file() else 'missing'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation-pid', type=int, required=True)
    parser.add_argument('--attempts-complete', type=Path, required=True)
    parser.add_argument('--external-pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    if (args.output.exists() or args.poll_seconds < 10
            or state(args.external_pid) != 'T'
            or state(args.generation_pid) in ('missing', 'Z')):
        raise ValueError('stopped external evaluator and live official generator required')
    try:
        while not args.attempts_complete.is_file():
            print(json.dumps({'phase': 'WAIT_OFFICIAL_GENERATION_BEFORE_GPU7_LEARNED_METRICS',
                              'generator': state(args.generation_pid),
                              'external': state(args.external_pid)}), flush=True)
            if state(args.generation_pid) in ('missing', 'Z'):
                raise RuntimeError('official generation ended without attempts closure')
            if state(args.external_pid) != 'T':
                raise RuntimeError('external evaluator resumed before GPU7 release')
            time.sleep(args.poll_seconds)
        attempts = json.loads(args.attempts_complete.read_text())
        if (attempts['status'] != 'PASS_OFFICIAL_RGB1_TEST508_ATTEMPTS_CLOSED'
                or attempts['attempted'] != 508):
            raise ValueError('official TEST508 attempt receipt differs')
        result = {'status': 'PASS_GPU7_GENERATION_THEN_EXTERNAL_LEARNED_METRICS',
                  'official_attempts': str(args.attempts_complete),
                  'external_pid': args.external_pid,
                  'attempted': attempts['attempted']}
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    finally:
        if state(args.external_pid) == 'T':
            os.kill(args.external_pid, signal.SIGCONT)
            print(json.dumps({'phase': 'RESUME_EXTERNAL_EVALUATOR',
                              'external_pid': args.external_pid}), flush=True)


if __name__ == '__main__':
    main()
