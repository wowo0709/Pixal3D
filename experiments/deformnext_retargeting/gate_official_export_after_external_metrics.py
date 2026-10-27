"""Serialize node16 GPU7 external learned metrics before official GLB exports."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time


def live(pid: int) -> bool:
    stat = Path(f'/proc/{pid}/stat')
    return stat.is_file() and stat.read_text().split(') ', 1)[1][0] != 'Z'


def terminate(_signum, _frame) -> None:
    raise SystemExit(143)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external-completion', type=Path, required=True)
    parser.add_argument('--external-pid', type=int, required=True)
    parser.add_argument('--official-export-pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    if (args.output.exists() or args.poll_seconds < 10
            or not live(args.external_pid) or not live(args.official_export_pid)):
        raise ValueError('fresh gate and two live node16 GPU7 jobs required')
    signal.signal(signal.SIGTERM, terminate)
    os.kill(args.official_export_pid, signal.SIGSTOP)
    print(json.dumps({'phase': 'PAUSE_OFFICIAL_EXPORT_UNTIL_EXTERNAL_METRICS',
                      'official_export_pid': args.official_export_pid,
                      'external_pid': args.external_pid}), flush=True)
    try:
        while not args.external_completion.is_file():
            print(json.dumps({'phase': 'WAIT_EXTERNAL_LEARNED_METRICS',
                              'external_live': live(args.external_pid),
                              'official_export_paused': live(args.official_export_pid)}),
                  flush=True)
            if not live(args.external_pid):
                raise RuntimeError('external evaluator ended without completion')
            if not live(args.official_export_pid):
                raise RuntimeError('official exporter vanished while gated')
            time.sleep(args.poll_seconds)
        external = json.loads(args.external_completion.read_text())
        if external['status'] != 'PASS_SOURCE_M0_FULL502_SEVEN_METHODS_TEN_METRICS':
            raise ValueError('external metric completion receipt differs')
        result = {'status': 'PASS_GPU7_EXTERNAL_METRICS_THEN_OFFICIAL_EXPORT',
                  'external_completion': str(args.external_completion),
                  'external_pid': args.external_pid,
                  'official_export_pid': args.official_export_pid}
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    finally:
        if live(args.official_export_pid):
            os.kill(args.official_export_pid, signal.SIGCONT)
            print(json.dumps({'phase': 'RESUME_OFFICIAL_EXPORT',
                              'official_export_pid': args.official_export_pid}), flush=True)


if __name__ == '__main__':
    main()
