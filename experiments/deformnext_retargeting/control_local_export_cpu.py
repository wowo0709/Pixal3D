"""Keep local export throughput while respecting a busy shared CPU host."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time


DEFAULT_GROUPS = (360663, 360664, 360665, 360666, 360667, 360668)


def state(pid: int) -> str | None:
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('State:'):
                value = line.split()[1]
                return None if value == 'Z' else value
    except FileNotFoundError:
        return None
    raise ValueError('process status lacks State')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', type=int, nargs='+', default=DEFAULT_GROUPS)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--interval-seconds', type=int, default=60)
    args = parser.parse_args()
    if len(set(args.groups)) != 6 or args.interval_seconds < 30:
        raise ValueError('expected six distinct worker process groups')
    while True:
        states = {pid: state(pid) for pid in args.groups}
        live = {pid: status for pid, status in states.items() if status is not None}
        if not live:
            args.status.write_text(json.dumps({'status': 'ALL_LOCAL_EXPORT_WORKERS_EXITED'}, indent=2) + '\n')
            break
        active = sorted(pid for pid, status in live.items() if status != 'T')
        paused = sorted(pid for pid, status in live.items() if status == 'T')
        load1 = os.getloadavg()[0]
        desired = 2
        if load1 < 85:
            desired = 3
        if load1 < 65:
            desired = 4
        if load1 < 45:
            desired = 6
        desired = min(desired, len(live))
        actions = []
        while len(active) > desired:
            pid = active.pop()
            try:
                os.killpg(pid, signal.SIGSTOP)
                paused.append(pid)
                actions.append(['pause', pid])
            except ProcessLookupError:
                actions.append(['exited', pid])
        while len(active) < desired and paused:
            pid = paused.pop(0)
            try:
                os.killpg(pid, signal.SIGCONT)
                active.append(pid)
                actions.append(['resume', pid])
            except ProcessLookupError:
                actions.append(['exited', pid])
        report = {'status': 'ADAPTIVE_LOCAL_EXPORT_CPU_CONTROL',
                  'load1': load1, 'active': sorted(active),
                  'paused': sorted(paused), 'desired_active': desired,
                  'actions': actions, 'timestamp_unix': time.time()}
        args.status.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
        time.sleep(args.interval_seconds)


if __name__ == '__main__':
    main()
