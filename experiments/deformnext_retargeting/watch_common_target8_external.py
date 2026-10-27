"""Publish the common-502 external target8 scores when all saved renders close."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--benchmark-repo', type=Path, required=True)
    parser.add_argument('--prep-root', type=Path, required=True)
    parser.add_argument('--max-wait-hours', type=float, default=24.)
    args = parser.parse_args()
    output = args.root / 'target8-external-502.json'
    if output.exists():
        raise FileExistsError(output)
    start = time.monotonic()
    while True:
        count = sum(1 for _ in (args.root / 'target8').rglob('*.receipt.json'))
        print(json.dumps({'external_target8_receipts': count, 'expected': 4054}), flush=True)
        if count > 4054:
            raise ValueError('more target8 renders than the frozen contract')
        if count == 4054:
            break
        if (time.monotonic() - start) / 3600 > args.max_wait_hours:
            raise TimeoutError('external target8 renders did not close')
        time.sleep(120)
    command = [str(args.benchmark_repo / 'setup/envs/benchmark-cpu/bin/python'),
               str(args.root / 'scripts/score_common_target8.py'),
               '--benchmark-repo', str(args.benchmark_repo),
               '--prep-root', str(args.prep_root),
               '--jobs', str(args.root / 'target8-all-jobs.jsonl'),
               '--ours-render-records', str(args.root / 'render-records'),
               '--subset', str(args.root / 'common-success-502.json'),
               '--external-only', '--output', str(output)]
    subprocess.run(command, check=True)
    print(json.dumps({'status': 'PASS_EXTERNAL_TARGET8_502', 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
