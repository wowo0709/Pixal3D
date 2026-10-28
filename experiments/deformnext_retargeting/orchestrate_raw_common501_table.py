"""Build the raw full-metric table when the matched main table closes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--main-root', type=Path, required=True)
    p.add_argument('--main-pid', type=int, required=True)
    p.add_argument('--raw-external', type=Path, required=True)
    p.add_argument('--native-success', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--paper-tex', type=Path, required=True)
    p.add_argument('--poll-seconds', type=int, default=60)
    a = p.parse_args()
    if a.output_root.exists() or a.paper_tex.exists() or a.poll_seconds < 10:
        raise ValueError('fresh raw output and bounded poll required')
    a.output_root.mkdir(parents=True)
    receipt = a.main_root / 'complete.json'
    while not receipt.is_file():
        try: os.kill(a.main_pid, 0); live = True
        except ProcessLookupError: live = False
        print(json.dumps({'phase': 'WAIT_COMMON_MAIN_TABLE', 'main_watcher_live': live}), flush=True)
        if not live: raise RuntimeError('main table watcher ended without completion receipt')
        time.sleep(a.poll_seconds)
    if json.loads(receipt.read_text())['status'] != 'PASS_COMMON_SUCCESS_TABLE_TEN_METRICS':
        raise ValueError('main table gate differs')
    output = a.output_root / 'raw-common501-table.json'
    command = [sys.executable, '-m', 'experiments.deformnext_retargeting.compose_raw_common501_table',
               '--raw-external', str(a.raw_external), '--native-success', str(a.native_success),
               '--output-tex', str(a.paper_tex), '--output-json', str(output)]
    for key in ('nine-summary', 'ours-summary', 'common-manifest', 'external-target8',
                'official-target8', 'ours-target8'):
        command.extend(('--' + key, str(a.main_root / (key + '.json'))))
    subprocess.run(command, check=True)
    result = json.loads(output.read_text())
    (a.output_root / 'complete.json').write_text(json.dumps({
        'status': result['status'], 'samples': result['samples'],
        'json': str(output), 'tex': str(a.paper_tex)}, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'samples': result['samples']}), flush=True)


if __name__ == '__main__':
    main()
