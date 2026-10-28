"""Compute raw V1/Selected paired intervals after full matched results close."""
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
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--raw-pid', type=int, required=True)
    p.add_argument('--paired-root', type=Path, required=True)
    p.add_argument('--paired-pid', type=int, required=True)
    p.add_argument('--main-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--poll-seconds', type=int, default=60)
    a = p.parse_args()
    if a.output_root.exists() or a.poll_seconds < 10:
        raise ValueError('fresh raw paired output and bounded polling required')
    a.output_root.mkdir(parents=True)
    gates = ((a.raw_root / 'complete.json', a.raw_pid),
             (a.paired_root / 'complete.json', a.paired_pid))
    while not all(path.is_file() for path, _ in gates):
        for path, pid in gates:
            if path.is_file(): continue
            try: os.kill(pid, 0)
            except ProcessLookupError: raise RuntimeError('required paired/table watcher ended without receipt')
        print(json.dumps({'phase': 'WAIT_RAW_TABLE_AND_OURS_PAIRED',
                          'raw_done': gates[0][0].is_file(), 'paired_done': gates[1][0].is_file()}), flush=True)
        time.sleep(a.poll_seconds)
    raw_gate = json.loads(gates[0][0].read_text())
    paired_gate = json.loads(gates[1][0].read_text())
    if (raw_gate['status'] != 'PASS_RAW_COMMON501_FULL_TABLE'
            or paired_gate['status'] != 'PASS_COMMON_SUCCESS_OURS_PAIRED_EIGHT_ENDPOINTS'
            or raw_gate['samples'] != paired_gate['samples']):
        raise ValueError('matched raw table and paired gates required')
    output = a.output_root / 'raw-paired-eight-endpoints.json'
    subprocess.run([sys.executable, '-m', 'experiments.deformnext_retargeting.score_scale_ungiven_ours_paired',
                    '--ours-evaluation', str(a.paired_root / 'evaluation-ours-four'),
                    '--ours-target8', str(a.main_root / 'ours-target8.json'),
                    '--common-manifest', str(a.main_root / 'common-manifest.json'),
                    '--mode', 'raw', '--output', str(output)], check=True)
    result = json.loads(output.read_text())
    table = json.loads((a.raw_root / 'raw-common501-table.json').read_text())
    if (result['status'] != 'PASS_GT_SCALE_FREE_OURS_V1_SELECTED_RAW_COMMON_PAIRED_EIGHT_ENDPOINTS'
            or result['samples'] != table['samples'] or len(result['metrics']) != 8):
        raise ValueError('raw paired result and table differ')
    for metric, row in result['metrics'].items():
        if (abs(row['v1_mean'] - table['values']['ours_v1_source_m0'][metric]) > 1e-6
                or abs(row['selected_mean'] - table['values']['ours_selected_source_m0'][metric]) > 1e-6
                or len(row['family16_bonferroni_ci95']) != 2):
            raise ValueError('raw paired means or 16-endpoint interval differ: ' + metric)
    receipt = {'status': 'PASS_COMMON_SUCCESS_OURS_RAW_PAIRED_EIGHT_ENDPOINTS',
               'samples': result['samples'], 'source_clusters': result['source_clusters'],
               'output': str(output)}
    (a.output_root / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
