"""Run the final paper claim audit after every matched result gate closes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


EXPECTED = (
    'PASS_COMMON_SUCCESS_TABLE_TEN_METRICS',
    'PASS_RAW_COMMON501_FULL_TABLE',
    'PASS_COMMON_SUCCESS_OURS_PAIRED_EIGHT_ENDPOINTS',
    'PASS_COMMON_SUCCESS_OURS_RAW_PAIRED_EIGHT_ENDPOINTS',
    'PASS_COMMON_TEST_OFFICIAL_CONTRASTS_ARCHIVED',
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-root', type=Path, required=True)
    parser.add_argument('--main-pid', type=int, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--raw-pid', type=int, required=True)
    parser.add_argument('--paired-root', type=Path, required=True)
    parser.add_argument('--paired-pid', type=int, required=True)
    parser.add_argument('--raw-paired-root', type=Path, required=True)
    parser.add_argument('--raw-paired-pid', type=int, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--official-pid', type=int, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    if args.output_root.exists() or args.poll_seconds < 10:
        raise ValueError('fresh audit output and bounded polling required')
    args.output_root.mkdir(parents=True)
    gates = (
        (args.main_root / 'complete.json', args.main_pid),
        (args.raw_root / 'complete.json', args.raw_pid),
        (args.paired_root / 'complete.json', args.paired_pid),
        (args.raw_paired_root / 'complete.json', args.raw_paired_pid),
        (args.official_root / 'complete.json', args.official_pid),
    )
    while not all(path.is_file() for path, _ in gates):
        states = []
        for path, pid in gates:
            done = path.is_file()
            states.append(done)
            if not done:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError as error:
                    raise RuntimeError('upstream watcher ended without receipt') from error
        print(json.dumps({'phase': 'WAIT_ALL_FAIR_TEST_GATES',
                          'complete': states}), flush=True)
        time.sleep(args.poll_seconds)
    receipts = [json.loads(path.read_text()) for path, _ in gates]
    if tuple(receipt['status'] for receipt in receipts) != EXPECTED:
        raise ValueError('upstream gate status differs from frozen contract')
    samples = {receipt['samples'] for receipt in receipts}
    if len(samples) != 1:
        raise ValueError('upstream gates use different common populations')
    output = args.output_root / 'claim-audit.json'
    subprocess.run([
        sys.executable, '-m',
        'experiments.deformnext_retargeting.audit_fair_test_claims',
        '--main-table-json', str(args.main_root / 'scale-ungiven-main-table.json'),
        '--raw-table-json', str(args.raw_root / 'raw-common501-table.json'),
        '--paired-json', str(args.paired_root / 'paired-eight-endpoints.json'),
        '--raw-paired-json', str(args.raw_paired_root / 'raw-paired-eight-endpoints.json'),
        '--official-contrasts-json',
        str(args.official_root / 'official-contrasts-sixteen-endpoints.json'),
        '--common-manifest', str(args.main_root / 'common-manifest.json'),
        '--output', str(output),
    ], check=True)
    result = json.loads(output.read_text())
    if (result['status'] != 'PASS_FAIR_TEST_CLAIM_AUDIT'
            or result['samples'] != samples.pop()
            or len(result['comparisons']['selected_vs_v1']) != 8
            or len(result['comparisons']['raw_selected_vs_v1']) != 8):
        raise ValueError('claim audit receipt is incomplete')
    complete = {'status': 'PASS_FAIR_TEST_CLAIM_AUDIT_ARCHIVED',
                'samples': result['samples'], 'output': str(output)}
    (args.output_root / 'complete.json').write_text(
        json.dumps(complete, indent=2) + '\n')
    print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    main()
