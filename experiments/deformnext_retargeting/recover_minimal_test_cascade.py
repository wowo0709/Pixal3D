"""Close the frozen TEST cascade after a render-bake GPU worker recovery."""
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
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--stopped-cascade-pid', type=int, required=True)
    parser.add_argument('--remote-base', required=True)
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    if (args.poll_seconds < 10 or state(args.stopped_cascade_pid) != 'T'
            or (args.output_root / 'complete.json').exists()):
        raise ValueError('stopped incomplete cascade and fresh completion required')
    campaign = json.loads(args.campaign.read_text())
    if (campaign['status'] != 'PREPARED_FROZEN_V1_SELECTED_TEST508_IMAGE_SCALE'
            or campaign['samples'] != 508 or len(campaign['plans']) != 1016):
        raise ValueError('frozen 1016-plan TEST campaign required')
    while True:
        shape = top1 = exports = 0
        for entry in campaign['plans']:
            plan = json.loads(Path(entry['shape_plan']).read_text())
            shape_receipt = Path(plan['output']) / 'archive-receipt.json'
            if shape_receipt.is_file():
                row = json.loads(shape_receipt.read_text())
                if (row['status'] != 'PASS_FILE4_ARCHIVE_CHECKSUM_VERIFIED'
                        or row['remote'] != f"{args.remote_base}/{entry['arm']}/{entry['sample_id']}/shape"):
                    raise ValueError('shape archive provenance differs: ' + entry['sample_id'])
                shape += 1
                if entry['arm'] == 'V1':
                    exports += 'benchmark.glb' in row['sha256']
            if entry['arm'] == 'SELECTED':
                pbr_plan = json.loads(Path(entry['top1_pbr_plan']).read_text())
                pbr_receipt = Path(pbr_plan['output']) / 'archive-receipt.json'
                if pbr_receipt.is_file():
                    row = json.loads(pbr_receipt.read_text())
                    if (row['status'] != 'PASS_FILE4_ARCHIVE_CHECKSUM_VERIFIED'
                            or row['remote'] != f"{args.remote_base}/SELECTED/{entry['sample_id']}/top1-pbr"):
                        raise ValueError('PBR archive provenance differs: ' + entry['sample_id'])
                    top1 += 1
                    exports += 'benchmark.glb' in row['sha256']
        print(json.dumps({'phase': 'WAIT_RECOVERED_TEST_CASCADE',
                          'shape': shape, 'top1': top1, 'exports': exports,
                          'stopped_original': state(args.stopped_cascade_pid)}), flush=True)
        if shape == 1016 and top1 == 508 and exports == 1016:
            break
        if state(args.stopped_cascade_pid) != 'T':
            raise RuntimeError('original cascade resumed before archive closure')
        time.sleep(args.poll_seconds)
    completion = {
        'status': 'PASS_FROZEN_V1_SELECTED_TEST508_CASCADES_FILE4_ARCHIVED',
        'samples': 508, 'shape_cases': shape,
        'selected_top1_pbr_cases': top1, 'render_exports': exports,
        'remote': args.remote_base, 'training_updates': 0,
        'physical_gpus': ['0', '1', '2', '3', '6', '7'],
        'recovery': 'GPU3 render-bake geometry exception; disjoint replacement worker',
    }
    tmp = args.output_root / 'complete.json.tmp'
    tmp.write_text(json.dumps(completion, indent=2) + '\n')
    tmp.replace(args.output_root / 'complete.json')
    print(json.dumps(completion), flush=True)
    os.kill(args.stopped_cascade_pid, signal.SIGCONT)


if __name__ == '__main__':
    main()
