"""Launch only frozen V1/Selected TEST508 cascades after their SS artifacts close."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--ss-complete', type=Path, required=True)
    parser.add_argument('--ss-watcher-pid', type=int, required=True)
    parser.add_argument('--runtime-python', type=Path, required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--remote-host', required=True)
    parser.add_argument('--remote-base', required=True)
    parser.add_argument('--ssh-port', type=int, default=55555)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    gpus = args.gpus.split(',')
    if (args.output_root.exists() or len(gpus) < 1 or len(set(gpus)) != len(gpus)
            or any(not gpu.isdigit() for gpu in gpus) or args.poll_seconds < 10
            or not args.campaign.is_file() or not args.runtime_python.is_file()):
        raise ValueError('fresh output and explicit physical GPU assignment required')
    args.output_root.mkdir(parents=True)
    while not args.ss_complete.is_file():
        print(json.dumps({'phase': 'WAIT_FROZEN_TEST508_SS',
                          'ss_watcher_live': process_exists(args.ss_watcher_pid)}), flush=True)
        if not process_exists(args.ss_watcher_pid):
            raise RuntimeError('frozen TEST508 SS watcher ended before completion')
        time.sleep(args.poll_seconds)
    ss = json.loads(args.ss_complete.read_text())
    campaign = json.loads(args.campaign.read_text())
    if (ss['status'] != 'PASS_FROZEN_TEST508_V1_SELECTED_SS_IMAGE_SCALE'
            or ss['samples'] != 508
            or campaign['status'] != 'PREPARED_FROZEN_V1_SELECTED_TEST508_IMAGE_SCALE'
            or campaign['samples'] != 508 or len(campaign['plans']) != 1016):
        raise ValueError('frozen SS and 1016 case plans must match')
    # Exercise one complete paired case before reserving all four GPUs. The
    # regular workers below skip its checksum-verified archive receipts.
    pilot = [sys.executable, '-u', '-m',
             'experiments.deformnext_retargeting.run_minimal_test_cascade_worker',
             '--campaign', str(args.campaign), '--ss-complete', str(args.ss_complete),
             '--runtime-python', str(args.runtime_python),
             '--remote-host', args.remote_host, '--remote-base', args.remote_base,
             '--ssh-port', str(args.ssh_port), '--physical-gpu', gpus[0],
             '--allowed-gpus', args.gpus, '--rank', '0', '--world-size', '508']
    with (args.output_root / 'pilot.log').open('w') as log:
        subprocess.run(pilot, cwd=Path(__file__).resolve().parents[2],
                       stdin=subprocess.DEVNULL, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    print(json.dumps({'phase': 'PASS_ONE_PAIRED_TEST_CASE_PILOT',
                      'physical_gpu': gpus[0]}), flush=True)
    processes = []
    for rank, gpu in enumerate(gpus):
        log = (args.output_root / f'gpu{gpu}.log').open('w')
        command = [sys.executable, '-u', '-m',
                   'experiments.deformnext_retargeting.run_minimal_test_cascade_worker',
                   '--campaign', str(args.campaign), '--ss-complete', str(args.ss_complete),
                   '--runtime-python', str(args.runtime_python),
                   '--remote-host', args.remote_host, '--remote-base', args.remote_base,
                   '--ssh-port', str(args.ssh_port), '--physical-gpu', gpu,
                   '--allowed-gpus', args.gpus, '--rank', str(rank),
                   '--world-size', str(len(gpus))]
        proc = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                                stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT)
        processes.append((gpu, proc, log))
    print(json.dumps({'phase': 'GENERATE_TWO_FROZEN_TEST_CASCADES',
                      'physical_gpus': gpus,
                      'pids': {gpu: proc.pid for gpu, proc, _ in processes}}), flush=True)
    codes = []
    for gpu, proc, log in processes:
        codes.append((gpu, proc.wait()))
        log.close()
    if any(code for _, code in codes):
        raise RuntimeError('frozen TEST cascade GPU worker failures: ' + repr(codes))
    shape, top1, exports = 0, 0, 0
    for row in campaign['plans']:
        plan = json.loads(Path(row['shape_plan']).read_text())
        receipt = Path(plan['output']) / 'archive-receipt.json'
        shape += receipt.is_file()
        if row['arm'] == 'V1' and receipt.is_file():
            exports += 'benchmark.glb' in json.loads(receipt.read_text())['sha256']
        if row['arm'] == 'SELECTED':
            pbr = json.loads(Path(row['top1_pbr_plan']).read_text())
            receipt = Path(pbr['output']) / 'archive-receipt.json'
            top1 += receipt.is_file()
            if receipt.is_file():
                exports += 'benchmark.glb' in json.loads(receipt.read_text())['sha256']
    if shape != 1016 or top1 != 508 or exports != 1016:
        raise ValueError(f'incomplete file4 archives: shape={shape}, top1={top1}, glb={exports}')
    result = {'status': 'PASS_FROZEN_V1_SELECTED_TEST508_CASCADES_FILE4_ARCHIVED',
              'samples': 508, 'shape_cases': shape, 'selected_top1_pbr_cases': top1,
              'render_exports': exports,
              'remote': args.remote_base, 'training_updates': 0,
              'physical_gpus': gpus}
    (args.output_root / 'complete.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
