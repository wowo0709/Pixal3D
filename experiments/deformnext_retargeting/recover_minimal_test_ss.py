"""Close frozen TEST508 SS after the Selected-only projection-root recovery."""
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
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--v1-pid', type=int, required=True)
    parser.add_argument('--selected-pid', type=int, required=True)
    parser.add_argument('--stopped-original-watcher-pid', type=int, required=True)
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    if (args.poll_seconds < 10 or (args.output_root / 'complete.json').exists()
            or state(args.stopped_original_watcher_pid) != 'T'):
        raise ValueError('original SS watcher must be deliberately stopped and receipt absent')
    paths = {arm: (args.output_root / arm / 'evaluation.json',
                   args.output_root / arm / 'scale-contract.json')
             for arm in ('V1', 'SELECTED')}
    while True:
        done = {arm: all(path.is_file() for path in pair)
                for arm, pair in paths.items()}
        processes = {'V1': state(args.v1_pid), 'SELECTED': state(args.selected_pid)}
        print(json.dumps({'phase': 'WAIT_RECOVERED_TEST508_SS',
                          'receipts': done, 'processes': processes}), flush=True)
        if all(done.values()):
            break
        if any(not done[arm] and processes[arm] in ('missing', 'Z') for arm in done):
            raise RuntimeError('SS producer exited without complete frozen receipts')
        if state(args.stopped_original_watcher_pid) != 'T':
            raise RuntimeError('original SS watcher resumed before recovery closure')
        time.sleep(args.poll_seconds)
    frozen = json.loads(args.freeze.read_text())
    expected_ids = frozen['population']['sample_ids']
    if len(expected_ids) != 508 or len(set(expected_ids)) != 508:
        raise ValueError('TEST508 frozen population differs')
    records = {}
    for arm in ('V1', 'SELECTED'):
        result = json.loads(paths[arm][0].read_text())
        contract = json.loads(paths[arm][1].read_text())
        if (result['status'] != 'PASS' or result['split'] != 'test'
                or result['optimizer_updates_during_evaluation'] != 0
                or result['quick_panel_ids'] != expected_ids
                or contract['status'] != 'PASS_TEST_SS_SCALE_UNPROVIDED_FROZEN'
                or contract['samples'] != 508
                or contract['target_M1_scale_not_used_for_inference'] is not True
                or contract['source_M0_mesh_used_in_generation'] is not False
                or result['native_SS_artifacts']['archived_asset_count'] != 508):
            raise ValueError('TEST508 frozen SS provenance/closure differs: ' + arm)
        records[arm] = {'output': str(args.output_root / arm),
                        'checkpoint': result['checkpoint'],
                        'voxel_iou': result['quick_metrics']['voxel_iou']}
    selected_root = frozen['population']['manifest']['path']
    expected_projection = str(Path(selected_root).parents[2] / 'projection_maps_world_v4')
    selected_contract = json.loads(paths['SELECTED'][1].read_text())
    if selected_contract['projection_root'] != expected_projection:
        raise ValueError('recovered Selected did not use frozen TEST508 projection bytes')
    completion = {'status': 'PASS_FROZEN_TEST508_V1_SELECTED_SS_IMAGE_SCALE',
                  'samples': 508, 'arms': records, 'training_updates': 0,
                  'recovery': 'Selected-only TEST508 projection-root binding; V1 retained',
                  'selected_projection_root': expected_projection}
    target = args.output_root / 'complete.json'
    tmp = args.output_root / 'complete.json.tmp'
    tmp.write_text(json.dumps(completion, indent=2) + '\n')
    tmp.replace(target)
    print(json.dumps(completion), flush=True)
    os.kill(args.stopped_original_watcher_pid, signal.SIGCONT)


if __name__ == '__main__':
    main()
