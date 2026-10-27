"""Run only frozen V1 and Selected TEST508 SS after the external render GPUs free."""
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
    parser.add_argument('--baseline-plan', type=Path, required=True)
    parser.add_argument('--baseline-render-pid', type=int, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--scale-model', type=Path, required=True)
    parser.add_argument('--panel-file', type=Path, required=True)
    parser.add_argument('--seed-map-file', type=Path, required=True)
    parser.add_argument('--runtime-python', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--gpus', required=True, help='Two unique physical GPU indices')
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    gpus = args.gpus.split(',')
    if (args.output_root.exists() or len(gpus) != 2 or len(set(gpus)) != 2
            or any(not gpu.isdigit() for gpu in gpus) or args.poll_seconds < 10
            or any(not path.is_file() for path in
                   (args.freeze, args.scale_model, args.panel_file,
                    args.seed_map_file, args.runtime_python))):
        raise ValueError('fresh output, frozen inputs, and two explicit GPUs required')
    args.output_root.mkdir(parents=True)
    chunks = [args.baseline_plan / f'chunk{i:03d}' for i in range(32)]
    if any(not (chunk / 'plan.json').is_file() for chunk in chunks):
        raise ValueError('frozen 32-chunk external render plan required')
    while True:
        done = sum((chunk / 'completed.json').is_file()
                   and json.loads((chunk / 'completed.json').read_text()).get('status')
                   == 'PASS_RENDER_CHUNK_ARCHIVED_ON_FILE4' for chunk in chunks)
        print(json.dumps({'phase': 'WAIT_EXTERNAL_RENDER_GPU_RELEASE', 'chunks_complete': done,
                          'baseline_render_live': process_exists(args.baseline_render_pid)}),
              flush=True)
        if done == 32:
            break
        if not process_exists(args.baseline_render_pid):
            raise RuntimeError('external renderer stopped before 32 verified chunks')
        time.sleep(args.poll_seconds)
    package = 'experiments.deformnext_retargeting.evaluate_ungiven_scale_val_ss'
    root = Path(__file__).resolve().parents[2]
    processes = []
    for arm, gpu in zip(('V1', 'SELECTED'), gpus):
        output = args.output_root / arm
        log = (args.output_root / f'{arm}.log').open('w')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                   BENCHMARK_ASSIGNED_GPU=gpu, OMP_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='1', ATTN_BACKEND='sdpa')
        command = [str(args.runtime_python), '-u', '-m', package,
                   '--freeze', str(args.freeze), '--arm', arm,
                   '--mode', 'image-estimated', '--scale-model', str(args.scale_model),
                   '--panel-file', str(args.panel_file),
                   '--seed-map-file', str(args.seed_map_file), '--output', str(output),
                   '--generation-batch', '8', '--split', 'test']
        proc = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
        processes.append((arm, gpu, proc, log))
    print(json.dumps({'phase': 'GENERATE_FROZEN_TEST_SS',
                      'pids': {arm: proc.pid for arm, _, proc, _ in processes},
                      'physical_gpus': {arm: gpu for arm, gpu, _, _ in processes}}), flush=True)
    failed = []
    for arm, _, proc, log in processes:
        code = proc.wait()
        log.close()
        if code:
            failed.append((arm, code))
    if failed:
        raise RuntimeError('frozen TEST508 SS generation failed: ' + repr(failed))
    frozen = json.loads(args.freeze.read_text())
    expected_ids = frozen['population']['sample_ids']
    records = {}
    for arm in ('V1', 'SELECTED'):
        result = json.loads((args.output_root / arm / 'evaluation.json').read_text())
        contract = json.loads((args.output_root / arm / 'scale-contract.json').read_text())
        if (result['status'] != 'PASS' or result['split'] != 'test'
                or result['optimizer_updates_during_evaluation'] != 0
                or result['quick_panel_ids'] != expected_ids
                or contract['status'] != 'PASS_TEST_SS_SCALE_UNPROVIDED_FROZEN'
                or contract['samples'] != 508 or contract['target_M1_scale_not_used_for_inference'] is not True
                or contract['source_M0_mesh_used_in_generation'] is not False):
            raise ValueError('TEST508 frozen SS provenance/closure differs: ' + arm)
        artifacts = result['native_SS_artifacts']
        if artifacts['archived_asset_count'] != 508:
            raise ValueError('TEST508 SS artifact count differs: ' + arm)
        records[arm] = {'output': str(args.output_root / arm),
                        'checkpoint': result['checkpoint'],
                        'voxel_iou': result['quick_metrics']['voxel_iou']}
    completion = {'status': 'PASS_FROZEN_TEST508_V1_SELECTED_SS_IMAGE_SCALE',
                  'samples': 508, 'arms': records, 'training_updates': 0}
    (args.output_root / 'complete.json').write_text(json.dumps(completion, indent=2) + '\n')
    print(json.dumps(completion), flush=True)


if __name__ == '__main__':
    main()
