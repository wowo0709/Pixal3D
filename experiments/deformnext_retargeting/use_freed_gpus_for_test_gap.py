"""Fill the disjoint high-index TEST gap after official rendering releases GPUs 6/7."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path('/root/dev/.cache/deformnext-retargeting/scale-ungiven-test508-20261028')
CASCADE = ROOT / 'cascade-two-arm-plans-v1/campaign.json'
SS_COMPLETE = ROOT / 'ss-two-arms-v1/complete.json'
CASCADE_COMPLETE = ROOT / 'cascade-two-arm-execution-v2-direct-file4/complete.json'
OFFICIAL_COMPLETE = Path('/root/dev/.cache/deformnext-benchmark-runtime/'
                         'official-rgb1-render-campaign-v2/complete.json')
REMOTE_BASE = ('/file4/youngwoo/deformation/experiments/retargeting-v1/'
               'scale-ungiven-test508-20261028/cascade-v1')


def alive(pid: int) -> bool:
    path = Path(f'/proc/{pid}/stat')
    return path.is_file() and path.read_text().split(') ', 1)[1][0] not in ('Z', 'X')


def receipt_paths(campaign: dict) -> list[tuple[Path, Path, Path]]:
    v1 = [row for row in campaign['plans'] if row['arm'] == 'V1']
    selected = {row['sample_id']: row for row in campaign['plans']
                if row['arm'] == 'SELECTED'}
    if len(v1) != 508 or len(selected) != 508:
        raise ValueError('frozen two-arm TEST508 campaign required')
    paths = []
    for row in v1:
        other = selected[row['sample_id']]
        vroot = Path(json.loads(Path(row['shape_plan']).read_text())['output'])
        sroot = Path(json.loads(Path(other['shape_plan']).read_text())['output'])
        proot = Path(json.loads(Path(other['top1_pbr_plan']).read_text())['output'])
        paths.append((vroot / 'archive-receipt.json',
                      sroot / 'archive-receipt.json',
                      proot / 'archive-receipt.json'))
    return paths


def complete(paths: tuple[Path, Path, Path]) -> bool:
    return all(path.is_file() for path in paths)


def frontier(paths: list[tuple[Path, Path, Path]], rank: int) -> int:
    return next((index for index in range(rank, 508, 4)
                 if not complete(paths[index])), 508)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--primary-pids', required=True,
                        help='Four comma-separated live primary GPU worker PIDs')
    parser.add_argument('--low-index', type=int, default=380)
    parser.add_argument('--high-index', type=int, default=399)
    parser.add_argument('--minimum-gap', type=int, default=32)
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--gpus', default='6,7',
                        help='Comma-separated subset of 6,7; allows a freed GPU to continue alone')
    args = parser.parse_args()
    pids = [int(item) for item in args.primary_pids.split(',')]
    gpus = [int(item) for item in args.gpus.split(',')]
    if (args.output_root.exists() or len(pids) != 4 or len(set(pids)) != 4
            or not gpus or len(set(gpus)) != len(gpus) or not set(gpus) <= {6, 7}
            or not 300 <= args.low_index < args.high_index < 400
            or args.minimum_gap < 24 or args.poll_seconds < 10
            or any(not alive(pid) for pid in pids)
            or not Path('/tmp/file4-ctl-v2').is_socket()):
        raise ValueError('fresh disjoint gap campaign and live primary workers required')
    campaign = json.loads(CASCADE.read_text())
    if (campaign['status'] != 'PREPARED_FROZEN_V1_SELECTED_TEST508_IMAGE_SCALE'
            or campaign['samples'] != 508):
        raise ValueError('frozen campaign differs')
    paths = receipt_paths(campaign)
    args.output_root.mkdir(parents=True)
    while not OFFICIAL_COMPLETE.is_file():
        if any(not alive(pid) and frontier(paths, rank) < 508
               for rank, pid in enumerate(pids)):
            raise RuntimeError('primary GPU worker ended before official render handoff')
        print(json.dumps({'phase': 'WAIT_OFFICIAL_RENDER_GPU_RELEASE',
                          'paired': sum(map(complete, paths))}), flush=True)
        time.sleep(args.poll_seconds)
    official = json.loads(OFFICIAL_COMPLETE.read_text())
    if (official['status'] != 'PASS_OFFICIAL_RGB1_COMMON502_RENDER_ARCHIVED'
            or official['samples'] != 501):
        raise ValueError('501-case official render closure required before GPU reuse')
    print(json.dumps({'phase': 'OFFICIAL_RENDER_RELEASED_GPUS_6_7',
                      'paired': sum(map(complete, paths))}), flush=True)
    results: dict[str, list[int]] = {'6': [], '7': []}
    repo = Path(__file__).resolve().parents[2]

    def worker(gpu: int, parity: int) -> None:
        env = dict(os.environ, PIXAL3D_FILE4_CONTROL_SOCKET='/tmp/file4-ctl-v2')
        for index in range(args.high_index, args.low_index - 1, -1):
            if index % 2 != parity or CASCADE_COMPLETE.is_file():
                continue
            if complete(paths[index]):
                continue
            rank = index % 4
            next_primary = frontier(paths, rank)
            if index - next_primary < args.minimum_gap:
                print(json.dumps({'phase': 'SKIP_AT_PRIMARY_SAFETY_GAP',
                                  'gpu': gpu, 'index': index,
                                  'primary_frontier': next_primary,
                                  'required_gap': args.minimum_gap}), flush=True)
                continue
            if not alive(pids[rank]):
                raise RuntimeError(f'primary rank{rank} stopped before gap case {index}')
            command = [sys.executable, '-u', '-m',
                       'experiments.deformnext_retargeting.run_minimal_test_cascade_worker',
                       '--campaign', str(CASCADE), '--ss-complete', str(SS_COMPLETE),
                       '--runtime-python', '/opt/conda/envs/pixal3d/bin/python',
                       '--remote-host', 'rvi@10.0.12.170', '--remote-base', REMOTE_BASE,
                       '--physical-gpu', str(gpu), '--allowed-gpus', '0,1,2,3,6,7',
                       '--rank', str(index), '--world-size', '508']
            log = args.output_root / f'gpu{gpu}-index{index:03d}.log'
            with log.open('w') as stream:
                run = subprocess.run(command, cwd=repo, env=env,
                                     stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT,
                                     timeout=5400)
            if run.returncode or not complete(paths[index]):
                raise RuntimeError(f'post-official tail case failed: {log}')
            results[str(gpu)].append(index)
            print(json.dumps({'phase': 'PASS_POST_OFFICIAL_DISJOINT_TEST_CASE',
                              'gpu': gpu, 'index': index}), flush=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, gpu % 2) for gpu in gpus]
        for future in futures:
            future.result()
    receipt = {'status': 'PASS_POST_OFFICIAL_DISJOINT_TEST_GAP',
               'official_render': str(OFFICIAL_COMPLETE),
               'paired_after': sum(map(complete, paths)),
               'completed_indices': results,
               'minimum_gap': args.minimum_gap}
    (args.output_root / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
