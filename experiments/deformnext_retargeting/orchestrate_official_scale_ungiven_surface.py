"""Score the explicit common-success intersection after all official TEST508 attempts."""
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
    parser.add_argument('--generation-pid', type=int, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--common-subset', type=Path, required=True)
    parser.add_argument('--pixal3d-repo', type=Path, required=True)
    parser.add_argument('--benchmark-repo', type=Path, required=True)
    parser.add_argument('--metrics-python', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--shards', type=int, default=4)
    args = parser.parse_args()
    if args.shards < 1 or args.shards > 8 or args.output_root.exists():
        raise ValueError('new output root and 1–8 shards required')
    args.output_root.mkdir(parents=True)
    frozen_subset = json.loads(args.common_subset.read_text())['sample_ids']
    if len(frozen_subset) != 502 or len(set(frozen_subset)) != 502:
        raise ValueError('frozen common502 required')
    while True:
        attempts_path = args.official_root / 'attempts-complete.json'
        completed = len(list((args.official_root / 'cases').glob('*/generation.json')))
        failures = len(list((args.official_root / 'cases').glob('*/failure-receipt.json')))
        print(json.dumps({'phase': 'WAIT_OFFICIAL_ATTEMPT_CLOSURE', 'successes': completed,
                          'model_failures': failures, 'expected_attempts': 508,
                          'pid_live': process_exists(args.generation_pid)}), flush=True)
        if attempts_path.is_file():
            break
        if completed + failures > 508 or not process_exists(args.generation_pid):
            raise RuntimeError('official generation ended before 508 closed attempts')
        time.sleep(60)
    attempts = json.loads(attempts_path.read_text())
    successes = set(attempts['success_sample_ids'])
    failed = set(attempts['failure_sample_ids'])
    if (attempts['status'] != 'PASS_OFFICIAL_RGB1_TEST508_ATTEMPTS_CLOSED'
            or attempts['attempted'] != 508 or len(successes) + len(failed) != 508
            or successes & failed or attempts['succeeded'] != len(successes)
            or attempts['model_failed_empty_sparse'] != len(failed)):
        raise ValueError('official TEST508 attempt closure differs')
    subset = [sid for sid in frozen_subset if sid in successes]
    excluded = [sid for sid in frozen_subset if sid in failed]
    if len(subset) + len(excluded) != 502 or not subset:
        raise ValueError('frozen common502 × official success intersection differs')
    subset_path = args.output_root / 'official-success-common.json'
    subset_path.write_text(json.dumps({'status': 'PASS_OFFICIAL_SUCCESS_INTERSECTION',
                                       'sample_ids': subset, 'excluded_model_failures': excluded,
                                       'frozen_external_common': 502}, indent=2) + '\n')

    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(args.pixal3d_repo), str(args.benchmark_repo)))
    env['OMP_NUM_THREADS'] = '4'
    env['OPENBLAS_NUM_THREADS'] = '1'
    work = Path(__file__).resolve().parent
    processes = []
    for rank in range(args.shards):
        output = args.output_root / f'shard{rank}.jsonl'
        log = (args.output_root / f'shard{rank}.log').open('w')
        command = [str(args.metrics_python), '-u', str(work / 'score_official_pixal3d_scale_ungiven.py'),
                   '--official-root', str(args.official_root), '--manifest', str(args.manifest),
                   '--common-subset', str(subset_path), '--expected-samples', str(len(subset)),
                   '--pixal3d-repo', str(args.pixal3d_repo),
                   '--output', str(output), '--shard-count', str(args.shards), '--shard-index', str(rank)]
        processes.append((subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    print(json.dumps({'phase': 'SCORE_OFFICIAL_COMMON502', 'shards': args.shards,
                      'pids': [p.pid for p, _ in processes]}), flush=True)
    codes = []
    for process, log in processes:
        codes.append(process.wait())
        log.close()
    if any(code != 0 for code in codes):
        raise RuntimeError(f'official scoring shard failures: {codes}')
    scored = [json.loads(line) for rank in range(args.shards)
              for line in (args.output_root / f'shard{rank}.jsonl').read_text().splitlines()]
    if len(scored) != len(subset) or {row['sample_id'] for row in scored} != set(subset):
        raise ValueError('final official-success surface score closure failed')
    result = {'status': 'PASS_OFFICIAL_RGB1_SCALE_UNGIVEN_COMMON_SUCCESS_SURFACE',
              'method': 'pixal3d_official_given_camera_scale_ungiven_1024_rgb1',
              'samples': len(scored), 'source_clusters': len({row['source_cluster_id'] for row in scored}),
              'frozen_external_common': 502, 'excluded_model_failures': excluded,
              'shards': args.shards, 'raw_and_source_M0_sim3': True,
              'test_M1_only_used_for_scoring': True}
    (args.output_root / 'complete.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
