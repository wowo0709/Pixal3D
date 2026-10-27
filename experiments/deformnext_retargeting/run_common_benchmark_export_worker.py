"""Resume one deterministic shard of the frozen TEST508 common export."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--exporter', type=Path, required=True)
    parser.add_argument('--pixal3d-repo', type=Path, required=True)
    parser.add_argument('--benchmark-repo', type=Path, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size or args.gpu < 0:
        raise ValueError('invalid worker assignment')
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    if len(rows) != 3048:
        raise ValueError('frozen manifest must contain exactly 3048 rows')
    log_root = args.root / 'export-logs' / f'rank{args.rank}'
    log_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    env['OMP_NUM_THREADS'] = '4'
    env['OPENBLAS_NUM_THREADS'] = '1'
    done = 0
    for index, row in enumerate(rows):
        if index % args.world_size != args.rank:
            continue
        output = args.root / row['export_relpath']
        report_path = output.with_suffix('.json')
        if output.is_file() and report_path.is_file():
            report = json.loads(report_path.read_text())
            if (report.get('status') != 'PASS_COMMON_BENCHMARK_MESH_EXPORT'
                    or report.get('prediction') != row['prediction']
                    or report.get('texture_size') != 2048):
                raise ValueError('existing export receipt mismatch: ' + str(output))
            continue
        if output.exists() or report_path.exists():
            raise FileExistsError('partial export needs explicit review: ' + str(output))
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(args.exporter),
                   '--pixal3d-repo', str(args.pixal3d_repo),
                   '--benchmark-repo', str(args.benchmark_repo),
                   '--prediction', row['prediction'],
                   '--render-geometry', row['render_geometry'],
                   '--anchor-evaluation', row['anchor_evaluation'],
                   '--generation', row['generation'],
                   '--mesh-scale', str(row['mesh_scale']),
                   '--output', str(output), '--texture-size', '2048']
        started = time.monotonic()
        proc = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
        log = {'index': index, 'sample_id': row['sample_id'], 'arm': row['arm'],
               'gpu': args.gpu, 'returncode': proc.returncode,
               'seconds': time.monotonic() - started, 'output': proc.stdout,
               'export': str(output)}
        (log_root / f'{index:04d}.json').write_text(json.dumps(log, indent=2) + '\n')
        print(json.dumps({k: log[k] for k in ('index', 'sample_id', 'arm', 'gpu',
                                             'returncode', 'seconds')}), flush=True)
        if proc.returncode or not output.is_file() or not report_path.is_file():
            raise RuntimeError('export failed; inspect ' + str(log_root / f'{index:04d}.json'))
        done += 1
        if args.limit and done >= args.limit:
            break
    print(json.dumps({'rank': args.rank, 'new_exports': done, 'status': 'PASS_WORKER_SHARD'}), flush=True)


if __name__ == '__main__':
    main()
