"""Blender entry point that reuses one Cycles process across assigned jobs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--indices', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--device', choices=('OPTIX', 'CUDA', 'CPU'), required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    sys.path.insert(0, str(args.benchmark_root))
    from deformnext_benchmark.render_blender import main as render_one

    rows = [json.loads(line) for line in args.jobs.read_text().splitlines()]
    indices = json.loads(args.indices.read_text())
    if not indices or len(set(indices)) != len(indices) or any(not 0 <= i < len(rows) for i in indices):
        raise ValueError('valid unique render indices required')
    started = time.monotonic()
    for position, index in enumerate(indices, 1):
        destination = args.root / rows[index]['output_rgba']
        receipt = destination.with_suffix('.receipt.json')
        if destination.is_file() and receipt.is_file():
            print(json.dumps({'index': index, 'status': 'REUSED', 'progress': position}), flush=True)
            continue
        if destination.exists() or receipt.exists():
            raise FileExistsError('partial render needs inspection: ' + str(destination))
        sys.argv = [str(args.benchmark_root / 'deformnext_benchmark/render_blender.py'), '--',
                    '--protocol', str(args.protocol), '--jobs', str(args.jobs),
                    '--job-index', str(index), '--root', str(args.root), '--device', args.device]
        render_one()
        if not (destination.is_file() and receipt.is_file()):
            raise RuntimeError('render did not produce PNG and receipt: ' + str(destination))
        print(json.dumps({'index': index, 'status': 'PASS', 'progress': position,
                          'total': len(indices), 'elapsed_seconds': round(time.monotonic() - started, 2)}), flush=True)


if __name__ == '__main__':
    main()
