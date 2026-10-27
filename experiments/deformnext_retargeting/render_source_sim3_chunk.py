"""Render one staged source-M0-aligned chunk on explicitly assigned GPUs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunk', type=Path, required=True)
    parser.add_argument('--stage-root', type=Path, required=True)
    parser.add_argument('--blender', type=Path, required=True)
    parser.add_argument('--batch-script', type=Path, required=True)
    parser.add_argument('--benchmark-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--threads-per-gpu', type=int, default=4)
    args = parser.parse_args()
    gpus = args.gpus.split(',')
    if (not gpus or len(gpus) != len(set(gpus)) or any(not gpu.isdigit() for gpu in gpus)
            or args.threads_per_gpu < 1):
        raise ValueError('explicit unique physical GPUs and positive thread count required')
    if args.stage_root != args.chunk / 'stage':
        raise ValueError('keep rendered artifacts within their immutable chunk')
    jobs = [json.loads(line) for line in (args.chunk / 'jobs.jsonl').read_text().splitlines()]
    if not jobs or len(jobs) != json.loads((args.chunk / 'plan.json').read_text())['jobs']:
        raise ValueError('chunk job count mismatch')
    for mesh in (args.chunk / 'input-files.txt').read_text().splitlines():
        if not (args.stage_root / mesh).is_file():
            raise FileNotFoundError(args.stage_root / mesh)
    logs = args.chunk / 'logs'
    logs.mkdir(exist_ok=True)

    shards = [list(range(rank, len(jobs), len(gpus))) for rank in range(len(gpus))]
    processes = []
    for gpu, indices in zip(gpus, shards):
        index_file = args.chunk / f'gpu{gpu}-indices.json'
        index_file.write_text(json.dumps(indices) + '\n')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, BENCHMARK_ASSIGNED_GPU=gpu,
                   OMP_NUM_THREADS=str(args.threads_per_gpu), OPENBLAS_NUM_THREADS='1')
        command = [str(args.blender), '-b', '-noaudio', '-t', str(args.threads_per_gpu),
                   '--python', str(args.batch_script), '--',
                   '--benchmark-root', str(args.benchmark_root), '--protocol', str(args.protocol),
                   '--jobs', str(args.chunk / 'jobs.jsonl'), '--indices', str(index_file),
                   '--root', str(args.stage_root), '--device', 'OPTIX']
        stream = (logs / f'gpu{gpu}.log').open('w')
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        processes.append((gpu, process, stream))
    failures = []
    for gpu, process, stream in processes:
        code = process.wait()
        stream.close()
        if code:
            failures.append((gpu, code))
    if failures:
        raise RuntimeError(f'render failures {failures}; inspect {logs}')
    for job in jobs:
        png = args.stage_root / job['output_rgba']
        if not png.is_file() or not png.with_suffix('.receipt.json').is_file():
            raise RuntimeError('incomplete render chunk: ' + str(png))
    summary = {'status': 'PASS_M0_SIM3_CHUNK_RENDER', 'jobs': len(jobs),
               'gpus': gpus}
    (args.chunk / 'render-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
