"""Finish the paired input-M0 evaluation after all aligned renders are archived.

Run this on node16, where the frozen baseline meshes, GT renders, and offline
learned metric registry are available. The GPU is used only for learned metrics.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def run(argv: list[str], *, env: dict[str, str] | None = None) -> None:
    print(json.dumps({'command': argv}), flush=True)
    subprocess.run(argv, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True,
                        help='Readable mirror of the frozen GT root; data must map to /file4')
    parser.add_argument('--score-root', type=Path, required=True)
    parser.add_argument('--gt-target8-jobs', type=Path, required=True)
    parser.add_argument('--raw-target8-jobs', type=Path, required=True)
    parser.add_argument('--metric-python', type=Path, required=True)
    parser.add_argument('--metric-registry', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--wait-process-name', default='',
                        help='Defer learned GPU metrics while this named producer remains live')
    args = parser.parse_args()
    if not str(args.campaign).startswith('/file4/youngwoo/deformation/experiments/retargeting-v1/source-sim3-test508-20261028/'):
        raise ValueError('declared source-only file4 campaign required')
    frozen_gt = args.baseline_root / 'shared/gt-a02'
    if ((args.reference_root / 'data').resolve() != Path('/file4')
            or (args.reference_root / 'renders').resolve() != (frozen_gt / 'renders').resolve()):
        raise ValueError('readable reference root must preserve the original GT files and renders')
    if args.work_root.exists():
        raise FileExistsError('new work root required: ' + str(args.work_root))
    args.work_root.mkdir(parents=True)
    script_root = Path(__file__).parent
    benchmark_repo = args.baseline_root.parents[2]
    metrics_python = args.metric_python.parent / 'python'
    if (not (benchmark_repo / 'deformnext_benchmark/io.py').is_file()
            or not metrics_python.is_file()):
        raise FileNotFoundError('node16 benchmark package or metrics Python is missing')
    script_env = dict(os.environ, PYTHONPATH=str(benchmark_repo),
                      OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    reference = args.baseline_root / 'launch/trellis2-follow-independent-a01/references.jsonl'
    shards = [args.score_root / f'source-sim3-shard{i}.jsonl' for i in range(4)]
    expected = [882, 882, 875, 875]
    while True:
        chunks = sorted(args.campaign.glob('chunk[0-9][0-9][0-9]'))
        complete = [chunk for chunk in chunks if (chunk / 'completed.json').is_file()]
        scores = [sum(1 for _ in path.open()) if path.is_file() else 0 for path in shards]
        print(json.dumps({'phase': 'WAIT_RENDER_CAMPAIGN', 'chunks': len(chunks),
                          'complete_chunks': len(complete), 'score_rows': scores}), flush=True)
        if len(chunks) == len(complete) == 32 and scores == expected:
            break
        time.sleep(args.poll_seconds)
    binding = args.work_root / 'binding'
    run([str(metrics_python), str(script_root / 'build_source_sim3_eval_manifests.py'),
         '--render-plan-root', str(args.campaign), '--scores', *map(str, shards),
         '--original-references', str(reference), '--file4-output-root', str(args.campaign),
         '--output-root', str(binding)], env=script_env)
    target8 = args.work_root / 'target8-paired.json'
    run([str(metrics_python), str(script_root / 'score_source_sim3_target8.py'),
         '--render-plan-root', str(args.campaign), '--file4-output-root', str(args.campaign),
         '--raw-target8-jobs', str(args.raw_target8_jobs),
         '--gt-target8-jobs', str(args.gt_target8_jobs),
         '--original-references', str(reference), '--output', str(target8)], env=script_env)
    if args.wait_process_name:
        while True:
            processes = subprocess.check_output(['ps', '-eo', 'args='], text=True).splitlines()
            busy = [line for line in processes if ('/' + args.wait_process_name + ' ') in line]
            if not busy:
                break
            print(json.dumps({'phase': 'WAIT_GPU_PRODUCER',
                              'name': args.wait_process_name,
                              'matching_processes': len(busy)}), flush=True)
            time.sleep(args.poll_seconds)
    evaluation = args.work_root / 'evaluation'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
               BENCHMARK_ASSIGNED_GPU=str(args.gpu),
               PYTHONPATH=str(args.baseline_root.parents[2]),
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    run([str(args.metric_python), '-m', 'deformnext_benchmark', 'evaluate',
         '--references', str(binding / 'references.jsonl'),
         '--predictions', str(binding / 'predictions.jsonl'),
         '--protocol', str(args.protocol),
         '--reference-root', str(args.reference_root),
         '--prediction-root', '/', '--output', str(evaluation),
         '--metric-registry', str(args.metric_registry), '--device', 'cuda:0'], env=env)
    summary = json.loads((evaluation / 'summary.json').read_text())
    if (set(summary['methods']) != {'trellis2', 'pixal3d', 'reconviagen', 'mvadapter',
                                    'corradapter', 'shapeup', 'spacecontrol'}
            or any(row['successful_predictions'] != 502 or not row['full_metric_computation']
                   for row in summary['methods'].values())):
        raise ValueError('full 7-method common502 evaluation did not pass')
    archive = args.campaign / 'final-evaluation'
    if archive.exists():
        raise FileExistsError(archive)
    archive.mkdir()
    for path in [target8, binding / 'references.jsonl', binding / 'predictions.jsonl',
                 binding / 'manifest.json', evaluation / 'summary.json', evaluation / 'table.csv']:
        shutil.copy2(path, archive / path.name)
    (archive / 'completion.json').write_text(json.dumps({
        'status': 'PASS_SOURCE_M0_FULL502_SEVEN_METHODS_TEN_METRICS',
        'source_alignment_reference': 'input_source_M0_only',
        'target_M1_used_for_registration': False,
        'evaluation_reference_root': str(args.reference_root),
        'cases': 502, 'methods': 7, 'cameras': [8, 9, 10],
        'paired_target8': str(target8), 'evaluation': str(evaluation),
    }, indent=2) + '\n')
    print(json.dumps({'status': 'PASS_SOURCE_M0_FULL502_SEVEN_METHODS_TEN_METRICS',
                      'archive': str(archive)}), flush=True)


if __name__ == '__main__':
    main()
