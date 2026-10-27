"""Evaluate only four new Ours conditions after the reused nine-method result."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


WORK = '/home/youngwoo/retargeting-unified-test508'
OURS_RENDER = ('/file4/youngwoo/deformation/experiments/retargeting-v1/'
               'source-sim3-test508-20261028/ours-v1-selected-scale-ungiven-common502-v1')
OFFICIAL_SURFACE = WORK + '/official-scale-ungiven-source-sim3-common-success-v2'
NINE_ROOT = WORK + '/unified-nine-method-final-evaluation-v4'
OURS_SURFACE = WORK + '/ours-v1-selected-source-sim3-final-v1'
OUT = WORK + '/ours-four-final-evaluation-v1'
BASELINE = '/home/seunghyun/editing_reproduction/outputs/deformnext-baselines/2026-10-24-frozen-test'
BENCHMARK = '/home/seunghyun/editing_reproduction'
PROTOCOL = BENCHMARK + '/baseline-benchmark-preparation-v3-20261020/protocol.json'


def live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def remote(host: str, port: int, command: list[str]) -> str:
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-p', str(port), host,
                             ' '.join(shlex.quote(part) for part in command)],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:] + result.stdout[-1500:])
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ours-render-complete', type=Path, required=True)
    parser.add_argument('--ours-render-watcher-pid', type=int, required=True)
    parser.add_argument('--nine-evaluator-pid', type=int, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--remote-host', default='youngwoo@n16.unist.info')
    parser.add_argument('--ssh-port', type=int, default=55555)
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    if args.output_root.exists() or args.poll_seconds < 10:
        raise ValueError('fresh Ours evaluation monitor required')
    args.output_root.mkdir(parents=True)
    nine_completion = (OURS_RENDER.replace('ours-v1-selected-scale-ungiven-common502-v1',
                       'official-rgb1-scale-ungiven-common-success-v2')
                       + '/final-evaluation/completion.json')
    while True:
        local_done = args.ours_render_complete.is_file()
        status = json.loads(remote(args.remote_host, args.ssh_port, [
            'python3', '-c',
            ('import json,os;from pathlib import Path;'
             f'print(json.dumps({{"nine_done":Path({nine_completion!r}).is_file(),'
             f'"nine_live":os.path.exists("/proc/{args.nine_evaluator_pid}")}}))')]))
        print(json.dumps({'phase': 'WAIT_REUSED_NINE_AND_NEW_OURS_FOUR',
                          'ours_render_done': local_done,
                          'ours_watcher_live': live(args.ours_render_watcher_pid),
                          **status}), flush=True)
        if local_done and status['nine_done']:
            break
        if not local_done and not live(args.ours_render_watcher_pid):
            raise RuntimeError('Ours renderer stopped before file4 closure')
        if not status['nine_done'] and not status['nine_live']:
            raise RuntimeError('nine-method evaluator stopped before closure')
        time.sleep(args.poll_seconds)
    ours = json.loads(args.ours_render_complete.read_text())
    if (ours['status'] != 'PASS_OURS_V1_SELECTED_RAW_AND_M0_COMMON502_RENDER_ARCHIVED'
            or ours['samples'] != 502 or ours['jobs'] != 6024):
        raise ValueError('frozen 502×2×2×3 Ours render closure required')
    nine = json.loads(remote(args.remote_host, args.ssh_port, ['cat', nine_completion]))
    if (nine['status'] != 'PASS_NINE_METHOD_COMMON_SUCCESS_TEN_METRICS'
            or not 1 <= nine['samples'] <= 502):
        raise ValueError('reused nine-method final evaluation differs')
    n = nine['samples']
    scripts = Path(__file__).parent
    for name in ('build_minimal_test_ours_eval_manifests.py',
                 'score_minimal_test_ours_target8.py',
                 'merge_common_thirteen_method_manifests.py'):
        subprocess.run(['rsync', '-a', '-e', f'ssh -o BatchMode=yes -p {args.ssh_port}',
                        str(scripts / name), f'{args.remote_host}:{WORK}/{name}'], check=True)
    remote(args.remote_host, args.ssh_port, ['test', '!', '-e', OUT])
    remote(args.remote_host, args.ssh_port, ['mkdir', '-p', OUT])
    metric_py = '/home/seunghyun/editing_reproduction/setup/envs/metrics/bin/python'
    metric_run = '/home/seunghyun/editing_reproduction/setup/envs/metrics/bin/M0Sim3-metrics'
    script_env = ['env', 'PYTHONPATH=' + BENCHMARK,
                  'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=1']
    subset = OFFICIAL_SURFACE + '/official-success-common.json'
    original = BASELINE + '/launch/trellis2-follow-independent-a01/references.jsonl'
    scores = [f'{OURS_SURFACE}/shard{i}.jsonl' for i in range(4)]
    binding = OUT + '/binding-ours-four'
    print(remote(args.remote_host, args.ssh_port,
                 [*script_env, metric_py, WORK + '/build_minimal_test_ours_eval_manifests.py',
                  '--render-plan-root', OURS_RENDER, '--scores', *scores,
                  '--common-subset', subset, '--original-references', original,
                  '--file4-output-root', OURS_RENDER, '--output-root', binding,
                  '--expected-samples', str(n)]), flush=True)
    print(remote(args.remote_host, args.ssh_port,
                 [*script_env, metric_py, WORK + '/merge_common_thirteen_method_manifests.py',
                  '--nine-binding', NINE_ROOT + '/binding-unified-nine',
                  '--ours-binding', binding, '--output-root', OUT + '/binding-common-thirteen',
                  '--expected-samples', str(n)]), flush=True)
    print(remote(args.remote_host, args.ssh_port,
                 [*script_env, metric_py, WORK + '/score_minimal_test_ours_target8.py',
                  '--render-plan-root', OURS_RENDER, '--file4-output-root', OURS_RENDER,
                  '--common-subset', subset,
                  '--gt-target8-jobs', WORK + '/target8-gt-jobs.jsonl',
                  '--original-references', original,
                  '--output', OUT + '/target8-ours-four.json',
                  '--expected-samples', str(n)]), flush=True)
    # The nine already-scored methods are reused; only four new Ours conditions
    # consume learned-metric GPU time on the exact same reference population.
    evaluation = OUT + '/evaluation-ours-four'
    command = ['env', 'CUDA_VISIBLE_DEVICES=7', 'BENCHMARK_ASSIGNED_GPU=7',
               'PYTHONPATH=' + BENCHMARK, 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=1',
               metric_run, '-m', 'deformnext_benchmark', 'evaluate',
               '--references', binding + '/references.jsonl',
               '--predictions', binding + '/predictions.jsonl',
               '--protocol', PROTOCOL,
               '--reference-root', WORK + '/gt-accessible-root',
               '--prediction-root', '/', '--output', evaluation,
               '--metric-registry', BENCHMARK + '/setup/metric-registry/registry.json',
               '--device', 'cuda:0']
    with (args.output_root / 'learned-metrics.log').open('w') as log:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-p', str(args.ssh_port),
                                 args.remote_host,
                                 ' '.join(shlex.quote(part) for part in command)],
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Ours four-condition learned metrics failed; inspect local log')
    summary = json.loads(remote(args.remote_host, args.ssh_port,
                                ['cat', evaluation + '/summary.json']))
    expected = {'ours_v1_raw', 'ours_v1_source_m0',
                'ours_selected_raw', 'ours_selected_source_m0'}
    if (set(summary['methods']) != expected
            or any(row['successful_predictions'] != n or not row['full_metric_computation']
                   for row in summary['methods'].values())):
        raise ValueError('four new methods lack full matched learned metrics')
    archive = OURS_RENDER + '/final-evaluation'
    remote(args.remote_host, args.ssh_port, ['mkdir', '-p', archive])
    for source, name in ((evaluation + '/summary.json', 'summary-ours-four.json'),
                         (evaluation + '/table.csv', 'table-ours-four.csv'),
                         (OUT + '/target8-ours-four.json', 'target8-ours-four.json'),
                         (OUT + '/binding-common-thirteen/manifest.json', 'common-thirteen-manifest.json')):
        remote(args.remote_host, args.ssh_port, ['cp', source, archive + '/' + name])
    report = {'status': 'PASS_FOUR_NEW_OURS_METHODS_TEN_METRICS_REUSING_NINE',
              'samples': n, 'methods': sorted(expected),
              'nine_method_evaluation_reused': nine_completion,
              'source_alignment_reference': 'input_source_M0_only',
              'target_M1_used_for_registration': False,
              'evaluation': evaluation}
    (args.output_root / 'complete.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
