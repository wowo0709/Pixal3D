"""Score official RGB1 twice and exactly replay cached external metrics on common N.

This is a faster successor to the waiting nine-method evaluator. Its old PID
must be deliberately stopped so downstream watchers retain a live prerequisite
without launching the redundant full nine-method learned evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


EXTERNAL = {'trellis2', 'pixal3d', 'reconviagen', 'mvadapter',
            'corradapter', 'shapeup', 'spacecontrol'}
OFFICIAL = {'pixal3d_official_rgb1_raw', 'pixal3d_official_rgb1_source_m0'}


def state(pid: int) -> str:
    path = Path(f'/proc/{pid}/stat')
    return path.read_text().split(') ', 1)[1][0] if path.is_file() else 'missing'


def run(command: list[str], *, env: dict[str, str] | None = None,
        log: Path | None = None) -> None:
    print(json.dumps({'phase': 'RUN', 'command': command}), flush=True)
    if log is None:
        subprocess.run(command, check=True, env=env)
    else:
        with log.open('w') as stream:
            subprocess.run(command, check=True, env=env,
                           stdout=stream, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-watcher-pid', type=int, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--official-scores', type=Path, required=True)
    parser.add_argument('--official-exports', type=Path, required=True)
    parser.add_argument('--external-evaluation-complete', type=Path, required=True)
    parser.add_argument('--external-full-evaluation', type=Path, required=True)
    parser.add_argument('--external-binding', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--gt-target8-jobs', type=Path, required=True)
    parser.add_argument('--raw-target8-jobs', type=Path, required=True)
    parser.add_argument('--metric-python', type=Path, required=True)
    parser.add_argument('--metrics-script-python', type=Path, required=True)
    parser.add_argument('--metric-registry', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    if (not str(args.campaign).startswith(
            '/file4/youngwoo/deformation/experiments/retargeting-v1/'
            'source-sim3-test508-20261028/official-rgb1-scale-ungiven-common-success-v2')
            or args.gpu != 7 or args.poll_seconds < 10
            or not args.work_root.is_dir()
            or list(args.work_root.iterdir())
            or state(args.old_watcher_pid) != 'T'):
        raise ValueError('paused old watcher and empty inherited nine-method root required')
    frozen_gt = args.baseline_root / 'shared/gt-a02'
    if ((args.reference_root / 'data').resolve() != Path('/file4')
            or (args.reference_root / 'renders').resolve()
            != (frozen_gt / 'renders').resolve()
            or not args.metric_python.is_file()
            or not args.metrics_script_python.is_file()):
        raise ValueError('frozen node16 metric/reference paths required')
    script_root = Path(__file__).parent
    for name in ('build_official_scale_ungiven_eval_manifests.py',
                 'merge_common_success_evaluator_manifests.py',
                 'score_official_scale_ungiven_target8.py',
                 'score_source_sim3_target8.py',
                 'replay_common_subset_evaluation.py'):
        if not (script_root / name).is_file():
            raise FileNotFoundError(script_root / name)
    score_receipt = args.official_scores / 'complete.json'
    export_receipt = args.official_exports / 'complete.json'
    while True:
        n = (json.loads(score_receipt.read_text())['samples']
             if score_receipt.is_file() else None)
        chunks = sorted(args.campaign.glob('chunk[0-9][0-9][0-9]'))
        completed = sum((chunk / 'completed.json').is_file() for chunk in chunks)
        expected = (n + 15) // 16 if n else None
        print(json.dumps({'phase': 'WAIT_OFFICIAL_RENDER_AND_EXTERNAL',
                          'official_samples': n, 'render_chunks': len(chunks),
                          'completed_chunks': completed, 'expected_chunks': expected,
                          'external_done': args.external_evaluation_complete.is_file(),
                          'paused_old_watcher': state(args.old_watcher_pid)}), flush=True)
        if state(args.old_watcher_pid) != 'T':
            raise RuntimeError('old full nine-method evaluator is no longer paused')
        if (expected is not None and len(chunks) == completed == expected
                and export_receipt.is_file()
                and args.external_evaluation_complete.is_file()):
            break
        time.sleep(args.poll_seconds)
    receipts = [(score_receipt, 'PASS_OFFICIAL_RGB1_SCALE_UNGIVEN_COMMON_SUCCESS_SURFACE'),
                (export_receipt, 'PASS_OFFICIAL_RGB1_SCALE_UNGIVEN_COMMON_SUCCESS_EXPORT'),
                (args.external_evaluation_complete,
                 'PASS_SOURCE_M0_FULL502_SEVEN_METHODS_TEN_METRICS')]
    loaded = [json.loads(path.read_text()) for path, _ in receipts]
    if any(row['status'] != expected for row, (_, expected) in zip(loaded, receipts)):
        raise ValueError('official or external prerequisite receipt differs')
    n = loaded[0]['samples']
    if (not 1 <= n <= 502 or loaded[1]['samples'] != n
            or loaded[0]['excluded_model_failures'] != loaded[1]['excluded_model_failures']
            or not (args.external_full_evaluation / 'summary.json').is_file()):
        raise ValueError('final source population or external full evaluation differs')
    for chunk in chunks:
        row = json.loads((chunk / 'completed.json').read_text())
        if row['status'] != 'PASS_RENDER_CHUNK_ARCHIVED_ON_FILE4':
            raise ValueError('official render chunk archive differs: ' + chunk.name)
    original = args.baseline_root / 'launch/trellis2-follow-independent-a01/references.jsonl'
    scores = [args.official_scores / f'shard{i}.jsonl' for i in range(4)]
    script_env = dict(os.environ, PYTHONPATH=str(args.baseline_root.parents[2]),
                      OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    official_binding = args.work_root / 'binding-official'
    run([str(args.metrics_script_python), str(script_root / 'build_official_scale_ungiven_eval_manifests.py'),
         '--render-plan-root', str(args.campaign), '--scores', *map(str, scores),
         '--original-references', str(original), '--file4-output-root', str(args.campaign),
         '--output-root', str(official_binding), '--expected-samples', str(n)], env=script_env)
    binding = args.work_root / 'binding-unified-nine'
    run([str(args.metrics_script_python), str(script_root / 'merge_common_success_evaluator_manifests.py'),
         '--external-binding', str(args.external_binding), '--official-binding', str(official_binding),
         '--output-root', str(binding), '--expected-samples', str(n)], env=script_env)
    official_target8 = args.work_root / 'target8-official-paired.json'
    run([str(args.metrics_script_python), str(script_root / 'score_official_scale_ungiven_target8.py'),
         '--render-plan-root', str(args.campaign), '--file4-output-root', str(args.campaign),
         '--gt-target8-jobs', str(args.gt_target8_jobs), '--original-references', str(original),
         '--output', str(official_target8), '--expected-samples', str(n)], env=script_env)
    external_target8 = args.work_root / 'target8-external-paired.json'
    external_campaign = args.external_evaluation_complete.parent.parent
    run([str(args.metrics_script_python), str(script_root / 'score_source_sim3_target8.py'),
         '--render-plan-root', str(external_campaign),
         '--file4-output-root', str(external_campaign),
         '--raw-target8-jobs', str(args.raw_target8_jobs),
         '--gt-target8-jobs', str(args.gt_target8_jobs),
         '--original-references', str(original),
         '--subset', str(args.official_scores / 'official-success-common.json'),
         '--output', str(external_target8)], env=script_env)
    official_two = args.work_root / 'evaluation-official-two'
    metric_env = dict(script_env, CUDA_VISIBLE_DEVICES=str(args.gpu),
                      BENCHMARK_ASSIGNED_GPU=str(args.gpu))
    run([str(args.metric_python), '-m', 'deformnext_benchmark', 'evaluate',
         '--references', str(official_binding / 'references.jsonl'),
         '--predictions', str(official_binding / 'predictions.jsonl'),
         '--protocol', str(args.protocol), '--reference-root', str(args.reference_root),
         '--prediction-root', '/', '--output', str(official_two),
         '--metric-registry', str(args.metric_registry), '--device', 'cuda:0'],
        env=metric_env, log=args.work_root / 'official-two-metrics.log')
    evaluation = args.work_root / 'evaluation'
    run([str(args.metrics_script_python), str(script_root / 'replay_common_subset_evaluation.py'),
         '--full-evaluation', str(args.external_full_evaluation),
         '--full-binding', str(args.external_binding),
         '--official-evaluation', str(official_two),
         '--official-binding', str(official_binding),
         '--nine-binding', str(binding), '--protocol', str(args.protocol),
         '--output', str(evaluation)], env=script_env,
        log=args.work_root / 'cache-replay.log')
    summary = json.loads((evaluation / 'summary.json').read_text())
    replay = json.loads((evaluation / 'cache-replay-provenance.json').read_text())
    if (set(summary['methods']) != EXTERNAL | OFFICIAL
            or any(row['successful_predictions'] != n
                   or not row['full_metric_computation']
                   or row['distribution']['image_count'] != n * 2
                   for row in summary['methods'].values())
            or replay['status'] != 'PASS_EXACT_CACHED_FEATURE_SUBSET_REPLAY'
            or replay['final_samples'] != n
            or replay['maximum_full_population_replay_error'] > 1e-8):
        raise ValueError('cached nine-method common evaluation differs')
    archive = args.campaign / 'final-evaluation'
    if archive.exists():
        raise FileExistsError('official final archive already exists')
    archive.mkdir()
    for path in (official_target8, external_target8,
                 binding / 'references.jsonl', binding / 'predictions.jsonl',
                 binding / 'manifest.json', evaluation / 'summary.json',
                 evaluation / 'table.csv', evaluation / 'cache-replay-provenance.json'):
        shutil.copy2(path, archive / path.name)
    result = {'status': 'PASS_NINE_METHOD_COMMON_SUCCESS_TEN_METRICS',
              'samples': n, 'frozen_external_common': 502,
              'excluded_model_failures': loaded[0]['excluded_model_failures'],
              'methods': sorted(EXTERNAL | OFFICIAL), 'cameras': [8, 9, 10],
              'source_alignment_reference': 'input_source_M0_only',
              'target_M1_used_for_registration': False,
              'evaluation': str(evaluation),
              'evaluation_mode': 'exact_external_feature_cache_subset_replay_plus_fresh_official_two',
              'replay_provenance': str(evaluation / 'cache-replay-provenance.json')}
    (archive / 'completion.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.work_root / 'cached-evaluation-complete.json').write_text(
        json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
