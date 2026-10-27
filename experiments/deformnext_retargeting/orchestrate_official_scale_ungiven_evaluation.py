"""Evaluate the complete official RGB1 raw/+M0 render campaign on node16."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def run(argv: list[str], *, env: dict[str, str] | None = None) -> None:
    print(json.dumps({'command': argv}), flush=True)
    subprocess.run(argv, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--official-scores', type=Path, required=True)
    parser.add_argument('--official-exports', type=Path, required=True)
    parser.add_argument('--external-evaluation-complete', type=Path, required=True)
    parser.add_argument('--external-evaluator-pid', type=int, required=True)
    parser.add_argument('--external-binding', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--gt-target8-jobs', type=Path, required=True)
    parser.add_argument('--raw-target8-jobs', type=Path, required=True)
    parser.add_argument('--metric-python', type=Path, required=True)
    parser.add_argument('--metrics-script-python', type=Path, required=True,
                        help='Python environment with numpy and deformnext_benchmark for binding')
    parser.add_argument('--metric-registry', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    if (not str(args.campaign).startswith(
            '/file4/youngwoo/deformation/experiments/retargeting-v1/'
            'source-sim3-test508-20261028/official-rgb1-scale-ungiven-common-success-v2')
            or args.gpu != 7 or args.poll_seconds < 10 or args.work_root.exists()):
        raise ValueError('fresh official file4 campaign and node16 GPU7 required')
    frozen_gt = args.baseline_root / 'shared/gt-a02'
    if ((args.reference_root / 'data').resolve() != Path('/file4')
            or (args.reference_root / 'renders').resolve() != (frozen_gt / 'renders').resolve()):
        raise ValueError('reference root must preserve original GT data and renders')
    args.work_root.mkdir(parents=True)
    script_root = Path(__file__).parent
    script_env = dict(os.environ, PYTHONPATH=str(args.baseline_root.parents[2]),
                      OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    original = args.baseline_root / 'launch/trellis2-follow-independent-a01/references.jsonl'
    scores = [args.official_scores / f'shard{i}.jsonl' for i in range(4)]
    while True:
        chunks = sorted(args.campaign.glob('chunk[0-9][0-9][0-9]'))
        complete_chunks = sum((chunk / 'completed.json').is_file() for chunk in chunks)
        surface_path = args.official_scores / 'complete.json'
        export_path = args.official_exports / 'complete.json'
        surface_done = surface_path.is_file()
        export_done = export_path.is_file()
        baseline_done = args.external_evaluation_complete.is_file()
        samples = json.loads(surface_path.read_text())['samples'] if surface_done else None
        expected_chunks = (samples + 15) // 16 if samples else None
        print(json.dumps({'phase': 'WAIT_OFFICIAL_COMMON_EVALUATION',
                          'render_chunks': len(chunks), 'completed_render_chunks': complete_chunks,
                          'expected_chunks': expected_chunks, 'official_success_samples': samples,
                          'surface_done': surface_done, 'export_done': export_done,
                          'external_evaluation_done': baseline_done,
                          'external_evaluator_live': process_exists(args.external_evaluator_pid)}),
              flush=True)
        if (expected_chunks is not None and len(chunks) == complete_chunks == expected_chunks
                and export_done and baseline_done):
            break
        if not baseline_done and not process_exists(args.external_evaluator_pid):
            raise RuntimeError('external evaluation ended before its completion receipt')
        time.sleep(args.poll_seconds)
    receipts = []
    for path, status in ((args.official_scores / 'complete.json',
                          'PASS_OFFICIAL_RGB1_SCALE_UNGIVEN_COMMON_SUCCESS_SURFACE'),
                         (args.official_exports / 'complete.json',
                          'PASS_OFFICIAL_RGB1_SCALE_UNGIVEN_COMMON_SUCCESS_EXPORT'),
                         (args.external_evaluation_complete,
                          'PASS_SOURCE_M0_FULL502_SEVEN_METHODS_TEN_METRICS')):
        receipt = json.loads(path.read_text())
        if receipt['status'] != status:
            raise ValueError('prerequisite receipt status differs: ' + str(path))
        receipts.append(receipt)
    sample_count = receipts[0]['samples']
    if (receipts[1]['samples'] != sample_count
            or receipts[0]['excluded_model_failures'] != receipts[1]['excluded_model_failures']):
        raise ValueError('official scorer/export populations differ')
    official_binding = args.work_root / 'binding-official'
    run([str(args.metrics_script_python), str(script_root / 'build_official_scale_ungiven_eval_manifests.py'),
         '--render-plan-root', str(args.campaign), '--scores', *map(str, scores),
         '--original-references', str(original), '--file4-output-root', str(args.campaign),
         '--output-root', str(official_binding), '--expected-samples', str(sample_count)], env=script_env)
    binding = args.work_root / 'binding-unified-nine'
    run([str(args.metrics_script_python), str(script_root / 'merge_common_success_evaluator_manifests.py'),
         '--external-binding', str(args.external_binding),
         '--official-binding', str(official_binding), '--output-root', str(binding),
         '--expected-samples', str(sample_count)], env=script_env)
    official_target8 = args.work_root / 'target8-official-paired.json'
    run([str(args.metrics_script_python), str(script_root / 'score_official_scale_ungiven_target8.py'),
         '--render-plan-root', str(args.campaign), '--file4-output-root', str(args.campaign),
         '--gt-target8-jobs', str(args.gt_target8_jobs),
         '--original-references', str(original), '--output', str(official_target8),
         '--expected-samples', str(sample_count)], env=script_env)
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
    evaluation = args.work_root / 'evaluation'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
               BENCHMARK_ASSIGNED_GPU=str(args.gpu),
               PYTHONPATH=str(args.baseline_root.parents[2]),
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    run([str(args.metric_python), '-m', 'deformnext_benchmark', 'evaluate',
         '--references', str(binding / 'references.jsonl'),
         '--predictions', str(binding / 'predictions.jsonl'),
         '--protocol', str(args.protocol), '--reference-root', str(args.reference_root),
         '--prediction-root', '/', '--output', str(evaluation),
         '--metric-registry', str(args.metric_registry), '--device', 'cuda:0'], env=env)
    summary = json.loads((evaluation / 'summary.json').read_text())
    expected = {'trellis2', 'pixal3d', 'reconviagen', 'mvadapter', 'corradapter',
                'shapeup', 'spacecontrol', 'pixal3d_official_rgb1_raw',
                'pixal3d_official_rgb1_source_m0'}
    if (set(summary['methods']) != expected
            or any(row['successful_predictions'] != sample_count or not row['full_metric_computation']
                   for row in summary['methods'].values())):
        raise ValueError('official two-method common502 full evaluation did not pass')
    archive = args.campaign / 'final-evaluation'
    if archive.exists():
        raise FileExistsError('official evaluation archive exists')
    archive.mkdir()
    for path in [official_target8, external_target8,
                 binding / 'references.jsonl', binding / 'predictions.jsonl',
                 binding / 'manifest.json', evaluation / 'summary.json', evaluation / 'table.csv']:
        shutil.copy2(path, archive / path.name)
    result = {'status': 'PASS_NINE_METHOD_COMMON_SUCCESS_TEN_METRICS',
              'samples': sample_count, 'frozen_external_common': 502,
              'excluded_model_failures': receipts[0]['excluded_model_failures'],
              'methods': sorted(expected), 'cameras': [8, 9, 10],
              'source_alignment_reference': 'input_source_M0_only',
              'target_M1_used_for_registration': False,
              'evaluation': str(evaluation)}
    (archive / 'completion.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
