"""Close frozen common renders, evaluate them, and write one unified table."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(command: list[str], log: Path, *, env: dict | None = None) -> None:
    if log.exists():
        raise FileExistsError('immutable evaluation command log exists: ' + str(log))
    with log.open('w') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                env=env, cwd='/home/seunghyun/editing_reproduction')
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}): {log}')


def count(root: Path, suffix: str) -> int:
    return sum(1 for _ in root.rglob('*' + suffix)) if root.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--benchmark-repo', type=Path, required=True)
    parser.add_argument('--prep-root', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--max-wait-hours', type=float, default=72.)
    args = parser.parse_args()
    root = args.root
    progress = root / 'completion-progress.json'
    complete = root / 'unified-completion.json'
    if complete.exists():
        raise FileExistsError('unified result already complete')
    start = time.monotonic()
    expected = {'exports': 3048, 'ours_renders': 9144, 'baseline_target8': 4054}
    while True:
        actual = {'exports': count(root / 'exports', '.glb'),
                  'ours_renders': count(root / 'renders', '.receipt.json'),
                  'baseline_target8': count(root / 'target8', '.receipt.json')}
        progress.write_text(json.dumps({'status': 'WAITING_FOR_COMMON_RENDER_CLOSURE',
                                        'expected': expected, 'actual': actual,
                                        'elapsed_hours': (time.monotonic() - start) / 3600}, indent=2) + '\n')
        if any(actual[k] > expected[k] for k in expected):
            raise ValueError('more artifacts than frozen TEST508 contract')
        if actual == expected:
            break
        if (time.monotonic() - start) / 3600 > args.max_wait_hours:
            raise TimeoutError('common rendering did not close within declared window')
        time.sleep(120)

    repo = args.benchmark_repo
    scripts = root / 'scripts'
    run(['/usr/bin/python3', str(scripts / 'bind_common_benchmark_ours.py'),
         '--manifest', str(root / 'export-manifest-v2.jsonl'),
         '--prep-root', str(args.prep_root), '--baseline-root', str(args.baseline_root),
         '--output-root', str(root)], root / 'bind-source910.log')

    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0', BENCHMARK_ASSIGNED_GPU='0',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    ours_score = root / 'source910-ours-evaluation'
    run([str(repo / 'setup/envs/metrics/bin/python'), '-m', 'deformnext_benchmark', 'evaluate',
         '--references', str(root / 'source910-references.jsonl'),
         '--predictions', str(root / 'source910-ours-predictions.jsonl'),
         '--protocol', str(args.prep_root / 'protocol.json'),
         '--reference-root', '/', '--prediction-root', '/', '--output', str(ours_score),
         '--metric-registry', str(repo / 'setup/metric-registry/registry.json'),
         '--device', 'cuda:0'], root / 'source910-ours-evaluation.log', env=env)

    common_source = root / 'common-source-all13-502.json'
    source_env = os.environ.copy()
    source_env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    run([str(repo / 'setup/envs/metrics/bin/python'),
         str(scripts / 'score_common_success_source.py'),
         '--benchmark-repo', str(repo),
         '--protocol', str(args.prep_root / 'protocol.json'),
         '--subset', str(root / 'common-success-502.json'),
         '--external-metrics-root', str(args.baseline_root / 'metrics'),
         '--mode', 'all', '--ours-eval-root', str(ours_score),
         '--external-score', str(root / 'common-source-external-502.json'),
         '--output', str(common_source)], root / 'common-source-all13-502.log', env=source_env)

    target_scores = root / 'target8-scores.json'
    run([str(repo / 'setup/envs/benchmark-cpu/bin/python'),
         str(scripts / 'score_common_target8.py'),
         '--benchmark-repo', str(repo), '--prep-root', str(args.prep_root),
         '--jobs', str(root / 'target8-all-jobs.jsonl'),
         '--ours-render-records', str(root / 'render-records'),
         '--subset', str(root / 'common-success-502.json'),
         '--output', str(target_scores)], root / 'target8-score.log')

    table = root / 'unified-test508-table.tex'
    run(['/usr/bin/python3', str(scripts / 'compose_unified_test508_table.py'),
         '--common-source-score', str(common_source),
         '--target8-scores', str(target_scores), '--output', str(table)],
        root / 'compose-table.log')
    result = {'status': 'PASS_UNIFIED_COMMON_RENDERER_TEST508_TABLE',
              'source_protocol_sha256': json.loads((ours_score / 'summary.json').read_text())['protocol_sha256'],
              'counts': expected, 'ours_summary': str(ours_score / 'summary.json'),
              'common_source_scores': str(common_source),
              'common_subset': str(root / 'common-success-502.json'),
              'target8_scores': str(target_scores), 'table': str(table),
              'table_sha256': sha256(table)}
    complete.write_text(json.dumps(result, indent=2) + '\n')
    progress.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
