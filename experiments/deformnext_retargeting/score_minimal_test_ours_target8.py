"""Score frozen V1/Selected target camera8 images on one common subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = tuple(f'ours_{arm}_{mode}' for arm in ('v1', 'selected')
                for mode in ('raw', 'source_m0'))


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-plan-root', type=Path, required=True)
    parser.add_argument('--file4-output-root', type=Path, required=True)
    parser.add_argument('--common-subset', type=Path, required=True)
    parser.add_argument('--gt-target8-jobs', type=Path, required=True)
    parser.add_argument('--original-references', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-samples', type=int, required=True)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.expected_samples <= 502:
        raise ValueError('new target8 output and declared common count required')
    from deformnext_benchmark.io import canonical_hash, digest, read_rgba
    from deformnext_benchmark.metrics import image_metrics
    subset = json.loads(args.common_subset.read_text())['sample_ids']
    if len(subset) != args.expected_samples or len(set(subset)) != len(subset):
        raise ValueError('frozen final common subset differs')
    gt_jobs = {job['sample_id']: job for job in rows(args.gt_target8_jobs)}
    refs = {row['sample_id']: row for row in rows(args.original_references)}
    if not set(subset) <= gt_jobs.keys() or not set(subset) <= refs.keys():
        raise ValueError('target GT/reference population differs')
    jobs = {}
    for chunk in sorted(args.render_plan_root.glob('chunk[0-9][0-9][0-9]')):
        receipt = json.loads((chunk / 'completed.json').read_text())
        if receipt['status'] != 'PASS_RENDER_CHUNK_ARCHIVED_ON_FILE4':
            raise ValueError('Ours render chunk not verified on file4')
        for job in rows(chunk / 'jobs.jsonl'):
            if job['camera_id'] != 8:
                continue
            key = job['method'], job['sample_id']
            if key in jobs:
                raise ValueError('duplicate Ours target8 render job')
            jobs[key] = job, args.file4_output_root / chunk.name / job['output_rgba']
    if len(jobs) != 502 * 4:
        raise ValueError('full Ours raw/+M0 target8 render closure required')
    scored = []
    for sid in subset:
        gt = Path('/') / gt_jobs[sid]['output_rgba']
        target, target_alpha = read_rgba(gt, resolution=512, background=[1, 1, 1])
        metrics = {}
        for method in METHODS:
            job, prediction = jobs[method, sid]
            receipt_path = prediction.with_suffix('.receipt.json')
            if not prediction.is_file() or not receipt_path.is_file():
                raise FileNotFoundError('Ours target8 render/receipt missing: ' + sid)
            receipt = json.loads(receipt_path.read_text())
            if (receipt['job_sha256'] != canonical_hash(job)
                    or receipt['output_sha256'] != digest(prediction)):
                raise ValueError('Ours target8 render receipt/hash differs: ' + sid)
            rgb, alpha = read_rgba(prediction, resolution=512, background=[1, 1, 1])
            metrics[method] = image_metrics(rgb, target, alpha, target_alpha)
        scored.append({'sample_id': sid, 'source_cluster_id': refs[sid]['source_cluster_id'],
                       'methods': metrics})
    clusters = sorted({row['source_cluster_id'] for row in scored})
    rng = np.random.default_rng(20261028)
    draws = rng.integers(0, len(clusters), size=(20_000, len(clusters)))
    indices = {cluster: [i for i, row in enumerate(scored)
                         if row['source_cluster_id'] == cluster] for cluster in clusters}

    def contrast(left: str, right: str, metric: str) -> dict:
        left_values = np.asarray([row['methods'][left][metric] for row in scored])
        right_values = np.asarray([row['methods'][right][metric] for row in scored])
        delta = right_values - left_values
        sums = np.asarray([delta[indices[cluster]].sum() for cluster in clusters])
        counts = np.asarray([len(indices[cluster]) for cluster in clusters])
        bootstrap = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        return {'left_mean': float(left_values.mean()), 'right_mean': float(right_values.mean()),
                'paired_delta': float(delta.mean()),
                'paired_delta_cluster_ci95': [float(x) for x in np.quantile(bootstrap, [.025, .975])]}

    summary = {}
    for metric in ('PSNR', 'SSIM', 'M_IoU'):
        summary[metric] = {
            'selected_minus_v1_raw': contrast('ours_v1_raw', 'ours_selected_raw', metric),
            'selected_minus_v1_source_m0': contrast('ours_v1_source_m0',
                                                    'ours_selected_source_m0', metric),
            'v1_source_m0_minus_raw': contrast('ours_v1_raw', 'ours_v1_source_m0', metric),
            'selected_source_m0_minus_raw': contrast('ours_selected_raw',
                                                    'ours_selected_source_m0', metric)}
    result = {'status': 'PASS_OURS_V1_SELECTED_RAW_AND_M0_TARGET8_PAIRED',
              'samples': len(scored), 'source_clusters': len(clusters),
              'sample_ids': subset,
              'metrics': summary, 'per_sample': scored}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'samples': len(scored)}), flush=True)


if __name__ == '__main__':
    main()
