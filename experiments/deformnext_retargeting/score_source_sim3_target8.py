"""Paired raw versus input-M0-aligned target-camera image metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = ('trellis2', 'pixal3d', 'reconviagen', 'mvadapter', 'corradapter', 'shapeup', 'spacecontrol')


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-plan-root', type=Path, required=True)
    parser.add_argument('--file4-output-root', type=Path, required=True)
    parser.add_argument('--raw-target8-jobs', type=Path, required=True)
    parser.add_argument('--gt-target8-jobs', type=Path, required=True)
    parser.add_argument('--original-references', type=Path, required=True)
    parser.add_argument('--subset', type=Path,
                        help='Explicit common-success sample IDs; default is frozen common502')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from deformnext_benchmark.io import canonical_hash, digest, read_rgba
    from deformnext_benchmark.metrics import image_metrics
    raw_jobs = {(job['method'], job['sample_id']): job for job in read_jsonl(args.raw_target8_jobs)}
    gt_jobs = {job['sample_id']: job for job in read_jsonl(args.gt_target8_jobs)}
    refs = {row['sample_id']: row for row in read_jsonl(args.original_references)}
    aligned_jobs = {}
    for chunk in sorted(args.render_plan_root.glob('chunk[0-9][0-9][0-9]')):
        if not (chunk / 'completed.json').is_file():
            raise ValueError('render chunk is not archived')
        for job in read_jsonl(chunk / 'jobs.jsonl'):
            if job['camera_id'] == 8:
                key = job['method'], job['sample_id']
                if key in aligned_jobs:
                    raise ValueError('duplicate aligned target8 job')
                aligned_jobs[key] = (job, args.file4_output_root / chunk.name / job['output_rgba'])
    if len(aligned_jobs) != 502 * len(METHODS):
        raise ValueError('full paired external common502 target8 jobs required')
    ids = (json.loads(args.subset.read_text())['sample_ids'] if args.subset else
           sorted({sid for _, sid in aligned_jobs}))
    if (not ids or len(ids) != len(set(ids))
            or not set(ids) <= {sid for _, sid in aligned_jobs}):
        raise ValueError('declared common-success subset differs')
    rows = []
    for sid in ids:
        gt = Path('/') / gt_jobs[sid]['output_rgba']
        reference, reference_alpha = read_rgba(gt, resolution=512, background=[1, 1, 1])
        for method in METHODS:
            raw = Path('/') / raw_jobs[method, sid]['output_rgba']
            source_job, aligned = aligned_jobs[method, sid]
            if not raw.is_file() or not aligned.is_file() or not aligned.with_suffix('.receipt.json').is_file():
                raise FileNotFoundError('paired raw/aligned render missing')
            receipt = json.loads(aligned.with_suffix('.receipt.json').read_text())
            if receipt.get('job_sha256') != canonical_hash(source_job) or receipt.get('output_sha256') != digest(aligned):
                raise ValueError('aligned target8 receipt mismatch')
            raw_rgb, raw_alpha = read_rgba(raw, resolution=512, background=[1, 1, 1])
            aligned_rgb, aligned_alpha = read_rgba(aligned, resolution=512, background=[1, 1, 1])
            rows.append({'method': method, 'sample_id': sid,
                         'source_cluster_id': refs[sid]['source_cluster_id'],
                         'raw': image_metrics(raw_rgb, reference, raw_alpha, reference_alpha),
                         'source_aligned': image_metrics(aligned_rgb, reference, aligned_alpha, reference_alpha)})
    clusters = sorted({row['source_cluster_id'] for row in rows})
    if len(ids) == 502 and len(clusters) != 133:
        raise ValueError('expected 133 source clusters')
    rng = np.random.default_rng(20261028)
    draws = rng.integers(0, len(clusters), size=(2000, len(clusters)))
    summary = {}
    for method in METHODS:
        data = [row for row in rows if row['method'] == method]
        if len(data) != len(ids):
            raise ValueError('method missing common population')
        values = {}
        for metric in ('PSNR', 'SSIM', 'M_IoU'):
            raw = np.asarray([row['raw'][metric] for row in data])
            aligned = np.asarray([row['source_aligned'][metric] for row in data])
            diff = aligned - raw
            by_cluster = {cluster: [i for i, row in enumerate(data) if row['source_cluster_id'] == cluster]
                          for cluster in clusters}
            sums = np.asarray([diff[by_cluster[cluster]].sum() for cluster in clusters])
            counts = np.asarray([len(by_cluster[cluster]) for cluster in clusters])
            bootstrap = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
            values[metric] = {'raw_mean': float(raw.mean()), 'source_aligned_mean': float(aligned.mean()),
                              'paired_delta': float(diff.mean()),
                              'paired_delta_cluster_ci95': [float(x) for x in np.quantile(bootstrap, [.025, .975])],
                              'fraction_improved': float((diff > 0).mean())}
        summary[method] = values
    output = {'schema': ('deformnext-input-M0-Sim3-target8-paired-common-success-v2'
                         if args.subset else 'deformnext-input-M0-Sim3-target8-paired-common502-v1'),
              'samples': len(ids), 'sample_ids': ids,
              'source_clusters': len(clusters), 'methods': summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'PASS_SOURCE_M0_TARGET8_PAIRED', 'samples': len(ids),
                      'methods': len(summary)}))


if __name__ == '__main__':
    main()
