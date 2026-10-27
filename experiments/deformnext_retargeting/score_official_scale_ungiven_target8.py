"""Score official RGB1 target8 renders in paired raw and source-M0 modes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = {'raw': 'pixal3d_official_rgb1_raw',
           'source_m0': 'pixal3d_official_rgb1_source_m0'}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-plan-root', type=Path, required=True)
    parser.add_argument('--file4-output-root', type=Path, required=True)
    parser.add_argument('--gt-target8-jobs', type=Path, required=True)
    parser.add_argument('--original-references', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-samples', type=int, default=502)
    args = parser.parse_args()
    if args.output.exists() or args.expected_samples < 1:
        raise ValueError('new output and positive expected sample count required')
    from deformnext_benchmark.io import canonical_hash, digest, read_rgba
    from deformnext_benchmark.metrics import image_metrics
    gt_jobs = {job['sample_id']: job for job in read_jsonl(args.gt_target8_jobs)}
    refs = {row['sample_id']: row for row in read_jsonl(args.original_references)}
    jobs = {}
    for chunk in sorted(args.render_plan_root.glob('chunk[0-9][0-9][0-9]')):
        receipt = json.loads((chunk / 'completed.json').read_text())
        if receipt['status'] != 'PASS_RENDER_CHUNK_ARCHIVED_ON_FILE4':
            raise ValueError('official render chunk not verified on file4')
        for job in read_jsonl(chunk / 'jobs.jsonl'):
            if job['camera_id'] != 8:
                continue
            key = job['method'], job['sample_id']
            if key in jobs:
                raise ValueError('duplicate official target8 job')
            jobs[key] = (job, args.file4_output_root / chunk.name / job['output_rgba'])
    ids = sorted({sid for _, sid in jobs})
    if len(ids) != args.expected_samples or len(jobs) != args.expected_samples * 2:
        raise ValueError('complete official raw/+M0 target8 population required')
    if not set(ids) <= set(gt_jobs) or not set(ids) <= set(refs):
        raise ValueError('target8 GT/reference population differs')
    rows = []
    for sid in ids:
        gt = Path('/') / gt_jobs[sid]['output_rgba']
        target, target_alpha = read_rgba(gt, resolution=512, background=[1, 1, 1])
        metrics = {}
        for mode, method in METHODS.items():
            job, prediction = jobs[method, sid]
            receipt_path = prediction.with_suffix('.receipt.json')
            if not prediction.is_file() or not receipt_path.is_file():
                raise FileNotFoundError('official target8 render/receipt missing: ' + sid)
            receipt = json.loads(receipt_path.read_text())
            if (receipt['job_sha256'] != canonical_hash(job)
                    or receipt['output_sha256'] != digest(prediction)):
                raise ValueError('official target8 receipt/hash differs: ' + sid)
            rgb, alpha = read_rgba(prediction, resolution=512, background=[1, 1, 1])
            metrics[mode] = image_metrics(rgb, target, alpha, target_alpha)
        rows.append({'sample_id': sid, 'source_cluster_id': refs[sid]['source_cluster_id'],
                     **metrics})
    clusters = sorted({row['source_cluster_id'] for row in rows})
    if args.expected_samples == 502 and len(clusters) != 133:
        raise ValueError('expected 133 frozen source clusters')
    rng = np.random.default_rng(20261028)
    draws = rng.integers(0, len(clusters), size=(20_000, len(clusters)))
    summary = {}
    for metric in ('PSNR', 'SSIM', 'M_IoU'):
        raw = np.asarray([row['raw'][metric] for row in rows])
        aligned = np.asarray([row['source_m0'][metric] for row in rows])
        diff = aligned - raw
        by_cluster = {cluster: [i for i, row in enumerate(rows)
                                if row['source_cluster_id'] == cluster] for cluster in clusters}
        sums = np.asarray([diff[by_cluster[cluster]].sum() for cluster in clusters])
        counts = np.asarray([len(by_cluster[cluster]) for cluster in clusters])
        bootstrap = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        summary[metric] = {'raw_mean': float(raw.mean()),
                           'source_m0_mean': float(aligned.mean()),
                           'paired_delta': float(diff.mean()),
                           'paired_delta_cluster_ci95': [float(x) for x in
                                                         np.quantile(bootstrap, [.025, .975])],
                           'fraction_improved': float((diff > 0).mean())}
    result = {'status': 'PASS_OFFICIAL_RGB1_RAW_AND_M0_TARGET8_PAIRED',
              'samples': len(rows), 'source_clusters': len(clusters),
              'metrics': summary, 'per_sample': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'samples': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
