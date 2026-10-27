"""Paired raw and input-M0-aligned surface metrics for frozen external baselines.

The registration consumes only the prediction and source M0 surface. Target M1
geometry is sampled for evaluation and is never passed to the registration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


LAUNCHES = {
    'trellis2': 'trellis2-follow-independent-a01',
    'pixal3d': 'pixal3d-follow-independent-a01',
    'reconviagen': 'reconviagen-follow',
    'mvadapter': 'mvadapter-follow',
    'corradapter': 'corradapter-follow',
    'shapeup': 'shapeup-follow-independent-a01',
    'spacecontrol': 'spacecontrol-follow-a02',
}


def trimmed_fit_score(moving: np.ndarray, fixed: np.ndarray, fraction: float = .7) -> float:
    """Symmetric, coverage-aware input-only registration objective."""
    forward = cKDTree(fixed).query(moving, workers=1)[0]
    reverse = cKDTree(moving).query(fixed, workers=1)[0]
    keep_forward = max(3, int(len(forward) * fraction))
    keep_reverse = max(3, int(len(reverse) * fraction))
    return float((np.partition(forward, keep_forward - 1)[:keep_forward].mean()
                  + np.partition(reverse, keep_reverse - 1)[:keep_reverse].mean()) / 2)


def similarity(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Least squares proper Sim(3) mapping source rows to destination rows."""
    center_source, center_destination = source.mean(axis=0), destination.mean(axis=0)
    a, b = source - center_source, destination - center_destination
    u, singular, vt = np.linalg.svd(a.T @ b / len(source))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = vt.T @ correction @ u.T
    variance = np.square(a).sum() / len(source)
    if variance <= 1e-12:
        raise ValueError('degenerate moving correspondences')
    scale = float((singular * np.diag(correction)).sum() / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('invalid similarity scale')
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = center_destination - matrix[:3, :3] @ center_source
    return matrix


def apply(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def trimmed_pairs(moving: np.ndarray, fixed: np.ndarray, fraction: float = .7):
    forward, forward_index = cKDTree(fixed).query(moving, workers=1)
    reverse, reverse_index = cKDTree(moving).query(fixed, workers=1)
    ni, nj = max(3, int(len(moving) * fraction)), max(3, int(len(fixed) * fraction))
    i = np.argpartition(forward, ni - 1)[:ni]
    j = np.argpartition(reverse, nj - 1)[:nj]
    return np.concatenate((moving[i], moving[reverse_index[j]])), np.concatenate((fixed[forward_index[i]], fixed[j]))


def register_m0(prediction: np.ndarray, source_m0: np.ndarray, *, iterations: int = 20):
    """Two source-only initializations; choose lowest M0 fit, including identity."""
    identity = np.eye(4)
    initial = [identity]
    pred_center, source_center = np.median(prediction, axis=0), np.median(source_m0, axis=0)
    pred_radius = np.median(np.linalg.norm(prediction - pred_center, axis=1))
    source_radius = np.median(np.linalg.norm(source_m0 - source_center, axis=1))
    if pred_radius <= 1e-9 or source_radius <= 1e-9:
        raise ValueError('degenerate source or prediction surface')
    initial_scale = float(source_radius / pred_radius)
    if .25 <= initial_scale <= 4.:
        normalized = np.eye(4)
        normalized[:3, :3] *= initial_scale
        normalized[:3, 3] = source_center - initial_scale * pred_center
        initial.append(normalized)
    candidates = []
    for guess in initial:
        matrix = guess.copy()
        best_score = trimmed_fit_score(apply(prediction, matrix), source_m0)
        best_matrix = matrix.copy()
        for _ in range(iterations):
            transformed = apply(prediction, matrix)
            moving, fixed = trimmed_pairs(transformed, source_m0)
            update = similarity(moving, fixed)
            candidate = update @ matrix
            candidate_scale = float(np.cbrt(np.linalg.det(candidate[:3, :3])))
            if not .25 <= candidate_scale <= 4.:
                break
            score = trimmed_fit_score(apply(prediction, candidate), source_m0)
            if score >= best_score - 1e-9:
                break
            matrix, best_matrix, best_score = candidate, candidate.copy(), score
        candidates.append((best_score, best_matrix))
    score, matrix = min(candidates, key=lambda item: item[0])
    return matrix, score, trimmed_fit_score(prediction, source_m0)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--dataset-manifest', type=Path, required=True)
    parser.add_argument('--common-subset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-samples', type=int, default=0, help='Pilot only; zero scores the full common subset')
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError('valid shard count/index required')

    # Keep the same sampler, score definition and sample seeds as node16 benchmark.
    from deformnext_benchmark.io import canonical_hash, read_mesh
    from deformnext_benchmark.frames import transform_vertices
    from deformnext_benchmark.metrics import sample_surface, surface_metrics

    manifest = {row['sample_id']: row for row in load_rows(args.dataset_manifest)}
    subset = json.loads(args.common_subset.read_text())['sample_ids']
    ids = subset[:args.max_samples] if args.max_samples else subset
    ids = ids[args.shard_index::args.shard_count]
    predictions = {}
    for method, launch in LAUNCHES.items():
        rows = load_rows(args.baseline_root / 'launch' / launch / 'predictions.jsonl')
        predictions[method] = {row['sample_id']: row for row in rows if row.get('status') == 'ok'}
    results = []
    for index, sid in enumerate(ids):
        row = manifest[sid]
        m0_path = Path(row['source_geometry'])
        m1_path = Path(row['target_geometry'])
        sv, sf = read_mesh(m0_path)
        tv, tf = read_mesh(m1_path)
        stable = int(canonical_hash(sid)[:8], 16)
        m0_points = sample_surface(sv, sf, count=4096, seed=stable+101)
        m1_points = sample_surface(tv, tf, count=20000, seed=stable+1)
        for method in LAUNCHES:
            pred = predictions[method].get(sid)
            if pred is None:
                raise ValueError(f'common-success sample missing: {method} {sid}')
            pred_path = args.baseline_root / pred['mesh']
            pv, pf = read_mesh(pred_path)
            pv = transform_vertices(pv, pred['to_world'])
            pred_points = sample_surface(pv, pf, count=20000, seed=stable)
            registration_points = sample_surface(pv, pf, count=4096, seed=stable+103)
            correction, fit_after, fit_before = register_m0(registration_points, m0_points)
            raw = surface_metrics(pred_points, m1_points, threshold=.02)
            aligned = surface_metrics(apply(pred_points, correction), m1_points, threshold=.02)
            results.append({
                'sample_id': sid, 'source_cluster_id': row['source_cluster_id'], 'method': method,
                'prediction_mesh_sha256': hashlib.sha256(pred_path.read_bytes()).hexdigest(),
                'source_geometry_sha256': hashlib.sha256(m0_path.read_bytes()).hexdigest(),
                'registration_reference': 'input_source_M0_only',
                'source_fit_before': fit_before, 'source_fit_after': fit_after,
                'correction_sim3': correction.tolist(),
                'correction_scale': float(np.cbrt(np.linalg.det(correction[:3, :3]))),
                'raw_target_M1': raw, 'source_aligned_target_M1': aligned,
            })
        if (index + 1) % 25 == 0:
            print(f'{index+1}/{len(ids)} samples', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        for item in results:
            handle.write(json.dumps(item, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'status': 'PASS_SOURCE_ONLY_SIM3', 'samples': len(ids), 'methods': list(LAUNCHES),
                      'rows': len(results), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
