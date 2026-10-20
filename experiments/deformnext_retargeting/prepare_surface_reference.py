"""Prepare evaluation-only GT surface points without loading any model outputs.

Use the existing cascade's source-defined common frame, sampler and seed.
These points are ground truth for CD/F-Score after generation, never model
conditioning or test-set model selection/calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .surface_metrics import sample_surface


def surface_reference(source_vertices, source_faces, target_vertices, target_faces,
                      *, count=20000, seed=101):
    source_vertices = np.asarray(source_vertices)
    target_vertices = np.asarray(target_vertices)
    source_faces = np.asarray(source_faces)
    target_faces = np.asarray(target_faces)
    if (source_vertices.ndim != 2 or source_vertices.shape[1:] != (3,)
            or target_vertices.shape != source_vertices.shape or not len(source_vertices)
            or not np.isfinite(source_vertices).all() or not np.isfinite(target_vertices).all()):
        raise ValueError('paired finite source/target vertex arrays required')
    if not np.array_equal(source_faces, target_faces):
        raise ValueError('the existing source-topology metric requires exact target faces')
    # Preserve the original float32 source normalization BEFORE the existing
    # sampler converts normalized points to float64. No target recentering.
    center = (source_vertices.min(0) + source_vertices.max(0)) / 2
    extent = float(np.ptp(source_vertices, axis=0).max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError('positive source-defined extent required')
    normalized = (target_vertices - center) / extent
    return dict(target_points=sample_surface(normalized, source_faces, count=count, seed=seed),
                source_center=center, source_extent=np.asarray(extent))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--file4-cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('explicitly CPU-only evaluation-reference preparation required')
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    ids = [row['sample_id'] for row in rows]
    if len(rows) != 508 or len(set(ids)) != 508 or any(row['split'] != 'test' for row in rows):
        raise ValueError('exact disjoint test508 population required')
    args.output.mkdir(exist_ok=False)
    (args.output / 'points').mkdir()

    def local(path):
        canonical = Path(path)
        if not canonical.is_absolute() or '..' in canonical.parts:
            raise ValueError('explicit canonical file4 paths required')
        return args.file4_cache_root / canonical.relative_to('/file4')

    records = []
    for number, row in enumerate(rows):
        source_path = local(row['source_geometry'])
        target_path = local(row['target_geometry'])
        vertex_path = local(row['target_vertices'])
        with np.load(source_path, allow_pickle=False) as mesh:
            source, source_faces = mesh['V'].copy(), mesh['F'].copy()
        with np.load(target_path, allow_pickle=False) as mesh:
            target, target_faces = mesh['V'].copy(), mesh['F'].copy()
        final_vertices = np.load(vertex_path, allow_pickle=False)
        if not np.array_equal(target, final_vertices):
            raise ValueError('canonical target and final vertices differ: ' + row['sample_id'])
        payload = surface_reference(source, source_faces, final_vertices, target_faces)
        filename = hashlib.sha256(row['sample_id'].encode()).hexdigest()[:24] + '.npz'
        path = args.output / 'points' / filename
        np.savez_compressed(path, **payload, sample_id=np.asarray(row['sample_id']),
            source_cluster_id=np.asarray(row['source_cluster_id']),
            point_count=np.asarray(20000), surface_seed=np.asarray(101),
            schema=np.asarray('source-defined-GT-surface-reference-v1'))
        with np.load(path, allow_pickle=False) as actual:
            if (not np.array_equal(actual['target_points'], payload['target_points'])
                    or str(actual['sample_id']) != row['sample_id']):
                raise ValueError('surface-reference NPZ readback failed')
        records.append(dict(sample_id=row['sample_id'], source_cluster_id=row['source_cluster_id'],
            source_uid=row['source_uid'], reference='points/' + filename,
            reference_SHA256=sha(path), point_count=20000, surface_seed=101,
            source_geometry=row['source_geometry'], target_geometry=row['target_geometry'],
            target_vertices=row['target_vertices'], source_geometry_SHA256=sha(source_path),
            target_geometry_SHA256=sha(target_path), target_vertices_SHA256=sha(vertex_path),
            source_center=payload['source_center'].tolist(), source_extent=float(payload['source_extent']),
            target_faces_equal_source=True, final_target_vertices_exact=True))
        (args.output / 'progress.json').write_text(json.dumps(dict(current=number + 1, total=508)) + '\n')
        if (number + 1) % 50 == 0:
            print(json.dumps(dict(current=number + 1, total=508, GPU_used=False)), flush=True)
    report = dict(status='PASS_ALL508_GT_SURFACE_REFERENCES', samples=508,
        source_clusters=len({row['source_cluster_id'] for row in rows}),
        manifest=str(args.manifest), manifest_SHA256=sha(args.manifest),
        source_SHA256={str(Path(__file__).resolve()): sha(__file__),
            str(Path(__file__).with_name('surface_metrics.py').resolve()): sha(Path(__file__).with_name('surface_metrics.py'))},
        surface_samples=20000, surface_seed=101, FScore_threshold=.02,
        metrics=['chamfer_L1', 'chamfer_squared_L2', 'surface_FScore'],
        source_defined_normalization=True, per_output_alignment=False,
        target_recentered_or_rescaled=False, geometry_frame='source-defined normalized WORLD',
        prediction_to_world='native mesh vertices @ anchor_rotation / row.mesh_scale, then source center/extent',
        all508_source_target_topologies_and_final_vertices_exact=True,
        reference_points_readback_exact=True, model_predictions_loaded=False,
        model_weights_loaded=False, GPU_used=False, training_or_test_fitting=False,
        model_quality_scores_measured=False, final_test_evaluation_complete=False,
        records=records, full_goal_complete=False)
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'records'}), flush=True)


if __name__ == '__main__':
    main()
