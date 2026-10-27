"""Score one native Pixal3D camera switch with the frozen world surface metric."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from deformnext_benchmark.io import canonical_hash
from deformnext_benchmark.metrics import sample_surface, surface_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--sample-id', required=True)
    parser.add_argument('--official-ungiven-surface', type=Path, required=True)
    parser.add_argument('--official-given-surface', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.manifest.open() as stream:
        row = next((item for line in stream if
                    (item := json.loads(line))['sample_id'] == args.sample_id), None)
    if row is None:
        raise ValueError('sample missing from frozen manifest')
    with np.load(args.official_ungiven_surface) as saved:
        old_v, old_f = saved['vertices'].copy(), saved['faces'].copy()
    with np.load(args.official_given_surface) as saved:
        given_v, given_f = saved['vertices'].copy(), saved['faces'].copy()
    with np.load(row['source_geometry']) as saved:
        target_f = saved['F'].copy()
    target_v = np.load(row['target_vertices'])
    # Exactly the official export's fixed rotation and GLB axis conversion.
    # Do not fit or rescale either prediction to GT.
    rotation = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float64)
    native_to_glb = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    glb_to_world = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    given_world = given_v @ (glb_to_world @ rotation @ native_to_glb).T
    stable = int(canonical_hash(args.sample_id)[:8], 16)
    reference = sample_surface(target_v, target_f, count=20000, seed=stable + 1)
    values = {}
    for name, vertices, faces in (('official_ungiven', old_v, old_f),
                                  ('official_given', given_world, given_f)):
        prediction = sample_surface(vertices, faces, count=20000, seed=stable)
        values[name] = surface_metrics(prediction, reference, threshold=.02)
    report = {'status': 'PASS_ONE_CASE_PIXAL3D_CAMERA_SURFACE_DIAGNOSTIC',
              'sample_id': args.sample_id, 'surface_samples': 20000,
              'surface_seed': stable, 'no_GT_alignment_or_scale_fit': True,
              'same_official_fixed_export_transform': True,
              'metrics': values, 'representative_or_paper_score': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
