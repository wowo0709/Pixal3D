"""Create common-camera Blender jobs after source-M0-only Sim(3) fitting.

Meshes remain the frozen baseline GLBs. Only each job's explicit to_world
matrix changes; every camera and renderer setting remains pinned.
"""
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
    parser.add_argument('--scores', type=Path, nargs='+', required=True)
    parser.add_argument('--target8-jobs', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--chunk-samples', type=int, default=24)
    parser.add_argument('--expected-samples', type=int, default=502, help='Use 1 for a mechanical smoke only')
    args = parser.parse_args()
    if args.output_root.exists() or args.chunk_samples < 1:
        raise ValueError('new output root and positive chunk size required')
    scores = {}
    for path in args.scores:
        for row in read_jsonl(path):
            key = row['method'], row['sample_id']
            if key in scores or row['registration_reference'] != 'input_source_M0_only':
                raise ValueError('duplicate or non-M0 source fit')
            scores[key] = row
    ids = sorted({sid for _, sid in scores})
    if len(ids) != args.expected_samples or len(scores) != args.expected_samples * len(METHODS):
        raise ValueError('all seven methods on common502 required')
    native_jobs = {}
    for job in read_jsonl(args.target8_jobs):
        key = job['method'], job['sample_id']
        if key in native_jobs:
            raise ValueError('duplicate native mesh binding')
        native_jobs[key] = job
    references = {row['sample_id']: row for row in read_jsonl(args.references)}
    args.output_root.mkdir(parents=True)
    total_jobs = 0
    for start in range(0, len(ids), args.chunk_samples):
        chunk_ids = ids[start:start + args.chunk_samples]
        chunk = args.output_root / f'chunk{start//args.chunk_samples:03d}'
        chunk.mkdir()
        render_jobs = []
        mesh_paths = set()
        metric_mesh_paths = set()
        for sid in chunk_ids:
            reference = references[sid]
            reference_cameras = {view['camera_id']: view['camera'] for view in reference['views']}
            if not {9, 10} <= reference_cameras.keys():
                raise ValueError('official common camera9/10 required')
            for method in METHODS:
                key = method, sid
                original, score = native_jobs[key], scores[key]
                correction = np.asarray(score['correction_sim3'], dtype=np.float64)
                native_to_world = np.asarray(original['to_world'], dtype=np.float64)
                aligned_to_world = correction @ native_to_world
                if (correction.shape != (4, 4) or native_to_world.shape != (4, 4)
                        or not np.isfinite(aligned_to_world).all()
                        or not np.allclose(aligned_to_world[3], [0, 0, 0, 1])):
                    raise ValueError('invalid source-fitted affine')
                if original['protocol_sha256'] != reference['protocol_sha256']:
                    raise ValueError('native/reference protocol mismatch')
                mesh_paths.add(original['mesh'])
                metric_mesh_paths.add(original['metric_mesh'])
                cameras = {8: original['camera'], 9: reference_cameras[9], 10: reference_cameras[10]}
                for camera_id in (8, 9, 10):
                    job = dict(original, camera_id=camera_id, camera=cameras[camera_id],
                               to_world=aligned_to_world.tolist(),
                               output_rgba=f'renders/{method}/{sid}/view{camera_id:02d}.png')
                    render_jobs.append(job)
        (chunk / 'jobs.jsonl').write_text(''.join(json.dumps(job, sort_keys=True) + '\n' for job in render_jobs))
        (chunk / 'input-files.txt').write_text(''.join(path + '\n' for path in sorted(mesh_paths | metric_mesh_paths)))
        (chunk / 'plan.json').write_text(json.dumps({'sample_ids': chunk_ids, 'jobs': len(render_jobs),
                                                     'render_meshes': len(mesh_paths),
                                                     'metric_meshes': len(metric_mesh_paths),
                                                     'cameras': [8, 9, 10]}, indent=2) + '\n')
        total_jobs += len(render_jobs)
    print(json.dumps({'status': 'PASS_SOURCE_M0_RENDER_PLAN', 'samples': len(ids), 'jobs': total_jobs,
                      'chunks': (len(ids) + args.chunk_samples - 1) // args.chunk_samples}))


if __name__ == '__main__':
    main()
