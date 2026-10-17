"""Materialize the historical three-source-camera GT slot protocol on a fixed panel.

CPU ray casting only. These routes contain GT geometry and are diagnostic-only;
they never replace inference-safe correspondence caches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .historical_geometry import _coord64, _surface_route
from .topology_labels import SCHEMA as LABEL_SCHEMA, fingerprint

SCHEMA = 'deformnext-historical-three-view-route-v2'


def frame_contract(row):
    return json.dumps({key: row[key] for key in ['mesh_scale', 'latent_geometry_schema',
                       'target_world_center', 'target_world_extent']}, sort_keys=True)


def selected_rows(manifest, panel):
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    ids = json.loads(panel.read_text())['sample_ids']
    lookup = {row['sample_id']: row for row in rows}
    if len(lookup) != len(rows) or len(set(ids)) != len(ids) or not set(ids) <= lookup.keys():
        raise ValueError('fixed reproduction panel has duplicate or missing identities')
    return [lookup[sid] for sid in ids]


def build(row, resolve):
    with np.load(resolve(row['source_geometry'])) as data:
        source, faces = data['V'].copy(), data['F'].copy()
    path = resolve(row['target_vertices'])
    if path.suffix != '.npy':
        raise ValueError('reproduction requires the explicitly rendered M1 vertex array')
    target = np.load(path)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError('M0/M1 vertex topology differs')
    if row.get('latent_geometry_schema') != 'deformnext-m1-render-world-v4':
        raise ValueError('reproduction requires the audited render-world frame')
    cameras = [dict(transform_matrix=matrix, camera_angle_x=angle)
               for matrix, angle in zip(row['transform_matrix'], row['camera_angle_x'], strict=True)]
    if len(cameras) != 4 or row['conditioning_camera_ids'] != [8, 9, 10, 11]:
        raise ValueError('expected the fixed target08 and three source09/10/11 cameras')
    routes = [_surface_route(source, target, faces, camera, cameras[0]) for camera in cameras[1:]]
    arrays = {key: np.stack([route[key] for route in routes]) for key in routes[0]}
    if arrays['valid'].shape != (3, 1024) or np.any(arrays['valid'].sum(1) < 2):
        raise ValueError('historical wrong-address control requires at least two active slots per view')
    if not np.isfinite(arrays['source_pixel'][arrays['valid']]).all():
        raise ValueError('nonfinite valid source read address')
    # These are sampled surface-route regions, not the old full SS authority masks.
    # Keep the name explicit; full regional BCE requires its own certified masks.
    hidden = arrays['valid'] & ~arrays['target_anchor_visible']
    points = arrays['target_point'][hidden]
    coords = _coord64(points, cameras[0], float(row['mesh_scale']))
    arrays['route_hidden_coords64'] = np.unique(coords, axis=0)
    arrays['route_hidden_coords32'] = np.unique(coords // 2, axis=0)
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--certification-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--path-map', action='append', required=True)
    args = parser.parse_args()
    # Avoid taking all CPU cores while GPU training/data loading is active.
    if hasattr(os, 'sched_getaffinity'):
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, set(available[-2:]))
    mappings = [item.split('=', 1) for item in args.path_map]
    def resolve(value):
        for source, target in mappings:
            if value == source or value.startswith(source.rstrip('/') + '/'):
                return Path(target + value[len(source):])
        return Path(value)
    rows = selected_rows(args.manifest, args.panel)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'data').mkdir(exist_ok=True)
    started = time.monotonic()
    records = []
    for row in rows:
        digest = fingerprint(row, resolve)
        label = args.certification_root / (row['sample_id'] + '.npz')
        with np.load(label) as check:
            if str(check['schema']) != LABEL_SCHEMA or str(check['input_sha256']) != digest:
                raise ValueError('existing image/mesh raster certification does not match inputs')
        path = args.output / 'data' / (row['sample_id'] + '.npz')
        if path.exists():
            with np.load(path) as old:
                if (str(old['schema']) != SCHEMA or str(old['input_sha256']) != digest
                        or str(old['frame_contract']) != frame_contract(row)):
                    raise ValueError('immutable reproduction route identity changed')
                arrays = {key: old[key].copy() for key in ['valid', 'target_anchor_visible']}
        else:
            arrays = build(row, resolve)
            temporary = path.with_name('.' + path.name + f'.{os.getpid()}.tmp.npz')
            np.savez(temporary, schema=SCHEMA, sample_id=row['sample_id'], split=row['split'],
                     input_sha256=digest, conditioning_input_sha256=row['conditioning_input_sha256'],
                     frame_contract=frame_contract(row),
                     source_camera_ids=np.asarray(row['conditioning_camera_ids'][1:]),
                     diagnostic_only=np.asarray(True), **arrays)
            os.replace(temporary, path)
        records.append({'sample_id': row['sample_id'], 'source_cluster_id': row['source_cluster_id'],
                        'valid_per_view': arrays['valid'].sum(1).tolist(),
                        'target_hidden_valid': int((arrays['valid'] & ~arrays['target_anchor_visible']).sum()),
                        'route_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        print(json.dumps({'completed': len(records), 'total': len(rows), **records[-1]}), flush=True)
    summary = {'status': 'PASS', 'schema': SCHEMA, 'diagnostic_only': True,
               'samples': len(rows), 'records': records, 'seconds': time.monotonic() - started,
               'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
               'panel_sha256': hashlib.sha256(args.panel.read_bytes()).hexdigest(),
               'geometry_protocol': 'exact historical _surface_route; pixel7.5+16n; visibility tolerance0.002',
               'scope': 'Three source-camera GT read/write maps. Sampled hidden-surface coordinates are not full regional SS authority masks.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output / 'panel-manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    print(json.dumps({'status': 'PASS', 'samples': len(rows), 'seconds': summary['seconds']}), flush=True)


if __name__ == '__main__':
    main()
