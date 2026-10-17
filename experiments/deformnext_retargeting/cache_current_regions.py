"""Current-data physical SS64/32 supervision for the original regional loss.

GT geometry is label/evaluator-only. RGB-predicted O/D queries live separately.
Native current camera/scale is never replaced by an oracle pair-union scale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .cache_task_rgb_transfer import surface_scene
from .cache_reproduction_routes import frame_contract
from .historical_geometry import camera_rays_for_pixels, barycentric_points, world_to_target_anchor_points
from .historical_region_authority import (_surface_support, _camera_clip, target_anchor_to_world_points,
    voxel_grid_center_corner_points, classify_projected_voxel_cubes, pool_support, FREE, SURFACE)
from .historical_reproduction import make_loss_regions
from .topology_labels import fingerprint, mask_agreement
from .reproduce_q2c import read, reference, write_new

SCHEMA = 'deformnext-current-native1024-authority-regions64-v3'


def coordinate_set(coords):
    return set(map(tuple, np.asarray(coords, np.int32).reshape(-1, 3).tolist()))


def coordinate_array(coords):
    return np.asarray(sorted(coords), np.int32).reshape(-1, 3)


def lock_target_authority(rebuilt64, saved_coords, ss_source_hash):
    """Require exact current1024->64 geometry and actual encoder-input parity."""
    if hashlib.sha256(saved_coords.tobytes()).hexdigest() != ss_source_hash:
        raise ValueError('saved support does not match the actual SS latent source hash')
    physical_set, saved_set = coordinate_set(rebuilt64), coordinate_set(saved_coords)
    if not physical_set or physical_set != saved_set:
        raise ValueError(f'native1024 pooled target differs from actual SS input: missing{len(saved_set-physical_set)}, extra{len(physical_set-saved_set)}')
    return coordinate_array(saved_set), {
        'actual_encoder_support_source_hash_exact': True,
        'actual_encoder_support_source_hash': ss_source_hash,
        'native1024_pooled_target_equals_saved_encoder_support': True,
        'rebuilt_native1024_vs_encoder_support_iou': 1.,
        'encoder_only_target_voxels': 0, 'rebuilt_native_only_target_voxels': 0}


def visible_coords(points, row):
    camera = {'transform_matrix': row['transform_matrix'][0]}
    canonical = world_to_target_anchor_points(points, camera, scale=row['mesh_scale'])
    inside = ((canonical >= -.5) & (canonical < .5)).all(-1)
    coords = np.floor((canonical[inside] + .5) * 64).astype(np.int32)
    return coordinate_set(np.unique(coords, axis=0)), int((~inside).sum())


def raster(scene, row, view, vertices, faces, image):
    import open3d as o3d
    yy, xx = np.meshgrid(np.arange(512), np.arange(512), indexing='ij')
    pixels = np.stack((xx, yy), -1).reshape(-1, 2).astype(np.float32)
    camera = {'transform_matrix': row['transform_matrix'][view], 'camera_angle_x': row['camera_angle_x'][view]}
    rays = camera_rays_for_pixels(pixels, np.asarray(camera['transform_matrix']), camera['camera_angle_x'], resolution=512)
    hit = scene.cast_rays(o3d.core.Tensor(rays), nthreads=2)
    ids, uv = hit['primitive_ids'].numpy().astype(np.int64), hit['primitive_uvs'].numpy()
    valid = np.isfinite(hit['t_hit'].numpy()) & (ids < len(faces))
    bary = np.stack((1-uv[:, 0]-uv[:, 1], uv[:, 0], uv[:, 1]), -1)
    points = barycentric_points(vertices, faces, ids[valid], bary[valid])
    with Image.open(image) as source:
        if source.size != (512, 512):
            raise ValueError('region authority requires the audited512px RGB input')
        alpha = (np.asarray(source.convert('RGBA'))[..., 3] > 127).reshape(-1)
    agreement = mask_agreement(torch.from_numpy(valid.reshape(512, 512)),
        torch.from_numpy(alpha.reshape(512, 512)), alpha_cutouts=True)
    observed = alpha[valid]
    ndc_depth = np.zeros(512 * 512, np.float32)
    clip = _camera_clip(points, camera)
    ndc_depth[valid] = clip[:, 2] / clip[:, 3]
    # Original conservative cube classifier indexes raw OpenGL bottom-up rows.
    return {'ids': ids[valid][observed], 'bary': bary[valid][observed], 'points': points[observed],
        'depth_gl': np.flipud(ndc_depth.reshape(512, 512)).copy(),
        'mask_gl': np.flipud(valid.reshape(512, 512)).copy(),
        'IoU': agreement[0], 'containment': agreement[1]}


def build(row, resolve):
    if row['latent_geometry_schema'] != 'deformnext-m1-render-world-v4' or row['conditioning_camera_ids'] != [8, 9, 10, 11]:
        raise ValueError('region labels require audited world-v4 native frame')
    with np.load(resolve(row['source_geometry'])) as data:
        source, faces = data['V'].copy(), data['F'].copy()
    target = np.load(resolve(row['target_vertices']))
    if source.shape != target.shape:
        raise ValueError('source/target topology differs')
    camera = {'transform_matrix': row['transform_matrix'][0], 'camera_angle_x': row['camera_angle_x'][0]}
    canonical = [world_to_target_anchor_points(v, camera, scale=row['mesh_scale']) for v in [source, target]]
    # Match the current native Shape1024->SS64 input construction for both
    # states. Direct64 voxelization does not commute with high-res pooling.
    source64, rebuilt_target64 = [pool_support(_surface_support(v, faces, resolution=1024),
        source_resolution=1024, target_resolution=64) for v in canonical]
    with np.load(resolve(row['shape_latent'])) as saved:
        saved_coords = saved['coords'].copy()
    with np.load(resolve(row['ss_latent'])) as encoded:
        if str(encoded['schema']) != 'deformnext-ss64-m1-anchor-v3':
            raise ValueError('unexpected current SS latent schema')
        ss_source_hash = str(encoded['source_coords_sha256'])
    target64, target_audit = lock_target_authority(rebuilt_target64, saved_coords, ss_source_hash)
    source_set, target_set = coordinate_set(source64), coordinate_set(target64)
    target_raster = raster(surface_scene(target, faces), row, 0, target, faces, resolve(row['target_anchor']))
    target_visible, target_outside = visible_coords(target_raster['points'], row)
    target_visible &= target_set
    scene = surface_scene(source, faces)
    corresponding, source_visible, agreements, outside = set(), set(), [], []
    for view in [1, 2, 3]:
        observed = raster(scene, row, view, source, faces, resolve(row['source_images'][view-1]))
        moved = barycentric_points(target, faces, observed['ids'], observed['bary'])
        moved_coords, dropped_target = visible_coords(moved, row)
        source_coords, dropped_source = visible_coords(observed['points'], row)
        corresponding |= moved_coords & target_set
        source_visible |= source_coords & source_set
        agreements.append({'view': view, 'IoU': observed['IoU'], 'containment': observed['containment']})
        outside.append({'view': view, 'target_pixels': dropped_target, 'source_pixels': dropped_source})
    matched = coordinate_array(corresponding - target_visible)
    stale = coordinate_array((source_set - target_set) & source_visible)
    indices, samples = voxel_grid_center_corner_points(64)
    clip = _camera_clip(target_anchor_to_world_points(samples, camera, scale=row['mesh_scale']).reshape(-1, 3), camera)
    sample_ndc = (clip[:, :3] / clip[:, 3:]).reshape(len(indices), 9, 3)
    def membership(coords):
        result = np.zeros((64, 64, 64), bool)
        if len(coords):
            result[tuple(np.asarray(coords).T)] = True
        return result.reshape(-1)
    states = classify_projected_voxel_cubes(sample_ndc, target_support=membership(target64),
        visible_target_support=membership(coordinate_array(target_visible)),
        raster_depth=target_raster['depth_gl'], raster_mask=target_raster['mask_gl'], depth_tolerance=1e-4)
    arrays = {'source64': source64, 'target64': target64, 'rebuilt_target64': rebuilt_target64,
        'free64': indices[states == FREE],
        'surface64': indices[states == SURFACE], 'matched_source_exclusive64': matched, 'stale_visible64': stale}
    for name in ['matched_source_exclusive', 'stale_visible']:
        arrays[name+'32'] = pool_support(arrays[name+'64'], source_resolution=64, target_resolution=32)
    regions = make_loss_regions(arrays, arrays)
    counts = {str(res): {name: int(value.sum()) for name, value in masks.items()} for res, masks in regions.items()}
    # Zero source/stale regions are recorded, never replaced using observed loss.
    if not len(arrays['free64']) or not len(arrays['surface64']):
        raise ValueError('current region authority lacks observed FREE/SURFACE')
    return arrays, {'region_counts': counts, **target_audit,
        'source_vertices_outside_native_AABB': int(((canonical[0] < -.5) | (canonical[0] >= .5)).any(-1).sum()),
        'target_vertices_outside_native_AABB': int(((canonical[1] < -.5) | (canonical[1] >= .5)).any(-1).sum()),
        'observed_target_pixels_outside_native_AABB': target_outside,
        'source_views_outside_native_AABB': outside,
        'raster_agreement': [{'view': 0, 'IoU': target_raster['IoU'], 'containment': target_raster['containment']}, *agreements]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['population', 'manifest', 'file4-mirror', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--preflight-summary', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    population = read(args.population)
    if population['status'] != 'PASS' or not population['full_population'] or len(population['records']) != 320:
        raise ValueError('expected frozen256/64 current RGB pilot population')
    chosen = sum(([r for r in population['records'] if r['role'] == role][:32] for role in ['train', 'calibration']), [])
    if args.preflight_only:
        chosen = chosen[:1]
    elif (args.preflight_summary is None or read(args.preflight_summary)['status'] != 'PASS_PREFLIGHT_ONLY'
            or read(args.preflight_summary)['schema'] != SCHEMA):
        raise ValueError('full label cache requires actual aligned one-pair CPU preflight')
    lookup = {r['sample_id']: r for r in (json.loads(line) for line in args.manifest.read_text().splitlines())}
    resolve = lambda value: args.file4_mirror/value.removeprefix('/file4/') if value.startswith('/file4/') else Path(value)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'summary.json').exists():
        raise FileExistsError('immutable current regional authority exists')
    started, records = time.monotonic(), []
    for selected in chosen:
        row = lookup[selected['sample_id']]
        digest = fingerprint(row, resolve)
        if digest != selected['geometry_input_sha256'] or row['source_cluster_id'] != selected['source_cluster_id']:
            raise ValueError('frozen region population/geometry fingerprint differs')
        path = args.output/'data'/(row['sample_id']+'.npz')
        if path.exists():
            raise FileExistsError('preserve partial region cache; choose fresh namespace')
        arrays, audit = build(row, resolve)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name('.'+path.name+f'.{os.getpid()}.tmp.npz')
        np.savez(temporary, schema=SCHEMA, supervision_only=True, sample_id=row['sample_id'],
            role=selected['role'], input_sha256=digest, frame_contract=frame_contract(row), **arrays)
        os.replace(temporary, path)
        records.append({'sample_id': row['sample_id'], 'source_cluster_id': row['source_cluster_id'],
            'role': selected['role'], 'artifact': reference(path), 'geometry_input_sha256': digest,
            'frame_contract': frame_contract(row), 'actual_encoder_input': reference(resolve(row['shape_latent'])),
            'actual_SS_latent': reference(resolve(row['ss_latent'])), **audit})
        print(json.dumps({'completed': len(records), 'of': len(chosen), 'seconds': time.monotonic()-started,
            'last_region_counts': audit['region_counts']}), flush=True)
    write_new(args.output/'summary.json', {'status': 'PASS_PREFLIGHT_ONLY' if args.preflight_only else 'PASS',
        'schema': SCHEMA, 'records': records, 'population': reference(args.population),
        'manifest': reference(args.manifest), 'seconds': time.monotonic()-started, 'supervision_only': True,
        'selection': 'first32 per role from pre-frozen category/hash pilot, no metric/region-size selection',
        'frame': 'current audited native world-v4 target camera/scale; no oracle pair-union rescaling',
        'label_backend': 'CPU Open3D pixel-center barycentrics and original flexible voxelization at current native1024 then exact pool64; original cube-ray/loss functions',
        'target64_authority': 'rebuilt native1024 pool64 must exactly equal actual encoder input, which must match SS latent source_coords_sha256',
        'alpha_policy': 'observed RGBA foreground restricts positive/source evidence; conservative FREE uses opaque geometry',
        'out_of_AABB_policy': 'drop out-of-grid interpolated points; never clamp them onto boundary voxels',
        'visible_device_for_optional_package_import': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'query_cache_modified': False, 'training_started': False})


if __name__ == '__main__':
    main()
