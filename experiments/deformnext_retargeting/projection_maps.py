"""Anchor-frame addresses matching dev-jaehyeok's native MV projection."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np

SCHEMA = 'deformnext-anchor-projection-v2'


def anchor_patch_indices(row, coords, resolution):
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('coords must be [N,3]')
    if np.any((coords < 0) | (coords >= resolution)):
        raise ValueError('coordinate outside projection grid')
    # Native ProjGrid linspace, rotation and scale, in the latent anchor frame.
    points = (-1 + 2 * coords.astype(np.float32) / (resolution - 1))
    points = points[:, [0, 2, 1]] * np.array([1, -1, 1], np.float32)
    points = points / float(row['mesh_scale']) / 2
    radius = np.linalg.norm(np.asarray(row['transform_matrix'][0], np.float32)[:3, 3])
    # Native canonical c2w = [[1,0,0,0],[0,0,-1,-r],[0,1,0,0],[0,0,0,1]].
    camera_x, camera_y, depth = points[:, 0], points[:, 2], points[:, 1] + radius
    focal = 256 / np.tan(float(row['camera_angle_x'][0]) / 2)
    xy = np.stack((256 + focal * camera_x / depth,
                   256 - focal * camera_y / depth), -1)
    patch = np.floor(xy / 16).astype(np.int64).clip(0, 31)
    return patch[:, 1] * 32 + patch[:, 0]


def remap(value, mappings):
    for source, dest in mappings:
        if value == source or value.startswith(source.rstrip('/') + '/'):
            return Path(dest + value[len(source):])
    return Path(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--path-map', action='append', default=[])
    args = parser.parse_args()
    mappings = [tuple(x.split('=', 1)) for x in args.path_map]
    args.output_root.mkdir(parents=True, exist_ok=True)
    grid = np.indices((16, 16, 16)).reshape(3, -1).T
    count = 0
    for line in args.manifest.read_text().splitlines():
        row = json.loads(line)
        path = args.output_root / (row['sample_id'] + '.npz')
        with np.load(remap(row['shape_latent'], mappings)) as archive:
            coords = archive['coords']
        if coords.shape[1] == 4:
            if np.any(coords[:, 0] != 0):
                raise ValueError('stored batch column must be zero')
            coords = coords[:, 1:]
        temporary = path.with_name(f'.{path.name}.{os.getpid()}.npz')
        np.savez(temporary, schema=np.asarray(SCHEMA),
                 patch_grid=anchor_patch_indices(row, grid, 16).reshape(16, 16, 16),
                 patch_ids=anchor_patch_indices(row, coords, 64))
        os.replace(temporary, path)
        count += 1
    print(json.dumps({'schema': SCHEMA, 'samples': count}), flush=True)


if __name__ == '__main__':
    main()
