"""Version M1 latent paths and retain the inverse Blender dump transform."""
from __future__ import annotations
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from data_toolkit.utils import sphere_normalize_torch, transform_mesh
from experiments.deformnext_retargeting.target_frame import frame_metadata, SCHEMA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, action='append', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--published-root', required=True)
    parser.add_argument('--path-map', action='append', default=[])
    args = parser.parse_args()
    torch.set_num_threads(1)
    mappings = [tuple(value.split('=', 1)) for value in args.path_map]
    def resolve(value):
        for source, dest in mappings:
            if value.startswith(source + '/'):
                return Path(dest + value[len(source):])
        return Path(value)
    def convert(row):
        path = resolve(row['target_vertices'])
        if path.suffix == '.npy':
            vertices = np.load(path)
        else:
            with np.load(path) as archive:
                vertices = archive['V'].copy()
        row.update(frame_metadata(vertices))
        row.pop('pbr_tokens', None)
        normalized, _, radius = sphere_normalize_torch(torch.from_numpy(vertices).float())
        transformed = transform_mesh(normalized, {'transform_matrix': row['transform_matrix'][0]})
        row['mesh_scale_previous'] = row['mesh_scale']
        row['mesh_scale'] = .49999 / transformed.abs().max().item() / radius.item()
        for field, folder in [('shape_latent', 'shape1024_latents_world_v4'), ('pbr_latent', 'pbr_latents_world_v4')]:
            row[field + '_previous'] = row[field]
            row[field] = str(Path(args.published_root) / folder / (row['sample_id'] + '.npz'))
        return row
    args.output_root.mkdir(parents=True, exist_ok=True)
    for path in args.manifest:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        with ThreadPoolExecutor(max_workers=4) as pool:
            converted = list(pool.map(convert, rows))
        destination = args.output_root / path.name
        temp = destination.with_name(f'.{destination.name}.{os.getpid()}')
        temp.write_text(''.join(json.dumps(row) + '\n' for row in converted))
        os.replace(temp, destination)
        print(json.dumps({'split': path.stem, 'count': len(converted), 'schema': SCHEMA}), flush=True)


if __name__ == '__main__':
    main()
