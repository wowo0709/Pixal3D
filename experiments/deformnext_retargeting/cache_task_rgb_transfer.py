"""Build current-data controlled RGB values without recoloring matcher inputs.

The original surface-cue generator is applied to M0 vertices. Pixel barycentrics
of M0/M1 then produce foreground-normalized patch values. Fixed RAW proposals
and epoch7 descriptors are copied unchanged; geometry/cues supervise loss only.
CPU ray casting is bounded to two cores while the four assigned GPUs work.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
import torch

from .historical_geometry import camera_rays_for_pixels
from .historical_surface_cue import procedural_surface_cue
from .reproduce_q2c import read, reference, write_new
from .topology_labels import SCHEMA as TEACHER_SCHEMA, fingerprint, mask_agreement

SCHEMA = 'deformnext-controlled-patch-RGB-task-v2'


def stable_seed(sample_id, role, cue):
    return int.from_bytes(hashlib.sha256(f'patchRGB-v1|{sample_id}|{role}|{cue}'.encode()).digest()[:8], 'big')


def choose(rows, count, role):
    groups = defaultdict(list)
    for row in rows:
        groups[row['category']].append(row)
    for values in groups.values():
        values.sort(key=lambda r: hashlib.sha256(f'{role}|{r["sample_id"]}'.encode()).hexdigest())
    chosen, used = [], set()
    while len(chosen) < count:
        added = False
        for category in sorted(groups):
            candidates = groups[category]
            candidates[:] = [r for r in candidates if r['source_cluster_id'] not in used]
            if candidates and len(chosen) < count:
                row = candidates.pop(0)
                chosen.append(row)
                used.add(row['source_cluster_id'])
                added = True
        if not added:
            raise ValueError('insufficient distinct source clusters for frozen pilot')
    return chosen


def freeze(panels, output, train_count=256, calibration_count=64):
    read_rows = lambda name: [json.loads(line) for line in (panels / (name + '.jsonl')).read_text().splitlines() if line]
    train, calibration, validation = [read_rows(name) for name in ['head_train', 'head_calibration', 'head_validation']]
    for a, b in [(train, calibration), (train, validation), (calibration, validation)]:
        if {r['source_cluster_id'] for r in a} & {r['source_cluster_id'] for r in b}:
            raise ValueError('upstream head train/calibration/validation clusters overlap')
    selected = {'train': choose(train, train_count, 'train'),
                'calibration': choose(calibration, calibration_count, 'calibration')}
    output.mkdir(parents=True, exist_ok=True)
    for role, rows in selected.items():
        with (output / (role + '.jsonl')).open('x') as stream:
            stream.write(''.join(json.dumps(row) + '\n' for row in rows))
    write_new(output / 'population.json', {'status': 'PASS', 'schema': SCHEMA,
        'train_assets': train_count, 'calibration_assets': calibration_count,
        'source_clusters_disjoint': True, 'one_pair_per_source_cluster': True,
        'selection': 'category round-robin, deterministic role/sample hash; no metric selection',
        'validation_508_used_for_training_or_task_evaluation': False,
        'upstream': {name: reference(panels / (name + '.jsonl'))
                     for name in ['head_train', 'head_calibration', 'head_validation']}})


def patch_operator(face_index, barycentric, valid, alpha, faces, vertex_count, image_size=512, patch_size=16):
    """Pixel interpolation/pooling as a sparse matrix; no nearest-patch labels."""
    if (image_size % patch_size or face_index.shape != (image_size * image_size,)
            or barycentric.shape != (len(face_index), 3) or alpha.shape != (image_size, image_size)
            or valid.shape != face_index.shape or not np.isfinite(alpha).all()
            or np.any((alpha < 0) | (alpha > 1))):
        raise ValueError('raster/alpha/patch shape or range differs')
    keep = valid & (alpha.reshape(-1) > 0)
    ids = np.flatnonzero(keep)
    selected = face_index[ids]
    weights = barycentric[ids]
    if (np.any((selected < 0) | (selected >= len(faces))) or not np.isfinite(weights).all()
            or not np.allclose(weights.sum(-1), 1., atol=1e-5)):
        raise ValueError('valid pixel face/barycentric authority differs')
    grid = image_size // patch_size
    patch = (ids // image_size // patch_size) * grid + (ids % image_size // patch_size)
    factor = alpha.reshape(-1)[ids].astype(np.float64)
    mass = np.bincount(patch, weights=factor, minlength=grid * grid)
    # Foreground means remove a dark-background/patch-area shortcut. The three
    # barycentric weights retain native interpolation across palette boundaries.
    normalized = weights.astype(np.float64) * (factor / np.maximum(mass[patch], 1e-12))[:, None]
    operator = coo_matrix((normalized.reshape(-1),
        (np.repeat(patch, 3), faces[selected].reshape(-1))), shape=(grid * grid, vertex_count)).tocsr()
    return operator, (mass / (patch_size * patch_size)).astype(np.float32)


def surface_scene(vertices, faces):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(vertices, np.float32)),
        o3d.core.Tensor(np.asarray(faces, np.uint32))))
    return scene


def raster_operator(scene, row, view, faces, vertex_count, image_path):
    import open3d as o3d
    yy, xx = np.meshgrid(np.arange(512), np.arange(512), indexing='ij')
    pixels = np.stack((xx, yy), -1).reshape(-1, 2).astype(np.float32)
    rays = camera_rays_for_pixels(pixels, np.asarray(row['transform_matrix'][view]),
                                  row['camera_angle_x'][view], resolution=512)
    hit = scene.cast_rays(o3d.core.Tensor(rays), nthreads=2)
    face = hit['primitive_ids'].numpy().astype(np.int64)
    uv = hit['primitive_uvs'].numpy()
    valid = np.isfinite(hit['t_hit'].numpy()) & (face < len(faces))
    bary = np.stack((1. - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]), -1)
    with Image.open(image_path) as image:
        if image.size != (512, 512):
            raise ValueError('controlled patch task requires the actual512px RGB input')
        alpha = np.asarray(image.convert('RGBA'), np.float32)[..., 3] / 255.
    iou, containment = mask_agreement(torch.from_numpy(valid.reshape(512, 512)),
        torch.from_numpy(alpha > .5), alpha_cutouts=True)
    operator, mass = patch_operator(face, bary, valid, alpha, faces, vertex_count)
    observed_mass = alpha.reshape(32, 16, 32, 16).mean((1, 3)).reshape(-1)
    return operator, mass, observed_mass, {'IoU': iou, 'containment': containment}


def build(row, args):
    resolve = lambda path: args.file4_mirror / path.removeprefix('/file4/') if path.startswith('/file4/') else Path(path)
    teacher_path = args.labels / (row['sample_id'] + '.npz')
    digest = fingerprint(row, resolve)
    with np.load(teacher_path, allow_pickle=False) as teacher:
        if (str(teacher['schema']) != TEACHER_SCHEMA or str(teacher['input_sha256']) != digest
                or str(teacher['sample_id']) != row['sample_id'] or str(teacher['split']) != row['split']):
            raise ValueError('cached topology/image certification differs')
        observable = teacher['positive_visible'].any(-1)
        positives = teacher['positive_indices'].copy()
        positive_visible = teacher['positive_visible'].copy()
    with np.load(resolve(row['source_geometry']), allow_pickle=False) as geometry:
        original, faces = geometry['V'].copy(), geometry['F'].copy()
    target = np.load(resolve(row['target_vertices']), allow_pickle=False)
    if original.shape != target.shape or row['latent_geometry_schema'] != 'deformnext-m1-render-world-v4':
        raise ValueError('M0/M1 topology or render-world frame differs')
    scenes = [surface_scene(target, faces), surface_scene(original, faces)]
    operators, masses, observed, agreement = [], [], [], []
    for view, image_path in enumerate([row['target_anchor'], *row['source_images']]):
        operator, mass, alpha, check = raster_operator(scenes[0 if view == 0 else 1], row,
            view, faces, len(original), resolve(image_path))
        operators.append(operator); masses.append(mass); observed.append(alpha); agreement.append(check)
    cache_path = resolve(row['transport_cache'])
    with np.load(cache_path, allow_pickle=False) as cache:
        if (str(cache['conditioning_input_sha256']) != row['conditioning_input_sha256']
                or int(cache['omega_completed_updates']) != 23597):
            raise ValueError('RGB or frozen Omega epoch7 fingerprint differs')
        source_valid = cache['source_valid'].copy()
        if not np.array_equal(source_valid, np.stack(observed[1:]) > .05):
            raise ValueError('native alpha-based source validity changed')
        query = cache['target_keys'][:, :1024].copy()
        source = cache['source_keys'][..., :1024].reshape(3072, 1024).copy()
    proposal_path = args.proposals / (row['sample_id'] + '.npz')
    with np.load(proposal_path, allow_pickle=False) as proposal:
        if str(proposal['input_sha256']) != row['conditioning_input_sha256']:
            raise ValueError('fixed RAW proposal fingerprint differs')
        candidates = proposal['indices'][:, :64].copy().astype(np.int64)
        raw = proposal['cosine'][:, :64].copy().astype(np.float32)
        valid_candidates = proposal['valid'][:, :64].copy()
    eligible = observable & (masses[0] > .05) & (observed[0] > .05) & valid_candidates.all(-1)
    if candidates.shape != (1024, 64) or np.any(np.diff(np.sort(candidates, -1), axis=-1) == 0):
        raise ValueError('original fixed64 distinct proposal contract differs')
    arrays = {}
    for role in ['label', 'evaluation']:
        colors = np.stack([procedural_surface_cue(original, active_vertices=np.ones(len(original), bool),
                         seed=stable_seed(row['sample_id'], role, cue)) for cue in range(16)])
        vertex_matrix = colors.transpose(1, 0, 2).reshape(len(original), 48)
        values = [np.clip(op @ vertex_matrix, 0, 1).reshape(1024, 16, 3).transpose(1, 0, 2).astype(np.float16)
                  for op in operators]
        arrays[role + '_target_rgb'] = values[0]
        arrays[role + '_source_rgb'] = np.concatenate(values[1:], axis=1)
    positive = ((candidates[..., None] == positives[:, None]) & positive_visible[:, None]).any(-1)
    record = {'sample_id': row['sample_id'], 'source_cluster_id': row['source_cluster_id'],
              'original_split': row['split'], 'eligible_queries': int(eligible.sum()),
              'foreground_queries': int((observed[0] > .05).sum()),
              'observable_queries': int(observable.sum()), 'fixed64_exact_patch_hits': int((positive.any(-1) & eligible).sum()),
              'source_usable_tokens': int(source_valid.sum()), 'raster_agreement': agreement,
              'conditioning_input_sha256': row['conditioning_input_sha256'], 'geometry_input_sha256': digest,
              'descriptor_cache': reference(cache_path), 'fixed_proposals': reference(proposal_path)}
    return {**arrays, 'query_L12': query, 'source_L12': source, 'candidates': candidates,
            'raw_scores': raw, 'eligible_queries': eligible, 'source_patch_mass': np.concatenate(masses[1:]),
            'target_patch_mass': masses[0]}, record


def materialize(args):
    began = time.monotonic()
    roles = ['train', 'calibration']
    selected = [(role, json.loads(line)) for role in roles
                for line in (args.population / (role + '.jsonl')).read_text().splitlines() if line]
    selected = selected[:args.limit] if args.limit else selected
    if read(args.population / 'population.json')['schema'] != SCHEMA:
        raise ValueError('frozen pilot/schema differs')
    records = []
    (args.output / 'data').mkdir(parents=True, exist_ok=True)
    (args.output / 'evaluation').mkdir(exist_ok=True)
    for role, row in selected:
        path = args.output / 'data' / (row['sample_id'] + '.npz')
        evaluation_path = args.output / 'evaluation' / path.name
        if path.exists() and evaluation_path.exists():
            with np.load(path, allow_pickle=False) as previous:
                record = json.loads(str(previous['metadata']))
                if (str(previous['schema']) != SCHEMA or str(previous['role']) != role
                        or record['conditioning_input_sha256'] != row['conditioning_input_sha256']
                        or record['geometry_input_sha256'] != fingerprint(row, lambda p:
                            args.file4_mirror / p.removeprefix('/file4/') if p.startswith('/file4/') else Path(p))):
                    raise ValueError('immutable task cache changed')
            records.append({**record, 'role': role, 'artifact': reference(path),
                            'evaluation_artifact': reference(evaluation_path)})
            print(json.dumps({'reused': len(records), 'total': len(selected)}), flush=True)
            continue
        if path.exists() or evaluation_path.exists():
            raise ValueError('partially published input pair; preserve and use a fresh output namespace')
        arrays, record = build(row, args)
        # Separate files keep the evaluator cue bank outside every training
        # record. Only the matching descriptor/proposal file is shared.
        for dest, values in [(path, {k: v for k, v in arrays.items() if not k.startswith('evaluation_')}),
                             (evaluation_path, {k: v for k, v in arrays.items() if k.startswith('evaluation_')})]:
            temporary = dest.with_name('.' + dest.name + f'.{os.getpid()}.tmp.npz')
            np.savez(temporary, schema=SCHEMA, sample_id=row['sample_id'], role=role,
                     supervision_only=True, metadata=json.dumps(record, sort_keys=True), **values)
            os.replace(temporary, dest)
        records.append({**record, 'role': role, 'artifact': reference(path),
                        'evaluation_artifact': reference(evaluation_path)})
        print(json.dumps({'completed': len(records), 'total': len(selected),
                          'seconds': time.monotonic()-began, **record}), flush=True)
    write_new(args.output / 'summary.json', {'status': 'PASS', 'schema': SCHEMA, 'records': records,
              'population': reference(args.population / 'population.json'), 'seconds': time.monotonic()-began,
              'full_population': args.limit is None, 'supervision_only': True,
              'matcher_inputs': 'unchanged epoch7 FRAME1024 and unchanged original global RAW64 candidates',
              'value_task': '16 label/16 disjoint evaluator source-space surface cues, barycentric foreground patch means',
              'validation508_opened': False, 'production_promotion': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'cache'])
    parser.add_argument('--panels', type=Path)
    parser.add_argument('--population', type=Path)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--proposals', type=Path)
    parser.add_argument('--file4-mirror', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    torch.set_num_threads(2)
    if args.mode == 'freeze':
        freeze(args.panels, args.output)
    else:
        if any(getattr(args, key) is None for key in ['population', 'labels', 'proposals', 'file4_mirror']):
            parser.error('cache requires population/labels/proposals/file4-mirror')
        materialize(args)


if __name__ == '__main__':
    main()
