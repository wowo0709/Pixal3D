"""Measure target-patch reuse and native projection variation on fixed cases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    ProjGrid, compute_multiview_projection_matrices,
)


def fingerprint(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


def within_patch_variation(features, ids):
    """Variability among actual voxel features sharing a cached target patch."""
    if features.ndim != 2 or ids.shape != features.shape[:1] or not len(ids):
        raise ValueError('nonempty actual voxel groups required')
    count = torch.bincount(ids, minlength=1024)
    sums = features.new_zeros(1024, features.shape[1]).index_add_(0, ids, features)
    means = sums / count[:, None].clamp_min(1)
    error = float((features - means[ids]).square().sum())
    energy = float(features.square().sum())
    return {'squared_within_patch_error': error, 'squared_feature_energy': energy,
            'within_patch_relative_RMS': (error / max(energy, 1e-30)) ** .5}


def run(root):
    torch.set_grad_enabled(False)
    torch.set_num_threads(2)
    output = root / 'semantic-patch-voxel-reuse-first8-v1'
    if output.exists():
        raise FileExistsError('preserve actual structural diagnosis')
    reference_path = root / 'matching-address-first8-v1/summary.json'
    reference = json.loads(reference_path.read_text())
    cfg_path = root / 'configs/task-full.json'
    cfg = json.loads(cfg_path.read_text())
    if cfg['trainer']['args']['image_cond_model']['args']['image_size'] != 512:
        raise ValueError('declared512 RGB canvas required')
    rows = {row['sample_id']: row for row in map(json.loads, open(reference['manifest']['path']))}
    projector = ProjGrid(grid_resolution=16, image_resolution=512).eval()
    cases = []
    for case in reference['cases']:
        row = rows[case['sample_id']]
        if row['conditioning_camera_ids'] != [8, 9, 10, 11]:
            raise ValueError('declared camera order differs')
        path = Path(cfg['validation']['dataset_args']['projection_root']) / (row['sample_id'] + '.npz')
        with np.load(path) as archive:
            if str(archive['schema']) != 'deformnext-anchor-projection-v2':
                raise ValueError('projection coordinate schema differs')
            grid = torch.from_numpy(archive['patch_grid'].copy()).long().flatten()
        if len(grid) != 4096 or grid.min() < 0 or grid.max() >= 1024:
            raise ValueError('declared SS16 and32patch grids required')
        with Image.open(case['inputs'][2]['path']) as image:
            if image.size != (512, 512):
                raise ValueError('prepared512 canvas required')
            alpha = np.asarray(image.convert('RGBA'))[8::16, 8::16, 3].reshape(-1) > 127
        keep = torch.from_numpy(alpha.copy())[grid]
        ids = grid[keep]
        counts = torch.bincount(ids, minlength=1024)
        if int((counts > 0).sum()) != case['active_queries']:
            raise ValueError('actual foreground-support count differs from address inspection')
        with np.load(case['inputs'][0]['path']) as archive:
            if str(archive['conditioning_input_sha256']) != row['conditioning_input_sha256']:
                raise ValueError('RGB transport identity differs')
            target = torch.from_numpy(archive['target_query'].copy()).float()
            sources = torch.from_numpy(archive['source_values'].copy()).float()
        if target.shape != (1024, 1024) or sources.shape != (3, 1024, 1024):
            raise ValueError('native cached DINO map shape differs')
        transforms = torch.tensor(row['transform_matrix'], dtype=torch.float32)[None]
        distances = transforms[..., :3, 3].norm(dim=-1)
        angles = torch.tensor(row['camera_angle_x'], dtype=torch.float32)[None]
        scale = torch.tensor([row['mesh_scale']], dtype=torch.float32)
        projection, _ = compute_multiview_projection_matrices(
            transforms, distances, projector.front_view_transform_matrix)
        projected = []
        for view, patches in enumerate([target, *sources]):
            projected.append(projector(patches.reshape(1, 32, 32, 1024), angles[:, view],
                                       distances[:, view], scale, projection[:, view])[0])
        native = torch.stack(projected).mean(0)
        source_mean = torch.stack(projected[1:]).mean(0)
        constant_per_patch = target[grid]
        constant_check = within_patch_variation(constant_per_patch[keep], ids)
        # Repeated float32 accumulation may produce a tiny nonzero residual.
        if constant_check['within_patch_relative_RMS'] > 2e-6:
            raise ValueError('group statistic does not recover repeated patch features')
        record = {
            'sample_id': row['sample_id'], 'category': row['category'],
            'all_SS_voxels': 4096, 'voxels_on_foreground_target_patches': int(keep.sum()),
            'foreground_patches_used': int((counts > 0).sum()),
            'mean_voxels_per_used_patch': float(counts.sum() / (counts > 0).sum()),
            'max_voxels_per_used_patch': int(counts.max()),
            'native_four_view_context': within_patch_variation(native[keep], ids),
            'native_three_source_context': within_patch_variation(source_mean[keep], ids),
            'native_target_only_context': within_patch_variation(projected[0][keep], ids),
            'constant_patch_group_reference': constant_check,
            'projection_cache': fingerprint(path),
            'RGB_and_transport_inputs': case['inputs'],
        }
        cases.append(record)
        print(json.dumps({'category': row['category'], 'mean_voxels_per_patch': record['mean_voxels_per_used_patch'],
                          'native_context_within_patch_relative_RMS': record['native_four_view_context']['within_patch_relative_RMS']}), flush=True)
    groups = ['native_four_view_context', 'native_three_source_context', 'native_target_only_context']
    total_voxels = sum(case['voxels_on_foreground_target_patches'] for case in cases)
    total_patches = sum(case['foreground_patches_used'] for case in cases)
    report = {'status': 'PASS_ACTUAL_PATCH_REUSE_AND_CPU_NATIVE_PROJECTION_VARIATION',
              'cases': cases, 'foreground_voxels': total_voxels, 'foreground_patches': total_patches,
              'voxel_to_patch_reuse_ratio': total_voxels / total_patches,
              'pooled_within_patch_relative_RMS': {group: (
                  sum(case[group]['squared_within_patch_error'] for case in cases) /
                  sum(case[group]['squared_feature_energy'] for case in cases)) ** .5 for group in groups},
              'reference': fingerprint(reference_path), 'recipe': fingerprint(cfg_path),
              'driver': fingerprint(Path(__file__)), 'device': 'CPU_FP32', 'optimizer_updates': 0,
              'scope': 'First8 fixed cases. Native ProjGrid applied to cached RGB DINO maps using declared cameras/scale. Multiple voxel centers sharing a nearest target-patch bin are not necessarily the identical physical ray. This measures available voxel-dependent feature variation,not matching accuracy,SS quality,a causal failure mechanism,or a GPU/BF16 numerical replay.',
              'possible_unverified_followup': 'Use native per-voxel context as utility query while preserving frozen semantic64 priors/values. Such a model has not been implemented or trained by this inspection.',
              'model_adopted': False, 'full_goal_complete': False}
    output.mkdir()
    with (output / 'summary.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'status': report['status'], 'reuse_ratio': report['voxel_to_patch_reuse_ratio'],
                      'native_context_variation': report['pooled_within_patch_relative_RMS']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)


if __name__ == '__main__':
    main()
