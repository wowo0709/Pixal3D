"""Engineering-only sparse ray/projection check on a declared real PBR asset.

No denoiser, image encoder, learned conditioner, optimizer, or scientific
training is constructed. Passing this check does not admit PBR training.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .cascade_view_roles import ClosedInputs, file_sha256
from pixal3d.modules.pbr_source_ray_geometry import (
    pool_sparse_source_ray_queries, project_sparse_pbr_source_delta,
    sparse_source_ray_membership,
)
from pixal3d.modules.ray_group_query import native_ray_membership
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    ProjGrid, compute_multiview_projection_matrices,
)


def validate_declared_input(row, latent, transport):
    if (row['split'] != 'train' or row['target_anchor_kind'] != 'rgb'
            or row['conditioning_camera_ids'] != [8, 9, 10, 11]
            or str(latent['schema']) != row['latent_geometry_schema']
            or latent['coords'].shape != (row['pbr_tokens'], 3)
            or not 0 < row['pbr_tokens'] <= 32768
            or float(latent['mesh_scale']) != row['mesh_scale']
            or str(transport['schema']) != 'DEFORMNEXT_PIXAL_VALUES_PBR1024_V1'
            or transport['source_values'].shape != (3, 1024, 2048)
            or transport['target_query'].shape != (1024, 1024)
            or transport['patch_ids'].shape != (row['pbr_tokens'],)
            or str(transport['conditioning_input_sha256']) != row['conditioning_input_sha256']
            or not np.isfinite(transport['source_values']).all()
            or not np.isfinite(transport['target_query']).all()
            or transport['patch_ids'].dtype != np.int64
            or ((transport['patch_ids'] < 0) | (transport['patch_ids'] >= 1024)).any()):
        raise ValueError('actual aligned TRAIN PBR support, RGB identity and DINO+NAF cache required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (visible not in {'0', '1', '2', '3'}
            or visible not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(',')):
        raise ValueError('one exclusively assigned physical GPU0-3 required')
    plan = json.loads(args.plan.read_text())
    check = ClosedInputs(plan['canonical_local_root'])

    def guard():
        check.check(args.plan, args.plan.parent / 'file4-closure.json')
        for path, digest in {**plan['source_SHA256'], **plan['input_SHA256']}.items():
            if file_sha256(path) != digest:
                raise ValueError('declared input/source changed: ' + path)
        for path, receipt in plan['closed_inputs'].items():
            check.check(path, receipt)

    guard()
    row = json.loads(Path(plan['row_snapshot']).read_text())
    with np.load(plan['latent_snapshot'], allow_pickle=False) as latent, np.load(
            plan['transport_snapshot'], allow_pickle=False) as transport:
        validate_declared_input(row, latent, transport)
        coords = torch.from_numpy(latent['coords'].astype(np.int32)).cuda()
        source = torch.from_numpy(transport['source_values'].copy()).float().cuda()[None]
        target = torch.from_numpy(transport['target_query'].copy()).float().cuda()[None]
        patches = torch.from_numpy(transport['patch_ids'].copy()).long().cuda()
    if row['sample_id'] != plan['sample_id']:
        raise ValueError('predeclared real sample differs')
    coords = torch.cat((torch.zeros_like(coords[:, :1]), coords), 1)
    transform = torch.tensor(row['transform_matrix'], dtype=torch.float32, device='cuda')[None]
    camera = dict(transform_matrix=transform,
        distance=torch.linalg.vector_norm(transform[:, :, :3, 3], dim=-1),
        camera_angle_x=torch.tensor(row['camera_angle_x'], dtype=torch.float32, device='cuda')[None],
        mesh_scale=torch.tensor([row['mesh_scale']], dtype=torch.float32, device='cuda'))
    grid = ProjGrid(64, 1024).cuda()
    flat = ((coords[:, 1].long() * 64 + coords[:, 2]) * 64 + coords[:, 3]).long()
    matrices, _ = compute_multiview_projection_matrices(transform, camera['distance'],
        grid.front_view_transform_matrix)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        membership = sparse_source_ray_membership(grid, coords, camera)
        dense_membership = native_ray_membership(grid, **camera, views=(1, 2, 3))
        if not torch.equal(membership, dense_membership[0, :, flat].transpose(0, 1)):
            raise ValueError('real native64 dense and packed source-ray addresses differ')
        pooled = pool_sparse_source_ray_queries(target[0, patches], membership, coords[:, 0], batch_size=1)
        if not (pooled.voxel_count.sum((0, 2)) == (membership >= 0).sum(0)).all():
            raise ValueError('real packed source-ray member count differs')
        zero = torch.zeros_like(source[..., :1])
        actual = project_sparse_pbr_source_delta(grid, source, zero, source, coords, camera)
        expected = None
        for source_index in range(3):
            view = source_index + 1
            dense = grid(source[:, source_index].reshape(1, 32, 32, 2048),
                camera['camera_angle_x'][:, view], camera['distance'][:, view],
                camera['mesh_scale'], matrices[:, view]).float()
            selected = dense[0, flat].clone()
            del dense
            expected = selected if expected is None else expected + selected
        expected = expected / 4
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)
        error = (actual - expected).abs()
        max_error = [float(x.max()) for x in error.split(1024, dim=-1)]
        rms = [float(x.square().mean().sqrt()) for x in actual.split(1024, dim=-1)]
        if not all(x > 0 for x in rms):
            raise ValueError('actual DINO and NAF delivery must both be exercised')
        mass = torch.full_like(zero, .125)
        untouched = project_sparse_pbr_source_delta(grid, mass * source, mass, source, coords, camera)
        if untouched.count_nonzero():
            raise ValueError('same-source replacement must be exactly native-zero')
    guard()
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='PASS_ACTUAL_PBR64_SPARSE_SOURCE_RAY_GEOMETRY',
        sample_id=row['sample_id'], split='train', pbr_tokens=len(coords),
        physical_GPU=visible, native_grid_resolution=64, input_image_resolution=1024,
        source_ray_resolution=32, value_channels=dict(DINO=1024, NAF=1024),
        source_member_addresses_exact_to_native_dense=True,
        actual_pooled_members_per_source=pooled.voxel_count.sum((0, 2)).tolist(),
        actual_DINO_NAF_max_absolute_error=max_error, actual_DINO_NAF_output_RMS=rms,
        projection_tolerance=dict(atol=2e-5, rtol=1e-5),
        same_source_zero_delta_exact=True, visibility_write_gate=False,
        peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
        optimizer_updates=0, neural_models_constructed=0,
        learned_R_or_attention_forward_proven=False,
        original_PBR_LoRA_initial_prediction_proven=False,
        scientific_microbatch_capacity_proven=False,
        PBR_training_admitted=False, quality_gain_proven=False,
        full_goal_complete=False, inputs=check.inputs)
    (output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
