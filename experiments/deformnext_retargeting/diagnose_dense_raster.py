"""Isolate native CUDA rasterization of an archived, unmodified cascade mesh.

The prefix arm is an engineering diagnostic only. Its pixels are never used
for a model comparison. Every mode runs in a fresh process because a CUDA
illegal-address error poisons that process's context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if gpu not in {'0', '1', '2', '3'} or gpu not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(','):
        raise ValueError('one explicitly assigned GPU0-3 is required')
    if plan['mode'] not in {'raw-prefix', 'raw-full', 'native-simplify'}:
        raise ValueError('explicit engineering diagnostic mode required')
    source = Path(plan['prediction'])
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8388608), b''):
            digest.update(chunk)
    if digest.hexdigest() != plan['prediction_SHA256']:
        raise ValueError('archived prediction bytes differ')
    out = Path(plan['output'])
    out.mkdir(parents=True, exist_ok=False)

    def record(name, values):
        (out / name).write_text(json.dumps(values, indent=2) + '\n')

    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.8, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    with np.load(source) as z:
        vertices = z['vertices'].copy()
        faces = z['faces'].copy()
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.dtype != np.float32
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype != np.int32
            or not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('finite native vertices and in-range native int32 triangles required')
    before = dict(vertices=len(vertices), faces=len(faces), min=vertices.min(0).tolist(), max=vertices.max(0).tolist())
    scale = float(plan['row']['mesh_scale'])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('positive original mesh_scale required')
    vertices = torch.from_numpy(vertices).cuda()
    faces = torch.from_numpy(faces).cuda()
    started = time.monotonic()
    if plan['mode'] == 'raw-prefix':
        faces = faces[:1000000].contiguous()
    elif plan['mode'] == 'native-simplify':
        from pixal3d.representations import Mesh
        mesh = Mesh(vertices, faces)
        mesh.simplify(target=1000000)
        vertices, faces = mesh.vertices, mesh.faces
        torch.cuda.synchronize()
        np.savez_compressed(out / 'render-geometry.npz',
            vertices=vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy())
    after = dict(vertices=len(vertices), faces=len(faces), min=vertices.min(0).values.tolist(), max=vertices.max(0).values.tolist())
    record('geometry-audit.json', dict(mode=plan['mode'], before=before, after=after,
        scale=scale, geometry_frame='original decoder-normalized coordinates',
        original_prediction_SHA256=plan['prediction_SHA256'],
        original_prediction_unchanged=True, per_output_alignment=False,
        raw_mesh_geometry_metrics_unchanged=True, prefix_is_not_a_quality_evaluation=True,
        simplification_seconds=time.monotonic()-started))
    import nvdiffrast.torch as dr
    import utils3d
    from pixal3d.renderers.pbr_mesh_renderer import intrinsics_to_projection
    with np.load(plan['geometry_evaluation']) as z:
        rotation = torch.eye(4, device='cuda')
        rotation[:3, :3] = torch.from_numpy(z['anchor_rotation']).cuda()
    i = plan['view_index']
    c2w = torch.tensor(plan['row']['transform_matrix'][i], device='cuda', dtype=torch.float32)
    flip = torch.diag(torch.tensor([1., -1., -1., 1.], device='cuda'))
    ext = flip @ torch.linalg.inv(c2w) @ torch.linalg.inv(rotation)
    fov = torch.tensor(plan['row']['camera_angle_x'][i], device='cuda')
    intr = utils3d.torch.intrinsics_from_fov(fov_x=fov, fov_y=fov)
    projection = intrinsics_to_projection(intr, .01, 100)
    vertices = vertices / scale
    homogeneous = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=-1)[None]
    clip = torch.bmm(homogeneous, (projection @ ext)[None].transpose(-1, -2))
    if not torch.isfinite(clip).all():
        raise ValueError('actual native clip positions must all be finite')
    record('before-raster.json', dict(vertices=len(vertices), faces=len(faces),
        clip_shape=list(clip.shape), clip_finite=True, face_dtype=str(faces.dtype),
        allocated_GiB=torch.cuda.memory_allocated()/2**30, phase='native DepthPeeler four layers'))
    context = dr.RasterizeCudaContext(device='cuda')
    layers = []
    with torch.inference_mode(), dr.DepthPeeler(context, clip, faces, (512, 512)) as peeler:
        for layer in range(4):
            rast, _ = peeler.rasterize_next_layer()
            torch.cuda.synchronize()
            layers.append(dict(layer=layer, foreground=int((rast[..., 3] > 0).sum()),
                finite=bool(torch.isfinite(rast).all()), max_triangle_id=float(rast[..., 3].max())))
            record('layers.json', layers)
    report = dict(status='PASS_ACTUAL_NATIVE_RASTER_DIAGNOSTIC', mode=plan['mode'],
        geometry_before=before, geometry_after=after, layers=layers,
        seconds=time.monotonic()-started, peak_GiB=torch.cuda.max_memory_allocated()/2**30,
        scientific_training_updates=0, model_quality_gain_claimed=False)
    record('result.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
