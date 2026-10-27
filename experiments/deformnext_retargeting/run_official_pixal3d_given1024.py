"""Generate official Pixal3D 1024 TEST predictions with a GT-free fixed scale.

Each worker loads the official model once. No target geometry, target depth,
evaluation image, or target-derived mesh scale is read during generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_official_cache(native_setup: Path, cache: Path) -> None:
    runtime_root = native_setup.parent
    for name in ('TencentARC/Pixal3D',
                 'camenduru/dinov3-vitl16-pretrain-lvd1689m',
                 'briaai/RMBG-2.0'):
        record = json.loads((native_setup / 'manifests' /
                             ('checkpoint-' + name.replace('/', '--') + '.json')).read_text())
        if record['status'] != 'complete':
            raise ValueError('pinned checkpoint incomplete: ' + name)
        relative = Path(record['local_dir']).relative_to('/root/dev/editing_reproduction')
        actual = runtime_root / relative
        if not actual.is_dir():
            raise FileNotFoundError(actual)
        model_dir = cache / ('models--' + name.replace('/', '--'))
        refs, snapshots = model_dir / 'refs', model_dir / 'snapshots'
        refs.mkdir(parents=True, exist_ok=True)
        snapshots.mkdir(exist_ok=True)
        ref = refs / 'main'
        if ref.exists() and ref.read_text().strip() != record['revision']:
            raise ValueError('pinned revision changed: ' + name)
        ref.write_text(record['revision'])
        snapshot = snapshots / record['revision']
        if not snapshot.exists():
            snapshot.symlink_to(actual, target_is_directory=True)
        elif snapshot.resolve() != actual.resolve():
            raise ValueError('snapshot source mismatch: ' + name)


def load_pipeline(args: argparse.Namespace):
    cache = args.output_root / f'hf-cache-rank{args.rank}'
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.physical_gpu),
                      HF_HUB_CACHE=str(cache),
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                      ATTN_BACKEND=args.attention_backend, CUDA_HOME='/usr/local/cuda-12.8',
                      TORCH_CUDA_ARCH_LIST='12.0', OPENCV_IO_ENABLE_OPENEXR='1')
    prepare_official_cache(args.native_setup, cache)
    sys.path.insert(0, str(args.pixal3d_repo))
    import torch
    if args.attention_backend == 'xformers':
        from xformers.ops.fmha.dispatch import _set_use_fa3
        _set_use_fa3(False)
    torch.set_num_threads(4)
    torch.hub.set_dir(str(args.native_setup / 'checkpoints/official-torch-hub'))
    original_hub = torch.hub.load

    def pinned_hub(repo, name, *hub_args, **hub_kwargs):
        if repo not in ('valeoai/NAF', 'valeoai/NAF:main'):
            raise ValueError('unprovisioned torch hub source: ' + repo)
        hub_kwargs.pop('source', None)
        return original_hub(str(args.native_setup / 'sources/NAF'), name,
                            *hub_args, source='local', **hub_kwargs)

    torch.hub.load = pinned_hub
    import inference as native
    return torch, native.init_pipeline('TencentARC/Pixal3D', low_vram=False)


def run_one(row: dict, args: argparse.Namespace, torch, pipeline) -> dict:
    import numpy as np
    from PIL import Image

    sid = row['sample_id']
    output = args.output_root / 'cases' / sid
    receipt_path = output / 'generation.json'
    target = Path(row['target_anchor'])
    target_hash = sha256(target)
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get('status') != 'PASS_OFFICIAL_PIXAL3D_GIVEN1024_GT_FREE'
                or receipt.get('sample_id') != sid
                or receipt.get('target_rgb_sha256') != target_hash
                or receipt.get('mesh_scale') != args.fixed_mesh_scale
                or not (output / 'prediction.npz').is_file()
                or receipt.get('prediction_sha256') != sha256(output / 'prediction.npz')):
            raise ValueError('existing case receipt mismatch: ' + sid)
        return {'sample_id': sid, 'status': 'REUSED', 'seconds': 0.0}
    if output.exists():
        raise FileExistsError('partial case needs review: ' + str(output))
    if row.get('target_anchor_kind') != 'rgb' or len(row.get('camera_angle_x', [])) != 4:
        raise ValueError('frozen target RGB and four camera records required')
    matrix = np.asarray(row['transform_matrix'][0], dtype=np.float64)
    camera = {'camera_angle_x': float(row['camera_angle_x'][0]),
              'distance': float(np.linalg.norm(matrix[:3, 3])),
              'mesh_scale': args.fixed_mesh_scale}
    if not 0 < camera['camera_angle_x'] < np.pi or camera['distance'] <= 0:
        raise ValueError('invalid provided camera')
    output.mkdir(parents=True)
    started = time.monotonic()
    try:
        image = pipeline.preprocess_image(Image.open(target))
        torch.manual_seed(0)
        with torch.inference_mode():
            meshes, (_, _, resolution) = pipeline.run(
                image, camera_params=camera, seed=0,
                sparse_structure_sampler_params={'steps': 12, 'guidance_strength': 7.5,
                                                 'guidance_rescale': .7, 'rescale_t': 5.},
                shape_slat_sampler_params={'steps': 12, 'guidance_strength': 7.5,
                                           'guidance_rescale': .5, 'rescale_t': 3.},
                tex_slat_sampler_params={'steps': 12, 'guidance_strength': 1.,
                                         'guidance_rescale': 0., 'rescale_t': 3.},
                preprocess_image=False, return_latent=True,
                pipeline_type='1024_cascade', max_num_tokens=32768)
        torch.cuda.synchronize()
        if len(meshes) != 1 or int(resolution) != 1024:
            raise ValueError('official 1024 cascade did not complete')
        mesh = meshes[0]
        vertices = mesh.vertices.detach().cpu().numpy().copy()
        faces = mesh.faces.detach().cpu().numpy().copy()
        if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
            raise ValueError('invalid official predicted surface')
        prediction = output / 'prediction.npz'
        temporary = output / 'prediction.tmp.npz'
        np.savez_compressed(temporary, vertices=vertices, faces=faces,
                            coords=mesh.coords.detach().cpu().numpy().copy(),
                            attrs=mesh.attrs.detach().cpu().numpy().copy())
        temporary.replace(prediction)
        receipt = {'status': 'PASS_OFFICIAL_PIXAL3D_GIVEN1024_GT_FREE',
                   'sample_id': sid, 'target_rgb': str(target),
                   'target_rgb_sha256': target_hash,
                   'prediction': str(prediction),
                   'prediction_sha256': sha256(prediction),
                   'camera_params': camera, 'mesh_scale': args.fixed_mesh_scale,
                   'scale_provenance': 'predeclared fixed 0.96, common across TRAIN/VAL/TEST prior manifests',
                   'seed': 0, 'actual_resolution': int(resolution),
                   'vertices': len(vertices), 'faces': len(faces),
                   'generation_seconds': time.monotonic() - started,
                   'GT_target_geometry_or_depth_used': False}
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        return {'sample_id': sid, 'status': 'PASS',
                'seconds': round(receipt['generation_seconds'], 2)}
    except BaseException as exc:
        (output / 'failure.json').write_text(json.dumps({
            'sample_id': sid, 'error': repr(exc),
            'traceback': traceback.format_exc()}, indent=2) + '\n')
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--pixal3d-repo', type=Path, required=True)
    parser.add_argument('--native-setup', type=Path, required=True)
    parser.add_argument('--physical-gpu', type=int, choices=tuple(range(8)), required=True)
    parser.add_argument('--fixed-mesh-scale', type=float, required=True)
    parser.add_argument('--attention-backend', choices=('xformers', 'sdpa'), default='xformers')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.fixed_mesh_scale != .96 or args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError('only predeclared GT-free scale 0.96 and valid shard allowed')
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    if len(rows) != 508 or len({row['sample_id'] for row in rows}) != 508:
        raise ValueError('frozen TEST508 manifest required')
    args.output_root.mkdir(parents=True, exist_ok=True)
    torch, pipeline = load_pipeline(args)
    done = 0
    for index, row in enumerate(rows):
        if index % args.world_size != args.rank:
            continue
        result = run_one(row, args, torch, pipeline)
        print(json.dumps({'index': index, 'rank': args.rank, **result}), flush=True)
        done += 1
        if args.limit and done >= args.limit:
            break
    print(json.dumps({'status': 'PASS_WORKER_SHARD', 'rank': args.rank,
                      'cases_seen': done}), flush=True)


if __name__ == '__main__':
    main()
