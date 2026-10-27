"""One-case native Pixal3D camera/resolution factorial feasibility probe.

The official denoisers, preprocessing, sampler, and seed match the node16
runner. The two planned interventions are camera_params and cascade resolution.
This probe does not choose a method on TEST results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--sample-id', required=True)
    parser.add_argument('--pixal3d-repo', type=Path, required=True)
    parser.add_argument('--native-setup', type=Path, required=True)
    parser.add_argument('--physical-gpu', type=int, choices=tuple(range(8)), required=True)
    parser.add_argument('--camera-mode', choices=('given', 'predicted'), default='given')
    parser.add_argument('--given-mesh-scale', type=float,
                        help='GT-free fixed scale; required in given mode')
    parser.add_argument('--camera-record', type=Path)
    parser.add_argument('--resolution', type=int, choices=(1024, 1536), default=1536)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('immutable probe output already exists')

    row = None
    with args.manifest.open() as stream:
        for line in stream:
            candidate = json.loads(line)
            if candidate['sample_id'] == args.sample_id:
                row = candidate
                break
    if row is None:
        raise ValueError('sample not in frozen manifest')
    if row['target_anchor_kind'] != 'rgb' or len(row['camera_angle_x']) != 4:
        raise ValueError('RGB target and four camera records required')
    target = Path(row['target_anchor'])
    if not target.is_file():
        raise FileNotFoundError(target)
    matrix = row['transform_matrix'][0]
    distance = math.sqrt(sum(float(matrix[index][3]) ** 2 for index in range(3)))
    if args.camera_mode == 'given':
        if args.given_mesh_scale is None or not 0 < args.given_mesh_scale < 2:
            raise ValueError('given mode requires an explicit GT-free --given-mesh-scale')
        camera = {'camera_angle_x': float(row['camera_angle_x'][0]),
                  'distance': distance, 'mesh_scale': args.given_mesh_scale}
        camera_source = 'provided camera from frozen manifest; explicit GT-free mesh scale'
    else:
        if args.camera_record is None:
            raise ValueError('predicted mode requires a frozen official MoGe record')
        original = json.loads(args.camera_record.read_text())
        if original.get('status') != 'ok' or original['sample_id'] != args.sample_id:
            raise ValueError('official MoGe record identity or status mismatch')
        camera = {key: float(original['camera_params'][key])
                  for key in ('camera_angle_x', 'distance', 'mesh_scale')}
        if original.get('inputs') and original['inputs'][0]['sha256'] != sha256(target):
            raise ValueError('official MoGe camera was estimated from a different RGB')
        camera_source = str(args.camera_record)
    if not 0 < camera['camera_angle_x'] < math.pi or not 0 < camera['mesh_scale'] < 2:
        raise ValueError('camera input invalid')

    # Node16's pinned manifests use /root/dev paths from the original account;
    # the identical checkpoint bytes are also present in the ACL-readable
    # /home/seunghyun mirror. Build an isolated HF cache of symlinks only.
    runtime_root = args.native_setup.parent
    cache = args.output.parent / 'hf-cache'
    for name in ('TencentARC/Pixal3D',
                 'camenduru/dinov3-vitl16-pretrain-lvd1689m',
                 'briaai/RMBG-2.0'):
        record = json.loads((args.native_setup / 'manifests' /
                             ('checkpoint-' + name.replace('/', '--') + '.json')).read_text())
        if record['status'] != 'complete':
            raise ValueError('pinned checkpoint incomplete: ' + name)
        pinned = Path(record['local_dir'])
        relative = pinned.relative_to('/root/dev/editing_reproduction')
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
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.physical_gpu),
                      HF_HUB_CACHE=str(cache), HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4',
                      OPENBLAS_NUM_THREADS='1', ATTN_BACKEND='xformers',
                      CUDA_HOME='/usr/local/cuda-12.8', TORCH_CUDA_ARCH_LIST='12.0',
                      OPENCV_IO_ENABLE_OPENEXR='1')
    sys.path.insert(0, str(args.pixal3d_repo))
    import torch
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
    import numpy as np
    from PIL import Image
    import inference as native

    start = time.monotonic()
    pipeline = native.init_pipeline('TencentARC/Pixal3D', low_vram=False)
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
            pipeline_type=f'{args.resolution}_cascade',
            max_num_tokens=32768 if args.resolution == 1024 else 49152)
    torch.cuda.synchronize()
    if len(meshes) != 1 or int(resolution) != args.resolution:
        raise ValueError('official requested cascade did not complete')
    mesh = meshes[0]
    vertices = mesh.vertices.detach().cpu().numpy().copy()
    faces = mesh.faces.detach().cpu().numpy().copy()
    if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
        raise ValueError('invalid official predicted surface')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, vertices=vertices, faces=faces,
                        coords=mesh.coords.detach().cpu().numpy().copy(),
                        attrs=mesh.attrs.detach().cpu().numpy().copy())
    receipt = {'status': 'PASS_PIXAL3D_CAMERA_RESOLUTION_FEASIBILITY',
               'sample_id': args.sample_id, 'target_rgb': str(target),
               'target_rgb_sha256': sha256(target), 'camera_params': camera,
               'camera_source': camera_source, 'camera_mode': args.camera_mode,
               'given_mesh_scale_provenance': ('explicit GT-free input' if args.camera_mode == 'given' else None),
               'same_official_weights_sampler_seed_preprocessing': True,
               'changed_from_node16_official_ungiven': [
                   *(['camera_params'] if args.camera_mode == 'given' else []),
                   *(['cascade_resolution_and_token_cap'] if args.resolution == 1024 else [])],
               'actual_resolution': int(resolution), 'vertices': len(vertices),
               'faces': len(faces), 'elapsed_seconds': time.monotonic() - start,
               'raw_surface': str(args.output), 'raw_surface_sha256': sha256(args.output),
               'metrics_computed': False}
    args.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
