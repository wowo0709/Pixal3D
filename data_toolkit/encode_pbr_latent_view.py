import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import argparse
from pathlib import Path
import tempfile
import torch
import numpy as np
import pandas as pd
import o_voxel
from easydict import EasyDict as edict

if __package__:
    from .utils import parse_view_indices
else:
    from utils import parse_view_indices

from data_toolkit.pipeline.atomic_io import atomic_copy, atomic_save_npz
from data_toolkit.pipeline.sparse_batching import (
    batch_sparse_tensors,
    run_encoder_tasks,
    split_sparse_tensor,
    validate_record_prefix,
)
from data_toolkit.pipeline.validation import (
    validate_scale,
    validate_sparse_latent,
)
from data_toolkit.encode_shape_latent_view import (
    _coordinates_to_uint8,
    _run_bounded_pipeline,
)

import pixal3d.models as models
import pixal3d.modules.sparse as sp

def is_valid_sparse_tensor(tensor):
    return torch.isfinite(tensor.feats).all() and torch.isfinite(tensor.coords).all()

def clear_cuda_error():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _sync_parent(path):
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
    directory = os.open(Path(path).parent, flags)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _atomic_write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f'.{path.stem}.',
            suffix='.csv',
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        frame.to_csv(temporary, index=False)
        pd.read_csv(temporary)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_parent(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _existing_sparse_tokens(path, grid_resolution):
    path = Path(path)
    if not path.exists():
        return None
    try:
        validate_sparse_latent(
            path,
            grid_resolution=grid_resolution,
            max_tokens=grid_resolution**3,
        )
        with np.load(path, allow_pickle=False) as data:
            return int(data['coords'].shape[0])
    except Exception as error:
        print(f'Removing corrupt sparse latent {path}: {error}')
        path.unlink(missing_ok=True)
        return None


def _publish_sparse_latent(path, feats, coords, grid_resolution):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f'.{path.stem}.',
            suffix='.npz',
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        atomic_save_npz(temporary, feats=feats, coords=coords)
        validate_sparse_latent(
            temporary,
            grid_resolution=grid_resolution,
            max_tokens=grid_resolution**3,
        )
        os.replace(temporary, path)
        _sync_parent(path)
        try:
            validate_sparse_latent(
                path,
                grid_resolution=grid_resolution,
                max_tokens=grid_resolution**3,
            )
        except Exception:
            path.unlink(missing_ok=True)
            _sync_parent(path)
            raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _encode_sparse_output(
    path,
    encode,
    grid_resolution,
    latent_dtype='float32',
    scale_source=None,
    scale_destination=None,
    cancel_event=None,
):
    path = Path(path)
    scale_source = Path(scale_source) if scale_source is not None else Path('')
    scale_destination = (
        Path(scale_destination) if scale_destination is not None else Path('')
    )
    validate_scale(scale_source)
    existing_tokens = _existing_sparse_tokens(path, grid_resolution)
    if existing_tokens is not None:
        try:
            validate_scale(scale_destination)
            return existing_tokens
        except Exception:
            path.unlink(missing_ok=True)
            scale_destination.unlink(missing_ok=True)

    z = encode()
    if not torch.isfinite(z.feats).all() or not torch.isfinite(z.coords).all():
        raise ValueError('encoder produced a non-finite sparse latent')
    feature_dtype = np.float16 if latent_dtype == 'float16' else np.float32
    raw_coords = z.coords[:, 1:].cpu().numpy()
    coords = _coordinates_to_uint8(raw_coords, grid_resolution)
    copied_scale = False
    try:
        if scale_destination.exists():
            try:
                validate_scale(scale_destination)
            except Exception:
                scale_destination.unlink(missing_ok=True)
        if not scale_destination.exists():
            atomic_copy(scale_source, scale_destination)
            copied_scale = True
        validate_scale(scale_destination)
        if cancel_event is not None and cancel_event.is_set():
            raise TimeoutError('saver cancelled before latent publication')
        _publish_sparse_latent(
            path,
            feats=z.feats.cpu().numpy().astype(feature_dtype),
            coords=coords,
            grid_resolution=grid_resolution,
        )
        if cancel_event is not None and cancel_event.is_set():
            raise TimeoutError('saver cancelled after latent publication')
    except Exception:
        path.unlink(missing_ok=True)
        if copied_scale:
            scale_destination.unlink(missing_ok=True)
        raise
    return int(z.coords.shape[0])


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--pbr_voxel_root', type=str, default=None,
                        help='Directory containing the pbr voxels')
    parser.add_argument('--pbr_latent_root', type=str, default=None,
                        help='Directory to save the pbr latent files')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--resolution', type=int, default=1024,
                        help='Sparse voxel resolution')
    parser.add_argument('--enc_pretrained', type=str, default='microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16',
                        help='Pretrained encoder model')
    parser.add_argument('--model_root', type=str,
                        help='Root directory of models')
    parser.add_argument('--enc_model', type=str,
                        help='Encoder model. if specified, use this model instead of pretrained model')
    parser.add_argument('--ckpt', type=str,
                        help='Checkpoint to load')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--view_indices', type=str, default=None,
                        help='View indices to process, e.g., "0,1,2" or "0-5". None for all views')
    parser.add_argument('--num_views', type=int, default=24,
                        help='Total number of views (used when view_indices is None)')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--loader_workers', type=int, default=2)
    parser.add_argument('--saver_workers', type=int, default=1)
    parser.add_argument('--latent_dtype', choices=('float32', 'float16'), default='float32')
    parser.add_argument('--timeout_seconds', type=int, default=900)
    parser.add_argument('--micro_batch_size', type=int, required=True)
    parser.add_argument('--gpu_memory_target_percent', type=float, default=80.0)
    parser.add_argument('--record_prefix', default='')
    opt = parser.parse_args()
    if opt.loader_workers <= 0:
        parser.error('--loader_workers must be positive')
    if opt.saver_workers <= 0:
        parser.error('--saver_workers must be positive')
    if opt.timeout_seconds <= 0:
        parser.error('--timeout_seconds must be positive')
    if opt.micro_batch_size <= 0:
        parser.error('--micro_batch_size must be positive')
    if not 0 < opt.gpu_memory_target_percent < 100:
        parser.error('--gpu_memory_target_percent must be in (0, 100)')
    opt.record_prefix = validate_record_prefix(opt.record_prefix)
    opt = edict(vars(opt))
    opt.pbr_voxel_root = opt.pbr_voxel_root or opt.root
    opt.pbr_latent_root = opt.pbr_latent_root or opt.root

    # Parse view_indices
    view_indices = parse_view_indices(opt.view_indices)
    if view_indices is None:
        view_indices = list(range(opt.num_views))
    
    print(f'View indices to process: {view_indices}')

    if opt.enc_model is None:
        latent_name = f'{opt.enc_pretrained.split("/")[-1]}_{opt.resolution}'
        encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()
    else:
        latent_name = f'{opt.enc_model.split("/")[-1]}_{opt.ckpt}_{opt.resolution}'
        cfg = edict(json.load(open(os.path.join(opt.model_root, opt.enc_model, 'config.json'), 'r')))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
        ckpt_path = os.path.join(opt.model_root, opt.enc_model, 'ckpts', f'encoder_{opt.ckpt}.pt')
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
        encoder.eval()
        print(f'Loaded model from {ckpt_path}')
    
    # Multi-view latent output directory
    latent_view_name = f'{latent_name}_view_fix'
    os.makedirs(os.path.join(opt.pbr_latent_root, 'pbr_latents', latent_view_name, 'new_records'), exist_ok=True)
    
    # Get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        aesthetic_metadata = pd.read_csv(os.path.join(opt.root, 'aesthetic_scores','metadata.csv')).set_index('sha256')
        metadata = metadata.join(aesthetic_metadata, how='left', rsuffix='_aesthetic')
    
    # Check pbr_voxels_view_fix metadata
    pbr_voxel_view_path = os.path.join(opt.pbr_voxel_root, f'pbr_voxels_view_fix_{opt.resolution}', 'metadata.csv')
    if os.path.exists(pbr_voxel_view_path):
        pbr_voxel_metadata = pd.read_csv(pbr_voxel_view_path).set_index('sha256')
        metadata = metadata.join(pbr_voxel_metadata, how='left', rsuffix='_pbr_voxel')
    
    # Check pbr_latent_view metadata (used to skip already completed tasks)
    pbr_latent_view_metadata_path = os.path.join(opt.pbr_latent_root, 'pbr_latents', latent_view_name, 'metadata.csv')
    if os.path.exists(pbr_latent_view_metadata_path):
        pbr_latent_view_metadata = pd.read_csv(pbr_latent_view_metadata_path).set_index('sha256')
        metadata = metadata.join(pbr_latent_view_metadata, how='left', rsuffix='_pbr_latent_view')
        print(f'Loaded pbr_latent_view metadata with {len(pbr_latent_view_metadata)} records')
    else:
        print(f'Warning: pbr_latent_view metadata not found at {pbr_latent_view_metadata_path}')
    
    metadata = metadata.reset_index()
    
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        
        # Filter to objects that have pbr_voxels_view_fix data
        # Use first view as indicator
        first_view_col = f'pbr_voxelized_view_fix{view_indices[0]:02d}_{opt.resolution}'
        if first_view_col in metadata.columns:
            metadata = metadata[metadata[first_view_col] == True]
        else:
            print(f'Warning: Column {first_view_col} not found in metadata, will check files directly')
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    records = []
    
    # Build all tasks. Files, not stale metadata, are the source of resume truth.
    all_tasks = []
    for _, row in metadata.iterrows():
        sha256 = row['sha256']
        for view_idx in view_indices:
            all_tasks.append((sha256, view_idx))
    
    # Split tasks by rank after filtering completed ones
    start = len(all_tasks) * opt.rank // opt.world_size
    end = len(all_tasks) * (opt.rank + 1) // opt.world_size
    tasks = all_tasks[start:end]
    
    print(f'Total tasks to validate or process: {len(all_tasks)}, This rank: {len(tasks)}')

    def task_paths(task):
        sha256, view_idx = task
        output_path = Path(opt.pbr_latent_root) / 'pbr_latents' / latent_view_name / sha256 / f'view{view_idx:02d}.npz'
        source_scale = Path(opt.pbr_voxel_root) / f'pbr_voxels_view_fix_{opt.resolution}' / sha256 / f'view{view_idx:02d}_scale.json'
        destination_scale = output_path.with_name(f'view{view_idx:02d}_scale.json')
        vxz_path = Path(opt.pbr_voxel_root) / f'pbr_voxels_view_fix_{opt.resolution}' / sha256 / f'view{view_idx:02d}.vxz'
        return output_path, source_scale, destination_scale, vxz_path

    def load(task, cancel_event):
        sha256, view_idx = task
        output_path, source_scale, destination_scale, vxz_path = task_paths(task)
        num_tokens = _existing_sparse_tokens(output_path, opt.resolution)
        if num_tokens is not None:
            try:
                validate_scale(source_scale)
                validate_scale(destination_scale)
                return None, {
                    'sha256': sha256,
                    f'pbr_latent_view{view_idx:02d}_encoded': True,
                    f'pbr_latent_view{view_idx:02d}_tokens': num_tokens,
                }
            except Exception as error:
                output_path.unlink(missing_ok=True)
                destination_scale.unlink(missing_ok=True)
                print(f'[Loader Repair] {sha256}/view{view_idx:02d}: {error}')
        try:
            validate_scale(source_scale)
        except Exception as error:
            print(f'[Loader Skip] {sha256}/view{view_idx:02d}: {error}')
            return None, None
        if not vxz_path.exists():
            print(f'[Loader Skip] {sha256}/view{view_idx:02d}: vxz file not found')
            return None, None
        attrs = ['base_color', 'metallic', 'roughness', 'alpha']
        coords, attr = o_voxel.io.read_vxz(str(vxz_path), num_threads=1)
        feats = torch.concat([attr[key] for key in attrs], dim=-1) / 255.0 * 2 - 1
        voxels = sp.SparseTensor(
            feats.float(),
            torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1),
        )
        if not is_valid_sparse_tensor(voxels):
            print(f'[Loader Skip] {sha256}/view{view_idx:02d}: NaN/Inf in input')
            return None, None
        return voxels, None

    def process_batch(voxels):
        z = encoder(batch_sparse_tensors(voxels).cuda())
        torch.cuda.synchronize()
        outputs = split_sparse_tensor(z)
        if any(not torch.isfinite(output.feats).all() for output in outputs):
            clear_cuda_error()
            return [
                output if torch.isfinite(output.feats).all() else None
                for output in outputs
            ]
        return outputs

    def save(task, z, cancel_event):
        sha256, view_idx = task
        output_path, source_scale, destination_scale, _ = task_paths(task)
        num_tokens = _encode_sparse_output(
            output_path,
            lambda: z,
            grid_resolution=opt.resolution,
            latent_dtype=opt.latent_dtype,
            scale_source=source_scale,
            scale_destination=destination_scale,
            cancel_event=cancel_event,
        )
        return {
            'sha256': sha256,
            f'pbr_latent_view{view_idx:02d}_encoded': True,
            f'pbr_latent_view{view_idx:02d}_tokens': num_tokens,
        }

    def cleanup(task):
        output_path, _, destination_scale, _ = task_paths(task)
        output_path.unlink(missing_ok=True)
        destination_scale.unlink(missing_ok=True)

    records = run_encoder_tasks(
        tasks=tasks,
        micro_batch_size=opt.micro_batch_size,
        load=load,
        process_batch=process_batch,
        save=save,
        loader_workers=opt.loader_workers,
        saver_workers=opt.saver_workers,
        timeout_seconds=opt.timeout_seconds,
        gpu_memory_target_percent=opt.gpu_memory_target_percent,
    )

    records = pd.DataFrame.from_records(records)
    if len(records.columns) == 0:
        records = pd.DataFrame(columns=['sha256'])
    _atomic_write_csv(
        records,
        Path(opt.pbr_latent_root) / 'pbr_latents' / latent_view_name / 'new_records' / f'{opt.record_prefix}part_{opt.rank}.csv',
    )
