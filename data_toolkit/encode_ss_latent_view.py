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
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full, Queue

if __package__:
    from .utils import parse_view_indices
else:
    from utils import parse_view_indices

from data_toolkit.pipeline.atomic_io import atomic_copy, atomic_save_npz
from data_toolkit.pipeline.validation import (
    validate_scale,
    validate_sparse_latent,
    validate_ss_latent,
)

import pixal3d.models as models

torch.set_grad_enabled(False)

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


def _existing_ss_output(path):
    path = Path(path)
    if not path.exists():
        return False
    try:
        validate_ss_latent(path)
        return True
    except Exception as error:
        print(f'Removing corrupt SS latent {path}: {error}')
        path.unlink(missing_ok=True)
        return False


def _publish_ss_latent(path, z):
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
        atomic_save_npz(temporary, z=z)
        validate_ss_latent(temporary)
        os.replace(temporary, path)
        _sync_parent(path)
        validate_ss_latent(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _encode_ss_output(path, encode, latent_dtype='float32'):
    path = Path(path)
    if _existing_ss_output(path):
        return
    z = encode()
    if not torch.isfinite(z).all():
        raise ValueError('encoder produced a non-finite SS latent')
    feature_dtype = np.float16 if latent_dtype == 'float16' else np.float32
    _publish_ss_latent(
        path,
        z=z[0].cpu().numpy().astype(feature_dtype),
    )


def _copy_valid_scale(source, destination):
    source = Path(source)
    destination = Path(destination)
    if destination.exists():
        try:
            validate_scale(destination)
            return
        except Exception:
            destination.unlink(missing_ok=True)
    if not source.exists():
        return
    try:
        validate_scale(source)
    except Exception as error:
        print(f'[Scale Skip] Invalid source {source}: {error}')
        return
    atomic_copy(source, destination)
    validate_scale(destination)


def _put_with_timeout(queue, item, timeout_seconds):
    try:
        queue.put(item, timeout=timeout_seconds)
        return True
    except Full:
        print(f'[Loader Timeout] Output queue stayed full for {timeout_seconds}s')
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--shape_latent_root', type=str, default=None,
                        help='Directory containing the shape latent files')
    parser.add_argument('--ss_latent_root', type=str, default=None,
                        help='Directory to save the ss latent files')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--resolution', type=int, default=32,
                        help='SS latent resolution')
    parser.add_argument('--shape_latent_name', type=str, required=True,
                        help='Name of the shape latent files (e.g., shape_enc_next_dc_f16c32_fp16_512)')
    parser.add_argument('--enc_pretrained', type=str, default='microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16',
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
    opt = parser.parse_args()
    if opt.loader_workers <= 0:
        parser.error('--loader_workers must be positive')
    if opt.saver_workers <= 0:
        parser.error('--saver_workers must be positive')
    if opt.timeout_seconds <= 0:
        parser.error('--timeout_seconds must be positive')
    opt = edict(vars(opt))
    opt.shape_latent_root = opt.shape_latent_root or opt.root
    opt.ss_latent_root = opt.ss_latent_root or opt.root

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
    
    # Multi-view shape_latent and ss_latent directory names
    shape_latent_view_name = f'{opt.shape_latent_name}_view'
    ss_latent_view_name = f'{latent_name}_view'
    
    os.makedirs(os.path.join(opt.ss_latent_root, 'ss_latents', ss_latent_view_name, 'new_records'), exist_ok=True)
    
    # Get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        aesthetic_metadata = pd.read_csv(os.path.join(opt.root, 'aesthetic_scores','metadata.csv')).set_index('sha256')
        metadata = metadata.join(aesthetic_metadata, how='left', rsuffix='_aesthetic')
    
    # Check shape_latent_view metadata
    shape_latent_view_metadata_path = os.path.join(opt.shape_latent_root, 'shape_latents', shape_latent_view_name, 'metadata.csv')
    if os.path.exists(shape_latent_view_metadata_path):
        shape_latent_view_metadata = pd.read_csv(shape_latent_view_metadata_path).set_index('sha256')
        metadata = metadata.join(shape_latent_view_metadata, how='left', rsuffix='_shape_latent_view')
        print(f'Loaded shape_latent_view metadata with {len(shape_latent_view_metadata)} records')
    else:
        print(f'Warning: shape_latent_view metadata not found at {shape_latent_view_metadata_path}')
    
    # Check ss_latent_view metadata (used to skip already completed tasks)
    ss_latent_view_metadata_path = os.path.join(opt.ss_latent_root, 'ss_latents', ss_latent_view_name, 'metadata.csv')
    if os.path.exists(ss_latent_view_metadata_path):
        ss_latent_view_metadata = pd.read_csv(ss_latent_view_metadata_path).set_index('sha256')
        metadata = metadata.join(ss_latent_view_metadata, how='left', rsuffix='_ss_latent_view')
        print(f'Loaded ss_latent_view metadata with {len(ss_latent_view_metadata)} records')
    else:
        print(f'Warning: ss_latent_view metadata not found at {ss_latent_view_metadata_path}')
    
    metadata = metadata.reset_index()
    
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        
        # Filter to objects that have shape_latent_view data
        # Use first view as indicator
        first_view_col = f'shape_latent_view{view_indices[0]:02d}_encoded'
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

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []
    
    # Build all tasks. Files, not stale metadata, are the source of resume truth.
    tasks = []
    for _, row in metadata.iterrows():
        sha256 = row['sha256']
        for view_idx in view_indices:
            tasks.append((sha256, view_idx))

    print(f'Total tasks to validate or process: {len(tasks)}')

    load_queue = Queue(maxsize=max(2, opt.loader_workers * 2))
    saver_futures = []

    with ThreadPoolExecutor(max_workers=opt.loader_workers) as loader_executor, \
         ThreadPoolExecutor(max_workers=opt.saver_workers) as saver_executor:

        def loader(task):
            sha256, view_idx = task
            try:
                output_path = os.path.join(
                    opt.ss_latent_root,
                    'ss_latents',
                    ss_latent_view_name,
                    sha256,
                    f'view{view_idx:02d}.npz'
                )
                if _existing_ss_output(output_path):
                    src_scale_path = Path(opt.shape_latent_root) / 'shape_latents' / shape_latent_view_name / sha256 / f'view{view_idx:02d}_scale.json'
                    dst_scale_path = Path(output_path).with_name(f'view{view_idx:02d}_scale.json')
                    _copy_valid_scale(src_scale_path, dst_scale_path)
                    records.append({
                        'sha256': sha256,
                        f'ss_latent_view{view_idx:02d}_encoded': True,
                    })
                    _put_with_timeout(
                        load_queue,
                        (sha256, view_idx, None),
                        opt.timeout_seconds,
                    )
                    return
                
                # shape_latent_view path: shape_latents/{shape_latent_view_name}/{sha256}/view{idx:02d}.npz
                npz_path = os.path.join(
                    opt.shape_latent_root, 
                    'shape_latents',
                    shape_latent_view_name, 
                    sha256, 
                    f'view{view_idx:02d}.npz'
                )
                
                if not os.path.exists(npz_path):
                    print(f"[Loader Skip] {sha256}/view{view_idx:02d}: npz file not found at {npz_path}")
                    _put_with_timeout(
                        load_queue,
                        (sha256, view_idx, None),
                        opt.timeout_seconds,
                    )
                    return

                validate_sparse_latent(
                    Path(npz_path),
                    grid_resolution=opt.resolution,
                    max_tokens=opt.resolution**3,
                )
                with np.load(npz_path, allow_pickle=False) as data:
                    coords = np.asarray(data['coords'])
                
                # Validate coords are within resolution range
                assert np.all(coords < opt.resolution), f"{sha256}/view{view_idx:02d}: Invalid coords (max={coords.max()}, resolution={opt.resolution})"
                
                coords = torch.from_numpy(coords).long()
                ss = torch.zeros(1, opt.resolution, opt.resolution, opt.resolution, dtype=torch.long)
                ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
                
                _put_with_timeout(
                    load_queue,
                    (sha256, view_idx, ss),
                    opt.timeout_seconds,
                )
            except Exception as e:
                print(f"[Loader Error] {sha256}/view{view_idx:02d}: {e}")
                _put_with_timeout(
                    load_queue,
                    (sha256, view_idx, None),
                    opt.timeout_seconds,
                )

        loader_executor.map(loader, tasks)
        
        def saver(sha256, view_idx, z):
            sha256_dir = os.path.join(opt.ss_latent_root, 'ss_latents', ss_latent_view_name, sha256)
            os.makedirs(sha256_dir, exist_ok=True)
            save_path = os.path.join(sha256_dir, f'view{view_idx:02d}.npz')
            _encode_ss_output(
                save_path,
                lambda: z,
                latent_dtype=opt.latent_dtype,
            )
            
            # Copy scale.json from shape_latent_view directory
            src_scale_path = os.path.join(
                opt.shape_latent_root,
                'shape_latents',
                shape_latent_view_name,
                sha256,
                f'view{view_idx:02d}_scale.json'
            )
            dst_scale_path = os.path.join(sha256_dir, f'view{view_idx:02d}_scale.json')
            _copy_valid_scale(src_scale_path, dst_scale_path)
            
            records.append({
                'sha256': sha256,
                f'ss_latent_view{view_idx:02d}_encoded': True,
            })
            
        for _ in tqdm(range(len(tasks)), desc="Extracting SS view latents"):
            try:
                sha256, view_idx, ss = load_queue.get(
                    timeout=opt.timeout_seconds
                )
                if ss is None:
                    continue
                
                ss = ss.cuda()[None].float()
                z = encoder(ss, sample_posterior=False)
                torch.cuda.synchronize()

                if not torch.isfinite(z).all():
                    print(f"[Skip] {sha256}/view{view_idx:02d}: Non-finite latent")
                    clear_cuda_error()
                    continue

                saver_futures.append(
                    saver_executor.submit(saver, sha256, view_idx, z)
                )

            except Empty:
                print(f'[Loader Timeout] No result received for {opt.timeout_seconds}s')
                break
            except Exception as e:
                print(f"[Error] {sha256}/view{view_idx:02d}: {e}")
                clear_cuda_error()
                continue

        for future in saver_futures:
            future.result(timeout=opt.timeout_seconds)

    records = pd.DataFrame.from_records(records)
    if len(records.columns) == 0:
        records = pd.DataFrame(columns=['sha256'])
    _atomic_write_csv(
        records,
        Path(opt.ss_latent_root) / 'ss_latents' / ss_latent_view_name / 'new_records' / f'part_{opt.rank}.csv',
    )
