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
import threading
import time
from easydict import EasyDict as edict
from queue import Empty, Full, Queue

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


def _coordinates_to_uint8(coords, grid_resolution):
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f'invalid coordinate shape: {coords.shape}')
    if not (
        np.issubdtype(coords.dtype, np.integer)
        or np.issubdtype(coords.dtype, np.floating)
    ):
        raise ValueError(f'invalid coordinate dtype: {coords.dtype}')
    if not np.isfinite(coords).all():
        raise ValueError('coordinates must be finite')
    if not np.equal(coords, np.trunc(coords)).all():
        raise ValueError('coordinates must be integral')
    if (coords < 0).any():
        raise ValueError('coordinates must be non-negative')
    if (coords >= grid_resolution).any():
        raise ValueError('coordinates outside grid resolution')
    if (coords > np.iinfo(np.uint8).max).any():
        raise ValueError('coordinates exceed uint8 storage range')
    narrowed = coords.astype(np.uint8)
    if not np.array_equal(narrowed.astype(np.float64), coords.astype(np.float64)):
        raise ValueError('coordinate narrowing to uint8 was not lossless')
    return narrowed


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


def _put_cancellable(queue, item, cancel_event):
    while not cancel_event.is_set():
        try:
            queue.put(item, timeout=0.05)
            return True
        except Full:
            continue
    return False


def _worker_error(error_queue):
    try:
        stage, task, error = error_queue.get_nowait()
    except Empty:
        return None
    return RuntimeError(f'{stage} failed for {task}: {error}')


def _wait_for_result(queue, error_queue, timeout_seconds, stage):
    deadline = time.monotonic() + timeout_seconds
    while True:
        error = _worker_error(error_queue)
        if error is not None:
            raise error
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f'{stage} inactivity timed out after {timeout_seconds} seconds'
            )
        try:
            return queue.get(timeout=min(0.05, remaining))
        except Empty:
            continue


def _run_bounded_pipeline(
    tasks,
    load,
    process,
    save,
    cleanup,
    loader_workers,
    saver_workers,
    timeout_seconds,
):
    tasks = list(tasks)
    cancel_event = threading.Event()
    loader_tasks = Queue(maxsize=max(2, loader_workers * 2))
    loader_results = Queue(maxsize=max(2, loader_workers * 2))
    saver_tasks = Queue(maxsize=max(2, saver_workers * 2))
    saver_results = Queue(maxsize=max(2, saver_workers * 2))
    errors = Queue(maxsize=max(2, loader_workers + saver_workers))
    sentinel = object()

    def report_error(stage, task, error):
        try:
            errors.put_nowait((stage, task, error))
        except Full:
            pass
        cancel_event.set()

    def produce():
        for task in tasks:
            if not _put_cancellable(loader_tasks, task, cancel_event):
                return
        for _ in range(loader_workers):
            if not _put_cancellable(loader_tasks, sentinel, cancel_event):
                return

    def loader_worker():
        while not cancel_event.is_set():
            try:
                task = loader_tasks.get(timeout=0.05)
            except Empty:
                continue
            if task is sentinel:
                return
            try:
                result = load(task, cancel_event)
            except BaseException as error:
                report_error('loader', task, error)
                return
            if not _put_cancellable(
                loader_results, (task, result), cancel_event
            ):
                return

    def saver_worker():
        while not cancel_event.is_set():
            try:
                task, payload = saver_tasks.get(timeout=0.05)
            except Empty:
                continue
            try:
                record = save(task, payload, cancel_event)
            except BaseException as error:
                report_error('saver', task, error)
                return
            if not _put_cancellable(
                saver_results, (task, record), cancel_event
            ):
                return

    threads = [threading.Thread(target=produce, daemon=True)]
    threads.extend(
        threading.Thread(target=loader_worker, daemon=True)
        for _ in range(loader_workers)
    )
    threads.extend(
        threading.Thread(target=saver_worker, daemon=True)
        for _ in range(saver_workers)
    )
    for thread in threads:
        thread.start()

    records = []
    pending_saves = []

    def drain_savers():
        while True:
            error = _worker_error(errors)
            if error is not None:
                raise error
            try:
                task, record = saver_results.get_nowait()
            except Empty:
                return
            pending_saves.remove(task)
            if record is not None:
                records.append(record)

    try:
        for _ in range(len(tasks)):
            drain_savers()
            task, (payload, immediate_record) = _wait_for_result(
                loader_results, errors, timeout_seconds, 'loader'
            )
            if immediate_record is not None:
                records.append(immediate_record)
            if payload is None:
                continue
            save_payload = process(task, payload)
            if save_payload is None:
                continue
            deadline = time.monotonic() + timeout_seconds
            while True:
                drain_savers()
                try:
                    saver_tasks.put_nowait((task, save_payload))
                    pending_saves.append(task)
                    break
                except Full:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            'saver queue timed out while awaiting capacity'
                        )
                    time.sleep(0.01)
        while pending_saves:
            task, record = _wait_for_result(
                saver_results, errors, timeout_seconds, 'saver'
            )
            pending_saves.remove(task)
            if record is not None:
                records.append(record)
        return records
    except BaseException:
        cancel_event.set()
        for task in list(pending_saves):
            cleanup(task)
        raise
    finally:
        cancel_event.set()

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--dual_grid_root', type=str, default=None,
                        help='Directory containing the dual grids')
    parser.add_argument('--shape_latent_root', type=str, default=None,
                        help='Directory to save the shape latent files')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--resolution', type=int, default=1024,
                        help='Sparse voxel resolution')
    parser.add_argument('--enc_pretrained', type=str, default='microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16',
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
    opt.dual_grid_root = opt.dual_grid_root or opt.root
    opt.shape_latent_root = opt.shape_latent_root or opt.root

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
    latent_view_name = f'{latent_name}_view'
    os.makedirs(os.path.join(opt.shape_latent_root, 'shape_latents', latent_view_name, 'new_records'), exist_ok=True)
    
    # Get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        aesthetic_metadata = pd.read_csv(os.path.join(opt.root, 'aesthetic_scores','metadata.csv')).set_index('sha256')
        metadata = metadata.join(aesthetic_metadata, how='left', rsuffix='_aesthetic')
    
    # Check dual_grid_view metadata
    dual_grid_view_path = os.path.join(opt.dual_grid_root, f'dual_grid_view_{opt.resolution}', 'metadata.csv')
    if os.path.exists(dual_grid_view_path):
        dual_grid_metadata = pd.read_csv(dual_grid_view_path).set_index('sha256')
        metadata = metadata.join(dual_grid_metadata, how='left', rsuffix='_dual_grid')
    
    # Check shape_latent_view metadata (used to skip already completed tasks)
    shape_latent_view_metadata_path = os.path.join(opt.shape_latent_root, 'shape_latents', latent_view_name, 'metadata.csv')
    if os.path.exists(shape_latent_view_metadata_path):
        shape_latent_view_metadata = pd.read_csv(shape_latent_view_metadata_path).set_index('sha256')
        metadata = metadata.join(shape_latent_view_metadata, how='left', rsuffix='_shape_latent_view')
        print(f'Loaded shape_latent_view metadata with {len(shape_latent_view_metadata)} records')
    else:
        print(f'Warning: shape_latent_view metadata not found at {shape_latent_view_metadata_path}')
    
    metadata = metadata.reset_index()
    
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        
        # Filter to objects that have dual_grid_view data
        # Use first view as indicator
        first_view_col = f'dual_grid_view{view_indices[0]:02d}_converted_{opt.resolution}'
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

    def task_paths(task):
        sha256, view_idx = task
        output_path = Path(opt.shape_latent_root) / 'shape_latents' / latent_view_name / sha256 / f'view{view_idx:02d}.npz'
        source_scale = Path(opt.dual_grid_root) / f'dual_grid_view_{opt.resolution}' / sha256 / f'view{view_idx:02d}_scale.json'
        destination_scale = output_path.with_name(f'view{view_idx:02d}_scale.json')
        vxz_path = Path(opt.dual_grid_root) / f'dual_grid_view_{opt.resolution}' / sha256 / f'view{view_idx:02d}.vxz'
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
                    f'shape_latent_view{view_idx:02d}_encoded': True,
                    f'shape_latent_view{view_idx:02d}_tokens': num_tokens,
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
        coords, attr = o_voxel.io.read_vxz(str(vxz_path), num_threads=1)
        vertices = sp.SparseTensor(
            (attr['vertices'] / 255.0).float(),
            torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1),
        )
        intersected = vertices.replace(torch.cat([
            attr['intersected'] % 2,
            attr['intersected'] // 2 % 2,
            attr['intersected'] // 4 % 2,
        ], dim=-1).bool())
        if not (
            is_valid_sparse_tensor(vertices)
            and is_valid_sparse_tensor(intersected)
        ):
            print(f'[Loader Skip] {sha256}/view{view_idx:02d}: NaN/Inf in input')
            return None, None
        return (vertices, intersected), None

    def process_batch(payloads):
        vertices = batch_sparse_tensors(
            [payload[0] for payload in payloads]
        )
        intersected = batch_sparse_tensors(
            [payload[1] for payload in payloads]
        )
        if not torch.equal(vertices.coords, intersected.coords):
            raise ValueError('shape encoder batch coordinates do not align')
        z = encoder(vertices.cuda(), intersected.cuda())
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
            f'shape_latent_view{view_idx:02d}_encoded': True,
            f'shape_latent_view{view_idx:02d}_tokens': num_tokens,
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
        Path(opt.shape_latent_root) / 'shape_latents' / latent_view_name / 'new_records' / f'{opt.record_prefix}part_{opt.rank}.csv',
    )
