"""
voxelize_pbr_view.py - Multi-view transform PBR voxelization
Extends voxelize_pbr.py with scale and mesh rotation logic
Based on dual_grid_view.py and test_ovoxel_pbr_transform.py implementation
"""
import os
import copy
import sys
import importlib
import argparse
import ctypes
import json
import math
import multiprocessing
from pathlib import Path
import signal
import tempfile
import time
import traceback
import pandas as pd
import pickle
import numpy as np
import torch
from easydict import EasyDict as edict
from functools import partial
import o_voxel

if __package__:
    from .utils import get_new_camera_matrix, sphere_normalize_torch
    from .pipeline.atomic_io import atomic_write_json
    from .pipeline.dataset_adapter import process_single_metadata_row
    from .pipeline.validation import validate_scale
else:
    from utils import get_new_camera_matrix, sphere_normalize_torch
    from pipeline.atomic_io import atomic_write_json
    from pipeline.dataset_adapter import process_single_metadata_row
    from pipeline.validation import validate_scale


def _sync_parent(path):
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
    directory = os.open(Path(path).parent, flags)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _atomic_write_vxz(path, coord, attr, native_threads):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f'.{path.stem}.',
            suffix='.vxz',
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        o_voxel.io.write_vxz(
            str(temporary), coord, attr, num_threads=native_threads
        )
        info = o_voxel.io.read_vxz_info(str(temporary))
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_parent(path)
        try:
            return o_voxel.io.read_vxz_info(str(path)) or info
        except Exception:
            path.unlink(missing_ok=True)
            _sync_parent(path)
            raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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


def _read_vxz_output(vxz_path, scale_path):
    info = o_voxel.io.read_vxz_info(str(vxz_path))
    validate_scale(Path(scale_path))
    return info


def _publish_vxz_pair(
    vxz_path,
    scale_path,
    scale_info,
    coord,
    attr,
    native_threads,
):
    vxz_path = Path(vxz_path)
    scale_path = Path(scale_path)
    scale_published = False
    try:
        atomic_write_json(scale_path, scale_info)
        scale_published = True
        validate_scale(scale_path)
        return _atomic_write_vxz(
            vxz_path,
            coord,
            attr,
            native_threads=native_threads,
        )
    except Exception:
        vxz_path.unlink(missing_ok=True)
        if scale_published:
            scale_path.unlink(missing_ok=True)
        if vxz_path.parent.exists():
            _sync_parent(vxz_path)
        raise


def _foreach_child(
    result_path,
    error_path,
    dataset_utils,
    metadata,
    output_dir,
    func,
    desc,
):
    temporary = None
    try:
        result = process_single_metadata_row(
            dataset_utils, metadata, output_dir, func
        )
        result_path = Path(result_path)
        with tempfile.NamedTemporaryFile(
            dir=result_path.parent,
            prefix=f'.{result_path.stem}.',
            suffix='.pickle',
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            pickle.dump(result, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, result_path)
        temporary = None
        _sync_parent(result_path)
    except BaseException:
        Path(error_path).write_text(traceback.format_exc())
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _linux_syscall(number, *arguments):
    result = ctypes.CDLL(None, use_errno=True).syscall(number, *arguments)
    if result < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return int(result)


def _pidfd_open(pid):
    return _linux_syscall(434, pid, 0)


def _pidfd_send_signal(pidfd, sent_signal, siginfo=None, flags=0):
    return _linux_syscall(424, pidfd, int(sent_signal), 0, flags)


def _terminate_process(process):
    try:
        pidfd = _pidfd_open(process.pid)
    except ProcessLookupError:
        process.join(0.5)
        if process.is_alive():
            raise RuntimeError(f'worker process disappeared: {process.pid}')
        return
    try:
        _pidfd_send_signal(pidfd, signal.SIGTERM)
        process.join(0.5)
        if process.is_alive():
            _pidfd_send_signal(pidfd, signal.SIGKILL)
            process.join(0.5)
    finally:
        os.close(pidfd)
    if process.is_alive():
        raise RuntimeError(f'could not kill worker process {process.pid}')


def _run_foreach_bounded(
    dataset_utils,
    metadata,
    output_dir,
    func,
    max_workers,
    desc,
    timeout_seconds,
):
    context = multiprocessing.get_context('fork')
    worker_limit = (
        max_workers
        if max_workers is not None and max_workers > 0
        else os.cpu_count() or 1
    )
    results = {}
    failures = []

    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_dir = Path(temporary_dir)
        active = {}
        next_position = 0

        def launch(position):
            result_path = temporary_dir / f'{position:08d}.result.pickle'
            error_path = temporary_dir / f'{position:08d}.error.txt'
            row = metadata.iloc[[position]].copy()
            asset = str(row.iloc[0].get('sha256', f'row {position}'))
            process = context.Process(
                target=_foreach_child,
                args=(
                    result_path,
                    error_path,
                    dataset_utils,
                    row,
                    output_dir,
                    func,
                    f'{desc}: {asset}',
                ),
            )
            process.start()
            active[position] = {
                'asset': asset,
                'process': process,
                'result_path': result_path,
                'error_path': error_path,
                'deadline': time.monotonic() + timeout_seconds,
            }

        try:
            while next_position < len(metadata) or active:
                while (
                    next_position < len(metadata)
                    and len(active) < worker_limit
                ):
                    launch(next_position)
                    next_position += 1

                progressed = False
                now = time.monotonic()
                for position, state in list(active.items()):
                    process = state['process']
                    process.join(0)
                    if not process.is_alive():
                        exit_code = process.exitcode
                        process.close()
                        del active[position]
                        progressed = True
                        if state['error_path'].exists():
                            failures.append((
                                position,
                                RuntimeError,
                                f"{state['asset']}: "
                                f"{state['error_path'].read_text()}",
                            ))
                        elif exit_code != 0 or not state['result_path'].exists():
                            failures.append((
                                position,
                                RuntimeError,
                                f"{state['asset']}: worker exited with code "
                                f'{exit_code}',
                            ))
                        else:
                            try:
                                with state['result_path'].open('rb') as stream:
                                    results[position] = pickle.load(stream)
                            except Exception as error:
                                failures.append((
                                    position,
                                    RuntimeError,
                                    f"{state['asset']}: invalid worker result: "
                                    f'{error}',
                                ))
                    elif now >= state['deadline']:
                        _terminate_process(process)
                        process.close()
                        del active[position]
                        progressed = True
                        failures.append((
                            position,
                            TimeoutError,
                            f"{state['asset']}: timed out after "
                            f'{timeout_seconds} seconds',
                        ))

                if not progressed and active:
                    next_deadline = min(
                        state['deadline'] for state in active.values()
                    )
                    time.sleep(min(0.01, max(0, next_deadline - now)))
        finally:
            for state in active.values():
                process = state['process']
                if process.is_alive():
                    _terminate_process(process)
                process.close()

    if failures:
        failures.sort(key=lambda failure: failure[0])
        message = f'{desc} failed: ' + '; '.join(
            failure[2] for failure in failures
        )
        error_type = (
            TimeoutError
            if any(failure[1] is TimeoutError for failure in failures)
            else RuntimeError
        )
        raise error_type(message)

    ordered_results = [results[position] for position in range(len(metadata))]
    if not ordered_results:
        return pd.DataFrame()
    if all(isinstance(result, pd.DataFrame) for result in ordered_results):
        return pd.concat(ordered_results, ignore_index=True)
    return ordered_results[0] if len(ordered_results) == 1 else ordered_results


# ==================== PBR-specific transform functions ====================

def transform_vertices(vertices, frame):
    """
    Apply multi-view transform to vertices based on camera transform matrix.

    Args:
        vertices: torch.Tensor, shape [N, 3], vertex coordinates
        frame: dict containing transform_matrix

    Returns:
        transformed_vertices: torch.Tensor, shape [N, 3]
    """
    device = vertices.device
    c2w_orig = torch.tensor(frame['transform_matrix'], dtype=torch.float32, device=device)

    # Old and new camera matrices
    radius = c2w_orig[:3, 3].norm().item()
    c2w_new = get_new_camera_matrix(radius=radius, yaw=-90/180.0*math.pi, pitch=0.0,
                                dtype=torch.float32, device=device)
    w2c_orig = torch.inverse(c2w_orig)

    # Initial and final axis alignment matrices
    R_init = torch.tensor([
        [1.0, 0.0,  0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0,  0.0, 0.0],
        [0.0, 0.0,  0.0, 1.0]
    ], dtype=torch.float32, device=device)

    R_back = torch.tensor([
        [1.0,  0.0, 0.0, 0.0],
        [0.0,  0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0,  0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=device)

    R_ply = torch.tensor([
        [1.0,  0.0, 0.0, 0.0],
        [0.0,  0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0,  0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=device)

    T_cam = c2w_new @ w2c_orig @ R_ply
    T_final = R_back @ T_cam @ R_init

    # Apply transform
    vertices = vertices.reshape(-1, 3)
    verts_h = torch.cat([vertices, torch.ones((vertices.shape[0], 1), dtype=torch.float32, device=device)], dim=1)
    verts_trans = (T_final @ verts_h.T).T[:, :3]

    return verts_trans


def transform_normals(normals, frame):
    """
    Apply multi-view transform to normals (rotation only).
    Consistent with test_ovoxel_pbr_transform.py implementation.

    Args:
        normals: torch.Tensor or np.ndarray, shape [N, 3] or [N, 3, 3]
        frame: dict containing transform_matrix

    Returns:
        transformed_normals: np.ndarray (always returns numpy for dump compatibility)
    """
    is_numpy = isinstance(normals, np.ndarray)
    if is_numpy:
        normals = torch.from_numpy(normals).float()

    device = normals.device
    original_shape = normals.shape

    # Flatten to [N, 3] for processing
    if len(original_shape) == 3:
        normals_flat = normals.reshape(-1, 3)
    else:
        normals_flat = normals

    c2w_orig = torch.tensor(frame['transform_matrix'], dtype=torch.float32, device=device)

    # Old and new camera matrices
    radius = c2w_orig[:3, 3].norm().item()
    c2w_new = get_new_camera_matrix(radius=radius, yaw=-90/180.0*math.pi, pitch=0.0,
                                dtype=torch.float32, device=device)
    w2c_orig = torch.inverse(c2w_orig)

    # Axis alignment matrices (rotation part only, 3x3)
    R_init = torch.tensor([
        [1.0, 0.0,  0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0,  0.0]
    ], dtype=torch.float32, device=device)

    R_back = torch.tensor([
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
        [0.0, -1.0, 0.0]
    ], dtype=torch.float32, device=device)

    R_ply = torch.tensor([
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
        [0.0, -1.0, 0.0]
    ], dtype=torch.float32, device=device)

    # Use rotation part only
    T_cam_rot = c2w_new[:3, :3] @ w2c_orig[:3, :3] @ R_ply
    T_final_rot = R_back @ T_cam_rot @ R_init

    # Apply rotation transform
    normals_trans = torch.matmul(normals_flat, T_final_rot.T)

    # Re-normalize
    normals_trans = torch.nn.functional.normalize(normals_trans, dim=-1)

    # Restore original shape
    if len(original_shape) == 3:
        normals_trans = normals_trans.reshape(original_shape)

    # Always return numpy array for dump compatibility
    return normals_trans.numpy()


def prepare_pbr_dump(dump):
    """
    Prepare PBR dump data for processing.
    Consistent with voxelize_pbr.py preprocessing.

    Args:
        dump: raw PBR dump data

    Returns:
        processed dump data
    """
    dump = copy.deepcopy(dump)

    # Fix dump alpha map
    for mat in dump['materials']:
        if mat['alphaTexture'] is not None and mat['alphaMode'] == 'OPAQUE':
            mat['alphaMode'] = 'BLEND'

    # Append default material
    dump['materials'].append({
        "baseColorFactor": [0.8, 0.8, 0.8],
        "alphaFactor": 1.0,
        "metallicFactor": 0.0,
        "roughnessFactor": 0.5,
        "alphaMode": "OPAQUE",
        "alphaCutoff": 0.5,
        "baseColorTexture": None,
        "alphaTexture": None,
        "metallicTexture": None,
        "roughnessTexture": None,
    })

    # Filter out empty objects
    dump['objects'] = [
        obj for obj in dump['objects']
        if obj['vertices'].size != 0 and obj['faces'].size != 0
    ]

    return dump


def transform_pbr_dump(dump, frame):
    """
    Apply multi-view transform to entire PBR dump data.

    Processing flow (based on test_ovoxel_pbr_transform.py):
    1. Box normalize all vertices (scale only, no center shift)
    2. Sphere normalize
    3. Apply multi-view transform
    4. Normalize back to [-0.5, 0.5]^3

    Note: All object vertices are processed together (not per-object) for consistency.

    Args:
        dump: PBR dump data (already preprocessed via prepare_pbr_dump)
        frame: camera frame info

    Returns:
        transformed_dump: transformed dump data
        total_scale: total scale from original mesh to final mesh
    """
    transformed_dump = copy.deepcopy(dump)

    # 1. Collect all vertices
    all_vertices_list = []
    vertex_counts = []
    for obj in transformed_dump['objects']:
        all_vertices_list.append(obj['vertices'])
        vertex_counts.append(len(obj['vertices']))

    if len(all_vertices_list) == 0:
        return transformed_dump, 1.0

    all_vertices = np.concatenate(all_vertices_list, axis=0)

    # 2. Box normalize (scale only, no center shift, consistent with original rendering)
    vertices_min = all_vertices.min(axis=0)
    vertices_max = all_vertices.max(axis=0)
    box_scale_init = 0.99999 / (vertices_max - vertices_min).max()
    all_vertices_box_normalized = all_vertices * box_scale_init

    all_vertices_tensor = torch.from_numpy(all_vertices_box_normalized).float()

    # 3. Sphere normalize all vertices together
    all_vertices_sphere, sphere_center, sphere_radius = sphere_normalize_torch(all_vertices_tensor)

    # 4. Multi-view transform
    all_transformed = transform_vertices(all_vertices_sphere, frame)

    # 5. Normalize back to [-0.5, 0.5]^3 (all vertices together)
    abs_max = all_transformed.abs().max().item()
    box_scale_final = 0.49999 / abs_max
    all_transformed_normalized = all_transformed * box_scale_final

    # Compute total scale (from original mesh to final normalized mesh)
    total_scale = box_scale_init * box_scale_final / sphere_radius.item()

    # 6. Split back to individual objects
    start_idx = 0
    for i, obj in enumerate(transformed_dump['objects']):
        end_idx = start_idx + vertex_counts[i]
        obj['vertices'] = all_transformed_normalized[start_idx:end_idx].numpy()
        start_idx = end_idx

        # Transform normals
        if obj['normals'] is not None and obj['normals'].size > 0:
            obj['normals'] = transform_normals(obj['normals'], frame)

        # Fix mat_ids (replace -1 with default material index)
        obj['mat_ids'][obj['mat_ids'] == -1] = len(transformed_dump['materials']) - 1

        # Validate range
        assert np.all(obj['mat_ids'] >= 0), 'invalid mat_ids'
        assert np.all(obj['vertices'] >= -0.5) and np.all(obj['vertices'] <= 0.5), 'vertices out of range'

    return transformed_dump, total_scale


def _pbr_voxelize_view(
    file,
    sha256,
    pbr_dump_root,
    transform_root,
    root,
    resolutions,
    native_threads,
    view_indices=None,
):
    """
    Process multi-view PBR voxelization for a single sha256.

    Args:
        file: local_path from metadata
        sha256: sha256 string
        pbr_dump_root: directory containing PBR dump files
        transform_root: directory containing transform json files
        root: output directory for PBR voxels
        view_indices: list of view indices to process, None for all views
    """
    try:
        pack = {'sha256': sha256}
        dump = None

        # Load transforms
        transform_path = os.path.join(transform_root, sha256, 'transforms.json')
        if not os.path.exists(transform_path):
            print(f'Transform file not found for {sha256}, skipping')
            return {'sha256': sha256, 'error': 'Transform file not found'}

        with open(transform_path, 'r') as f:
            transforms_json = json.load(f)
        transform_mats = transforms_json['frames']

        # Determine views to process
        if view_indices is None:
            view_indices = list(range(len(transform_mats)))
        else:
            view_indices = [i for i in view_indices if i < len(transform_mats)]

        # Track processed and skipped counts
        processed_count = 0
        skipped_count = 0

        for view_idx in view_indices:
            for res in resolutions:
                need_process = False

                # Check if already processed
                # Path structure: pbr_voxels_view_fix_{res}/{sha256}/view{idx:02d}.vxz
                sha256_dir = os.path.join(root, f'pbr_voxels_view_fix_{res}', sha256)
                vxz_path = os.path.join(sha256_dir, f'view{view_idx:02d}.vxz')
                scale_path = os.path.join(sha256_dir, f'view{view_idx:02d}_scale.json')
                if os.path.exists(vxz_path):
                    try:
                        info = _read_vxz_output(vxz_path, scale_path)
                        pack[f'pbr_voxelized_view_fix{view_idx:02d}_{res}'] = True
                        pack[f'num_pbr_voxels_view_fix{view_idx:02d}_{res}'] = info['num_voxel']
                        skipped_count += 1
                    except Exception as e:
                        print(f'Error reading {sha256}/view{view_idx:02d}.vxz: {e}, will reprocess')
                        Path(vxz_path).unlink(missing_ok=True)
                        Path(scale_path).unlink(missing_ok=True)
                        need_process = True
                else:
                    need_process = True

                # Process PBR dump
                if need_process:
                    # Lazy load dump
                    if dump is None:
                        pbr_dump_file = os.path.join(pbr_dump_root, 'pbr_dumps', f'{sha256}.pickle')
                        if not os.path.exists(pbr_dump_file):
                            print(f'PBR dump not found for {sha256}, skipping')
                            return {'sha256': sha256, 'error': 'PBR dump not found'}

                        with open(pbr_dump_file, 'rb') as f:
                            dump = pickle.load(f)

                        # Prepare dump data
                        dump = prepare_pbr_dump(dump)

                        if len(dump['objects']) == 0:
                            print(f'No valid objects in PBR dump for {sha256}, skipping')
                            return {'sha256': sha256, 'error': 'No valid objects in PBR dump'}

                    # Get transform for current view
                    frame = transform_mats[view_idx]

                    # Multi-view transform (deep copy from original dump each time)
                    transformed_dump, total_scale = transform_pbr_dump(dump, frame)

                    # PBR voxelization
                    coord, attr = o_voxel.convert.blender_dump_to_volumetric_attr(
                        transformed_dump,
                        grid_size=res,
                        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        mip_level_offset=0,
                        verbose=False,
                        timing=False
                    )

                    # Remove normal and emissive (consistent with voxelize_pbr.py)
                    del attr['normal']
                    del attr['emissive']

                    # Save .vxz file
                    os.makedirs(sha256_dir, exist_ok=True)
                    scale_info = {
                        'view_idx': view_idx,
                        'total_scale': float(total_scale),
                    }
                    _publish_vxz_pair(
                        vxz_path,
                        scale_path,
                        scale_info,
                        coord,
                        attr,
                        native_threads=native_threads,
                    )

                    pack[f'pbr_voxelized_view_fix{view_idx:02d}_{res}'] = True
                    pack[f'num_pbr_voxels_view_fix{view_idx:02d}_{res}'] = len(coord)
                    pack[f'pbr_voxel_scale_view_fix{view_idx:02d}_{res}'] = float(total_scale)
                    processed_count += 1

        # Record processing stats
        pack['_processed_count'] = processed_count
        pack['_skipped_count'] = skipped_count

        return pack

    except Exception as e:
        print(f'Error processing {sha256}: {e}')
        import traceback
        traceback.print_exc()
        return {'sha256': sha256, 'error': str(e)}


if __name__ == '__main__':
    dataset_name = (
        sys.argv[1]
        if len(sys.argv) > 1 and not sys.argv[1].startswith('-')
        else None
    )
    dataset_utils = (
        importlib.import_module(f'datasets.{dataset_name}')
        if dataset_name is not None
        else None
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--pbr_dump_root', type=str, default=None,
                        help='Directory to load PBR dumps')
    parser.add_argument('--transform_root', type=str, default=None,
                        help='Directory to load transform json files (renders_cond)')
    parser.add_argument('--pbr_voxel_root', type=str, default=None,
                        help='Directory to save voxelized PBR attributes')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--view_indices', type=str, default=None,
                        help='View indices to process, e.g., "0,1,2" or "0-5". None for all views')
    parser.add_argument('--skip_list', type=str, default=None,
                        help='Path to a file containing sha256 hashes to skip (one per line). '
                             'Supports format: "sha256" or "dataset/sha256"')
    parser.add_argument('--clean_pbr_dir', type=str, default=None,
                        help='Path to clean_pbr directory. Will auto-load {dataset}_clean_output.txt as ok-list, '
                             'only sha256 in ok-list will be processed')
    parser.add_argument('--clean_pbr_name', type=str, default=None,
                        help='Dataset name prefix for clean_pbr file (e.g., ObjaverseXL_github). '
                             'Defaults to sys.argv[1] if not specified')
    if dataset_utils is not None:
        dataset_utils.add_args(parser)
    parser.add_argument('--resolution', type=str, default='1024')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=0)
    parser.add_argument('--native_threads', type=int, default=4)
    parser.add_argument('--timeout_seconds', type=int, default=900)
    opt = parser.parse_args(sys.argv[2:] if dataset_name is not None else sys.argv[1:])
    if dataset_utils is None:
        parser.error('dataset name is required')
    if opt.native_threads <= 0:
        parser.error('--native_threads must be positive')
    if opt.timeout_seconds <= 0:
        parser.error('--timeout_seconds must be positive')
    opt = edict(vars(opt))
    opt.resolution = sorted([int(x) for x in opt.resolution.split(',')], reverse=True)
    opt.pbr_dump_root = opt.pbr_dump_root or opt.root
    opt.transform_root = opt.transform_root or os.path.join(opt.root, 'renders_cond')
    opt.pbr_voxel_root = opt.pbr_voxel_root or opt.root

    # Parse view_indices
    view_indices = None
    if opt.view_indices is not None:
        view_indices = []
        for part in opt.view_indices.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                view_indices.extend(range(start, end + 1))
            else:
                view_indices.append(int(part))
        view_indices = list(set(view_indices))  # Deduplicate
        view_indices.sort()

    # Load skip list (sha256 hashes to skip)
    skip_set = set()
    if opt.skip_list is not None and os.path.exists(opt.skip_list):
        with open(opt.skip_list, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Support "dataset/sha256" and plain "sha256" format, extract pure sha256
                    skip_set.add(line.split('/')[-1])
        print(f'Loaded {len(skip_set)} items from skip_list: {opt.skip_list}')

    # Load clean_pbr ok-list (only process approved sha256)
    ok_set = None
    if opt.clean_pbr_dir is not None:
        dataset_name = opt.clean_pbr_name or sys.argv[1]
        clean_file = os.path.join(opt.clean_pbr_dir, f'{dataset_name}_clean_output.txt')
        if os.path.exists(clean_file):
            ok_set = set()
            with open(clean_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        ok_set.add(line.split('/')[-1])
            print(f'Loaded {len(ok_set)} ok items from clean_pbr: {clean_file}')
        else:
            print(f'Warning: clean_pbr file not found: {clean_file}, proceeding without ok-list filter')

    for res in opt.resolution:
        os.makedirs(os.path.join(opt.pbr_voxel_root, f'pbr_voxels_view_fix_{res}', 'new_records'), exist_ok=True)

    # Get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.root, 'aesthetic_scores','metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.pbr_dump_root, 'pbr_dumps', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.pbr_dump_root, 'pbr_dumps', 'metadata.csv')).set_index('sha256'))

    # Check already processed pbr_voxels_view_fix
    for res in opt.resolution:
        if os.path.exists(os.path.join(opt.pbr_voxel_root, f'pbr_voxels_view_fix_{res}', 'metadata.csv')):
            pbr_voxel_metadata = pd.read_csv(os.path.join(opt.pbr_voxel_root, f'pbr_voxels_view_fix_{res}', 'metadata.csv')).set_index('sha256')
            metadata = metadata.combine_first(pbr_voxel_metadata)

    metadata = metadata.reset_index()

    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata['pbr_dumped'] == True]

    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    # Apply skip_list filter (exclude specified sha256)
    if skip_set:
        before_skip = len(metadata)
        metadata = metadata[~metadata['sha256'].isin(skip_set)]
        print(f'Skip list: filtered out {before_skip - len(metadata)} objects, {len(metadata)} remaining')

    # Apply clean_pbr ok-list filter (only keep approved sha256)
    if ok_set is not None:
        before_ok = len(metadata)
        metadata = metadata[metadata['sha256'].isin(ok_set)]
        print(f'Ok list: kept {len(metadata)} objects out of {before_ok} (filtered {before_ok - len(metadata)})')

    metadata = metadata.sample(frac=1, random_state=444).reset_index(drop=True)
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]

    print(f'Processing {len(metadata)} objects...')
    if view_indices:
        print(f'View indices to process: {view_indices}')
    else:
        print('Processing all available views')

    # Process objects
    func = partial(_pbr_voxelize_view,
                   pbr_dump_root=opt.pbr_dump_root,
                   transform_root=opt.transform_root,
                   root=opt.pbr_voxel_root,
                   resolutions=opt.resolution,
                   native_threads=opt.native_threads,
                   view_indices=view_indices)
    pbr_voxelized = _run_foreach_bounded(
        dataset_utils,
        metadata,
        opt.root,
        func,
        max_workers=opt.max_workers,
        desc='Voxelizing PBR views',
        timeout_seconds=opt.timeout_seconds,
    )

    # Processing summary
    total_processed = pbr_voxelized['_processed_count'].sum() if '_processed_count' in pbr_voxelized.columns else 0
    total_skipped = pbr_voxelized['_skipped_count'].sum() if '_skipped_count' in pbr_voxelized.columns else 0
    print(f'\n========== Processing Summary ==========')
    print(f'Total processed (new): {int(total_processed)}')
    print(f'Total skipped (existing): {int(total_skipped)}')
    print(f'Total items: {int(total_processed + total_skipped)}')
    print(f'=========================================\n')

    if 'error' in pbr_voxelized.columns:
        errors = pbr_voxelized[pbr_voxelized['error'].notna()]
        if len(errors) > 0:
            with open('errors_pbr_view.txt', 'w') as f:
                f.write('\n'.join(errors['sha256'].tolist()))
            print(f'Errors written to errors_pbr_view.txt ({len(errors)} errors)')

    # Save metadata
    for res in opt.resolution:
        # Collect all view-related columns
        view_cols = [col for col in pbr_voxelized.columns if f'pbr_voxelized_view_fix' in col and f'_{res}' in col]
        if view_cols:
            # Save metadata for each view
            pbr_voxel_metadata = pbr_voxelized[pbr_voxelized[view_cols].any(axis=1)]
            if len(pbr_voxel_metadata) > 0:
                # Save simplified metadata
                cols_to_save = ['sha256'] + [col for col in pbr_voxelized.columns if f'_{res}' in col]
                cols_to_save = [col for col in cols_to_save if col in pbr_voxelized.columns]
                _atomic_write_csv(
                    pbr_voxel_metadata[cols_to_save],
                    Path(opt.pbr_voxel_root) / f'pbr_voxels_view_fix_{res}' / 'new_records' / f'part_{opt.rank}.csv',
                )

    print('Done!')
