"""
dual_grid_view.py - Multi-view transform dual grid processing
Extends dual_grid.py with scale and mesh rotation logic
Based on test_ovoxel_transform.py implementation
"""
import os
import sys
import importlib
import argparse
import inspect
import json
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np
import torch
import pickle
import o_voxel
from easydict import EasyDict as edict
from functools import partial

if __package__:
    from .utils import get_new_camera_matrix, transform_mesh, sphere_normalize_torch
    from .pipeline.atomic_io import atomic_write_json
    from .pipeline.validation import validate_scale
else:
    from utils import get_new_camera_matrix, transform_mesh, sphere_normalize_torch
    from pipeline.atomic_io import atomic_write_json
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
        return o_voxel.io.read_vxz_info(str(path)) or info
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


def _dual_grid_mesh_view(
    file,
    sha256,
    mesh_dump_root,
    transform_root,
    root,
    resolutions,
    native_threads,
    view_indices=None,
):
    """
    Process multi-view dual grid conversion for a single sha256.
    
    Args:
        file: local_path from metadata
        sha256: sha256 string
        mesh_dump_root: directory containing mesh dump files
        transform_root: directory containing transform json files
        root: output directory for dual grids
        view_indices: list of view indices to process, None for all views
    """
    try:
        pack = {'sha256': sha256}
        vertices_sphere = None
        sphere_radius = None
        faces = None
        
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
                # Path structure: dual_grid_view_{res}/{sha256}/view{idx:02d}.vxz
                sha256_dir = os.path.join(root, f'dual_grid_view_{res}', sha256)
                vxz_path = os.path.join(sha256_dir, f'view{view_idx:02d}.vxz')
                scale_path = os.path.join(sha256_dir, f'view{view_idx:02d}_scale.json')
                if os.path.exists(vxz_path):
                    try:
                        info = _read_vxz_output(vxz_path, scale_path)
                        pack[f'dual_grid_view{view_idx:02d}_converted_{res}'] = True
                        pack[f'dual_grid_view{view_idx:02d}_size_{res}'] = info['num_voxel']
                        skipped_count += 1
                    except Exception as e:
                        print(f'Error reading {sha256}/view{view_idx:02d}.vxz: {e}')
                        Path(vxz_path).unlink(missing_ok=True)
                        Path(scale_path).unlink(missing_ok=True)
                        need_process = True
                else:
                    need_process = True
                
                # Process mesh
                if need_process:
                    # Lazy load mesh
                    if vertices_sphere is None:
                        mesh_file = os.path.join(mesh_dump_root, 'mesh_dumps', f'{sha256}.pickle')
                        if not os.path.exists(mesh_file):
                            print(f'Mesh dump not found for {sha256}, skipping')
                            return {'sha256': sha256, 'error': 'Mesh dump not found'}
                        
                        with open(mesh_file, 'rb') as f:
                            dump = pickle.load(f)
                        
                        start = 0
                        vertices_list = []
                        faces_list = []
                        for obj in dump['objects']:
                            if obj['vertices'].size == 0 or obj['faces'].size == 0:
                                continue
                            vertices_list.append(obj['vertices'])
                            faces_list.append(obj['faces'] + start)
                            start += len(obj['vertices'])
                        
                        if len(vertices_list) == 0:
                            print(f'No valid mesh data for {sha256}, skipping')
                            return {'sha256': sha256, 'error': 'No valid mesh data'}
                        
                        vertices = torch.from_numpy(np.concatenate(vertices_list, axis=0)).float().contiguous()
                        faces = torch.from_numpy(np.concatenate(faces_list, axis=0)).long().contiguous()
                        
                        # Sphere normalization (for multi-view transform) - CPU only
                        vertices_sphere, sphere_center, sphere_radius = sphere_normalize_torch(vertices)
                    
                    # Get transform for current view
                    transform = transform_mats[view_idx]
                    
                    # Multi-view transform - CPU only
                    transformed_vertices = transform_mesh(vertices_sphere, transform)
                    
                    # Post-transform normalization: scale by abs max to [-0.5, 0.5]^3
                    # Only scale, no center shift, to preserve relative model position
                    abs_max = transformed_vertices.abs().max().item()
                    box_scale = 0.49999 / abs_max  # Normalize to [-0.5, 0.5] range
                    transformed_normalized = transformed_vertices * box_scale
                    transformed_normalized_cpu = transformed_normalized.contiguous()
                    
                    # Compute total scale (from original mesh to final normalized mesh)
                    total_scale = box_scale / sphere_radius.item()
                    
                    # Validate range
                    assert torch.all(transformed_normalized_cpu >= -0.5) and torch.all(transformed_normalized_cpu <= 0.5), \
                        f'vertices out of range for {sha256} view {view_idx}'
                    
                    # Ensure vertices and faces are on CPU with correct types and contiguous memory
                    # CPU only, consistent with process_dual_grid in test_ovoxel_transform.py
                    vertices_for_grid = transformed_normalized_cpu.float().contiguous()
                    faces_for_grid = faces.long().contiguous()
                    data_for_grid = {'vertices': vertices_for_grid, 'faces': faces_for_grid}
                    
                    # Dual grid encoding
                    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
                        **data_for_grid,
                        grid_size=res,
                        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        face_weight=1.0,
                        boundary_weight=0.2,
                        regularization_weight=1e-2,
                        timing=False,
                    )
                    
                    # Convert to intra-voxel offsets and quantize
                    dual_vertices = dual_vertices.float()
                    voxel_indices_float = voxel_indices.float()
                    dual_vertices = dual_vertices * res - voxel_indices_float
                    assert torch.all(dual_vertices >= -1e-3) and torch.all(dual_vertices <= 1+1e-3), \
                        f'dual_vertices out of range for {sha256} view {view_idx}'
                    dual_vertices = torch.clamp(dual_vertices, 0, 1)
                    dual_vertices = (dual_vertices * 255).type(torch.uint8)
                    intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)
                    
                    # Save .vxz file
                    os.makedirs(sha256_dir, exist_ok=True)
                    _atomic_write_vxz(
                        vxz_path,
                        voxel_indices,
                        {'vertices': dual_vertices, 'intersected': intersected},
                        native_threads=native_threads,
                    )
                    
                    # Save scale info
                    scale_info = {
                        'view_idx': view_idx,
                        'total_scale': total_scale,
                        'sphere_radius': sphere_radius.item(),
                        'box_scale': box_scale,
                    }
                    atomic_write_json(Path(scale_path), scale_info)
                    validate_scale(Path(scale_path))
                    
                    pack[f'dual_grid_view{view_idx:02d}_converted_{res}'] = True
                    pack[f'dual_grid_view{view_idx:02d}_size_{res}'] = len(voxel_indices)
                    pack[f'dual_grid_view{view_idx:02d}_scale_{res}'] = total_scale
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
    parser.add_argument('--mesh_dump_root', type=str, default=None,
                        help='Directory to load mesh dumps')
    parser.add_argument('--transform_root', type=str, default=None,
                        help='Directory to load transform json files (renders_cond)')
    parser.add_argument('--dual_grid_root', type=str, default=None,
                        help='Directory to save dual grids')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--view_indices', type=str, default=None,
                        help='View indices to process, e.g., "0,1,2" or "0-5". None for all views')
    if dataset_utils is not None:
        dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--resolution', type=str, default='256')
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
    opt.resolution = [int(x) for x in opt.resolution.split(',')]
    opt.mesh_dump_root = opt.mesh_dump_root or opt.root
    opt.transform_root = opt.transform_root or os.path.join(opt.root, 'renders_cond')
    opt.dual_grid_root = opt.dual_grid_root or opt.root
    
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

    for res in opt.resolution:
        os.makedirs(os.path.join(opt.dual_grid_root, f'dual_grid_view_{res}', 'new_records'), exist_ok=True)

    # Get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.root, 'aesthetic_scores','metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.mesh_dump_root, 'mesh_dumps', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.mesh_dump_root, 'mesh_dumps', 'metadata.csv')).set_index('sha256'))
    
    # Check already processed dual_grid_view
    for res in opt.resolution:
        if os.path.exists(os.path.join(opt.dual_grid_root, f'dual_grid_view_{res}', 'metadata.csv')):
            dual_grid_metadata = pd.read_csv(os.path.join(opt.dual_grid_root, f'dual_grid_view_{res}', 'metadata.csv')).set_index('sha256')
            metadata = metadata.combine_first(dual_grid_metadata)
    
    metadata = metadata.reset_index()
    
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata['mesh_dumped'] == True]
        
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
    
    print(f'Processing {len(metadata)} objects...')
    if view_indices:
        print(f'View indices to process: {view_indices}')
    else:
        print('Processing all available views')

    # Process objects
    func = partial(_dual_grid_mesh_view, 
                   root=opt.dual_grid_root, 
                   mesh_dump_root=opt.mesh_dump_root,
                   transform_root=opt.transform_root,
                   resolutions=opt.resolution,
                   native_threads=opt.native_threads,
                   view_indices=view_indices)
    foreach_kwargs = {
        'max_workers': opt.max_workers,
        'desc': 'Dual griding views',
    }
    if 'timeout' in inspect.signature(dataset_utils.foreach_instance).parameters:
        foreach_kwargs['timeout'] = opt.timeout_seconds
    dual_grids = dataset_utils.foreach_instance(
        metadata, opt.root, func, **foreach_kwargs
    )
    
    # Processing summary
    total_processed = dual_grids['_processed_count'].sum() if '_processed_count' in dual_grids.columns else 0
    total_skipped = dual_grids['_skipped_count'].sum() if '_skipped_count' in dual_grids.columns else 0
    print(f'\n========== Processing Summary ==========')
    print(f'Total processed (new): {int(total_processed)}')
    print(f'Total skipped (existing): {int(total_skipped)}')
    print(f'Total items: {int(total_processed + total_skipped)}')
    print(f'=========================================\n')
    
    if 'error' in dual_grids.columns:
        errors = dual_grids[dual_grids['error'].notna()]
        if len(errors) > 0:
            with open('errors_view.txt', 'w') as f:
                f.write('\n'.join(errors['sha256'].tolist()))
            print(f'Errors written to errors_view.txt ({len(errors)} errors)')
    
    # Save metadata
    for res in opt.resolution:
        # Collect all view-related columns
        view_cols = [col for col in dual_grids.columns if f'dual_grid_view' in col and f'_{res}' in col and 'converted' in col]
        if view_cols:
            # Save metadata for each view
            dual_grid_metadata = dual_grids[dual_grids[view_cols].any(axis=1)]
            if len(dual_grid_metadata) > 0:
                # Save simplified metadata
                cols_to_save = ['sha256'] + [col for col in dual_grids.columns if f'_{res}' in col]
                cols_to_save = [col for col in cols_to_save if col in dual_grids.columns]
                _atomic_write_csv(
                    dual_grid_metadata[cols_to_save],
                    Path(opt.dual_grid_root) / f'dual_grid_view_{res}' / 'new_records' / f'part_{opt.rank}.csv',
                )
    
    print('Done!')
