import os
import sys
import time
import glob
import importlib
import argparse
from pathlib import Path
import tempfile
import pandas as pd
from easydict import EasyDict as edict


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


def _list_optional_directory(path):
    path = Path(path)
    return sorted(os.listdir(path)) if path.is_dir() else []


def update_metadata(path, opt):
    path = Path(path)
    if not path.exists():
        return None
    timestamp = str(int(time.time()))
    merged_records = path / 'merged_records'
    new_records = path / 'new_records'
    merged_records.mkdir(parents=True, exist_ok=True)
    new_records.mkdir(parents=True, exist_ok=True)
    if opt.from_merged_records:
        source_directory = merged_records
        record_start = opt.record_start if opt.record_start is not None else 0
        df_files = []
        for file_name in _list_optional_directory(source_directory):
            if not file_name.endswith('.csv'):
                continue
            try:
                if int(file_name.split('_')[0]) >= record_start:
                    df_files.append(file_name)
            except ValueError:
                print(f'Skipping merged record with invalid timestamp: {file_name}')
    else:
        source_directory = new_records
        df_files = [
            file_name
            for file_name in _list_optional_directory(source_directory)
            if file_name.startswith('part_') and file_name.endswith('.csv')
        ]
    df_parts = []
    for f in df_files:
        try:
            df_parts.append(pd.read_csv(source_directory / f))
        except Exception as e:
            print(f"Failed to read {f}: {e}")
    if len(df_parts) > 0:
        metadata_path = path / 'metadata.csv'
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path)
        else:
            columns = df_parts[0].columns
            metadata = pd.DataFrame(columns=columns)
        metadata.set_index('sha256', inplace=True)
        if metadata.index.duplicated().any():
            metadata = metadata.groupby(level=0).first()
        for df_part in df_parts:
            if 'sha256' in df_part.columns:
                df_part.set_index('sha256', inplace=True)
                if df_part.index.duplicated().any():
                    df_part = df_part.groupby(level=0).first()
                metadata = df_part.combine_first(metadata)
        _atomic_write_csv(metadata.reset_index(), metadata_path)
        if not opt.from_merged_records:
            for f in df_files:
                os.replace(new_records / f, merged_records / f'{timestamp}_{f}')
        return metadata
    else:
        metadata_path = path / 'metadata.csv'
        if metadata_path.exists():
            return pd.read_csv(metadata_path)
    return None


def build_downloaded_metadata_from_files(raw_root, global_metadata):
    """Scan local files under raw_root to build download metadata.
    
    Walks through raw_root to find downloaded 3D files (.glb, .obj, .fbx, .usdz, .gltf, .zip),
    matches them against global_metadata via file_identifier (uid extracted from URL) to recover
    the sha256 -> local_path mapping.
    """
    extensions = ('.glb', '.obj', '.fbx', '.usdz', '.gltf', '.zip')
    
    # Build uid -> sha256 mapping from global metadata
    uid_to_sha256 = {}
    if 'file_identifier' in global_metadata.columns:
        for _, row in global_metadata.iterrows():
            uid = str(row['file_identifier']).split('/')[-1]
            uid_to_sha256[uid] = row['sha256']
    
    # Scan files
    records = []
    for dirpath, dirnames, filenames in os.walk(raw_root):
        for fname in filenames:
            if not fname.lower().endswith(extensions):
                continue
            uid = os.path.splitext(fname)[0]
            sha256 = uid_to_sha256.get(uid)
            if sha256 is not None:
                full_path = os.path.join(dirpath, fname)
                # Store path relative to parent of raw_root (i.e. download_root)
                rel_path = os.path.relpath(full_path, os.path.dirname(raw_root))
                records.append({'sha256': sha256, 'local_path': rel_path})
    
    if len(records) == 0:
        return None
    
    df = pd.DataFrame(records).set_index('sha256')
    print(f'  [from_file] Found {len(df)} downloaded files under {raw_root}')
    
    # Save as metadata.csv under raw_root
    os.makedirs(raw_root, exist_ok=True)
    _atomic_write_csv(df.reset_index(), Path(raw_root) / 'metadata.csv')
    return df


# Check if directory is a multi-view directory (ending with _view or _view_fix)
def _is_view_dir(dirname):
    return dirname.endswith('_view') or dirname.endswith('_view_fix')


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
    parser.add_argument('--download_root', type=str, default=None,
                        help='Directory to save the downloaded files')
    parser.add_argument('--thumbnail_root', type=str, default=None,
                        help='Directory to save the thumbnail files')
    parser.add_argument('--render_cond_root', type=str, default=None,
                        help='Directory to save the render condition files')
    parser.add_argument('--mesh_dump_root', type=str, default=None,
                        help='Directory to save the mesh files')
    parser.add_argument('--pbr_dump_root', type=str, default=None,
                        help='Directory to save the pbr files')
    parser.add_argument('--dual_grid_root', type=str, default=None,
                        help='Directory to save the dual grid files')
    parser.add_argument('--pbr_voxel_root', type=str, default=None,
                        help='Directory to save the pbr voxel files')
    parser.add_argument('--ss_latent_root', type=str, default=None,
                        help='Directory to save the sparse structure latent files')
    parser.add_argument('--shape_latent_root', type=str, default=None,
                        help='Directory to save the shape latent files')
    parser.add_argument('--pbr_latent_root', type=str, default=None,
                        help='Directory to save the pbr latent files')
    parser.add_argument('--field', type=str, default='all',
                        help='Fields to process, separated by commas')
    parser.add_argument('--from_file', action='store_true',
                        help='Build metadata from file instead of from records of processings.' +
                             'Useful when some processing fail to generate records but file already exists.')
    parser.add_argument('--from_merged_records', action='store_true',
                        help='Build metadata from merged records')
    parser.add_argument('--record_start', type=int)
    parser.add_argument('--rebuild', action='store_true',
                        help='Rebuild metadata from scratch, ignore existing metadata.')
    if dataset_utils is not None:
        dataset_utils.add_args(parser)
    opt = parser.parse_args(sys.argv[2:] if dataset_name is not None else sys.argv[1:])
    if dataset_utils is None:
        parser.error('dataset name is required')
    opt = edict(vars(opt))
    opt.download_root = opt.download_root or opt.root
    opt.thumbnail_root = opt.thumbnail_root or opt.root
    opt.render_cond_root = opt.render_cond_root or opt.root
    opt.mesh_dump_root = opt.mesh_dump_root or opt.root
    opt.pbr_dump_root = opt.pbr_dump_root or opt.root
    opt.dual_grid_root = opt.dual_grid_root or opt.root
    opt.pbr_voxel_root = opt.pbr_voxel_root or opt.root
    opt.ss_latent_root = opt.ss_latent_root or opt.root
    opt.shape_latent_root = opt.shape_latent_root or opt.root
    opt.pbr_latent_root = opt.pbr_latent_root or opt.root

    os.makedirs(opt.root, exist_ok=True)

    opt.field = opt.field.split(',')
    
    # get file list
    if os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        print('Loading previous metadata...')
        metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv'))
    else:
        metadata = dataset_utils.get_metadata(**opt)
        _atomic_write_csv(metadata, Path(opt.root) / 'metadata.csv')
    
    # merge downloaded
    if opt.from_file:
        downloaded_metadata = build_downloaded_metadata_from_files(
            os.path.join(opt.download_root, 'raw'), metadata)
    else:
        downloaded_metadata = update_metadata(os.path.join(opt.download_root, 'raw'), opt)

    # merge thumbnails
    thumbnail_metadata = update_metadata(os.path.join(opt.thumbnail_root, 'thumbnails'), opt)
    
    # merge aesthetic scores
    aesthetic_score_metadata = update_metadata(os.path.join(opt.root, 'aesthetic_scores'), opt)
    
    # merge render conditions
    render_cond_metadata = update_metadata(os.path.join(opt.render_cond_root, 'renders_cond'), opt)

    # merge mesh dumped
    mesh_dumped_metadata = update_metadata(os.path.join(opt.mesh_dump_root, 'mesh_dumps'), opt)
        
    # merge pbr dumped
    pbr_dumped_metadata = update_metadata(os.path.join(opt.pbr_dump_root, 'pbr_dumps'), opt)
    
    # merge asset stats
    asset_stats_metadata = update_metadata(os.path.join(opt.root, 'asset_stats'), opt)
        
    # merge dual grid (original, no view transform)
    dual_grid_resolutions = []
    for dir in _list_optional_directory(opt.dual_grid_root):
        if os.path.isdir(os.path.join(opt.dual_grid_root, dir)) and dir.startswith('dual_grid_') and not dir.startswith('dual_grid_view_'):
            dual_grid_resolutions.append(int(dir.split('_')[-1]))
    dual_grid_metadata = {}
    for res in dual_grid_resolutions:
        dual_grid_metadata[res] = update_metadata(os.path.join(opt.dual_grid_root, f'dual_grid_{res}'), opt)
    
    # merge dual grid view (multi-view)
    dual_grid_view_resolutions = []
    for dir in _list_optional_directory(opt.dual_grid_root):
        if os.path.isdir(os.path.join(opt.dual_grid_root, dir)) and dir.startswith('dual_grid_view_'):
            dual_grid_view_resolutions.append(int(dir.split('_')[-1]))
    dual_grid_view_metadata = {}
    for res in dual_grid_view_resolutions:
        dual_grid_view_metadata[res] = update_metadata(os.path.join(opt.dual_grid_root, f'dual_grid_view_{res}'), opt)
    
    # merge pbr voxelized (single view)
    pbr_voxel_resolutions = []
    for dir in _list_optional_directory(opt.pbr_voxel_root):
        if os.path.isdir(os.path.join(opt.pbr_voxel_root, dir)) and dir.startswith('pbr_voxels_') and not dir.startswith('pbr_voxels_view_'):
            pbr_voxel_resolutions.append(int(dir.split('_')[-1]))
    pbr_voxel_metadata = {}
    for res in pbr_voxel_resolutions:
        pbr_voxel_metadata[res] = update_metadata(os.path.join(opt.pbr_voxel_root, f'pbr_voxels_{res}'), opt)
    
    # merge pbr voxelized view (multi-view)
    # Supports both pbr_voxels_view_{res} and pbr_voxels_view_fix_{res} directory names
    pbr_voxel_view_dirs = {}  # res -> dir_name
    for dir in _list_optional_directory(opt.pbr_voxel_root):
        if os.path.isdir(os.path.join(opt.pbr_voxel_root, dir)) and dir.startswith('pbr_voxels_view_') and not dir.startswith('pbr_voxels_view_fix_'):
            res = int(dir.split('_')[-1])
            pbr_voxel_view_dirs[res] = dir
        elif os.path.isdir(os.path.join(opt.pbr_voxel_root, dir)) and dir.startswith('pbr_voxels_view_fix_'):
            res = int(dir.split('_')[-1])
            pbr_voxel_view_dirs[res] = dir
    pbr_voxel_view_resolutions = sorted(pbr_voxel_view_dirs.keys())
    pbr_voxel_view_metadata = {}
    for res in pbr_voxel_view_resolutions:
        pbr_voxel_view_metadata[res] = update_metadata(os.path.join(opt.pbr_voxel_root, pbr_voxel_view_dirs[res]), opt)
        
    # merge ss latents
    ss_latent_models = []
    if os.path.exists(os.path.join(opt.ss_latent_root, 'ss_latents')):
        ss_latent_models = os.listdir(os.path.join(opt.ss_latent_root, 'ss_latents'))
    ss_latent_metadata = {}
    for model in ss_latent_models:
        ss_latent_metadata[model] = update_metadata(os.path.join(opt.ss_latent_root, f'ss_latents/{model}'), opt)
        
    # merge shape latents (original, no view transform)
    shape_latent_models = []
    if os.path.exists(os.path.join(opt.shape_latent_root, 'shape_latents')):
        for dir in os.listdir(os.path.join(opt.shape_latent_root, 'shape_latents')):
            if os.path.isdir(os.path.join(opt.shape_latent_root, 'shape_latents', dir)) and not _is_view_dir(dir):
                shape_latent_models.append(dir)
    shape_latent_metadata = {}
    for model in shape_latent_models:
        shape_latent_metadata[model] = update_metadata(os.path.join(opt.shape_latent_root, f'shape_latents/{model}'), opt)
    
    # merge shape latents view (multi-view, including _view and _view_fix)
    shape_latent_view_models = []
    if os.path.exists(os.path.join(opt.shape_latent_root, 'shape_latents')):
        for dir in os.listdir(os.path.join(opt.shape_latent_root, 'shape_latents')):
            if os.path.isdir(os.path.join(opt.shape_latent_root, 'shape_latents', dir)) and _is_view_dir(dir):
                shape_latent_view_models.append(dir)
    shape_latent_view_metadata = {}
    for model in shape_latent_view_models:
        shape_latent_view_metadata[model] = update_metadata(os.path.join(opt.shape_latent_root, f'shape_latents/{model}'), opt)
        
    # merge pbr latents (single view)
    pbr_latent_models = []
    if os.path.exists(os.path.join(opt.pbr_latent_root, 'pbr_latents')):
        for dir in os.listdir(os.path.join(opt.pbr_latent_root, 'pbr_latents')):
            if os.path.isdir(os.path.join(opt.pbr_latent_root, 'pbr_latents', dir)) and not _is_view_dir(dir):
                pbr_latent_models.append(dir)
    pbr_latent_metadata = {}
    for model in pbr_latent_models:
        pbr_latent_metadata[model] = update_metadata(os.path.join(opt.pbr_latent_root, f'pbr_latents/{model}'), opt)
    
    # merge pbr latents view (multi-view, including _view and _view_fix)
    pbr_latent_view_models = []
    if os.path.exists(os.path.join(opt.pbr_latent_root, 'pbr_latents')):
        for dir in os.listdir(os.path.join(opt.pbr_latent_root, 'pbr_latents')):
            if os.path.isdir(os.path.join(opt.pbr_latent_root, 'pbr_latents', dir)) and _is_view_dir(dir):
                pbr_latent_view_models.append(dir)
    pbr_latent_view_metadata = {}
    for model in pbr_latent_view_models:
        pbr_latent_view_metadata[model] = update_metadata(os.path.join(opt.pbr_latent_root, f'pbr_latents/{model}'), opt)

    # Merge all sub-metadata back into main metadata and save
    metadata = metadata.set_index('sha256')
    sub_metadata_list = [
        downloaded_metadata,
        thumbnail_metadata,
        aesthetic_score_metadata,
        render_cond_metadata,
        mesh_dumped_metadata,
        pbr_dumped_metadata,
        asset_stats_metadata,
    ]
    for res in dual_grid_resolutions:
        sub_metadata_list.append(dual_grid_metadata.get(res))
    for res in dual_grid_view_resolutions:
        sub_metadata_list.append(dual_grid_view_metadata.get(res))
    for res in pbr_voxel_resolutions:
        sub_metadata_list.append(pbr_voxel_metadata.get(res))
    for res in pbr_voxel_view_resolutions:
        sub_metadata_list.append(pbr_voxel_view_metadata.get(res))
    for model in ss_latent_models:
        sub_metadata_list.append(ss_latent_metadata.get(model))
    for model in shape_latent_models:
        sub_metadata_list.append(shape_latent_metadata.get(model))
    for model in shape_latent_view_models:
        sub_metadata_list.append(shape_latent_view_metadata.get(model))
    for model in pbr_latent_models:
        sub_metadata_list.append(pbr_latent_metadata.get(model))
    for model in pbr_latent_view_models:
        sub_metadata_list.append(pbr_latent_view_metadata.get(model))
    if metadata.index.duplicated().any():
        metadata = metadata.groupby(level=0).first()
    for sub in sub_metadata_list:
        if sub is not None:
            if 'sha256' in sub.columns:
                sub = sub.set_index('sha256')
            if sub.index.duplicated().any():
                sub = sub.groupby(level=0).first()
            metadata = metadata.combine_first(sub)
    metadata = metadata.reset_index()
    _atomic_write_csv(metadata, Path(opt.root) / 'metadata.csv')
    print(f'Saved merged metadata with {len(metadata)} entries and columns: {list(metadata.columns)}')

    # statistics
    num_downloaded = downloaded_metadata['local_path'].count() if downloaded_metadata is not None else 0
    with open(os.path.join(opt.root, 'statistics.txt'), 'w') as f:
        f.write('Statistics:\n')
        f.write(f'  - Number of assets: {len(metadata)}\n')
        f.write(f'  - Number of assets downloaded: {num_downloaded}\n')
        if thumbnail_metadata is not None:
            f.write(f'  - Number of assets with thumbnails: {thumbnail_metadata["thumbnailed"].sum()}\n')
        if aesthetic_score_metadata is not None:
            f.write(f'  - Number of assets with aesthetic scores: {aesthetic_score_metadata["aesthetic_score"].count()}\n')
        if render_cond_metadata is not None:
            f.write(f'  - Number of assets with render conditions: {render_cond_metadata["cond_rendered"].count()}\n')
        if mesh_dumped_metadata is not None:
            f.write(f'  - Number of assets with mesh dumped: {mesh_dumped_metadata["mesh_dumped"].sum()}\n')
        if pbr_dumped_metadata is not None:
            f.write(f'  - Number of assets with PBR dumped: {pbr_dumped_metadata["pbr_dumped"].sum()}\n')
        if asset_stats_metadata is not None:
            f.write(f'  - Number of assets with asset stats: {len(asset_stats_metadata)}\n')
        if len(dual_grid_resolutions) != 0:
            f.write(f'  - Number of assets with dual grid:\n')
            for res in dual_grid_resolutions:
                if dual_grid_metadata[res] is not None:
                    f.write(f'    - {res}: {dual_grid_metadata[res]["dual_grid_converted"].sum()}\n')
        if len(dual_grid_view_resolutions) != 0:
            f.write(f'  - Number of assets with dual grid view:\n')
            for res in dual_grid_view_resolutions:
                if dual_grid_view_metadata[res] is not None:
                    col_name = f'dual_grid_view00_converted_{res}'
                    if col_name in dual_grid_view_metadata[res].columns:
                        f.write(f'    - {res}: {dual_grid_view_metadata[res][col_name].sum()}\n')
                    else:
                        f.write(f'    - {res}: {len(dual_grid_view_metadata[res])}\n')
        if len(pbr_voxel_resolutions) != 0:
            f.write(f'  - Number of assets with PBR voxelization:\n')
            for res in sorted(pbr_voxel_resolutions):
                if pbr_voxel_metadata[res] is not None:
                    f.write(f'    - {res}: {pbr_voxel_metadata[res]["pbr_voxelized"].sum()}\n')
        if len(pbr_voxel_view_resolutions) != 0:
            f.write(f'  - Number of assets with PBR voxelization view:\n')
            for res in sorted(pbr_voxel_view_resolutions):
                if pbr_voxel_view_metadata[res] is not None:
                    dir_name = pbr_voxel_view_dirs[res]
                    col_name_old = 'pbr_voxelized_view00'
                    col_name_new = f'pbr_voxelized_view_fix00_{res}'
                    if col_name_old in pbr_voxel_view_metadata[res].columns:
                        f.write(f'    - {dir_name}: {pbr_voxel_view_metadata[res][col_name_old].sum()}\n')
                    elif col_name_new in pbr_voxel_view_metadata[res].columns:
                        f.write(f'    - {dir_name}: {pbr_voxel_view_metadata[res][col_name_new].sum()}\n')
                    else:
                        f.write(f'    - {dir_name}: {len(pbr_voxel_view_metadata[res])}\n')
        if len(ss_latent_models) != 0:
            f.write(f'  - Number of assets with sparse structure latents:\n')
            for model in ss_latent_models:
                if ss_latent_metadata[model] is not None:
                    if 'ss_latent_encoded' in ss_latent_metadata[model].columns:
                        f.write(f'    - {model}: {ss_latent_metadata[model]["ss_latent_encoded"].sum()}\n')
                    elif 'ss_latent_view00_encoded' in ss_latent_metadata[model].columns:
                        f.write(f'    - {model}: {ss_latent_metadata[model]["ss_latent_view00_encoded"].sum()}\n')
                    else:
                        f.write(f'    - {model}: {len(ss_latent_metadata[model])}\n')
        if len(shape_latent_models) != 0:
            f.write(f'  - Number of assets with shape latents:\n')
            for model in shape_latent_models:
                if shape_latent_metadata[model] is not None:
                    f.write(f'    - {model}: {shape_latent_metadata[model]["shape_latent_encoded"].sum()}\n')
        if len(shape_latent_view_models) != 0:
            f.write(f'  - Number of assets with shape latents view:\n')
            for model in shape_latent_view_models:
                if shape_latent_view_metadata[model] is not None:
                    col_name = 'shape_latent_view00_encoded'
                    if col_name in shape_latent_view_metadata[model].columns:
                        f.write(f'    - {model}: {shape_latent_view_metadata[model][col_name].sum()}\n')
                    else:
                        f.write(f'    - {model}: {len(shape_latent_view_metadata[model])}\n')
        if len(pbr_latent_models) != 0:
            f.write(f'  - Number of assets with PBR latents:\n')
            for model in pbr_latent_models:
                if pbr_latent_metadata[model] is not None:
                    f.write(f'    - {model}: {pbr_latent_metadata[model]["pbr_latent_encoded"].sum()}\n')
        if len(pbr_latent_view_models) != 0:
            f.write(f'  - Number of assets with PBR latents view:\n')
            for model in pbr_latent_view_models:
                if pbr_latent_view_metadata[model] is not None:
                    col_name = 'pbr_latent_view00_encoded'
                    if col_name in pbr_latent_view_metadata[model].columns:
                        f.write(f'    - {model}: {pbr_latent_view_metadata[model][col_name].sum()}\n')
                    else:
                        f.write(f'    - {model}: {len(pbr_latent_view_metadata[model])}\n')
        
    with open(os.path.join(opt.root, 'statistics.txt'), 'r') as f:
        print(f.read())
