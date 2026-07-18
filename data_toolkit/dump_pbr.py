import os
import sys
import importlib
import argparse
import inspect
from pathlib import Path
import pickle
import subprocess
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
import tempfile


BLENDER_LINK = 'https://ftp.halifax.rwth-aachen.de/blender/release/Blender4.5/blender-4.5.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-4.5.1-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6 libxfixes3 libgl1')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-4.5.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')
    os.system(f'{BLENDER_PATH} -b --python {os.path.join(os.path.dirname(__file__), "blender_script", "install_pillow.py")}')


def _sync_parent(path):
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
    directory = os.open(Path(path).parent, flags)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _read_pickle(path):
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def _pbr_failure_record(sha256, reason):
    reason = str(reason).strip() or 'Failed to dump PBR'
    if 'Material is not supported' in reason:
        category = 'unsupported_shader'
    elif reason.startswith('PBR dump timed out'):
        category = 'timeout'
    else:
        category = 'pbr_dump_failure'
    return {
        'sha256': sha256,
        'pbr_dumped': False,
        'error_category': category,
        'error_reason': reason,
    }


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


def _dump_pbr(file_path, sha256, root, timeout_seconds=900):
    output_path = Path(root) / 'pbr_dumps' / f'{sha256}.pickle'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        try:
            _read_pickle(output_path)
            return {'sha256': sha256, 'pbr_dumped': True}
        except Exception:
            output_path.unlink(missing_ok=True)

    temporary = None
    error_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f'.{output_path.name}.',
            suffix='.pickle',
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        error_path = Path(f'{temporary}_error.txt')
        args = [
            BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'dump_pbr.py'),
            '--',
            '--object', os.path.expanduser(file_path),
            '--output_path', os.path.expanduser(temporary)
        ]
        if file_path.endswith('.blend'):
            args.insert(1, file_path)

        try:
            subprocess.run(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return _pbr_failure_record(
                sha256,
                f'PBR dump timed out after {timeout_seconds} seconds',
            )

        try:
            _read_pickle(temporary)
        except Exception:
            if error_path.exists():
                reason = error_path.read_text().strip()
                if 'Material is not supported' not in reason:
                    reason = f'Material is not supported: {reason}'
                return _pbr_failure_record(
                    sha256, reason
                )
            return _pbr_failure_record(
                sha256, f'Failed to dump PBR. File {file_path}.'
            )

        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
        _sync_parent(output_path)
        try:
            _read_pickle(output_path)
        except Exception:
            output_path.unlink(missing_ok=True)
            _sync_parent(output_path)
            raise
        return {'sha256': sha256, 'pbr_dumped': True}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if error_path is not None:
            error_path.unlink(missing_ok=True)

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
    parser.add_argument('--pbr_dump_root', type=str, default=None,
                        help='Directory to save the mesh dumps')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    if dataset_utils is not None:
        dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=0)
    parser.add_argument('--timeout_seconds', type=int, default=900)
    opt = parser.parse_args(sys.argv[2:] if dataset_name is not None else sys.argv[1:])
    if dataset_utils is None:
        parser.error('dataset name is required')
    if opt.timeout_seconds <= 0:
        parser.error('--timeout_seconds must be positive')
    opt = edict(vars(opt))
    opt.download_root = opt.download_root or opt.root
    opt.pbr_dump_root = opt.pbr_dump_root or opt.root

    os.makedirs(os.path.join(opt.pbr_dump_root, 'pbr_dumps', 'new_records'), exist_ok=True)
    
    # install blender
    print('Checking blender...', flush=True)
    _install_blender()

    # get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.root, 'aesthetic_scores','metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.download_root, 'raw', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.download_root, 'raw', 'metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.pbr_dump_root, 'pbr_dumps', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.pbr_dump_root, 'pbr_dumps', 'metadata.csv')).set_index('sha256'))
    metadata = metadata.reset_index()
    if opt.instances is None:
        metadata = metadata[metadata['local_path'].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
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

    # filter out objects that are already processed
    sha256_list = []
    for file_name in os.listdir(os.path.join(opt.pbr_dump_root, 'pbr_dumps')):
        if not file_name.endswith('.pickle'):
            continue
        path = Path(opt.pbr_dump_root) / 'pbr_dumps' / file_name
        try:
            _read_pickle(path)
            sha256_list.append(path.stem)
        except Exception as error:
            print(f'Removing corrupt PBR dump {path}: {error}')
            path.unlink(missing_ok=True)
    for sha256 in sha256_list:
        records.append({'sha256': sha256, 'pbr_dumped': True})
    print(f'Found {len(sha256_list)} dumped PBRs')
    metadata = metadata[~metadata['sha256'].isin(sha256_list)]
       
    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(
        _dump_pbr,
        root=opt.pbr_dump_root,
        timeout_seconds=opt.timeout_seconds,
    )
    foreach_kwargs = {
        'max_workers': opt.max_workers,
        'desc': 'Dumping PBR',
    }
    if 'timeout' in inspect.signature(dataset_utils.foreach_instance).parameters:
        foreach_kwargs['timeout'] = opt.timeout_seconds
    pbr_dumped = dataset_utils.foreach_instance(
        metadata, opt.download_root, func, **foreach_kwargs
    )
    pbr_dumped = pd.concat([pbr_dumped, pd.DataFrame.from_records(records)])
    if len(pbr_dumped.columns) == 0:
        pbr_dumped = pd.DataFrame(columns=['sha256', 'pbr_dumped'])
    _atomic_write_csv(
        pbr_dumped,
        Path(opt.pbr_dump_root) / 'pbr_dumps' / 'new_records' / f'part_{opt.rank}.csv',
    )
