import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import objaverse.xl as oxl


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--source', type=str, default='sketchfab',
                        help='Data source to download annotations from (github, sketchfab)')


def get_metadata(source, **kwargs):
    if source == 'sketchfab':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_sketchfab.csv")
    elif source == 'github':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_github.csv")
    else:
        raise ValueError(f"Invalid source: {source}")
    return metadata
        

def download(metadata: pd.DataFrame, output_dir: str, **kwargs) -> pd.DataFrame:
    max_workers = min(int(kwargs.get("max_workers", 8)), 8)
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)

    # download annotations
    annotations = oxl.get_annotations()
    annotations = annotations[annotations['sha256'].isin(metadata['sha256'].values)]
    
    # download and render objects
    file_paths = oxl.download_objects(
        annotations,
        download_dir=os.path.join(output_dir, "raw"),
        save_repo_format="zip",
        processes=max_workers,
    )
    
    downloaded = {}
    metadata = metadata.set_index("file_identifier")
    for k, v in file_paths.items():
        sha256 = metadata.loc[k, "sha256"]
        downloaded[sha256] = os.path.relpath(v, output_dir)

    return pd.DataFrame(downloaded.items(), columns=['sha256', 'local_path'])


def _process_instance(args):
    """Worker function for ProcessPoolExecutor (must be top-level for pickling)"""
    import os, tempfile, zipfile
    metadatum, output_dir, func = args
    try:
        local_path = metadatum['local_path']
        sha256 = metadatum['sha256']
        
        direct_file_path = os.path.join(output_dir, local_path)
        if os.path.exists(direct_file_path):
            file = direct_file_path
            record = func(file, sha256)
        elif local_path.startswith('raw/github/repos/'):
            path_parts = local_path.split('/')
            file_name = os.path.join(*path_parts[5:])
            zip_file = os.path.join(output_dir, *path_parts[:5])
            if os.path.exists(zip_file):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(tmp_dir)
                    file = os.path.join(tmp_dir, file_name)
                    record = func(file, sha256)
            else:
                # zip file not found, pass local_path directly (for tasks like dual_grid_view that don't need the original file)
                file = local_path
                record = func(file, sha256)
        else:
            file = os.path.join(output_dir, local_path)
            record = func(file, sha256)
        return record
    except Exception as e:
        print(f"Error processing object {metadatum.get('sha256', '?')}: {e}")
        return None


def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects', log_interval=500, timeout=None) -> pd.DataFrame:
    print("================")
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
    from tqdm import tqdm
    
    # load metadata
    metadata = metadata.to_dict('records')

    max_workers = max_workers or os.cpu_count()
    records = []
    
    # Track processed/skipped counts
    total_processed = 0
    total_skipped = 0
    timeout_count = 0
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_instance, (m, output_dir, func)): m['sha256']
                for m in metadata
            }
            pbar = tqdm(as_completed(futures), total=len(futures), desc=desc)
            for future in pbar:
                sha256 = futures[future]
                try:
                    r = future.result(timeout=timeout)
                    if r is not None:
                        records.append(r)
                        # Update stats
                        if '_processed_count' in r:
                            total_processed += r['_processed_count']
                        if '_skipped_count' in r:
                            total_skipped += r['_skipped_count']
                        # Update progress bar display
                        pbar.set_postfix(processed=total_processed, skipped=total_skipped, timeout=timeout_count, refresh=False)
                except TimeoutError:
                    timeout_count += 1
                    print(f"Timeout processing object {sha256} (>{timeout}s)")
                    records.append({'sha256': sha256, 'error': f'Timeout (>{timeout}s)'})
                    pbar.set_postfix(processed=total_processed, skipped=total_skipped, timeout=timeout_count, refresh=False)
                except Exception as e:
                    print(f"Error processing object {sha256}: {e}")
    except Exception as e:
        print(f"Error happened during processing: {e}")
    
    if timeout_count > 0:
        print(f"Total timeout: {timeout_count} objects")
        
    return pd.DataFrame.from_records(records)
