import os
import argparse
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

import pandas as pd

try:
    from ..utils import get_file_hash
except ImportError:  # Legacy execution from data_toolkit/download.py.
    from utils import get_file_hash


ABO_ARCHIVE_URL = (
    "https://amazon-berkeley-objects.s3.amazonaws.com/archives/"
    "abo-3dmodels.tar"
)


def add_args(parser: argparse.ArgumentParser):
    pass


def get_metadata(**kwargs):
    metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ABO.csv")
    return metadata
        

def _safe_destination(root: Path, member_name: str) -> Path:
    archive_path = PurePosixPath(member_name)
    if archive_path.is_absolute() or ".." in archive_path.parts:
        raise ValueError(f"Unsafe TAR member: {member_name}")
    destination = (root / Path(*archive_path.parts)).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe TAR member: {member_name}")
    return destination


def _validate_tar_member(root: Path, member: tarfile.TarInfo) -> None:
    if member.issym() or member.islnk():
        raise ValueError(f"Unsafe TAR member: {member.name}")
    if not (member.isfile() or member.isdir()):
        raise ValueError(f"Unsafe TAR member: {member.name}")
    _safe_destination(root, member.name)


def download(
    metadata: pd.DataFrame, output_dir: str, **kwargs
) -> pd.DataFrame:
    max_workers = min(int(kwargs.get("max_workers", 8)), 8)
    if max_workers < 1:
        raise ValueError("max_workers must be positive")

    raw_dir = Path(output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "abo-3dmodels.tar"
    if not archive_path.is_file():
        subprocess.run(
            ["wget", "-O", str(archive_path), ABO_ARCHIVE_URL], check=True
        )

    downloaded = []
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        for member in members:
            _validate_tar_member(raw_dir, member)
        members_by_name = {member.name: member for member in members}

        for record in metadata.to_dict("records"):
            identifier = str(record["file_identifier"])
            member_name = f"3dmodels/original/{identifier}"
            member = members_by_name.get(member_name)
            if member is None or not member.isfile():
                continue
            archive.extract(member, raw_dir)
            local_file = _safe_destination(raw_dir, member_name)
            actual_sha256 = get_file_hash(str(local_file))
            if actual_sha256 == record["sha256"]:
                downloaded.append(
                    {
                        "sha256": actual_sha256,
                        "local_path": f"raw/{member_name}",
                    }
                )

    return pd.DataFrame(downloaded, columns=["sha256", "local_path"])


def _process_instance(args):
    """Worker function for ProcessPoolExecutor (must be top-level for pickling)"""
    import os
    metadatum, output_dir, func = args
    try:
        local_path = metadatum['local_path']
        sha256 = metadatum['sha256']
        file = os.path.join(output_dir, local_path)
        record = func(file, sha256)
        return record
    except Exception as e:
        print(f"Error processing object {metadatum.get('sha256', '?')}: {e}")
        return None


def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects') -> pd.DataFrame:
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    
    # load metadata
    metadata = metadata.to_dict('records')

    max_workers = max_workers or os.cpu_count()
    records = []
    
    # Track processed/skipped counts
    total_processed = 0
    total_skipped = 0
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_instance, (m, output_dir, func)): m['sha256']
                for m in metadata
            }
            pbar = tqdm(as_completed(futures), total=len(futures), desc=desc)
            for future in pbar:
                try:
                    r = future.result()
                    if r is not None:
                        records.append(r)
                        # Update stats
                        if '_processed_count' in r:
                            total_processed += r['_processed_count']
                        if '_skipped_count' in r:
                            total_skipped += r['_skipped_count']
                        # Update progress bar display
                        pbar.set_postfix(processed=total_processed, skipped=total_skipped, refresh=False)
                except Exception as e:
                    sha256 = futures[future]
                    print(f"Error processing object {sha256}: {e}")
    except Exception as e:
        print(f"Error happened during processing: {e}")
        
    return pd.DataFrame.from_records(records)
