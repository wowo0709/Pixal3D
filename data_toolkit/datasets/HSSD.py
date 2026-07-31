import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path, PurePosixPath

import huggingface_hub
import pandas as pd
from tqdm import tqdm

try:
    from ..utils import get_file_hash
except ImportError:  # Legacy execution from data_toolkit/download.py.
    from utils import get_file_hash


def add_args(parser: argparse.ArgumentParser):
    pass


def get_metadata(**kwargs):
    return pd.read_csv(
        "hf://datasets/JeffreyXiang/TRELLIS-500K/HSSD.csv"
    )


def _local_file(raw_dir: Path, identifier: str) -> Path:
    archive_path = PurePosixPath(identifier)
    if archive_path.is_absolute() or ".." in archive_path.parts:
        raise ValueError(f"Unsafe HSSD path: {identifier}")
    destination = (raw_dir / Path(*archive_path.parts)).resolve()
    if not destination.is_relative_to(raw_dir.resolve()):
        raise ValueError(f"Unsafe HSSD path: {identifier}")
    return destination


def download(
    metadata: pd.DataFrame, output_dir: str, **kwargs
) -> pd.DataFrame:
    max_workers = min(int(kwargs.get("max_workers", 8)), 8)
    if max_workers < 1:
        raise ValueError("max_workers must be positive")

    raw_dir = Path(output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    identifiers = metadata["file_identifier"].astype(str).tolist()
    for identifier in identifiers:
        _local_file(raw_dir, identifier)

    try:
        huggingface_hub.whoami()
    except Exception:
        huggingface_hub.login()

    huggingface_hub.snapshot_download(
        repo_id="hssd/hssd-models",
        repo_type="dataset",
        allow_patterns=identifiers,
        local_dir=str(raw_dir),
        max_workers=max_workers,
    )

    downloaded = []
    for record in metadata.to_dict("records"):
        identifier = str(record["file_identifier"])
        local_file = _local_file(raw_dir, identifier)
        if not local_file.is_file():
            continue
        actual_sha256 = get_file_hash(str(local_file))
        if actual_sha256 == record["sha256"]:
            downloaded.append(
                {
                    "sha256": actual_sha256,
                    "local_path": f"raw/{identifier}",
                }
            )
    return pd.DataFrame(downloaded, columns=["sha256", "local_path"])


def foreach_instance(
    metadata,
    output_dir,
    func,
    max_workers=None,
    desc="Processing objects",
) -> pd.DataFrame:
    records = []
    metadata_records = metadata.to_dict("records")
    max_workers = max_workers or os.cpu_count()

    def worker(metadatum):
        try:
            file = os.path.join(output_dir, metadatum["local_path"])
            return func(file, metadatum["sha256"])
        except Exception as error:
            print(
                f"Error processing object {metadatum.get('sha256', '?')}: "
                f"{error}"
            )
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for record in tqdm(
            executor.map(worker, metadata_records),
            total=len(metadata_records),
            desc=desc,
        ):
            if record is not None:
                records.append(record)
    return pd.DataFrame.from_records(records)
