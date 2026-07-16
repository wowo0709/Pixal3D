import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile

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
        "hf://datasets/JeffreyXiang/TRELLIS-500K/3D-FUTURE.csv"
    )


def _safe_destination(root: Path, member_name: str) -> Path:
    archive_path = PurePosixPath(member_name)
    if archive_path.is_absolute() or ".." in archive_path.parts:
        raise ValueError(f"Unsafe ZIP member: {member_name}")
    destination = (root / Path(*archive_path.parts)).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe ZIP member: {member_name}")
    return destination


def _validate_zip_member(root: Path, member: zipfile.ZipInfo) -> None:
    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError(f"Unsafe ZIP member: {member.filename}")
    _safe_destination(root, member.filename)


def download(
    metadata: pd.DataFrame, output_dir: str, **kwargs
) -> pd.DataFrame:
    max_workers = min(int(kwargs.get("max_workers", 8)), 8)
    if max_workers < 1:
        raise ValueError("max_workers must be positive")

    output_path = Path(output_dir)
    archive_path = output_path / "3D-FUTURE-model.zip"
    if not archive_path.is_file():
        raise FileNotFoundError(f"3D-FUTURE-model.zip not found: {archive_path}")

    raw_dir = output_path / "raw"
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        for member in members:
            _validate_zip_member(raw_dir, member)
        raw_dir.mkdir(parents=True, exist_ok=True)

        def worker(record):
            identifier = str(record["file_identifier"]).rstrip("/")
            prefix = f"{identifier}/"
            selected = [
                member
                for member in members
                if member.filename.startswith(prefix) and not member.is_dir()
            ]
            if not selected:
                return None
            for member in selected:
                archive.extract(member, raw_dir)
            image_path = _safe_destination(raw_dir, f"{identifier}/image.jpg")
            model_path = _safe_destination(
                raw_dir, f"{identifier}/raw_model.obj"
            )
            if not image_path.is_file() or not model_path.is_file():
                return None
            actual_sha256 = get_file_hash(str(image_path))
            if actual_sha256 != record["sha256"]:
                return None
            return {
                "sha256": actual_sha256,
                "local_path": f"raw/{identifier}/raw_model.obj",
            }

        records = metadata.to_dict("records")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            downloaded = list(executor.map(worker, records))

    return pd.DataFrame(
        [record for record in downloaded if record is not None],
        columns=["sha256", "local_path"],
    )


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
