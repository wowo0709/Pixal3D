import argparse
import fcntl
import importlib
import os
from pathlib import Path
import sys
import tempfile

from easydict import EasyDict as edict
import pandas as pd


OBJAVERSE_ALIASES = {
    "ObjaverseXL_sketchfab": "sketchfab",
    "ObjaverseXL_github": "github",
}


def _adapter_target(dataset_name: str) -> tuple[str, str | None]:
    source = OBJAVERSE_ALIASES.get(dataset_name)
    return ("ObjaverseXL" if source else dataset_name, source)


def _import_adapter(adapter_name: str):
    package_name = f"data_toolkit.datasets.{adapter_name}"
    legacy_name = f"datasets.{adapter_name}"
    candidates = (
        (package_name, legacy_name)
        if __package__
        else (legacy_name, package_name)
    )
    try:
        return importlib.import_module(candidates[0])
    except ModuleNotFoundError as error:
        missing = error.name or ""
        if not (
            candidates[0] == missing
            or candidates[0].startswith(f"{missing}.")
        ):
            raise
    return importlib.import_module(candidates[1])


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            frame.to_csv(temporary, index=False)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _publish_download_records(
    download_root: Path, rank: int, downloaded: pd.DataFrame
) -> None:
    part = download_root / "raw/new_records" / f"part_{rank}.csv"
    _atomic_write_csv(downloaded, part)


def _read_records(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"sha256": str})


def _merge_download_records(download_root: Path) -> None:
    raw_dir = download_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    lock_path = raw_dir / ".metadata.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            parts = sorted((raw_dir / "new_records").glob("part_*.csv"))
            if not parts:
                return
            output = raw_dir / "metadata.csv"
            frames = []
            if output.is_file():
                frames.append(_read_records(output))
            frames.extend(_read_records(path) for path in parts)
            combined = pd.concat(frames, ignore_index=True, sort=False)
            merged = (
                combined.groupby("sha256", as_index=False, sort=True)
                .last()
                .reset_index(drop=True)
            )
            _atomic_write_csv(merged, output)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit("dataset name is required")

    adapter_name, canonical_source = _adapter_target(argv[0])
    dataset_utils = _import_adapter(adapter_name)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, required=True, help="Directory to save the metadata"
    )
    parser.add_argument(
        "--download_root",
        type=str,
        default=None,
        help="Directory to download the objects",
    )
    parser.add_argument(
        "--filter_low_aesthetic_score",
        type=float,
        default=None,
        help="Filter objects with aesthetic score lower than this value",
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Only check if the objects are already downloaded",
    )
    parser.add_argument(
        "--instances", type=str, default=None, help="Instances to process"
    )
    dataset_utils.add_args(parser)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    opt = edict(vars(parser.parse_args(argv[1:])))
    if canonical_source is not None:
        opt.source = canonical_source
    opt.download_root = opt.download_root or opt.root

    os.makedirs(opt.root, exist_ok=True)
    os.makedirs(opt.download_root, exist_ok=True)
    new_records = Path(opt.download_root) / "raw/new_records"
    new_records.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(opt.root) / "metadata.csv"
    if not metadata_path.is_file():
        raise ValueError("metadata.csv not found")
    metadata = _read_records(metadata_path).set_index("sha256")
    aesthetic_path = Path(opt.root) / "aesthetic_scores/metadata.csv"
    if aesthetic_path.is_file():
        metadata = metadata.combine_first(
            _read_records(aesthetic_path).set_index("sha256")
        )
    downloaded_metadata = Path(opt.download_root) / "raw/metadata.csv"
    if downloaded_metadata.is_file():
        metadata = metadata.combine_first(
            _read_records(downloaded_metadata).set_index("sha256")
        )
    metadata = metadata.reset_index()

    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[
                metadata["aesthetic_score"] >= opt.filter_low_aesthetic_score
            ]
        if "local_path" in metadata.columns:
            metadata = metadata[metadata["local_path"].isna()]
    else:
        instances_path = Path(opt.instances)
        if instances_path.is_file():
            instances = instances_path.read_text().splitlines()
        else:
            instances = opt.instances.split(",")
        metadata = metadata[metadata["sha256"].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata.iloc[start:end]
    print(f"Processing {len(metadata)} objects...")

    downloaded = dataset_utils.download(
        metadata, output_dir=opt.download_root, **opt
    )
    download_root = Path(opt.download_root)
    _publish_download_records(download_root, opt.rank, downloaded)
    _merge_download_records(download_root)


if __name__ == "__main__":
    main()
