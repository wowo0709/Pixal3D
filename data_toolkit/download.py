import argparse
import importlib
import os
from pathlib import Path
import sys

from easydict import EasyDict as edict
import pandas as pd


OBJAVERSE_ALIASES = {
    "ObjaverseXL_sketchfab": "sketchfab",
    "ObjaverseXL_github": "github",
}


def _adapter_target(dataset_name: str) -> tuple[str, str | None]:
    source = OBJAVERSE_ALIASES.get(dataset_name)
    return ("ObjaverseXL" if source else dataset_name, source)


def _merge_download_records(download_root: Path) -> None:
    raw_dir = download_root / "raw"
    parts = sorted((raw_dir / "new_records").glob("part_*.csv"))
    if not parts:
        return
    frames = [pd.read_csv(path) for path in parts]
    merged = pd.concat(frames, ignore_index=True)
    if "sha256" in merged.columns:
        merged = (
            merged.drop_duplicates(subset="sha256", keep="last")
            .sort_values("sha256")
            .reset_index(drop=True)
        )
    output = raw_dir / "metadata.csv"
    temporary = output.with_suffix(output.suffix + ".tmp")
    merged.to_csv(temporary, index=False)
    os.replace(temporary, output)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit("dataset name is required")

    adapter_name, canonical_source = _adapter_target(argv[0])
    dataset_utils = importlib.import_module(f"datasets.{adapter_name}")

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
    metadata = pd.read_csv(metadata_path).set_index("sha256")
    aesthetic_path = Path(opt.root) / "aesthetic_scores/metadata.csv"
    if aesthetic_path.is_file():
        metadata = metadata.combine_first(
            pd.read_csv(aesthetic_path).set_index("sha256")
        )
    downloaded_metadata = Path(opt.download_root) / "raw/metadata.csv"
    if downloaded_metadata.is_file():
        metadata = metadata.combine_first(
            pd.read_csv(downloaded_metadata).set_index("sha256")
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
    downloaded.to_csv(new_records / f"part_{opt.rank}.csv", index=False)
    _merge_download_records(Path(opt.download_root))


if __name__ == "__main__":
    main()
