from enum import Enum
from hashlib import sha256 as digest
import json
import os
from pathlib import Path

import pandas as pd


class AssetState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    QUARANTINED = "quarantined"


def camera_seed(asset_sha256: str, policy: str) -> int:
    return int.from_bytes(
        digest(f"{policy}:{asset_sha256}".encode()).digest()[:8], "big"
    )


def split_for_sha(asset_sha256: str) -> str:
    return "validation" if int(asset_sha256[:8], 16) % 100 == 0 else "train"


def canonicalize_sources(
    source_frames: dict[str, pd.DataFrame],
    policy: str,
    source_order: tuple[str, ...],
) -> pd.DataFrame:
    priority = {source: index for index, source in enumerate(source_order)}
    rows = [
        {**record, "source": source, "source_priority": priority[source]}
        for source, frame in source_frames.items()
        for record in frame.to_dict("records")
    ]
    merged = pd.DataFrame(rows).sort_values(
        ["sha256", "source_priority", "file_identifier"]
    )
    result = []
    for asset_sha, group in merged.groupby("sha256", sort=True):
        owner = group.iloc[0].to_dict()
        owner_source = owner.pop("source")
        owner.pop("source_priority")
        owner.update(
            owner_source=owner_source,
            duplicate_sources=json.dumps(
                sorted(set(group["source"]) - {owner_source})
            ),
            split=split_for_sha(asset_sha),
            camera_seed=camera_seed(asset_sha, policy),
            pipeline_state=AssetState.PENDING.value,
            attempt_count=0,
            last_error_category="",
            last_error="",
        )
        result.append(owner)
    return pd.DataFrame(result).sort_values("sha256").reset_index(drop=True)


def assign_shards(frame: pd.DataFrame, shard_size: int) -> pd.DataFrame:
    parts = []
    for source, source_frame in frame.groupby("owner_source", sort=False):
        ordered = source_frame.sort_values("sha256").copy()
        ordered["shard_id"] = [
            f"{source}-{index // shard_size:05d}" for index in range(len(ordered))
        ]
        parts.append(ordered)
    return pd.concat(parts, ignore_index=True)


def write_compat_metadata(frame: pd.DataFrame, source: str, path: Path) -> None:
    selected = frame.loc[frame["owner_source"] == source].sort_values("sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    selected.to_csv(temporary, index=False)
    os.replace(temporary, path)


class RegistryStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)

    def save(self, frame: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, self.path)

    def update_state(
        self,
        asset_sha256: str,
        field: str,
        state: AssetState,
        error: str = "",
    ) -> None:
        frame = self.load()
        selected = frame["sha256"] == asset_sha256
        if selected.sum() != 1:
            raise KeyError(asset_sha256)
        frame.loc[selected, field] = state.value
        if error:
            frame.loc[selected, "last_error"] = error
        self.save(frame)
