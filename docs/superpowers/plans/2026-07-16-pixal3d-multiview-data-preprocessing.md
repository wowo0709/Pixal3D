# Pixal3D Multi-View Data Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and operate a safe, resumable pipeline that downloads the public 500,777-row Pixal3D/TRELLIS-500K pool and prepares eight-view conditions plus two-anchor Stage 1/2/3 training packs before fine-tuning starts.

**Architecture:** Existing `data_toolkit` scripts remain leaf workers. A focused `data_toolkit.pipeline` package owns configuration, canonical Parquet state, deterministic sampling, resource admission, subprocess orchestration, validation, packing, reporting, and recovery. Production data flows through bounded local scratch and is published as verified uncompressed tar packs on `/root/data2`, with raw archives on `/root/data3`.

**Tech Stack:** Python 3.11 in conda environment `pixal3d`, PyTorch 2.8+, CUDA 12.8+, pandas, PyArrow, Pillow, NumPy, psutil, pytest, Blender 4.5.1 LTS, OptiX, existing Pixal3D and O-Voxel modules.

**Primary References:** `data_toolkit/README.md`, the current `data_toolkit/*.py` leaf workers, [Pixal3D issue #9](https://github.com/TencentARC/Pixal3D/issues/9), and the upstream [TRELLIS dataset adapters](https://github.com/microsoft/TRELLIS/tree/main/dataset_toolkits/datasets).

## Global Constraints

- Run Python and pytest through `conda run -n pixal3d`.
- Use `/root/data2/pixal3d` for raw downloads, control state, staging, reports, and prepared packs.
- Use `/root/data3/pixal3d/archive` for verified raw archives, failed assets, and retired artifacts.
- Use `/root/pixal3d-data` for local preprocessing scratch and training `active`/`next`. Never use `/root/data/pixal3d`.
- Process ObjaverseXL Sketchfab, ObjaverseXL GitHub, ABO, HSSD, then 3D-FUTURE, sequentially.
- Keep Toys4K evaluation-only. Exclude TexVerse until its source data is supplied.
- Render exactly eight deterministic 512 by 512 RGBA conditions.
- Generate targets only for `view00` and `view01`.
- Generate SS at source resolution 64 and shape/PBR at 256, 512, and 1024.
- Deduplicate globally by SHA and split with `int(sha256[:8], 16) % 100`; bucket 0 is validation, buckets 1-99 are training.
- Never overlap preprocessing and fine-tuning.
- Start CPU-heavy work at no more than 32 runnable CPU threads.
- With seven encoder ranks, use two loader threads and one saver thread per rank; including each rank's main thread, the planned encoder-side total is 28 threads.
- Pause after CPU exceeds 80% or load exceeds 72 for two minutes; stop safely after CPU exceeds 90% for five minutes.
- Keep 96 GiB RAM normally available and stop safely below 64 GiB.
- Keep local free space above `max(15%, 120 GiB)`, data2 project use below 16 TiB soft and 18 TiB hard, and data3 project use below 26 TiB.
- Sample resources every five seconds, roll up every 30 seconds, and attach the last five minutes to escalations.
- Never silently change source scope, camera policy, dtype, encoder checkpoint, or retention.
- Follow TDD: failing focused test, minimal implementation, focused pass, relevant suite, focused commit.
- At execution time use `superpowers:using-git-worktrees`; preserve unrelated mode-only worktree changes.

---

## File Map

- `data_toolkit/pipeline/config.py`: strict typed YAML configuration.
- `data_toolkit/pipeline/registry.py`: canonical assets, deduplication, splits, shards, and atomic Parquet state.
- `data_toolkit/pipeline/preflight.py`: source credentials, manual archives, models, mounts, and hardware readiness.
- `data_toolkit/pipeline/camera.py`: deterministic camera generation.
- `data_toolkit/pipeline/blender.py`: pinned Blender installation and checksum.
- `data_toolkit/pipeline/atomic_io.py`: atomic NPZ, JSON, copy, and publication primitives.
- `data_toolkit/pipeline/validation.py`: render, camera, latent, scale, and pack validation.
- `data_toolkit/pipeline/packing.py`: deterministic tar packs and manifests.
- `data_toolkit/pipeline/resources.py`: telemetry and admission policy.
- `data_toolkit/pipeline/commands.py`: exact leaf command DAG.
- `data_toolkit/pipeline/orchestrator.py`: retries, state, cleanup, publication, archive, and resume.
- `data_toolkit/pipeline/reporting.py`: smoke, pilot, capacity, source, audit, and handoff reports.
- `data_toolkit/pipeline/cli.py`: operator entry point.
- `data_toolkit/configs/multiview_preprocess.yaml`: production policy.
- `tests/data_toolkit/`: unit and synthetic integration tests.

---

### Task 1: Test Harness, Package Skeleton, and Typed Configuration

**Files:**
- Create: `pytest.ini`
- Create: `data_toolkit/__init__.py`
- Create: `data_toolkit/requirements.txt`
- Create: `data_toolkit/pipeline/__init__.py`
- Create: `data_toolkit/pipeline/config.py`
- Create: `data_toolkit/configs/multiview_preprocess.yaml`
- Create: `tests/data_toolkit/test_config.py`
- Modify: `data_toolkit/setup.sh:1`

**Interfaces:**
- Produces: `PipelineConfig`, `load_config(path: Path) -> PipelineConfig`, `PipelineConfig.config_hash() -> str`.
- Consumes: no earlier task.

- [ ] **Step 1: Write the failing configuration tests**

```python
# tests/data_toolkit/test_config.py
from pathlib import Path
import pytest
from data_toolkit.pipeline.config import load_config

CONFIG = Path("data_toolkit/configs/multiview_preprocess.yaml")

def test_fixed_contract():
    cfg = load_config(CONFIG)
    assert cfg.paths.local_root == Path("/root/pixal3d-data")
    assert cfg.paths.data2_root == Path("/root/data2/pixal3d")
    assert cfg.paths.data3_root == Path("/root/data3/pixal3d")
    assert cfg.sources == ("ObjaverseXL_sketchfab", "ObjaverseXL_github", "ABO", "HSSD", "3D-FUTURE")
    assert (cfg.render.num_views, cfg.render.resolution) == (8, 512)
    assert cfg.targets.views == (0, 1)
    assert cfg.targets.resolutions == (256, 512, 1024)
    assert cfg.limits.cpu_soft_percent == 80.0
    assert cfg.limits.ram_soft_available_gib == 96

def test_unknown_root_key_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(CONFIG.read_text() + "\nunknown_key: true\n")
    with pytest.raises(ValueError, match="unknown_key"):
        load_config(path)

def test_hash_is_stable():
    assert load_config(CONFIG).config_hash() == load_config(CONFIG).config_hash()
    assert len(load_config(CONFIG).config_hash()) == 64
```

- [ ] **Step 2: Verify the tests fail for the missing package**

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_config.py -v`

Expected: FAIL with `ModuleNotFoundError: data_toolkit.pipeline`.

- [ ] **Step 3: Add dependencies and pytest settings**

```text
# data_toolkit/requirements.txt
pandas==2.3.3
pyarrow==21.0.0
psutil==7.2.2
PyYAML==6.0.3
pytest==8.4.2
huggingface_hub==0.36.2
```

```ini
# pytest.ini
[pytest]
testpaths = tests
markers =
    integration: requires external binaries, models, or data
    gpu: requires CUDA
```

```bash
# data_toolkit/setup.sh
#!/usr/bin/env bash
set -euo pipefail
python -m pip install -r "$(dirname "$0")/requirements.txt"
python -m pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless pandas open3d objaverse 'huggingface_hub[cli]' open_clip_torch
```

- [ ] **Step 4: Implement the typed configuration**

```python
# data_toolkit/pipeline/config.py
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import yaml

@dataclass(frozen=True)
class PathConfig:
    data2_root: Path
    data3_root: Path
    local_root: Path

@dataclass(frozen=True)
class RenderConfig:
    num_views: int
    resolution: int
    fov_min_degrees: float
    fov_max_degrees: float
    camera_policy: str
    blender_version: str
    cycles_device: str

@dataclass(frozen=True)
class TargetConfig:
    views: tuple[int, ...]
    resolutions: tuple[int, ...]
    ss_resolution: int
    latent_dtype: str

@dataclass(frozen=True)
class WorkerConfig:
    cpu_threads: int
    dump_workers: int
    voxel_workers: int
    voxel_threads_per_worker: int
    render_workers: int
    encoder_ranks: int
    encoder_loader_threads: int
    encoder_saver_threads: int

@dataclass(frozen=True)
class LimitConfig:
    cpu_soft_percent: float
    cpu_hard_percent: float
    load_soft: float
    io_wait_soft_percent: float
    ram_soft_available_gib: int
    ram_hard_available_gib: int
    local_free_percent: float
    local_free_gib: int
    data2_soft_tib: int
    data2_hard_tib: int
    data2_fs_free_tib: int
    data3_soft_tib: int
    data3_fs_free_tib: int

@dataclass(frozen=True)
class PipelineConfig:
    pipeline_version: str
    sources: tuple[str, ...]
    evaluation_sources: tuple[str, ...]
    shard_size: int
    paths: PathConfig
    render: RenderConfig
    targets: TargetConfig
    workers: WorkerConfig
    limits: LimitConfig

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return sha256(payload).hexdigest()

ROOT_KEYS = {"pipeline_version", "sources", "evaluation_sources", "shard_size", "paths", "render", "targets", "workers", "limits"}

def load_config(path: Path) -> PipelineConfig:
    raw = yaml.safe_load(path.read_text())
    unknown = set(raw) - ROOT_KEYS
    if unknown:
        raise ValueError(f"unknown_key: {sorted(unknown)}")
    return PipelineConfig(
        pipeline_version=raw["pipeline_version"],
        sources=tuple(raw["sources"]),
        evaluation_sources=tuple(raw["evaluation_sources"]),
        shard_size=int(raw["shard_size"]),
        paths=PathConfig(**{key: Path(value) for key, value in raw["paths"].items()}),
        render=RenderConfig(**raw["render"]),
        targets=TargetConfig(tuple(raw["targets"]["views"]), tuple(raw["targets"]["resolutions"]), raw["targets"]["ss_resolution"], raw["targets"]["latent_dtype"]),
        workers=WorkerConfig(**raw["workers"]),
        limits=LimitConfig(**raw["limits"]),
    )
```

Use this exact YAML:

```yaml
pipeline_version: pixal3d-mv-v1
sources: [ObjaverseXL_sketchfab, ObjaverseXL_github, ABO, HSSD, 3D-FUTURE]
evaluation_sources: [Toys4k]
shard_size: 5000
paths: {data2_root: /root/data2/pixal3d, data3_root: /root/data3/pixal3d, local_root: /root/pixal3d-data}
render: {num_views: 8, resolution: 512, fov_min_degrees: 10.0, fov_max_degrees: 70.0, camera_policy: pixal3d-mv-camera-v1, blender_version: 4.5.1, cycles_device: OPTIX}
targets: {views: [0, 1], resolutions: [256, 512, 1024], ss_resolution: 64, latent_dtype: float32}
workers: {cpu_threads: 32, dump_workers: 24, voxel_workers: 8, voxel_threads_per_worker: 4, render_workers: 7, encoder_ranks: 7, encoder_loader_threads: 2, encoder_saver_threads: 1}
limits: {cpu_soft_percent: 80.0, cpu_hard_percent: 90.0, load_soft: 72.0, io_wait_soft_percent: 10.0, ram_soft_available_gib: 96, ram_hard_available_gib: 64, local_free_percent: 15.0, local_free_gib: 120, data2_soft_tib: 16, data2_hard_tib: 18, data2_fs_free_tib: 2, data3_soft_tib: 26, data3_fs_free_tib: 4}
```

- [ ] **Step 5: Install, test, and commit**

Run: `conda run -n pixal3d bash data_toolkit/setup.sh`

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_config.py -v`

Expected: 3 passed.

```bash
git add pytest.ini data_toolkit/__init__.py data_toolkit/requirements.txt data_toolkit/setup.sh data_toolkit/pipeline/__init__.py data_toolkit/pipeline/config.py data_toolkit/configs/multiview_preprocess.yaml tests/data_toolkit/test_config.py
git commit -m "feat: add preprocessing pipeline configuration"
```

---

### Task 2: Canonical Registry, Deduplication, Splits, and Shards

**Files:**
- Create: `data_toolkit/pipeline/registry.py`
- Create: `tests/data_toolkit/test_registry.py`

**Interfaces:**
- Consumes: `PipelineConfig`.
- Produces: `AssetState`, `camera_seed`, `split_for_sha`, `canonicalize_sources`, `assign_shards`, `RegistryStore`.

- [ ] **Step 1: Write failing registry tests**

```python
# tests/data_toolkit/test_registry.py
import pandas as pd
from data_toolkit.pipeline.registry import RegistryStore, assign_shards, camera_seed, canonicalize_sources, split_for_sha, write_compat_metadata

def test_seed_and_split_are_deterministic():
    sha = "01234567" + "a" * 56
    assert camera_seed(sha, "pixal3d-mv-camera-v1") == camera_seed(sha, "pixal3d-mv-camera-v1")
    assert split_for_sha(sha) == "train"
    assert split_for_sha("00000000" + "b" * 56) == "validation"

def test_global_deduplication():
    shared = "a" * 64
    frames = {
        "ABO": pd.DataFrame([{"sha256": shared, "file_identifier": "a.glb"}]),
        "HSSD": pd.DataFrame([{"sha256": shared, "file_identifier": "h.glb"}]),
    }
    result = canonicalize_sources(frames, "pixal3d-mv-camera-v1", ("ABO", "HSSD"))
    assert len(result) == 1
    assert result.iloc[0]["owner_source"] == "ABO"
    assert result.iloc[0]["duplicate_sources"] == '["HSSD"]'

def test_stable_shards_and_atomic_roundtrip(tmp_path):
    frame = pd.DataFrame({"sha256": [f"{i:064x}" for i in range(6)], "owner_source": ["ABO"] * 6})
    first = assign_shards(frame, 2).set_index("sha256")["shard_id"].to_dict()
    second = assign_shards(frame.sample(frac=1, random_state=7), 2).set_index("sha256")["shard_id"].to_dict()
    assert first == second
    store = RegistryStore(tmp_path / "assets.parquet")
    store.save(frame)
    assert store.load().to_dict("records") == frame.to_dict("records")
    assert not (tmp_path / "assets.parquet.tmp").exists()

def test_compat_metadata_contains_only_owned_rows(tmp_path):
    frame = pd.DataFrame([
        {"sha256": "a" * 64, "owner_source": "ABO", "file_identifier": "a.glb"},
        {"sha256": "b" * 64, "owner_source": "HSSD", "file_identifier": "b.glb"},
    ])
    path = tmp_path / "ABO" / "metadata.csv"
    write_compat_metadata(frame, "ABO", path)
    assert pd.read_csv(path)["sha256"].tolist() == ["a" * 64]
    assert not path.with_suffix(".csv.tmp").exists()
```

- [ ] **Step 2: Verify tests fail, then implement the registry**

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_registry.py -v`

Expected: FAIL importing `data_toolkit.pipeline.registry`.

```python
# data_toolkit/pipeline/registry.py
from enum import StrEnum
from hashlib import sha256 as digest
import json
import os
from pathlib import Path
import pandas as pd

class AssetState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    QUARANTINED = "quarantined"

def camera_seed(asset_sha256: str, policy: str) -> int:
    return int.from_bytes(digest(f"{policy}:{asset_sha256}".encode()).digest()[:8], "big")

def split_for_sha(asset_sha256: str) -> str:
    return "validation" if int(asset_sha256[:8], 16) % 100 == 0 else "train"

def canonicalize_sources(source_frames: dict[str, pd.DataFrame], policy: str, source_order: tuple[str, ...]) -> pd.DataFrame:
    priority = {source: index for index, source in enumerate(source_order)}
    rows = [{**record, "source": source, "source_priority": priority[source]} for source, frame in source_frames.items() for record in frame.to_dict("records")]
    merged = pd.DataFrame(rows).sort_values(["sha256", "source_priority", "file_identifier"])
    result = []
    for asset_sha, group in merged.groupby("sha256", sort=True):
        owner = group.iloc[0].to_dict()
        owner_source = owner.pop("source")
        owner.pop("source_priority")
        owner.update(
            owner_source=owner_source,
            duplicate_sources=json.dumps(sorted(set(group["source"]) - {owner_source})),
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
        ordered["shard_id"] = [f"{source}-{index // shard_size:05d}" for index in range(len(ordered))]
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

    def update_state(self, asset_sha256: str, field: str, state: AssetState, error: str = "") -> None:
        frame = self.load()
        selected = frame["sha256"] == asset_sha256
        if selected.sum() != 1:
            raise KeyError(asset_sha256)
        frame.loc[selected, field] = state.value
        if error:
            frame.loc[selected, "last_error"] = error
        self.save(frame)
```

- [ ] **Step 3: Run tests and commit**

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_registry.py tests/data_toolkit/test_config.py -v`

Expected: 7 passed.

```bash
git add data_toolkit/pipeline/registry.py tests/data_toolkit/test_registry.py
git commit -m "feat: add canonical preprocessing registry"
```

### Task 3: Dataset Adapters and Source Access Preflight

**Files:**
- Create: `data_toolkit/datasets/HSSD.py`
- Create: `data_toolkit/datasets/3D-FUTURE.py`
- Create: `data_toolkit/datasets/Toys4k.py`
- Create: `data_toolkit/pipeline/preflight.py`
- Create: `tests/data_toolkit/test_dataset_adapters.py`
- Create: `tests/data_toolkit/test_preflight.py`
- Modify: `data_toolkit/datasets/ABO.py:20-63`
- Modify: `data_toolkit/download.py:12-64`

**Interfaces:**
- Consumes: `PipelineConfig`.
- Produces: adapters with `download(metadata: pd.DataFrame, output_dir: str, **kwargs) -> pd.DataFrame`; `PreflightStatus`; `PreflightResult`; `run_preflight`.

- [ ] **Step 1: Write failing adapter and preflight tests**

```python
# tests/data_toolkit/test_dataset_adapters.py
import importlib
import inspect
import pandas as pd

def test_adapter_download_contract():
    for name in ("ABO", "HSSD", "3D-FUTURE", "Toys4k", "ObjaverseXL"):
        parameters = inspect.signature(importlib.import_module(f"data_toolkit.datasets.{name}").download).parameters
        assert "metadata" in parameters
        assert "output_dir" in parameters

def test_public_metadata_paths(monkeypatch):
    seen = []
    monkeypatch.setattr(pd, "read_csv", lambda path: seen.append(path) or pd.DataFrame())
    for name in ("HSSD", "3D-FUTURE", "Toys4k"):
        importlib.import_module(f"data_toolkit.datasets.{name}").get_metadata()
    assert seen == [
        "hf://datasets/JeffreyXiang/TRELLIS-500K/HSSD.csv",
        "hf://datasets/JeffreyXiang/TRELLIS-500K/3D-FUTURE.csv",
        "hf://datasets/JeffreyXiang/TRELLIS-500K/Toys4k.csv",
    ]
```

```python
# tests/data_toolkit/test_preflight.py
from dataclasses import replace
from data_toolkit.pipeline.preflight import PreflightStatus, run_preflight

def test_missing_manual_archives_block(config, tmp_path):
    cfg = replace(config, paths=replace(config.paths, data2_root=tmp_path / "data2"))
    results = {item.source: item for item in run_preflight(cfg, check_remote=False)}
    assert results["3D-FUTURE"].status == PreflightStatus.BLOCKED
    assert "3D-FUTURE-model.zip" in results["3D-FUTURE"].message
    assert results["Toys4k"].status == PreflightStatus.BLOCKED
```

```python
# tests/data_toolkit/conftest.py
from pathlib import Path
import pytest
from data_toolkit.pipeline.config import load_config

@pytest.fixture
def config():
    return load_config(Path("data_toolkit/configs/multiview_preprocess.yaml"))
```

- [ ] **Step 2: Verify missing adapters and preflight fail**

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_dataset_adapters.py tests/data_toolkit/test_preflight.py -v`

Expected: FAIL importing HSSD and preflight modules.

- [ ] **Step 3: Port and normalize the source adapters**

Port source-specific acquisition and hash rules from:

- `https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/datasets/HSSD.py`
- `https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/datasets/3D-FUTURE.py`
- `https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/datasets/Toys4k.py`

Every adapter must use this signature and bound:

```python
def download(metadata: pd.DataFrame, output_dir: str, **kwargs) -> pd.DataFrame:
    max_workers = min(int(kwargs.get("max_workers", 8)), 8)
```

Implement the adapters with these exact source rules:

- HSSD calls `huggingface_hub.snapshot_download(repo_id="hssd/hssd-models", repo_type="dataset", allow_patterns=metadata["file_identifier"].tolist(), local_dir=<output_dir>/raw, max_workers=max_workers)` after preflight login, checks each selected file against the metadata SHA, and returns only verified `sha256,local_path` rows. It never exceeds eight transfer workers.
- 3D-FUTURE requires `<output_dir>/3D-FUTURE-model.zip`, validates every ZIP member before extraction under `<output_dir>/raw`, extracts only metadata-selected model directories, hashes each selected `image.jpg` as required by the public metadata, and returns the verified directory's `raw_model.obj` relative path.
- Toys4K requires `<output_dir>/toys4k_blend_files.zip`, validates every ZIP member before extraction under `<output_dir>/raw`, extracts only selected `.blend` members, hashes each result, and returns verified relative paths.
- ABO changes `download(metadata, root, **kwargs)` to `download(metadata, output_dir, **kwargs)`, downloads with `subprocess.run(..., check=True)`, validates tar member paths, and extracts only metadata-selected members.
- ObjaverseXL retains its existing adapter but receives `max_workers <= 8`; the wrapper maps canonical sources to `ObjaverseXL --source sketchfab` and `ObjaverseXL --source github`.

Before tar or ZIP extraction, reject absolute names, `..` components, symlinks, hard links, and any resolved path escaping `<output_dir>/raw`. Add `--max_workers` to `download.py`, pass it to the adapter, and merge verified `raw/new_records/part_*.csv` into an atomic `raw/metadata.csv` before local staging.

- [ ] **Step 4: Implement explicit preflight statuses**

```python
# data_toolkit/pipeline/preflight.py
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import huggingface_hub
from .config import PipelineConfig

class PreflightStatus(StrEnum):
    READY = "ready"
    BLOCKED = "blocked"
    ERROR = "error"

@dataclass(frozen=True)
class PreflightResult:
    source: str
    status: PreflightStatus
    message: str

def _manual(source: str, path: Path) -> PreflightResult:
    status = PreflightStatus.READY if path.is_file() else PreflightStatus.BLOCKED
    message = str(path) if path.is_file() else f"Missing manual archive: {path}"
    return PreflightResult(source, status, message)

def run_preflight(config: PipelineConfig, check_remote: bool = True) -> tuple[PreflightResult, ...]:
    raw = config.paths.data2_root / "raw"
    results = [
        PreflightResult("ObjaverseXL_sketchfab", PreflightStatus.READY, "ObjaverseXL API"),
        PreflightResult("ObjaverseXL_github", PreflightStatus.READY, "ObjaverseXL API"),
        PreflightResult("ABO", PreflightStatus.READY, "ABO public archive"),
    ]
    if check_remote:
        try:
            huggingface_hub.whoami()
            huggingface_hub.hf_hub_download(repo_id="hssd/hssd-models", filename="README.md", repo_type="dataset")
            hssd = PreflightResult("HSSD", PreflightStatus.READY, "HSSD access verified")
        except Exception as error:
            hssd = PreflightResult("HSSD", PreflightStatus.BLOCKED, f"HSSD access failed: {error}")
    else:
        hssd = PreflightResult("HSSD", PreflightStatus.BLOCKED, "HSSD remote check skipped")
    results.extend([
        hssd,
        _manual("3D-FUTURE", raw / "3D-FUTURE" / "3D-FUTURE-model.zip"),
        _manual("Toys4k", raw / "Toys4k" / "toys4k_blend_files.zip"),
    ])
    return tuple(results)
```

- [ ] **Step 5: Test and commit**

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_dataset_adapters.py tests/data_toolkit/test_preflight.py -v`

Expected: all tests pass without network access.

```bash
git add data_toolkit/datasets/ABO.py data_toolkit/datasets/HSSD.py data_toolkit/datasets/3D-FUTURE.py data_toolkit/datasets/Toys4k.py data_toolkit/download.py data_toolkit/pipeline/preflight.py tests/data_toolkit/conftest.py tests/data_toolkit/test_dataset_adapters.py tests/data_toolkit/test_preflight.py
git commit -m "feat: add dataset source adapters and preflight"
```

---

### Task 4: Deterministic Cameras and Pinned Blender Rendering

**Files:**
- Create: `data_toolkit/pipeline/camera.py`
- Create: `data_toolkit/pipeline/blender.py`
- Create: `tests/data_toolkit/test_camera.py`
- Create: `tests/data_toolkit/test_blender.py`
- Modify: `data_toolkit/render_cond.py:17-72,78-154`
- Modify: `data_toolkit/blender_script/render_cond.py:38-59,435-528`

**Interfaces:**
- Consumes: `RenderConfig`, `camera_seed`.
- Produces: `build_condition_views`, `verify_archive`, `ensure_blender`.

- [ ] **Step 1: Write failing deterministic camera and checksum tests**

```python
# tests/data_toolkit/test_camera.py
from data_toolkit.pipeline.camera import build_condition_views

def test_views_are_deterministic_and_bounded(config):
    first = build_condition_views("d" * 64, config.render)
    assert first == build_condition_views("d" * 64, config.render)
    assert len(first) == 8
    assert all(10 <= item["fov_degrees"] <= 70 for item in first)
    assert first != build_condition_views("e" * 64, config.render)
```

```python
# tests/data_toolkit/test_blender.py
from hashlib import sha256
import pytest
from data_toolkit.pipeline.blender import verify_archive

def test_checksum_verification(tmp_path):
    path = tmp_path / "blender.tar.xz"
    path.write_bytes(b"fixture")
    verify_archive(path, sha256(b"fixture").hexdigest())
    with pytest.raises(ValueError, match="checksum"):
        verify_archive(path, "0" * 64)
```

- [ ] **Step 2: Verify tests fail**

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_camera.py tests/data_toolkit/test_blender.py -v`

Expected: FAIL importing camera and Blender helpers.

- [ ] **Step 3: Implement deterministic cameras**

```python
# data_toolkit/pipeline/camera.py
import numpy as np
from data_toolkit.utils import sphere_hammersley_sequence
from .config import RenderConfig
from .registry import camera_seed

def build_condition_views(asset_sha256: str, config: RenderConfig) -> list[dict[str, float]]:
    rng = np.random.Generator(np.random.PCG64(camera_seed(asset_sha256, config.camera_policy)))
    offset = tuple(float(value) for value in rng.random(2))
    fov_min = np.deg2rad(config.fov_min_degrees)
    fov_max = np.deg2rad(config.fov_max_degrees)
    radius_min = np.sqrt(3) / 2 / np.sin(fov_max / 2)
    radius_max = np.sqrt(3) / 2 / np.sin(fov_min / 2)
    radii = 1 / np.sqrt(rng.uniform(1 / radius_max**2, 1 / radius_min**2, config.num_views))
    result = []
    for index, radius in enumerate(radii):
        yaw, pitch = sphere_hammersley_sequence(index, config.num_views, offset)
        fov = float(2 * np.arcsin(np.sqrt(3) / 2 / radius))
        result.append({"yaw": float(yaw), "pitch": float(pitch), "radius": float(radius), "fov": fov, "fov_degrees": float(np.rad2deg(fov))})
    return result
```

- [ ] **Step 4: Implement the pinned Blender installer**

```python
# data_toolkit/pipeline/blender.py
from hashlib import sha256
import os
from pathlib import Path
import shutil
import tarfile
from urllib.request import urlopen

BLENDER_URL = "https://download.blender.org/release/Blender4.5/blender-4.5.1-linux-x64.tar.xz"
BLENDER_SHA256 = "085a7ed4ed80c3cb66783bad76f236f39897de5d33884abd133e0c6db94c0f14"
BLENDER_DIR = "blender-4.5.1-linux-x64"

def verify_archive(path: Path, expected: str) -> None:
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    if value.hexdigest() != expected:
        raise ValueError(f"Blender checksum mismatch: {path}")

def ensure_blender(tool_root: Path) -> Path:
    binary = tool_root / BLENDER_DIR / "blender"
    if binary.is_file():
        return binary
    tool_root.mkdir(parents=True, exist_ok=True)
    archive = tool_root / Path(BLENDER_URL).name
    partial = archive.with_suffix(archive.suffix + ".part")
    with urlopen(BLENDER_URL) as source, partial.open("wb") as target:
        shutil.copyfileobj(source, target, 8 * 1024 * 1024)
    os.replace(partial, archive)
    verify_archive(archive, BLENDER_SHA256)
    with tarfile.open(archive, "r:xz") as bundle:
        bundle.extractall(tool_root, filter="data")
    if not binary.is_file():
        raise FileNotFoundError(binary)
    return binary
```

- [ ] **Step 5: Harden both render scripts**

Add `--cond_resolution`, `--blender_path`, `--cycles_device`, and `--timeout_seconds`. Call `build_condition_views`, not `np.random`; pass `download_root`, not `render_cond_root`, to `foreach_instance`; use `subprocess.run(check=True, timeout=...)`; render to a temporary sibling directory and rename only after validation.

In Blender, scale the 130-pixel boundary target by `resolution / 1024` and select OptiX explicitly:

```python
preferences = bpy.context.preferences.addons["cycles"].preferences
preferences.compute_device_type = arg.cycles_device
preferences.get_devices()
selected = []
for device in preferences.devices:
    device.use = device.type != "CPU"
    if device.use:
        selected.append(device.name)
if not selected:
    raise RuntimeError(f"No {arg.cycles_device} device selected")
bpy.context.scene.cycles.device = "GPU"
```

Write selected device names into `transforms.json` so hardware preflight can reject CPU fallback.

- [ ] **Step 6: Test, syntax-check, and commit**

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_camera.py tests/data_toolkit/test_blender.py -v`

Run: `conda run -n pixal3d python -m compileall -q data_toolkit`

Expected: tests pass and compile exits 0.

```bash
git add data_toolkit/pipeline/camera.py data_toolkit/pipeline/blender.py data_toolkit/render_cond.py data_toolkit/blender_script/render_cond.py tests/data_toolkit/test_camera.py tests/data_toolkit/test_blender.py
git commit -m "feat: make condition rendering deterministic"
```

---

### Task 5: Atomic I/O and Content Validators

**Files:**
- Create: `data_toolkit/pipeline/atomic_io.py`
- Create: `data_toolkit/pipeline/validation.py`
- Create: `tests/data_toolkit/test_atomic_io.py`
- Create: `tests/data_toolkit/test_validation.py`

**Interfaces:**
- Produces: `atomic_save_npz`, `atomic_write_json`, `atomic_copy`, `ValidationError`, render/latent/scale validators.

- [ ] **Step 1: Write failing tests**

```python
# tests/data_toolkit/test_atomic_io.py
import numpy as np
from data_toolkit.pipeline.atomic_io import atomic_save_npz

def test_atomic_npz(tmp_path):
    path = tmp_path / "view00.npz"
    atomic_save_npz(path, feats=np.ones((2, 3), np.float32), coords=np.zeros((2, 3), np.uint8))
    with np.load(path) as data:
        assert data["feats"].shape == (2, 3)
    assert list(tmp_path.glob("*.tmp")) == []
```

```python
# tests/data_toolkit/test_validation.py
import json
import numpy as np
from PIL import Image
import pytest
from data_toolkit.pipeline.validation import ValidationError, validate_render_dir, validate_sparse_latent

def test_valid_render_directory(tmp_path):
    frames = []
    for index in range(8):
        rgba = np.zeros((512, 512, 4), np.uint8)
        rgba[128:384, 128:384, 3] = 255
        Image.fromarray(rgba, "RGBA").save(tmp_path / f"{index:03d}.png")
        frames.append({"file_path": f"{index:03d}.png", "camera_angle_x": 0.7, "transform_matrix": np.eye(4).tolist(), "radius": 2.0})
    (tmp_path / "transforms.json").write_text(json.dumps({"frames": frames}))
    validate_render_dir(tmp_path, 8, 512)

def test_non_finite_latent_is_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, feats=np.array([[np.nan]], np.float32), coords=np.zeros((1, 3), np.uint8))
    with pytest.raises(ValidationError, match="non-finite"):
        validate_sparse_latent(path, 16, 8192)
```

- [ ] **Step 2: Verify tests fail, then implement atomic writers**

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_atomic_io.py tests/data_toolkit/test_validation.py -v`

Expected: FAIL importing Task 5 modules.

```python
# data_toolkit/pipeline/atomic_io.py
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np

def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)

def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
```

- [ ] **Step 3: Implement validators**

```python
# data_toolkit/pipeline/validation.py
import json
from pathlib import Path
import numpy as np
from PIL import Image

class ValidationError(ValueError):
    pass

def validate_render_dir(path: Path, expected_views: int, resolution: int) -> None:
    try:
        frames = json.loads((path / "transforms.json").read_text())["frames"]
    except Exception as error:
        raise ValidationError(f"invalid transforms: {error}") from error
    if len(frames) != expected_views:
        raise ValidationError(f"expected {expected_views} frames, found {len(frames)}")
    for index, frame in enumerate(frames):
        with Image.open(path / f"{index:03d}.png") as image:
            rgba = np.asarray(image.convert("RGBA"))
        if rgba.shape != (resolution, resolution, 4):
            raise ValidationError(f"wrong image shape: {rgba.shape}")
        alpha_fraction = float((rgba[..., 3] > 0).mean())
        if not 0.01 <= alpha_fraction <= 0.95:
            raise ValidationError(f"invalid alpha fraction: {alpha_fraction}")
        matrix = np.asarray(frame["transform_matrix"], np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-8:
            raise ValidationError(f"invalid camera frame {index}")

def validate_sparse_latent(path: Path, grid_resolution: int, max_tokens: int) -> None:
    with np.load(path) as data:
        feats, coords = data["feats"], data["coords"]
    if feats.shape[0] != coords.shape[0] or coords.ndim != 2 or coords.shape[1] != 3:
        raise ValidationError(f"shape mismatch: {path}")
    if len(coords) > max_tokens:
        raise ValidationError(f"token limit exceeded: {len(coords)}")
    if not np.isfinite(feats).all():
        raise ValidationError(f"non-finite features: {path}")
    if (coords < 0).any() or (coords >= grid_resolution).any():
        raise ValidationError(f"coordinates outside grid: {path}")

def validate_ss_latent(path: Path) -> None:
    with np.load(path) as data:
        if not np.isfinite(data["z"]).all():
            raise ValidationError(f"non-finite SS latent: {path}")

def validate_scale(path: Path) -> None:
    values = np.asarray(list(json.loads(path.read_text()).values()), np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValidationError(f"invalid scale metadata: {path}")
```

- [ ] **Step 4: Test and commit**

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_atomic_io.py tests/data_toolkit/test_validation.py -v`

Expected: all Task 5 tests pass.

```bash
git add data_toolkit/pipeline/atomic_io.py data_toolkit/pipeline/validation.py tests/data_toolkit/test_atomic_io.py tests/data_toolkit/test_validation.py
git commit -m "feat: add atomic preprocessing validation"
```

### Task 6: Bounded and Resumable Leaf Workers

**Files:**
- Create: `tests/data_toolkit/test_leaf_worker_contracts.py`
- Modify: `data_toolkit/dump_mesh.py:27-44,56-126`
- Modify: `data_toolkit/dump_pbr.py:28-45,57-127`
- Modify: `data_toolkit/dual_grid_view.py:59-153,202-350`
- Modify: `data_toolkit/voxelize_pbr_view.py:294-397,404-596`
- Modify: `data_toolkit/encode_shape_latent_view.py:30-262`
- Modify: `data_toolkit/encode_pbr_latent_view.py:30-271`
- Modify: `data_toolkit/encode_ss_latent_view.py:25-257`
- Modify: `data_toolkit/build_metadata.py:12-48,151-317`

**Interfaces:**
- Consumes: Task 5 writers and validators.
- Produces: common `--timeout_seconds`; voxel `--native_threads`; encoder `--loader_workers`, `--saver_workers`, `--latent_dtype`; content-aware resume.

- [ ] **Step 1: Write failing CLI contract tests**

```python
# tests/data_toolkit/test_leaf_worker_contracts.py
import subprocess
import sys
import pytest

CPU_SCRIPTS = ("dump_mesh.py", "dump_pbr.py", "dual_grid_view.py", "voxelize_pbr_view.py")
ENCODERS = ("encode_shape_latent_view.py", "encode_pbr_latent_view.py", "encode_ss_latent_view.py")

@pytest.mark.parametrize("script", CPU_SCRIPTS + ENCODERS)
def test_timeout_flag(script):
    result = subprocess.run([sys.executable, f"data_toolkit/{script}", "--help"], capture_output=True, text=True)
    assert "--timeout_seconds" in result.stdout

@pytest.mark.parametrize("script", ENCODERS)
def test_encoder_bounds_and_dtype(script):
    result = subprocess.run([sys.executable, f"data_toolkit/{script}", "--help"], capture_output=True, text=True)
    assert "--loader_workers" in result.stdout
    assert "--saver_workers" in result.stdout
    assert "--latent_dtype" in result.stdout

@pytest.mark.parametrize("script", ("dual_grid_view.py", "voxelize_pbr_view.py"))
def test_voxel_native_thread_bound(script):
    result = subprocess.run([sys.executable, f"data_toolkit/{script}", "--help"], capture_output=True, text=True)
    assert "--native_threads" in result.stdout
```

- [ ] **Step 2: Verify flags are initially missing**

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_leaf_worker_contracts.py -v`

Expected: FAIL because the new flags are absent.

- [ ] **Step 3: Add bounded worker and timeout arguments**

Add this to CPU leaf parsers:

```python
parser.add_argument("--timeout_seconds", type=int, default=900)
```

Add `parser.add_argument("--native_threads", type=int, default=4)` to `dual_grid_view.py` and `voxelize_pbr_view.py`, and pass `opt.native_threads` to every O-Voxel `num_threads` argument instead of a literal or `os.cpu_count()`.

Add this to all three encoders:

```python
parser.add_argument("--loader_workers", type=int, default=2)
parser.add_argument("--saver_workers", type=int, default=1)
parser.add_argument("--latent_dtype", choices=("float32", "float16"), default="float32")
parser.add_argument("--timeout_seconds", type=int, default=900)
```

Replace hard-coded 32-thread encoder pools with parsed values and set queue size to `max(2, opt.loader_workers * 2)`. Save shape/PBR as:

```python
feature_dtype = np.float16 if opt.latent_dtype == "float16" else np.float32
atomic_save_npz(
    Path(save_path),
    feats=z.feats.cpu().numpy().astype(feature_dtype),
    coords=z.coords[:, 1:].cpu().numpy().astype(np.uint8),
)
```

Save SS with `atomic_save_npz(Path(save_path), z=z[0].cpu().numpy())`, and scale JSON with `atomic_copy`. Validate an existing output before skipping; delete and regenerate it if validation fails.

- [ ] **Step 4: Make remaining outputs and metadata atomic**

Write `.vxz`, pickle, and CSV outputs to sibling `.tmp` paths, reopen them with their native reader, then call `os.replace`. In `build_metadata.update_metadata`, read `merged_records` when `--from_merged_records` is selected instead of reading `new_records`. Treat absent optional root directories as empty rather than calling `os.listdir` on them.

Add a test that places a corrupt NPZ at an expected encoder output, stubs the encoder/input, runs one asset, and asserts the file is replaced by a valid NPZ.

- [ ] **Step 5: Test, compile, and commit**

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_leaf_worker_contracts.py -v`

Run: `conda run -n pixal3d python -m compileall -q data_toolkit`

Expected: tests pass and compile exits 0.

```bash
git add data_toolkit/dump_mesh.py data_toolkit/dump_pbr.py data_toolkit/dual_grid_view.py data_toolkit/voxelize_pbr_view.py data_toolkit/encode_shape_latent_view.py data_toolkit/encode_pbr_latent_view.py data_toolkit/encode_ss_latent_view.py data_toolkit/build_metadata.py tests/data_toolkit/test_leaf_worker_contracts.py
git commit -m "feat: make preprocessing workers resumable"
```

---

### Task 7: Deterministic Training Packs and Publication

**Files:**
- Create: `data_toolkit/pipeline/packing.py`
- Create: `tests/data_toolkit/test_packing.py`

**Interfaces:**
- Consumes: `atomic_write_json`, `ValidationError`.
- Produces: `PackMember`, `PackManifest`, `build_pack`, `verify_pack`, `publish_pack`.

- [ ] **Step 1: Write failing pack tests**

```python
# tests/data_toolkit/test_packing.py
from pathlib import Path
import pytest
from data_toolkit.pipeline.packing import build_pack, verify_pack

def test_pack_is_deterministic(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a").write_text("a")
    (source / "b").write_text("b")
    metadata = {"batch_id": "batch000", "family": "common", "config_hash": "c" * 64, "tool_commit": "abc123", "asset_sha256s": ("a" * 64,), "completed_count": 1, "quarantined_count": 0}
    first = build_pack(source, [Path("b"), Path("a")], tmp_path / "one.tar", "ABO-00000", **metadata)
    second = build_pack(source, [Path("a"), Path("b")], tmp_path / "two.tar", "ABO-00000", **metadata)
    assert first.pack_sha256 == second.pack_sha256
    verify_pack(tmp_path / "one.tar", tmp_path / "one.tar.manifest.json")

def test_symlink_is_rejected(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "target").write_text("x")
    (source / "link").symlink_to("target")
    with pytest.raises(ValueError, match="symlink"):
        build_pack(source, [Path("link")], tmp_path / "bad.tar", "ABO-00000", batch_id="batch000", family="common", config_hash="c" * 64, tool_commit="abc123", asset_sha256s=("a" * 64,), completed_count=1, quarantined_count=0)
```

- [ ] **Step 2: Verify the packing module is missing**

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_packing.py -v`

Expected: FAIL importing packing.

- [ ] **Step 3: Implement deterministic tar and manifests**

```python
# data_toolkit/pipeline/packing.py
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tarfile
from .atomic_io import atomic_write_json
from .validation import ValidationError

@dataclass(frozen=True)
class PackMember:
    path: str
    size: int
    sha256: str

@dataclass(frozen=True)
class PackManifest:
    shard_id: str
    batch_id: str
    family: str
    config_hash: str
    tool_commit: str
    asset_sha256s: tuple[str, ...]
    completed_count: int
    quarantined_count: int
    created_at: str
    validated_at: str
    pack_sha256: str
    members: tuple[PackMember, ...]

def file_sha(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def build_pack(source_root: Path, members: list[Path], output: Path, shard_id: str, *, batch_id: str, family: str, config_hash: str, tool_commit: str, asset_sha256s: tuple[str, ...], completed_count: int, quarantined_count: int) -> PackManifest:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    recorded = []
    with tarfile.open(temporary, "w", format=tarfile.PAX_FORMAT) as bundle:
        for relative in sorted(members, key=lambda item: item.as_posix()):
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe member: {relative}")
            path = source_root / relative
            if path.is_symlink():
                raise ValueError(f"symlink member: {relative}")
            info = bundle.gettarinfo(str(path), relative.as_posix())
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as stream:
                bundle.addfile(info, stream)
            recorded.append(PackMember(relative.as_posix(), path.stat().st_size, file_sha(path)))
    os.replace(temporary, output)
    manifest = PackManifest(
        shard_id,
        batch_id,
        family,
        config_hash,
        tool_commit,
        tuple(sorted(asset_sha256s)),
        completed_count,
        quarantined_count,
        datetime.now(timezone.utc).isoformat(),
        "",
        file_sha(output),
        tuple(recorded),
    )
    atomic_write_json(output.with_suffix(output.suffix + ".manifest.json"), asdict(manifest))
    return manifest

def verify_pack(pack_path: Path, manifest_path: Path) -> None:
    expected = json.loads(manifest_path.read_text())
    if file_sha(pack_path) != expected["pack_sha256"]:
        raise ValidationError(f"pack checksum mismatch: {pack_path}")
    members = {item["path"]: item for item in expected["members"]}
    with tarfile.open(pack_path) as bundle:
        actual = {}
        for item in bundle.getmembers():
            path = Path(item.name)
            if path.is_absolute() or ".." in path.parts or item.issym() or item.islnk() or not item.isfile():
                raise ValidationError(f"unsafe tar member: {item.name}")
            if item.name in actual:
                raise ValidationError(f"duplicate tar member: {item.name}")
            member_sha = sha256()
            stream = bundle.extractfile(item)
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                member_sha.update(block)
            actual[item.name] = (item.size, member_sha.hexdigest())
    if set(actual) != set(members):
        raise ValidationError("pack member set mismatch")
    for name, (size, digest_value) in actual.items():
        if size != members[name]["size"] or digest_value != members[name]["sha256"]:
            raise ValidationError(f"pack member size mismatch: {name}")
```

Implement `publish_pack` to build under `data2/staging/<shard>/<batch>`, verify, write `validated_at` into the manifest, then atomically rename tar and manifest into `prepared`. Refuse to overwrite a valid published pack with a different checksum. Publish exactly eight families per work batch: common, SS-64, shape-256/512/1024, and PBR-256/512/1024. A top-level logical-shard index records every batch and all eight manifest checksums so Stage 1, Stage 2, and Stage 3 can be materialized independently.

- [ ] **Step 4: Test and commit**

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_packing.py -v`

Expected: all pack tests pass.

```bash
git add data_toolkit/pipeline/packing.py tests/data_toolkit/test_packing.py
git commit -m "feat: add verified training pack publication"
```

---

### Task 8: Continuous Resource Telemetry and Guardrails

**Files:**
- Create: `data_toolkit/pipeline/resources.py`
- Create: `tests/data_toolkit/test_resources.py`

**Interfaces:**
- Consumes: `PipelineConfig.limits` and configured roots.
- Produces: `ResourceSnapshot`, `ResourceAction`, `ResourceDecision`, `ResourcePolicy`, `ResourceGuard`, telemetry JSONL.

- [ ] **Step 1: Write failing threshold tests**

```python
# tests/data_toolkit/test_resources.py
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from data_toolkit.pipeline.resources import ResourceAction, ResourcePolicy, ResourceSnapshot

def sample(now, **changes):
    base = ResourceSnapshot(now, 20.0, 10.0, 1.0, 400.0, 0, 300.0, 40.0, 1.0, 20.0, 1.0, 30.0)
    return replace(base, **changes)

def test_cpu_soft_and_hard_durations(config):
    start = datetime(2026, 7, 16, tzinfo=timezone.utc)
    policy = ResourcePolicy(config.limits)
    assert policy.evaluate(sample(start, cpu_percent=85.0)).action == ResourceAction.RUN
    assert policy.evaluate(sample(start + timedelta(minutes=2), cpu_percent=85.0)).action == ResourceAction.PAUSE
    policy = ResourcePolicy(config.limits)
    policy.evaluate(sample(start, cpu_percent=95.0))
    assert policy.evaluate(sample(start + timedelta(minutes=5), cpu_percent=95.0)).action == ResourceAction.STOP

def test_local_absolute_floor_stops(config):
    decision = ResourcePolicy(config.limits).evaluate(sample(datetime.now(timezone.utc), local_free_gib=119.0))
    assert decision.action == ResourceAction.STOP

def test_nfs_free_space_floors_stop(config):
    now = datetime.now(timezone.utc)
    assert ResourcePolicy(config.limits).evaluate(sample(now, data2_fs_free_tib=1.9)).action == ResourceAction.STOP
    assert ResourcePolicy(config.limits).evaluate(sample(now, data3_fs_free_tib=3.9)).action == ResourceAction.STOP

def test_swap_activity_pauses_new_work(config):
    decision = ResourcePolicy(config.limits).evaluate(sample(datetime.now(timezone.utc), swap_in_bytes=4096))
    assert decision.action == ResourceAction.PAUSE
```

- [ ] **Step 2: Verify the resource module is missing**

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_resources.py -v`

Expected: FAIL importing resources.

- [ ] **Step 3: Implement resource state and policy**

```python
# data_toolkit/pipeline/resources.py
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import json
from pathlib import Path
import os
import psutil
from .config import LimitConfig

@dataclass(frozen=True)
class ResourceSnapshot:
    timestamp: datetime
    cpu_percent: float
    load_1m: float
    io_wait_percent: float
    available_ram_gib: float
    swap_in_bytes: int
    local_free_gib: float
    local_free_percent: float
    data2_project_tib: float
    data2_fs_free_tib: float
    data3_project_tib: float
    data3_fs_free_tib: float

class ResourceAction(StrEnum):
    RUN = "run"
    PAUSE = "pause"
    STOP = "stop"

@dataclass(frozen=True)
class ResourceDecision:
    action: ResourceAction
    reasons: tuple[str, ...]

class ResourceLimitExceeded(RuntimeError):
    def __init__(self, reasons: tuple[str, ...]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons

class ResourcePolicy:
    def __init__(self, limits: LimitConfig):
        self.limits = limits
        self.first_seen: dict[str, datetime] = {}

    def duration(self, key: str, active: bool, now: datetime) -> timedelta:
        if not active:
            self.first_seen.pop(key, None)
            return timedelta()
        self.first_seen.setdefault(key, now)
        return now - self.first_seen[key]

    def evaluate(self, value: ResourceSnapshot) -> ResourceDecision:
        now, hard, soft = value.timestamp, [], []
        if value.local_free_gib < self.limits.local_free_gib or value.local_free_percent < self.limits.local_free_percent:
            hard.append("local free-space floor")
        if value.data2_project_tib >= self.limits.data2_hard_tib:
            hard.append("data2 hard project limit")
        if value.data2_fs_free_tib < self.limits.data2_fs_free_tib:
            hard.append("data2 filesystem free-space floor")
        if value.data3_fs_free_tib < self.limits.data3_fs_free_tib:
            hard.append("data3 filesystem free-space floor")
        if value.available_ram_gib < self.limits.ram_hard_available_gib:
            hard.append("RAM hard floor")
        if self.duration("cpu_hard", value.cpu_percent > self.limits.cpu_hard_percent, now) >= timedelta(minutes=5):
            hard.append("CPU hard duration")
        if self.duration("cpu_soft", value.cpu_percent > self.limits.cpu_soft_percent, now) >= timedelta(minutes=2):
            soft.append("CPU soft duration")
        if self.duration("load", value.load_1m > self.limits.load_soft, now) >= timedelta(minutes=2):
            soft.append("load soft duration")
        if self.duration("iowait", value.io_wait_percent > self.limits.io_wait_soft_percent, now) >= timedelta(minutes=2):
            soft.append("I/O wait")
        if value.available_ram_gib < self.limits.ram_soft_available_gib:
            soft.append("RAM soft floor")
        if value.swap_in_bytes > 0:
            soft.append("swap-in activity")
        if value.data2_project_tib >= self.limits.data2_soft_tib or value.data3_project_tib >= self.limits.data3_soft_tib:
            soft.append("project storage soft limit")
        return ResourceDecision(ResourceAction.STOP if hard else ResourceAction.PAUSE if soft else ResourceAction.RUN, tuple(hard or soft))
```

Implement sampling with psutil for CPU, iowait, RAM, swap delta, and filesystem free space. Use registry-accounted bytes for five-second project size samples and reconcile with a directory walk only at shard boundaries. Query GPU metrics with `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits`; a query failure is reported but does not bypass CPU/storage admission.

Append JSONL every five seconds; flush and fsync every 30 seconds. Implement `ResourceGuard(sample, policy, telemetry_writer, clock, sleeper)` with injected dependencies so tests do not wait in real time. `check(shard_id, command) -> ResourceDecision` samples once, writes telemetry, and converts recovery-period `RUN` values to `PAUSE` until five uninterrupted stable minutes pass. `wait_for_admission(shard_id, command)` calls `check` every five seconds, raises `ResourceLimitExceeded(decision.reasons)` immediately on `STOP`, and blocks admission on `PAUSE`. Keep a 60-snapshot deque for `last_five_minutes()` and write the shard/command with every JSONL sample. An initial `RUN` with no preceding pause returns immediately.

Long-running subprocesses are not exempt: Task 10 calls `check` every five seconds while every download, Blender, voxel, and encoder process group is alive. A soft decision sends `SIGSTOP` to the process group and samples until recovery permits `SIGCONT`; a hard decision sends `SIGCONT` if necessary, then `SIGTERM`, waits 60 seconds, sends `SIGKILL` only if still alive, checkpoints, and escalates. Atomic leaf outputs make this interruption resumable.

- [ ] **Step 4: Test stable recovery, telemetry, and commit**

Add fake-time tests proving five stable minutes are required after a pause and telemetry serialization includes ISO timestamps.

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_resources.py -v`

Expected: all resource tests pass.

```bash
git add data_toolkit/pipeline/resources.py tests/data_toolkit/test_resources.py
git commit -m "feat: add preprocessing resource guardrails"
```

---

### Task 9: Exact Per-Shard Command DAG

**Files:**
- Create: `data_toolkit/pipeline/commands.py`
- Create: `tests/data_toolkit/test_commands.py`

**Interfaces:**
- Consumes: `PipelineConfig`.
- Produces: `ShardContext`, `CommandSpec`, `expand_ranked`, `build_preprocessing_dag`.

- [ ] **Step 1: Write failing DAG tests**

```python
# tests/data_toolkit/test_commands.py
from data_toolkit.pipeline.commands import ShardContext, build_preprocessing_dag

def test_dag_order_and_anchor_views(config, tmp_path):
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")
    dag = build_preprocessing_dag(context, config)
    names = [item.name for item in dag]
    assert names[:6] == ["download", "stage_raw", "dump_mesh", "dump_pbr", "asset_stats", "render_cond"]
    assert names.index("cleanup_voxels_256") > names.index("encode_pbr_256")
    assert names.index("dual_grid_512") > names.index("cleanup_voxels_256")
    assert names.index("encode_ss_64") > names.index("encode_shape_1024")
    assert names[-4:] == ["validate_outputs", "build_packs", "archive_raw", "cleanup_local"]
    voxel = [item for item in dag if item.name.startswith(("dual_grid", "voxelize_pbr"))]
    assert all("0-1" in item.argv for item in voxel)

def test_objaversexl_source_mapping_and_blender_path(config, tmp_path):
    context = ShardContext.for_test(tmp_path, "ObjaverseXL_github", "ObjaverseXL_github-00000")
    dag = build_preprocessing_dag(context, config)
    download = next(item for item in dag if item.name == "download")
    render = next(item for item in dag if item.name == "render_cond")
    assert download.argv[2:5] == ("ObjaverseXL", "--source", "github")
    assert "--blender_path" in render.argv
    assert render.argv[render.argv.index("--blender_path") + 1].endswith("blender-4.5.1-linux-x64/blender")
    assert render.gpu_ranks == config.workers.render_workers
    assert render.argv[render.argv.index("--max_workers") + 1] == "1"
```

- [ ] **Step 2: Verify the command module is missing**

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_commands.py -v`

Expected: FAIL importing commands.

- [ ] **Step 3: Implement typed command construction**

```python
# data_toolkit/pipeline/commands.py
from dataclasses import dataclass
from pathlib import Path
from .config import PipelineConfig

@dataclass(frozen=True)
class ShardContext:
    source: str
    shard_id: str
    instances: Path
    metadata_root: Path
    source_root: Path
    download_root: Path
    work_root: Path
    output_root: Path
    batch_id: str

    @classmethod
    def for_test(cls, root: Path, source: str, shard_id: str):
        return cls(source, shard_id, root / "instances.txt", root / "metadata", root / "source", root / "raw", root / "work", root / "output", "batch000")

    @classmethod
    def from_config(cls, config: PipelineConfig, source: str, shard_id: str, batch_id: str):
        local = config.paths.local_root / "preprocess" / "active" / shard_id / batch_id
        control = config.paths.data2_root / "control"
        return cls(
            source,
            shard_id,
            control / "shards" / source / shard_id / f"{batch_id}.txt",
            control / "metadata" / source,
            config.paths.data2_root / "raw" / source,
            local / "source",
            local / "work",
            local / "output",
            batch_id,
        )

@dataclass(frozen=True)
class CommandSpec:
    name: str
    argv: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()
    gpu_ranks: int = 0

CPU_ENV = (("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"))
RENDER_ENV = (("OMP_NUM_THREADS", "2"), ("MKL_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"))

def python_command(script: str, *args: str) -> tuple[str, ...]:
    return ("python", f"data_toolkit/{script}", *args)

def dataset_args(source: str) -> tuple[str, ...]:
    if source == "ObjaverseXL_sketchfab":
        return ("ObjaverseXL", "--source", "sketchfab")
    if source == "ObjaverseXL_github":
        return ("ObjaverseXL", "--source", "github")
    return (source,)

def expand_ranked(command: CommandSpec) -> tuple[tuple[tuple[str, ...], tuple[tuple[str, str], ...]], ...]:
    if not command.gpu_ranks:
        return ((command.argv, command.env),)
    return tuple(
        (
            (*command.argv, "--rank", str(rank), "--world_size", str(command.gpu_ranks)),
            (*command.env, ("CUDA_VISIBLE_DEVICES", str(rank))),
        )
        for rank in range(command.gpu_ranks)
    )

def build_preprocessing_dag(context: ShardContext, config: PipelineConfig) -> tuple[CommandSpec, ...]:
    dataset = dataset_args(context.source)
    base = (*dataset, "--root", str(context.metadata_root), "--instances", str(context.instances))
    blender = config.paths.local_root / "tools" / "blender-4.5.1-linux-x64" / "blender"
    commands = [
        CommandSpec("download", python_command("download.py", *base, "--download_root", str(context.source_root), "--max_workers", "8"), CPU_ENV),
        CommandSpec("stage_raw", ("internal:stage_raw",)),
        CommandSpec("dump_mesh", python_command("dump_mesh.py", *base, "--download_root", str(context.download_root), "--mesh_dump_root", str(context.work_root), "--max_workers", str(config.workers.dump_workers)), CPU_ENV),
        CommandSpec("dump_pbr", python_command("dump_pbr.py", *base, "--download_root", str(context.download_root), "--pbr_dump_root", str(context.work_root), "--max_workers", str(config.workers.dump_workers)), CPU_ENV),
        CommandSpec("asset_stats", python_command("asset_stats.py", "--root", str(context.metadata_root), "--instances", str(context.instances), "--mesh_dump_root", str(context.work_root), "--pbr_dump_root", str(context.work_root), "--max_workers", str(config.workers.dump_workers)), CPU_ENV),
        CommandSpec("render_cond", python_command("render_cond.py", *base, "--download_root", str(context.download_root), "--render_cond_root", str(context.output_root), "--num_cond_views", "8", "--cond_resolution", "512", "--blender_path", str(blender), "--cycles_device", "OPTIX", "--max_workers", "1"), RENDER_ENV, gpu_ranks=config.workers.render_workers),
    ]
    for resolution in config.targets.resolutions:
        common = ("--resolution", str(resolution), "--view_indices", "0-1")
        commands.extend([
            CommandSpec(f"dual_grid_{resolution}", python_command("dual_grid_view.py", *base, "--mesh_dump_root", str(context.work_root), "--transform_root", str(context.output_root / "renders_cond"), "--dual_grid_root", str(context.work_root), *common, "--max_workers", str(config.workers.voxel_workers), "--native_threads", str(config.workers.voxel_threads_per_worker)), CPU_ENV),
            CommandSpec(f"voxelize_pbr_{resolution}", python_command("voxelize_pbr_view.py", *base, "--pbr_dump_root", str(context.work_root), "--transform_root", str(context.output_root / "renders_cond"), "--pbr_voxel_root", str(context.work_root), *common, "--max_workers", str(config.workers.voxel_workers), "--native_threads", str(config.workers.voxel_threads_per_worker)), CPU_ENV),
            CommandSpec(f"encode_shape_{resolution}", python_command("encode_shape_latent_view.py", "--root", str(context.metadata_root), "--instances", str(context.instances), "--dual_grid_root", str(context.work_root), "--shape_latent_root", str(context.output_root), *common, "--loader_workers", str(config.workers.encoder_loader_threads), "--saver_workers", str(config.workers.encoder_saver_threads), "--latent_dtype", config.targets.latent_dtype), CPU_ENV, gpu_ranks=config.workers.encoder_ranks),
            CommandSpec(f"encode_pbr_{resolution}", python_command("encode_pbr_latent_view.py", "--root", str(context.metadata_root), "--instances", str(context.instances), "--pbr_voxel_root", str(context.work_root), "--pbr_latent_root", str(context.output_root), *common, "--loader_workers", str(config.workers.encoder_loader_threads), "--saver_workers", str(config.workers.encoder_saver_threads), "--latent_dtype", config.targets.latent_dtype), CPU_ENV, gpu_ranks=config.workers.encoder_ranks),
            CommandSpec(f"cleanup_voxels_{resolution}", ("internal:cleanup_voxels", str(resolution))),
        ])
    commands.extend([
        CommandSpec("encode_ss_64", python_command("encode_ss_latent_view.py", "--root", str(context.metadata_root), "--instances", str(context.instances), "--shape_latent_root", str(context.output_root), "--ss_latent_root", str(context.output_root), "--shape_latent_name", "shape_enc_next_dc_f16c32_fp16_1024", "--resolution", "64", "--view_indices", "0-1", "--loader_workers", str(config.workers.encoder_loader_threads), "--saver_workers", str(config.workers.encoder_saver_threads)), CPU_ENV, gpu_ranks=config.workers.encoder_ranks),
        CommandSpec("validate_outputs", ("internal:validate_outputs",)),
        CommandSpec("build_packs", ("internal:build_packs",)),
        CommandSpec("archive_raw", ("internal:archive_raw",)),
        CommandSpec("cleanup_local", ("internal:cleanup_local",)),
    ])
    return tuple(commands)
```

`stage_raw` copies only the shard's verified raw paths from `source_root` into `download_root`, preserving adapter-relative paths, then atomically writes `<download_root>/raw/metadata.csv` with local paths that resolve under local scratch. Reject paths that escape either root. For Objaverse GitHub repository ZIPs, extract only selected members into scratch; retain a registry reference count for the shared ZIP.

`archive_raw` creates a per-work-batch uncompressed tar from the exact staged raw inputs, publishes it under `/root/data3/pixal3d/archive/raw/<source>/<shard_id>/<batch_id>.tar`, and verifies file count, total bytes, member hashes, and whole-tar SHA before marking archive state complete. Delete a data2 raw file only when the registry reports no pending shard references; shared Objaverse repository ZIPs remain until their final referencing shard completes. `cleanup_local` runs only after pack and archive validation.

The orchestrator calls `expand_ranked` and starts all ranks in one monitored command group. Each rank gets a distinct `CUDA_VISIBLE_DEVICES`, `--rank`, and `--world_size`; rendering uses one asset worker and at most two OpenMP threads per GPU. CPU stages remain sequential and use the configured aggregate worker maxima.

- [ ] **Step 4: Test and commit**

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_commands.py -v`

Expected: all DAG tests pass.

```bash
git add data_toolkit/pipeline/commands.py tests/data_toolkit/test_commands.py
git commit -m "feat: define preprocessing shard DAG"
```

### Task 10: Resumable Orchestrator and Immediate Escalation

**Files:**
- Create: `data_toolkit/pipeline/orchestrator.py`
- Create: `tests/data_toolkit/test_orchestrator.py`

**Interfaces:**
- Consumes: registry, resources, DAG, validators, and packer.
- Produces: `PipelineCheckpoint`, `EscalationReport`, `PipelineStopped`, `PipelineRunner.run_shard`, `PipelineRunner.resume_shard`.

- [ ] **Step 1: Write failing resume and stop tests**

```python
# tests/data_toolkit/test_orchestrator.py
from collections import defaultdict
import pytest
from data_toolkit.pipeline.commands import ShardContext
from data_toolkit.pipeline.orchestrator import PipelineCheckpoint, PipelineRunner, PipelineStopped, plan_work_batches
from data_toolkit.pipeline.resources import ResourceLimitExceeded

class FakeResourceGuard:
    def __init__(self):
        self.reason = None

    def stop_next(self, reason):
        self.reason = reason

    def wait_for_admission(self, shard_id, command):
        if self.reason:
            raise ResourceLimitExceeded((self.reason,))

    def last_five_minutes(self):
        return ({"cpu_percent": 95.0},)

class RecordingRunner(PipelineRunner):
    def __init__(self, config):
        validators = defaultdict(lambda: lambda: True)
        handlers = defaultdict(lambda: lambda: None)
        super().__init__(config, FakeResourceGuard(), validators, handlers)
        self.checkpoint = PipelineCheckpoint("ABO-00000")
        self.checkpoint.was_saved = False
        self.executed = []

    def load_checkpoint(self, path, shard_id):
        return self.checkpoint

    def save_checkpoint(self, path, checkpoint):
        checkpoint.was_saved = True

    def execute(self, command, shard_id):
        self.executed.append(command.name)
        self.validators[command.name] = lambda: True

@pytest.fixture
def shard_context(tmp_path):
    return ShardContext.for_test(tmp_path, "ABO", "ABO-00000")

@pytest.fixture
def fake_runner(config):
    return RecordingRunner(config)

def test_work_batches_fit_reserved_local_budget():
    shas = tuple(f"{index:064x}" for index in range(10))
    batches = plan_work_batches(shas, local_usable_bytes=1000, p95_peak_bytes=300, shard_size=5000)
    assert tuple(sha for batch in batches for sha in batch) == shas
    assert all(len(batch) <= 2 for batch in batches)

def test_resume_skips_only_valid_outputs(fake_runner, shard_context):
    fake_runner.checkpoint.complete("dump_mesh")
    fake_runner.validators["dump_mesh"] = lambda: True
    fake_runner.run_shard(shard_context)
    assert "dump_mesh" not in fake_runner.executed
    assert "dump_pbr" in fake_runner.executed

def test_corrupt_complete_output_is_regenerated(fake_runner, shard_context):
    fake_runner.checkpoint.complete("render_cond")
    fake_runner.validators["render_cond"] = lambda: False
    fake_runner.run_shard(shard_context)
    assert "render_cond" in fake_runner.executed

def test_hard_resource_stop_checkpoints_and_escalates(fake_runner, shard_context):
    fake_runner.resource_guard.stop_next("CPU hard duration")
    with pytest.raises(PipelineStopped) as caught:
        fake_runner.run_shard(shard_context)
    assert caught.value.report.reason == "CPU hard duration"
    assert caught.value.report.shard_id == shard_context.shard_id
    assert caught.value.report.recent_telemetry
    assert fake_runner.checkpoint.was_saved
```

- [ ] **Step 2: Verify the orchestrator is missing**

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_orchestrator.py -v`

Expected: FAIL importing orchestrator.

- [ ] **Step 3: Implement checkpointed sequential execution**

```python
# data_toolkit/pipeline/orchestrator.py
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from .atomic_io import atomic_write_json
from .commands import CommandSpec, ShardContext, build_preprocessing_dag, expand_ranked
from .config import PipelineConfig
from .resources import ResourceAction, ResourceLimitExceeded

@dataclass
class PipelineCheckpoint:
    shard_id: str
    completed_commands: list[str] = field(default_factory=list)
    attempts: dict[str, int] = field(default_factory=dict)

    def complete(self, command: str) -> None:
        if command not in self.completed_commands:
            self.completed_commands.append(command)

@dataclass(frozen=True)
class EscalationReport:
    source: str
    shard_id: str
    command: str
    reason: str
    recent_telemetry: tuple[dict, ...]
    safe_resume_command: str
    created_at: str

class PipelineStopped(RuntimeError):
    def __init__(self, report: EscalationReport, exit_code: int):
        super().__init__(report.reason)
        self.report = report
        self.exit_code = exit_code

def plan_work_batches(asset_sha256s: tuple[str, ...], local_usable_bytes: int, p95_peak_bytes: int, shard_size: int) -> tuple[tuple[str, ...], ...]:
    per_asset = max(1, int(p95_peak_bytes * 1.25))
    batch_size = max(1, min(shard_size, int(local_usable_bytes * 0.80) // per_asset))
    ordered = tuple(sorted(asset_sha256s))
    return tuple(ordered[index:index + batch_size] for index in range(0, len(ordered), batch_size))

class PipelineRunner:
    def __init__(self, config: PipelineConfig, resource_guard, validators, internal_handlers):
        self.config = config
        self.resource_guard = resource_guard
        self.validators = validators
        self.internal_handlers = internal_handlers

    def run_shard(self, context: ShardContext) -> None:
        checkpoint_path = context.work_root / "checkpoint.json"
        checkpoint = self.load_checkpoint(checkpoint_path, context.shard_id)
        for command in build_preprocessing_dag(context, self.config):
            if command.name in checkpoint.completed_commands and self.validators[command.name]():
                continue
            try:
                self.resource_guard.wait_for_admission(context.shard_id, command.name)
            except ResourceLimitExceeded as error:
                self.save_checkpoint(checkpoint_path, checkpoint)
                self.stop(context, command.name, "; ".join(error.reasons), checkpoint, exit_code=3)
            try:
                self.execute(command, context.shard_id)
                if not self.validators[command.name]():
                    raise ValueError(f"validation failed: {command.name}")
            except Exception as error:
                checkpoint.attempts[command.name] = checkpoint.attempts.get(command.name, 0) + 1
                self.save_checkpoint(checkpoint_path, checkpoint)
                if isinstance(error, ResourceLimitExceeded):
                    self.stop(context, command.name, str(error), checkpoint, exit_code=3)
                if checkpoint.attempts[command.name] >= 3 or self.is_infrastructure_error(error):
                    exit_code = 4 if "validation failed" in str(error).lower() else 2
                    self.stop(context, command.name, str(error), checkpoint, exit_code=exit_code)
                continue
            checkpoint.complete(command.name)
            self.save_checkpoint(checkpoint_path, checkpoint)

    def resume_shard(self, context: ShardContext) -> None:
        self.run_shard(context)

    def execute(self, command: CommandSpec, shard_id: str) -> None:
        if command.argv[0].startswith("internal:"):
            self.internal_handlers[command.name]()
            return
        processes = []
        for argv, additions in expand_ranked(command):
            environment = os.environ.copy()
            environment.update(dict(additions))
            processes.append(subprocess.Popen(argv, env=environment, start_new_session=True))
        paused = False
        while any(process.poll() is None for process in processes):
            decision = self.resource_guard.check(shard_id, command.name)
            if decision.action == ResourceAction.STOP:
                for process in processes:
                    if process.poll() is None:
                        if paused:
                            os.killpg(process.pid, signal.SIGCONT)
                        os.killpg(process.pid, signal.SIGTERM)
                deadline = time.monotonic() + 60
                alive = [process for process in processes if process.poll() is None]
                while alive and time.monotonic() < deadline:
                    time.sleep(1)
                    alive = [process for process in alive if process.poll() is None]
                for process in alive:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise ResourceLimitExceeded(decision.reasons)
            if decision.action == ResourceAction.PAUSE and not paused:
                for process in processes:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGSTOP)
                paused = True
            elif decision.action == ResourceAction.RUN and paused:
                for process in processes:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGCONT)
                paused = False
            time.sleep(5)
        failed = next((process for process in processes if process.returncode), None)
        if failed:
            raise subprocess.CalledProcessError(failed.returncode, command.argv)

    def load_checkpoint(self, path: Path, shard_id: str) -> PipelineCheckpoint:
        if not path.is_file():
            return PipelineCheckpoint(shard_id)
        return PipelineCheckpoint(**json.loads(path.read_text()))

    def save_checkpoint(self, path: Path, checkpoint: PipelineCheckpoint) -> None:
        atomic_write_json(path, asdict(checkpoint))

    def is_infrastructure_error(self, error: Exception) -> bool:
        text = str(error).lower()
        return any(term in text for term in ("authentication", "checkpoint", "checksum", "no optix device", "resource hard"))

    def stop(self, context: ShardContext, command: str, reason: str, checkpoint: PipelineCheckpoint, exit_code: int) -> None:
        report = EscalationReport(
            context.source,
            context.shard_id,
            command,
            reason,
            tuple(self.resource_guard.last_five_minutes()),
            f"python -m data_toolkit.pipeline.cli resume --source {context.source} --shard {context.shard_id}",
            datetime.now(timezone.utc).isoformat(),
        )
        self.save_checkpoint(context.work_root / "checkpoint.json", checkpoint)
        atomic_write_json(self.config.paths.data2_root / "control/reports/escalations" / f"{context.shard_id}.json", asdict(report))
        raise PipelineStopped(report, exit_code)
```

Import `ResourceLimitExceeded` from `resources.py`. Register explicit internal handlers named `stage_raw`, `cleanup_voxels_256`, `cleanup_voxels_512`, `cleanup_voxels_1024`, `validate_outputs`, `build_packs`, `archive_raw`, and `cleanup_local`; reject any unknown internal name. Handlers must enforce this order: stage selected verified raw and compatibility metadata, validate each encoded resolution, delete only that resolution's dual-grid/PBR voxels, validate all final outputs, publish common/SS/shape/PBR packs, copy the selected raw archive to data3, verify file count/bytes/SHA, delete only unreferenced data2 raw, then clear local scratch.

Add `build_services(config: PipelineConfig) -> PipelineServices`. `PipelineServices` owns the registry store, resource guard, validators, internal handlers, and `PipelineRunner`; it exposes `build_registry`, `plan`, `run`, `resume`, `audit`, and `report` methods used by the CLI in Task 11. Construction must be side-effect free so read-only `plan` does not create data directories.

For production, `PipelineServices.plan` reads the pilot p95 peak-local-bytes per asset, computes `local_usable_bytes` after subtracting `max(15% filesystem size, 120 GiB)` from current free space, calls `plan_work_batches`, and atomically freezes the returned SHA lists as `batch000.txt`, `batch001.txt`, and so on. Resume always reuses those manifests even if free space later changes. `run` processes and audits one batch at a time; each batch publishes eight pack volumes and one raw archive before local cleanup. A logical shard becomes complete only after its index verifies every frozen batch.

- [ ] **Step 4: Test retries and rolling quality gates**

Add tests that feed 500 outcomes with 51 end-to-end failures and 26 schema failures. Both must stop. Lower rates must allow admission. Assert every stop report includes source, shard, command, reason, telemetry, completed counts, safe resume, and recovery choices.

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_orchestrator.py -v`

Expected: all resume, retry, resource, and quality tests pass.

- [ ] **Step 5: Commit Task 10**

```bash
git add data_toolkit/pipeline/orchestrator.py tests/data_toolkit/test_orchestrator.py
git commit -m "feat: orchestrate resumable preprocessing shards"
```

---

### Task 11: CLI, Reports, Audits, and Runbook

**Files:**
- Create: `data_toolkit/pipeline/reporting.py`
- Create: `data_toolkit/pipeline/cli.py`
- Create: `tests/data_toolkit/test_reporting.py`
- Create: `tests/data_toolkit/test_cli.py`
- Modify: `tests/data_toolkit/conftest.py`
- Modify: `data_toolkit/README.md:1-214`

**Interfaces:**
- Consumes: all pipeline services.
- Produces: `preflight`, `registry`, `plan`, `run`, `resume`, `audit`, `report`; JSON and Markdown reports.

- [ ] **Step 1: Write failing CLI/report tests**

```python
# tests/data_toolkit/test_reporting.py
import pandas as pd
from data_toolkit.pipeline.reporting import capacity_projection

def test_capacity_projection_uses_p95_and_headroom():
    result = capacity_projection(pd.DataFrame({"final_bytes": [100, 120, 140, 160]}), 500_777, 1.25)
    assert result["assets"] == 500_777
    assert result["projected_bytes"] >= 500_777 * 140 * 1.25
```

```python
# tests/data_toolkit/test_cli.py
from data_toolkit.pipeline.cli import main

def test_plan_is_read_only(tmp_config, capsys):
    assert main(["plan", "--config", str(tmp_config), "--gate", "smoke"]) == 0
    output = capsys.readouterr().out
    assert "dump_mesh" in output and "build_packs" in output
    assert not tmp_config.parent.joinpath("data2").exists()
```

Append this concrete fixture:

```python
# tests/data_toolkit/conftest.py
import yaml

@pytest.fixture
def tmp_config(tmp_path):
    raw = yaml.safe_load(Path("data_toolkit/configs/multiview_preprocess.yaml").read_text())
    raw["paths"] = {
        "data2_root": str(tmp_path / "data2"),
        "data3_root": str(tmp_path / "data3"),
        "local_root": str(tmp_path / "local"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path
```

- [ ] **Step 2: Verify CLI/reporting imports fail**

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_cli.py tests/data_toolkit/test_reporting.py -v`

Expected: FAIL importing Task 11 modules.

- [ ] **Step 3: Implement reporting**

```python
# data_toolkit/pipeline/reporting.py
from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd

def capacity_projection(pilot: pd.DataFrame, total_assets: int, headroom: float = 1.25) -> dict:
    p95 = float(pilot["final_bytes"].quantile(0.95))
    return {"assets": total_assets, "p95_final_bytes": p95, "headroom": headroom, "projected_bytes": int(p95 * total_assets * headroom)}

def write_report(root: Path, name: str, payload: dict) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    json_path, markdown_path = root / f"{name}.json", root / f"{name}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    lines = [f"# {name}", "", f"Generated: {datetime.now(timezone.utc).isoformat()}", ""]
    lines.extend(f"- {key}: {value}" for key, value in sorted(payload.items()))
    markdown_path.write_text("\n".join(lines) + "\n")
    return json_path, markdown_path
```

Add builders for source counts, failure categories, resource peaks, throughput quantiles, byte quantiles, FP16 parity, checksums, split overlap, capacity, and handoff.

- [ ] **Step 4: Implement CLI parsing and dispatch**

```python
# data_toolkit/pipeline/cli.py
import argparse
from pathlib import Path
from typing import Sequence
from .config import load_config
from .orchestrator import PipelineStopped, build_services
from .preflight import PreflightStatus, run_preflight

def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    children = root.add_subparsers(dest="command", required=True)
    for name in ("preflight", "registry", "plan", "run", "resume", "audit", "report"):
        child = children.add_parser(name)
        child.add_argument("--config", type=Path, required=True)
    for name in ("plan", "run"):
        children.choices[name].add_argument("--gate", choices=("smoke", "pilot", "production"), required=True)
    for name in ("plan", "run", "resume", "audit"):
        children.choices[name].add_argument("--source")
        children.choices[name].add_argument("--shard")
    children.choices["plan"].add_argument("--count", type=int)
    children.choices["report"].add_argument("--gate", choices=("pilot", "production"))
    children.choices["report"].add_argument("--hardware-check", action="store_true")
    return root

def dispatch(args, config) -> int:
    services = build_services(config)
    try:
        if args.command == "registry":
            services.build_registry()
        elif args.command == "plan":
            for line in services.plan(args.gate, args.source, args.shard, args.count):
                print(line)
        elif args.command == "run":
            services.run(args.gate, args.source, args.shard)
        elif args.command == "resume":
            services.resume(args.source, args.shard)
        elif args.command == "audit":
            services.audit(args.source, args.shard)
        elif args.command == "report":
            services.report(args.gate, args.hardware_check)
        else:
            raise AssertionError(f"unhandled command: {args.command}")
    except PipelineStopped as error:
        print(error.report.reason)
        return error.exit_code
    return 0

def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "preflight":
        results = run_preflight(config)
        for item in results:
            print(f"{item.source}: {item.status}: {item.message}")
        return 2 if any(item.status != PreflightStatus.READY for item in results) else 0
    return dispatch(args, config)

if __name__ == "__main__":
    raise SystemExit(main())
```

`plan` creates no directory. `build_services` and `plan` must remain side-effect free until a mutating service method is called. Production requires passed smoke/pilot reports with the same config hash. Exit codes: 0 success, 2 operator-blocked, 3 resource-stop, 4 data-quality-stop; encode the category in `PipelineStopped` rather than deriving it from arbitrary message text.

- [ ] **Step 5: Rewrite the operator runbook**

Document roots, environment/tests, HSSD access, manual archives, CLI commands, resource thresholds, pause/stop, escalation reports, archive verification, cleanup, and Stage pack extraction.

- [ ] **Step 6: Test and commit**

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_cli.py tests/data_toolkit/test_reporting.py -v`

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit -v`

Expected: all unit tests pass; external tests remain marked.

```bash
git add data_toolkit/pipeline/reporting.py data_toolkit/pipeline/cli.py data_toolkit/README.md tests/data_toolkit/conftest.py tests/data_toolkit/test_reporting.py tests/data_toolkit/test_cli.py
git commit -m "feat: add preprocessing CLI and operator reports"
```

---

### Task 12: Synthetic End-to-End Integration

**Files:**
- Create: `tests/data_toolkit/test_pipeline_integration.py`
- Create: `tests/data_toolkit/fixtures/fake_leaf_worker.py`
- Modify: `tests/data_toolkit/conftest.py`
- Modify: `data_toolkit/pipeline/commands.py`

**Interfaces:**
- Consumes: complete CLI and services.
- Produces: proof that a two-asset shard validates, packs, cleans, and resumes without external data.

- [ ] **Step 1: Write the failing integration test**

```python
# tests/data_toolkit/test_pipeline_integration.py
import tarfile
import pytest
from data_toolkit.pipeline.cli import main

@pytest.mark.integration
def test_two_asset_shard_runs_and_resumes(synthetic_config, monkeypatch):
    monkeypatch.setenv("PIXAL3D_LEAF_WORKER", "tests/data_toolkit/fixtures/fake_leaf_worker.py")
    arguments = ["--config", str(synthetic_config), "--source", "Synthetic", "--shard", "Synthetic-00000"]
    assert main(["run", *arguments, "--gate", "smoke"]) == 0
    assert main(["resume", *arguments]) == 0
    packs = sorted((synthetic_config.parent / "data2/prepared").rglob("*.tar"))
    assert len(packs) == 8
    for path in packs:
        with tarfile.open(path) as bundle:
            assert bundle.getmembers()
```

Append a fixture that rewrites `tmp_config` with `sources: [Synthetic]`, `evaluation_sources: []`, `shard_size: 2`, and creates a two-row canonical metadata fixture plus `instances.txt`. Return only the config path, and derive all output assertions from the configured paths rather than production roots:

```python
# tests/data_toolkit/conftest.py
@pytest.fixture
def synthetic_config(tmp_config):
    raw = yaml.safe_load(tmp_config.read_text())
    raw["sources"] = ["Synthetic"]
    raw["evaluation_sources"] = []
    raw["shard_size"] = 2
    tmp_config.write_text(yaml.safe_dump(raw, sort_keys=False))
    return tmp_config
```

The integration setup service creates deterministic SHA rows and the shard instance manifest on first mutating `run`; `plan` remains read-only.

- [ ] **Step 2: Verify synthetic dispatch is absent**

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_pipeline_integration.py -m integration -v`

Expected: FAIL because fake leaf dispatch is not connected.

- [ ] **Step 3: Implement fake outputs and test-only dispatch**

The fake worker uses atomic writers to create eight valid RGBA images/transforms, two SS files, and two shape/PBR files per resolution. It records command counts to prove resume does not regenerate valid outputs.

```python
def python_command(script: str, *args: str) -> tuple[str, ...]:
    override = os.environ.get("PIXAL3D_LEAF_WORKER")
    if override:
        return ("python", override, "--original-script", script, *args)
    return ("python", f"data_toolkit/{script}", *args)
```

Do not expose this variable in production config or docs.

- [ ] **Step 4: Run integration, suite, diff check, and commit**

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit/test_pipeline_integration.py -m integration -v`

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit -v`

Run: `git diff --check`

Expected: integration and suite pass; diff check is empty.

```bash
git add tests/data_toolkit/conftest.py tests/data_toolkit/test_pipeline_integration.py tests/data_toolkit/fixtures/fake_leaf_worker.py data_toolkit/pipeline/commands.py
git commit -m "test: cover preprocessing pipeline end to end"
```

### Task 13: External Access and Hardware Preflight

**Files:**
- Runtime output: `/root/data2/pixal3d/control/reports/preflight.json`
- Runtime output: `/root/data2/pixal3d/control/reports/hardware.json`

**Interfaces:**
- Consumes: production config and completed code.
- Produces: explicit ready or blocking report before bulk download.

- [ ] **Step 1: Run code verification**

Run: `conda run -n pixal3d python -m pytest tests/data_toolkit -v`

Expected: all unit and synthetic integration tests pass.

- [ ] **Step 2: Run source access preflight**

Run: `conda run -n pixal3d python -m data_toolkit.pipeline.cli preflight --config data_toolkit/configs/multiview_preprocess.yaml`

Expected when ready: exit 0 for ObjaverseXL Sketchfab/GitHub, ABO, HSSD, 3D-FUTURE, and Toys4k.

If exit 2, stop and ask the user immediately. The manual archives must be exactly:

```text
/root/data2/pixal3d/raw/3D-FUTURE/3D-FUTURE-model.zip
/root/data2/pixal3d/raw/Toys4k/toys4k_blend_files.zip
```

- [ ] **Step 3: Validate Blender and each GPU**

Run: `conda run -n pixal3d python -m data_toolkit.pipeline.cli report --config data_toolkit/configs/multiview_preprocess.yaml --hardware-check`

Expected: seven GPU entries, one distinct visible OptiX device each, non-empty cube render, no CPU fallback.

- [ ] **Step 4: Benchmark local/data2/data3 sequential I/O**

The hardware check writes, fsyncs, reads, and deletes one bounded 10 GiB fixture per root while no other workload runs. Expected fields: read/write MiB/s, elapsed time, free space before/after, and zero fixture files left behind.

- [ ] **Step 5: Stop on any failed preflight**

Send the report path, failed command, last five telemetry minutes, and recovery choices. Do not partially begin production downloads.

---

### Task 14: 100-Asset Smoke Gate

**Files:**
- Runtime manifests: `/root/data2/pixal3d/control/shards/smoke/`
- Runtime report: `/root/data2/pixal3d/control/reports/smoke/summary.json`

- [ ] **Step 1: Build source metadata and registries**

Run: `conda run -n pixal3d python -m data_toolkit.pipeline.cli registry --config data_toolkit/configs/multiview_preprocess.yaml`

Expected before deduplication: ObjaverseXL Sketchfab 168,307; ObjaverseXL GitHub 311,843; ABO 4,485; HSSD 6,670; 3D-FUTURE 9,472; total training pool 500,777; Toys4K 3,229. Require no split overlap and stable repeated counts.

- [ ] **Step 2: Plan 20 assets per source**

Run: `conda run -n pixal3d python -m data_toolkit.pipeline.cli plan --config data_toolkit/configs/multiview_preprocess.yaml --gate smoke`

Expected: five manifests and 100 unique canonical SHAs covering formats and size bands.

- [ ] **Step 3: Run smoke**

Run: `conda run -n pixal3d python -m data_toolkit.pipeline.cli run --config data_toolkit/configs/multiview_preprocess.yaml --gate smoke`

Expected: 100 terminal attempts; zero infrastructure, adapter, checkpoint, OptiX, and schema failures; valid packs.

- [ ] **Step 4: Inspect samples and telemetry**

Decode at least two `view00`/`view01` examples per source and output family. Verify pauses and floors were never bypassed.

- [ ] **Step 5: Stop on smoke failure**

Do not start pilot. Report failed commands, categories, sample paths, and telemetry.

---

### Task 15: 1,000-Asset Pilot and Capacity Decision

**Files:**
- Runtime manifests: `/root/data2/pixal3d/control/shards/pilot/`
- Runtime reports: `/root/data2/pixal3d/control/reports/pilot/`

- [ ] **Step 1: Plan the stratified pilot**

Run: `conda run -n pixal3d python -m data_toolkit.pipeline.cli plan --config data_toolkit/configs/multiview_preprocess.yaml --gate pilot --count 1000`

Expected: 1,000 unique SHAs stratified by source, extension, raw size, mesh complexity, materials, and alpha.

- [ ] **Step 2: Run pilot with FP32 output**

Run: `conda run -n pixal3d python -m data_toolkit.pipeline.cli run --config data_toolkit/configs/multiview_preprocess.yaml --gate pilot`

Expected: all attempts terminal, resource policy enforced, no unverified publication.

- [ ] **Step 3: Run FP16 parity**

Compare at least 32 decoded assets for shape and PBR at each resolution. Require exact coordinates, zero non-finite values, absolute-error p99 at most 0.01, and decode degradation at most 0.1%. If all pass, change `latent_dtype` to `float16`, rerun tests, commit, and rebuild pilot packs. Otherwise retain FP32.

- [ ] **Step 4: Produce projections**

Run: `conda run -n pixal3d python -m data_toolkit.pipeline.cli report --config data_toolkit/configs/multiview_preprocess.yaml --gate pilot`

Expected: p50/p95/p99 times, failures, resource peaks, byte distributions, total data2/data3/local projection, and ETA.

- [ ] **Step 5: Require user acceptance**

Stop and ask if final data projects above 16 TiB, archive above 26 TiB, success below 90%, a source has abnormal failures, or FP16 fails. Production waits for explicit acceptance.

---

### Task 16: First 5,000-Asset Production Shard

**Files:**
- Runtime shard index: `/root/data2/pixal3d/control/shards/ObjaverseXL_sketchfab/ObjaverseXL_sketchfab-00000/index.json`
- Runtime batch manifests: `/root/data2/pixal3d/control/shards/ObjaverseXL_sketchfab/ObjaverseXL_sketchfab-00000/batch*.txt`
- Runtime report: `/root/data2/pixal3d/control/reports/production/ObjaverseXL_sketchfab-00000.json`

- [ ] **Step 1: Dry-run the production DAG**

Run: `conda run -n pixal3d python -m data_toolkit.pipeline.cli plan --config data_toolkit/configs/multiview_preprocess.yaml --gate production --source ObjaverseXL_sketchfab --shard ObjaverseXL_sketchfab-00000`

Expected: ordered resolutions, views 0-1 only, cleanup after validation, archive after publication.

- [ ] **Step 2: Run the first production shard**

Run: `conda run -n pixal3d python -m data_toolkit.pipeline.cli run --config data_toolkit/configs/multiview_preprocess.yaml --gate production --source ObjaverseXL_sketchfab --shard ObjaverseXL_sketchfab-00000`

Expected: terminal rows, valid packs, verified data3 archive, reclaimed local scratch, maintained floors.

- [ ] **Step 3: Verify idempotent resume**

Run: `conda run -n pixal3d python -m data_toolkit.pipeline.cli resume --config data_toolkit/configs/multiview_preprocess.yaml --source ObjaverseXL_sketchfab --shard ObjaverseXL_sketchfab-00000`

Expected: validators run, valid outputs are not regenerated, exit 0.

- [ ] **Step 4: Audit before next shard**

Run: `conda run -n pixal3d python -m data_toolkit.pipeline.cli audit --config data_toolkit/configs/multiview_preprocess.yaml --source ObjaverseXL_sketchfab --shard ObjaverseXL_sketchfab-00000`

Expected: checksums pass, success at least 90%, no split leakage, usage inside projections.

---

### Task 17: Full Production and Training Handoff

**Files:**
- Runtime manifests: `/root/data2/pixal3d/control/shards/`
- Runtime reports: `/root/data2/pixal3d/control/reports/production/`
- Runtime handoff: `/root/data2/pixal3d/control/splits/training_handoff.json`

- [ ] **Step 1: Finish ObjaverseXL Sketchfab one shard at a time**

Never schedule a next shard before audit. Never overlap data3 archive with another CPU/GPU-heavy phase.

- [ ] **Step 2: Audit the source**

Run: `conda run -n pixal3d python -m data_toolkit.pipeline.cli audit --config data_toolkit/configs/multiview_preprocess.yaml --source ObjaverseXL_sketchfab`

Expected: every canonical asset complete or quarantined, success at least 90%, verified packs/archives, exact failures.

- [ ] **Step 3: Repeat fixed source order**

Process ObjaverseXL GitHub, ABO, HSSD, then 3D-FUTURE. Stop immediately for authentication, checkpoint, checksum, resource hard limit, rolling success below 90%, or schema error above 5%.

- [ ] **Step 4: Prepare Toys4K separately**

Generate evaluation-only conditions and ground truth. Audit no Toys4K SHA occurs in training packs.

- [ ] **Step 5: Run the global audit**

Run: `conda run -n pixal3d python -m data_toolkit.pipeline.cli audit --config data_toolkit/configs/multiview_preprocess.yaml`

Expected:

```text
all source rows resolve to complete, quarantined, or duplicate
no SHA overlap across train, validation, and Toys4K
all pack and archive checksums valid
all common, SS, shape, and PBR families represented
source and global success rates at least 90 percent
data2, data3, and local usage inside configured limits
```

- [ ] **Step 6: Materialize one handoff shard per Stage**

Extract common+SS for Stage 1, common+shape for Stage 2, and common+shape+PBR for Stage 3 under `/root/pixal3d-data/train/active`. Load one sample per resolution and anchor with Pixal3D dataset classes.

- [ ] **Step 7: Freeze handoff and stop preprocessing**

Write config hash, registry checksum, pack checksums, counts, splits, resolutions, anchors, and path mappings to `training_handoff.json`. Confirm no preprocessing process remains before fine-tuning.

---

## Final Verification

After code Tasks 1-12:

```bash
conda run -n pixal3d python -m compileall -q data_toolkit
conda run -n pixal3d python -m pytest tests/data_toolkit -v
git diff --check
git log --oneline --max-count=15
```

Expected: compile exit 0, all tests pass, no whitespace errors, focused commits.

Before declaring preprocessing complete:

```bash
conda run -n pixal3d python -m data_toolkit.pipeline.cli audit --config data_toolkit/configs/multiview_preprocess.yaml
```

Expected: global audit exit 0 with no checksum, split, schema, resource, or handoff error.
