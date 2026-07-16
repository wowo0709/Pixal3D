from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pytest
import yaml

from data_toolkit.pipeline.config import load_config
from data_toolkit.pipeline.runtime import (
    CanonicalRegistryBuilder,
    RuntimeReportBuilder,
)


@pytest.fixture
def config():
    return load_config(Path("data_toolkit/configs/multiview_preprocess.yaml"))


@pytest.fixture
def tmp_config(tmp_path):
    raw = yaml.safe_load(
        Path("data_toolkit/configs/multiview_preprocess.yaml").read_text()
    )
    raw["paths"] = {
        "data2_root": str(tmp_path / "data2"),
        "data3_root": str(tmp_path / "data3"),
        "local_root": str(tmp_path / "local"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def _hardware_evidence(config):
    return {
        "schema_version": 2,
        "artifact_type": "hardware_preflight_evidence",
        "config_hash": config.config_hash(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "software": {
            "cuda_version": "synthetic",
            "torch_version": "synthetic",
            "blender_version": config.render.blender_version,
            "optix_enabled": True,
        },
        "gpus": [
            {
                "index": index,
                "name": f"Synthetic GPU {index}",
                "cuda_visible_device": str(index),
                "cycles_device": config.render.cycles_device,
                "cpu_fallback_detected": False,
                "cube_render_sha256": f"{index + 1:064x}",
            }
            for index in range(7)
        ],
        "storage": {
            root: {
                "fixture_bytes": 10 * 1024**3,
                "write_elapsed_seconds": 10.0,
                "read_elapsed_seconds": 8.0,
                "write_sha256": "c" * 64,
                "read_sha256": "c" * 64,
                "total_bytes": 100 * 1024**4,
                "free_bytes_before": 50 * 1024**4,
                "free_bytes_after": 50 * 1024**4,
                "fixture_removed": True,
            }
            for root in ("local", "data2", "data3")
        },
        "source_measurements": {source: [256] for source in config.sources},
    }


@pytest.fixture
def synthetic_config(tmp_config):
    raw = yaml.safe_load(tmp_config.read_text())
    raw["sources"] = ["Synthetic"]
    raw["evaluation_sources"] = ["SyntheticEval"]
    raw["shard_size"] = 2
    tmp_config.write_text(yaml.safe_dump(raw, sort_keys=False))
    config = load_config(tmp_config)

    payloads = (b"pixal3d synthetic asset zero", b"pixal3d synthetic asset one")
    training = config.paths.data2_root / "control/metadata/Synthetic/metadata.csv"
    training.parent.mkdir(parents=True)
    training.write_text(
        "sha256,file_identifier,fixture_payload\n"
        + "".join(
            f"{sha256(payload).hexdigest()},objects/asset-{index}.glb,"
            f"{payload.decode('ascii')}\n"
            for index, payload in enumerate(payloads)
        )
    )
    evaluation = (
        config.paths.data2_root
        / "control/metadata/SyntheticEval/metadata.csv"
    )
    evaluation.parent.mkdir(parents=True)
    evaluation.write_text(
        "sha256,file_identifier\n"
        f"{sha256(b'pixal3d synthetic evaluation').hexdigest()},"
        "evaluation/unused.glb\n"
    )
    CanonicalRegistryBuilder(config)()

    hardware = (
        config.paths.data2_root / "control/report_inputs/hardware.json"
    )
    hardware.parent.mkdir(parents=True, exist_ok=True)
    hardware.write_text(json.dumps(_hardware_evidence(config)))
    RuntimeReportBuilder(config)(None, True)
    return tmp_config
