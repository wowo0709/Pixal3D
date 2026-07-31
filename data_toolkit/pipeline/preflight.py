from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import huggingface_hub

from .config import PipelineConfig


class PreflightStatus(str, Enum):
    READY = "ready"
    BLOCKED = "blocked"
    ERROR = "error"


@dataclass(frozen=True)
class PreflightResult:
    source: str
    status: PreflightStatus
    message: str


def _manual(source: str, path: Path) -> PreflightResult:
    status = (
        PreflightStatus.READY if path.is_file() else PreflightStatus.BLOCKED
    )
    message = str(path) if path.is_file() else f"Missing manual archive: {path}"
    return PreflightResult(source, status, message)


def run_preflight(
    config: PipelineConfig, check_remote: bool = True
) -> tuple[PreflightResult, ...]:
    raw = config.paths.data2_root / "raw"
    results = [
        PreflightResult(
            "ObjaverseXL_sketchfab", PreflightStatus.READY, "ObjaverseXL API"
        ),
        PreflightResult(
            "ObjaverseXL_github", PreflightStatus.READY, "ObjaverseXL API"
        ),
        PreflightResult(
            "ABO", PreflightStatus.READY, "ABO public archive"
        ),
    ]
    if check_remote:
        try:
            huggingface_hub.whoami()
            huggingface_hub.hf_hub_download(
                repo_id="hssd/hssd-models",
                filename="README.md",
                repo_type="dataset",
            )
            hssd = PreflightResult(
                "HSSD", PreflightStatus.READY, "HSSD access verified"
            )
        except Exception as error:
            hssd = PreflightResult(
                "HSSD",
                PreflightStatus.BLOCKED,
                f"HSSD access failed: {error}",
            )
    else:
        hssd = PreflightResult(
            "HSSD", PreflightStatus.BLOCKED, "HSSD remote check skipped"
        )
    results.extend(
        [
            hssd,
            _manual(
                "3D-FUTURE",
                raw / "3D-FUTURE" / "3D-FUTURE-model.zip",
            ),
            _manual(
                "Toys4k", raw / "Toys4k" / "toys4k_blend_files.zip"
            ),
        ]
    )
    return tuple(results)
