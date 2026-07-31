from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from .orchestrator import (
    _atomic_write_bytes_nofollow,
    _read_regular_bytes_nofollow,
)


GPU_POLICY_SCHEMA_VERSION = 1


class GpuPolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class GpuRuntimePolicy:
    target_percent: int
    hard_percent: int


def _validate(
    policy: GpuRuntimePolicy,
    canonical_target_percent: int,
) -> None:
    if (
        type(policy.target_percent) is not int
        or type(policy.hard_percent) is not int
        or policy.target_percent != canonical_target_percent
        or not 0 < policy.target_percent < policy.hard_percent <= 100
    ):
        raise GpuPolicyError(
            "GPU runtime policy must preserve the canonical target and "
            "satisfy 0 < target < hard <= 100"
        )


def read_gpu_runtime_policy(
    path: Path,
    *,
    canonical_target_percent: int,
    canonical_hard_percent: int,
) -> GpuRuntimePolicy:
    fallback = GpuRuntimePolicy(
        canonical_target_percent,
        canonical_hard_percent,
    )
    payload = _read_regular_bytes_nofollow(Path(path), missing_ok=True)
    if payload is None:
        _validate(fallback, canonical_target_percent)
        return fallback
    try:
        value = json.loads(payload)
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "target_percent",
            "hard_percent",
            "updated_at",
        }:
            raise ValueError("unexpected fields")
        if value["schema_version"] != GPU_POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported schema version")
        updated_at = datetime.fromisoformat(value["updated_at"])
        if updated_at.tzinfo is None:
            raise ValueError("naive update time")
        policy = GpuRuntimePolicy(
            value["target_percent"],
            value["hard_percent"],
        )
        _validate(policy, canonical_target_percent)
        return policy
    except (GpuPolicyError, TypeError, ValueError, UnicodeError) as error:
        raise GpuPolicyError(
            f"invalid GPU runtime policy: {error}"
        ) from error


def write_gpu_runtime_policy(
    path: Path,
    policy: GpuRuntimePolicy,
    *,
    canonical_target_percent: int,
    now: datetime,
) -> None:
    _validate(policy, canonical_target_percent)
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise GpuPolicyError("GPU policy timestamp must be timezone-aware")
    value = {
        "schema_version": GPU_POLICY_SCHEMA_VERSION,
        **asdict(policy),
        "updated_at": now.astimezone(timezone.utc).isoformat(),
    }
    _atomic_write_bytes_nofollow(
        Path(path),
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"),
    )
