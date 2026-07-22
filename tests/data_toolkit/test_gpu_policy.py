from datetime import datetime, timezone

import pytest

from data_toolkit.pipeline.gpu_policy import (
    GpuPolicyError,
    GpuRuntimePolicy,
    read_gpu_runtime_policy,
    write_gpu_runtime_policy,
)


NOW = datetime(2026, 7, 22, 13, 0, tzinfo=timezone.utc)


def test_missing_policy_uses_canonical_values(tmp_path):
    assert read_gpu_runtime_policy(
        tmp_path / "gpu_policy.json",
        canonical_target_percent=80,
        canonical_hard_percent=90,
    ) == GpuRuntimePolicy(80, 90)


def test_policy_round_trips_atomically(tmp_path):
    path = tmp_path / "control/runtime/gpu_policy.json"
    write_gpu_runtime_policy(
        path,
        GpuRuntimePolicy(80, 100),
        canonical_target_percent=80,
        now=NOW,
    )

    assert read_gpu_runtime_policy(
        path,
        canonical_target_percent=80,
        canonical_hard_percent=90,
    ) == GpuRuntimePolicy(80, 100)


@pytest.mark.parametrize(
    "policy",
    [
        GpuRuntimePolicy(True, 100),
        GpuRuntimePolicy(79, 100),
        GpuRuntimePolicy(80, 80),
        GpuRuntimePolicy(80, 101),
    ],
)
def test_write_rejects_invalid_or_target_mismatched_policy(tmp_path, policy):
    with pytest.raises(GpuPolicyError, match="target < hard <= 100"):
        write_gpu_runtime_policy(
            tmp_path / "gpu_policy.json",
            policy,
            canonical_target_percent=80,
            now=NOW,
        )


def test_malformed_policy_never_falls_back_silently(tmp_path):
    path = tmp_path / "gpu_policy.json"
    path.write_text('{"schema_version":1,"target_percent":80}')

    with pytest.raises(GpuPolicyError, match="invalid GPU runtime policy"):
        read_gpu_runtime_policy(
            path,
            canonical_target_percent=80,
            canonical_hard_percent=90,
        )
