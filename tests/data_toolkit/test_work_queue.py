from datetime import datetime, timedelta, timezone

import pytest

from data_toolkit.pipeline.work_queue import (
    LeaseLostError,
    ProductionWorkQueue,
    WorkUnit,
)


NOW = datetime(2026, 7, 22, 4, 30, tzinfo=timezone.utc)


def units():
    return (
        WorkUnit("ABO", "ABO-00000", "batch000", 256),
        WorkUnit("HSSD", "HSSD-00000", "batch000", 256),
    )


def test_claims_are_atomic_and_distinct(tmp_path):
    queue = ProductionWorkQueue(tmp_path, lease_timeout=timedelta(minutes=5))
    queue.initialize("a" * 64, units(), now=NOW)

    first = queue.claim("node17", now=NOW, token="token-node17")
    second = queue.claim("node16", now=NOW, token="token-node16")

    assert first is not None
    assert second is not None
    assert first.unit != second.unit
    assert queue.claim("node18", now=NOW, token="token-node18") is None


def test_stale_lease_is_reclaimed_and_fenced(tmp_path):
    queue = ProductionWorkQueue(tmp_path, lease_timeout=timedelta(minutes=5))
    queue.initialize("a" * 64, units()[:1], now=NOW)
    stale = queue.claim("node17", now=NOW, token="old-token")

    replacement = queue.claim(
        "node16", now=NOW + timedelta(minutes=6), token="new-token"
    )

    assert replacement is not None
    assert replacement.unit == stale.unit
    assert replacement.attempt == 2
    with pytest.raises(LeaseLostError):
        queue.heartbeat(stale, stage="render_cond", now=NOW + timedelta(minutes=6))
    with pytest.raises(LeaseLostError):
        queue.complete(stale, now=NOW + timedelta(minutes=6))


def test_heartbeat_extends_lease_and_tracks_stage(tmp_path):
    queue = ProductionWorkQueue(tmp_path, lease_timeout=timedelta(minutes=5))
    queue.initialize("a" * 64, units()[:1], now=NOW)
    lease = queue.claim("node17", now=NOW, token="token")

    renewed = queue.heartbeat(
        lease, stage="dual_grid_512", now=NOW + timedelta(minutes=4)
    )

    assert renewed.stage == "dual_grid_512"
    assert queue.claim(
        "node16", now=NOW + timedelta(minutes=8), token="other"
    ) is None


def test_completed_unit_is_never_claimed_again(tmp_path):
    queue = ProductionWorkQueue(tmp_path, lease_timeout=timedelta(minutes=5))
    queue.initialize("a" * 64, units()[:1], now=NOW)
    lease = queue.claim("node17", now=NOW, token="token")

    queue.complete(lease, now=NOW + timedelta(minutes=1))

    assert queue.claim(
        "node16", now=NOW + timedelta(minutes=10), token="other"
    ) is None
    status = queue.status(now=NOW + timedelta(minutes=10))
    assert status == {
        "pending": 0,
        "running": 0,
        "completed": 1,
        "failed": 0,
        "stale": 0,
        "total": 1,
    }


def test_initialize_is_idempotent_but_rejects_different_scope(tmp_path):
    queue = ProductionWorkQueue(tmp_path, lease_timeout=timedelta(minutes=5))
    queue.initialize("a" * 64, units(), now=NOW)
    queue.initialize("a" * 64, units(), now=NOW + timedelta(minutes=1))

    with pytest.raises(ValueError, match="different production scope"):
        queue.initialize("b" * 64, units(), now=NOW)
