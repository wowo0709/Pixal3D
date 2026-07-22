from datetime import datetime, timedelta, timezone
import os

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


def test_source_priority_defaults_to_first_manifest_appearance(tmp_path):
    queue = ProductionWorkQueue(tmp_path, lease_timeout=timedelta(minutes=5))
    queue.initialize("a" * 64, units(), now=NOW)

    assert queue.source_priority() == ("ABO", "HSSD")


def test_source_priority_round_trips_without_changing_manifest(tmp_path):
    queue = ProductionWorkQueue(tmp_path, lease_timeout=timedelta(minutes=5))
    queue.initialize("a" * 64, units(), now=NOW)
    manifest_before = queue.manifest_path.read_bytes()

    queue.set_source_priority(("HSSD", "ABO"), now=NOW)

    assert queue.source_priority() == ("HSSD", "ABO")
    assert queue.manifest_path.read_bytes() == manifest_before


@pytest.mark.parametrize(
    "sources",
    [(), ("ABO",), ("ABO", "ABO"), ("ABO", "unknown")],
)
def test_source_priority_rejects_incomplete_duplicate_and_unknown_sources(
    tmp_path, sources
):
    queue = ProductionWorkQueue(tmp_path, lease_timeout=timedelta(minutes=5))
    queue.initialize("a" * 64, units(), now=NOW)

    with pytest.raises(ValueError, match="every queue source exactly once"):
        queue.set_source_priority(sources, now=NOW)


def test_malformed_source_priority_stops_reads_and_claims(tmp_path):
    queue = ProductionWorkQueue(tmp_path, lease_timeout=timedelta(minutes=5))
    queue.initialize("a" * 64, units(), now=NOW)
    queue.priority_path.write_text(
        '{"schema_version":1,"sources":["ABO"]}'
    )

    with pytest.raises(ValueError, match="source priority"):
        queue.source_priority()
    with pytest.raises(ValueError, match="source priority"):
        queue.claim("node17", now=NOW, token="token")


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
    queue.assert_config_hash("a" * 64)
    with pytest.raises(ValueError, match="config hash mismatch"):
        queue.assert_config_hash("b" * 64)


def test_released_unit_retries_then_becomes_terminal_failure(tmp_path):
    queue = ProductionWorkQueue(
        tmp_path, lease_timeout=timedelta(minutes=5), max_attempts=3
    )
    queue.initialize("a" * 64, units()[:1], now=NOW)

    for attempt in range(1, 4):
        lease = queue.claim(
            f"node{attempt}",
            now=NOW + timedelta(minutes=attempt),
            token=f"token{attempt}",
        )
        assert lease.attempt == attempt
        queue.release(
            lease,
            reason="infrastructure failure",
            now=NOW + timedelta(minutes=attempt, seconds=1),
        )

    assert queue.claim(
        "node4", now=NOW + timedelta(minutes=4), token="token4"
    ) is None
    assert queue.status(now=NOW + timedelta(minutes=4))["failed"] == 1


def test_adopt_completed_marks_verified_legacy_batch(tmp_path):
    queue = ProductionWorkQueue(tmp_path, lease_timeout=timedelta(minutes=5))
    queue.initialize("a" * 64, units()[:1], now=NOW)

    queue.adopt_completed(units()[0], now=NOW, node_id="legacy-node17")

    assert queue.claim("node16", now=NOW, token="token") is None
    assert queue.status(now=NOW)["completed"] == 1


def test_snapshot_reports_live_node_batch_attempt_and_stage(tmp_path):
    queue = ProductionWorkQueue(tmp_path, lease_timeout=timedelta(minutes=5))
    queue.initialize("a" * 64, units()[:1], now=NOW)
    lease = queue.claim("node17", now=NOW, token="token")
    queue.heartbeat(
        lease, stage="render_cond", now=NOW + timedelta(seconds=30)
    )

    snapshot = queue.snapshot(now=NOW + timedelta(seconds=31))

    assert snapshot["counts"]["running"] == 1
    assert snapshot["active"] == [
        {
            "unit_id": "ABO--ABO-00000--batch000",
            "source": "ABO",
            "shard_id": "ABO-00000",
            "batch_id": "batch000",
            "node_id": "node17",
            "attempt": 1,
            "stage": "render_cond",
            "heartbeat_at": "2026-07-22T04:30:30+00:00",
            "stale": False,
        }
    ]


def test_orphaned_claim_directory_is_recovered_after_lease_timeout(tmp_path):
    queue = ProductionWorkQueue(tmp_path, lease_timeout=timedelta(minutes=5))
    queue.initialize("a" * 64, units()[:1], now=NOW)
    orphan = queue.leases_root / units()[0].unit_id
    orphan.mkdir()
    old = NOW.timestamp()
    os.utime(orphan, (old, old))

    lease = queue.claim(
        "node16", now=NOW + timedelta(minutes=6), token="replacement"
    )

    assert lease is not None
    assert lease.attempt == 1
