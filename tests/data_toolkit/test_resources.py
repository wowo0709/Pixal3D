from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from data_toolkit.pipeline import resources
from data_toolkit.pipeline.resources import (
    GpuMetric,
    ProjectStorageAccounting,
    ResourceAction,
    ResourceDecision,
    ResourceGuard,
    ResourceLimitExceeded,
    ResourcePolicy,
    ResourceSampler,
    ResourceSnapshot,
    TelemetryWriter,
)


def sample(now, **changes):
    base = ResourceSnapshot(
        now,
        20.0,
        10.0,
        1.0,
        400.0,
        0,
        300.0,
        40.0,
        1.0,
        20.0,
        1.0,
        30.0,
    )
    return replace(base, **changes)


def test_cpu_soft_and_hard_durations(config):
    start = datetime(2026, 7, 16, tzinfo=timezone.utc)
    policy = ResourcePolicy(config.limits)
    assert (
        policy.evaluate(
            sample(start, cpu_percent=85.0, monotonic_seconds=0.0)
        ).action
        == ResourceAction.RUN
    )
    assert (
        policy.evaluate(
            sample(
                start + timedelta(minutes=2),
                cpu_percent=85.0,
                monotonic_seconds=120.0,
            )
        ).action
        == ResourceAction.PAUSE
    )
    policy = ResourcePolicy(config.limits)
    policy.evaluate(sample(start, cpu_percent=95.0, monotonic_seconds=0.0))
    assert (
        policy.evaluate(
            sample(
                start + timedelta(minutes=5),
                cpu_percent=95.0,
                monotonic_seconds=300.0,
            )
        ).action
        == ResourceAction.STOP
    )


def test_policy_elapsed_thresholds_ignore_forward_and_backward_wall_clock_jumps(
    config,
):
    wall = datetime(2026, 7, 16, tzinfo=timezone.utc)
    policy = ResourcePolicy(config.limits)

    assert (
        policy.evaluate(
            sample(wall, cpu_percent=85.0, monotonic_seconds=1_000.0)
        ).action
        == ResourceAction.RUN
    )
    assert (
        policy.evaluate(
            sample(
                wall + timedelta(days=30),
                cpu_percent=85.0,
                monotonic_seconds=1_119.0,
            )
        ).action
        == ResourceAction.RUN
    )
    assert (
        policy.evaluate(
            sample(
                wall - timedelta(days=30),
                cpu_percent=85.0,
                monotonic_seconds=1_120.0,
            )
        ).action
        == ResourceAction.PAUSE
    )


def test_policy_soft_duration_resets_after_stable_sample(config):
    wall = datetime(2026, 7, 16, tzinfo=timezone.utc)
    policy = ResourcePolicy(config.limits)

    policy.evaluate(sample(wall, cpu_percent=85.0, monotonic_seconds=0.0))
    policy.evaluate(sample(wall, cpu_percent=20.0, monotonic_seconds=119.0))
    assert (
        policy.evaluate(
            sample(wall, cpu_percent=85.0, monotonic_seconds=1_000.0)
        ).action
        == ResourceAction.RUN
    )
    assert (
        policy.evaluate(
            sample(wall, cpu_percent=85.0, monotonic_seconds=1_120.0)
        ).action
        == ResourceAction.PAUSE
    )


@pytest.mark.parametrize(
    ("field", "high", "reason"),
    [
        ("cpu_percent", 85.0, "CPU soft duration"),
        ("load_1m", 73.0, "load soft duration"),
        ("io_wait_percent", 11.0, "I/O wait"),
    ],
)
def test_duration_soft_policies_require_continuous_violation(
    config, field, high, reason
):
    wall = datetime(2026, 7, 16, tzinfo=timezone.utc)
    policy = ResourcePolicy(config.limits)

    assert (
        policy.evaluate(sample(wall, monotonic_seconds=0.0, **{field: high})).action
        == ResourceAction.RUN
    )
    decision = policy.evaluate(
        sample(wall, monotonic_seconds=120.0, **{field: high})
    )
    assert decision.action == ResourceAction.PAUSE
    assert reason in decision.reasons
    policy.evaluate(sample(wall, monotonic_seconds=121.0))
    assert (
        policy.evaluate(
            sample(wall, monotonic_seconds=1_000.0, **{field: high})
        ).action
        == ResourceAction.RUN
    )


def test_cpu_hard_duration_resets_after_stable_sample(config):
    wall = datetime(2026, 7, 16, tzinfo=timezone.utc)
    policy = ResourcePolicy(config.limits)

    policy.evaluate(sample(wall, cpu_percent=95.0, monotonic_seconds=0.0))
    policy.evaluate(sample(wall, cpu_percent=20.0, monotonic_seconds=299.0))
    assert (
        policy.evaluate(
            sample(wall, cpu_percent=95.0, monotonic_seconds=1_000.0)
        ).action
        == ResourceAction.RUN
    )
    assert (
        policy.evaluate(
            sample(wall, cpu_percent=95.0, monotonic_seconds=1_299.0)
        ).action
        == ResourceAction.PAUSE
    )
    assert (
        policy.evaluate(
            sample(wall, cpu_percent=95.0, monotonic_seconds=1_300.0)
        ).action
        == ResourceAction.STOP
    )


def test_local_absolute_floor_stops(config):
    decision = ResourcePolicy(config.limits).evaluate(
        sample(datetime.now(timezone.utc), local_free_gib=119.0)
    )
    assert decision.action == ResourceAction.STOP


def test_nfs_free_space_floors_stop(config):
    now = datetime.now(timezone.utc)
    assert (
        ResourcePolicy(config.limits).evaluate(
            sample(now, data2_fs_free_tib=1.9)
        ).action
        == ResourceAction.STOP
    )
    assert (
        ResourcePolicy(config.limits).evaluate(
            sample(now, data3_fs_free_tib=3.9)
        ).action
        == ResourceAction.STOP
    )


def test_small_swap_activity_does_not_pause_new_work(config):
    decision = ResourcePolicy(config.limits).evaluate(
        sample(datetime.now(timezone.utc), swap_in_bytes=4096)
    )
    assert decision.action == ResourceAction.RUN


def test_sustained_swap_activity_pauses_new_work(config):
    policy = ResourcePolicy(config.limits)
    start = datetime(2026, 7, 16, tzinfo=timezone.utc)
    for index in range(2):
        assert (
            policy.evaluate(
                sample(
                    start,
                    swap_in_bytes=128 * 1024**2,
                    monotonic_seconds=index * 20.0,
                )
            ).action
            == ResourceAction.RUN
        )
    assert (
        policy.evaluate(
            sample(
                start,
                swap_in_bytes=128 * 1024**2,
                monotonic_seconds=40.0,
            )
        ).action
        == ResourceAction.PAUSE
    )


def test_cpu_temperature_requires_sustained_soft_threshold(config):
    policy = ResourcePolicy(config.limits)
    start = datetime(2026, 7, 16, tzinfo=timezone.utc)
    assert (
        policy.evaluate(
            sample(start, cpu_max_temperature_celsius=86, monotonic_seconds=0)
        ).action
        == ResourceAction.RUN
    )
    assert (
        policy.evaluate(
            sample(start, cpu_max_temperature_celsius=86, monotonic_seconds=29)
        ).action
        == ResourceAction.RUN
    )
    assert (
        policy.evaluate(
            sample(start, cpu_max_temperature_celsius=86, monotonic_seconds=30)
        ).action
        == ResourceAction.PAUSE
    )


def test_gpu_temperature_hard_threshold_stops_immediately(config):
    decision = ResourcePolicy(config.limits).evaluate(
        sample(
            datetime.now(timezone.utc),
            gpu_metrics=(GpuMetric(0, 0, 0, 88, 0),),
        )
    )
    assert decision.action == ResourceAction.STOP


class FakePsutil:
    def __init__(self, roots, *, local_free_gib=300.0):
        self.roots = roots
        self.local_free_gib = local_free_gib
        self.swap_values = iter((10_000, 14_096))

    def cpu_percent(self, interval=None):
        assert interval is None
        return 37.5

    def cpu_times_percent(self, interval=None):
        assert interval is None
        return SimpleNamespace(iowait=2.5)

    def getloadavg(self):
        return (12.0, 10.0, 8.0)

    def virtual_memory(self):
        return SimpleNamespace(available=200 * 1024**3)

    def swap_memory(self):
        return SimpleNamespace(sin=next(self.swap_values))

    def disk_usage(self, path):
        if Path(path) == self.roots.local_root:
            return SimpleNamespace(
                total=1_000 * 1024**3,
                free=int(self.local_free_gib * 1024**3),
            )
        if Path(path) == self.roots.data2_root:
            return SimpleNamespace(total=30 * 1024**4, free=20 * 1024**4)
        if Path(path) == self.roots.data3_root:
            return SimpleNamespace(total=40 * 1024**4, free=30 * 1024**4)
        raise AssertionError(path)


def test_sampler_uses_cached_project_bytes_and_swap_delta(config, tmp_path):
    roots = replace(
        config.paths,
        data2_root=tmp_path / "data2",
        data3_root=tmp_path / "data3",
        local_root=tmp_path / "local",
    )
    config = replace(config, paths=roots)
    roots.data2_root.mkdir()
    roots.data3_root.mkdir()
    walk_calls = []

    def directory_size(root):
        walk_calls.append(root)
        return 7 if root == roots.data2_root else 11

    accounting = ProjectStorageAccounting(
        roots.data2_root,
        roots.data3_root,
        data2_bytes=1 * 1024**4,
        data3_bytes=2 * 1024**4,
        directory_size=directory_size,
    )
    runner_calls = []

    def gpu_runner(argv, **kwargs):
        runner_calls.append((argv, kwargs))
        return SimpleNamespace(stdout="0, 75, 1234, 55, 210.5\n")

    now = datetime(2026, 7, 16, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    monotonic_values = iter((100.0, 105.0))
    sampler = ResourceSampler(
        config,
        accounting,
        psutil_api=FakePsutil(roots),
        gpu_runner=gpu_runner,
        clock=lambda: now,
        monotonic_clock=lambda: next(monotonic_values),
    )

    first = sampler()
    second = sampler()

    assert first.swap_in_bytes == 0
    assert second.swap_in_bytes == 4096
    assert first.timestamp == datetime(2026, 7, 15, 18, 30, tzinfo=timezone.utc)
    assert first.monotonic_seconds == 100.0
    assert second.monotonic_seconds == 105.0
    assert first.data2_project_tib == 1.0
    assert first.data3_project_tib == 2.0
    assert first.gpu_metrics[0].power_watts == 210.5
    assert first.gpu_query_error is None
    assert walk_calls == []
    assert runner_calls[0] == (
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        {"capture_output": True, "text": True, "check": True, "timeout": 5.0},
    )

    assert sampler.reconcile_at_shard_boundary() == (7, 11)
    assert walk_calls == [roots.data2_root, roots.data3_root]


def test_project_accounting_accepts_registry_deltas_without_walking(tmp_path):
    data2 = tmp_path / "data2"
    data3 = tmp_path / "data3"
    data2.mkdir()
    data3.mkdir()
    accounting = ProjectStorageAccounting(
        data2,
        data3,
        data2_bytes=100,
        data3_bytes=200,
        directory_size=lambda root: pytest.fail("unexpected directory walk"),
    )

    accounting.record_registry_delta(data2 / "prepared" / "pack.tar", 25)
    accounting.record_registry_delta(data3 / "archive" / "raw.tar", -50)

    assert accounting.current_bytes() == (125, 150)
    with pytest.raises(ValueError, match="outside configured project roots"):
        accounting.record_registry_delta(tmp_path / "other" / "file", 1)
    with pytest.raises(ValueError, match="negative project accounting"):
        accounting.record_registry_delta(data2 / "prepared" / "pack.tar", -126)


@pytest.mark.parametrize(
    ("data2_bytes", "data3_bytes"), [(None, None), (0, None), (None, 0)]
)
def test_project_accounting_requires_initialized_registry_totals(
    tmp_path, data2_bytes, data3_bytes
):
    data2 = tmp_path / "data2"
    data3 = tmp_path / "data3"
    data2.mkdir()
    data3.mkdir()

    with pytest.raises(RuntimeError, match="registry totals must be initialized"):
        ProjectStorageAccounting(
            data2, data3, data2_bytes=data2_bytes, data3_bytes=data3_bytes
        )


def test_project_accounting_rejects_missing_or_non_directory_roots(tmp_path):
    data2 = tmp_path / "data2"
    data3 = tmp_path / "data3"
    data2.mkdir()
    data3.write_text("not a directory")

    with pytest.raises(RuntimeError, match="project root is not a directory"):
        ProjectStorageAccounting(data2, data3, data2_bytes=0, data3_bytes=0)
    with pytest.raises(RuntimeError, match="project root is not a directory"):
        ProjectStorageAccounting(
            tmp_path / "missing", data2, data2_bytes=0, data3_bytes=0
        )


def test_directory_size_reraises_walk_errors(tmp_path, monkeypatch):
    root = tmp_path / "data2"
    root.mkdir()

    def failing_walk(path, *, onerror):
        onerror(PermissionError("walk denied"))
        return ()

    monkeypatch.setattr(resources.os, "walk", failing_walk)

    with pytest.raises(PermissionError, match="walk denied"):
        resources._directory_size(root)


def test_reconciliation_failure_does_not_partially_publish_cache(tmp_path):
    data2 = tmp_path / "data2"
    data3 = tmp_path / "data3"
    data2.mkdir()
    data3.mkdir()

    def directory_size(root):
        if root == data2:
            return 1_000
        raise OSError("data3 walk failed")

    accounting = ProjectStorageAccounting(
        data2,
        data3,
        data2_bytes=100,
        data3_bytes=200,
        directory_size=directory_size,
    )

    with pytest.raises(OSError, match="data3 walk failed"):
        accounting.reconcile_at_shard_boundary()

    assert accounting.current_bytes() == (100, 200)


def test_reconciliation_walks_outside_lock_and_rejects_concurrent_delta(tmp_path):
    data2 = tmp_path / "data2"
    data3 = tmp_path / "data3"
    data2.mkdir()
    data3.mkdir()
    walk_started = threading.Event()
    release_walk = threading.Event()

    def directory_size(root):
        if root == data2:
            walk_started.set()
            assert release_walk.wait(timeout=2)
            return 1_000
        return 2_000

    accounting = ProjectStorageAccounting(
        data2,
        data3,
        data2_bytes=100,
        data3_bytes=200,
        directory_size=directory_size,
    )
    errors = []

    def reconcile():
        try:
            accounting.reconcile_at_shard_boundary()
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=reconcile)
    worker.start()
    assert walk_started.wait(timeout=2)
    accounting.record_registry_delta(data2 / "prepared" / "new.tar", 25)
    release_walk.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert "accounting changed during reconciliation" in str(errors[0])
    assert accounting.current_bytes() == (125, 200)


def test_parallel_registry_deltas_are_not_lost(tmp_path):
    data2 = tmp_path / "data2"
    data3 = tmp_path / "data3"
    data2.mkdir()
    data3.mkdir()
    accounting = ProjectStorageAccounting(
        data2, data3, data2_bytes=0, data3_bytes=0
    )
    start = threading.Barrier(9)

    def add_deltas():
        start.wait()
        for _ in range(1_000):
            accounting.record_registry_delta(data2 / "prepared" / "pack.tar", 1)

    workers = [threading.Thread(target=add_deltas) for _ in range(8)]
    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for worker in workers:
            worker.start()
        start.wait()
        for worker in workers:
            worker.join(timeout=5)
    finally:
        sys.setswitchinterval(previous_interval)

    assert all(not worker.is_alive() for worker in workers)
    assert accounting.current_bytes() == (8_000, 0)


def test_gpu_query_failure_is_reported_without_bypassing_hard_policy(
    config, tmp_path
):
    roots = replace(
        config.paths,
        data2_root=tmp_path / "data2",
        data3_root=tmp_path / "data3",
        local_root=tmp_path / "local",
    )
    config = replace(config, paths=roots)
    roots.data2_root.mkdir()
    roots.data3_root.mkdir()
    accounting = ProjectStorageAccounting(
        roots.data2_root, roots.data3_root, data2_bytes=0, data3_bytes=0
    )

    def gpu_runner(argv, **kwargs):
        raise OSError("driver unavailable")

    snapshot = ResourceSampler(
        config,
        accounting,
        psutil_api=FakePsutil(roots, local_free_gib=119.0),
        gpu_runner=gpu_runner,
        clock=lambda: datetime(2026, 7, 16, tzinfo=timezone.utc),
    )()

    assert snapshot.gpu_metrics == ()
    assert snapshot.gpu_query_error == "driver unavailable"
    decision = ResourcePolicy(config.limits).evaluate(snapshot)
    assert decision.action == ResourceAction.STOP
    assert decision.reasons == ("local free-space floor",)


def test_gpu_query_timeout_is_bounded_and_does_not_bypass_hard_policy(
    config, tmp_path
):
    roots = replace(
        config.paths,
        data2_root=tmp_path / "data2",
        data3_root=tmp_path / "data3",
        local_root=tmp_path / "local",
    )
    config = replace(config, paths=roots)
    roots.data2_root.mkdir()
    roots.data3_root.mkdir()
    accounting = ProjectStorageAccounting(
        roots.data2_root, roots.data3_root, data2_bytes=0, data3_bytes=0
    )
    calls = []

    def gpu_runner(argv, **kwargs):
        calls.append((argv, kwargs))
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    snapshot = ResourceSampler(
        config,
        accounting,
        psutil_api=FakePsutil(roots, local_free_gib=119.0),
        gpu_runner=gpu_runner,
        clock=lambda: datetime(2026, 7, 16, tzinfo=timezone.utc),
    )()

    assert calls[0][1]["timeout"] == 5.0
    assert "timed out after 5.0 seconds" in snapshot.gpu_query_error
    assert ResourcePolicy(config.limits).evaluate(snapshot).action == ResourceAction.STOP


class FakeClock:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        if isinstance(self.value, datetime):
            self.value += timedelta(seconds=seconds)
        else:
            self.value += seconds


class SequencePolicy:
    def __init__(self, *actions):
        self.actions = iter(actions)
        self.last = actions[-1]
        self.limits = SimpleNamespace(recovery_stable_seconds=30)

    def evaluate(self, snapshot):
        try:
            self.last = next(self.actions)
        except StopIteration:
            pass
        return ResourceDecision(self.last, (f"raw {self.last.value}",))


class RecordingTelemetry:
    def __init__(self):
        self.records = []

    def write(self, snapshot, decision, shard_id, command):
        self.records.append((snapshot, decision, shard_id, command))


def test_telemetry_serializes_iso_timestamp_and_syncs_every_thirty_seconds(
    tmp_path, monkeypatch
):
    now = datetime(2026, 7, 16, 12, 30, tzinfo=timezone.utc)
    monotonic = FakeClock(1_000.0)
    fsync_calls = []
    monkeypatch.setattr(
        "data_toolkit.pipeline.resources.os.fsync", lambda fd: fsync_calls.append(fd)
    )
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path, clock=monotonic)
    decision = ResourceDecision(ResourceAction.RUN, ())

    writer.write(sample(now), decision, "ABO-00000", "download")
    assert fsync_calls == []
    monotonic.advance(29)
    writer.write(
        sample(now + timedelta(days=30)), decision, "ABO-00000", "download"
    )
    assert fsync_calls == []
    monotonic.advance(6)
    writer.write(
        sample(now - timedelta(days=30)), decision, "ABO-00000", "download"
    )
    assert len(fsync_calls) == 1
    writer.close()
    assert len(fsync_calls) == 2

    records = [__import__("json").loads(line) for line in path.read_text().splitlines()]
    assert records[0]["timestamp"] == "2026-07-16T12:30:00+00:00"
    assert records[0]["shard_id"] == "ABO-00000"
    assert records[0]["command"] == "download"
    assert records[0]["action"] == "run"
    assert "monotonic_seconds" not in records[0]


def test_telemetry_close_closes_once_and_preserves_durability_failure(
    tmp_path, monkeypatch
):
    writer = TelemetryWriter(tmp_path / "telemetry.jsonl", clock=lambda: 0.0)
    monkeypatch.setattr(
        "data_toolkit.pipeline.resources.os.fsync",
        lambda fd: (_ for _ in ()).throw(OSError("fsync failed")),
    )

    with pytest.raises(OSError, match="fsync failed"):
        writer.close()

    assert writer._closed
    assert writer._stream.closed
    writer.close()


def test_telemetry_context_keeps_body_error_primary_when_close_fails(
    tmp_path, monkeypatch
):
    writer = TelemetryWriter(tmp_path / "telemetry.jsonl", clock=lambda: 0.0)
    monkeypatch.setattr(
        "data_toolkit.pipeline.resources.os.fsync",
        lambda fd: (_ for _ in ()).throw(OSError("fsync failed")),
    )

    with pytest.raises(ValueError, match="body failed") as raised:
        with writer:
            raise ValueError("body failed")

    assert isinstance(raised.value.__cause__, OSError)
    assert writer._closed
    assert writer._stream.closed


def test_guard_requires_thirty_uninterrupted_stable_seconds_after_pause():
    start = datetime(2026, 7, 16, tzinfo=timezone.utc)
    monotonic = FakeClock(0.0)
    telemetry = RecordingTelemetry()
    policy = SequencePolicy(
        ResourceAction.PAUSE,
        ResourceAction.RUN,
    )
    guard = ResourceGuard(
        lambda: sample(start), policy, telemetry, monotonic, lambda seconds: None
    )

    assert guard.check("shard", "command").action == ResourceAction.PAUSE
    monotonic.advance(5)
    assert guard.check("shard", "command").action == ResourceAction.PAUSE
    monotonic.advance(24)
    assert guard.check("shard", "command").action == ResourceAction.PAUSE
    monotonic.advance(6)
    assert guard.check("shard", "command").action == ResourceAction.RUN
    assert telemetry.records[-1][2:] == ("shard", "command")


def test_guard_recovery_ignores_forward_and_backward_wall_clock_jumps():
    wall = FakeClock(datetime(2026, 7, 16, tzinfo=timezone.utc))
    monotonic = FakeClock(0.0)
    guard = ResourceGuard(
        lambda: sample(wall()),
        SequencePolicy(ResourceAction.PAUSE, ResourceAction.RUN),
        RecordingTelemetry(),
        monotonic,
        lambda seconds: None,
    )

    assert guard.check("shard", "command").action == ResourceAction.PAUSE
    monotonic.advance(5)
    wall.advance(30 * 24 * 60 * 60)
    assert guard.check("shard", "command").action == ResourceAction.PAUSE
    monotonic.advance(24)
    wall.advance(-60 * 24 * 60 * 60)
    assert guard.check("shard", "command").action == ResourceAction.PAUSE
    monotonic.advance(6)
    assert guard.check("shard", "command").action == ResourceAction.RUN


def test_wait_for_admission_polls_every_five_seconds_and_initial_run_is_immediate():
    start = datetime(2026, 7, 16, tzinfo=timezone.utc)
    monotonic = FakeClock(0.0)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        monotonic.advance(seconds)

    guard = ResourceGuard(
        lambda: sample(start),
        SequencePolicy(ResourceAction.PAUSE, ResourceAction.RUN),
        RecordingTelemetry(),
        monotonic,
        sleep,
    )
    assert guard.wait_for_admission("shard", "download").action == ResourceAction.RUN
    assert sleeps == [5] * 7

    sleeps.clear()
    immediate = ResourceGuard(
        lambda: sample(start),
        SequencePolicy(ResourceAction.RUN),
        RecordingTelemetry(),
        monotonic,
        sleep,
    )
    assert immediate.wait_for_admission("shard", "download").action == ResourceAction.RUN
    assert sleeps == []


def test_wait_for_admission_raises_stop_without_sleeping():
    now = datetime(2026, 7, 16, tzinfo=timezone.utc)
    sleeps = []
    guard = ResourceGuard(
        lambda: sample(now),
        SequencePolicy(ResourceAction.STOP),
        RecordingTelemetry(),
        lambda: 0.0,
        sleeps.append,
    )

    with pytest.raises(ResourceLimitExceeded, match="raw stop") as raised:
        guard.wait_for_admission("shard", "download")

    assert raised.value.reasons == ("raw stop",)
    assert sleeps == []


def test_guard_history_is_bounded_and_json_serializable():
    start = datetime(2026, 7, 16, tzinfo=timezone.utc)
    wall = FakeClock(start)
    monotonic = FakeClock(0.0)
    guard = ResourceGuard(
        lambda: sample(wall()),
        SequencePolicy(ResourceAction.RUN),
        RecordingTelemetry(),
        monotonic,
        lambda seconds: None,
    )

    for _ in range(65):
        guard.check("shard", "command")
        wall.advance(5)
        monotonic.advance(5)

    history = guard.last_five_minutes()
    assert len(history) == 60
    assert history[0]["timestamp"] == (start + timedelta(seconds=25)).isoformat()
    __import__("json").dumps(history)
