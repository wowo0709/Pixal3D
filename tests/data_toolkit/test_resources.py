from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_toolkit.pipeline.resources import (
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
    assert policy.evaluate(sample(start, cpu_percent=85.0)).action == ResourceAction.RUN
    assert (
        policy.evaluate(sample(start + timedelta(minutes=2), cpu_percent=85.0)).action
        == ResourceAction.PAUSE
    )
    policy = ResourcePolicy(config.limits)
    policy.evaluate(sample(start, cpu_percent=95.0))
    assert (
        policy.evaluate(sample(start + timedelta(minutes=5), cpu_percent=95.0)).action
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


def test_swap_activity_pauses_new_work(config):
    decision = ResourcePolicy(config.limits).evaluate(
        sample(datetime.now(timezone.utc), swap_in_bytes=4096)
    )
    assert decision.action == ResourceAction.PAUSE


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

    now = datetime(2026, 7, 16, tzinfo=timezone.utc)
    sampler = ResourceSampler(
        config,
        accounting,
        psutil_api=FakePsutil(roots),
        gpu_runner=gpu_runner,
        clock=lambda: now,
    )

    first = sampler()
    second = sampler()

    assert first.swap_in_bytes == 0
    assert second.swap_in_bytes == 4096
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
        {"capture_output": True, "text": True, "check": True},
    )

    assert sampler.reconcile_at_shard_boundary() == (7, 11)
    assert walk_calls == [roots.data2_root, roots.data3_root]


def test_project_accounting_accepts_registry_deltas_without_walking(tmp_path):
    data2 = tmp_path / "data2"
    data3 = tmp_path / "data3"
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
    accounting = ProjectStorageAccounting(roots.data2_root, roots.data3_root)

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


class FakeClock:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += timedelta(seconds=seconds)


class SequencePolicy:
    def __init__(self, *actions):
        self.actions = iter(actions)
        self.last = actions[-1]

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
    clock = FakeClock(now)
    fsync_calls = []
    monkeypatch.setattr(
        "data_toolkit.pipeline.resources.os.fsync", lambda fd: fsync_calls.append(fd)
    )
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path, clock=clock)
    decision = ResourceDecision(ResourceAction.RUN, ())

    writer.write(sample(now), decision, "ABO-00000", "download")
    assert fsync_calls == []
    clock.advance(30)
    writer.write(sample(clock()), decision, "ABO-00000", "download")
    assert len(fsync_calls) == 1
    writer.close()
    assert len(fsync_calls) == 2

    records = [__import__("json").loads(line) for line in path.read_text().splitlines()]
    assert records[0]["timestamp"] == "2026-07-16T12:30:00+00:00"
    assert records[0]["shard_id"] == "ABO-00000"
    assert records[0]["command"] == "download"
    assert records[0]["action"] == "run"


def test_guard_requires_five_uninterrupted_stable_minutes_after_pause():
    start = datetime(2026, 7, 16, tzinfo=timezone.utc)
    clock = FakeClock(start)
    telemetry = RecordingTelemetry()
    policy = SequencePolicy(
        ResourceAction.PAUSE,
        ResourceAction.RUN,
        ResourceAction.RUN,
        ResourceAction.PAUSE,
        ResourceAction.RUN,
        ResourceAction.RUN,
        ResourceAction.RUN,
    )
    guard = ResourceGuard(
        lambda: sample(clock()), policy, telemetry, clock, lambda seconds: None
    )

    assert guard.check("shard", "command").action == ResourceAction.PAUSE
    clock.advance(5)
    assert guard.check("shard", "command").action == ResourceAction.PAUSE
    clock.advance(299)
    assert guard.check("shard", "command").action == ResourceAction.PAUSE
    clock.advance(1)
    assert guard.check("shard", "command").action == ResourceAction.PAUSE
    clock.advance(5)
    assert guard.check("shard", "command").action == ResourceAction.PAUSE
    clock.advance(299)
    assert guard.check("shard", "command").action == ResourceAction.PAUSE
    clock.advance(1)
    assert guard.check("shard", "command").action == ResourceAction.RUN
    assert telemetry.records[-1][2:] == ("shard", "command")


def test_wait_for_admission_polls_every_five_seconds_and_initial_run_is_immediate():
    start = datetime(2026, 7, 16, tzinfo=timezone.utc)
    clock = FakeClock(start)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock.advance(seconds)

    guard = ResourceGuard(
        lambda: sample(clock()),
        SequencePolicy(ResourceAction.PAUSE, ResourceAction.RUN),
        RecordingTelemetry(),
        clock,
        sleep,
    )
    assert guard.wait_for_admission("shard", "download").action == ResourceAction.RUN
    assert sleeps == [5] * 61

    sleeps.clear()
    immediate = ResourceGuard(
        lambda: sample(clock()),
        SequencePolicy(ResourceAction.RUN),
        RecordingTelemetry(),
        clock,
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
        lambda: now,
        sleeps.append,
    )

    with pytest.raises(ResourceLimitExceeded, match="raw stop") as raised:
        guard.wait_for_admission("shard", "download")

    assert raised.value.reasons == ("raw stop",)
    assert sleeps == []


def test_guard_history_is_bounded_and_json_serializable():
    start = datetime(2026, 7, 16, tzinfo=timezone.utc)
    clock = FakeClock(start)
    guard = ResourceGuard(
        lambda: sample(clock()),
        SequencePolicy(ResourceAction.RUN),
        RecordingTelemetry(),
        clock,
        lambda seconds: None,
    )

    for _ in range(65):
        guard.check("shard", "command")
        clock.advance(5)

    history = guard.last_five_minutes()
    assert len(history) == 60
    assert history[0]["timestamp"] == (start + timedelta(seconds=25)).isoformat()
    __import__("json").dumps(history)
