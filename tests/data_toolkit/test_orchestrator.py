from collections import defaultdict, deque
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
import zipfile

import pandas as pd
import pytest

from data_toolkit.pipeline.commands import CommandSpec, ShardContext
from data_toolkit.pipeline.config import PathConfig
from data_toolkit.pipeline.orchestrator import (
    CheckpointError,
    EscalationCategory,
    InfrastructureError,
    IntegrationProviderRequired,
    PipelineCheckpoint,
    PipelineRunner,
    PipelineServices,
    PipelineStopped,
    ProcessGroupSafetyError,
    RollingQualityGate,
    build_services,
    plan_work_batches,
)
from data_toolkit.pipeline.packing import (
    PACK_FAMILIES,
    publish_pack,
    verify_pack,
)
from data_toolkit.pipeline.resources import (
    ResourceAction,
    ResourceDecision,
    ResourceLimitExceeded,
)
from data_toolkit.pipeline.validation import ValidationError


class FakeResourceGuard:
    def __init__(self, decisions=()):
        self.reason = None
        self.decisions = deque(decisions)

    def stop_next(self, reason):
        self.reason = reason

    def wait_for_admission(self, shard_id, command):
        if self.reason:
            raise ResourceLimitExceeded((self.reason,))

    def check(self, shard_id, command):
        if self.decisions:
            decision = self.decisions.popleft()
            if isinstance(decision, BaseException):
                raise decision
            return decision
        return ResourceDecision(ResourceAction.RUN, ())

    def last_five_minutes(self):
        return ({"cpu_percent": 95.0},)


class RecordingRunner(PipelineRunner):
    def __init__(self, config, commands=None, *, reports=None):
        validators = defaultdict(lambda: lambda: True)
        handlers = {}
        super().__init__(
            config,
            FakeResourceGuard(),
            validators,
            handlers,
            command_builder=(lambda context, config: commands)
            if commands is not None
            else None,
            report_writer=(reports.append if reports is not None else None),
        )
        self.checkpoint = PipelineCheckpoint("ABO-00000")
        self.checkpoint.was_saved = False
        self.executed = []
        self.failures = {}

    def load_checkpoint(self, path, shard_id):
        return self.checkpoint

    def save_checkpoint(self, path, checkpoint):
        checkpoint.was_saved = True

    def execute(self, command, shard_id):
        self.executed.append(command.name)
        failures = self.failures.get(command.name)
        if failures:
            raise failures.pop(0)
        self.validators[command.name] = lambda: True


@pytest.fixture
def shard_context(tmp_path):
    return ShardContext.for_test(tmp_path, "ABO", "ABO-00000")


@pytest.fixture
def isolated_config(config, tmp_path):
    return replace(
        config,
        paths=PathConfig(
            data2_root=tmp_path / "data2",
            data3_root=tmp_path / "data3",
            local_root=tmp_path / "local",
        ),
    )


def test_work_batches_fit_reserved_local_budget():
    shas = tuple(f"{index:064x}" for index in range(10))
    batches = plan_work_batches(
        shas,
        local_usable_bytes=1000,
        p95_peak_bytes=300,
        shard_size=5000,
    )
    assert tuple(sha for batch in batches for sha in batch) == shas
    assert all(len(batch) <= 2 for batch in batches)


def test_work_batches_reject_capacity_that_cannot_fit_one_asset():
    with pytest.raises(ValueError, match="cannot fit one"):
        plan_work_batches(
            ("a" * 64,),
            local_usable_bytes=100,
            p95_peak_bytes=100,
            shard_size=5000,
        )


def test_resume_skips_only_valid_outputs(isolated_config, shard_context):
    fake_runner = RecordingRunner(isolated_config)
    fake_runner.checkpoint.complete("dump_mesh")
    fake_runner.validators["dump_mesh"] = lambda: True
    fake_runner.run_shard(shard_context)
    assert "dump_mesh" not in fake_runner.executed
    assert "dump_pbr" in fake_runner.executed


def test_corrupt_complete_output_is_regenerated(isolated_config, shard_context):
    fake_runner = RecordingRunner(isolated_config)
    fake_runner.checkpoint.complete("render_cond")
    fake_runner.validators["render_cond"] = lambda: False
    fake_runner.run_shard(shard_context)
    assert "render_cond" in fake_runner.executed


def test_hard_resource_stop_checkpoints_and_escalates(
    isolated_config, shard_context
):
    reports = []
    fake_runner = RecordingRunner(isolated_config, reports=reports)
    fake_runner.resource_guard.stop_next("CPU hard duration")
    with pytest.raises(PipelineStopped) as caught:
        fake_runner.run_shard(shard_context)
    assert caught.value.exit_code == 3
    assert caught.value.report.category == EscalationCategory.RESOURCE
    assert caught.value.report.reason == "CPU hard duration"
    assert caught.value.report.shard_id == shard_context.shard_id
    assert caught.value.report.recent_telemetry
    assert caught.value.report.completed_counts == {"commands": 0, "outcomes": 0}
    assert caught.value.report.safe_resume_command
    assert caught.value.report.recovery_choices
    assert fake_runner.checkpoint.was_saved
    assert reports == [caught.value.report]


def test_recoverable_command_gets_at_most_three_total_attempts(
    isolated_config, shard_context
):
    command = CommandSpec("recoverable", ("worker",))
    runner = RecordingRunner(isolated_config, (command,))
    runner.failures[command.name] = [
        subprocess.CalledProcessError(1, command.argv),
        subprocess.CalledProcessError(1, command.argv),
    ]

    runner.run_shard(shard_context)

    assert runner.executed == [command.name] * 3
    assert runner.checkpoint.attempts == {command.name: 2}
    assert runner.checkpoint.completed_commands == [command.name]


def test_resume_does_not_reset_failed_attempt_budget(
    isolated_config, shard_context
):
    command = CommandSpec("recoverable", ("worker",))
    runner = RecordingRunner(isolated_config, (command,))
    runner.checkpoint.attempts[command.name] = 2
    runner.failures[command.name] = [subprocess.CalledProcessError(1, command.argv)]

    with pytest.raises(PipelineStopped) as caught:
        runner.resume_shard(shard_context)

    assert runner.executed == [command.name]
    assert runner.checkpoint.attempts[command.name] == 3
    assert caught.value.exit_code == 2


def test_infrastructure_error_escalates_without_retry(
    isolated_config, shard_context
):
    command = CommandSpec("checkpointed", ("worker",))
    runner = RecordingRunner(isolated_config, (command,))
    runner.failures[command.name] = [InfrastructureError("missing checkpoint")]

    with pytest.raises(PipelineStopped) as caught:
        runner.run_shard(shard_context)

    assert runner.executed == [command.name]
    assert caught.value.exit_code == 2
    assert caught.value.report.category == EscalationCategory.INFRASTRUCTURE


def test_os_infrastructure_error_escalates_without_retry(
    isolated_config, shard_context
):
    command = CommandSpec("launch", ("worker",))
    runner = RecordingRunner(isolated_config, (command,))
    runner.failures[command.name] = [OSError("exec format error")]

    with pytest.raises(PipelineStopped) as caught:
        runner.run_shard(shard_context)

    assert runner.executed == [command.name]
    assert caught.value.exit_code == 2
    assert caught.value.report.category == EscalationCategory.INFRASTRUCTURE


def test_checkpoint_roundtrip_is_atomic_and_validates_identity(
    isolated_config, tmp_path
):
    runner = PipelineRunner(isolated_config, FakeResourceGuard(), {}, {})
    path = tmp_path / "nested" / "checkpoint.json"
    checkpoint = PipelineCheckpoint("ABO-00000")
    checkpoint.complete("dump_mesh")
    checkpoint.attempts["render_cond"] = 2

    runner.save_checkpoint(path, checkpoint)

    assert runner.load_checkpoint(path, "ABO-00000") == checkpoint
    assert not list(path.parent.glob("*.tmp"))
    with pytest.raises(CheckpointError, match="shard identity"):
        runner.load_checkpoint(path, "ABO-00001")


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        json.dumps({"shard_id": "ABO-00000"}),
        json.dumps(
            {
                "schema_version": 99,
                "shard_id": "ABO-00000",
                "completed_commands": [],
                "attempts": {},
            }
        ),
        json.dumps(
            {
                "schema_version": 1,
                "shard_id": "ABO-00000",
                "completed_commands": ["dump_mesh", "dump_mesh"],
                "attempts": {},
            }
        ),
        json.dumps(
            {
                "schema_version": 1,
                "shard_id": "ABO-00000",
                "completed_commands": [],
                "attempts": {"dump_mesh": True},
            }
        ),
    ],
)
def test_corrupt_checkpoint_fails_closed(
    payload, isolated_config, tmp_path
):
    path = tmp_path / "checkpoint.json"
    path.write_text(payload)
    runner = PipelineRunner(isolated_config, FakeResourceGuard(), {}, {})

    with pytest.raises(CheckpointError):
        runner.load_checkpoint(path, "ABO-00000")


def test_checkpoint_symlink_fails_closed(isolated_config, tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shard_id": "ABO-00000",
                "completed_commands": [],
                "attempts": {},
            }
        )
    )
    path = tmp_path / "checkpoint.json"
    path.symlink_to(outside)
    runner = PipelineRunner(isolated_config, FakeResourceGuard(), {}, {})

    with pytest.raises(CheckpointError, match="regular file"):
        runner.load_checkpoint(path, "ABO-00000")


def test_unknown_internal_command_fails_closed(isolated_config):
    runner = PipelineRunner(
        isolated_config,
        FakeResourceGuard(),
        {},
        defaultdict(lambda: lambda: None),
    )

    with pytest.raises(InfrastructureError, match="unknown internal command"):
        runner.execute(CommandSpec("erase_everything", ("internal:erase",)), "shard")


def test_completed_command_with_missing_validator_escalates(
    isolated_config, shard_context
):
    command = CommandSpec("completed", ("worker",))
    reports = []
    runner = PipelineRunner(
        isolated_config,
        FakeResourceGuard(),
        {},
        {},
        command_builder=lambda context, config: (command,),
        report_writer=reports.append,
    )
    checkpoint = PipelineCheckpoint(shard_context.shard_id)
    checkpoint.complete(command.name)
    runner.save_checkpoint(
        shard_context.work_root / "checkpoint.json", checkpoint
    )

    with pytest.raises(PipelineStopped) as caught:
        runner.resume_shard(shard_context)

    assert caught.value.exit_code == 2
    assert caught.value.report.category == EscalationCategory.INFRASTRUCTURE
    assert reports == [caught.value.report]


def test_internal_dispatch_is_keyed_by_exact_command_name(isolated_config):
    calls = []
    runner = PipelineRunner(
        isolated_config,
        FakeResourceGuard(),
        {},
        {"stage_raw": lambda: calls.append("stage_raw")},
    )

    runner.execute(CommandSpec("stage_raw", ("internal:stage_raw",)), "shard")

    assert calls == ["stage_raw"]


def test_known_internal_name_cannot_dispatch_external_argv(isolated_config):
    runner = PipelineRunner(
        isolated_config,
        FakeResourceGuard(),
        {},
        {"stage_raw": lambda: None},
    )

    with pytest.raises(InfrastructureError, match="unknown internal command"):
        runner.execute(CommandSpec("stage_raw", ("external-worker",)), "shard")


@pytest.mark.parametrize(
    ("end_to_end_failures", "schema_failures", "should_stop"),
    [
        (51, 0, True),
        (50, 0, False),
        (49, 0, False),
        (0, 26, True),
        (0, 25, False),
        (0, 24, False),
    ],
)
def test_rolling_quality_gate_boundaries(
    end_to_end_failures, schema_failures, should_stop
):
    gate = RollingQualityGate()
    for index in range(500):
        gate.record(
            succeeded=index >= end_to_end_failures,
            schema_failure=index < schema_failures,
        )
    assert (gate.violation_reason() is not None) is should_stop


def test_quality_stop_report_has_complete_operator_context(
    isolated_config, shard_context
):
    command = CommandSpec("quality_checked", ("worker",))
    reports = []
    runner = RecordingRunner(isolated_config, (command,), reports=reports)
    for index in range(500):
        runner.quality_gate.record(succeeded=index >= 51, schema_failure=False)

    with pytest.raises(PipelineStopped) as caught:
        runner.run_shard(shard_context)

    report = caught.value.report
    assert caught.value.exit_code == 4
    assert report.source == shard_context.source
    assert report.shard_id == shard_context.shard_id
    assert report.command == command.name
    assert report.category == EscalationCategory.DATA_QUALITY
    assert "end-to-end" in report.reason
    assert report.recent_telemetry
    assert report.completed_counts == {"commands": 0, "outcomes": 500}
    assert "resume" in report.safe_resume_command
    assert len(report.recovery_choices) >= 2


class FakeProcess:
    def __init__(self, pid, polls):
        self.pid = pid
        self._polls = deque(polls)
        self.returncode = None
        self.waited = 0

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        if self._polls:
            value = self._polls.popleft()
            if value is not None:
                self.returncode = value
        return self.returncode

    def wait(self):
        self.waited += 1
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def process_runner(config, guard, processes, *, group_ids=None):
    created = []
    signals = []
    clock = FakeClock()

    def process_factory(argv, **kwargs):
        process = processes[len(created)]
        created.append((tuple(argv), kwargs, process))
        return process

    by_pid = {process.pid: process for process in processes}

    def killpg(group, sent_signal):
        signals.append((group, sent_signal))
        if sent_signal == signal.SIGKILL:
            by_pid[group].returncode = -sent_signal

    runner = PipelineRunner(
        config,
        guard,
        {},
        {},
        process_factory=process_factory,
        monotonic_clock=clock,
        sleeper=clock.sleep,
        killpg=killpg,
        getpgid=lambda pid: (group_ids or {}).get(pid, pid),
        termination_grace_seconds=2,
        monitor_interval_seconds=1,
    )
    return runner, created, signals


def test_external_ranks_are_reaped_after_normal_completion(isolated_config):
    processes = [FakeProcess(101, [None, 0]), FakeProcess(102, [None, 0])]
    runner, created, signals = process_runner(
        isolated_config, FakeResourceGuard(), processes
    )
    command = CommandSpec("encode", ("worker",), gpu_ranks=2)

    runner.execute(command, "ABO-00000")

    assert len(created) == 2
    assert all(item[1]["start_new_session"] for item in created)
    assert [item[0][-4:] for item in created] == [
        ("--rank", "0", "--world_size", "2"),
        ("--rank", "1", "--world_size", "2"),
    ]
    assert all(process.waited == 1 for process in processes)
    assert signals == []


def test_failed_rank_terminates_and_reaps_its_siblings(isolated_config):
    processes = [FakeProcess(201, [7]), FakeProcess(202, [None, None, None])]
    runner, _, signals = process_runner(
        isolated_config, FakeResourceGuard(), processes
    )

    with pytest.raises(subprocess.CalledProcessError):
        runner.execute(CommandSpec("encode", ("worker",), gpu_ranks=2), "shard")

    assert (202, signal.SIGTERM) in signals
    assert (202, signal.SIGKILL) in signals
    assert all(process.waited == 1 for process in processes)


def test_paused_groups_resume_before_term_and_all_are_reaped(isolated_config):
    pause = ResourceDecision(ResourceAction.PAUSE, ("CPU soft",))
    stop = ResourceDecision(ResourceAction.STOP, ("CPU hard",))
    processes = [FakeProcess(301, [None] * 10), FakeProcess(302, [None] * 10)]
    runner, _, signals = process_runner(
        isolated_config, FakeResourceGuard((pause, stop)), processes
    )

    with pytest.raises(ResourceLimitExceeded):
        runner.execute(CommandSpec("encode", ("worker",), gpu_ranks=2), "shard")

    for pid in (301, 302):
        assert signals.index((pid, signal.SIGSTOP)) < signals.index(
            (pid, signal.SIGCONT)
        )
        assert signals.index((pid, signal.SIGCONT)) < signals.index(
            (pid, signal.SIGTERM)
        )
        assert (pid, signal.SIGKILL) in signals
    assert all(process.waited == 1 for process in processes)


def test_monitor_exception_still_terminates_and_reaps_all_ranks(isolated_config):
    processes = [FakeProcess(401, [None] * 10), FakeProcess(402, [None] * 10)]
    runner, _, signals = process_runner(
        isolated_config,
        FakeResourceGuard((RuntimeError("sampler failed"),)),
        processes,
    )

    with pytest.raises(RuntimeError, match="sampler failed"):
        runner.execute(CommandSpec("encode", ("worker",), gpu_ranks=2), "shard")

    assert all((process.pid, signal.SIGTERM) in signals for process in processes)
    assert all(process.waited == 1 for process in processes)


def test_process_group_identity_mismatch_never_signals_unrelated_group(
    isolated_config,
):
    process = FakeProcess(501, [None] * 10)
    runner, _, signals = process_runner(
        isolated_config,
        FakeResourceGuard((RuntimeError("sampler failed"),)),
        [process],
        group_ids={501: 999},
    )

    with pytest.raises(ProcessGroupSafetyError, match="process group"):
        runner.execute(CommandSpec("worker", ("worker",)), "shard")

    assert signals == []
    assert process.waited == 1


def test_negative_rank_count_fails_instead_of_becoming_noop(isolated_config):
    runner, created, signals = process_runner(
        isolated_config, FakeResourceGuard(), []
    )

    with pytest.raises(InfrastructureError, match="rank count"):
        runner.execute(
            CommandSpec("malformed", ("worker",), gpu_ranks=-1), "shard"
        )

    assert created == []
    assert signals == []


def test_build_services_is_side_effect_free(isolated_config):
    services = build_services(isolated_config)

    assert services.registry.path == isolated_config.paths.data2_root / "control/assets.parquet"
    assert not isolated_config.paths.data2_root.exists()
    assert not isolated_config.paths.data3_root.exists()
    assert not isolated_config.paths.local_root.exists()


class FakeRegistry:
    def __init__(self, frame, path):
        self.frame = frame
        self.path = path

    def load(self):
        return self.frame.copy()


class FakePilotReader:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def p95_peak_local_bytes(self, source):
        self.calls.append(source)
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value


class FakeReferenceCounter:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def pending_references(
        self,
        source,
        raw_relative_path,
        *,
        excluding_shard_id,
        excluding_batch_id,
    ):
        self.calls.append(
            (
                source,
                raw_relative_path,
                excluding_shard_id,
                excluding_batch_id,
            )
        )
        return self.value


class FakeShardRunner:
    def __init__(self):
        self.runs = []
        self.resumes = []

    def run_shard(self, context):
        self.runs.append(context)

    def resume_shard(self, context):
        self.resumes.append(context)


def test_plan_uses_exact_local_reserve_and_freezes_immutable_batches(
    isolated_config,
):
    gib = 1024**3
    shas = tuple(f"{index:064x}" for index in range(10))
    registry = FakeRegistry(
        pd.DataFrame(
            {
                "sha256": shas,
                "owner_source": ["ABO"] * len(shas),
                "shard_id": ["ABO-00000"] * len(shas),
            }
        ),
        isolated_config.paths.data2_root / "control/assets.parquet",
    )
    pilot = FakePilotReader(100 * gib)
    disk_calls = []

    def disk_usage(path):
        disk_calls.append(path)
        return SimpleNamespace(total=1000 * gib, free=900 * gib)

    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        registry_store=registry,
        pilot_reader=pilot,
        disk_usage=disk_usage,
    )

    lines = services.plan(
        "production", "ABO", "ABO-00000", None, freeze=True
    )

    batch_root = (
        isolated_config.paths.data2_root
        / "control/shards/ABO/ABO-00000"
    )
    assert [len((batch_root / f"batch{index:03d}.txt").read_text().splitlines()) for index in range(3)] == [4, 4, 2]
    assert (batch_root / "batches.json").is_file()
    assert len(lines) == 3
    assert pilot.calls == ["ABO"]
    assert disk_calls == [isolated_config.paths.local_root]

    before = {
        path: path.read_bytes() for path in batch_root.iterdir() if path.is_file()
    }
    pilot.value = AssertionError("frozen plans must not reread pilot")
    services.disk_usage = lambda path: (_ for _ in ()).throw(
        AssertionError("frozen plans must not reread free space")
    )
    services.plan("production", "ABO", "ABO-00000", None, freeze=True)
    assert {
        path: path.read_bytes() for path in batch_root.iterdir() if path.is_file()
    } == before


def test_read_only_plan_does_not_create_configured_roots(isolated_config):
    gib = 1024**3
    sha = "a" * 64
    registry = FakeRegistry(
        pd.DataFrame(
            {
                "sha256": [sha],
                "owner_source": ["ABO"],
                "shard_id": ["ABO-00000"],
            }
        ),
        isolated_config.paths.data2_root / "control/assets.parquet",
    )
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        registry_store=registry,
        pilot_reader=FakePilotReader(100),
        disk_usage=lambda path: SimpleNamespace(
            total=1000 * gib, free=1000 * gib
        ),
    )

    services.plan("production", "ABO", "ABO-00000", None)

    assert not isolated_config.paths.data2_root.exists()
    assert not isolated_config.paths.data3_root.exists()
    assert not isolated_config.paths.local_root.exists()


@pytest.mark.parametrize("pilot_value", [0, -1, True, 1.5])
def test_plan_rejects_unvalidated_pilot_p95(isolated_config, pilot_value):
    sha = "a" * 64
    registry = FakeRegistry(
        pd.DataFrame(
            {
                "sha256": [sha],
                "owner_source": ["ABO"],
                "shard_id": ["ABO-00000"],
            }
        ),
        isolated_config.paths.data2_root / "control/assets.parquet",
    )
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        registry_store=registry,
        pilot_reader=FakePilotReader(pilot_value),
        disk_usage=lambda path: SimpleNamespace(total=1000, free=1000),
    )

    with pytest.raises(IntegrationProviderRequired, match="positive integer"):
        services.plan("production", "ABO", "ABO-00000", None)


def test_resume_reuses_frozen_batches_without_replanning(isolated_config):
    gib = 1024**3
    shas = tuple(f"{index:064x}" for index in range(3))
    registry = FakeRegistry(
        pd.DataFrame(
            {
                "sha256": shas,
                "owner_source": ["ABO"] * 3,
                "shard_id": ["ABO-00000"] * 3,
            }
        ),
        isolated_config.paths.data2_root / "control/assets.parquet",
    )
    runner = FakeShardRunner()
    audits = []
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        registry_store=registry,
        pilot_reader=FakePilotReader(100),
        disk_usage=lambda path: SimpleNamespace(
            total=1000 * gib, free=1000 * gib
        ),
        runner=runner,
        batch_auditor=audits.append,
        published_batch_verifier=lambda context: None,
    )
    services.plan("production", "ABO", "ABO-00000", None, freeze=True)
    services.pilot_reader = FakePilotReader(
        AssertionError("resume must not replan")
    )
    services.disk_usage = lambda path: (_ for _ in ()).throw(
        AssertionError("resume must not inspect free space")
    )

    services.resume("ABO", "ABO-00000")

    assert [item.batch_id for item in runner.resumes] == ["batch000"]
    assert audits == runner.resumes


def test_resume_accepts_frozen_gate_subset_without_expanding_scope(
    isolated_config,
):
    gib = 1024**3
    shas = tuple(f"{index:064x}" for index in range(5))
    registry = FakeRegistry(
        pd.DataFrame(
            {
                "sha256": shas,
                "owner_source": ["ABO"] * len(shas),
                "shard_id": ["ABO-00000"] * len(shas),
            }
        ),
        isolated_config.paths.data2_root / "control/assets.parquet",
    )
    runner = FakeShardRunner()
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        registry_store=registry,
        pilot_reader=FakePilotReader(100),
        disk_usage=lambda path: SimpleNamespace(
            total=1000 * gib, free=1000 * gib
        ),
        runner=runner,
        batch_auditor=lambda context: None,
        published_batch_verifier=lambda context: None,
    )
    services.plan(
        "smoke", "ABO", "ABO-00000", count=2, freeze=True
    )

    services.resume("ABO", "ABO-00000")

    assert [context.instances.read_text().splitlines() for context in runner.resumes] == [
        list(shas[:2])
    ]


def test_corrupt_frozen_batch_manifest_fails_closed(isolated_config):
    gib = 1024**3
    sha = "a" * 64
    registry = FakeRegistry(
        pd.DataFrame(
            {
                "sha256": [sha],
                "owner_source": ["ABO"],
                "shard_id": ["ABO-00000"],
            }
        ),
        isolated_config.paths.data2_root / "control/assets.parquet",
    )
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        registry_store=registry,
        pilot_reader=FakePilotReader(100),
        disk_usage=lambda path: SimpleNamespace(
            total=1000 * gib, free=1000 * gib
        ),
    )
    services.plan("production", "ABO", "ABO-00000", None, freeze=True)
    batch = (
        isolated_config.paths.data2_root
        / "control/shards/ABO/ABO-00000/batch000.txt"
    )
    batch.write_text("b" * 64 + "\n")

    with pytest.raises(InfrastructureError, match="frozen batch"):
        services.resume("ABO", "ABO-00000")


def write_raw_metadata(context, records):
    metadata = context.source_root / "raw/metadata.csv"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(metadata, index=False)


def write_instances(context, shas):
    context.instances.parent.mkdir(parents=True, exist_ok=True)
    context.instances.write_text("".join(f"{sha}\n" for sha in shas))


def test_stage_raw_preserves_verified_adapter_relative_layout(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(tmp_path / "batch", "ABO", "ABO-00000")
    contents = b"verified raw"
    asset_sha = sha256(contents).hexdigest()
    relative = "raw/models/item.glb"
    source = context.source_root / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(contents)
    write_instances(context, (asset_sha,))
    write_raw_metadata(
        context, ({"sha256": asset_sha, "local_path": relative},)
    )
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )

    services.stage_raw(context)

    assert (context.download_root / relative).read_bytes() == contents
    staged = pd.read_csv(context.download_root / "raw/metadata.csv")
    assert staged.to_dict("records") == [
        {"sha256": asset_sha, "local_path": relative}
    ]


@pytest.mark.parametrize(
    "relative", ["../escape.glb", "/absolute.glb", "raw\\bad.glb"]
)
def test_stage_raw_rejects_escape_paths(isolated_config, tmp_path, relative):
    context = ShardContext.for_test(tmp_path / "batch", "ABO", "ABO-00000")
    asset_sha = "a" * 64
    write_instances(context, (asset_sha,))
    write_raw_metadata(
        context, ({"sha256": asset_sha, "local_path": relative},)
    )
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )

    with pytest.raises(ValidationError, match="unsafe raw path"):
        services.stage_raw(context)


def test_stage_raw_rejects_symlinked_source(isolated_config, tmp_path):
    context = ShardContext.for_test(tmp_path / "batch", "ABO", "ABO-00000")
    contents = b"outside"
    asset_sha = sha256(contents).hexdigest()
    outside = tmp_path / "outside.glb"
    outside.write_bytes(contents)
    relative = "raw/models/item.glb"
    source = context.source_root / relative
    source.parent.mkdir(parents=True)
    source.symlink_to(outside)
    write_instances(context, (asset_sha,))
    write_raw_metadata(
        context, ({"sha256": asset_sha, "local_path": relative},)
    )
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )

    with pytest.raises(ValidationError, match="symlink"):
        services.stage_raw(context)


def test_stage_raw_extracts_only_selected_shared_zip_member(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "batch", "ObjaverseXL_github", "ObjaverseXL_github-00000"
    )
    contents = b"selected glb"
    asset_sha = sha256(contents).hexdigest()
    archive_relative = "raw/github/repos/repo.zip"
    selected_relative = f"{archive_relative}/models/selected.glb"
    archive = context.source_root / archive_relative
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("models/selected.glb", contents)
        bundle.writestr("models/unselected.glb", b"not selected")
    write_instances(context, (asset_sha,))
    write_raw_metadata(
        context,
        ({"sha256": asset_sha, "local_path": selected_relative},),
    )
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )

    services.stage_raw(context)

    assert (context.download_root / selected_relative).read_bytes() == contents
    assert not (
        context.download_root
        / f"{archive_relative}/models/unselected.glb"
    ).exists()


def test_stage_raw_rejects_duplicate_zip_members(isolated_config, tmp_path):
    context = ShardContext.for_test(
        tmp_path / "batch", "ObjaverseXL_github", "ObjaverseXL_github-00000"
    )
    contents = b"selected glb"
    asset_sha = sha256(contents).hexdigest()
    archive_relative = "raw/github/repos/repo.zip"
    selected_relative = f"{archive_relative}/models/selected.glb"
    archive = context.source_root / archive_relative
    archive.parent.mkdir(parents=True)
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("models/selected.glb", contents)
            bundle.writestr("models/selected.glb", contents)
    write_instances(context, (asset_sha,))
    write_raw_metadata(
        context,
        ({"sha256": asset_sha, "local_path": selected_relative},),
    )
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )

    with pytest.raises(ValidationError, match="duplicate ZIP member"):
        services.stage_raw(context)


def test_build_packs_publishes_exactly_eight_family_layouts(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(tmp_path / "batch", "ABO", "ABO-00000")
    asset_sha = "a" * 64
    write_instances(context, (asset_sha,))
    for relative in (
        f"renders_cond/{asset_sha}/000.png",
        f"ss_latents/ss_enc_conv3d_16l8_fp16_64_view/{asset_sha}/view00.npz",
        f"shape_latents/shape_enc_next_dc_f16c32_fp16_256_view/{asset_sha}/view00.npz",
        f"shape_latents/shape_enc_next_dc_f16c32_fp16_512_view/{asset_sha}/view00.npz",
        f"shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view/{asset_sha}/view00.npz",
        f"pbr_latents/tex_enc_next_dc_f16c32_fp16_256_view_fix/{asset_sha}/view00.npz",
        f"pbr_latents/tex_enc_next_dc_f16c32_fp16_512_view_fix/{asset_sha}/view00.npz",
        f"pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix/{asset_sha}/view00.npz",
    ):
        path = context.output_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"output")
    published = []

    def pack_publisher(data2_root, source_root, members, shard_id, **kwargs):
        published.append((data2_root, source_root, members, shard_id, kwargs))
        return tuple(SimpleNamespace(validated_at="now") for _ in PACK_FAMILIES)

    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        output_validator=lambda context: None,
        pack_publisher=pack_publisher,
        pack_member_builder=lambda context: {
            family: [Path(f"renders_cond/{asset_sha}/000.png")]
            for family in PACK_FAMILIES
        },
        published_batch_verifier=lambda context: None,
        tool_commit="test-commit",
    )

    services.build_packs(context)

    assert len(published) == 1
    assert tuple(published[0][2]) == PACK_FAMILIES


def test_build_packs_rejects_inexact_family_mapping(isolated_config, tmp_path):
    context = ShardContext.for_test(tmp_path / "batch", "ABO", "ABO-00000")
    write_instances(context, ("a" * 64,))
    calls = []
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        output_validator=lambda context: None,
        pack_member_builder=lambda context: {"common": []},
        pack_publisher=lambda *args, **kwargs: calls.append(args),
        published_batch_verifier=lambda context: None,
    )

    with pytest.raises(ValidationError, match="exactly eight"):
        services.build_packs(context)

    assert calls == []


def publish_dummy_batch(config, context, asset_sha):
    members = {}
    for family in PACK_FAMILIES:
        relative = Path(family) / "payload.bin"
        path = context.output_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(family.encode())
        members[family] = [relative]
    return publish_pack(
        config.paths.data2_root,
        context.output_root,
        members,
        context.shard_id,
        source=context.source,
        batch_id=context.batch_id,
        config_hash=config.config_hash(),
        tool_commit="test-commit",
        asset_sha256s=(asset_sha,),
        completed_count=1,
        quarantined_count=0,
    )


def test_published_index_must_point_to_canonical_family_path(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(tmp_path / "batch", "ABO", "ABO-00000")
    asset_sha = "a" * 64
    write_instances(context, (asset_sha,))
    publish_dummy_batch(isolated_config, context, asset_sha)
    prepared = isolated_config.paths.data2_root / "prepared"
    index_path = prepared / "index/ABO/ABO-00000.json"
    index = json.loads(index_path.read_text())
    entry = index["batches"]["batch000"]["common"]
    canonical_pack = prepared / entry["pack"]
    canonical_manifest = prepared / entry["manifest"]
    alternate_pack = prepared / "alternate/common.tar"
    alternate_manifest = alternate_pack.with_suffix(".tar.manifest.json")
    alternate_pack.parent.mkdir(parents=True)
    alternate_pack.write_bytes(canonical_pack.read_bytes())
    alternate_manifest.write_bytes(canonical_manifest.read_bytes())
    entry["pack"] = alternate_pack.relative_to(prepared).as_posix()
    entry["manifest"] = alternate_manifest.relative_to(prepared).as_posix()
    index_path.write_text(json.dumps(index))
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )

    with pytest.raises(ValidationError, match="canonical pack path"):
        services._verify_published_batch(context)


def test_logical_shard_index_rejects_unfrozen_extra_batch(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(tmp_path / "batch", "ABO", "ABO-00000")
    asset_sha = "a" * 64
    write_instances(context, (asset_sha,))
    publish_dummy_batch(isolated_config, context, asset_sha)
    index_path = (
        isolated_config.paths.data2_root
        / "prepared/index/ABO/ABO-00000.json"
    )
    index = json.loads(index_path.read_text())
    index["batches"]["batch999"] = index["batches"]["batch000"]
    index_path.write_text(json.dumps(index))
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )

    with pytest.raises(ValidationError, match="batch set"):
        services._verify_logical_index(
            "ABO", "ABO-00000", ((asset_sha,),)
        )


def configured_context(config, source="ABO"):
    return ShardContext.from_config(config, source, f"{source}-00000", "batch000")


def test_archive_verifies_before_zero_reference_deletion(isolated_config):
    context = configured_context(isolated_config)
    contents = b"raw archive payload"
    asset_sha = sha256(contents).hexdigest()
    relative = "raw/models/item.glb"
    source = context.source_root / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(contents)
    write_instances(context, (asset_sha,))
    write_raw_metadata(
        context, ({"sha256": asset_sha, "local_path": relative},)
    )
    counter = FakeReferenceCounter(1)
    audit_calls = []
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        reference_counter=counter,
        published_batch_verifier=audit_calls.append,
        tool_commit="test-commit",
    )
    services.stage_raw(context)

    services.archive_raw(context)

    archive = (
        isolated_config.paths.data3_root
        / "archive/raw/ABO/ABO-00000/batch000.tar"
    )
    verify_pack(archive, archive.with_suffix(".tar.manifest.json"))
    manifest = json.loads(archive.with_suffix(".tar.manifest.json").read_text())
    assert len(manifest["members"]) == 1
    assert sum(item["size"] for item in manifest["members"]) == len(contents)
    assert manifest["validated_at"]
    assert source.exists()
    assert audit_calls == [context]

    counter.value = 0
    services.archive_raw(context)
    assert not source.exists()
    assert counter.calls[-1][1] == relative

    services.archive_raw(context)
    assert not source.exists()


def test_archive_fails_closed_without_reference_provider(isolated_config):
    context = configured_context(isolated_config)
    contents = b"raw archive payload"
    asset_sha = sha256(contents).hexdigest()
    relative = "raw/models/item.glb"
    source = context.source_root / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(contents)
    write_instances(context, (asset_sha,))
    write_raw_metadata(
        context, ({"sha256": asset_sha, "local_path": relative},)
    )
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        published_batch_verifier=lambda context: None,
        tool_commit="test-commit",
    )
    services.stage_raw(context)

    with pytest.raises(IntegrationProviderRequired, match="reference counter"):
        services.archive_raw(context)

    assert source.exists()


def test_archive_rejects_symlinked_final_output(isolated_config):
    context = configured_context(isolated_config)
    contents = b"raw archive payload"
    asset_sha = sha256(contents).hexdigest()
    relative = "raw/models/item.glb"
    source = context.source_root / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(contents)
    write_instances(context, (asset_sha,))
    write_raw_metadata(
        context, ({"sha256": asset_sha, "local_path": relative},)
    )
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        reference_counter=FakeReferenceCounter(1),
        published_batch_verifier=lambda context: None,
        tool_commit="test-commit",
    )
    services.stage_raw(context)
    services.archive_raw(context)
    archive, _ = services._raw_archive_paths(context)
    target = archive.with_name("target.tar")
    archive.rename(target)
    archive.symlink_to(target.name)

    with pytest.raises(ValidationError, match="symlink"):
        services.archive_raw(context)


def test_resolution_cleanup_requires_valid_encoded_outputs(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(tmp_path / "batch", "ABO", "ABO-00000")
    dual = context.work_root / "dual_grid_view_256"
    pbr = context.work_root / "pbr_voxels_view_fix_256"
    dual.mkdir(parents=True)
    pbr.mkdir(parents=True)
    valid = False

    def validate_resolution(context, resolution):
        if not valid:
            raise ValidationError("encoded output corrupt")

    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        resolution_validator=validate_resolution,
    )

    with pytest.raises(ValidationError, match="encoded output corrupt"):
        services.cleanup_voxels(context, 256)
    assert dual.exists() and pbr.exists()

    valid = True
    services.cleanup_voxels(context, 256)
    assert not dual.exists() and not pbr.exists()


def test_local_cleanup_requires_pack_and_archive_audits(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(tmp_path / "batch", "ABO", "ABO-00000")
    for root in (context.download_root, context.work_root, context.output_root):
        root.mkdir(parents=True)
        (root / "payload").write_text("x")
    calls = []

    def batch_audit(context):
        calls.append("packs")

    def archive_audit(context):
        calls.append("raw")

    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        published_batch_verifier=batch_audit,
        raw_archive_verifier=archive_audit,
    )

    services.cleanup_local(context)

    assert calls == ["packs", "raw"]
    assert not context.download_root.exists()
    assert not context.work_root.exists()
    assert not context.output_root.exists()
