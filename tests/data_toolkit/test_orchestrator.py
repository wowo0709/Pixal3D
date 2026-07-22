from collections import defaultdict, deque
from dataclasses import asdict, replace
import ast
import errno
from hashlib import sha256
import json
import io
import os
import pickle
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import zipfile

import pandas as pd
import numpy as np
import pytest

import data_toolkit.pipeline.orchestrator as orchestrator_module
from data_toolkit.pipeline.commands import CommandSpec, ShardContext
from data_toolkit.pipeline.config import PathConfig
from data_toolkit.pipeline.orchestrator import (
    CheckpointError,
    EscalationCategory,
    InfrastructureError,
    IntegrationProviderRequired,
    OutputValidationError,
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
    ResourceAccountingError,
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

    def load_checkpoint(self, path, shard_id, gate="production"):
        self.checkpoint.gate = gate
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


def test_production_batches_above_64_use_parallel_chunk_scheduler(
    isolated_config, shard_context
):
    class Runner:
        def __init__(self):
            self.calls = []
            self.command_builder = "original"

        def run_shard(self, context):
            self.calls.append(("run", context.batch_id))

        def resume_shard(self, context):
            self.calls.append(("resume", context.batch_id))

    class Scheduler:
        def __init__(self):
            self.calls = []

        def run_batch(self, context, assets):
            self.calls.append((context.batch_id, tuple(assets)))

    runner = Runner()
    scheduler = Scheduler()
    services = PipelineServices(
        isolated_config,
        runner=runner,
        parallel_scheduler_factory=lambda _context: scheduler,
    )
    context = replace(shard_context, gate="production")
    assets = tuple(f"{index:064x}" for index in range(65))

    services._execute_batch(context, assets, resume=False)

    assert scheduler.calls == [("batch000", assets)]
    assert runner.calls == [("run", "batch000")]
    assert runner.command_builder == "original"


def test_parallel_raw_metadata_omits_quarantined_download_assets(
    isolated_config, shard_context
):
    completed = "a" * 64
    quarantined = "b" * 64
    parent = replace(shard_context, gate="production")
    parent.instances.parent.mkdir(parents=True, exist_ok=True)
    parent.instances.write_text(f"{completed}\n{quarantined}\n")

    runner = RecordingRunner(isolated_config)
    runner.checkpoint.quality_outcomes = {
        completed: "completed",
        quarantined: "failure",
    }
    services = PipelineServices(isolated_config, runner=runner)
    child_instances = parent.work_root / "chunks/chunk000/instances.txt"
    child_instances.parent.mkdir(parents=True, exist_ok=True)
    child_instances.write_text(f"{completed}\n{quarantined}\n")
    child = replace(
        parent,
        instances=child_instances,
        download_root=parent.work_root / "chunks/chunk000/source",
    )
    child_metadata = child.download_root / "raw/metadata.csv"
    child_metadata.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "sha256": completed,
                "local_path": f"raw/{completed}.glb",
                "content_sha256": completed,
                "companion_files": "{}",
            }
        ]
    ).to_csv(child_metadata, index=False)
    chunk = SimpleNamespace(
        assets=lambda: (completed, quarantined),
    )
    executor = SimpleNamespace(
        service_context=lambda _chunk: (services, child),
    )

    services._write_parallel_raw_metadata(parent, (chunk,), executor)

    records = services._read_raw_records(
        parent.download_root / "raw/metadata.csv", (completed,)
    )
    assert tuple(record["sha256"] for record in records) == (completed,)


def test_parallel_raw_metadata_allows_quarantined_chunk_without_metadata(
    isolated_config, shard_context
):
    quarantined = "b" * 64
    parent = replace(shard_context, gate="production")
    parent.instances.parent.mkdir(parents=True, exist_ok=True)
    parent.instances.write_text(f"{quarantined}\n")
    runner = RecordingRunner(isolated_config)
    runner.checkpoint.quality_outcomes = {quarantined: "failure"}
    services = PipelineServices(isolated_config, runner=runner)
    child = replace(
        parent,
        download_root=parent.work_root / "chunks/chunk000/source",
    )
    chunk = SimpleNamespace(assets=lambda: (quarantined,))
    executor = SimpleNamespace(
        service_context=lambda _chunk: (services, child),
    )

    services._write_parallel_raw_metadata(parent, (chunk,), executor)

    assert services._read_raw_records(
        parent.download_root / "raw/metadata.csv", ()
    ) == ()


def test_build_packs_creates_empty_output_root_for_quarantined_batch(
    isolated_config, shard_context
):
    quarantined = "b" * 64
    context = replace(shard_context, gate="production")
    context.instances.parent.mkdir(parents=True, exist_ok=True)
    context.instances.write_text(f"{quarantined}\n")
    runner = RecordingRunner(isolated_config)
    runner.checkpoint.quality_outcomes = {quarantined: "failure"}
    runner.active_context = context
    runner.active_checkpoint = runner.checkpoint
    runner._active_quality_ledger = {"family_exclusions": {}}

    def publish(_data2, source_root, members, *_args, **_kwargs):
        assert source_root.is_dir()
        assert all(not family_members for family_members in members.values())
        return tuple(
            SimpleNamespace(validated_at="now") for _ in PACK_FAMILIES
        )

    services = PipelineServices(
        isolated_config,
        runner=runner,
        output_validator=lambda _context: None,
        pack_publisher=publish,
        published_batch_verifier=lambda _context: None,
        tool_commit="test-commit",
    )

    services.build_packs(context)

    assert context.output_root.is_dir()


@pytest.mark.parametrize("gate", ["smoke", "pilot"])
def test_qualification_batches_keep_sequential_reference_path(
    isolated_config, shard_context, gate
):
    class Runner:
        def __init__(self):
            self.calls = []

        def run_shard(self, context):
            self.calls.append(context.gate)

    runner = Runner()
    services = PipelineServices(
        isolated_config,
        runner=runner,
        parallel_scheduler_factory=lambda _context: pytest.fail(
            "qualification must not construct a parallel scheduler"
        ),
    )
    context = replace(shard_context, gate=gate)

    services._execute_batch(
        context, tuple(f"{index:064x}" for index in range(65)), resume=False
    )

    assert runner.calls == [gate]


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


def test_work_batches_respect_explicit_gate_cap():
    shas = tuple(f"{index:064x}" for index in range(200))
    batches = plan_work_batches(
        shas,
        local_usable_bytes=100_000_000_000,
        p95_peak_bytes=115 * 1024**2,
        shard_size=5000,
        max_batch_assets=64,
    )
    assert len(batches) == 4
    assert [len(batch) for batch in batches] == [64, 64, 64, 8]


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


def test_completed_output_revalidation_preserves_pipeline_stop(
    isolated_config, shard_context
):
    runner = RecordingRunner(isolated_config)
    runner.checkpoint.complete("dump_mesh")
    stopped = PipelineStopped(SimpleNamespace(reason="quality stop"), 4)
    runner.validators["dump_mesh"] = lambda: (_ for _ in ()).throw(stopped)

    with pytest.raises(PipelineStopped) as caught:
        runner.run_shard(shard_context)

    assert caught.value is stopped
    assert caught.value.exit_code == 4


@pytest.mark.parametrize("error_type", [AssertionError, AttributeError, MemoryError])
def test_quality_restore_does_not_hide_programmer_defects(
    error_type, isolated_config, shard_context
):
    runner = RecordingRunner(isolated_config, commands=())
    defect = error_type("quality restore programmer defect")
    runner._restore_quality_state = lambda context, checkpoint: (
        _ for _ in ()
    ).throw(defect)

    with pytest.raises(error_type) as caught:
        runner.run_shard(shard_context)

    assert caught.value is defect


@pytest.mark.parametrize("error_type", [AssertionError, AttributeError, MemoryError])
def test_completed_output_validation_does_not_hide_programmer_defects(
    error_type, isolated_config, shard_context
):
    command = CommandSpec("completed", ("worker",))
    runner = RecordingRunner(isolated_config, (command,))
    runner.checkpoint.complete(command.name)
    defect = error_type("completed validator programmer defect")
    runner.validators[command.name] = lambda: (_ for _ in ()).throw(defect)

    with pytest.raises(error_type) as caught:
        runner.run_shard(shard_context)

    assert caught.value is defect


@pytest.mark.parametrize("error_type", [AssertionError, AttributeError, MemoryError])
def test_command_execution_does_not_hide_programmer_defects(
    error_type, isolated_config, shard_context
):
    command = CommandSpec("defective", ("worker",))
    runner = RecordingRunner(isolated_config, (command,))
    defect = error_type("command programmer defect")
    runner.failures[command.name] = [defect]

    with pytest.raises(error_type) as caught:
        runner.run_shard(shard_context)

    assert caught.value is defect


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
    assert caught.value.report.completed_counts == {
        "commands": 0,
        "outcomes": 0,
        "completed_assets": 0,
        "quarantined_assets": 0,
        "failure_assets": 0,
        "schema_failure_assets": 0,
    }
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
    assert runner.checkpoint.attempts == {command.name: 3}
    assert runner.checkpoint.completed_commands == [command.name]


def test_runner_records_per_command_wall_seconds(
    isolated_config, shard_context
):
    command = CommandSpec("timed", ("worker",))
    runner = RecordingRunner(isolated_config, (command,))
    ticks = iter((10.0, 12.5))
    runner.monotonic_clock = lambda: next(ticks)

    runner.run_shard(shard_context)

    assert runner.last_command_timings == {"timed": pytest.approx(2.5)}


def test_failed_render_retry_steps_down_workers_at_retry_boundary(
    isolated_config, shard_context
):
    command = CommandSpec(
        "render_cond",
        ("worker",),
        gpu_ranks=7,
        workers_per_gpu=4,
    )
    runner = RecordingRunner(isolated_config, (command,))
    launched_workers = []

    def execute(candidate, shard_id):
        launched_workers.append(candidate.workers_per_gpu)
        if len(launched_workers) == 1:
            raise subprocess.CalledProcessError(1, candidate.argv)
        runner.validators[candidate.name] = lambda: True

    runner.execute = execute

    runner.run_shard(shard_context)

    assert launched_workers == [4, 3]


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


def test_exhausted_download_checkpoint_validates_without_new_launch(
    isolated_config, shard_context
):
    command = CommandSpec("download", ("worker",))
    runner = RecordingRunner(isolated_config, (command,))
    runner.checkpoint.attempts[command.name] = 3
    runner.validators[command.name] = lambda: True

    runner.resume_shard(shard_context)

    assert runner.executed == []
    assert runner.checkpoint.completed_commands == [command.name]
    assert runner.checkpoint.attempts[command.name] == 3


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


@pytest.mark.parametrize("error_type", [AssertionError, AttributeError, MemoryError])
def test_checkpoint_parser_does_not_hide_programmer_defects(
    error_type, isolated_config, monkeypatch, tmp_path
):
    path = tmp_path / "checkpoint.json"
    path.write_text("{}")
    runner = PipelineRunner(isolated_config, FakeResourceGuard(), {}, {})
    defect = error_type("checkpoint parser programmer defect")
    monkeypatch.setattr(
        orchestrator_module.json,
        "loads",
        lambda payload: (_ for _ in ()).throw(defect),
    )

    with pytest.raises(error_type) as caught:
        runner.load_checkpoint(path, "ABO-00000")

    assert caught.value is defect


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
            asset_sha=f"{index:064x}",
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
    assets = tuple(f"{index:064x}" for index in range(500))
    write_instances(shard_context, assets)
    runner.checkpoint.quality_outcomes = {
        asset_sha: "failure" if index < 51 else "completed"
        for index, asset_sha in enumerate(assets)
    }

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
    assert report.completed_counts == {
        "commands": 0,
        "outcomes": 500,
        "completed_assets": 449,
        "quarantined_assets": 51,
        "failure_assets": 51,
        "schema_failure_assets": 0,
    }
    assert "resume" in report.safe_resume_command
    assert len(report.recovery_choices) >= 2


class FakeProcess:
    def __init__(self, pid, polls):
        self.pid = pid
        self._polls = deque(polls)
        self.returncode = None
        self.waited = 0
        self.wait_timeouts = []
        self.control = None

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        if self._polls:
            value = self._polls.popleft()
            if value is not None:
                self.returncode = value
        return self.returncode

    def wait(self, timeout):
        self.waited += 1
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired(("supervisor",), timeout)
        return self.returncode

    def send_control(self, action):
        if self.control is None:
            raise AssertionError("missing fake supervisor control")
        self.control(action)


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

    def supervisor_factory(argv, environment):
        process = processes[len(created)]
        created.append(
            (
                tuple(argv),
                {"env": dict(environment), "start_new_session": True},
                process,
            )
        )

        def control(action):
            if (group_ids or {}).get(process.pid, process.pid) != process.pid:
                raise ProcessGroupSafetyError("reused stable supervisor")
            sent_signal = {
                "pause": signal.SIGSTOP,
                "resume": signal.SIGCONT,
                "terminate": signal.SIGTERM,
                "kill": signal.SIGKILL,
            }[action]
            signals.append((process.pid, sent_signal))
            if action == "kill":
                process.returncode = -sent_signal

        process.control = control
        return process

    runner = PipelineRunner(
        config,
        guard,
        {},
        {},
        supervisor_factory=supervisor_factory,
        monotonic_clock=clock,
        sleeper=clock.sleep,
        termination_grace_seconds=2,
        reap_timeout_seconds=3,
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


def test_short_leaf_completion_is_polled_without_five_second_tail_latency(
    isolated_config,
):
    process = FakeProcess(103, [None, 0])
    runner, _, _ = process_runner(
        isolated_config, FakeResourceGuard(), [process]
    )

    runner.execute(CommandSpec("worker", ("worker",)), "ABO-00000")

    assert runner.monotonic_clock() == pytest.approx(0.1)


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


def test_failed_blender_worker_terminates_and_reaps_all_fourteen_workers(
    isolated_config,
):
    processes = [FakeProcess(220, [7])] + [
        FakeProcess(221 + index, [None, None, None]) for index in range(13)
    ]
    runner, created, signals = process_runner(
        isolated_config, FakeResourceGuard(), processes
    )
    command = CommandSpec(
        "render_cond",
        ("worker",),
        gpu_ranks=7,
        workers_per_gpu=2,
    )

    with pytest.raises(subprocess.CalledProcessError):
        runner.execute(command, "shard")

    assert len(created) == 14
    assert all(process.waited == 1 for process in processes)
    assert all(
        (process.pid, signal.SIGTERM) in signals for process in processes[1:]
    )


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


def test_stable_supervisor_identity_mismatch_never_signals_unrelated_process(
    isolated_config,
):
    process = FakeProcess(501, [None] * 10)
    runner, _, signals = process_runner(
        isolated_config,
        FakeResourceGuard((RuntimeError("sampler failed"),)),
        [process],
        group_ids={501: 999},
    )

    with pytest.raises(ProcessGroupSafetyError, match="stable supervisor"):
        runner.execute(CommandSpec("worker", ("worker",)), "shard")

    assert signals == []
    assert process.waited == 2


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
        self.loads = 0

    def load(self):
        self.loads += 1
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


def test_pipeline_services_cache_immutable_registry_for_shard_planning(
    isolated_config,
):
    registry = FakeRegistry(
        pd.DataFrame(
            {
                "sha256": ("a" * 64, "b" * 64),
                "owner_source": ("ABO", "HSSD"),
                "shard_id": ("ABO-00000", "HSSD-00000"),
            }
        ),
        isolated_config.paths.data2_root / "control/assets.parquet",
    )
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        registry_store=registry,
    )

    assert services._registry_shas("ABO", "ABO-00000") == ("a" * 64,)
    assert services._registry_shas("HSSD", "HSSD-00000") == ("b" * 64,)
    assert registry.loads == 1


def test_pipeline_services_accept_verified_archive_tool_commit(
    isolated_config, monkeypatch
):
    expected = "a" * 40
    monkeypatch.setenv("PIXAL3D_TOOL_COMMIT", expected)
    monkeypatch.setattr(
        orchestrator_module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("archive deployment ran git"),
    )
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
    )

    assert services._resolved_tool_commit() == expected


def test_pipeline_services_reject_invalid_archive_tool_commit(
    isolated_config, monkeypatch
):
    monkeypatch.setenv("PIXAL3D_TOOL_COMMIT", "not-a-commit")
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
    )

    with pytest.raises(InfrastructureError, match="deployment identity"):
        services._resolved_tool_commit()


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
        gate="production",
    ):
        self.calls.append(
            (
                source,
                raw_relative_path,
                excluding_shard_id,
                excluding_batch_id,
                gate,
            )
        )
        return self.value


class FakeAccounting:
    def __init__(self, *, failure=None):
        self.failure = failure
        self.deltas = []
        self.reconciliations = 0

    def record_registry_delta(self, path, delta_bytes):
        if self.failure is not None:
            raise self.failure
        self.deltas.append((Path(path), delta_bytes))

    def reconcile_at_shard_boundary(self):
        if self.failure is not None:
            raise self.failure
        self.reconciliations += 1
        return (0, 0)


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


def test_freeze_writes_independent_batch_files_concurrently(
    isolated_config, monkeypatch
):
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
    )
    shas = tuple(f"{index:064x}" for index in range(8))
    batches = tuple((asset,) for asset in shas)
    original = orchestrator_module._atomic_write_text
    lock = threading.Lock()
    active = 0
    maximum = 0

    def measured_write(path, payload):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.03)
        try:
            return original(path, payload)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(
        orchestrator_module, "_atomic_write_text", measured_write
    )

    services._freeze_batches(
        "production", "ABO", "ABO-00000", batches, shas
    )

    assert maximum > 1


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
        project_accounting=FakeAccounting(),
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

    services.resume("production", "ABO", "ABO-00000")

    assert [item.batch_id for item in runner.resumes] == ["batch000"]
    assert audits == runner.resumes


def test_run_batch_executes_only_the_claimed_frozen_batch(isolated_config):
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
    accounting = FakeAccounting()
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        registry_store=registry,
        runner=runner,
        project_accounting=accounting,
        batch_auditor=audits.append,
        published_batch_verifier=lambda context: None,
    )
    services._freeze_batches(
        "production",
        "ABO",
        "ABO-00000",
        (shas[:2], shas[2:]),
        shas,
    )

    context = services.run_batch(
        "production", "ABO", "ABO-00000", "batch001"
    )

    assert context.batch_id == "batch001"
    assert [item.batch_id for item in runner.runs] == ["batch001"]
    assert audits == runner.runs
    assert accounting.reconciliations == 1


def test_run_batch_rejects_unknown_batch(isolated_config):
    asset = "a" * 64
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        registry_store=FakeRegistry(
            pd.DataFrame(
                {
                    "sha256": [asset],
                    "owner_source": ["ABO"],
                    "shard_id": ["ABO-00000"],
                }
            ),
            isolated_config.paths.data2_root / "control/assets.parquet",
        ),
    )
    services._freeze_batches(
        "production", "ABO", "ABO-00000", ((asset,),), (asset,)
    )

    with pytest.raises(ValueError, match="unknown frozen batch"):
        services.run_batch(
            "production", "ABO", "ABO-00000", "batch999"
        )


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
        project_accounting=FakeAccounting(),
        batch_auditor=lambda context: None,
        published_batch_verifier=lambda context: None,
    )
    services.plan(
        "smoke", "ABO", "ABO-00000", count=2, freeze=True
    )

    services.resume("smoke", "ABO", "ABO-00000")

    assert [context.instances.read_text().splitlines() for context in runner.resumes] == [
        list(shas[:2])
    ]


@pytest.mark.parametrize("gate", ["smoke", "pilot"])
def test_qualification_audit_reloads_gate_checkpoint_with_real_runner(
    gate, isolated_config
):
    gib = 1024**3
    asset_sha = "a" * 64
    registry = FakeRegistry(
        pd.DataFrame(
            {
                "sha256": [asset_sha],
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
        raw_archive_verifier=lambda context: None,
        project_accounting=FakeAccounting(),
    )
    services.plan(gate, "ABO", "ABO-00000", count=1, freeze=True)
    context = ShardContext.from_config(
        isolated_config,
        "ABO",
        "ABO-00000",
        "batch000",
        gate=gate,
    )
    services.runner.save_checkpoint(
        services._checkpoint_path(context),
        PipelineCheckpoint(
            context.shard_id,
            gate=gate,
            quality_outcomes={asset_sha: "completed"},
        ),
    )
    services.published_batch_verifier = lambda active: services._quality_state(
        active
    )

    services.audit(gate, "ABO", "ABO-00000")


def test_gate_scopes_isolate_qualification_from_full_production_shard(
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
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        registry_store=registry,
        pilot_reader=FakePilotReader(100),
        disk_usage=lambda path: SimpleNamespace(
            total=1000 * gib, free=1000 * gib
        ),
    )

    services.plan("smoke", "ABO", "ABO-00000", count=2, freeze=True)
    services.plan("production", "ABO", "ABO-00000", freeze=True)

    smoke = (
        isolated_config.paths.data2_root
        / "control/qualification/smoke/shards/ABO/ABO-00000"
    )
    production = (
        isolated_config.paths.data2_root / "control/shards/ABO/ABO-00000"
    )
    assert (smoke / "batch000.txt").read_text().splitlines() == list(shas[:2])
    assert (production / "batch000.txt").read_text().splitlines() == list(shas)
    assert json.loads((smoke / "batches.json").read_text())["gate"] == "smoke"
    assert json.loads((production / "batches.json").read_text())["gate"] == "production"


def test_production_never_accepts_a_restricted_count(isolated_config):
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
    )

    with pytest.raises(ValueError, match="full canonical shard"):
        services.run("production", "ABO", "ABO-00000", count=1)


def test_resume_gate_never_falls_back_to_another_frozen_scope(isolated_config):
    gib = 1024**3
    shas = ("a" * 64, "b" * 64)
    registry = FakeRegistry(
        pd.DataFrame(
            {
                "sha256": shas,
                "owner_source": ["ABO"] * 2,
                "shard_id": ["ABO-00000"] * 2,
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
    services.plan("smoke", "ABO", "ABO-00000", count=1, freeze=True)

    with pytest.raises(InfrastructureError, match="production"):
        services.resume("production", "ABO", "ABO-00000")


@pytest.mark.parametrize("error_type", [AssertionError, AttributeError, MemoryError])
def test_frozen_manifest_parser_does_not_hide_programmer_defects(
    error_type, isolated_config, monkeypatch
):
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )
    root = services._batch_root("smoke", "ABO", "ABO-00000")
    root.mkdir(parents=True)
    (root / "batches.json").write_text("{}")
    defect = error_type("frozen manifest programmer defect")
    monkeypatch.setattr(
        orchestrator_module.json,
        "loads",
        lambda payload: (_ for _ in ()).throw(defect),
    )

    with pytest.raises(error_type) as caught:
        services._read_frozen_batches(
            "smoke", "ABO", "ABO-00000", ("a" * 64,)
        )

    assert caught.value is defect


def test_gate_identity_isolates_runtime_and_publication_paths(isolated_config):
    smoke = ShardContext.from_config(
        isolated_config,
        "ABO",
        "ABO-00000",
        "batch000",
        gate="smoke",
    )
    production = ShardContext.from_config(
        isolated_config,
        "ABO",
        "ABO-00000",
        "batch000",
        gate="production",
    )
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )

    assert smoke.instances != production.instances
    assert smoke.work_root != production.work_root
    assert services._checkpoint_path(smoke) != services._checkpoint_path(production)
    assert services._quality_ledger_path(smoke) != services._quality_ledger_path(production)
    assert services._raw_archive_paths(smoke) != services._raw_archive_paths(production)


def test_family_dependencies_make_pbr_and_ss_depend_on_shape(config):
    dependencies = orchestrator_module.family_dependencies(config)

    assert dependencies["PBR-256"] == frozenset({"shape-256"})
    assert dependencies["PBR-512"] == frozenset({"shape-512"})
    assert dependencies["PBR-1024"] == frozenset({"shape-1024"})
    assert dependencies["SS-64"] == frozenset({"shape-1024"})


def test_record_family_exclusion_round_trips_without_terminal_outcome(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "family-ledger", "ABO", "ABO-00000"
    )
    asset_sha = "a" * 64
    write_instances(context, (asset_sha,))
    runner = PipelineRunner(
        isolated_config,
        FakeResourceGuard(),
        {},
        {},
        command_builder=lambda _context, _config: (),
    )
    checkpoint_path = tmp_path / "checkpoint.json"
    ledger_path = tmp_path / "quality.json"
    runner.active_context = context
    runner.active_checkpoint = PipelineCheckpoint(context.shard_id)
    runner.active_checkpoint_path = checkpoint_path
    runner._active_quality_assets = (asset_sha,)
    runner._active_instances_sha256 = sha256(
        context.instances.read_bytes()
    ).hexdigest()
    runner._active_quality_ledger = orchestrator_module._empty_quality_ledger(
        context
    )
    runner._active_quality_ledger_path = ledger_path
    orchestrator_module._save_quality_ledger(
        ledger_path, runner._active_quality_ledger
    )

    runner.record_family_exclusion(
        asset_sha,
        ("PBR-256", "PBR-512", "PBR-1024"),
        category="unsupported_shader",
        stage="dump_pbr",
        reason="Material is not supported",
        attempts=1,
    )

    ledger = json.loads(ledger_path.read_text())
    assert ledger["schema_version"] == 3
    assert set(ledger["family_exclusions"][asset_sha]) == {
        "PBR-256",
        "PBR-512",
        "PBR-1024",
    }
    assert runner.family_exclusions(asset_sha)["PBR-256"] == {
        "category": "unsupported_shader",
        "stage": "dump_pbr",
        "reason": "Material is not supported",
        "attempts": 1,
    }
    assert asset_sha not in runner.active_checkpoint.quality_outcomes


def test_schema_two_quality_ledger_loads_with_empty_family_exclusions(
    tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "legacy-family-ledger", "ABO", "ABO-00000"
    )
    path = tmp_path / "legacy-quality.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source": context.source,
                "shard_id": context.shard_id,
                "gate": context.gate,
                "batches": {},
                "entries": [],
                "quarantine": {},
            }
        )
    )

    ledger = orchestrator_module._load_quality_ledger(path, context)

    assert ledger["schema_version"] == 3
    assert ledger["family_exclusions"] == {}


def _family_services(isolated_config, tmp_path, **service_kwargs):
    context = ShardContext.for_test(
        tmp_path / "family-routing", "ABO", "ABO-00000"
    )
    assets = ("a" * 64, "b" * 64)
    write_instances(context, assets)
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        **service_kwargs,
    )
    runner = services.runner
    runner.active_context = context
    runner.active_checkpoint = PipelineCheckpoint(context.shard_id)
    runner.active_checkpoint_path = tmp_path / "checkpoint.json"
    runner._active_quality_assets = assets
    runner._active_instances_sha256 = sha256(
        context.instances.read_bytes()
    ).hexdigest()
    runner._active_quality_ledger = orchestrator_module._empty_quality_ledger(
        context
    )
    runner._active_quality_ledger_path = tmp_path / "quality.json"
    orchestrator_module._save_quality_ledger(
        runner._active_quality_ledger_path, runner._active_quality_ledger
    )
    return services, context, runner, assets


def test_pbr_exclusion_keeps_geometry_command_instances(
    isolated_config, tmp_path
):
    services, context, runner, assets = _family_services(
        isolated_config, tmp_path
    )
    full_pbr, geometry_only = assets
    runner.record_family_exclusion(
        geometry_only,
        ("PBR-256", "PBR-512", "PBR-1024"),
        category="unsupported_shader",
        stage="dump_pbr",
        reason="Material is not supported",
        attempts=1,
    )
    pbr = CommandSpec(
        "encode_pbr_256",
        ("python", "worker.py", "--instances", str(context.instances)),
    )
    shape = replace(pbr, name="encode_shape_256")

    pbr_launch = runner._command_for_eligible_assets(
        pbr, context, runner.active_checkpoint
    )
    shape_launch = runner._command_for_eligible_assets(
        shape, context, runner.active_checkpoint
    )

    pbr_instances = Path(
        pbr_launch.argv[pbr_launch.argv.index("--instances") + 1]
    )
    shape_instances = Path(
        shape_launch.argv[shape_launch.argv.index("--instances") + 1]
    )
    assert pbr_instances.read_text().splitlines() == [full_pbr]
    assert shape_instances.read_text().splitlines() == list(assets)
    assert services._candidate_assets(context, "SS-64") == assets


def test_shape_resolution_exclusion_cascades_only_to_dependents(
    isolated_config, tmp_path
):
    services, context, runner, assets = _family_services(
        isolated_config, tmp_path
    )
    asset = assets[0]

    services._exclude_stage_failure(
        context,
        asset,
        "encode_shape_256",
        category="missing_output",
        reason="missing shape latent",
    )

    assert asset not in services._candidate_assets(context, "shape-256")
    assert asset not in services._candidate_assets(context, "PBR-256")
    assert asset in services._candidate_assets(context, "shape-512")
    assert asset in services._candidate_assets(context, "SS-64")
    assert asset not in runner.active_checkpoint.quality_outcomes


def _write_pbr_stage_records(context, rows):
    path = context.work_root / "pbr_dumps/new_records/part_0.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_csv(path, index=False)


def test_dump_pbr_validator_excludes_unsupported_material_only(
    isolated_config, tmp_path
):
    services, context, runner, assets = _family_services(
        isolated_config, tmp_path
    )
    full_pbr, geometry_only = assets
    output = context.work_root / "pbr_dumps" / f"{full_pbr}.pickle"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump({"objects": [], "materials": []}, stream)
    _write_pbr_stage_records(
        context,
        (
            {"sha256": full_pbr, "pbr_dumped": True},
            {
                "sha256": geometry_only,
                "pbr_dumped": False,
                "error_category": "unsupported_shader",
                "error_reason": "Material is not supported",
            },
        ),
    )
    runner.active_checkpoint.attempts["dump_pbr"] = 1

    assert services.validators["dump_pbr"]() is True

    assert set(runner.family_exclusions(geometry_only)) == {
        "PBR-256",
        "PBR-512",
        "PBR-1024",
    }
    assert geometry_only in services._candidate_assets(
        context, "shape-256"
    )
    assert geometry_only not in runner.active_checkpoint.quality_outcomes


def test_dump_pbr_validator_retries_transient_failure_before_exclusion(
    isolated_config, tmp_path
):
    services, context, runner, assets = _family_services(
        isolated_config, tmp_path
    )
    asset = assets[0]
    _write_pbr_stage_records(
        context,
        (
            {
                "sha256": asset,
                "pbr_dumped": False,
                "error_category": "timeout",
                "error_reason": "PBR dump timed out after 17 seconds",
            },
        ),
    )
    runner.active_checkpoint.attempts["dump_pbr"] = 1

    assert services.validators["dump_pbr"]() is False

    assert runner.family_exclusions(asset) == {}
    assert asset not in runner.active_checkpoint.quality_outcomes


@pytest.mark.parametrize(
    ("command_name", "expected_exclusions"),
    (
        ("dual_grid_256", {"shape-256", "PBR-256"}),
        ("voxelize_pbr_256", {"PBR-256"}),
        ("encode_ss_64", {"SS-64"}),
    ),
)
def test_family_stage_validator_does_not_globally_quarantine_missing_output(
    isolated_config, tmp_path, command_name, expected_exclusions
):
    services, context, runner, assets = _family_services(
        isolated_config, tmp_path
    )
    runner.active_checkpoint.attempts[command_name] = 1

    assert services.validators[command_name]() is True

    for asset in assets:
        assert set(runner.family_exclusions(asset)) == expected_exclusions
        assert asset not in runner.active_checkpoint.quality_outcomes


def test_terminal_validation_completes_geometry_only_asset(
    isolated_config, tmp_path
):
    calls = []

    def validate_family(context, asset_sha, family):
        calls.append((asset_sha, family))

    services, context, runner, assets = _family_services(
        isolated_config,
        tmp_path,
        family_output_validator=validate_family,
    )
    full_pbr, geometry_only = assets
    runner.record_family_exclusion(
        geometry_only,
        ("PBR-256", "PBR-512", "PBR-1024"),
        category="unsupported_shader",
        stage="dump_pbr",
        reason="Material is not supported",
        attempts=1,
    )

    services.validate_outputs(context)

    assert runner.active_checkpoint.quality_outcomes == {
        full_pbr: "completed",
        geometry_only: "completed",
    }
    geometry_families = {
        family for asset, family in calls if asset == geometry_only
    }
    assert geometry_families == {
        "common",
        "SS-64",
        "shape-256",
        "shape-512",
        "shape-1024",
    }


def test_family_pack_membership_keeps_pbr_subset_of_matching_shape(
    isolated_config, tmp_path
):
    services, _context, runner, assets = _family_services(
        isolated_config, tmp_path
    )
    full_pbr, geometry_only = assets
    runner.record_family_exclusion(
        geometry_only,
        ("PBR-256", "PBR-512", "PBR-1024"),
        category="unsupported_shader",
        stage="dump_pbr",
        reason="Material is not supported",
        attempts=1,
    )

    included = services._family_included_assets(assets)

    assert included["common"] == assets
    for resolution in isolated_config.targets.resolutions:
        assert included[f"shape-{resolution}"] == assets
        assert included[f"PBR-{resolution}"] == (full_pbr,)
        assert set(included[f"PBR-{resolution}"]) <= set(
            included[f"shape-{resolution}"]
        )


def test_published_pack_audit_restores_durable_family_exclusions(
    isolated_config, tmp_path
):
    services, context, runner, assets = _family_services(
        isolated_config, tmp_path, tool_commit="test-commit"
    )
    full_pbr, geometry_only = assets
    ledger_path = services._quality_ledger_path(context)
    runner._active_quality_ledger_path = ledger_path
    orchestrator_module._save_quality_ledger(
        ledger_path, runner._active_quality_ledger
    )
    runner.record_family_exclusion(
        geometry_only,
        ("PBR-256", "PBR-512", "PBR-1024"),
        category="unsupported_shader",
        stage="dump_pbr",
        reason="Material is not supported",
        attempts=1,
    )
    included = services._family_included_assets(assets)
    members = services._pack_members_for_assets(context, included)
    for family_members in members.values():
        for relative in family_members:
            path = context.output_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.as_posix().encode())
    publish_pack(
        isolated_config.paths.data2_root,
        context.output_root,
        members,
        context.shard_id,
        source=context.source,
        batch_id=context.batch_id,
        config_hash=isolated_config.config_hash(),
        tool_commit="test-commit",
        asset_sha256s=assets,
        included_asset_sha256s_by_family=included,
    )
    write_quality_checkpoint(
        services,
        context,
        {asset: "completed" for asset in assets},
    )
    runner.active_context = None
    runner.active_checkpoint = None
    runner.active_checkpoint_path = None
    runner._active_quality_ledger = None
    runner._active_quality_ledger_path = None

    services._verify_published_batch(context)


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
        services.resume("production", "ABO", "ABO-00000")


def write_raw_metadata(context, records):
    metadata = context.source_root / "raw/metadata.csv"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(metadata, index=False)


def write_instances(context, shas):
    context.instances.parent.mkdir(parents=True, exist_ok=True)
    context.instances.write_text("".join(f"{sha}\n" for sha in shas))


def test_read_raw_records_normalizes_content_and_companions(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "raw-contract", "3D-FUTURE", "3D-FUTURE-00000"
    )
    asset_sha = sha256(b"image identity").hexdigest()
    content_sha = sha256(b"obj content").hexdigest()
    companion = "raw/3D-FUTURE-model/item/model.mtl"
    companion_sha = sha256(b"material").hexdigest()
    write_raw_metadata(
        context,
        (
            {
                "sha256": asset_sha,
                "local_path": "raw/3D-FUTURE-model/item/raw_model.obj",
                "content_sha256": content_sha,
                "companion_files": json.dumps({companion: companion_sha}),
            },
        ),
    )
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )

    records = services._read_raw_records(
        context.source_root / "raw/metadata.csv", (asset_sha,)
    )

    assert records == (
        {
            "sha256": asset_sha,
            "local_path": "raw/3D-FUTURE-model/item/raw_model.obj",
            "content_sha256": content_sha,
            "companion_files": {companion: companion_sha},
        },
    )


def test_read_raw_records_rejects_duplicate_companion_json_keys(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "duplicate-companion",
        "3D-FUTURE",
        "3D-FUTURE-00000",
    )
    asset_sha = sha256(b"image identity").hexdigest()
    companion = "raw/3D-FUTURE-model/item/model.mtl"
    first_sha = sha256(b"first material").hexdigest()
    second_sha = sha256(b"second material").hexdigest()
    write_raw_metadata(
        context,
        (
            {
                "sha256": asset_sha,
                "local_path": "raw/3D-FUTURE-model/item/raw_model.obj",
                "content_sha256": sha256(b"obj content").hexdigest(),
                "companion_files": (
                    f'{{"{companion}":"{first_sha}",'
                    f'"{companion}":"{second_sha}"}}'
                ),
            },
        ),
    )
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )

    with pytest.raises(ValidationError, match="duplicate raw companion path"):
        services._read_raw_records(
            context.source_root / "raw/metadata.csv", (asset_sha,)
        )


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
        {
            "sha256": asset_sha,
            "local_path": relative,
            "content_sha256": asset_sha,
            "companion_files": "{}",
        }
    ]


def test_stage_raw_copies_full_declared_package_using_content_hashes(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "full-raw-package", "3D-FUTURE", "3D-FUTURE-00000"
    )
    asset_sha = sha256(b"image identity").hexdigest()
    files = {
        "raw/3D-FUTURE-model/item/raw_model.obj": b"mtllib model.mtl\nmesh",
        "raw/3D-FUTURE-model/item/model.mtl": b"map_Kd texture.png\n",
        "raw/3D-FUTURE-model/item/texture.png": b"texture",
        "raw/3D-FUTURE-model/item/image.jpg": b"image identity",
    }
    for relative, contents in files.items():
        source = context.source_root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(contents)
    primary = "raw/3D-FUTURE-model/item/raw_model.obj"
    companions = {
        relative: sha256(contents).hexdigest()
        for relative, contents in files.items()
        if relative != primary
    }
    write_instances(context, (asset_sha,))
    write_raw_metadata(
        context,
        (
            {
                "sha256": asset_sha,
                "local_path": primary,
                "content_sha256": sha256(files[primary]).hexdigest(),
                "companion_files": json.dumps(companions),
            },
        ),
    )
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )

    services.stage_raw(context)

    assert {
        relative: (context.download_root / relative).read_bytes()
        for relative in files
    } == files


def _download_validation_service(isolated_config, context, attempts):
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        published_batch_verifier=lambda _context: (_ for _ in ()).throw(
            ValidationError("published outputs are not present")
        ),
    )
    checkpoint = PipelineCheckpoint(context.shard_id, gate=context.gate)
    checkpoint.attempts["download"] = attempts
    services.runner.active_context = context
    services.runner.active_checkpoint = checkpoint
    outcomes = []

    def record(asset_sha, outcome):
        outcomes.append((asset_sha, outcome))
        checkpoint.quality_outcomes[asset_sha] = outcome

    services.runner.record_quality_outcome = record
    return services, checkpoint, outcomes


def test_partial_download_waits_until_third_attempt_before_quarantine(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "partial-download", "ObjaverseXL_github", "ObjaverseXL_github-00000"
    )
    first = sha256(b"first").hexdigest()
    missing = "b" * 64
    write_instances(context, (first, missing))
    write_raw_metadata(
        context, ({"sha256": first, "local_path": "raw/first.glb"},)
    )

    services, checkpoint, outcomes = _download_validation_service(
        isolated_config, context, 1
    )
    assert services.validators["download"]() is False
    assert outcomes == []

    checkpoint.attempts["download"] = 2
    assert services.validators["download"]() is False
    assert outcomes == []

    checkpoint.attempts["download"] = 3
    assert services.validators["download"]() is True
    assert outcomes == [(missing, "failure")]


def test_empty_partial_download_stays_fail_closed_at_terminal_attempt(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "empty-download", "ObjaverseXL_github", "ObjaverseXL_github-00000"
    )
    first = "a" * 64
    second = "b" * 64
    write_instances(context, (first, second))
    metadata = context.source_root / "raw/metadata.csv"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text("sha256,local_path\n")

    services, _checkpoint, outcomes = _download_validation_service(
        isolated_config, context, 3
    )
    assert services.validators["download"]() is False
    assert outcomes == []


def test_stage_raw_uses_only_non_quarantined_assets(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "stage-eligible", "ObjaverseXL_github", "ObjaverseXL_github-00000"
    )
    contents = b"eligible raw"
    completed = sha256(contents).hexdigest()
    quarantined = "f" * 64
    relative = "raw/models/item.glb"
    source = context.source_root / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(contents)
    write_instances(context, tuple(sorted((completed, quarantined))))
    write_raw_metadata(
        context, ({"sha256": completed, "local_path": relative},)
    )
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )
    checkpoint = PipelineCheckpoint(context.shard_id, gate=context.gate)
    checkpoint.quality_outcomes[quarantined] = "failure"
    services.runner.active_checkpoint = checkpoint

    services.stage_raw(context)

    staged = pd.read_csv(context.download_root / "raw/metadata.csv")
    assert staged.to_dict("records") == [
        {
            "sha256": completed,
            "local_path": relative,
            "content_sha256": completed,
            "companion_files": "{}",
        }
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
        project_accounting=FakeAccounting(),
        output_validator=lambda context: None,
        pack_publisher=pack_publisher,
        published_batch_verifier=lambda context: None,
        tool_commit="test-commit",
    )
    write_quality_checkpoint(services, context, {asset_sha: "completed"})

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
        project_accounting=FakeAccounting(),
        output_validator=lambda context: None,
        pack_member_builder=lambda context: {"common": []},
        pack_publisher=lambda *args, **kwargs: calls.append(args),
        published_batch_verifier=lambda context: None,
    )
    write_quality_checkpoint(services, context, {"a" * 64: "completed"})

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
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        project_accounting=FakeAccounting(),
        tool_commit="test-commit",
    )
    write_quality_checkpoint(services, context, {asset_sha: "completed"})
    members = services._pack_members(context)
    for family_members in members.values():
        for relative in family_members:
            path = context.output_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.as_posix().encode())
    publish_pack(
        isolated_config.paths.data2_root,
        context.output_root,
        members,
        context.shard_id,
        source=context.source,
        batch_id=context.batch_id,
        config_hash=isolated_config.config_hash(),
        tool_commit="test-commit",
        asset_sha256s=(asset_sha,),
        completed_count=1,
        quarantined_count=0,
    )
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
        project_accounting=FakeAccounting(),
        published_batch_verifier=audit_calls.append,
        tool_commit="test-commit",
    )
    write_quality_checkpoint(services, context, {asset_sha: "completed"})
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


def test_raw_archive_contains_primary_and_companion_content_hashes(
    isolated_config
):
    context = configured_context(isolated_config, "3D-FUTURE")
    asset_sha = sha256(b"image identity").hexdigest()
    primary = "raw/3D-FUTURE-model/item/raw_model.obj"
    files = {
        primary: b"mtllib model.mtl\nmesh",
        "raw/3D-FUTURE-model/item/model.mtl": b"map_Kd texture.png\n",
        "raw/3D-FUTURE-model/item/texture.png": b"texture",
    }
    for relative, contents in files.items():
        source = context.source_root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(contents)
    write_instances(context, (asset_sha,))
    write_raw_metadata(
        context,
        (
            {
                "sha256": asset_sha,
                "local_path": primary,
                "content_sha256": sha256(files[primary]).hexdigest(),
                "companion_files": json.dumps(
                    {
                        relative: sha256(contents).hexdigest()
                        for relative, contents in files.items()
                        if relative != primary
                    }
                ),
            },
        ),
    )
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        reference_counter=FakeReferenceCounter(1),
        project_accounting=FakeAccounting(),
        published_batch_verifier=lambda active_context: None,
        tool_commit="test-commit",
    )
    write_quality_checkpoint(services, context, {asset_sha: "completed"})
    services.stage_raw(context)

    services.archive_raw(context)

    _, manifest_path = services._raw_archive_paths(context)
    manifest = json.loads(manifest_path.read_text())
    assert {
        item["path"]: item["sha256"] for item in manifest["members"]
    } == {
        relative: sha256(contents).hexdigest()
        for relative, contents in files.items()
    }


def test_raw_archive_checks_primary_reference_before_companion_cleanup(
    isolated_config
):
    context = configured_context(isolated_config, "3D-FUTURE")
    asset_sha = sha256(b"image identity").hexdigest()
    primary = "raw/3D-FUTURE-model/item/raw_model.obj"
    companion = "raw/3D-FUTURE-model/item/model.mtl"
    files = {primary: b"mesh", companion: b"material"}
    for relative, contents in files.items():
        source = context.source_root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(contents)
    write_instances(context, (asset_sha,))
    write_raw_metadata(
        context,
        (
            {
                "sha256": asset_sha,
                "local_path": primary,
                "content_sha256": sha256(files[primary]).hexdigest(),
                "companion_files": json.dumps(
                    {companion: sha256(files[companion]).hexdigest()}
                ),
            },
        ),
    )
    counter = FakeReferenceCounter(0)
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        reference_counter=counter,
        project_accounting=FakeAccounting(),
        published_batch_verifier=lambda active_context: None,
        tool_commit="test-commit",
    )
    write_quality_checkpoint(services, context, {asset_sha: "completed"})
    services.stage_raw(context)

    services.archive_raw(context)

    assert counter.calls == [
        (
            "3D-FUTURE",
            primary,
            "3D-FUTURE-00000",
            "batch000",
            "production",
        )
    ]
    assert all(not (context.source_root / relative).exists() for relative in files)


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
        project_accounting=FakeAccounting(),
        published_batch_verifier=lambda context: None,
        tool_commit="test-commit",
    )
    write_quality_checkpoint(services, context, {asset_sha: "completed"})
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
        project_accounting=FakeAccounting(),
        published_batch_verifier=lambda context: None,
        tool_commit="test-commit",
    )
    write_quality_checkpoint(services, context, {asset_sha: "completed"})
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


def test_production_dag_shape_validation_precedes_pbr_and_cleanup_requires_both(
    isolated_config, tmp_path, monkeypatch
):
    context = ShardContext.for_test(
        tmp_path / "batch", "ABO", "ABO-00000"
    )
    asset_sha = "a" * 64
    write_instances(context, (asset_sha,))
    validated = []

    monkeypatch.setattr(
        orchestrator_module,
        "validate_sparse_latent",
        lambda path, resolution, tokens: validated.append(Path(path)),
    )
    monkeypatch.setattr(
        orchestrator_module, "validate_scale", lambda path: None
    )
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )
    services.runner.active_context = context
    for relative in (
        services._shape_directory(256),
        services._pbr_directory(256),
    ):
        for view in isolated_config.targets.views:
            output = (
                context.output_root
                / relative
                / asset_sha
                / f"view{view:02d}.npz"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"latent")
            output.with_name(f"view{view:02d}_scale.json").write_text("{}")

    assert services.validators["encode_shape_256"]() is True
    assert validated
    assert all("shape_latents" in path.parts for path in validated)

    validated.clear()
    (context.work_root / "dual_grid_view_256").mkdir(parents=True)
    (context.work_root / "pbr_voxels_view_fix_256").mkdir(parents=True)
    services.cleanup_voxels(context, 256)
    assert {"shape_latents", "pbr_latents"} <= {
        next(
            part
            for part in path.parts
            if part in {"shape_latents", "pbr_latents"}
        )
        for path in validated
    }


class FakeSupervisor:
    def __init__(self, pid, polls, *, control=None):
        self.pid = pid
        self._polls = deque(polls)
        self.returncode = None
        self.controls = []
        self.wait_timeouts = []
        self._control = control

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        if self._polls:
            value = self._polls.popleft()
            if value is not None:
                self.returncode = value
        return self.returncode

    def send_control(self, action):
        self.controls.append(action)
        if self._control is not None:
            self._control(self, action)
        elif action == "kill":
            self.returncode = -signal.SIGKILL

    def wait(self, timeout):
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired(("supervisor",), timeout)
        return self.returncode


def supervisor_runner(config, guard, supervisors):
    created = []
    clock = FakeClock()

    def factory(argv, environment):
        supervisor = supervisors[len(created)]
        created.append((tuple(argv), dict(environment), supervisor))
        return supervisor

    runner = PipelineRunner(
        config,
        guard,
        {},
        {},
        supervisor_factory=factory,
        monotonic_clock=clock,
        sleeper=clock.sleep,
        termination_grace_seconds=2,
        reap_timeout_seconds=3,
        monitor_interval_seconds=1,
    )
    return runner, created


def test_supervisor_cleanup_waits_are_bounded_and_preserve_resource_stop(
    isolated_config,
):
    stop = ResourceDecision(ResourceAction.STOP, ("disk hard",))
    supervisor = FakeSupervisor(700, [None] * 10)
    runner, _ = supervisor_runner(
        isolated_config, FakeResourceGuard((stop,)), [supervisor]
    )

    with pytest.raises(ResourceLimitExceeded, match="disk hard"):
        runner.execute(CommandSpec("worker", ("worker",)), "shard")

    assert supervisor.controls == ["terminate", "kill"]
    assert supervisor.wait_timeouts == [3]


def test_resource_stop_survives_supervisor_cleanup_failures(isolated_config):
    stop = ResourceDecision(ResourceAction.STOP, ("memory hard",))

    def fail_control(supervisor, action):
        raise ProcessGroupSafetyError(f"cannot {action} stable supervisor")

    supervisor = FakeSupervisor(701, [None] * 10, control=fail_control)
    runner, _ = supervisor_runner(
        isolated_config, FakeResourceGuard((stop,)), [supervisor]
    )

    with pytest.raises(ResourceLimitExceeded, match="memory hard") as caught:
        runner.execute(CommandSpec("worker", ("worker",)), "shard")

    assert supervisor.wait_timeouts == [3, 3]
    assert any(
        "cleanup" in note.lower() for note in getattr(caught.value, "__notes__", ())
    )


def test_pidfd_supervisor_exit_race_never_controls_reused_pid(isolated_config):
    stop = ResourceDecision(ResourceAction.STOP, ("CPU hard",))
    reused_process_controls = []

    def exit_before_control(supervisor, action):
        supervisor.returncode = 0
        raise ProcessLookupError("original supervisor exited")

    supervisor = FakeSupervisor(702, [None] * 10, control=exit_before_control)
    runner, _ = supervisor_runner(
        isolated_config, FakeResourceGuard((stop,)), [supervisor]
    )

    with pytest.raises(ResourceLimitExceeded, match="CPU hard"):
        runner.execute(CommandSpec("worker", ("worker",)), "shard")

    assert reused_process_controls == []
    assert supervisor.controls == ["terminate"]
    assert supervisor.wait_timeouts == [3]


class CrashAfterPersistRunner(RecordingRunner):
    def __init__(self, config, command):
        super().__init__(config, (command,))
        self.launch_state = None

    def execute(self, command, shard_id):
        self.launch_state = asdict(self.checkpoint)
        raise SystemExit("simulated SIGKILL boundary")


def test_attempt_is_durable_before_command_launch(isolated_config, shard_context):
    command = CommandSpec("worker", ("worker",))
    runner = CrashAfterPersistRunner(isolated_config, command)

    with pytest.raises(SystemExit, match="SIGKILL"):
        runner.run_shard(shard_context)

    assert runner.launch_state["attempts"] == {command.name: 1}
    assert runner.launch_state["active_attempt"] == {
        "command": command.name,
        "attempt": 1,
    }


def quality_services(config, context, failures):
    def validate_asset(active_context, asset_sha):
        failure = failures.get(asset_sha)
        if failure is not None:
            raise failure

    services = PipelineServices(
        config,
        resource_guard=FakeResourceGuard(),
        asset_output_validator=validate_asset,
    )
    services.runner.command_builder = lambda active_context, active_config: (
        CommandSpec("validate_outputs", ("internal:validate_outputs",)),
    )
    return services


def test_quality_outcomes_are_durable_deduplicated_and_quarantined(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "quality", "ABO", "ABO-00000"
    )
    shas = tuple(f"{index:064x}" for index in range(500))
    write_instances(context, shas)
    failures = {
        asset_sha: ValidationError("invalid schema")
        for asset_sha in shas[:25]
    }
    services = quality_services(isolated_config, context, failures)

    services.runner.run_shard(context)

    checkpoint_path = services._checkpoint_path(context)
    checkpoint = services.runner.load_checkpoint(
        checkpoint_path, context.shard_id
    )
    assert len(checkpoint.quality_outcomes) == 500
    assert list(checkpoint.quality_outcomes.values()).count("schema_failure") == 25
    assert list(checkpoint.quality_outcomes.values()).count("completed") == 475
    assert checkpoint.completed_commands == ["validate_outputs"]

    checkpoint.completed_commands.clear()
    services.runner.save_checkpoint(checkpoint_path, checkpoint)
    services.runner.resume_shard(context)
    resumed = services.runner.load_checkpoint(checkpoint_path, context.shard_id)
    assert len(resumed.quality_outcomes) == 500
    assert resumed.quality_outcomes == checkpoint.quality_outcomes


def test_quality_window_is_durable_across_batches_and_service_restart(
    isolated_config, tmp_path
):
    first_context = replace(
        ShardContext.for_test(
            tmp_path / "quality-batch-000", "ABO", "ABO-00000"
        ),
        batch_id="batch000",
    )
    second_context = replace(
        ShardContext.for_test(
            tmp_path / "quality-batch-001", "ABO", "ABO-00000"
        ),
        batch_id="batch001",
    )
    assets = tuple(f"{index:064x}" for index in range(500))
    first_assets = assets[:250]
    second_assets = assets[250:]
    write_instances(first_context, first_assets)
    write_instances(second_context, second_assets)

    first_services = quality_services(
        isolated_config,
        first_context,
        {asset: OutputValidationError("terminal") for asset in first_assets[:26]},
    )
    first_services.runner.run_shard(first_context)
    assert first_services.runner.quality_gate.count == 250

    second_services = quality_services(
        isolated_config,
        second_context,
        {
            asset: OutputValidationError("terminal")
            for asset in second_assets[:26]
        },
    )
    with pytest.raises(PipelineStopped) as caught:
        second_services.runner.run_shard(second_context)

    assert caught.value.report.category == EscalationCategory.DATA_QUALITY
    assert "52/500" in caught.value.report.reason
    assert second_services.runner.quality_gate.count == 500
    ledger = json.loads(
        second_services._quality_ledger_path(second_context).read_text()
    )
    assert len(ledger["entries"]) == 500
    assert ledger["batches"]["batch000"]["admitted_prefix"] == 250
    assert ledger["batches"]["batch001"]["admitted_prefix"] == 250


def test_sparse_quality_resume_preserves_frozen_positions_at_exact_threshold(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "sparse-quality-resume", "ABO", "ABO-00000"
    )
    assets = tuple(f"{index:064x}" for index in range(5000))
    write_instances(context, assets)
    command = CommandSpec("worker", ("worker",))
    runner = RecordingRunner(isolated_config, (command,))
    runner.checkpoint.quality_outcomes = {
        assets[index]: "failure" for index in range(9, 5000, 10)
    }

    runner.resume_shard(context)

    assert runner.executed == [command.name]
    assert runner.quality_gate.count == 0
    assert runner.quality_gate.violation_reason() is None
    assert len(runner.checkpoint.quality_outcomes) == 500


@pytest.mark.parametrize(
    ("failure_type", "count", "reason"),
    [
        (OutputValidationError, 51, "end-to-end"),
        (ValidationError, 26, "schema"),
    ],
)
def test_durable_quality_strict_threshold_stops_without_command_retry(
    isolated_config, tmp_path, failure_type, count, reason
):
    context = ShardContext.for_test(
        tmp_path / reason, "ABO", "ABO-00000"
    )
    shas = tuple(f"{index:064x}" for index in range(500))
    write_instances(context, shas)
    failures = {
        asset_sha: failure_type("terminal asset failure")
        for asset_sha in shas[:count]
    }
    services = quality_services(isolated_config, context, failures)

    with pytest.raises(PipelineStopped) as caught:
        services.runner.run_shard(context)

    checkpoint = services.runner.load_checkpoint(
        services._checkpoint_path(context), context.shard_id
    )
    assert caught.value.report.category == EscalationCategory.DATA_QUALITY
    assert reason in caught.value.report.reason
    assert len(checkpoint.quality_outcomes) == 500
    assert checkpoint.attempts == {"validate_outputs": 1}


def test_dump_stats_and_voxel_validators_reject_structurally_corrupt_outputs(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "artifacts", "ABO", "ABO-00000"
    )
    asset_sha = "a" * 64
    write_instances(context, (asset_sha,))
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )
    services.runner.active_context = context

    mesh = context.work_root / "mesh_dumps" / f"{asset_sha}.pickle"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"not a pickle")
    assert services.validators["dump_mesh"]() is False
    with mesh.open("wb") as stream:
        pickle.dump({"objects": []}, stream)
    assert services.validators["dump_mesh"]() is True

    stats = context.metadata_root / "asset_stats/new_records/part_0.csv"
    stats.parent.mkdir(parents=True)
    stats.write_text("sha256,num_faces\n" + asset_sha + ",1\n")
    assert services.validators["asset_stats"]() is False
    stats.write_text(
        "sha256,num_faces,num_vertices\n" + asset_sha + ",1,3\n"
    )
    assert services.validators["asset_stats"]() is True

    for view in isolated_config.targets.views:
        voxel = (
            context.work_root
            / "dual_grid_view_256"
            / asset_sha
            / f"view{view:02d}.vxz"
        )
        voxel.parent.mkdir(parents=True, exist_ok=True)
        voxel.write_bytes(b"not a voxel")
        voxel.with_name(f"view{view:02d}_scale.json").write_text(
            '{"scale": 1.0}'
        )
    assert services.validators["dual_grid_256"]() is False


def test_completed_validator_io_failure_is_immediate_infrastructure_stop(
    isolated_config, shard_context
):
    command = CommandSpec("dump", ("worker",))
    reports = []
    runner = RecordingRunner(isolated_config, (command,), reports=reports)
    runner.checkpoint.complete(command.name)
    runner.validators[command.name] = lambda: (_ for _ in ()).throw(
        OSError("metadata storage unavailable")
    )

    with pytest.raises(PipelineStopped) as caught:
        runner.run_shard(shard_context)

    assert runner.executed == []
    assert caught.value.report.category == EscalationCategory.INFRASTRUCTURE
    assert "storage unavailable" in caught.value.report.reason


def test_admission_provider_failure_is_immediate_infrastructure_stop(
    isolated_config, shard_context
):
    command = CommandSpec("worker", ("worker",))
    reports = []
    runner = RecordingRunner(isolated_config, (command,), reports=reports)
    runner.resource_guard.wait_for_admission = lambda shard, name: (
        _ for _ in ()
    ).throw(RuntimeError("telemetry writer failed"))

    with pytest.raises(PipelineStopped) as caught:
        runner.run_shard(shard_context)

    assert runner.executed == []
    assert caught.value.report.category == EscalationCategory.INFRASTRUCTURE
    assert "telemetry writer failed" in caught.value.report.reason


def test_monitor_provider_failure_is_immediate_infrastructure_stop(
    isolated_config, shard_context
):
    command = CommandSpec("worker", ("worker",))
    supervisors = [FakeSupervisor(800, [None] * 10)]
    reports = []
    runner, _ = supervisor_runner(
        isolated_config,
        FakeResourceGuard((RuntimeError("sampler failed"),)),
        supervisors,
    )
    runner.command_builder = lambda context, config: (command,)
    runner.validators = {command.name: lambda: True}
    runner.report_writer = reports.append
    runner.checkpoint_path = lambda context: context.work_root / "checkpoint.json"

    with pytest.raises(PipelineStopped) as caught:
        runner.run_shard(shard_context)

    assert caught.value.report.category == EscalationCategory.INFRASTRUCTURE
    assert "sampler failed" in caught.value.report.reason
    assert len(supervisors[0].wait_timeouts) == 1


def test_checkpoint_parent_symlink_is_never_followed(isolated_config, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(outside, target_is_directory=True)
    checkpoint_path = linked_parent / "checkpoint.json"
    runner = PipelineRunner(isolated_config, FakeResourceGuard(), {}, {})

    with pytest.raises(CheckpointError, match="unsafe checkpoint"):
        runner.save_checkpoint(
            checkpoint_path, PipelineCheckpoint("ABO-00000")
        )

    assert not (outside / "checkpoint.json").exists()


def test_raw_metadata_parent_symlink_is_never_followed(
    isolated_config, tmp_path
):
    root = tmp_path / "source"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    asset_sha = "a" * 64
    (outside / "metadata.csv").write_text(
        "sha256,local_path\n" + asset_sha + ",raw/item.glb\n"
    )
    (root / "raw").symlink_to(outside, target_is_directory=True)
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )

    with pytest.raises(ValidationError, match="raw metadata"):
        services._read_raw_records(
            root / "raw/metadata.csv", (asset_sha,)
        )


def test_staging_root_with_symlink_ancestor_is_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    root = linked / "download"
    payload = b"payload"

    with pytest.raises(ValidationError, match="staging"):
        orchestrator_module._atomic_stage_stream(
            io.BytesIO(payload),
            root,
            Path("raw/item.glb"),
            sha256(payload).hexdigest(),
        )

    assert not (outside / "download/raw/item.glb").exists()


def write_quality_checkpoint(services, context, outcomes):
    checkpoint = PipelineCheckpoint(context.shard_id)
    checkpoint.quality_outcomes.update(outcomes)
    checkpoint.complete("validate_outputs")
    services.runner.save_checkpoint(services._checkpoint_path(context), checkpoint)
    return checkpoint


def test_pack_publication_uses_frozen_quality_admission_counts(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "pack-quality", "ABO", "ABO-00000"
    )
    completed = "a" * 64
    quarantined = "b" * 64
    write_instances(context, (completed, quarantined))
    published = []

    def pack_publisher(data2_root, source_root, members, shard_id, **kwargs):
        published.append((members, kwargs))
        return tuple(SimpleNamespace(validated_at="now") for _ in PACK_FAMILIES)

    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        project_accounting=FakeAccounting(),
        output_validator=lambda context: None,
        pack_publisher=pack_publisher,
        published_batch_verifier=lambda context: None,
        tool_commit="test-commit",
    )
    write_quality_checkpoint(
        services,
        context,
        {completed: "completed", quarantined: "schema_failure"},
    )

    services.build_packs(context)

    members, kwargs = published[0]
    assert kwargs["asset_sha256s"] == (completed, quarantined)
    assert kwargs["included_asset_sha256s_by_family"] == {
        family: (completed,) for family in PACK_FAMILIES
    }
    assert all(
        quarantined not in relative.as_posix()
        for family_members in members.values()
        for relative in family_members
    )
    assert any(
        completed in relative.as_posix()
        for family_members in members.values()
        for relative in family_members
    )


def test_pack_default_validation_skips_quarantined_assets(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "pack-validation", "ABO", "ABO-00000"
    )
    completed = "a" * 64
    quarantined = "b" * 64
    write_instances(context, (completed, quarantined))
    validated = []

    def pack_publisher(*args, **kwargs):
        return tuple(
            SimpleNamespace(validated_at="now") for _ in PACK_FAMILIES
        )

    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        project_accounting=FakeAccounting(),
        asset_output_validator=lambda active_context, asset_sha: validated.append(
            asset_sha
        ),
        pack_publisher=pack_publisher,
        published_batch_verifier=lambda context: None,
        tool_commit="test-commit",
    )
    write_quality_checkpoint(
        services,
        context,
        {completed: "completed", quarantined: "failure"},
    )

    services.build_packs(context)

    assert validated == [completed]


def test_published_pack_rejects_stale_frozen_sha_identity(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "stale-pack", "ABO", "ABO-00000"
    )
    old_sha = "a" * 64
    new_sha = "b" * 64
    write_instances(context, (old_sha,))
    publish_dummy_batch(isolated_config, context, old_sha)
    write_instances(context, (new_sha,))
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        project_accounting=FakeAccounting(),
        tool_commit="test-commit",
    )
    write_quality_checkpoint(services, context, {new_sha: "completed"})

    with pytest.raises(ValidationError, match="frozen SHA"):
        services._verify_published_batch(context)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tool_commit", "wrong-commit"),
        ("completed_count", 0),
        ("quarantined_count", 1),
    ],
)
def test_raw_archive_audit_binds_tool_and_quality_counts(
    isolated_config, field, value
):
    context = configured_context(isolated_config)
    contents = b"raw identity"
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
        project_accounting=FakeAccounting(),
        published_batch_verifier=lambda context: None,
        tool_commit="test-commit",
    )
    write_quality_checkpoint(services, context, {asset_sha: "completed"})
    services.stage_raw(context)
    services.archive_raw(context)
    _, manifest_path = services._raw_archive_paths(context)
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValidationError, match="raw archive identity"):
        services._verify_raw_archive(context)


def test_archive_records_data3_publication_and_data2_deletion_deltas(
    isolated_config,
):
    context = configured_context(isolated_config)
    contents = b"accounted raw"
    asset_sha = sha256(contents).hexdigest()
    relative = "raw/models/item.glb"
    source = context.source_root / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(contents)
    write_instances(context, (asset_sha,))
    write_raw_metadata(
        context, ({"sha256": asset_sha, "local_path": relative},)
    )
    accounting = FakeAccounting()
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        reference_counter=FakeReferenceCounter(0),
        project_accounting=accounting,
        published_batch_verifier=lambda context: None,
        tool_commit="test-commit",
    )
    write_quality_checkpoint(services, context, {asset_sha: "completed"})
    services.stage_raw(context)

    services.archive_raw(context)

    assert any(
        isolated_config.paths.data3_root in path.parents and delta > 0
        for path, delta in accounting.deltas
    )
    assert (source, -len(contents)) in accounting.deltas


def test_resume_reconciles_accounting_at_batch_and_shard_boundaries(
    isolated_config,
):
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
    runner = FakeShardRunner()
    accounting = FakeAccounting()
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        registry_store=registry,
        pilot_reader=FakePilotReader(100),
        disk_usage=lambda path: SimpleNamespace(
            total=1000 * gib, free=1000 * gib
        ),
        runner=runner,
        project_accounting=accounting,
        batch_auditor=lambda context: None,
        published_batch_verifier=lambda context: None,
    )
    services.plan("production", "ABO", "ABO-00000", freeze=True)

    services.resume("production", "ABO", "ABO-00000")

    assert accounting.reconciliations == 2


def test_stop_persistence_failures_preserve_primary_and_write_local_fallback(
    isolated_config, shard_context, tmp_path
):
    command = CommandSpec("worker", ("worker",))
    guard = FakeResourceGuard()
    guard.stop_next("disk hard primary")
    fallback = tmp_path / "local-fallback/report.json"
    runner = PipelineRunner(
        isolated_config,
        guard,
        {command.name: lambda: True},
        {},
        command_builder=lambda context, config: (command,),
        report_writer=lambda report: (_ for _ in ()).throw(
            OSError("data2 report unavailable")
        ),
        checkpoint_path=lambda context: tmp_path / "checkpoint.json",
        fallback_report_path=lambda context, name: fallback,
    )
    runner.save_checkpoint = lambda path, checkpoint: (_ for _ in ()).throw(
        CheckpointError("checkpoint unavailable")
    )

    with pytest.raises(PipelineStopped) as caught:
        runner.run_shard(shard_context)

    assert caught.value.report.reason == "disk hard primary"
    assert caught.value.report.category == EscalationCategory.RESOURCE
    assert any(
        "checkpoint unavailable" in error
        for error in caught.value.report.persistence_errors
    )
    assert any(
        "data2 report unavailable" in error
        for error in caught.value.report.persistence_errors
    )
    fallback_report = json.loads(fallback.read_text())
    assert fallback_report["reason"] == "disk hard primary"
    assert fallback_report["persistence_errors"]


def test_fallback_failure_still_raises_pipeline_stopped(
    isolated_config, shard_context, tmp_path
):
    command = CommandSpec("worker", ("worker",))
    guard = FakeResourceGuard()
    guard.stop_next("resource primary")
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    runner = PipelineRunner(
        isolated_config,
        guard,
        {command.name: lambda: True},
        {},
        command_builder=lambda context, config: (command,),
        report_writer=lambda report: (_ for _ in ()).throw(
            OSError("primary report failed")
        ),
        checkpoint_path=lambda context: tmp_path / "checkpoint.json",
        fallback_report_path=lambda context, name: linked / "fallback.json",
    )

    with pytest.raises(PipelineStopped) as caught:
        runner.run_shard(shard_context)

    assert caught.value.report.reason == "resource primary"
    assert any(
        "fallback" in error.lower()
        for error in caught.value.report.persistence_errors
    )
    assert not (outside / "fallback.json").exists()


def test_success_checkpoint_failure_escalates_instead_of_escaping(
    isolated_config, shard_context, tmp_path
):
    command = CommandSpec("worker", ("worker",))
    reports = []

    class CompletionFailureRunner(RecordingRunner):
        def save_checkpoint(self, path, checkpoint):
            if command.name in checkpoint.completed_commands:
                raise CheckpointError("completion checkpoint failed")
            super().save_checkpoint(path, checkpoint)

    runner = CompletionFailureRunner(isolated_config, (command,), reports=reports)
    runner.fallback_report_path = (
        lambda context, name: tmp_path / "fallback.json"
    )

    with pytest.raises(PipelineStopped) as caught:
        runner.run_shard(shard_context)

    assert caught.value.report.category == EscalationCategory.INFRASTRUCTURE
    assert "completion checkpoint failed" in caught.value.report.reason
    assert reports == [caught.value.report]


def test_accounting_reconciliation_failure_checkpoints_and_reports(
    isolated_config,
):
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
    accounting = FakeAccounting(
        failure=ResourceAccountingError("registry stale")
    )
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        registry_store=registry,
        pilot_reader=FakePilotReader(100),
        disk_usage=lambda path: SimpleNamespace(
            total=1000 * gib, free=1000 * gib
        ),
        project_accounting=accounting,
        batch_auditor=lambda context: None,
        published_batch_verifier=lambda context: None,
    )
    services.runner.command_builder = lambda context, config: ()
    services.plan("production", "ABO", "ABO-00000", freeze=True)

    with pytest.raises(PipelineStopped) as caught:
        services.resume("production", "ABO", "ABO-00000")

    assert caught.value.report.category == EscalationCategory.INFRASTRUCTURE
    assert "registry stale" in caught.value.report.reason
    report_path = (
        isolated_config.paths.data2_root
        / "control/reports/escalations/ABO/ABO-00000/accounting_batch.json"
    )
    assert report_path.is_file()


def test_cleanup_rejects_symlinked_ancestor_without_deleting_outside(
    isolated_config, tmp_path
):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    context = ShardContext.for_test(
        linked / "batch", "ABO", "ABO-00000"
    )
    payload = context.output_root / "payload"
    payload.parent.mkdir(parents=True)
    payload.write_text("keep")
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        published_batch_verifier=lambda context: None,
        raw_archive_verifier=lambda context: None,
    )

    with pytest.raises(ValidationError, match="clean"):
        services.cleanup_local(context)

    assert payload.read_text() == "keep"


def test_invalid_monitor_decision_is_immediate_infrastructure_stop(
    isolated_config, shard_context
):
    command = CommandSpec("worker", ("worker",))
    supervisor = FakeSupervisor(900, [None] * 10)
    runner, _ = supervisor_runner(
        isolated_config, FakeResourceGuard((object(),)), [supervisor]
    )
    runner.command_builder = lambda context, config: (command,)
    runner.validators = {command.name: lambda: True}
    runner.checkpoint_path = lambda context: context.work_root / "checkpoint.json"

    with pytest.raises(PipelineStopped) as caught:
        runner.run_shard(shard_context)

    assert caught.value.report.category == EscalationCategory.INFRASTRUCTURE
    assert "monitor" in caught.value.report.reason
    assert len(supervisor.wait_timeouts) == 1


def test_pack_publication_records_data2_delta(isolated_config, tmp_path):
    context = ShardContext.for_test(
        tmp_path / "pack-accounting", "ABO", "ABO-00000"
    )
    asset_sha = "a" * 64
    write_instances(context, (asset_sha,))
    accounting = FakeAccounting()
    services = None

    def publisher(data2_root, source_root, members, shard_id, **kwargs):
        published = services._published_paths(context)[0]
        published.parent.mkdir(parents=True, exist_ok=True)
        published.write_bytes(b"published pack")
        return tuple(SimpleNamespace(validated_at="now") for _ in PACK_FAMILIES)

    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        project_accounting=accounting,
        output_validator=lambda context: None,
        pack_publisher=publisher,
        published_batch_verifier=lambda context: None,
        tool_commit="test-commit",
    )
    write_quality_checkpoint(services, context, {asset_sha: "completed"})

    services.build_packs(context)

    assert any(
        isolated_config.paths.data2_root in path.parents and delta > 0
        for path, delta in accounting.deltas
    )


def test_published_pack_rejects_stale_expected_members(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "stale-members", "ABO", "ABO-00000"
    )
    asset_sha = "a" * 64
    write_instances(context, (asset_sha,))
    publish_dummy_batch(isolated_config, context, asset_sha)
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        project_accounting=FakeAccounting(),
        tool_commit="test-commit",
    )
    write_quality_checkpoint(services, context, {asset_sha: "completed"})

    with pytest.raises(ValidationError, match="member identity"):
        services._verify_published_batch(context)


def test_published_pack_rejects_stale_quality_counts(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "stale-counts", "ABO", "ABO-00000"
    )
    asset_sha = "a" * 64
    write_instances(context, (asset_sha,))
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        project_accounting=FakeAccounting(),
        tool_commit="test-commit",
    )
    write_quality_checkpoint(services, context, {asset_sha: "completed"})
    members = services._pack_members(context)
    for family_members in members.values():
        for relative in family_members:
            path = context.output_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"member")
    publish_pack(
        isolated_config.paths.data2_root,
        context.output_root,
        members,
        context.shard_id,
        source=context.source,
        batch_id=context.batch_id,
        config_hash=isolated_config.config_hash(),
        tool_commit="test-commit",
        asset_sha256s=(asset_sha,),
        completed_count=1,
        quarantined_count=0,
    )
    prepared = isolated_config.paths.data2_root / "prepared"
    index_path = prepared / "index/ABO/ABO-00000.json"
    index = json.loads(index_path.read_text())
    entry = index["batches"]["batch000"]["common"]
    manifest_path = prepared / entry["manifest"]
    manifest = json.loads(manifest_path.read_text())
    manifest["completed_count"] = 0
    manifest_path.write_text(json.dumps(manifest))
    entry["manifest_sha256"] = sha256(manifest_path.read_bytes()).hexdigest()
    index_path.write_text(json.dumps(index))

    with pytest.raises(ValidationError, match="identity mismatch"):
        services._verify_published_batch(context)


def test_cleanup_removes_processed_roots_before_staged_raw(
    isolated_config, tmp_path, monkeypatch
):
    context = ShardContext.for_test(
        tmp_path / "cleanup-order", "ABO", "ABO-00000"
    )
    removed = []
    monkeypatch.setattr(
        orchestrator_module,
        "_remove_tree_nofollow",
        lambda path: removed.append(Path(path)),
    )
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        published_batch_verifier=lambda context: None,
        raw_archive_verifier=lambda context: None,
    )

    services.cleanup_local(context)

    assert removed == [
        context.output_root,
        context.work_root,
        context.download_root,
    ]


def test_telemetry_failure_during_stop_is_exposed_not_suppressing_primary(
    isolated_config, shard_context, tmp_path
):
    command = CommandSpec("worker", ("worker",))
    guard = FakeResourceGuard()
    guard.stop_next("resource primary")
    guard.last_five_minutes = lambda: (_ for _ in ()).throw(
        RuntimeError("telemetry unavailable")
    )
    reports = []
    runner = PipelineRunner(
        isolated_config,
        guard,
        {command.name: lambda: True},
        {},
        command_builder=lambda context, config: (command,),
        checkpoint_path=lambda context: tmp_path / "checkpoint.json",
        report_writer=reports.append,
    )

    with pytest.raises(PipelineStopped) as caught:
        runner.run_shard(shard_context)

    assert caught.value.report.reason == "resource primary"
    assert any(
        "telemetry unavailable" in error
        for error in caught.value.report.persistence_errors
    )
    assert reports == [caught.value.report]


def test_service_report_writer_rejects_symlinked_data2_ancestor(
    isolated_config, shard_context, tmp_path
):
    outside = tmp_path / "outside-report"
    outside.mkdir()
    isolated_config.paths.data2_root.symlink_to(
        outside, target_is_directory=True
    )
    guard = FakeResourceGuard()
    guard.stop_next("primary")
    command = CommandSpec("worker", ("worker",))
    producer = PipelineRunner(
        isolated_config,
        guard,
        {command.name: lambda: True},
        {},
        command_builder=lambda context, config: (command,),
        checkpoint_path=lambda context: tmp_path / "checkpoint.json",
    )
    with pytest.raises(PipelineStopped) as caught:
        producer.run_shard(shard_context)
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )

    with pytest.raises(OSError):
        services._write_escalation(caught.value.report)

    assert not list(outside.rglob("*.json"))


def test_published_validator_io_failure_is_not_downgraded(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "published-io", "ABO", "ABO-00000"
    )
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        published_batch_verifier=lambda context: (_ for _ in ()).throw(
            OSError("data2 I/O failed")
        ),
    )

    with pytest.raises(OSError, match="data2 I/O failed"):
        services._published_is_valid(context)


def test_raw_metadata_io_failure_is_not_downgraded(
    isolated_config, monkeypatch, tmp_path
):
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_read_regular_bytes_nofollow",
        lambda path: (_ for _ in ()).throw(OSError("metadata EIO")),
    )

    with pytest.raises(OSError, match="metadata EIO"):
        services._read_raw_records(
            tmp_path / "metadata.csv", ("a" * 64,)
        )


def test_accounting_delta_failure_reconciles_before_stopping(
    isolated_config,
):
    class RecoverableAccounting:
        def __init__(self):
            self.reconciliations = 0

        def record_registry_delta(self, path, delta):
            raise ResourceAccountingError("delta write failed")

        def reconcile_at_shard_boundary(self):
            self.reconciliations += 1
            return (1, 2)

    accounting = RecoverableAccounting()
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        project_accounting=accounting,
    )

    with pytest.raises(InfrastructureError, match="delta write failed"):
        services._record_delta(
            isolated_config.paths.data2_root / "prepared/pack.tar", 10
        )

    assert accounting.reconciliations == 1


def test_accounting_boundaries_do_not_hide_programmer_defects(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "accounting-defect", "ABO", "ABO-00000"
    )

    class DefectiveAccounting:
        def record_registry_delta(self, path, delta):
            raise AssertionError("delta programmer defect")

        def reconcile_at_shard_boundary(self):
            raise AssertionError("reconcile programmer defect")

    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        project_accounting=DefectiveAccounting(),
    )

    with pytest.raises(AssertionError, match="delta programmer defect"):
        services._record_delta(tmp_path / "pack.tar", 1)
    with pytest.raises(AssertionError, match="reconcile programmer defect"):
        services._reconcile_accounting(context, "batch")


def test_published_index_io_failure_propagates(
    isolated_config, monkeypatch, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "published-index-io", "ABO", "ABO-00000"
    )
    asset_sha = "a" * 64
    write_instances(context, (asset_sha,))
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        project_accounting=FakeAccounting(),
        tool_commit="test-commit",
    )
    write_quality_checkpoint(services, context, {asset_sha: "completed"})
    index_path = (
        isolated_config.paths.data2_root
        / "prepared/index/ABO/ABO-00000.json"
    )
    real_read = orchestrator_module._read_regular_bytes_nofollow

    def fail_index(path, *args, **kwargs):
        if Path(path) == index_path:
            raise OSError("published index EIO")
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(
        orchestrator_module, "_read_regular_bytes_nofollow", fail_index
    )

    with pytest.raises(OSError, match="published index EIO"):
        services._verify_published_batch(context)


def test_raw_archive_manifest_io_failure_propagates(
    isolated_config, monkeypatch
):
    context = configured_context(isolated_config)
    asset_sha = "a" * 64
    write_instances(context, (asset_sha,))
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        project_accounting=FakeAccounting(),
        tool_commit="test-commit",
    )
    write_quality_checkpoint(services, context, {asset_sha: "completed"})
    _, manifest_path = services._raw_archive_paths(context)
    real_read = orchestrator_module._read_regular_bytes_nofollow

    def fail_manifest(path, *args, **kwargs):
        if Path(path) == manifest_path:
            raise OSError("raw manifest EIO")
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(orchestrator_module, "verify_pack", lambda *args: None)
    monkeypatch.setattr(
        orchestrator_module,
        "_read_regular_bytes_nofollow",
        fail_manifest,
    )

    with pytest.raises(OSError, match="raw manifest EIO"):
        services._verify_raw_archive(context)


def test_logical_index_io_failure_propagates(
    isolated_config, monkeypatch
):
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )
    index_path = (
        isolated_config.paths.data2_root
        / "prepared/index/ABO/ABO-00000.json"
    )
    real_read = orchestrator_module._read_regular_bytes_nofollow

    def fail_index(path, *args, **kwargs):
        if Path(path) == index_path:
            raise OSError("logical index EIO")
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(
        orchestrator_module, "_read_regular_bytes_nofollow", fail_index
    )

    with pytest.raises(OSError, match="logical index EIO"):
        services._verify_logical_index(
            "ABO", "ABO-00000", (("a" * 64,),)
        )


def test_pidfd_launch_failure_closes_gate_before_bounded_reap(monkeypatch):
    class GatedProcess:
        pid = 777

        def __init__(self, gate_fd):
            self.gate_fd = os.dup(gate_fd)
            os.set_blocking(self.gate_fd, False)
            self.wait_timeouts = []

        def wait(self, timeout):
            self.wait_timeouts.append(timeout)
            try:
                value = os.read(self.gate_fd, 1)
            except BlockingIOError as error:
                raise subprocess.TimeoutExpired(("supervisor",), timeout) from error
            assert value == b""
            os.close(self.gate_fd)
            self.gate_fd = -1
            return 125

    created = []

    def process_factory(argv, **kwargs):
        process = GatedProcess(kwargs["pass_fds"][0])
        created.append(process)
        return process

    monkeypatch.setattr(
        orchestrator_module,
        "_linux_syscall",
        lambda *args: (_ for _ in ()).throw(OSError("pidfd unavailable")),
    )

    with pytest.raises(OSError, match="pidfd unavailable"):
        orchestrator_module._LinuxProcessSupervisor.launch(
            process_factory, ("worker",), {}
        )

    assert created[0].wait_timeouts == [1]
    assert created[0].gate_fd < 0


def test_pidfd_launch_cleanup_preserves_original_error_and_finally_kills(
    monkeypatch,
):
    class BrokenWaitProcess:
        pid = 778

        def __init__(self):
            self.wait_timeouts = []

        def wait(self, timeout):
            self.wait_timeouts.append(timeout)
            if len(self.wait_timeouts) == 1:
                raise RuntimeError("initial reap failed")
            return -signal.SIGKILL

    process = BrokenWaitProcess()
    sent_signals = []
    pidfd_template = os.open("/dev/null", os.O_RDONLY)

    def fake_syscall(number, *arguments):
        if number == orchestrator_module._LinuxProcessSupervisor._PIDFD_OPEN:
            return os.dup(pidfd_template)
        sent_signals.append(arguments[1])
        return 0

    monkeypatch.setattr(orchestrator_module, "_linux_syscall", fake_syscall)
    monkeypatch.setattr(
        orchestrator_module,
        "_wait_for_supervisor_ready",
        lambda *args: (_ for _ in ()).throw(TimeoutError("READY failed")),
    )

    try:
        with pytest.raises(TimeoutError, match="READY failed") as caught:
            orchestrator_module._LinuxProcessSupervisor.launch(
                lambda *args, **kwargs: process, ("worker",), {}
            )
    finally:
        os.close(pidfd_template)

    assert process.wait_timeouts == [1, 1]
    assert sent_signals == [signal.SIGKILL]
    assert any(
        "initial reap failed" in note
        for note in getattr(caught.value, "__notes__", ())
    )


def test_post_release_launch_failure_uses_supervisor_group_kill(monkeypatch):
    class ReleasedProcess:
        pid = 779

        def __init__(self, release_fd):
            self.release_fd = os.dup(release_fd)
            self.release = None
            self.killed = False
            self.wait_timeouts = []

        def wait(self, timeout):
            self.wait_timeouts.append(timeout)
            if self.release is None:
                self.release = os.read(self.release_fd, 1)
            if not self.killed:
                raise subprocess.TimeoutExpired(("supervisor",), timeout)
            os.close(self.release_fd)
            self.release_fd = -1
            return -signal.SIGKILL

    class FailAfterRelease(orchestrator_module._LinuxProcessSupervisor):
        def __init__(self, process, pidfd):
            raise KeyboardInterrupt("post-release failure")

    created = []
    sent_signals = []
    pidfd_template = os.open("/dev/null", os.O_RDONLY)

    def process_factory(argv, **kwargs):
        process = ReleasedProcess(kwargs["pass_fds"][0])
        created.append(process)
        return process

    def fake_syscall(number, *arguments):
        if number == FailAfterRelease._PIDFD_OPEN:
            return os.dup(pidfd_template)
        sent_signals.append(arguments[1])
        if arguments[1] == signal.SIGUSR2:
            created[0].killed = True
        return 0

    monkeypatch.setattr(orchestrator_module, "_linux_syscall", fake_syscall)
    monkeypatch.setattr(
        orchestrator_module, "_wait_for_supervisor_ready", lambda *args: None
    )

    try:
        with pytest.raises(KeyboardInterrupt, match="post-release failure"):
            FailAfterRelease.launch(process_factory, ("worker",), {})
    finally:
        os.close(pidfd_template)
        if created and created[0].release_fd >= 0:
            os.close(created[0].release_fd)

    assert created[0].release == b"1"
    assert created[0].wait_timeouts == [1, 1]
    assert created[0].release_fd < 0
    assert sent_signals == [signal.SIGUSR2]


def test_release_write_failure_uses_supervisor_group_kill(monkeypatch):
    class ReadyProcess:
        pid = 780

        def __init__(self, release_fd):
            self.release_fd = os.dup(release_fd)
            self.release = None
            self.killed = False

        def wait(self, timeout):
            if self.release is None:
                self.release = os.read(self.release_fd, 1)
            if not self.killed:
                raise subprocess.TimeoutExpired(("supervisor",), timeout)
            os.close(self.release_fd)
            self.release_fd = -1
            return -signal.SIGKILL

    created = []
    sent_signals = []
    pidfd_template = os.open("/dev/null", os.O_RDONLY)
    real_write = os.write

    def process_factory(argv, **kwargs):
        process = ReadyProcess(kwargs["pass_fds"][0])
        created.append(process)
        return process

    def fake_syscall(number, *arguments):
        if number == orchestrator_module._LinuxProcessSupervisor._PIDFD_OPEN:
            return os.dup(pidfd_template)
        sent_signals.append(arguments[1])
        if arguments[1] == signal.SIGUSR2:
            created[0].killed = True
        return 0

    def fail_release_write(file_descriptor, data):
        if data == b"1":
            raise OSError(errno.EIO, "release EIO")
        return real_write(file_descriptor, data)

    monkeypatch.setattr(orchestrator_module, "_linux_syscall", fake_syscall)
    monkeypatch.setattr(
        orchestrator_module, "_wait_for_supervisor_ready", lambda *args: None
    )
    monkeypatch.setattr(orchestrator_module.os, "write", fail_release_write)

    try:
        with pytest.raises(OSError, match="release EIO"):
            orchestrator_module._LinuxProcessSupervisor.launch(
                process_factory, ("worker",), {}
            )
    finally:
        os.close(pidfd_template)
        if created and created[0].release_fd >= 0:
            os.close(created[0].release_fd)

    assert created[0].release == b""
    assert sent_signals == [signal.SIGUSR2]


def test_default_dump_validator_propagates_source_io_failure(
    isolated_config, monkeypatch, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "dump-validator-io", "ABO", "ABO-00000"
    )
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_open_directory_nofollow",
        lambda path: (_ for _ in ()).throw(OSError("dump source EIO")),
    )

    with pytest.raises(OSError, match="dump source EIO"):
        services._validate_dump_output(
            context, "geometry_dumps", "a" * 64
        )


def test_default_voxel_validator_propagates_reader_io_failure(
    isolated_config, monkeypatch, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "voxel-validator-io", "ABO", "ABO-00000"
    )
    relative = Path("voxels") / ("a" * 64) / "view00.vxz"
    source = context.work_root / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(b"voxel")
    fake_voxel = SimpleNamespace(
        io=SimpleNamespace(
            read_vxz_info=lambda path: (_ for _ in ()).throw(
                OSError("voxel reader EIO")
            )
        )
    )
    monkeypatch.setitem(sys.modules, "o_voxel", fake_voxel)

    with pytest.raises(OSError, match="voxel reader EIO"):
        PipelineServices._validate_voxel_output(
            context, "voxels", "a" * 64, 0
        )


def test_supervisor_program_uses_ready_handshake_and_kernel_group_control():
    program = orchestrator_module._SUPERVISOR_PROGRAM

    assert "/proc" not in program
    assert "os.killpg(leader" in program
    assert program.index("signal.signal(signal.SIGUSR1") < program.index(
        "os.write(ready_fd"
    )
    assert program.index("os.write(ready_fd") < program.index(
        "os.read(release_fd"
    )
    assert program.index("os.read(release_fd") < program.index(
        "worker = subprocess.Popen"
    )


def test_supervisor_resume_handler_ignores_self_delivery_while_resuming_group():
    program = ast.parse(orchestrator_module._SUPERVISOR_PROGRAM)
    resume_node = next(
        node
        for node in program.body
        if isinstance(node, ast.FunctionDef) and node.name == "resume_group"
    )
    transitions = []

    class FakeSignal:
        SIGCONT = signal.SIGCONT
        SIG_IGN = object()
        current = None

        @classmethod
        def signal(cls, sent_signal, handler):
            assert sent_signal == cls.SIGCONT
            previous = cls.current
            cls.current = handler
            transitions.append(handler)
            return previous

    def killpg(leader, sent_signal):
        assert leader == 91
        assert sent_signal == signal.SIGCONT
        assert FakeSignal.current is FakeSignal.SIG_IGN

    namespace = {
        "signal": FakeSignal,
        "os": SimpleNamespace(killpg=killpg),
        "leader": 91,
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[resume_node], type_ignores=[])
            ),
            "<resume-handler>",
            "exec",
        ),
        namespace,
    )
    handler = namespace["resume_group"]
    FakeSignal.current = handler

    handler()

    assert transitions == [FakeSignal.SIG_IGN, handler]
    assert FakeSignal.current is handler


def test_cleanup_finally_kills_reaps_and_closes_stubborn_supervisor(
    isolated_config,
):
    stop = ResourceDecision(ResourceAction.STOP, ("disk hard",))
    kill_count = 0

    def exit_only_after_final_kill(supervisor, action):
        nonlocal kill_count
        if action == "kill":
            kill_count += 1
            if kill_count == 2:
                supervisor.returncode = -signal.SIGKILL

    supervisor = FakeSupervisor(
        1001, [None] * 20, control=exit_only_after_final_kill
    )
    supervisor.closed = False
    supervisor.close = lambda: setattr(supervisor, "closed", True)
    runner, _ = supervisor_runner(
        isolated_config, FakeResourceGuard((stop,)), [supervisor]
    )

    with pytest.raises(ResourceLimitExceeded, match="disk hard"):
        runner.execute(CommandSpec("worker", ("worker",)), "shard")

    assert supervisor.controls == ["terminate", "kill", "kill"]
    assert supervisor.wait_timeouts == [3, 3]
    assert supervisor.closed is True


def test_cleanup_closes_supervisor_when_poll_fails(isolated_config):
    class BrokenPollSupervisor:
        pid = 1002

        def __init__(self):
            self.closed = False

        def poll(self):
            raise RuntimeError("poll failed")

        def close(self):
            self.closed = True

    supervisor = BrokenPollSupervisor()
    runner, _ = supervisor_runner(
        isolated_config, FakeResourceGuard(), [supervisor]
    )

    with pytest.raises(RuntimeError, match="poll failed"):
        runner._terminate_and_reap([supervisor], set())

    assert supervisor.closed is True


def test_final_reap_failure_retains_supervisor_until_kill_and_close(
    isolated_config,
):
    class LateReapSupervisor:
        pid = 1003

        def __init__(self):
            self.first_poll = True
            self.returncode = None
            self.controls = []
            self.wait_timeouts = []
            self.closed = False

        def poll(self):
            if self.first_poll:
                self.first_poll = False
                return 0
            return self.returncode

        def send_control(self, action):
            self.controls.append(action)
            if action == "kill":
                self.returncode = -signal.SIGKILL

        def wait(self, timeout):
            self.wait_timeouts.append(timeout)
            if self.returncode is None:
                raise subprocess.TimeoutExpired(("supervisor",), timeout)
            return self.returncode

        def close(self):
            self.closed = True

    supervisor = LateReapSupervisor()
    runner, _ = supervisor_runner(
        isolated_config, FakeResourceGuard(), [supervisor]
    )

    with pytest.raises(ProcessGroupSafetyError, match="did not exit"):
        runner.execute(CommandSpec("worker", ("worker",)), "shard")

    assert supervisor.controls == ["terminate", "kill"]
    assert supervisor.wait_timeouts == [3, 3]
    assert supervisor.closed is True


def test_production_stage_quality_is_durable_and_filters_downstream_instances(
    isolated_config, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "stage-quality", "ABO", "ABO-00000"
    )
    completed, missing, invalid = tuple(
        f"{index:064x}" for index in range(3)
    )
    write_instances(context, (completed, missing, invalid))
    commands = (
        CommandSpec(
            "dump_mesh",
            ("worker", "--instances", str(context.instances)),
        ),
        CommandSpec(
            "dump_pbr",
            ("worker", "--instances", str(context.instances)),
        ),
        CommandSpec("validate_outputs", ("internal:validate_outputs",)),
    )
    downstream_instances = []
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        asset_output_validator=lambda active_context, asset_sha: None,
    )
    services.runner.command_builder = lambda active_context, config: commands
    internal_execute = services.runner.execute

    def execute(command, shard_id):
        if command.name == "dump_mesh":
            root = context.work_root / "mesh_dumps"
            root.mkdir(parents=True, exist_ok=True)
            with (root / f"{completed}.pickle").open("wb") as stream:
                pickle.dump({"objects": []}, stream)
            (root / f"{invalid}.pickle").write_bytes(b"corrupt")
        elif command.name == "dump_pbr":
            index = command.argv.index("--instances") + 1
            downstream_instances.extend(
                Path(command.argv[index]).read_text().splitlines()
            )
            root = context.work_root / "pbr_dumps"
            root.mkdir(parents=True, exist_ok=True)
            with (root / f"{completed}.pickle").open("wb") as stream:
                pickle.dump({"objects": [], "materials": []}, stream)
        else:
            return internal_execute(command, shard_id)

    services.runner.execute = execute

    services.runner.run_shard(context)

    checkpoint = services.runner.load_checkpoint(
        services._checkpoint_path(context), context.shard_id
    )
    assert checkpoint.quality_outcomes == {
        missing: "failure",
        invalid: "schema_failure",
        completed: "completed",
    }
    assert downstream_instances == [completed]
    assert checkpoint.attempts == {
        "dump_mesh": 1,
        "dump_pbr": 1,
        "validate_outputs": 1,
    }


def test_real_asset_stats_leaf_part_is_accepted_by_production_validator(
    isolated_config, monkeypatch, tmp_path
):
    context = ShardContext.for_test(
        tmp_path / "asset-stats-leaf", "ABO", "ABO-00000"
    )
    asset_sha = "a" * 64
    write_instances(context, (asset_sha,))
    context.metadata_root.mkdir(parents=True)
    pd.DataFrame([{"sha256": asset_sha}]).to_csv(
        context.metadata_root / "metadata.csv", index=False
    )
    mesh_records = context.work_root / "mesh_dumps/new_records"
    mesh_records.mkdir(parents=True)
    pd.DataFrame([{"sha256": asset_sha, "mesh_dumped": True}]).to_csv(
        mesh_records / "part_0.csv", index=False
    )
    pbr_records = context.work_root / "pbr_dumps/new_records"
    pbr_records.mkdir(parents=True)
    pd.DataFrame([{"sha256": asset_sha, "pbr_dumped": True}]).to_csv(
        pbr_records / "part_0.csv", index=False
    )
    dump = context.work_root / "pbr_dumps" / f"{asset_sha}.pickle"
    dump.parent.mkdir(parents=True, exist_ok=True)
    with dump.open("wb") as stream:
        pickle.dump(
            {
                "objects": [
                    {
                        "vertices": np.zeros((3, 3)),
                        "faces": np.zeros((1, 3)),
                    }
                ],
                "materials": [
                    {
                        "baseColorTexture": object(),
                        "metallicTexture": None,
                        "roughnessTexture": None,
                        "alphaTexture": None,
                    }
                ],
            },
            stream,
        )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "asset_stats.py",
            "--root",
            str(context.metadata_root),
            "--instances",
            str(context.instances),
            "--mesh_dump_root",
            str(context.work_root),
            "--pbr_dump_root",
            str(context.work_root),
            "--max_workers",
            "1",
        ],
    )

    runpy.run_module("data_toolkit.asset_stats", run_name="__main__")

    part = context.metadata_root / "asset_stats/new_records/part_0.csv"
    assert part.is_file()
    assert not (context.metadata_root / "asset_stats/metadata.csv").exists()
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )
    services._validate_asset_stats(context)


def test_escalation_report_includes_terminal_asset_counts(
    isolated_config, shard_context
):
    reports = []
    runner = PipelineRunner(
        isolated_config,
        FakeResourceGuard(),
        {},
        {},
        report_writer=reports.append,
    )
    checkpoint = PipelineCheckpoint(
        shard_context.shard_id,
        completed_commands=["dump_mesh"],
        quality_outcomes={
            "a" * 64: "completed",
            "b" * 64: "failure",
            "c" * 64: "schema_failure",
        },
    )
    runner.quality_gate.restore(
        checkpoint.quality_outcomes, tuple(checkpoint.quality_outcomes)
    )

    with pytest.raises(PipelineStopped) as caught:
        runner.stop(
            shard_context,
            "worker",
            "stop",
            checkpoint,
            category=EscalationCategory.DATA_QUALITY,
            exit_code=4,
        )

    assert caught.value.report.completed_counts == {
        "commands": 1,
        "outcomes": 3,
        "completed_assets": 1,
        "quarantined_assets": 2,
        "failure_assets": 1,
        "schema_failure_assets": 1,
    }


def test_staging_publication_never_replaces_existing_destination(tmp_path):
    root = tmp_path / "staging"
    relative = Path("raw/item.glb")
    destination = root / relative
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing")
    replacement = b"replacement"

    with pytest.raises(ValidationError, match="staging destination"):
        orchestrator_module._atomic_stage_stream(
            io.BytesIO(replacement),
            root,
            relative,
            sha256(replacement).hexdigest(),
        )

    assert destination.read_bytes() == b"existing"


def test_raw_delete_tombstone_detects_identity_swap_without_deleting_replacement(
    tmp_path,
):
    root = tmp_path / "raw-delete"
    relative = Path("models/item.glb")
    source = root / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(b"original")
    replacement = b"replacement"
    rename_calls = 0

    def swap_then_rename(
        source_name,
        destination_name,
        source_directory_fd,
        destination_directory_fd=None,
    ):
        nonlocal rename_calls
        rename_calls += 1
        if destination_directory_fd is None:
            destination_directory_fd = source_directory_fd
        if rename_calls > 1:
            os.rename(
                source_name,
                destination_name,
                src_dir_fd=source_directory_fd,
                dst_dir_fd=destination_directory_fd,
            )
            return
        os.rename(
            source_name,
            "original.saved",
            src_dir_fd=source_directory_fd,
            dst_dir_fd=source_directory_fd,
        )
        replacement_fd = os.open(
            source_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=source_directory_fd,
        )
        try:
            os.write(replacement_fd, replacement)
        finally:
            os.close(replacement_fd)
        os.rename(
            source_name,
            destination_name,
            src_dir_fd=source_directory_fd,
            dst_dir_fd=destination_directory_fd,
        )

    with pytest.raises(ValidationError, match="identity changed"):
        orchestrator_module._unlink_regular_beneath(
            root, relative, rename_noreplace=swap_then_rename
        )

    survivors = {
        path.name: path.read_bytes() for path in source.parent.iterdir()
    }
    assert b"original" in survivors.values()
    assert replacement in survivors.values()


def test_raw_delete_post_validation_swap_cannot_delete_replacement(
    tmp_path, monkeypatch
):
    root = tmp_path / "raw-delete-post-validation"
    relative = Path("models/item.glb")
    source = root / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(b"original")
    replacement = b"replacement"
    real_same_inode = orchestrator_module._same_inode

    def swap_public_tombstone_after_validation(first, second):
        matches = real_same_inode(first, second)
        public_tombstones = tuple(source.parent.glob(".*.delete"))
        if public_tombstones:
            public_tombstone = public_tombstones[0]
            public_tombstone.rename(source.parent / "validated-original.saved")
            public_tombstone.write_bytes(replacement)
        else:
            source.write_bytes(replacement)
        return matches

    monkeypatch.setattr(
        orchestrator_module, "_same_inode", swap_public_tombstone_after_validation
    )

    removed = orchestrator_module._unlink_regular_beneath(root, relative)

    assert removed == len(b"original")
    assert source.read_bytes() == replacement


def test_raw_archive_audit_rejects_unrelated_valid_member_mapping(
    isolated_config,
):
    context = configured_context(isolated_config)
    contents = b"raw mapping"
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
        project_accounting=FakeAccounting(),
        published_batch_verifier=lambda active_context: None,
        tool_commit="test-commit",
    )
    write_quality_checkpoint(services, context, {asset_sha: "completed"})
    services.stage_raw(context)
    services.archive_raw(context)
    write_raw_metadata(
        context,
        ({"sha256": asset_sha, "local_path": "raw/models/other.glb"},),
    )

    with pytest.raises(ValidationError, match="raw archive.*mapping"):
        services._verify_raw_archive(context)


@pytest.mark.parametrize(
    "fault",
    (
        OSError(errno.ENOSPC, "metadata full"),
        OSError("metadata write failed"),
    ),
)
def test_raw_metadata_write_io_failure_propagates(
    isolated_config, monkeypatch, tmp_path, fault
):
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_atomic_write_bytes_nofollow",
        lambda path, payload: (_ for _ in ()).throw(fault),
    )

    with pytest.raises(OSError, match="metadata"):
        services._write_raw_records(
            tmp_path / "metadata.csv",
            ({"sha256": "a" * 64, "local_path": "raw/item.glb"},),
        )


def test_zip_source_eio_propagates(isolated_config, monkeypatch, tmp_path):
    context = ShardContext.for_test(
        tmp_path / "zip-eio", "ABO", "ABO-00000"
    )
    asset_sha = "a" * 64
    write_instances(context, (asset_sha,))
    write_raw_metadata(
        context,
        (
            {
                "sha256": asset_sha,
                "local_path": "raw/bundle.zip/item.glb",
            },
        ),
    )
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_open_regular_beneath",
        lambda root, relative: (_ for _ in ()).throw(
            OSError(errno.EIO, "ZIP source EIO")
        ),
    )

    with pytest.raises(OSError, match="ZIP source EIO"):
        services.stage_raw(context)


@pytest.mark.parametrize(
    "fault",
    (OSError(errno.EIO, "cleanup EIO"), OSError("cleanup failed")),
)
def test_cleanup_io_failure_propagates(
    isolated_config, monkeypatch, tmp_path, fault
):
    context = ShardContext.for_test(
        tmp_path / "cleanup-eio", "ABO", "ABO-00000"
    )
    services = PipelineServices(
        isolated_config,
        resource_guard=FakeResourceGuard(),
        published_batch_verifier=lambda active_context: None,
        raw_archive_verifier=lambda active_context: None,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_remove_tree_nofollow",
        lambda path: (_ for _ in ()).throw(fault),
    )

    with pytest.raises(OSError, match="cleanup"):
        services.cleanup_local(context)


@pytest.mark.parametrize(
    "fault",
    (
        OSError(errno.ESTALE, "accounting ESTALE"),
        OSError("accounting probe failed"),
    ),
)
def test_accounting_probe_io_failure_propagates(
    isolated_config, monkeypatch, tmp_path, fault
):
    services = PipelineServices(
        isolated_config, resource_guard=FakeResourceGuard()
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_regular_file_size_nofollow",
        lambda path, missing_ok=False: (_ for _ in ()).throw(fault),
    )

    with pytest.raises(OSError, match="accounting"):
        services._path_size(tmp_path / "pack.tar")
