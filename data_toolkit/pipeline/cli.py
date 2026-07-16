from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Sequence

from .config import load_config
from .orchestrator import (
    CheckpointError,
    EscalationCategory,
    InfrastructureError,
    IntegrationProviderRequired,
    PipelineStopped,
)
from .preflight import PreflightStatus, run_preflight
from .reporting import ReportValidationError
from .resources import ResourceAccountingError, ResourceLimitExceeded
from .runtime import (
    ArtifactValidationError,
    build_mutating_services,
    build_read_only_services,
    read_gate_report,
)
from .validation import ValidationError


SUCCESS = 0
OPERATOR_BLOCKED = 2
RESOURCE_STOP = 3
DATA_QUALITY_STOP = 4
GATES = ("smoke", "pilot", "production")


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


class PipelineArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        result = super().parse_args(args, namespace)
        self._validate(result)
        return result

    def _validate(self, args) -> None:
        command = args.command
        source = getattr(args, "source", None)
        shard = getattr(args, "shard", None)
        count = getattr(args, "count", None)
        if command == "plan":
            if (source is None) != (shard is None):
                self.error("plan requires --source and --shard together")
            if count is not None and source is None:
                self.error("plan --count requires --source and --shard")
        if command == "run" and args.gate == "production" and count is not None:
            self.error("production run cannot be restricted with --count")
        if command == "report" and args.gate is None and not args.hardware_check:
            self.error("report requires --gate or --hardware-check")


def parser() -> argparse.ArgumentParser:
    root = PipelineArgumentParser(prog="pixal3d-preprocess")
    children = root.add_subparsers(dest="command", required=True)
    for name in (
        "preflight",
        "registry",
        "plan",
        "run",
        "resume",
        "audit",
        "report",
    ):
        child = children.add_parser(name)
        child.add_argument("--config", type=Path, required=True)

    for name in ("plan", "run", "resume", "audit"):
        children.choices[name].add_argument(
            "--gate", choices=GATES, required=True
        )
    for name in ("plan", "run"):
        children.choices[name].add_argument(
            "--count", type=_positive_integer
        )

    children.choices["plan"].add_argument("--source")
    children.choices["plan"].add_argument("--shard")
    for name in ("run", "resume", "audit"):
        children.choices[name].add_argument("--source", required=True)
        children.choices[name].add_argument("--shard", required=True)

    children.choices["report"].add_argument(
        "--gate", choices=GATES
    )
    children.choices["report"].add_argument(
        "--hardware-check", action="store_true"
    )
    return root


def _operator_error(error: BaseException) -> int:
    print(str(error), file=sys.stderr)
    return OPERATOR_BLOCKED


def _validate_scope(args, config) -> None:
    source = getattr(args, "source", None)
    shard = getattr(args, "shard", None)
    if source is None:
        return
    if source not in config.sources:
        if source in config.evaluation_sources:
            raise ArtifactValidationError(
                f"evaluation source cannot be used for training: {source}"
            )
        raise ArtifactValidationError(f"training source is not configured: {source}")
    if not re.fullmatch(rf"{re.escape(source)}-[0-9]{{5}}", shard or ""):
        raise ArtifactValidationError(
            f"shard does not belong to source {source}: {shard}"
        )


def _required_gates(args, config) -> None:
    if args.command not in {"run", "resume"}:
        return
    if args.gate == "pilot":
        read_gate_report(config, "smoke")
    elif args.gate == "production":
        smoke = read_gate_report(config, "smoke")
        pilot = read_gate_report(config, "pilot")
        if smoke["config_hash"] != pilot["config_hash"]:
            raise ArtifactValidationError(
                "smoke and pilot gate config hashes do not match"
            )


def _stopped_exit(error: PipelineStopped) -> int:
    print(error.report.reason, file=sys.stderr)
    if error.report.category == EscalationCategory.RESOURCE:
        return RESOURCE_STOP
    if error.report.category == EscalationCategory.DATA_QUALITY:
        return DATA_QUALITY_STOP
    return OPERATOR_BLOCKED


def _dispatch(args, config) -> int:
    if args.command == "preflight":
        results = run_preflight(config)
        for item in results:
            print(f"{item.source}: {item.status.value}: {item.message}")
        if any(item.status != PreflightStatus.READY for item in results):
            return OPERATOR_BLOCKED
        return SUCCESS

    _validate_scope(args, config)
    _required_gates(args, config)

    if args.command == "plan":
        services = build_read_only_services(config)
        lines = services.plan(
            args.gate,
            args.source,
            args.shard,
            args.count,
            freeze=False,
        )
        for line in lines:
            print(line)
        return SUCCESS

    with build_mutating_services(config) as runtime:
        services = runtime.services
        if args.command == "registry":
            frame = services.build_registry()
            print(f"registry: {len(frame)} assets")
        elif args.command == "run":
            services.run(args.gate, args.source, args.shard, args.count)
        elif args.command == "resume":
            services.resume(args.gate, args.source, args.shard)
        elif args.command == "audit":
            services.audit(args.gate, args.source, args.shard)
        elif args.command == "report":
            result = services.report(args.gate, args.hardware_check)
            for path in result:
                print(path)
        else:
            raise AssertionError(f"unhandled command: {args.command}")
    return SUCCESS


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except (OSError, ValueError) as error:
        return _operator_error(error)
    try:
        return _dispatch(args, config)
    except PipelineStopped as error:
        return _stopped_exit(error)
    except ResourceLimitExceeded as error:
        print(str(error), file=sys.stderr)
        return RESOURCE_STOP
    except ValidationError as error:
        print(str(error), file=sys.stderr)
        return DATA_QUALITY_STOP
    except (
        ArtifactValidationError,
        IntegrationProviderRequired,
        InfrastructureError,
        ResourceAccountingError,
        ReportValidationError,
    ) as error:
        return _operator_error(error)


if __name__ == "__main__":
    raise SystemExit(main())
