# Full-VRAM GPU Runtime Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep encoder GPU-memory target at 80% while applying one shared, auditable, config-hash-independent hard ceiling of 100% to all current and future production workers.

**Architecture:** Add a focused `gpu_policy.py` module that atomically persists and validates `control/runtime/gpu_policy.json`, falling back to canonical config values when absent. Worker startup reads that policy into `WorkerExecutionConfig` without changing canonical identity, while both resource brokers change their hard-ceiling comparison to admit an exact total of 100%. The workers CLI writes and reports the shared policy, enabling batch-boundary deployment without rebuilding the frozen queue.

**Tech Stack:** Python 3, standard-library dataclasses/datetime/json/pathlib, argparse, shared-filesystem no-follow atomic writes, pytest, existing worker registry and supervisors.

## Global Constraints

- Preserve `units.json` byte-for-byte and preserve its current config hash.
- Preserve `priority.json`, all leases, history, terminal records, checkpoints, packs, archives, and quality ledgers.
- Keep `gpu_memory_target_percent` at exactly 80 for this production scope.
- Set the effective `gpu_memory_hard_percent` to exactly 100.
- Permit an internal reservation sum equal to the hard ceiling.
- Do not interrupt a live batch merely to change the policy.
- Make the policy shared and auditable so dynamically added nodes cannot silently use a different ceiling.
- Keep existing CUDA OOM adaptive micro-batch reduction and batch-level retry behavior unchanged.
- Keep CUDA 12.8 and PyTorch 2.8 or newer on every production worker.

---

## File Structure

- `data_toolkit/pipeline/gpu_policy.py`: own the policy value, validation, fallback, no-follow read, and atomic write.
- `data_toolkit/pipeline/worker_runtime.py`: apply a validated runtime policy to node-local execution config while preserving canonical hash identity.
- `data_toolkit/pipeline/parallelism.py`: make the hard ceiling inclusive in both resource brokers.
- `data_toolkit/pipeline/cli.py`: expose shared policy mutation/status and load it before supervisor/worker startup.
- `tests/data_toolkit/test_gpu_policy.py`: prove artifact safety, fallback, and validation.
- `tests/data_toolkit/test_worker_runtime.py`: prove effective target/hard values and unchanged config hash.
- `tests/data_toolkit/test_parallelism.py`: prove exact-ceiling admission and above-ceiling rejection.
- `tests/data_toolkit/test_cli.py`: prove CLI persistence, misuse rejection, status visibility, and malformed-policy pre-claim stop.

### Task 1: Shared GPU Policy Artifact

**Files:**
- Create: `data_toolkit/pipeline/gpu_policy.py`
- Create: `tests/data_toolkit/test_gpu_policy.py`

**Interfaces:**
- Produces: `GpuRuntimePolicy(target_percent: int, hard_percent: int)`.
- Produces: `read_gpu_runtime_policy(path: Path, *, canonical_target_percent: int, canonical_hard_percent: int) -> GpuRuntimePolicy`.
- Produces: `write_gpu_runtime_policy(path: Path, policy: GpuRuntimePolicy, *, canonical_target_percent: int, now: datetime) -> None`.
- Produces: `GpuPolicyError`, raised for malformed, unsupported, or mismatched policy artifacts.

- [ ] **Step 1: Write failing fallback and round-trip tests**

```python
from datetime import datetime, timezone
from pathlib import Path

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
```

- [ ] **Step 2: Run tests and verify the missing-module failure**

Run: `PYTHONPATH=. /opt/conda/envs/pixal3d/bin/pytest -q tests/data_toolkit/test_gpu_policy.py`

Expected: collection FAILS with `ModuleNotFoundError: data_toolkit.pipeline.gpu_policy`.

- [ ] **Step 3: Implement the minimal policy module**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from .orchestrator import _atomic_write_bytes_nofollow, _read_regular_bytes_nofollow


GPU_POLICY_SCHEMA_VERSION = 1


class GpuPolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class GpuRuntimePolicy:
    target_percent: int
    hard_percent: int


def _validate(policy: GpuRuntimePolicy, canonical_target_percent: int) -> None:
    if (
        type(policy.target_percent) is not int
        or type(policy.hard_percent) is not int
        or policy.target_percent != canonical_target_percent
        or not 0 < policy.target_percent < policy.hard_percent <= 100
    ):
        raise GpuPolicyError(
            "GPU runtime policy must preserve the canonical target and satisfy "
            "0 < target < hard <= 100"
        )


def read_gpu_runtime_policy(
    path: Path,
    *,
    canonical_target_percent: int,
    canonical_hard_percent: int,
) -> GpuRuntimePolicy:
    fallback = GpuRuntimePolicy(
        canonical_target_percent, canonical_hard_percent
    )
    payload = _read_regular_bytes_nofollow(Path(path), missing_ok=True)
    if payload is None:
        _validate(fallback, canonical_target_percent)
        return fallback
    try:
        value = json.loads(payload)
        if not isinstance(value, dict) or set(value) != {
            "schema_version", "target_percent", "hard_percent", "updated_at"
        }:
            raise ValueError("unexpected fields")
        if value["schema_version"] != GPU_POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported schema version")
        updated_at = datetime.fromisoformat(value["updated_at"])
        if updated_at.tzinfo is None:
            raise ValueError("naive update time")
        policy = GpuRuntimePolicy(
            value["target_percent"], value["hard_percent"]
        )
        _validate(policy, canonical_target_percent)
        return policy
    except (
        GpuPolicyError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise GpuPolicyError(f"invalid GPU runtime policy: {error}") from error


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": GPU_POLICY_SCHEMA_VERSION,
        **asdict(policy),
        "updated_at": now.astimezone(timezone.utc).isoformat(),
    }
    _atomic_write_bytes_nofollow(
        path,
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8"),
    )
```

- [ ] **Step 4: Add exact invalid-policy tests**

```python
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
```

- [ ] **Step 5: Run the policy tests**

Run: `PYTHONPATH=. /opt/conda/envs/pixal3d/bin/pytest -q tests/data_toolkit/test_gpu_policy.py`

Expected: all selected tests PASS.

- [ ] **Step 6: Commit the artifact API**

```bash
git add data_toolkit/pipeline/gpu_policy.py tests/data_toolkit/test_gpu_policy.py
git commit -m "feat: persist shared GPU runtime policy"
```

### Task 2: Effective Worker Policy and Inclusive Admission

**Files:**
- Modify: `data_toolkit/pipeline/worker_runtime.py`
- Modify: `data_toolkit/pipeline/parallelism.py`
- Modify: `tests/data_toolkit/test_worker_runtime.py`
- Modify: `tests/data_toolkit/test_parallelism.py`

**Interfaces:**
- Consumes: `GpuRuntimePolicy` from Task 1.
- Changes: `execution_config(canonical, registration, gpu_policy=None) -> WorkerExecutionConfig`.
- Produces: node-local parallelism with policy target/hard and unchanged `config_hash()`.
- Changes: both brokers admit a resulting reservation equal to, but not greater than, their hard ceiling.

- [ ] **Step 1: Write the failing execution-config identity test**

```python
from data_toolkit.pipeline.gpu_policy import GpuRuntimePolicy


def test_execution_config_applies_gpu_policy_without_changing_identity():
    canonical = load_config(CONFIG)
    registration = WorkerRegistration(
        node_id="node16",
        ssh_target="node16",
        cpu_limit=40,
        gpu_indices=(0, 1, 2, 3),
        data2_root=Path("/file2/youngwoo/pixal3d"),
        data3_root=Path("/file3/youngwoo/pixal3d"),
        local_root=Path("/home/youngwoo/data/pixal3d"),
    )

    configured = execution_config(
        canonical, registration, GpuRuntimePolicy(80, 100)
    )

    assert configured.config_hash() == canonical.config_hash()
    assert canonical.parallelism.gpu_memory_hard_percent == 90
    assert configured.parallelism.gpu_memory_target_percent == 80
    assert configured.parallelism.gpu_memory_hard_percent == 100
```

- [ ] **Step 2: Run the execution-config test and verify signature failure**

Run: `PYTHONPATH=. /opt/conda/envs/pixal3d/bin/pytest -q tests/data_toolkit/test_worker_runtime.py::test_execution_config_applies_gpu_policy_without_changing_identity`

Expected: FAIL because `execution_config` accepts only two positional arguments.

- [ ] **Step 3: Apply the optional policy in worker execution config**

```python
# worker_runtime.py
from .gpu_policy import GpuRuntimePolicy

class WorkerExecutionConfig:
    def __init__(
        self,
        canonical: PipelineConfig,
        registration: WorkerRegistration,
        gpu_policy: GpuRuntimePolicy | None = None,
    ) -> None:
        effective = gpu_policy or GpuRuntimePolicy(
            canonical.parallelism.gpu_memory_target_percent,
            canonical.parallelism.gpu_memory_hard_percent,
        )
        if effective.target_percent != canonical.parallelism.gpu_memory_target_percent:
            raise ValueError("GPU runtime target must match canonical target")
        # Preserve the existing setup and replace parallelism as follows:
        self.parallelism = replace(
            canonical.parallelism,
            gpu_count=gpu_count,
            cpu_physical_cores=cpu_limit,
            gpu_memory_target_percent=effective.target_percent,
            gpu_memory_hard_percent=effective.hard_percent,
        )

def execution_config(
    canonical: PipelineConfig,
    registration: WorkerRegistration,
    gpu_policy: GpuRuntimePolicy | None = None,
) -> WorkerExecutionConfig:
    return WorkerExecutionConfig(canonical, registration, gpu_policy)
```

- [ ] **Step 4: Write failing exact-ceiling broker tests**

```python
def test_node_broker_admits_exact_hard_limit_but_not_more():
    broker = NodeResourceBroker(
        cpu_limit=44, gpu_count=1, gpu_hard_percent=100.0
    )
    render = broker.try_acquire(
        cpu_cores=0, gpu_indices=(0,), gpu_memory_percent=20.0
    )
    encode = broker.try_acquire(
        cpu_cores=0, gpu_indices=(0,), gpu_memory_percent=80.0
    )

    assert render is not None and encode is not None
    assert broker.try_acquire(
        cpu_cores=0, gpu_indices=(0,), gpu_memory_percent=0.1
    ) is None


def test_dynamic_broker_admits_observed_plus_reserved_exactly_at_hard_limit():
    broker = DynamicResourceBroker()
    broker.register(
        WorkerSpec(
            "node16", cpu_limit=4, gpu_indices=(0,), gpu_hard_percent=100.0
        )
    )
    broker.update_gpu_memory(
        "node16",
        (GpuMemoryState(index=0, used_mib=20.0, total_mib=100.0),),
    )

    assert broker.acquire_any(
        cpu_cores=1, gpu_count=1, gpu_memory_percent=80.0
    ) is not None
```

- [ ] **Step 5: Run the broker tests and verify exact-ceiling failures**

Run: `PYTHONPATH=. /opt/conda/envs/pixal3d/bin/pytest -q tests/data_toolkit/test_parallelism.py -k 'exact_hard_limit or exactly_at_hard_limit'`

Expected: FAIL because both brokers currently require totals to be strictly below the hard limit.

- [ ] **Step 6: Make both hard-ceiling comparisons inclusive**

```python
# NodeResourceBroker.try_acquire:
if any(
    self._gpu_allocated_percent[index] + gpu_memory_percent
    > self.gpu_hard_percent
    for index in gpu_indices
):
    return None

# DynamicResourceBroker.acquire_any selection:
if (
    worker["gpu_external"][index]
    + worker["gpu_allocated"][index]
    + float(gpu_memory_percent)
    <= float(spec.gpu_hard_percent)
)
```

- [ ] **Step 7: Run worker-runtime and broker modules**

Run: `PYTHONPATH=. /opt/conda/envs/pixal3d/bin/pytest -q tests/data_toolkit/test_worker_runtime.py tests/data_toolkit/test_parallelism.py`

Expected: all tests PASS, including unchanged OOM micro-batch reduction tests.

- [ ] **Step 8: Commit effective policy and admission semantics**

```bash
git add data_toolkit/pipeline/worker_runtime.py data_toolkit/pipeline/parallelism.py tests/data_toolkit/test_worker_runtime.py tests/data_toolkit/test_parallelism.py
git commit -m "feat: admit GPU reservations through full capacity"
```

### Task 3: Policy CLI, Status, and Worker Startup

**Files:**
- Modify: `data_toolkit/pipeline/cli.py`
- Modify: `tests/data_toolkit/test_cli.py`

**Interfaces:**
- Consumes: policy read/write functions from Task 1 and policy-aware `execution_config` from Task 2.
- Produces: `workers --action set-gpu-policy --gpu-target-percent 80 --gpu-hard-percent 100`.
- Produces: per-node status fields `gpu_memory_target_percent` and `gpu_memory_hard_percent`.
- Guarantees: supervisor and worker resolve policy from their registered shared data2 root before building execution config.

- [ ] **Step 1: Write failing policy CLI persistence/status test**

```python
def test_workers_cli_sets_and_reports_shared_gpu_policy(tmp_config, capsys):
    assert main([
        "workers", "--config", str(tmp_config), "--action", "register",
        "--node-id", "node17", "--ssh-target", "local",
        "--cpu-limit", "4", "--gpus", "0,1",
    ]) == 0
    capsys.readouterr()

    assert main([
        "workers", "--config", str(tmp_config),
        "--action", "set-gpu-policy",
        "--gpu-target-percent", "80",
        "--gpu-hard-percent", "100",
    ]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "gpu_memory_hard_percent": 100,
        "gpu_memory_target_percent": 80,
    }

    assert main([
        "workers", "--config", str(tmp_config), "--action", "status",
    ]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["node17"]["gpu_memory_target_percent"] == 80
    assert status["node17"]["gpu_memory_hard_percent"] == 100
```

- [ ] **Step 2: Run the focused CLI test and verify parser failure**

Run: `PYTHONPATH=. /opt/conda/envs/pixal3d/bin/pytest -q tests/data_toolkit/test_cli.py::test_workers_cli_sets_and_reports_shared_gpu_policy`

Expected: FAIL because argparse rejects `set-gpu-policy`.

- [ ] **Step 3: Add parser action and exclusive arguments**

```python
workers.add_argument(
    "--action",
    choices=(
        "register", "activate", "drain", "remove", "status",
        "set-gpu-policy",
    ),
    required=True,
)
workers.add_argument("--gpu-target-percent", type=_positive_integer)
workers.add_argument("--gpu-hard-percent", type=_positive_integer)

# PipelineArgumentParser._validate:
if command == "workers":
    policy_arguments = (
        args.gpu_target_percent is not None,
        args.gpu_hard_percent is not None,
    )
    if args.action == "set-gpu-policy" and not all(policy_arguments):
        self.error("workers set-gpu-policy requires target and hard percentages")
    if args.action != "set-gpu-policy" and any(policy_arguments):
        self.error("GPU policy arguments require workers set-gpu-policy")
```

- [ ] **Step 4: Add one shared-policy resolver and wire mutation/status**

```python
from .gpu_policy import (
    GpuPolicyError,
    GpuRuntimePolicy,
    read_gpu_runtime_policy,
    write_gpu_runtime_policy,
)

def _gpu_policy_path(data2_root: Path) -> Path:
    return Path(data2_root) / "control/runtime/gpu_policy.json"

def _effective_gpu_policy(config, data2_root: Path) -> GpuRuntimePolicy:
    return read_gpu_runtime_policy(
        _gpu_policy_path(data2_root),
        canonical_target_percent=config.parallelism.gpu_memory_target_percent,
        canonical_hard_percent=config.parallelism.gpu_memory_hard_percent,
    )

# In workers dispatch, before register/drain branches:
if args.action == "set-gpu-policy":
    policy = GpuRuntimePolicy(
        args.gpu_target_percent, args.gpu_hard_percent
    )
    write_gpu_runtime_policy(
        _gpu_policy_path(config.paths.data2_root),
        policy,
        canonical_target_percent=config.parallelism.gpu_memory_target_percent,
        now=datetime.now(timezone.utc),
    )
    print(json.dumps({
        "gpu_memory_target_percent": policy.target_percent,
        "gpu_memory_hard_percent": policy.hard_percent,
    }, sort_keys=True))
    return SUCCESS

# In status:
policy = _effective_gpu_policy(config, config.paths.data2_root)
# Add to each node dictionary:
"gpu_memory_target_percent": policy.target_percent,
"gpu_memory_hard_percent": policy.hard_percent,
```

Add `GpuPolicyError` to the operator-error exception tuple in `main()`.

- [ ] **Step 5: Load policy for supervisor and worker execution configs**

```python
def _registered_execution_config(config, registration):
    policy = _effective_gpu_policy(config, registration.data2_root)
    return execution_config(config, registration, policy)

# Replace both supervisor and worker calls:
held_config = _registered_execution_config(config, registration)
```

- [ ] **Step 6: Add parser misuse and malformed-policy pre-claim tests**

```python
@pytest.mark.parametrize("argv", [
    ["workers", "--action", "set-gpu-policy", "--gpu-target-percent", "80"],
    ["workers", "--action", "status", "--gpu-hard-percent", "100"],
])
def test_workers_gpu_policy_arguments_are_action_scoped(tmp_config, argv):
    with pytest.raises(SystemExit):
        parser().parse_args([*argv, "--config", str(tmp_config)])


def test_malformed_gpu_policy_stops_worker_before_claim(
    tmp_config, monkeypatch, capsys
):
    config = load_config(tmp_config)
    registry_path = config.paths.data2_root / "control/runtime/workers.json"
    assert main([
        "workers", "--config", str(tmp_config), "--action", "register",
        "--node-id", "node17", "--ssh-target", "local",
        "--cpu-limit", "4", "--gpus", "0,1",
    ]) == 0
    policy_path = config.paths.data2_root / "control/runtime/gpu_policy.json"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text("{}")
    monkeypatch.setattr(
        "data_toolkit.pipeline.cli.validate_worker_environment",
        lambda configured, registered: pytest.fail("validation must not run"),
    )

    assert main([
        "worker", "--config", str(tmp_config), "--node-id", "node17",
        "--worker-registry", str(registry_path), "--once",
    ]) == 2
    assert "invalid GPU runtime policy" in capsys.readouterr().err
```

- [ ] **Step 7: Run CLI, worker-runtime, and policy tests**

Run: `PYTHONPATH=. /opt/conda/envs/pixal3d/bin/pytest -q tests/data_toolkit/test_cli.py tests/data_toolkit/test_worker_runtime.py tests/data_toolkit/test_gpu_policy.py`

Expected: all tests PASS.

- [ ] **Step 8: Commit CLI and startup integration**

```bash
git add data_toolkit/pipeline/cli.py tests/data_toolkit/test_cli.py
git commit -m "feat: apply shared GPU policy to production workers"
```

### Task 4: Full Regression and Batch-Boundary Deployment

**Files:**
- Verify: `data_toolkit/pipeline/gpu_policy.py`
- Verify: `data_toolkit/pipeline/worker_runtime.py`
- Verify: `data_toolkit/pipeline/parallelism.py`
- Verify: `data_toolkit/pipeline/cli.py`
- Deploy: runtime files to `/home/youngwoo/Pixal3D/data_toolkit/pipeline/` on node16.
- Mutate through CLI: `/root/data2/pixal3d/control/runtime/gpu_policy.json`.

**Interfaces:**
- Consumes: all Tasks 1-3 and the existing source-priority supervisors/watchers.
- Produces: both nodes reporting target 80/hard 100 and continuing the shared priority queue on one verified runtime commit.

- [ ] **Step 1: Run whitespace and complete regression verification**

```bash
git diff --check HEAD~3..HEAD
PYTHONPATH=. /opt/conda/envs/pixal3d/bin/pytest -q tests/data_toolkit
```

Expected: no diff-check output; every data-toolkit test passes.

- [ ] **Step 2: Capture pre-cutover identity and live leases**

```bash
git rev-parse HEAD
sha256sum /root/data2/pixal3d/control/runtime/work_queue/units.json
PYTHONPATH=. /opt/conda/envs/pixal3d/bin/python -m data_toolkit.pipeline.cli \
  queue --config data_toolkit/configs/multiview_preprocess.yaml --action status
PYTHONPATH=. /opt/conda/envs/pixal3d/bin/python -m data_toolkit.pipeline.cli \
  workers --config data_toolkit/configs/multiview_preprocess.yaml --action status
```

Expected: current leases remain live/distinct and priority is `ABO`, `3D-FUTURE`, `HSSD`, `ObjaverseXL_sketchfab`.

- [ ] **Step 3: Drain each active node without terminating its lease**

```bash
for worker_node in node16 node17; do
  PYTHONPATH=. /opt/conda/envs/pixal3d/bin/python \
    -m data_toolkit.pipeline.cli workers \
    --config data_toolkit/configs/multiview_preprocess.yaml \
    --action drain --node-id "$worker_node"
done
```

Expected: registry state becomes draining while each existing lease heartbeat continues until its batch boundary.

- [ ] **Step 4: Deploy identical runtime files to node16 with rollback copies**

```bash
deployment_commit=$(git rev-parse HEAD)
ssh -p 55555 youngwoo@n16.unist.info \
  "mkdir -p /home/youngwoo/Pixal3D/.deploy-$deployment_commit/incoming \
             /home/youngwoo/Pixal3D/.deploy-$deployment_commit/previous"
scp -P 55555 \
  data_toolkit/pipeline/gpu_policy.py \
  data_toolkit/pipeline/worker_runtime.py \
  data_toolkit/pipeline/parallelism.py \
  data_toolkit/pipeline/cli.py \
  youngwoo@n16.unist.info:/home/youngwoo/Pixal3D/.deploy-$deployment_commit/incoming/
ssh -p 55555 youngwoo@n16.unist.info \
  "deployment_root=/home/youngwoo/Pixal3D/.deploy-$deployment_commit; \
   for runtime_file in gpu_policy.py worker_runtime.py parallelism.py cli.py; do \
     current=/home/youngwoo/Pixal3D/data_toolkit/pipeline/\$runtime_file; \
     if test -f \$current; then cp -p \$current \$deployment_root/previous/\$runtime_file; fi; \
     mv \$deployment_root/incoming/\$runtime_file \$current; \
   done; \
   sha256sum /home/youngwoo/Pixal3D/data_toolkit/pipeline/{gpu_policy.py,worker_runtime.py,parallelism.py,cli.py}"
sha256sum data_toolkit/pipeline/{gpu_policy.py,worker_runtime.py,parallelism.py,cli.py}
```

Expected: node16 and node17 print identical SHA-256 values for all four files.
Keep `previous/` until post-cutover verification passes.

- [ ] **Step 5: Verify node16 runtime before policy mutation**

Run inside `youngwoo_diyscene`:

```bash
cd /home/youngwoo/Pixal3D
PYTHONPATH=. /home/youngwoo/miniconda3/envs/pixal3d/bin/python -c \
  'import torch; from data_toolkit.pipeline.gpu_policy import GpuRuntimePolicy; assert torch.__version__.startswith("2.8."); assert torch.version.cuda == "12.8"; assert GpuRuntimePolicy(80, 100).hard_percent == 100'
```

Expected: exit status 0.

- [ ] **Step 6: Write the shared policy once and prove queue identity is unchanged**

```bash
before=$(sha256sum /root/data2/pixal3d/control/runtime/work_queue/units.json)
PYTHONPATH=. /opt/conda/envs/pixal3d/bin/python -m data_toolkit.pipeline.cli \
  workers --config data_toolkit/configs/multiview_preprocess.yaml \
  --action set-gpu-policy --gpu-target-percent 80 --gpu-hard-percent 100
after=$(sha256sum /root/data2/pixal3d/control/runtime/work_queue/units.json)
test "$before" = "$after"
```

Expected: policy output reports target 80/hard 100 and the test exits 0.

- [ ] **Step 7: Restart/reactivate each node independently at its batch boundary**

```bash
deployment_commit=$(git rev-parse HEAD)
tmux new-session -d -s pixal3d-node17-supervisor \
  -c /root/dev/Pixal3D/.worktrees/parallel-preprocessing \
  "exec env PYTHONPATH=. PIXAL3D_TOOL_COMMIT=$deployment_commit \
   /opt/conda/envs/pixal3d/bin/python -m data_toolkit.pipeline.cli supervisor \
   --config data_toolkit/configs/multiview_preprocess.yaml --node-id node17 \
   --worker-registry /root/data2/pixal3d/control/runtime/workers.json"
ssh -p 55555 youngwoo@n16.unist.info \
  "docker exec -d youngwoo_diyscene sh -lc \
   'cd /home/youngwoo/Pixal3D; exec env PYTHONPATH=. \
    PIXAL3D_TOOL_COMMIT=$deployment_commit \
    /home/youngwoo/miniconda3/envs/pixal3d/bin/python \
    -m data_toolkit.pipeline.cli supervisor \
    --config data_toolkit/configs/multiview_preprocess.yaml --node-id node16 \
    --worker-registry /root/data2/pixal3d/control/runtime/workers.json'"
for worker_node in node16 node17; do
  PYTHONPATH=. /opt/conda/envs/pixal3d/bin/python \
    -m data_toolkit.pipeline.cli workers \
    --config data_toolkit/configs/multiview_preprocess.yaml \
    --action activate --node-id "$worker_node"
done
```

Run each supervisor/activation block only after that node's pre-cutover worker
PID has exited. Expected: the first finished node resumes without waiting for
the second; no old lease is revoked.

- [ ] **Step 8: Verify policy, processes, leases, and VRAM telemetry**

```bash
PYTHONPATH=. /opt/conda/envs/pixal3d/bin/python -m data_toolkit.pipeline.cli \
  workers --config data_toolkit/configs/multiview_preprocess.yaml --action status
PYTHONPATH=. /opt/conda/envs/pixal3d/bin/python -m data_toolkit.pipeline.cli \
  queue --config data_toolkit/configs/multiview_preprocess.yaml --action status
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits
```

Expected: every node entry reports target 80/hard 100; new claims are distinct and obey source priority; rendering/encoding may use capacity through the 100% admission ceiling. OOM/retry is acceptable under the approved policy, but duplicate claims or a node reporting hard 90 is not.
