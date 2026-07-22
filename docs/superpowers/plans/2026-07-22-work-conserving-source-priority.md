# Work-Conserving Source Priority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the shared production queue prefer `ABO`, then `3D-FUTURE`, `HSSD`, and `ObjaverseXL_sketchfab`, while allowing idle workers to claim from the next source whenever all higher-priority work is already live-leased.

**Architecture:** Store mutable scheduling policy in an atomic queue-local `priority.json` that is deliberately separate from the frozen `units.json` and config hash. `ProductionWorkQueue.claim()` reads the effective priority on every claim and performs a stable source-priority sort before using the existing atomic lease-directory algorithm. The queue CLI writes and reports the policy without rebuilding work units or interrupting existing leases.

**Tech Stack:** Python 3, standard-library `dataclasses`/`datetime`/`json`/`pathlib`, argparse, pytest, shared-filesystem atomic rename and atomic directory creation.

## Global Constraints

- Preserve the frozen `units.json` manifest and its config hash.
- Preserve completed, failed, history, checkpoint, archive, and pack records.
- Keep source-local batch ordering identical to the frozen manifest.
- Keep node16 and node17 on one atomic shared queue without duplicate claims.
- Treat both completed and terminally quarantined batches as terminal.
- Reclaim a stale higher-priority lease before considering a lower-priority source, subject to the existing attempt budget.
- Do not introduce a strict barrier that idles a worker while a lower-priority batch is claimable.
- Do not revoke or interrupt any lease that was live before the priority update.

---

## File Structure

- `data_toolkit/pipeline/work_queue.py`: own priority artifact validation, atomic persistence, effective-priority lookup, stable ordered-unit view, and status exposure.
- `data_toolkit/pipeline/cli.py`: own `queue --action prioritize --sources ...` argument and command contract.
- `tests/data_toolkit/test_work_queue.py`: prove priority persistence, validation, work-conserving claims, stale recovery, and legacy behavior.
- `tests/data_toolkit/test_cli.py`: prove operator-facing priority write/status behavior and argument validation.

### Task 1: Queue Priority Artifact

**Files:**
- Modify: `data_toolkit/pipeline/work_queue.py`
- Test: `tests/data_toolkit/test_work_queue.py`

**Interfaces:**
- Consumes: frozen work units returned by `ProductionWorkQueue.units() -> tuple[WorkUnit, ...]`.
- Produces: `ProductionWorkQueue.set_source_priority(sources: tuple[str, ...], *, now: datetime) -> None` and `ProductionWorkQueue.source_priority() -> tuple[str, ...]`.

- [ ] **Step 1: Write failing artifact round-trip and legacy-fallback tests**

```python
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
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `pytest -q tests/data_toolkit/test_work_queue.py -k 'source_priority'`

Expected: FAIL because `source_priority` and `set_source_priority` do not exist.

- [ ] **Step 3: Implement schema, validation, fallback, and atomic persistence**

Add this queue state and API in `work_queue.py`, reusing `_read_json`, `_write_json`, `_timestamp`, and `_aware`:

```python
PRIORITY_SCHEMA_VERSION = 1

# In ProductionWorkQueue.__init__:
self.priority_path = self.root / "priority.json"

def set_source_priority(
    self, sources: tuple[str, ...], *, now: datetime
) -> None:
    _aware(now)
    requested = tuple(sources)
    expected = self._manifest_sources()
    if (
        not requested
        or len(set(requested)) != len(requested)
        or set(requested) != set(expected)
        or any(_IDENTIFIER.fullmatch(source) is None for source in requested)
    ):
        raise ValueError(
            "source priority must contain every queue source exactly once"
        )
    _write_json(
        self.priority_path,
        {
            "schema_version": PRIORITY_SCHEMA_VERSION,
            "sources": list(requested),
            "updated_at": _timestamp(now),
        },
    )

def source_priority(self) -> tuple[str, ...]:
    expected = self._manifest_sources()
    value = _read_json(self.priority_path, missing_ok=True)
    if value is None:
        return expected
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "sources", "updated_at"
    }:
        raise ValueError("invalid production source priority")
    if value["schema_version"] != PRIORITY_SCHEMA_VERSION:
        raise ValueError("unsupported production source priority")
    datetime.fromisoformat(value["updated_at"])
    sources = tuple(value["sources"]) if isinstance(value["sources"], list) else ()
    if (
        not sources
        or len(set(sources)) != len(sources)
        or set(sources) != set(expected)
        or any(not isinstance(source, str) or _IDENTIFIER.fullmatch(source) is None
               for source in sources)
    ):
        raise ValueError("invalid production source priority")
    return sources

def _manifest_sources(self) -> tuple[str, ...]:
    return tuple(dict.fromkeys(unit.source for unit in self.units()))
```

- [ ] **Step 4: Add exact validation tests**

```python
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
    queue.priority_path.write_text('{"schema_version":1,"sources":["ABO"]}')

    with pytest.raises(ValueError, match="source priority"):
        queue.source_priority()
    with pytest.raises(ValueError, match="source priority"):
        queue.claim("node17", now=NOW, token="token")
```

- [ ] **Step 5: Run artifact tests**

Run: `pytest -q tests/data_toolkit/test_work_queue.py -k 'source_priority'`

Expected: all selected tests PASS.

- [ ] **Step 6: Commit the artifact API**

```bash
git add data_toolkit/pipeline/work_queue.py tests/data_toolkit/test_work_queue.py
git commit -m "feat: persist production source priority"
```

### Task 2: Work-Conserving Priority Claims

**Files:**
- Modify: `data_toolkit/pipeline/work_queue.py`
- Test: `tests/data_toolkit/test_work_queue.py`

**Interfaces:**
- Consumes: `ProductionWorkQueue.source_priority() -> tuple[str, ...]` from Task 1.
- Produces: `_ordered_units() -> tuple[WorkUnit, ...]`, used only by `claim()`; all public lease interfaces remain unchanged.

- [ ] **Step 1: Write failing stable-order and work-conserving tests**

```python
def priority_units():
    return (
        WorkUnit("ObjaverseXL_sketchfab", "ObjaverseXL_sketchfab-00000", "batch000", 256),
        WorkUnit("ABO", "ABO-00000", "batch000", 256),
        WorkUnit("ABO", "ABO-00000", "batch001", 256),
        WorkUnit("3D-FUTURE", "3D-FUTURE-00000", "batch000", 256),
        WorkUnit("HSSD", "HSSD-00000", "batch000", 256),
    )


def test_claim_uses_priority_and_preserves_source_local_manifest_order(tmp_path):
    queue = ProductionWorkQueue(tmp_path, lease_timeout=timedelta(minutes=5))
    queue.initialize("a" * 64, priority_units(), now=NOW)
    queue.set_source_priority(
        ("ABO", "3D-FUTURE", "HSSD", "ObjaverseXL_sketchfab"), now=NOW
    )

    first = queue.claim("node17", now=NOW, token="first")
    second = queue.claim("node16", now=NOW, token="second")

    assert first.unit.batch_id == "batch000"
    assert second.unit.batch_id == "batch001"


def test_idle_worker_advances_when_all_higher_priority_units_are_live(tmp_path):
    queue = ProductionWorkQueue(tmp_path, lease_timeout=timedelta(minutes=5))
    queue.initialize("a" * 64, priority_units()[1:4], now=NOW)
    queue.set_source_priority(("ABO", "3D-FUTURE"), now=NOW)
    queue.claim("node17", now=NOW, token="abo-0")
    queue.claim("node16", now=NOW, token="abo-1")

    lease = queue.claim("node18", now=NOW, token="future")

    assert lease.unit.source == "3D-FUTURE"
```

- [ ] **Step 2: Run the claim tests and verify priority-order failure**

Run: `pytest -q tests/data_toolkit/test_work_queue.py -k 'claim_uses_priority or idle_worker_advances'`

Expected: the first test FAILS by selecting the manifest-leading Objaverse unit before ABO.

- [ ] **Step 3: Implement the stable ordered-unit view and route claims through it**

```python
def _ordered_units(self) -> tuple[WorkUnit, ...]:
    units = self.units()
    ranks = {source: rank for rank, source in enumerate(self.source_priority())}
    return tuple(sorted(units, key=lambda unit: ranks[unit.source]))

# In claim(), replace:
for unit in self.units():
# with:
for unit in self._ordered_units():
```

Python's stable sort preserves the frozen manifest position within each source.

- [ ] **Step 4: Add stale-high-priority and terminal-skip tests**

```python
def test_stale_high_priority_lease_is_reclaimed_before_lower_source(tmp_path):
    queue = ProductionWorkQueue(tmp_path, lease_timeout=timedelta(minutes=5))
    queue.initialize("a" * 64, priority_units()[1:4:2], now=NOW)
    queue.set_source_priority(("ABO", "3D-FUTURE"), now=NOW)
    stale = queue.claim("node17", now=NOW, token="stale")

    replacement = queue.claim(
        "node16", now=NOW + timedelta(minutes=6), token="replacement"
    )

    assert stale.unit.source == "ABO"
    assert replacement.unit.source == "ABO"
    assert replacement.attempt == 2


def test_terminal_high_priority_units_do_not_block_lower_source(tmp_path):
    selected = priority_units()[1:4:2]
    queue = ProductionWorkQueue(tmp_path, lease_timeout=timedelta(minutes=5))
    queue.initialize("a" * 64, selected, now=NOW)
    queue.set_source_priority(("ABO", "3D-FUTURE"), now=NOW)
    abo = queue.claim("node17", now=NOW, token="abo")
    queue.complete(abo, now=NOW)

    assert queue.claim("node16", now=NOW, token="future").unit.source == "3D-FUTURE"


def test_terminally_failed_high_priority_unit_does_not_block_lower_source(tmp_path):
    selected = priority_units()[1:4:2]
    queue = ProductionWorkQueue(
        tmp_path, lease_timeout=timedelta(minutes=5), max_attempts=1
    )
    queue.initialize("a" * 64, selected, now=NOW)
    queue.set_source_priority(("ABO", "3D-FUTURE"), now=NOW)
    abo = queue.claim("node17", now=NOW, token="abo")
    queue.release(abo, reason="unusable assets", now=NOW)

    assert queue.claim("node16", now=NOW, token="future").unit.source == "3D-FUTURE"
```

- [ ] **Step 5: Run all queue tests**

Run: `pytest -q tests/data_toolkit/test_work_queue.py`

Expected: all tests PASS, including existing atomic distinct-claim and retry tests.

- [ ] **Step 6: Commit claim scheduling**

```bash
git add data_toolkit/pipeline/work_queue.py tests/data_toolkit/test_work_queue.py
git commit -m "feat: prioritize work-conserving production claims"
```

### Task 3: Priority CLI and Auditable Status

**Files:**
- Modify: `data_toolkit/pipeline/cli.py`
- Modify: `data_toolkit/pipeline/work_queue.py`
- Test: `tests/data_toolkit/test_cli.py`
- Test: `tests/data_toolkit/test_work_queue.py`

**Interfaces:**
- Consumes: `set_source_priority()` and `source_priority()` from Task 1.
- Produces: `queue --action prioritize --sources <comma-separated-sources>` and snapshot key `source_priority: list[str]`.

- [ ] **Step 1: Write failing snapshot and CLI tests**

```python
def test_snapshot_reports_effective_source_priority(tmp_path):
    queue = ProductionWorkQueue(tmp_path, lease_timeout=timedelta(minutes=5))
    queue.initialize("a" * 64, units(), now=NOW)
    queue.set_source_priority(("HSSD", "ABO"), now=NOW)

    assert queue.snapshot(now=NOW)["source_priority"] == ["HSSD", "ABO"]
```

Add to `test_cli.py`:

```python
def test_queue_cli_persists_and_reports_source_priority(tmp_config, capsys):
    config = load_config(tmp_config)
    queue = ProductionWorkQueue(
        config.paths.data2_root / "control/runtime/work_queue",
        lease_timeout=timedelta(minutes=5),
    )
    queue.initialize(
        config.config_hash(),
        (
            WorkUnit("ABO", "ABO-00000", "batch000", 1),
            WorkUnit("HSSD", "HSSD-00000", "batch000", 1),
        ),
        now=datetime.now(timezone.utc),
    )

    assert main([
        "queue", "--config", str(tmp_config), "--action", "prioritize",
        "--sources", "ABO,HSSD",
    ]) == 0
    prioritized = json.loads(capsys.readouterr().out)
    assert prioritized["source_priority"] == ["ABO", "HSSD"]

    assert main([
        "queue", "--config", str(tmp_config), "--action", "status",
    ]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["source_priority"] == ["ABO", "HSSD"]
```

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `pytest -q tests/data_toolkit/test_work_queue.py::test_snapshot_reports_effective_source_priority tests/data_toolkit/test_cli.py::test_queue_cli_persists_and_reports_source_priority`

Expected: FAIL because snapshots omit priority and argparse rejects `prioritize`.

- [ ] **Step 3: Implement exact queue CLI parsing and validation**

Add a parser helper:

```python
def _source_priority(value: str) -> tuple[str, ...]:
    sources = tuple(value.split(","))
    if not sources or any(not source for source in sources):
        raise argparse.ArgumentTypeError("must be a comma-separated source list")
    return sources
```

Extend queue arguments and `PipelineArgumentParser._validate()`:

```python
queue.add_argument(
    "--action", choices=("init", "reconcile", "status", "prioritize"), required=True
)
queue.add_argument("--sources", type=_source_priority)

if command == "queue":
    if args.action == "prioritize" and args.sources is None:
        self.error("queue prioritize requires --sources")
    if args.action != "prioritize" and args.sources is not None:
        self.error("queue --sources requires --action prioritize")
```

Before creating mutating pipeline services in the queue command branch, add:

```python
if args.action == "prioritize":
    _assert_queue_config(queue, config)
    queue.set_source_priority(args.sources, now=datetime.now(timezone.utc))
    print(json.dumps(queue.snapshot(now=datetime.now(timezone.utc)), sort_keys=True))
    return SUCCESS
```

- [ ] **Step 4: Expose the effective priority in every snapshot**

```python
return {
    "counts": self.status(now=now),
    "active": active,
    "source_priority": list(self.source_priority()),
}
```

- [ ] **Step 5: Add parser misuse assertions**

```python
@pytest.mark.parametrize("argv", [
    ["queue", "--action", "prioritize"],
    ["queue", "--action", "status", "--sources", "ABO,HSSD"],
])
def test_queue_priority_cli_rejects_missing_or_misplaced_sources(
    tmp_config, argv
):
    with pytest.raises(SystemExit):
        parser().parse_args([*argv, "--config", str(tmp_config)])
```

- [ ] **Step 6: Run queue and CLI test modules**

Run: `pytest -q tests/data_toolkit/test_work_queue.py tests/data_toolkit/test_cli.py`

Expected: all tests PASS.

- [ ] **Step 7: Commit CLI and status support**

```bash
git add data_toolkit/pipeline/cli.py data_toolkit/pipeline/work_queue.py tests/data_toolkit/test_cli.py tests/data_toolkit/test_work_queue.py
git commit -m "feat: control and report production source priority"
```

### Task 4: Full Regression Verification

**Files:**
- Verify: `data_toolkit/pipeline/work_queue.py`
- Verify: `data_toolkit/pipeline/cli.py`
- Verify: `tests/data_toolkit/`

**Interfaces:**
- Consumes: complete queue and CLI behavior from Tasks 1-3.
- Produces: a locally verified commit suitable for identical deployment to node16 and node17.

- [ ] **Step 1: Run formatting/whitespace validation**

Run: `git diff --check HEAD~3..HEAD`

Expected: no output and exit status 0.

- [ ] **Step 2: Run the complete data-toolkit regression suite**

Run: `pytest -q tests/data_toolkit`

Expected: all tests PASS with no failure or error.

- [ ] **Step 3: Inspect the final diff and repository state**

Run: `git diff HEAD~3..HEAD -- data_toolkit/pipeline/work_queue.py data_toolkit/pipeline/cli.py tests/data_toolkit/test_work_queue.py tests/data_toolkit/test_cli.py && git status --short`

Expected: only the planned source/test changes are present and worktree status is clean.

### Task 5: Batch-Boundary Deployment and Live Cutover

**Files:**
- Deploy: the verified Git commit to the existing Pixal3D checkout used by node16.
- Mutate: shared queue artifact `/root/node17/data/pixal3d/control/runtime/work_queue/priority.json` through the CLI only.
- Observe: shared worker registry, leases, supervisor logs, GPU processes, and queue snapshots.

**Interfaces:**
- Consumes: verified Git commit from Task 4, existing supervisor/registry controls, and shared queue root.
- Produces: both nodes claiming under the same work-conserving source priority without disturbing their pre-cutover leases.

- [ ] **Step 1: Capture the pre-cutover queue and worker evidence**

Run on node17:

```bash
python -m data_toolkit.pipeline.cli queue \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --action status
python -m data_toolkit.pipeline.cli workers \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --action status
```

Expected: active leases identify their current node, batch, token/attempt, and stage; both intended nodes remain registered.

- [ ] **Step 2: Put both nodes into draining state without killing active work**

Run for `node16` and `node17`:

```bash
python -m data_toolkit.pipeline.cli workers \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --action drain --node-id NODE_ID
```

Expected: each registry entry becomes `draining`; currently owned lease directories remain present and their heartbeat/stage continues until the batch boundary.

- [ ] **Step 3: Deploy the same verified commit to the existing checkout on each node**

Use the repository's existing non-destructive deployment path and verify on each node:

```bash
git rev-parse HEAD
git status --short
```

Expected: identical commit hashes and no uncommitted deployment drift. Do not reset, delete scratch data, or replace the shared queue manifest.

- [ ] **Step 4: Write the requested shared source priority once**

Run on one node only:

```bash
python -m data_toolkit.pipeline.cli queue \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --action prioritize \
  --sources ABO,3D-FUTURE,HSSD,ObjaverseXL_sketchfab
```

Expected: output contains `"source_priority": ["ABO", "3D-FUTURE", "HSSD", "ObjaverseXL_sketchfab"]`; `units.json` hash and byte content remain unchanged.

- [ ] **Step 5: Reactivate node16 and node17 supervisors at their batch boundaries**

Run for each node after its prior lease reaches completed/failed history:

```bash
python -m data_toolkit.pipeline.cli workers \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --action activate --node-id NODE_ID
```

Expected: the persistent supervisor starts the worker on the deployed commit. No pre-cutover live lease is force-released.

- [ ] **Step 6: Verify live distinct claims and work-conserving priority**

Run:

```bash
python -m data_toolkit.pipeline.cli queue \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --action status
```

Expected: new leases have distinct unit IDs and owners. New claims are ABO while ABO has pending/stale work; if all remaining ABO work is live-leased, an idle node may hold a `3D-FUTURE` lease. Existing Objaverse work that began before cutover may continue until terminal.

- [ ] **Step 7: Record post-cutover operational evidence**

Record the deployed commit, `priority.json` payload, unchanged `units.json` checksum, both worker states, active leases, and supervisor process/log locations in the existing production runbook/status document. Re-run queue status after one claim transition to prove workers continue automatically.
