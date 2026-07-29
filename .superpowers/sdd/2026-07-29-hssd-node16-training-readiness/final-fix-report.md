# Final fix report — HSSD Node16 training readiness

## Status

All five Important findings in `final-review-findings.md` are addressed in
this single final fix wave. The production preparation gate remains
unexecuted: no real materialization, source preflight, model load, trainer,
CUDA workload, W&B operation, checkpoint operation, tmux launch, or external
service was invoked.

The final reviewed deployment manifest is intentionally not tracked here.
It must be generated as an external create-only sidecar from the exact commit
that passes the fresh final review; generating it before that review would bind
the wrong revision.

## Important 1 — exact boolean acceptance

- Acceptance evidence now requires an exact JSON boolean type and identity,
  rather than Python equality between `0/1` and `False/True`.
- Both canonical production-spec validation and private fixture structure
  validation reject integer booleans.
- Direct materialization, source preflight, publication, source-chain
  validation, combined resolution, and launch validation preserve the exact
  type check for ABO, 3D-FUTURE, and HSSD.

TDD evidence:

```text
Initial focused RED: 21 failed, 4 passed
Focused GREEN:       25 passed
Runtime bool RED:    2 failed
Runtime bool GREEN:  2 passed
```

## Important 2 — canonical production profile binding

`training_source_profiles.py` now owns one root-aware production contract for
each named source:

- exact source/profile mapping;
- exact ordered index suffixes and shard identities;
- exact batch sets;
- one shared data root;
- exact frozen and candidate-stage counts;
- exact nested fixed-count contract where applicable;
- exact acceptance mode and boolean gate result.

Every public production materialization, preflight, report/publish, combined
manifest, preparation, and launch-validation boundary invokes this contract.
Production launch validation additionally rereads and validates each pinned
index's source, shard, batch set, digest, and count evidence.

The final legacy audit found three historical opt-outs and closed them:

- the public historical ABO materializer no longer synthesizes a spec from
  caller-controlled fixture inputs;
- public ABO schema-1 report/handoff/publish functions now require a canonical
  root-aware ABO index and canonical publication topology;
- legacy CLI path options are either the exact historical defaults or rejected;
  root-aware operation derives every path from `--data2-root` and
  `--local-root`.

Synthetic mechanics remain available only through explicitly private
`_..._for_fixture` helpers used by tests. Production wrappers were not relaxed.
The Node16 root-aware ABO path accepts the exact checked-in config identity
under the reviewed repository root or the exact
`<local-root>/runtime-configs/<source-stem>.node16-workers1.json` identity
created and validated by the Node16 workflow. A boundary regression exercises
all four ABO stages with those runtime-config paths.

TDD evidence:

```text
Canonical profile/launch focused GREEN: 20 passed
Legacy public materializer RED:          1 failed
Legacy materializer focused GREEN:       5 passed
Materializer + preflight final GREEN:    216 passed
```

## Important 3 — HSSD waiver semantics

- Materialization emits `waiver=production-valid-subset` only for
  `valid_subset_user_waiver` sources.
- ABO and 3D-FUTURE retain their existing waiver requirement.
- HSSD remains `production_gate=true`, emits no waiver, and rejects a waiver
  at materialization, preflight, publication, source-chain, and launch
  validation boundaries.

TDD evidence:

```text
Waiver regressions RED:   2 failed
Waiver regressions GREEN: 2 passed
HSSD launch waiver regression: GREEN
```

## Important 4 — reviewed revision and file inventory

The preparation interface now requires all three deployment inputs in both
plan and execute modes:

```text
--expected-revision
--deployment-manifest
--deployment-manifest-sha256
```

Admission verifies an exact schema-1 sidecar containing:

```text
schema_version
revision
hash_algorithm=sha256
files=[{path, mode, size, sha256}, ...]
```

Verification is independent of `.git`, rejects symlinks/nonregular files,
requires a sorted exact inventory with no extras or omissions, validates
`100644`/`100755` modes, sizes, and SHA-256 bytes, and runs before planning,
admission, or mutation. The deployment binding is persisted in the final
report.

`scripts/generate_node16_deployment_manifest.py` builds the sidecar from the
exact `git archive` stream plus `git ls-tree`, requires a full lowercase
40-character commit ID, writes create-only outside the repository, and rejects
unsafe archive entries. The runbook verifies the sidecar and archive digest
before extraction and never relies on `.git` on Node16.

TDD evidence:

```text
Verifier API RED:                  5 failed
Verifier API GREEN:                5 passed
Required entrypoint binding RED:   2 failed
Required entrypoint binding GREEN: 2 passed
Generator suite GREEN:             19 passed
```

Post-review sidecar procedure:

```bash
FINAL_REVISION="$(git rev-parse HEAD)"
python scripts/generate_node16_deployment_manifest.py \
  --repo-root "$PWD" \
  --revision "$FINAL_REVISION" \
  --output /absolute/external/path/node16-deployment-manifest.json
sha256sum /absolute/external/path/node16-deployment-manifest.json
```

The output path must not exist and must remain outside the checkout. This
procedure is to be run only after the fresh reviewer approves the exact final
commit containing this report.

## Important 5 — complete low-level final-report schema

The low-level writer now rejects missing, extra, malformed, mistyped, or
semantically disconnected nested evidence. It requires:

- all three exact source identities and canonical artifact path/digest chains;
- all four stages and exact source/final count and scope contracts;
- exact HSSD standalone and three-source combined preflight results;
- exact runtime-config identity and `num_workers=1` semantics;
- exact combined manifest identity;
- all four stage-bound launch commands tied to the approved runtime configs,
  combined training manifest, six GPUs, W&B, output/checkpoint roots, and
  monitoring/log semantics;
- exact deployment revision and manifest binding.

The writer revalidates all three public source chains and every combined stage
instead of trusting caller-provided summaries. Empty but structurally shaped
evidence therefore cannot be published.

TDD evidence:

```text
Valid nested report initial RED:       failed
Malformed nested-schema regressions:   20 RED cases, then GREEN
Semantic source/combined sentinels:    RED, then GREEN
Materialization integer acceptance:    3 RED, then 3 GREEN
Node16 + manifest focused final:        163 passed
```

## Final verification

Required CPU-masked planned suite:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
  /opt/conda/envs/pixal3d/bin/python -m pytest -q -p no:cacheprovider \
  tests/multiview/test_generate_node16_deployment_manifest.py \
  tests/multiview/test_node16_training_prepare.py \
  tests/multiview/test_training_source_profiles.py \
  tests/multiview/test_training_materialization.py \
  tests/multiview/test_production_materialization.py \
  tests/multiview/test_training_preflight.py \
  tests/multiview/test_production_preflight.py \
  tests/multiview/test_training_manifest.py \
  tests/multiview/test_multisource_preflight.py \
  tests/multiview/test_training_entrypoint.py
```

Result:

```text
501 passed in 26.91s
```

Static verification:

```text
py_compile:      passed
git diff --check: passed
```

## Preserved contracts and concerns

- ABO remains schema 1.
- 3D-FUTURE and HSSD remain schema 2.
- ABO and 3D-FUTURE retain valid-subset waiver semantics.
- HSSD retains production-gate semantics with no waiver.
- Publication and preparation remain create-only.
- CPU-only preparation guards remain mandatory.
- The deferred unknown-stage `KeyError` and low-level final-inode/fsync minors
  were not expanded into this fix wave.
- Production remains blocked until a fresh review approves the exact final
  commit and the external deployment manifest is generated from that commit.
