# Task 2 Report — Source-Aware Multi-Index Materialization

## Outcome

Implemented the reusable, source-aware production materialization core while
preserving the historical ABO Python imports, call signatures, CLI defaults,
evidence shape, safety checks, and create-only publication behavior.

Implementation commit:

```text
a41378b feat: support multi-shard training materialization
```

This report is committed separately after the implementation hash is known.

## RED Evidence

The required first focused run was executed before the production module
existed:

```bash
conda run --no-capture-output -n pixal3d \
  python -m pytest tests/multiview/test_training_materialization.py -q
```

Result: exit 2 during collection, with:

```text
ModuleNotFoundError:
No module named 'data_toolkit.pipeline.training_materialization'
1 error in 0.10s
```

Additional red/green cycles covered:

- missing public `materialize_stage` and `materialize_all`;
- missing `--profile` CLI behavior and cross-profile index rejection;
- wrong 3D-FUTURE acceptance metadata;
- absent generic pack-exclusion evidence;
- source-index evidence incorrectly coupled to mapping insertion order;
- a verified catalog being relabeled with a foreign source spec;
- generic stage evidence claiming unobserved final counts for other stages;
- duplicate allowed `(shard_id, batch_id)` identities surviving the early
  provenance check.

Each failure was observed for the intended reason before the corresponding
minimal implementation or hardening change.

## Files and Interfaces

### `data_toolkit/pipeline/training_materialization.py`

Created the reusable core and exported:

- `ProductionSourceSpec`
- `ABO_SOURCE_SPEC`
- `THREED_FUTURE_SOURCE_SPEC`
- `FamilyPack` with `source` and `shard_id`
- `load_source_catalog(spec, prepared_root)`
- `compute_stage_scopes(catalog)`
- `materialize_stage(spec, stage, catalog, output_root)`
- `materialize_all(spec, prepared_root, output_root)`

The 3D-FUTURE profile binds:

- the exact `3D-FUTURE-00000.json` and `3D-FUTURE-00001.json` indexes;
- 20 and 18 exact batches, respectively;
- 9,472 frozen assets;
- candidate stage counts `8495/8513/8495/8495`;
- observed final counts (`fixed_count_contract=None`);
- `valid_subset_user_waiver` and a failed original 90-percent gate.

The catalog loader:

- verifies source, shard, gate, exact batch/family sets, manifest digests,
  pack/manifest agreement, manifests, and pack payloads;
- accepts duplicate batch names across different shards;
- rejects duplicate frozen assets within a shard or across shards;
- validates equal frozen scopes across required families;
- validates the exact expected source candidate counts before returning.

The generic materializer additionally binds a supplied catalog back to the
spec's exact source and unique `(shard_id, batch_id, family)` matrix. Generic
stage evidence records only the current stage's observed count mappings so it
does not claim final counts that have not yet been evaluated. The existing
safe-path, regular-file, member verification, eligibility filtering, cleanup,
locking, and atomic `RENAME_NOREPLACE` publication code was retained.

### `scripts/materialize_multiview_production.py`

Converted to a compatibility wrapper:

- historical imports and single-index ABO functions remain available;
- default CLI profile remains ABO;
- added `--profile {abo,3d-future}`;
- retained repeatable
  `--stage {ss64,shape512,shape1024,pbr1024}`;
- rejects index, prepared-root, or output-root values inconsistent with the
  selected profile;
- routes normal CLI materialization through the source-aware core.

### Tests

`tests/multiview/test_training_materialization.py` now verifies the exact
3D-FUTURE profile, count mode, and acceptance metadata.

`tests/multiview/test_production_materialization.py` now verifies:

- duplicate batch names across shards;
- cross-shard asset overlap rejection;
- wrong source and shard identity;
- missing expected batches;
- manifest digest mismatch;
- source/shard-aware stage and all-stage materialization;
- exact source-index and pack provenance;
- foreign and duplicate catalog identity rejection;
- stage-local observed count evidence;
- profile CLI exposure and cross-profile index rejection.

## Verification

Final focused command:

```bash
conda run --no-capture-output -n pixal3d \
  python -m pytest \
  tests/multiview/test_training_materialization.py \
  tests/multiview/test_production_materialization.py -q
```

Result:

```text
83 passed in 2.71s
```

Final broader CPU regression command:

```bash
conda run --no-capture-output -n pixal3d \
  python -m pytest tests/data_toolkit tests/multiview \
  -q -m "not gpu and not integration"
```

Result:

```text
1097 passed, 36 deselected, 7 warnings in 93.10s
```

The warnings are existing deprecation warnings from `torch.cross` and the
offline W&B serialization test. No production materialization, production
evidence write, CUDA job, external W&B run, or preprocessing-worker action was
performed.

Additional checks:

```bash
git diff --check
python -m py_compile \
  data_toolkit/pipeline/training_materialization.py \
  scripts/materialize_multiview_production.py
```

Both exited 0.

## Self-Review and Independent Review

Self-review checked every Task 2 brief interface and step, path/digest
verification, multi-shard identity ordering, observed-versus-fixed count
semantics, wrapper import compatibility, profile CLI behavior, failure
cleanup, and create-only publication.

An independent read-only reviewer initially identified two Important issues:

1. a caller could relabel verified packs with a different source spec;
2. observed-source evidence claimed candidate-as-final counts for stages not
   yet materialized.

Both were fixed with dedicated red/green regressions. The final reviewer
reported no Critical or Important findings. Its one Minor finding—requiring
the exact expected `(shard_id, batch_id)` identity set exactly once per
family—was also fixed with a dedicated regression before final verification.

## Concerns

No blocking concerns. Task 3 should aggregate the stage-local generic count
mappings across all four materializations and then call
`observed_count_contract` to publish the full source count contract.
