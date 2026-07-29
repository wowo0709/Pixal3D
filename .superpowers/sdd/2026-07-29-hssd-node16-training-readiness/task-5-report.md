# Task 5 Report: CPU-Only Node16 Preparation Driver

## Summary

- Added a CPU-only, create-only Node16 preparation driver for ABO,
  3D-FUTURE, HSSD, standalone HSSD validation, and the canonical
  ABO + 3D-FUTURE + HSSD manifest.
- Added conservative disk admission from fully validated source catalogs.
- Added four stage-bound runtime config copies whose only parsed JSON change
  is `trainer.args.num_workers = 1`.
- Added a CLI with dry-run planning and explicit execute mode.
- Added a final immutable JSON report containing source/stage counts, source
  and union-scope digests, eligibility exclusion counts, artifact digests,
  disk evidence, runtime semantics, and launch commands.
- Did not run real Node16 materialization, a model, a trainer, or W&B.

## Files

- `data_toolkit/pipeline/node16_training_prepare.py`
- `scripts/prepare_node16_training.py`
- `tests/multiview/test_node16_training_prepare.py`
- `.superpowers/sdd/2026-07-29-hssd-node16-training-readiness/task-5-report.md`

No Task 1–4 implementation, production artifact, plan, spec, runbook, model,
trainer, or W&B file was modified.

## RED Evidence

Initial Task 5 collection RED:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/multiview/test_node16_training_prepare.py

ERROR tests/multiview/test_node16_training_prepare.py
ImportError: cannot import name 'node16_training_prepare'
1 error in 0.13s
```

The new module did not exist, so the desired public interface could not be
imported.

The ABO publication regression was then observed RED:

```text
CUDA_VISIBLE_DEVICES="" ... pytest -q \
  tests/multiview/test_node16_training_prepare.py::\
test_publish_abo_uses_existing_fixed_count_publisher

1 failed in 2.34s
```

The failure proved that ABO was incorrectly routed to the generic schema-2
observed-count publisher. The implementation now composes the existing ABO
fixed-count schema-1 publisher, while 3D-FUTURE and HSSD continue to use the
source-aware schema-2 publisher.

Create-only hardening was observed RED:

```text
4 failed, 14 deselected in 0.28s
```

The four failures covered partial runtime configs, a partial combined-output
root, a partial evidence root, and absent combined union-scope evidence.

Final-report rerun behavior was observed RED:

```text
1 failed in 0.38s
```

A valid rerun changed only current free-space values and source `reused`
flags, but the byte-only report check rejected it. The final implementation
keeps the first report immutable, accepts those historical-field differences,
and still rejects any path, setting, count, scope, or artifact digest change.

## GREEN Evidence

Focused Task 5 verification:

```text
CUDA_VISIBLE_DEVICES="" \
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/multiview/test_node16_training_prepare.py

19 passed in 1.30s
```

Required Task 5 plus adjacent Task 1–4 CPU-masked verification:

```text
CUDA_VISIBLE_DEVICES="" \
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/multiview/test_node16_training_prepare.py \
  tests/multiview/test_training_source_profiles.py \
  tests/multiview/test_training_materialization.py \
  tests/multiview/test_training_preflight.py \
  tests/multiview/test_training_manifest.py \
  tests/multiview/test_multisource_preflight.py

154 passed in 15.92s
```

`git diff --check` exited 0. `py_compile` exited 0 for the preparation module
and CLI.

## Create-Only, Partial-State, and Reuse Evidence

The focused suite proves:

- a complete source calls `validate_source_training_data` and then strict
  preflight for every materialized stage before reuse;
- a source root without a validated `training_data.json` is refused before
  catalog loading or materialization;
- every discovered partial source path is reported and remains unchanged;
- partial runtime-config, combined-output, and final-evidence states are
  refused rather than completed or repaired;
- semantically identical existing runtime JSON is accepted without changing
  its bytes;
- mismatched runtime JSON is rejected without replacement;
- an immutable final report can be reused only when all invariant evidence
  still matches;
- no driver or CLI code calls `unlink`, `rmtree`, `remove`, `rename`,
  `replace`, or truncating file modes.

The source order is:

```text
materialize ABO -> publish ABO
materialize 3D-FUTURE -> publish 3D-FUTURE
materialize HSSD -> publish HSSD
preflight standalone HSSD
publish three-source combined input
preflight three-source combined input
write final report
```

## Disk-Estimate Evidence

`test_disk_estimate_counts_repeated_stage_family_consumption` uses literal
pack sizes and proves one source consumes:

```text
4 * common
+ 1 * SS-64
+ 1 * shape-512
+ 2 * shape-1024
+ 1 * PBR-1024
```

The test verifies the sum across all three source profiles, then verifies:

```text
required_bytes = stage_expanded_pack_bytes * 2 + 10 * 1024**3
```

It also proves estimation validates all three catalogs without creating the
local production root. Insufficient free space raises before any
materialization call.

## Runtime Semantic-Diff Evidence

`test_runtime_configs_change_only_num_workers` parses every original and
runtime JSON document, restores runtime `num_workers` to the original value,
and asserts complete parsed-object equality. Therefore every other nested
field is unchanged.

Runtime report validation additionally requires, per stage:

- batch per GPU: `8, 8, 2, 2`;
- batch split: `4, 4, 1, 1`;
- six-GPU global batch: `48, 48, 12, 12`;
- `max_steps = 20000`;
- `i_save = 2000`;
- `max_checkpoints = 5`;
- `i_sample = -1`;
- `num_workers = 1`;
- the matching `multiview_stage`.

## CUDA Import-Order Evidence

The CLI checks that `CUDA_VISIBLE_DEVICES` exists and equals the empty string
before importing the preparation module. Two fresh subprocess tests cover an
unset variable and a non-empty value, install an import hook, and prove that
neither rejected process imports `torch`.

The planning module itself is also torch-free:

```text
{"torch_imported_after_planning_module": False}
```

Execute-time source and combined loader paths lazy-import torch/preflight only
after the same environment gate, assert both
`torch.cuda.is_available() is False` and
`torch.cuda.is_initialized() is False` before configured Dataset import, and
repeat the checks after validation.

## Self-Review

- Confirmed the driver composes existing Task 1–4 catalog, materializer,
  source preflight, publisher, manifest resolver, combined publisher, and
  configured real-loader interfaces rather than reimplementing their source
  validation rules.
- Confirmed ABO uses its fixed-count schema-1 publication path; 3D-FUTURE and
  HSSD use the observed-count schema-2 path.
- Confirmed disk admission occurs before runtime or stage-root creation.
- Confirmed all source, runtime, combined, and report mutations are
  create-only.
- Confirmed complete source reuse includes both immutable chain validation and
  fresh materialized-stage preflight.
- Confirmed the report includes source counts/digests/exclusions, combined
  union counts/digests, artifact hashes, disk evidence, runtime evidence, and
  launch commands.
- Mutation review: removing repeated family consumption, either CUDA guard,
  source-chain validation, stage validation, partial-state refusal, ABO
  publisher selection, runtime semantic equality, orchestration order, or
  report invariant comparison is covered by a focused test.
- The optional pre-commit child reviewer was interrupted after it did not
  return within the controller deadline. The controller will perform the
  independent post-commit review.

## Concerns

- Per the Task 5 restriction, execute mode is covered only with unit and
  synthetic tests. Real Node16 indexes, packs, stage materialization, Dataset
  loading, and report publication were not run.
- A broad `CUDA_VISIBLE_DEVICES="" pytest -q tests/multiview` collection is
  not CPU-clean independently of Task 5. Three unrelated modules
  (`test_dataset_collation.py`, `test_inference_manifest.py`, and
  `test_pipeline_inputs.py`) import `flex_gemm`, which queries a CUDA device
  during collection and raises `RuntimeError: No CUDA GPUs are available`.
  The exact Task 5 required CPU suite passes 154 tests; no out-of-scope
  inference or dataset code was changed to mask the unrelated collection
  behavior.
