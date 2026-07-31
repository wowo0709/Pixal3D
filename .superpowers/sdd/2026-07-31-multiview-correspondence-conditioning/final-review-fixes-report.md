# Final whole-branch review fixes report

## Outcome

Implemented the four requested review fixes without expanding the research
method:

- aggregation is bounded to the 1024 cascade;
- oracle-mask preprocessing ambiguity fails closed;
- non-finite temperatures are rejected; and
- explicit-mean and consensus mask non-leakage evidence is strengthened.

Both CorrAdapter documents now describe the 1024-only contract, oracle-mask
coordinate-frame contract, exact scope of the CPU synthetic equality test, and
linear retained-memory scaling through the forced 1024 fallback.

## RED

The tests were added before production changes.

Command:

```bash
PYTHONPATH=/tmp/pixal3d-flex-gemm-stub.gIVSQA${PYTHONPATH:+:$PYTHONPATH} \
CUDA_VISIBLE_DEVICES='' \
conda run -n pixal3d python -m pytest \
  tests/multiview/test_projection_aggregation.py \
  tests/multiview/test_inference_manifest.py \
  tests/multiview/test_pipeline_inputs.py -q
```

Result:

```text
5 failed, 65 passed in 3.84s
```

The expected failures showed that NaN/+infinity temperatures were accepted,
aggregation-enabled inference selected 1536, and both new pipeline boundary
contracts were absent. Negative infinity already failed through the existing
non-positive check, while the parameterized case records the complete
non-finite contract.

## GREEN: directly affected tests

Command: same as RED.

Result:

```text
70 passed in 3.73s
```

## Focused six-file regression

Command:

```bash
PYTHONPATH=/tmp/pixal3d-flex-gemm-stub.gIVSQA${PYTHONPATH:+:$PYTHONPATH} \
CUDA_VISIBLE_DEVICES='' \
conda run -n pixal3d python -m pytest \
  tests/multiview/test_projection_aggregation.py \
  tests/multiview/test_conditioner.py \
  tests/multiview/test_online_mean.py \
  tests/multiview/test_projection_geometry.py \
  tests/multiview/test_pipeline_inputs.py \
  tests/multiview/test_inference_manifest.py -v
```

Result:

```text
93 passed in 4.57s
```

There were no skips, failures, or warnings.

## Full multi-view CPU regression

This host has no active CUDA/Triton driver. Verification used the established
external import-only stub at
`/tmp/pixal3d-flex-gemm-stub.gIVSQA`. A temporary external `.pth` entry placed
the stub first for child tests that remove `PYTHONPATH`; the stub was not
committed and was removed after verification.

The first full run exposed that a plain path-only `.pth` entry was ordered
after the installed `flex_gemm`, so one subprocess test loaded Triton:

```text
1 failed, 805 passed, 32 skipped, 6 warnings in 58.79s
```

After correcting only the external import-stub ordering, the final command was:

```bash
PYTHONPATH=/tmp/pixal3d-flex-gemm-stub.gIVSQA${PYTHONPATH:+:$PYTHONPATH} \
CUDA_VISIBLE_DEVICES='' \
conda run -n pixal3d python -m pytest tests/multiview -q
```

Final result:

```text
806 passed, 32 skipped, 6 warnings in 58.40s
```

All warnings were the existing `wandb` `DeprecationWarning`.

## Exact regression claims

The explicit sparse-first mean test now compares against the production
`_online_mean_tensor_groups` reducer using nontrivial synthetic bf16 per-view
values, noncontiguous active-coordinate gathering, and a grid-resolution
override. The observed result is bitwise equal for both global and projected
features on CPU. This does not claim bitwise real-DINO or GPU equivalence.

The consensus diagnostic test now compares routing weights with and without an
oracle mask and observes exact equality, in addition to the existing fused
feature check.
