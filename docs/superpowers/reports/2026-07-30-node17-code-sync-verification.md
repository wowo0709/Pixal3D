# Node17 code synchronization verification

Date: 2026-07-30 UTC

Verified code revision:
`84781942ac8a01015815c18d77b7f0e4d32d7287`
(`test: support CPU-only multiview collection`).

Tasks 1–5 end at `8c5038b`; the verified revision adds only the CPU-only
pytest collection harness and its regressions.

## CPU-only collection correction

The unmodified full-suite command failed consistently while collecting
`test_dataset_collation.py`, `test_inference_manifest.py`, and
`test_pipeline_inputs.py`. All three traces converged on the installed
`flex_gemm` autotune import calling `torch.cuda.get_device_name()` while
`CUDA_VISIBLE_DEVICES` was empty.

The correction is test-harness-only:

- `tests/multiview/conftest.py` temporarily returns the supported static
  device name `A100` only while pytest collects tests and only when
  `CUDA_VISIBLE_DEVICES=""`;
- a `finally` block restores the exact original
  `torch.cuda.get_device_name` callable before test execution;
- no production CUDA check, training code, or installed dependency changed.

RED:

```text
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_cpu_only_collection_guard.py -x -q

1 failed in 7.93s
```

The subprocess collection regression failed with
`RuntimeError: No CUDA GPUs are available`.

GREEN:

```text
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/multiview/test_cpu_only_collection_guard.py -q

2 passed in 6.46s
```

The second regression confirms test execution sees PyTorch's real
`torch.cuda.get_device_name` function, not the collection substitute.

## Node16 18-file review scope

Node16 was accessed read-only with key-based SSH:

```text
ssh -p 55555 -o BatchMode=yes -o ConnectTimeout=15 \
  youngwoo@n16.unist.info
```

The host identified itself as `rvi-node016`. The reference capture was
`/home/youngwoo/Pixal3D-multiview`; it is a plain file capture rather than a
Git checkout. Exactly the eight Task 1 paths and ten Task 2 paths were
streamed with remote `tar`. No other Node16 path was imported.

For every path, local and Node16 raw SHA-256 values were calculated. JSON
production configs were also decoded and compared structurally. Python
production files were parsed and compared with attribute-free `ast.dump`.
This separates execution behavior from formatting and Node16 trailing
whitespace.

| Scope | Count | Result |
| --- | ---: | --- |
| Production JSON configs | 4 | byte-for-byte and JSON-semantic match |
| Production Python (`BasicTrainer` plus nine camera-call files) | 10 | AST-identical; raw formatting differs |
| `test_train_smoke_override.py` | 1 | byte-for-byte match |
| `test_configs.py` | 1 | deliberate local policy assertions use `1000/1000/3`, plus `num_workers=2` and `i_print=i_log=10` |
| `test_wandb_multiview.py` | 1 | approved Task 1 startup fixture uses `i_sample=1000`; remaining difference is formatting |
| `test_projection_geometry.py` | 1 | approved Task 2 source/API and centered-intrinsics regressions are behavior-equivalent; representation/order differs |

The four production config SHA-256 pairs were identical:

| Path | Local and Node16 SHA-256 |
| --- | --- |
| `configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json` | `9b11d67b03bde579583b849eaac7349762107c0ef5cb239dd1abed7027d8fb62` |
| `configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json` | `e29bb5b35db9d3b7f893e6b27ca775449fef4b6c818616440c8954617533cd05` |
| `configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json` | `98c57360525f28e6008d178d2b63dd0589ab8e37976f2aa1e4883ea3b86f1053` |
| `configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json` | `a30a44a2b980650846250d3e4ba537a3cd3a65ed07383be7de3b7c9a7d99d978` |

The resulting production gate is 14/14 behavior-identical to Node16. The
test-only differences are the approved Task 1/2 regression forms and current
`1000/1000/3` policy, not production drift.

The complete `102a772..8c5038b` Task 1–5 path list was also checked. It
contains no `ops/` path, root-level July 31 document, cache, scratch, or
bytecode file. No such excluded path is tracked or staged.

## Complete CPU-only suite

The exact required command was run fresh at the verified code revision:

```text
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
/opt/conda/envs/pixal3d/bin/python -m pytest tests/multiview -q
```

Result:

```text
755 passed, 16 skipped, 6 warnings in 42.62s
```

The 16 skips are explicitly optional integrations: four released-checkpoint
tests whose checkpoint files are absent and twelve GPU smoke cases for which
CUDA is intentionally hidden. The six warnings are existing W&B SDK
deprecation warnings from the offline serialization test.

## Static and Git hygiene

Commands:

```text
/opt/conda/envs/pixal3d/bin/python -m compileall -q \
  data_toolkit/pipeline pixal3d scripts
git diff --check
git status --short
```

`compileall` exited 0, `git diff --check` was silent, and the worktree was
clean at the verified code revision. After this report was created,
`git status --short` contained only:

```text
?? docs/superpowers/reports/2026-07-30-node17-code-sync-verification.md
```

## Conclusion

The full CPU-only multi-view suite, static compilation, reviewed Node16
production-behavior comparison, and Git scope/hygiene gates pass. Revision
`84781942ac8a01015815c18d77b7f0e4d32d7287` is eligible for the separately
controlled production-data execution phase.
