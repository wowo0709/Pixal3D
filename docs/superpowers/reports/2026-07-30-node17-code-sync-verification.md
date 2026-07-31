# Node17 code synchronization verification

Date: 2026-07-30 UTC

Verified code revision:
`cdfc7a2a4a9795d1dd6dc71491f6f38327c48b88`
(`fix: narrow CPU-only collection preload`).

Tasks 1–5 end at `8c5038b`; the verified revision adds only test-harness,
test synchronization, regression, and verification-evidence changes.

## CPU-only collection correction

The unmodified full-suite command failed consistently while collecting
`test_dataset_collation.py`, `test_inference_manifest.py`, and
`test_pipeline_inputs.py`. All three traces converged on the installed
`flex_gemm` autotune import calling `torch.cuda.get_device_name()` while
`CUDA_VISIBLE_DEVICES` was empty.

The correction is test-harness-only:

- `tests/multiview/conftest.py` temporarily returns the supported static
  device name `A100` only while `pytest_configure` imports the known
  `flex_gemm.ops.grid_sample` dependency and only when
  `CUDA_VISIBLE_DEVICES=""`;
- a `finally` block restores the exact original
  `torch.cuda.get_device_name` callable immediately after that import,
  before pytest begins normal test-module collection;
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

Formal review found that the initial hook kept the substitute active for the
whole collection phase. A regression then loaded the harness against an
unrelated test module whose top-level code calls
`torch.cuda.get_device_name()`.

Fix-round RED:

```text
1 failed in 2.92s
```

The unrelated probe incorrectly collected with return code 0, proving the
original hook masked it. After narrowing the substitution to the one
dependency import, the full guard suite produced:

```text
3 passed in 8.93s
```

The regressions now prove the known flex-gemm-dependent module collects,
unrelated collection-time CUDA access still raises, and test execution sees
PyTorch's real callable.

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

Formal review required the two remaining test-only differences to be removed.
`test_wandb_multiview.py` and `test_projection_geometry.py` are now
byte-for-byte equal to the current Node16 capture. The complete fresh 18-file
checksum record is:

| Path | Local SHA-256 | Node16 SHA-256 | Mode/result |
| --- | --- | --- | --- |
| `configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json` | `9b11d67b03bde579583b849eaac7349762107c0ef5cb239dd1abed7027d8fb62` | `9b11d67b03bde579583b849eaac7349762107c0ef5cb239dd1abed7027d8fb62` | raw + JSON: match |
| `configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json` | `e29bb5b35db9d3b7f893e6b27ca775449fef4b6c818616440c8954617533cd05` | `e29bb5b35db9d3b7f893e6b27ca775449fef4b6c818616440c8954617533cd05` | raw + JSON: match |
| `configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json` | `98c57360525f28e6008d178d2b63dd0589ab8e37976f2aa1e4883ea3b86f1053` | `98c57360525f28e6008d178d2b63dd0589ab8e37976f2aa1e4883ea3b86f1053` | raw + JSON: match |
| `configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json` | `a30a44a2b980650846250d3e4ba537a3cd3a65ed07383be7de3b7c9a7d99d978` | `a30a44a2b980650846250d3e4ba537a3cd3a65ed07383be7de3b7c9a7d99d978` | raw + JSON: match |
| `pixal3d/trainers/basic.py` | `026809596b74e7e7dd9691379bb87acd8b9facc3a3faac0ba8335a31c449d084` | `8f1ca6f4c5818c9e331f833b478b411d0714414b8ab8d08ad8958018be409f1e` | AST: match (format differs) |
| `tests/multiview/test_configs.py` | `75e09eb2096b57fcfae7cbeef94bce2d2bb08291dcb96f1a7a39c57440e434e9` | `4e6fe6fe5b527e95988674bbe186be01874e07e38ece59cbdf5785dae6b2d547` | approved policy assertions, including `1000/1000/3` |
| `tests/multiview/test_train_smoke_override.py` | `a6c92927cd57a8b808a7f6b6622b6299b42e0c5f7264cd99a2f48e5512bfa372` | `a6c92927cd57a8b808a7f6b6622b6299b42e0c5f7264cd99a2f48e5512bfa372` | raw: match |
| `tests/multiview/test_wandb_multiview.py` | `4fe10ce69725440975114c7925545c5f99b612fc79d1552b42e5cf4e08bc60e9` | `4fe10ce69725440975114c7925545c5f99b612fc79d1552b42e5cf4e08bc60e9` | raw: match |
| `pixal3d/datasets/flexi_dual_grid.py` | `bd094f88f7878df6e43b1cdb5ba593c11a536b68fbe3cd76378052fb286df9c4` | `716d645deba17beaeaa788f6e7c81a7a4c3ecdbdff5d9682a7c8f8283fd2a3a5` | AST: match (format differs) |
| `pixal3d/datasets/sparse_structure_latent.py` | `dc34efb3a4e2d9fab4743e4a3da7cd420051e8753b4c115c5ef1efdd4d3959aa` | `17ebf1571a2a46403f85ad65b6cb306b0b8be0b142bd3cce4f9be12eeca9bd9c` | AST: match (format differs) |
| `pixal3d/datasets/sparse_voxel_pbr.py` | `5acbb57e148d125c4ad99344df346d770a2a68e41ceb44e403dc14b0be27cdc3` | `dcd2eab1f9cdcefb85cb3cb8784111d8c3a9a46f7878df0c42f34ef2a6f508b6` | AST: match (format differs) |
| `pixal3d/datasets/structured_latent.py` | `6f8d315c8612f991ca1285777a23142cf0daab2f110ce06d4d815a3067ee7b22` | `6ad1d32c0884ba7a3adfccc970d391893421cca0cd2478fdd8a64b13e052086f` | AST: match (format differs) |
| `pixal3d/datasets/structured_latent_shape.py` | `7d220f67da08bb24b6416a250a3c750586a02f801c7880d4aedc1481d8820b25` | `f3bce7dd2e772337654521b23e3da5bf63903b1b90e76d8f43ad71e4fcf384f3` | AST: match (format differs) |
| `pixal3d/datasets/structured_latent_svpbr.py` | `86c74dbfc9380ffaa1ee42b1c2ff3e8cc99066a48fa4e71b23fc01ca5aa8ae5e` | `e9924cbc17f0d2e867079f02110a4620097f19e20d4cc07dc56962edd2483faf` | AST: match (format differs) |
| `pixal3d/trainers/vae/pbr_vae.py` | `b595d7fc7f1eef654a85172b2e8c276e81dd7b75167a5e9bbe71c85053355d10` | `a8c13eb65ebcff575d3aa8856da973f20ab5273bce57788b806c18e63e468108` | AST: match (format differs) |
| `pixal3d/trainers/vae/shape_vae.py` | `5bade064c8c27e83ab18b255660439c65e9edcebfff6c29d67e6293d10ad1582` | `acc2f96178e33ae046d4aecd955fa3f27bb437d0c8dabb559acb1a1555716a1d` | AST: match (format differs) |
| `pixal3d/utils/render_utils.py` | `a6061b25839a10f44490922a1008b249b4a989df45dc5833fd59559196ab97f2` | `0001fa701adf788344a769b128ea8016147746ab924831918af172d5622a8726` | AST: match (format differs) |
| `tests/multiview/test_projection_geometry.py` | `59f40e0bc80a7a847f8f69f09816bd7c7a4bea96d2e6dc11da6cc04b02a99458` | `59f40e0bc80a7a847f8f69f09816bd7c7a4bea96d2e6dc11da6cc04b02a99458` | raw: match |

The resulting production gate is 14/14 behavior-identical to Node16. Seven
files match raw bytes, ten production Python files match ASTs, and the sole
remaining non-Node16-exact path is `test_configs.py`, whose approved policy
assertions include `num_workers=2`, `i_print=i_log=10`, and `1000/1000/3`.

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
756 passed, 16 skipped, 6 warnings in 59.94s
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
clean at the verified code revision. After this tracked report was modified,
`git status --short` contained only:

```text
 M docs/superpowers/reports/2026-07-30-node17-code-sync-verification.md
```

## Conclusion

The full CPU-only multi-view suite, static compilation, reviewed Node16
production-behavior comparison, and Git scope/hygiene gates pass. Revision
`cdfc7a2a4a9795d1dd6dc71491f6f38327c48b88` is eligible for the separately
controlled production-data execution phase.
