# Task 4 report — dataset-independent controlled-corruption CLI

Status: DONE

Commit: `feat: add controlled corruption cli` (the Task 4 commit containing
this report)

## Requirements delivered

- Added an explicit, path-agnostic CLI requiring `--transforms`,
  `--output-dir`, and finite positive `--mesh-scale`. It never discovers or
  selects a dataset path.
- Defaults are K=4, non-anchor corruption view index 1, and seed 42. The
  corruption index is constrained to the selected first-K range, and seeds are
  constrained to PyTorch's exact inclusive manual-seed range.
- Reused the Task 1 calibrated loader and foreground precedence without
  changing it. The CLI resolves every selected frame's explicit or meaningful
  RGBA-alpha mask and rejects the rembg fallback, keeping execution strictly
  model-free and CPU-only.
- Reused Task 2's deterministic CPU-only region sampling and C1 color, C2
  procedural-pattern, and C3 deletion generators. All three arms modify only
  the requested view (default index 1) and persist actual changed-pixel oracle
  masks.
- Reused Task 3's atomic/resumable bundle writer, canonical manifest, artifact
  hashes, contact sheet, and validation. Completed runs use a stable ID derived
  from K/view/seed. Failure diagnostics use a reason-digested failed ID so a
  corrected input can subsequently publish the completed run without deleting
  or overwriting failure evidence.
- Invalid calibrated input, escaped input paths, insufficient K, and empty
  foreground masks exit nonzero and publish a failed manifest/contact sheet
  when the explicit output path is writable. Existing hash-mismatched runs fail
  closed and remain untouched.
- Added subprocess-only tests that create four ordered RGBA images, an explicit
  mask, alpha masks, mixed frame/top-level FOV, and distinct finite transforms
  entirely in pytest temporary directories. No binary fixture was committed.
- The E2E runs the CLI twice and verifies untouched resumable output, source and
  camera order, original frame indices, translation-norm distances, all
  artifacts and SHA-256 values, mask provenance, nonempty foreground/oracle
  masks, exact C1-C3 changed-only masks, and the contact sheet.

## Required input and CLI

```text
<dataset>/
  transforms.json
  ordered_view_0.png
  ordered_view_1.png
  ordered_view_2.png
  ordered_view_3.png
  view_0_foreground.png       # optional with meaningful RGBA alpha
```

Every selected frame in `transforms.json` must provide an in-directory
`file_path`, a finite 4x4 `transform_matrix`, and frame- or top-level finite
`camera_angle_x`. Each frame needs an in-directory `foreground_mask_path` or a
nonempty, meaningful RGBA alpha channel.

```bash
python scripts/prepare_correspondence_corruptions.py \
  --transforms /path/to/transforms.json \
  --output-dir /path/to/output \
  --mesh-scale 1.0 \
  --num-views 4 \
  --corrupt-view-index 1 \
  --seed 42
```

## Changed files

- Created `scripts/prepare_correspondence_corruptions.py`.
- Created `tests/multiview/test_correspondence_corruption_cli.py`.
- Created `tests/fixtures/correspondence/README.md`.
- Created this Task 4 report.

## TDD and review evidence

- Parser RED: the initial subprocess suite reported `8 failed` because the CLI
  did not exist. Minimal argparse implementation made those tests GREEN.
- Synthetic E2E RED: `1 failed, 8 deselected`; the parser-only CLI returned 1
  and produced no bundle. C1-C3 orchestration and Task 1-3 integration made it
  GREEN.
- Negative RED: `3 failed, 1 passed, 13 deselected`; invalid inputs had no
  failed bundles, while Task 3 already failed closed on hash tampering. Failure
  publication made the invalid-input tests GREEN.
- Independent review found no Critical issues and two Important edge cases:
  failed-to-fixed recovery and out-of-range torch seeds. Focused RED reproduced
  both (`1 failed, 2 passed` for recovery/boundary selection and `2 failed` for
  out-of-range seeds); the fixes passed recovery `1/1` and seed boundary `4/4`.
  Re-review confirmed both addressed with no remaining Critical or Important
  findings.

## Exact final verification

Focused CLI E2E and parser/negative suite:

```text
$ CUDA_VISIBLE_DEVICES='' /opt/conda/envs/pixal3d/bin/python -m pytest tests/multiview/test_correspondence_corruption_cli.py -q
.....................                                                    [100%]
21 passed in 60.28s (0:01:00)
```

Task 1-4 correspondence integration (before the two review regressions were
added; all covered files were then exercised again by the final full suite):

```text
$ CUDA_VISIBLE_DEVICES='' /opt/conda/envs/pixal3d/bin/python -m pytest tests/multiview/test_correspondence_inputs.py tests/multiview/test_correspondence_corruptions.py tests/multiview/test_correspondence_artifacts.py tests/multiview/test_correspondence_corruption_cli.py -q
102 passed in 49.21s
```

Full CPU-only multiview suite:

```text
$ CUDA_VISIBLE_DEVICES='' /opt/conda/envs/pixal3d/bin/python -m pytest tests/multiview -q
982 passed, 16 skipped, 6 warnings in 135.94s (0:02:15)
```

The six warnings are existing W&B `DeprecationWarning` messages from
`test_wandb_multiview.py`.

Hygiene:

```text
$ git diff --check
(exit 0, no output)
```

## Execution boundary and concerns

- Track A execution is waiting for user-provided `--transforms` and
  `--output-dir` paths. No real dataset was selected or run.
- All checkpoint, flow, model, GPU, training, mesh/depth, and 3D stages remain
  waiting for user-provided paths/checkpoints and are outside this CLI.
- Model-based rembg is deliberately unavailable from this CPU-only command;
  opaque inputs must provide an explicit foreground mask.
- Failed diagnostic bundles are retained when corrected inputs later complete;
  this preserves evidence and avoids mutating Task 3's immutable bundles.
- No known blockers or unresolved correctness concerns.
