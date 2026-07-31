# Task 2 report — pure C1–C3 corruption generators

## Delivered

- Added deterministic foreground-region sampling with a local CPU generator,
  foreground containment, validation, and no global RNG advancement.
- Added C1 masked torchvision hue/saturation/brightness adjustment.
- Added deterministic, asset-free C2 `sole`, `stripes`, and `logo` patterns.
- Added C3 deletion using per-channel non-foreground median, with border-median
  fallback when the image has no background pixels.
- All C1–C3 oracle masks are derived from actual per-pixel changes greater than
  `1/255`, intersected with the requested region.
- Exported the Task 2 interfaces from `pixal3d.experiments.correspondence`.

## TDD evidence

Each behavior was introduced as a failing import/behavior test before its
implementation. The observed RED failures were missing imports for, in order,
`sample_foreground_region`, `corrupt_local_color`,
`corrupt_procedural_pattern`, and `corrupt_local_deletion`. Each cycle was then
made GREEN before starting the next behavior group.

## Verification

- Focused corruption tests: `25 passed`.
- Corruption plus correspondence foundation tests: `132 passed`.
- Full CPU-only multiview suite with `CUDA_VISIBLE_DEVICES=''`:
  `928 passed, 16 skipped, 6 warnings`.
- Running the full suite without hiding the available GPU triggers the existing
  CPU-only preflight guard (`torch.cuda.is_available()`); the same preflight
  fails alone in that environment and passes with CPU-only visibility.

## Scope boundary

No real dataset, model, conditioning path, flow checkpoint, GPU computation,
training, mesh/depth generation, or external pattern/logo asset was used.

## Fix round 1 — explicit CPU-only tensor contract

- Added early, named CPU-device validation for `image`, `foreground`, and
  `region`; non-CPU tensors are rejected rather than copied or operated on.
- Added hardware-independent regression coverage with meta tensors for region
  sampling and every tensor argument of C1, C2, and C3.

Exact focused verification:

```text
$ /opt/conda/envs/pixal3d/bin/python -m pytest tests/multiview/test_correspondence_corruptions.py -q
...................................                                      [100%]
35 passed in 1.08s
```
