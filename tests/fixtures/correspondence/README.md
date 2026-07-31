# Controlled-corruption CLI fixture contract

The CLI tests create all calibrated images, masks, and manifests beneath a
pytest temporary directory. This directory intentionally contains no binary
fixture files.

## Required input layout

```text
<dataset>/
  transforms.json
  ordered_view_0.png
  ordered_view_1.png
  ordered_view_2.png
  ordered_view_3.png
  view_0_foreground.png       # optional when meaningful RGBA alpha exists
```

`transforms.json` must contain an ordered `frames` list. Each selected frame
must name an image inside the manifest directory, provide a finite 4x4
`transform_matrix`, and obtain finite `camera_angle_x` from the frame or the
top level. Foreground selection requires either an in-directory
`foreground_mask_path` or meaningful, nonempty RGBA alpha. The CLI deliberately
does not invoke model-based rembg because this preparation stage is CPU-only.

Run it only with user-selected paths:

```bash
python scripts/prepare_correspondence_corruptions.py \
  --transforms /path/to/transforms.json \
  --output-dir /path/to/output \
  --mesh-scale 1.0 \
  --num-views 4 \
  --corrupt-view-index 1 \
  --seed 42
```

The default view count is 4, the default corrupted view is non-anchor index 1,
and the default seed is 42. Mesh scale is always explicit. Camera distance is
derived from the Euclidean norm of each transform translation.

## Execution boundary

Track A execution is waiting for a user-provided `--transforms` path and
`--output-dir`; the tests and implementation never guess or select a real
dataset. All checkpoint, flow, model, GPU, training, mesh/depth, and 3D stages
are outside this CLI and remain waiting for user-provided paths or checkpoints.
