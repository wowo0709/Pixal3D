# W&B Anchor-View Naming Design

## Goal

Remove the ambiguity between ground-truth data and the camera used for a
render by naming the first conditioning camera `anchor_view`.

## Naming

Future snapshots use these names:

| Old name | New name |
|---|---|
| `sample_gt_view` | `sample_anchor_view` |
| `sample_gt_gt_view` | `sample_gt_anchor_view` |
| `combined_views` | `combined_anchor_views` |
| `sample_gt_view_{attr}` | `sample_anchor_view_{attr}` |
| `sample_gt_gt_view_{attr}` | `sample_gt_anchor_view_{attr}` |
| `combined_views_{attr}` | `combined_anchor_views_{attr}` |

The subject prefix remains authoritative:

- `sample`: generated model output.
- `sample_gt`: ground-truth target latent.

The view suffix describes only the render camera:

- `anchor_view`: the camera associated with the first conditioning view.
- `multiview`: the existing fixed four-camera diagnostic render.

`sample_gt_multiview` remains unchanged because it unambiguously represents
the ground-truth target rendered from the fixed diagnostic cameras.

## Scope

- Rename generated snapshot dictionary keys.
- Rename local combined snapshot files and W&B media keys.
- Update W&B and snapshot tests to require the new names and reject the old
  ambiguous names.
- Do not emit compatibility aliases; the historical pilot runs remain
  unchanged.
- Do not change rendering, model inputs, camera selection, training, or
  checkpoint behavior.

## Verification

The focused W&B test suite must demonstrate the red-green transition and pass
after implementation. The full multi-view suite should also pass when CUDA is
available; if CUDA initialization is unavailable, report that environmental
collection blocker separately from the CPU-only focused suite.
