# Multi-view Training Output Defaults Design

## Goal

Standardize all four multi-view fine-tuning runs to 100,000 optimizer
iterations and make their default persistent output location
`/root/data3/pixal3d/ckpts`, without allowing stages to overwrite one
another.

## Stage mapping

Each checked-in multi-view config declares `max_steps: 100000` and one
top-level `default_output_dir`:

| Stage | Default output directory |
|---|---|
| `ss64` | `/root/data3/pixal3d/ckpts/ss64` |
| `shape512` | `/root/data3/pixal3d/ckpts/shape512` |
| `shape1024` | `/root/data3/pixal3d/ckpts/shape1024` |
| `pbr1024` | `/root/data3/pixal3d/ckpts/pbr1024` |

The trainer keeps its existing output layout below each stage directory.
For example, SS64 model, EMA, and optimizer checkpoints are written below
`/root/data3/pixal3d/ckpts/ss64/ckpts`, while logs, resolved configs, model
summaries, and any other run artifacts remain below
`/root/data3/pixal3d/ckpts/ss64`.

## Output resolution

`train.py` accepts `--output_dir` as an optional override instead of a
required argument. A small pure helper resolves paths with this precedence:

1. A non-empty CLI `--output_dir`.
2. The config's non-empty `default_output_dir`.
3. If neither exists, fail before initializing CUDA, datasets, models, or
   creating files.

`--load_dir` remains optional. A non-empty explicit value wins; otherwise it
resolves to the final output directory. Consequently, a normal run resumes
from the same stage-specific persistent directory, while remediation and
evaluation commands can still load from a different directory.

Configs outside the four multi-view fine-tuning configs are not assigned a
new default. They retain the requirement to provide `--output_dir`.

## Command-line and config compatibility

The new config field is named `default_output_dir`, rather than `output_dir`,
so it cannot accidentally override an explicit CLI value during the existing
config merge. Existing commands that pass `--output_dir` and `--load_dir`
continue to resolve to those paths.

Smoke overrides continue to change only smoke-related trainer fields.
They do not change the persistent output or resume directory.

## Storage behavior

`train.py` creates the resolved stage directory on node rank 0. Existing
trainer behavior then places checkpoints, logs, samples, model summaries,
command metadata, and resolved config artifacts under that directory.

Checkpoint retention remains unchanged: save every 5,000 iterations, keep the
latest five complete checkpoints, and publish retained checkpoints atomically
inside the destination filesystem.

If `WANDB_DIR` is explicitly set in the environment, the W&B client continues
to honor it for its own cache. Otherwise W&B files default to the resolved
stage output directory.

## Verification

Tests must prove:

- all four configs use exactly `max_steps: 100000`;
- all four configs have the exact stage-to-directory mapping above;
- config defaults work without `--output_dir`;
- an explicit `--output_dir` overrides the config default;
- an explicit `--load_dir` overrides the resume default;
- missing CLI and config output paths fail before side effects;
- the existing smoke override tests and multi-view config policy tests remain
  green.
