# Multi-view Training Output Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run every checked-in multi-view fine-tuning stage for 100,000 optimizer iterations and default all run artifacts to collision-free stage directories under `/root/data3/pixal3d/ckpts`.

**Architecture:** Each multi-view config declares a stage-specific `default_output_dir`. A pure `train.resolve_output_dirs` helper gives explicit CLI paths precedence, defaults resume to the resolved output directory, and rejects missing paths before training setup; `train.py` then writes its existing artifact layout below the resolved directory.

**Tech Stack:** Python, argparse, JSON training configs, pytest.

## Global Constraints

- All four multi-view fine-tuning configs set `trainer.args.max_steps` to exactly `100000`.
- `ss64` defaults to `/root/data3/pixal3d/ckpts/ss64`.
- `shape512` defaults to `/root/data3/pixal3d/ckpts/shape512`.
- `shape1024` defaults to `/root/data3/pixal3d/ckpts/shape1024`.
- `pbr1024` defaults to `/root/data3/pixal3d/ckpts/pbr1024`.
- A non-empty CLI `--output_dir` overrides `default_output_dir`.
- A non-empty CLI `--load_dir` overrides the resume default; otherwise `load_dir` equals the resolved `output_dir`.
- A config without `default_output_dir` still requires a non-empty CLI `--output_dir`.
- Do not change model architecture, dataset behavior, optimizer, batch settings, snapshot policy, checkpoint interval, or checkpoint retention.

---

### Task 1: Resolve persistent stage outputs and standardize 100k fine-tuning

**Files:**
- Modify: `train.py`
- Modify: `configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json`
- Modify: `configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json`
- Modify: `configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json`
- Modify: `configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json`
- Modify: `tests/multiview/test_configs.py`
- Modify: `tests/multiview/test_train_smoke_override.py`

**Interfaces:**
- Consumes: top-level config mapping, optional CLI `--output_dir`, and optional CLI `--load_dir`.
- Produces: `resolve_output_dirs(config, cli_output_dir=None, cli_load_dir="") -> tuple[str, str]`.

- [ ] **Step 1: Write failing config-policy assertions**

Extend the four-config policy test with the exact mapping and iteration count:

```python
OUTPUT_DIRS = {
    "ss64": "/root/data3/pixal3d/ckpts/ss64",
    "shape512": "/root/data3/pixal3d/ckpts/shape512",
    "shape1024": "/root/data3/pixal3d/ckpts/shape1024",
    "pbr1024": "/root/data3/pixal3d/ckpts/pbr1024",
}

assert trainer_args["max_steps"] == 100_000
assert config["default_output_dir"] == OUTPUT_DIRS[stage]
```

- [ ] **Step 2: Write failing output-resolution tests**

Import `resolve_output_dirs` from `train` and add these focused behaviors:

```python
def test_config_output_default_also_becomes_resume_default():
    assert resolve_output_dirs(
        {"default_output_dir": "/persistent/ss64"}
    ) == ("/persistent/ss64", "/persistent/ss64")


def test_cli_output_overrides_config_default():
    assert resolve_output_dirs(
        {"default_output_dir": "/persistent/ss64"},
        cli_output_dir="/override/run",
    ) == ("/override/run", "/override/run")


def test_explicit_load_dir_overrides_resume_default():
    assert resolve_output_dirs(
        {"default_output_dir": "/persistent/ss64"},
        cli_load_dir="/recovery/run",
    ) == ("/persistent/ss64", "/recovery/run")


def test_missing_cli_and_config_output_is_rejected():
    with pytest.raises(ValueError, match="output_dir"):
        resolve_output_dirs({})
```

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```bash
conda run --no-capture-output -n pixal3d python -m pytest -q \
  tests/multiview/test_configs.py::test_four_configs_use_batchwide_k_and_matching_checkpoints \
  tests/multiview/test_train_smoke_override.py
```

Expected: collection fails because `resolve_output_dirs` does not exist, or the new assertions fail because the configs still use 1,000,000 iterations and have no default output paths.

- [ ] **Step 4: Implement the pure resolver**

Add this helper near `apply_smoke_overrides` in `train.py`:

```python
def resolve_output_dirs(config, cli_output_dir=None, cli_load_dir=""):
    default_output_dir = config.get("default_output_dir")
    output_dir = cli_output_dir or default_output_dir
    if not output_dir:
        raise ValueError(
            "output_dir is required: pass --output_dir or set default_output_dir in the config"
        )
    load_dir = cli_load_dir or output_dir
    return output_dir, load_dir
```

- [ ] **Step 5: Apply the resolver before training setup**

Make `--output_dir` optional:

```python
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Output directory; defaults to config default_output_dir",
)
```

After parsing and loading the JSON, resolve the paths before calling
`torch.cuda.device_count()`:

```python
try:
    resolved_output_dir, resolved_load_dir = resolve_output_dirs(
        config,
        cli_output_dir=opt.output_dir,
        cli_load_dir=opt.load_dir,
    )
except ValueError as exc:
    parser.error(str(exc))

opt.output_dir = resolved_output_dir
opt.load_dir = resolved_load_dir
opt.num_gpus = torch.cuda.device_count() if opt.num_gpus == -1 else opt.num_gpus
```

After the existing option/config merge, explicitly restore the resolved paths
so an unrelated top-level config key cannot reverse CLI precedence:

```python
cfg.output_dir = resolved_output_dir
cfg.load_dir = resolved_load_dir
```

- [ ] **Step 6: Apply exact config defaults**

Add the appropriate top-level `default_output_dir` to each config and change:

```json
"max_steps": 100000
```

Leave every other training and model value unchanged.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run:

```bash
conda run --no-capture-output -n pixal3d python -m pytest -q \
  tests/multiview/test_configs.py::test_four_configs_use_batchwide_k_and_matching_checkpoints \
  tests/multiview/test_train_smoke_override.py \
  tests/multiview/test_checkpoint_retention.py
```

Expected: all selected tests pass.

- [ ] **Step 8: Run static and config verification**

Run:

```bash
conda run --no-capture-output -n pixal3d python -m compileall -q \
  train.py tests/multiview/test_configs.py tests/multiview/test_train_smoke_override.py
git diff --check
```

Expected: both commands exit with status 0.

- [ ] **Step 9: Commit**

```bash
git add \
  train.py \
  configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json \
  configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json \
  configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  tests/multiview/test_configs.py \
  tests/multiview/test_train_smoke_override.py
git commit -m "config: persist multiview fine-tuning outputs"
```
