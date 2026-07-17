#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/dev/Pixal3D/.worktrees/multiview-preprocess
CONFIG="$PROJECT/data_toolkit/configs/multiview_preprocess.yaml"
LOG_ROOT=/root/data2/pixal3d/control/automation
mkdir -p "$LOG_ROOT"
cd "$PROJECT"

run_cli() {
  conda run --no-capture-output -n pixal3d \
    python -m data_toolkit.pipeline.cli "$@"
}

abo_complete() {
  python - <<'PY'
import json
from pathlib import Path
root=Path('/root/data2/pixal3d/control/qualification/smoke/checkpoints/ABO/ABO-00000')
expected={
 'download','stage_raw','dump_mesh','dump_pbr','asset_stats','render_cond',
 'dual_grid_256','voxelize_pbr_256','encode_shape_256','encode_pbr_256',
 'cleanup_voxels_256','dual_grid_512','voxelize_pbr_512','encode_shape_512',
 'encode_pbr_512','cleanup_voxels_512','dual_grid_1024','voxelize_pbr_1024',
 'encode_shape_1024','encode_pbr_1024','cleanup_voxels_1024','encode_ss_64',
 'validate_outputs','build_packs','archive_raw','cleanup_local'}
files=sorted(root.glob('batch[0-2].json'))
if len(files)!=3: raise SystemExit(1)
for path in files:
 d=json.loads(path.read_text())
 if d.get('active_attempt') is not None or not expected.issubset(set(d.get('completed_commands',()))): raise SystemExit(1)
PY
}

echo "[$(date -Is)] waiting for ABO smoke completion" >> "$LOG_ROOT/after_abo_smoke.log"
while ! abo_complete; do sleep 60; done
echo "[$(date -Is)] ABO complete; resuming ObjaverseXL Sketchfab" >> "$LOG_ROOT/after_abo_smoke.log"
run_cli resume --config "$CONFIG" --gate smoke \
  --source ObjaverseXL_sketchfab --shard ObjaverseXL_sketchfab-00000 \
  >> "$LOG_ROOT/objaverse_sketchfab.log" 2>&1

echo "[$(date -Is)] resuming ObjaverseXL GitHub" >> "$LOG_ROOT/after_abo_smoke.log"
run_cli resume --config "$CONFIG" --gate smoke \
  --source ObjaverseXL_github --shard ObjaverseXL_github-00000 \
  >> "$LOG_ROOT/objaverse_github.log" 2>&1

echo "[$(date -Is)] starting HSSD smoke" >> "$LOG_ROOT/after_abo_smoke.log"
run_cli run --config "$CONFIG" --gate smoke \
  --source HSSD --shard HSSD-00000 --count 9 \
  >> "$LOG_ROOT/hssd.log" 2>&1

echo "[$(date -Is)] sequence complete" >> "$LOG_ROOT/after_abo_smoke.log"
