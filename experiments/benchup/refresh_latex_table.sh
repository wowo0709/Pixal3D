#!/usr/bin/env bash
set -u
if [ "$#" -eq 0 ]; then echo 'Pass live score and table-monitor PIDs explicitly.' >&2; exit 2; fi
R=/root/dev/.cache/benchup-runtime
REPO=/root/dev/Pixal3D/.worktrees/ray-score-prior-stage-20261019
REPORT="$REPO/docs/superpowers/reports/2026-10-30-benchup-common-table.tex"
cd "$REPO" || exit 1
while :; do
  /opt/conda/envs/pixal3d/bin/python -m experiments.benchup.write_latex_table --input "$R/all-method-common-metrics-v1.json" --output "$REPORT" || true
  rsync -a --checksum -e 'ssh -o BatchMode=yes -S /tmp/file4-ctl-v2 -p 55555' "$REPORT" rvi@10.0.12.170:/file4/youngwoo/deformation/experiments/retargeting-v1/benchup/reports/ || true
  live=0
  for pid in "$@"; do if kill -0 "$pid" 2>/dev/null; then live=1; fi; done
  [ "$live" -eq 0 ] && break
  sleep 60
done
