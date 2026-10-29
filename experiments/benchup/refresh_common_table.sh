#!/usr/bin/env bash
set -u
TASK_REPO=/root/dev/Pixal3D/.worktrees/ray-score-prior-stage-20261019
TASK_RUNTIME=/root/dev/.cache/benchup-runtime
TASK_BENCH=/root/dev/.cache/deformnext-benchmark-runtime
TASK_FIXED="$TASK_RUNTIME/orientation-correct-v2"
TASK_OUTPUT="$TASK_RUNTIME/all-method-common-metrics-v1.json"
TASK_REPORT="$TASK_REPO/docs/superpowers/reports/2026-10-30-benchup-common-table-ko.md"
TASK_PROGRESS="$TASK_REPO/docs/superpowers/reports/2026-10-30-benchup-progress-ko.md"
cd "$TASK_REPO" || exit 1
while :; do
  if OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /opt/conda/envs/pixal3d/bin/python -u -m experiments.benchup.aggregate_common_metrics --benchmark-root "$TASK_BENCH" --manifest "$TASK_RUNTIME/unit-scale-anchor-manifest-v1.jsonl" \
    --method V1 "$TASK_RUNTIME/v1-render-stream-v1" "$TASK_RUNTIME/v1-learned-metrics-v1" \
    --method Selected "$TASK_RUNTIME/selected-render-stream-v1" "$TASK_RUNTIME/selected-learned-metrics-v1" \
    --method MV-zero "$TASK_RUNTIME/mv-zero-render-stream-v1" "$TASK_RUNTIME/mv-zero-learned-metrics-v1" \
    --method Official-RGB1 "$TASK_RUNTIME/official-rgb1-render-stream-v1" "$TASK_RUNTIME/official-rgb1-learned-metrics-v1" \
    --method TRELLIS2 "$TASK_FIXED/trellis2-render-stream-v1" "$TASK_FIXED/trellis2-learned-metrics-v1" \
    --method ReconViaGen "$TASK_FIXED/recon-render-stream-v1" "$TASK_FIXED/recon-learned-metrics-v1" \
    --method ShapeUP "$TASK_RUNTIME/shapeup-render-stream-v1" "$TASK_RUNTIME/shapeup-learned-metrics-v1" \
    --method SpaceControl "$TASK_FIXED/spacecontrol-render-stream-v1" "$TASK_FIXED/spacecontrol-learned-metrics-v1" \
    --method MVAdapter-Recon "$TASK_FIXED/mvadapter-recon-render-stream-v1" "$TASK_FIXED/mvadapter-recon-learned-metrics-v1" \
    --method CorrAdapter-Recon "$TASK_FIXED/corradapter-recon-render-stream-v1" "$TASK_FIXED/corradapter-recon-learned-metrics-v1" \
    --output "$TASK_OUTPUT"; then
    /opt/conda/envs/pixal3d/bin/python -m experiments.benchup.write_common_table --input "$TASK_OUTPUT" --output "$TASK_REPORT" || true
    /opt/conda/envs/pixal3d/bin/python -m experiments.benchup.paired_cluster_bootstrap --runtime "$TASK_RUNTIME" --corrected-root "$TASK_FIXED" --manifest "$TASK_RUNTIME/unit-scale-anchor-manifest-v1.jsonl" --aggregate "$TASK_OUTPUT" --output "$TASK_RUNTIME/paired-cluster-bootstrap-v1.json" || true
    rsync -a --checksum -e 'ssh -o BatchMode=yes -S /tmp/file4-ctl-v2 -p 55555' "$TASK_OUTPUT" "$TASK_REPORT" "$TASK_RUNTIME/paired-cluster-bootstrap-v1.json" rvi@10.0.12.170:/file4/youngwoo/deformation/experiments/retargeting-v1/benchup/reports/ || true
    if [ -f "$TASK_PROGRESS" ]; then
      rsync -a --checksum -e 'ssh -o BatchMode=yes -S /tmp/file4-ctl-v2 -p 55555' "$TASK_PROGRESS" rvi@10.0.12.170:/file4/youngwoo/deformation/experiments/retargeting-v1/benchup/reports/ || true
    fi
  fi
  live=0
  for pid in "$@"; do if kill -0 "$pid" 2>/dev/null; then live=1; fi; done
  [ "$live" -eq 1 ] || exit 0
  sleep 60
done
