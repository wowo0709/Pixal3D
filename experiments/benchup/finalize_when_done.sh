#!/usr/bin/env bash
# Finish the currently running, explicitly identified score jobs before final audit.
set -u
if [ "$#" -eq 0 ]; then echo 'Pass live score and table-monitor PIDs explicitly.' >&2; exit 2; fi
R=/root/dev/.cache/benchup-runtime
F="$R/orientation-correct-v2"
B=/root/dev/.cache/deformnext-benchmark-runtime
REPO=/root/dev/Pixal3D/.worktrees/ray-score-prior-stage-20261019
cd "$REPO" || exit 1
while :; do
  live=0
  for pid in "$@"; do if kill -0 "$pid" 2>/dev/null; then live=1; fi; done
  [ "$live" -eq 0 ] && break
  sleep 30
done
# The monitor itself is among the supplied PIDs, so its atomic writes have ended.
bash experiments/benchup/refresh_common_table.sh || exit 1
/opt/conda/envs/pixal3d/bin/python -u -m experiments.benchup.audit_failures --runtime "$R" --output "$R/recorded-failures-audit-v1.json"
TASK_FAILURE_STATUS=$?
rsync -a --checksum -e 'ssh -o BatchMode=yes -S /tmp/file4-ctl-v2 -p 55555' "$R/recorded-failures-audit-v1.json" \
  rvi@10.0.12.170:/file4/youngwoo/deformation/experiments/retargeting-v1/benchup/reports/ || exit 1
[ "$TASK_FAILURE_STATUS" -eq 0 ] || exit "$TASK_FAILURE_STATUS"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 15 /opt/conda/envs/pixal3d/bin/python -u -m experiments.benchup.audit_common_evidence \
  --runtime "$R" --corrected-root "$F" --benchmark-root "$B" \
  --manifest "$R/unit-scale-anchor-manifest-v1.jsonl" --aggregate "$R/all-method-common-metrics-v1.json" \
  --output "$R/common-evidence-audit-v1.json" || exit 1
rsync -a --checksum -e 'ssh -o BatchMode=yes -S /tmp/file4-ctl-v2 -p 55555' "$R/common-evidence-audit-v1.json" \
  rvi@10.0.12.170:/file4/youngwoo/deformation/experiments/retargeting-v1/benchup/reports/ || exit 1
