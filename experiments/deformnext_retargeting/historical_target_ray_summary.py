"""Compare faithful target-ray reruns with matched R and their historical runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .cascade_view_roles import ClosedInputs
from .compare_ss_noise_panels import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if os.environ.get('CUDA_VISIBLE_DEVICES', '') or digest(args.plan) != args.plan_sha256:
        raise ValueError('pinned CPU-only reproduction summary required')
    plan = json.loads(args.plan.read_text())
    os.sched_setaffinity(0, set(plan['CPU_affinity']))
    os.nice(10)
    base = Path(plan['canonical_local_root'])
    closed = ClosedInputs(base)
    closed.check(args.plan, args.plan.parent/'file4-closure.json')
    for path, expected in plan['source_SHA256'].items():
        if digest(path) != expected:
            raise ValueError('reproduction summary source changed: '+path)
    records, historical = {}, {}
    for arm, item in plan['models'].items():
        records[arm], historical[arm] = [], []
        for group, destination in [('evaluations', records[arm]), ('historical', historical[arm])]:
            for row in item[group]:
                closed.check(row['path'], row['receipt'])
                value = json.loads(Path(row['path']).read_text())
                if (value['status'] != 'PASS' or value['step'] != 480 or len(value['per_asset']) != 128
                        or value['optimizer_updates_during_evaluation'] != 0):
                    raise ValueError('complete fixed final128x3 historical and new predictions required')
                destination.append(value)
        if len(records[arm]) != 3 or len(historical[arm]) != 3:
            raise ValueError('exact three sampling-noise repeats required')
    contrasts = {}
    for arm, baseline in [('direct-nodq', 'retargeting'), ('direct-dq', 'retargeting'), ('direct-dq', 'direct-nodq')]:
        contrasts[arm+' minus '+baseline] = compare([
            (str(n), records[baseline][n], records[arm][n]) for n in range(3)])
    for arm in records:
        contrasts[arm+' rerun minus historical'] = compare([
            (str(n), historical[arm][n], records[arm][n]) for n in range(3)])
    result = dict(status='PASS_ACTUAL_FAITHFUL_TARGET_PATCH_R96_FINAL128x3',
        contrasts=contrasts, input_SHA256=closed.inputs, training_seed=101,
        matched_total_steps=480, shared_R96_plus384=True,
        independent_training_seed_confirmation=False,
        query_axis_and_warmup_differ_from_source_ray_family=True,
        no_causal_claim_from_cross_family_difference=True,
        full_cascade_and_source_preservation_not_evaluated_by_voxel_metrics=True,
        no_test508_or_model_adoption=True, full_goal_complete=False)
    out = Path(plan['output']); out.mkdir(parents=True, exist_ok=False)
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    manifest = out/'summary-files.json'
    manifest.write_text(json.dumps([str((out/'summary.json').relative_to(base))], indent=2)+'\n')
    subprocess.run([plan['python'], plan['canonical_closer'], '--manifest', str(manifest),
        '--receipt', str((out/'file4-closure.json').relative_to(base)),
        '--scope', 'Actual faithful target-patch/R96 native fixed480 128x3 matched and historical comparisons; no independent seed, test or full-cascade claim'], check=True)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
