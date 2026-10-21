"""Append the faithful fallback only after closed development selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .cascade_view_roles import ClosedInputs
from .historical_target_ray_reproduction import stage_jobs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '' or sha(args.plan) != args.plan_sha256:
        raise ValueError('pinned CPU-only historical activation required')
    plan = json.loads(args.plan.read_text())
    os.sched_setaffinity(0, set(plan['CPU_affinity']))
    os.nice(10)
    if set(os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(',')) != set(map(str, plan['assigned_GPUs'])):
        raise ValueError('current dispatcher assignment must equal the explicit plan')
    base = Path(plan['canonical_local_root'])
    closed = ClosedInputs(base)
    closed.check(args.plan, args.plan.parent/'file4-closure.json')
    for path, digest in plan['source_SHA256'].items():
        if sha(path) != digest:
            raise ValueError('historical activation source changed: '+path)
    closed.check(plan['selection'], plan['selection_receipt'])
    closed.check(plan['templates_index'], plan['templates_receipt'])
    selection = json.loads(Path(plan['selection']).read_text())
    index = json.loads(Path(plan['templates_index']).read_text())
    new = stage_jobs(plan, selection, index)
    queue_path = Path(plan['queue_path'])
    before = queue_path.read_bytes()
    jobs = json.loads(before)
    if {j['name'] for j in jobs} & {j['name'] for j in new}:
        raise FileExistsError('preserve previous historical attempts')
    out = Path(plan['output'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'queue-before.json').write_bytes(before)
    after = (json.dumps(jobs+new, indent=2)+'\n').encode() if new else before
    (out/'queue-after.json').write_bytes(after)
    (out/'appended-jobs.json').write_text(json.dumps(new, indent=2)+'\n')
    report = dict(status='ACTIVATED_FAITHFUL_TARGET_RAY_FALLBACK' if new else 'SKIPPED_WITH_SUPPORTED_SOURCE_RAY_CANDIDATE',
        source_selection_status=selection['status'], selected_source_candidate=selection['selected_candidate'],
        appended_jobs=len(new), original_descriptors_unchanged=True,
        same_training_seed101_reproduction_not_independent_seed_confirmation=True,
        new_disposable_DDP2_capacity_and_both_rank_first97_exact_required=bool(new),
        heldout_test_predictions_read=False, model_adopted=False, full_goal_complete=False)
    (out/'activation.json').write_text(json.dumps(report, indent=2)+'\n')
    manifest = out/'activation-files.json'
    manifest.write_text(json.dumps([str(p.relative_to(base)) for p in out.iterdir() if p != manifest], indent=2)+'\n')
    subprocess.run([plan['python'], plan['canonical_closer'], '--manifest', str(manifest),
        '--receipt', str(Path(plan['activation_receipt']).relative_to(base)),
        '--scope', 'Actual conditional faithful target-patch/R96 activation after development-only source-ray decision; native entrypoints and full matched480 budget preserved'], check=True)
    if queue_path.read_bytes() != before:
        raise RuntimeError('concurrent queue writer: preserve candidate and current queue without overwrite')
    if new:
        temporary = queue_path.with_suffix('.historical-target-ray.tmp')
        with temporary.open('xb') as stream:
            stream.write(after); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, queue_path)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
