"""Execute a pinned native historical stage in a fresh owned namespace."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

import psutil

from .cascade_view_roles import ClosedInputs
from .historical_target_ray_reproduction import verify_capacity


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            value.update(block)
    return value.hexdigest()


def execute(plan_path, expected_sha):
    if sha(plan_path) != expected_sha:
        raise ValueError('pinned historical-stage plan changed')
    plan = json.loads(plan_path.read_text())
    base = Path(plan['canonical_local_root'])
    closed = ClosedInputs(base)

    def guard():
        if sha(plan_path) != expected_sha:
            raise ValueError('historical-stage plan changed during execution')
        closed.check(plan_path, plan_path.parent/'file4-closure.json')
        for path, digest in {**plan['source_SHA256'], **plan['input_SHA256']}.items():
            if sha(path) != digest:
                raise ValueError('historical source/input changed: '+path)
        for path, receipt in plan['closed_inputs'].items():
            closed.check(path, receipt)

    guard()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    allowed = os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(',')
    expected = 1 if plan['phase'] == 'evaluation' else 2
    if (set(allowed) != set(map(str, plan['assigned_GPUs'])) or len(visible) != expected
            or len(set(visible)) != expected or not set(visible) <= set(allowed)):
        raise ValueError('exact assigned historical DDP pair or one evaluation GPU required')
    os.sched_setaffinity(0, {cpu for g in visible for cpu in plan['CPU_affinity_per_GPU'][g]})
    out = Path(plan['output'])
    out.relative_to(base)
    out.mkdir(parents=True, exist_ok=False)

    def archive(receipt):
        manifest = out/'closed-files.json'
        files = sorted(p for p in out.rglob('*') if p.is_file()
                       and 'wandb' not in p.relative_to(out).parts
                       and p.name != manifest.name and not p.name.endswith('file4-closure.json'))
        manifest.write_text(json.dumps([str(p.relative_to(base)) for p in files], indent=2)+'\n')
        subprocess.run([plan['python'], plan['canonical_closer'], '--manifest', str(manifest),
            '--receipt', str((out/receipt).relative_to(base)),
            '--scope', 'Actual historical target-patch/R96 '+plan['phase']+'; unchanged native numerical entry and original matched budget'], check=True)

    checkpoint_before = {p: sha(p) for p in plan.get('checkpoint_inputs', [])}
    try:
        if plan['phase'] == 'science':
            capacity = json.loads(Path(plan['capacity_audit']).read_text())
            if capacity['status'] != 'PASS_ACTUAL_TWO_RANK_R96_TARGET_PATCH_CAPACITY_REPRODUCTION':
                raise ValueError('own two-rank first97-loss reproduction required before science')
        env = dict(os.environ, PYTHONPATH=plan['repo'], OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', ATTN_BACKEND='sdpa',
            CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        if plan['phase'] == 'science':
            env.update(WANDB_MODE='online', WANDB_DIR=str(out/'wandb'))
        with (out/'native.log').open('xb') as log:
            process = subprocess.Popen(plan['command'], cwd=plan['repo'], env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            birth = psutil.Process(process.pid).create_time()
            identity = dict(PID=process.pid, birth=birth, physical_GPUs=visible, command=plan['command'])
            (out/'active.json').write_text(json.dumps(identity, indent=2)+'\n')
            code = process.wait()
        (out/'terminal.json').write_text(json.dumps(dict(**identity, returncode=code), indent=2)+'\n')
        guard()
        if code != 0:
            raise RuntimeError('native historical stage exited with code '+str(code))
        native = Path(plan['native_output'])
        phase = plan['phase']
        if phase == 'capacity':
            reports = {rank: json.loads((native/f'joint-profile.rank{rank}.json').read_text()) for rank in [0, 1]}
            golden = {int(rank): json.loads(Path(path).read_text()) for rank, path in plan['golden_profiles'].items()}
            result = verify_capacity(reports, golden, arm=plan['arm'])
        elif phase == 'science':
            progress = json.loads((native/'progress.json').read_text())
            rows = [json.loads(line) for line in (native/'validation.jsonl').read_text().splitlines() if line.strip()]
            if (progress['stage'] != 'completed' or progress['current'] != 480
                    or [r['step'] for r in rows] != [96, 192, 288, 384, 480]
                    or any(len(r['quick_per_asset']) != 32 for r in rows)
                    or not all((native/f'ckpts/{key}_step0000480.pt').is_file()
                               for key in ['denoiser', 'deform_conditioner', 'misc'])):
                raise ValueError('complete original R96-to480 native training and validation required')
            for rank in [0, 1]:
                initial = json.loads((native/f'anchor-stage-initialization.rank{rank}.json').read_text())
                if initial['restored_common_Adam_states'] != 366 or initial['reference_step'] != 96:
                    raise ValueError('same R96 continuation required for both actual ranks')
            result = dict(status='PASS_COMPLETED_FAITHFUL_TARGET_PATCH_R96_SS480',
                arm=plan['arm'], initial_step=96, additional_updates=384, final_step=480,
                global_batch=96, microbatch=48, accumulation=1, world_size=2,
                training_seed=101, initial_quick_metrics=rows[0]['quick_metrics'],
                final_quick_metrics=rows[-1]['quick_metrics'], independent_training_seed=False)
        elif phase == 'evaluation':
            evaluation = json.loads((native/'evaluation.json').read_text())
            if (evaluation['status'] != 'PASS' or evaluation['step'] != 480
                    or evaluation['stage'] != 'ss64' or len(evaluation['per_asset']) != 128
                    or evaluation['optimizer_updates_during_evaluation'] != 0
                    or not evaluation['deterministic_evaluation']['actual_deterministic_algorithms']):
                raise ValueError('original deterministic final128 native evaluation required')
            result = dict(status='PASS_ACTUAL_FAITHFUL_TARGET_PATCH_FINAL128', arm=plan['arm'],
                sampling_noise=plan['noise'], metrics=evaluation['quick_metrics'], optimizer_updates=0)
        else:
            raise ValueError('explicit historical phase required')
        if checkpoint_before != {p: sha(p) for p in checkpoint_before}:
            raise ValueError('historical input checkpoint changed during execution')
        result.update(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source_SHA256=plan['source_SHA256'], input_SHA256=closed.inputs,
            physical_GPUs=visible, no_automatic_retry=True, full_goal_complete=False)
        (out/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
        archive('file4-closure.json')
        return result
    except Exception as error:
        (out/'failure.txt').write_text(traceback.format_exc())
        (out/'failure.json').write_text(json.dumps(dict(status='FAILED_HISTORICAL_TARGET_PATCH_STAGE',
            error=str(error), no_automatic_retry=True, full_goal_complete=False), indent=2)+'\n')
        archive('failure-file4-closure.json')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.plan, args.plan_sha256)), flush=True)


if __name__ == '__main__':
    main()
