"""Execute verified owned checkpoint-audit functions without their idle watcher."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import time
from types import ModuleType

from .semantic_topk_search import write_new


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_prefix(source, kind):
    """Preserve all imports/functions/checks and remove only the final watch loop."""
    tree = ast.parse(source)
    if not tree.body or not isinstance(tree.body[-1], ast.While):
        raise ValueError('owned auditor must end with its watcher loop')
    if any(isinstance(node, ast.While) for node in tree.body[:-1]):
        raise ValueError('unexpected extra top-level watcher loop')
    name = 'epoch' if kind == 'epoch' else 'audit'
    functions = [node for node in tree.body[:-1] if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(functions) != 1:
        raise ValueError('exact owned audit function required')
    digest = hashlib.sha256(ast.get_source_segment(source, functions[0]).encode()).hexdigest()
    return ast.Module(body=tree.body[:-1], type_ignores=tree.type_ignores), digest


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('checkpoint audits must run in an explicitly CPU-only process')
    if sha(args.implementation) != args.expected_sha:
        raise ValueError('owned immutable audit template changed')
    kind = args.kind
    if kind == 'epoch' and args.epoch not in range(1, 6):
        raise ValueError('declared epoch1..5 required')
    cfg = json.loads(args.config.read_text())
    root = args.root.resolve()
    name = cfg['experiment']['arm']
    if (args.config.resolve() != root / 'configs' / (name + '.json')
            or Path(cfg['default_output_dir']).resolve() != root / 'runs_ss64' / name):
        raise ValueError('owned run/config aliases required')
    jobs = json.loads((args.work_root / 'queue-jobs.json').read_text())
    matched = [job for job in jobs if job['name'] == args.job_name]
    if len(matched) != 1:
        raise ValueError('actual training queue entry required')
    job = matched[0]
    command = job['command']
    if (command[command.index('--mode') + 1] != 'train'
            or Path(command[command.index('--root') + 1]).resolve() != root
            or Path(command[command.index('--config') + 1]).resolve() != args.config.resolve()):
        raise ValueError('audit must bind its actual training command')
    folder = root / (f'epoch-audits-v1/{args.job_name}/epoch{args.epoch:02d}' if kind == 'epoch'
                     else f'final-weight-audits-v1/{args.job_name}')
    output = folder / 'once-driver-result-v1.json'
    if output.exists():
        raise FileExistsError('preserve actual completed audit-driver result')
    source = args.implementation.read_text()
    prefix, function_sha = audit_prefix(source, kind)
    own_sha = sha(Path(__file__))
    declared = {
        'status': 'DECLARED_CHECKPOINT_AUDIT_EXECUTION', 'kind': kind,
        'owned_template': str(args.implementation), 'template_SHA256': args.expected_sha,
        'audit_function_SHA256': function_sha, 'driver_SHA256': own_sha,
        'lifecycle': 'Remove final top-level watch loop only;retain every prefix statement/function/check.',
        'overridden_globals': {'N': str(root), 'W': str(args.work_root.resolve())},
        'actual_job': args.job_name, 'GPU_used': False, 'model_quality_claimed': False}
    binding = folder / 'once-driver-binding-v1.json'
    if binding.exists():
        if json.loads(binding.read_text()) != declared:
            raise ValueError('preserve earlier audit attempt; source/lifecycle binding differs')
    else:
        write_new(binding, declared)
    module = ModuleType('owned_saved_checkpoint_audit')
    module.__file__ = str(args.implementation)
    exec(compile(prefix, str(args.implementation), 'exec'), module.__dict__)
    module.N = root
    module.W = args.work_root.resolve()
    while True:
        if (sha(args.implementation) != args.expected_sha or sha(Path(__file__)) != own_sha):
            raise RuntimeError('owned audit source changed during execution')
        try:
            passed = (module.epoch(job, root, cfg, args.epoch) if kind == 'epoch'
                      else module.audit(job, root, cfg))
        except StopIteration:
            # W&B explicit-step data can flush at the following training log.
            # No training restart or acceptance of missing remote data.
            passed = False
            print('AWAITING_ACTUAL_WANDB_STEP', flush=True)
        if passed:
            receipt = folder / 'file4-closure.json'
            record = json.loads(receipt.read_text())
            if record.get('status') != 'PASS' or record.get('local_equals_actual_file4_SHA256') is not True:
                raise ValueError('actual canonical audit closure required')
            binding = folder / 'once-driver-binding-v1.json'
            rows = [row for row in record['files']
                    if row['relative_path'] == str(binding.relative_to(module.D))]
            if len(rows) != 1 or rows[0]['sha256'] != sha(binding):
                raise ValueError('actual file4 receipt must include the lifecycle/source binding')
            write_new(output, {'status': 'PASS_CLOSED_SAVED_CHECKPOINT_AUDIT_ONCE',
                'kind': kind, 'job': args.job_name, 'actual_receipt_SHA256': sha(receipt),
                'audit_function_SHA256': function_sha, 'driver_SHA256': own_sha,
                'template_SHA256': args.expected_sha, 'watcher_loop_not_executed': True,
                'training_restarts': 0, 'GPU_used': False})
            print('PASS_CLOSED_SAVED_CHECKPOINT_AUDIT_ONCE', kind, args.job_name, flush=True)
            return
        time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['epoch', 'final'], required=True)
    parser.add_argument('--implementation', type=Path, required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--job-name', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--epoch', type=int, default=0)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
