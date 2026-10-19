"""Add one score-prior control without replacing the five live ray-arm sources."""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import sys

from .compare_ray_group_epochs import verify_matched_recipes
from .orthogonal_utility_followups import close_job, cpu, require_source_contract
from .ray_group_queue import arm_jobs, commands_from_stage, native_proof_folder, read, sha
from .ray_group_study import recipe, source_record, validate_profile
from .semantic_topk_search import write_new


NAME = 'ray-no-score-prior'
MODULE = 'experiments.deformnext_retargeting.ray_group_queue'


def verify_contrast(full, control):
    verify_matched_recipes(full, control)
    expected = copy.deepcopy(full['trainer'])
    expected['args']['deform_conditioner']['task_attention']['score_prior'] = False
    if control['trainer'] != expected:
        raise ValueError('only scalar score_prior may differ from Full')


def verify_lineage(root):
    parent = read(root / 'score-prior-control-v1/source-isolation.json')['parent_components_SHA256']
    current = source_record()['component_SHA256']
    registry = 'experiments/deformnext_retargeting/ray_group_study.py'
    if set(parent) != set(current) or any(parent[k] != current[k] for k in parent if k != registry):
        raise ValueError('tensor model/data/trainer source must match the five-arm parent exactly')
    return current


def prepare(args):
    verify_lineage(args.root)
    full = read(args.root / 'configs/ray-full.json')
    validate_profile(full, args.root)
    control = recipe(full, args.root, NAME)
    verify_contrast(full, control)
    write_new(args.root / 'configs' / (NAME + '.json'), control)
    folder = args.root / 'score-prior-control-v1'
    archived = folder / 'prepared-recipe.json'
    write_new(archived, control)
    write_new(folder / 'prepared-config-contract.json', {
        'runtime_components': [dict(runtime_path=control['config'], archived_path=str(archived), sha256=sha(archived))],
        'scope': 'Prepared recipe archive;startup compares numerical fields after measured batch metadata release.'})
    print('PREPARED_NOT_TRAINED', NAME, flush=True)


def queue(args):
    root, work, previous = args.root, args.work_root, args.previous_work
    extension = root / 'score-prior-control-v1'
    require_source_contract(args.source_contract, args.source_receipt, root.parent)
    current = verify_lineage(root)
    cfg = read(root / 'configs' / (NAME + '.json'))
    validate_profile(cfg, root)
    verify_contrast(read(root / 'configs/ray-full.json'), cfg)
    proof = native_proof_folder(root, NAME)
    binding, native = read(proof / 'ray-source-binding-proof.json'), read(proof / 'native-proof.json')
    if (binding['component_SHA256'] != current or not binding['source_stable_through_execution']
            or native['status'] != 'PASS' or native['optimizer_updates'] != 0
            or any(v != 0 for v in native['condition_max_abs_errors'].values())):
        raise ValueError('actual current-source native/backward proof required')
    path = work / 'queue-jobs.json'
    before = path.read_bytes()
    existing = json.loads(before)
    if any(j['name'] == 'train-' + NAME for j in existing):
        raise FileExistsError('preserve already queued control')
    jobs = arm_jobs(args, NAME)
    startup = 'audit-ray-startup-' + NAME
    folder = root / 'startup-audits-v1' / NAME
    jobs.append(dict(name=startup, resource='cpu', priority=0,
        requires=[str(args.source_receipt), str(root / 'runs_ss64' / NAME / 'runtime_batch.rank1.jsonl')],
        outputs=[str(folder / 'summary.json')], command=cpu() + ['-m',
            'experiments.deformnext_retargeting.audit_ray_group_startup', '--root', str(root),
            '--work-root', str(work), '--arm', NAME, '--expected-sha',
            sha(Path(__file__).with_name('audit_ray_group_startup.py')),
            '--prepared-source-contract', str(extension / 'prepared-config-contract.json')]))
    startup_close = 'close-ray-startup-' + NAME
    files = [folder / name for name in ['summary.json', 'initial-validation-snapshot.json',
        'main-rank0-first3.json', 'main-rank1-first3.json',
        'benchmark-rank0-two-updates.json', 'benchmark-rank1-two-updates.json']]
    jobs.append(close_job(root, root.parent, work, previous, startup_close, [startup], files, folder / 'file4-closure.json'))
    pairs = []
    for n in range(1, 6):
        label = f'compare-ray-full-vs-no-score-prior-epoch{n:02d}'
        output = root / 'epoch-pair-comparisons-v1/ray-full-vs-ray-no-score-prior' / f'epoch{n:02d}.json'
        jobs.append(dict(name=label, resource='cpu', priority=0,
            depends_on=[f'audit-epoch-ray-full-{n:02d}', f'audit-epoch-{NAME}-{n:02d}'],
            requires=[str(args.source_receipt)], outputs=[str(output), str(output.with_suffix('.md'))],
            command=cpu() + ['-m', 'experiments.deformnext_retargeting.compare_ray_group_epochs',
                '--root', str(root), '--epoch', str(n), '--candidate', 'ray-full', '--baseline', NAME,
                '--expected-sha', sha(Path(__file__).with_name('compare_ray_group_epochs.py')),
                '--source-contract', str(args.source_contract), '--source-receipt', str(args.source_receipt)]))
        closed = f'close-ray-full-vs-no-score-prior-epoch{n:02d}'
        pairs.append(closed)
        jobs.append(close_job(root, root.parent, work, previous, closed, [label],
            [output, output.with_suffix('.md')], output.with_suffix('.file4.json')))
    summary = 'summarize-six-ray-arms-fixed-final'
    dependencies = ['close-five-ray-arms-fixed-final', startup_close, *pairs]
    dependencies += [j['name'] for j in jobs if j['name'].startswith(('final-', 'functional-QA-', 'audit-final-'))]
    dependencies += ['close-ray-value-' + name for name in ['ray-full', 'ray-weighted', 'ray-top1',
        'ray-native-spatial', 'ray-no-decoder', NAME]]
    outputs = [root / 'six-arm-fixed-final-report.json', root / 'six-arm-fixed-final-report.md']
    jobs.append(dict(name=summary, resource='cpu', priority=2, depends_on=dependencies,
        requires=[str(args.source_receipt)], outputs=[str(p) for p in outputs],
        command=cpu() + ['-m', MODULE, '--mode', 'summarize', '--root', str(root),
            '--source-contract', str(args.source_contract), '--source-receipt', str(args.source_receipt)]))
    jobs.append(close_job(root, root.parent, work, previous, 'close-six-ray-arms-fixed-final',
        [summary], outputs, root / 'six-arm-fixed-final-file4.json'))
    for item in jobs:
        item.setdefault('requires', []).append(str(args.source_receipt))
    commands_from_stage(jobs, Path(__file__).resolve().parents[2])
    names = [j['name'] for j in existing + jobs]
    if len(names) != len(set(names)) or any(d not in names for j in jobs for d in j.get('depends_on', [])):
        raise ValueError('queue collision or missing dependency')
    write_new(extension / 'queue-declaration.json', dict(status='PREPARED_FOR_QUEUE',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), new_jobs=jobs,
        original_jobs_preserved=len(existing), parent_queue_SHA256=sha(path),
        actual_native_proof=str(proof), source_receipt=str(args.source_receipt),
        budget='FreshSS4210;seed101;raw480;global96;DDP2;micro48;acc1;matching prior flag only',
        training_seed_replication=False, model_adopted=False))
    (extension / 'queue-before.json').write_bytes(before)
    if path.read_bytes() != before:
        raise RuntimeError('concurrent queue mutation;reconcile before writing')
    temporary = path.with_suffix('.score-prior.tmp')
    temporary.write_text(json.dumps(existing + jobs, indent=2) + '\n')
    os.replace(temporary, path)
    write_new(extension / 'queue-installed.json', dict(status='QUEUED_NOT_TRAINED',
        new_jobs=len(jobs), configured_jobs=len(names), original_jobs_preserved=len(existing),
        train_gpus=['0', '1'], source_stage=str(Path(__file__).resolve().parents[2])))
    print('QUEUED', NAME, len(jobs), 'new jobs,total', len(names), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['prepare', 'queue'], required=True)
    p.add_argument('--root', type=Path, required=True)
    for key in ['work-root', 'previous-work', 'source-contract', 'source-receipt', 'epoch-template', 'final-template']:
        p.add_argument('--' + key, type=Path)
    for key in ['epoch-sha', 'final-sha']:
        p.add_argument('--' + key)
    args = p.parse_args()
    prepare(args) if args.mode == 'prepare' else queue(args)


if __name__ == '__main__':
    main()
