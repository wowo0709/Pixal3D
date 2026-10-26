"""Retire completed DEV geometry raw duplicates after their late metrics finish.

This separate metadata entry does not modify any frozen scientific source.
Original case/terminal guards, actual canonical bytes, both queues, protected
process metadata, and a second reader/inode check precede each local unlink.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess

import psutil

from .cascade_view_roles import ClosedInputs, file_sha256 as sha
from .collect_module_seed_confirmation import verify_terminal
from .dev_cascade_cache_retirement import ARMS, validate_completed_case
from .operational_process_access import process_access_waits
from .retire_selected_final_raw import identity, remote_hashes
from .selected_pipeline_model_contract import require


SCHEMA = 'completed-DEV640-late-geometry-raw-retirement-v1'
TERMINAL = {'completed', 'failed', 'cancelled', 'canceled', 'terminated', 'skipped'}


def validate_evidence(value):
    require(value.get('status') == 'PASS_ACTUAL_COMPLETE_SS7_GEOMETRY12_FULL_FAMILY48_PAPER_EXPORT'
        and value.get('assets') == 128 and value.get('source_clusters') == 93
        and len(value.get('SS_rows', [])) == 7
        and {r['arm'] for r in value['SS_rows']} >= set(ARMS)
        and value.get('all_previous_six_means_exact') is True
        and value.get('actual_new_NN_forward_or_quality_computation') == 0
        and value.get('all48_intervals', {}).get('family_comparisons') == 48,
        'actual completed seven-model late geometry and full48-family evidence required')


def validate_geometry_candidate(path, *, case_root, owned_root, case, geometry,
        replay, canonical_entry, actual_local_SHA256, readers, late_evidence):
    validate_evidence(late_evidence)
    validate_completed_case(case_root=case_root, owned_root=owned_root,
        case=case, geometry=geometry, replay=replay)
    path, root = Path(path), Path(case_root)
    require(path == root / 'geometry/prediction.npz' and path.is_file()
        and not path.is_symlink() and path.resolve() == path and not readers,
        'only the exact completed late-consumed geometry prediction may retire')
    require(canonical_entry.get('sha256') == actual_local_SHA256
        and canonical_entry.get('bytes') == path.stat().st_size,
        'actual local raw must match its canonical archive')
    return dict(path=str(path), bytes=path.stat().st_size, sha256=actual_local_SHA256,
        local_duplicate_only=True, canonical_file4_retained=True)


def pending_reads(jobs, states, folders):
    """Inspect every declared unfinished plan once for the entire candidate set."""
    folders = {str(Path(p)) for p in folders}
    files = {str(Path(p) / 'prediction.npz') for p in folders}
    readers = set()
    for job in jobs:
        if states.get(job['name'], {}).get('status', 'pending').lower() in TERMINAL:
            continue
        command = job.get('command', [])
        documents = [job]
        if '--plan' in command:
            path = Path(command[command.index('--plan') + 1])
            require(path.is_file(), 'unreadable unfinished plan prevents retirement: ' + job['name'])
            documents.append(json.loads(path.read_text()))
        for document in documents:
            metadata_only = document.get('phase') == 'summary' or document.get('worker_mode') == 'summary'
            def scan(value, key=None):
                if isinstance(value, str):
                    if value in files or value in folders and not (metadata_only and key == 'generation'):
                        readers.add(job['name'])
                elif isinstance(value, list):
                    for item in value:
                        scan(item, key)
                elif isinstance(value, dict):
                    for k, item in value.items():
                        scan(k)
                        scan(item, k)
            scan(document)
    return sorted(readers)


def reader_snapshot(policy, folders):
    jobs, states, snapshots = [], {}, {}
    for queue in policy['reader_queues']:
        payload = Path(queue['jobs']).read_bytes()
        rows = json.loads(payload)
        require(len({r['name'] for r in rows}) == len(rows), 'unique actual queue jobs required')
        jobs.extend(rows)
        states.update(json.loads(Path(queue['states']).read_text()))
        snapshots[queue['jobs']] = sha(queue['jobs'])
        require(Path(queue['jobs']).read_bytes() == payload, 'queue changed during reader inspection')
    readers = pending_reads(jobs, states, folders)
    files = {str(Path(p) / 'prediction.npz') for p in folders}
    for process in psutil.process_iter():
        try:
            if any(row.path in files for row in process.open_files()):
                readers.append('actual open process ' + str(process.pid))
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
    return sorted(set(readers)), snapshots


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def close(folder, policy, label):
    folder, base = Path(folder), Path(policy['canonical_local_root'])
    manifest = Path(policy['manifest_directory']) / (policy['name'] + '-' + label + '-files.json')
    write(manifest, [str(p.relative_to(base)) for p in sorted(folder.rglob('*'))
        if p.is_file() and p != folder / 'file4-closure.json'])
    subprocess.run([*policy['closure_command'], '--manifest', str(manifest),
        '--receipt', str((folder / 'file4-closure.json').relative_to(base)),
        '--scope', 'Completed DEV late geometry raw local duplicates only; original canonical arrays, guides, pixels, metrics, data and checkpoints retained'], check=True)


def execute(path):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'explicit CPU metadata-only execution required')
    os.sched_setaffinity(0, {92, 93})
    os.nice(10)
    path = Path(path)
    policy = json.loads(path.read_text())
    require(policy['schema'] == SCHEMA and policy['repo'] == str(Path(__file__).resolve().parents[2]),
        'explicit new late-consumer retirement policy and actual checkout required')
    base, owned, output = (Path(policy[k]) for k in ['canonical_local_root', 'owned_outputs_root', 'output'])
    require(all(p.is_absolute() and p.resolve() == p for p in [base, owned, output])
        and owned != base and owned.is_relative_to(base) and output.is_relative_to(base)
        and not output.is_relative_to(owned) and not owned.is_relative_to(output),
        'separate resolved own DEV and journal roots required')
    require(not output.exists() and len(policy['scientific_source_SHA256']) == 505
        and str(Path(__file__).resolve()) in policy['operational_source_SHA256'],
        'fresh journal, original505 and separate pinned metadata worker required')
    require(type(policy['batch_size']) is int and 0 < policy['batch_size'] <= 32,
        'bounded canonical retirement batch required')
    check = ClosedInputs(base)
    check.check(path, path.parent / 'file4-closure.json')
    def read(p, receipt=None):
        p = Path(p)
        check.check(p, receipt or p.parent / 'file4-closure.json')
        return json.loads(p.read_text())
    evidence = read(policy['late_evidence'])
    validate_evidence(evidence)
    references = read(policy['references'])
    ids = references['sample_ids']
    require(len(ids) == len(set(ids)) == 128 and len(set(references['source_clusters_by_sample'].values())) == 93,
        'same original DEV128/93 references required')
    rows = read(policy['case_index'])
    require(len(rows) == 640 and {(r['arm'], r['sample_id']) for r in rows}
        == {(a, sid) for a in ARMS for sid in ids}, 'exactly five completed controls by original128 assets required')
    def guard():
        for key in ['scientific_source_SHA256', 'operational_source_SHA256']:
            require(all(sha(p) == d for p, d in policy[key].items()), 'pinned retirement or numerical source changed')
    guard()
    jobs, states = [], {}
    for queue in policy['reader_queues']:
        jobs.extend(json.loads(Path(queue['jobs']).read_text()))
        states.update(json.loads(Path(queue['states']).read_text()))
    output.mkdir(parents=True, exist_ok=False)
    def emit(event):
        with (output / 'process-access-events.jsonl').open('a') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n')
    selected = []
    with process_access_waits(emit=emit):
        for offset in range(0, len(rows), policy['batch_size']):
            batch = rows[offset:offset + policy['batch_size']]
            records = []
            for row in batch:
                root = Path(row['case_root'])
                matches = [j for j in jobs if j['name'] == row['case_job']]
                require(len(matches) == 1 and states[row['case_job']]['status'] == 'completed'
                    and states[row['case_job']]['returncode'] == 0,
                    'actual original completed case job required')
                command = matches[0]['command']
                require('--plan' in command and command[command.index('--plan') + 1] == row['case_plan'],
                    'original case plan identity required')
                plan = read(row['case_plan'])
                case = read(root / 'audit.json')
                require(case['SS_arm'] == row['arm'] and case['sample_id'] == row['sample_id']
                    and plan['output'] == str(root)
                    and plan['SS_model_binding'] == case['actual_SS_model_binding']
                    and plan['PBR_model_binding'] == case['actual_PBR_model_binding'],
                    'actual case and original physical model bindings required')
                for terminal in case['actual_owned_terminals']:
                    verify_terminal(terminal)
                geometry = read(root / 'geometry/generation.json')
                replay = read(root / 'PBR/generation.json')
                validate_completed_case(case_root=root, owned_root=owned, case=case, geometry=geometry, replay=replay)
                retained = [root / 'geometry/native-shape-guides.pt',
                    *[root / 'shared' / f'camera{c:02d}.npz' for c in [8, 9, 10]]]
                require(all(p.is_file() and not p.is_symlink() for p in retained), 'retain all actual guides and scored pixels')
                target = root / 'geometry/prediction.npz'
                before = identity(target)
                digest = check.check(target, target.parent / 'file4-closure.json')
                require(identity(target) == before, 'raw changed while hashing')
                receipt = json.loads((target.parent / 'file4-closure.json').read_text())
                entry = next(x for x in receipt['files'] if x['relative_path'] == str(target.relative_to(base)))
                records.append((target, root, case, geometry, replay, entry, digest, before))
            folder = output / f'batch-{offset:04d}'
            folder.mkdir()
            with Path(policy['reader_lock']).open('a+b') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                folders = [r[0].parent for r in records]
                readers, queues = reader_snapshot(policy, folders)
                require(not readers, 'unfinished or actual readers still consume DEV geometry raw')
                tokens = [validate_geometry_candidate(target, case_root=root, owned_root=owned,
                    case=case, geometry=geometry, replay=replay, canonical_entry=entry,
                    actual_local_SHA256=digest, readers=readers, late_evidence=evidence)
                    for target, root, case, geometry, replay, entry, digest, before in records]
                remote_rows = [dict(relative_path=str(Path(t['path']).relative_to(base)),
                    sha256=t['sha256'], bytes=t['bytes']) for t in tokens]
                actual = remote_hashes(policy, remote_rows)
                require(set(actual) == {r['relative_path'] for r in remote_rows}
                    and all(actual[r['relative_path']] == {k:r[k] for k in ['sha256', 'bytes']} for r in remote_rows),
                    'fresh actual remote hashes and sizes must match every selected raw')
                intent = folder / 'before-retirement'
                intent.mkdir()
                write(intent / 'audit.json', dict(status='PASS_ACTUAL_LATE_DEV_RAW_CANONICAL_BEFORE_RETIREMENT',
                    selected=tokens, actual_remote=actual, reader_queue_SHA256=queues,
                    local_inode_identities={str(r[0]):r[-1] for r in records}, full_goal_complete=False))
                close(intent, policy, f'batch-{offset:04d}-before')
                guard()
                readers2, queues2 = reader_snapshot(policy, folders)
                require(not readers2 and queues2 == queues, 'a new reader requires another actual retirement intent')
                require(all(identity(r[0]) == r[-1] for r in records), 'raw inode changed before local removal')
                for token in tokens:
                    target = Path(token['path'])
                    before = next(r[-1] for r in records if r[0] == target)
                    require(identity(target) == before, 'raw inode changed immediately before unlink')
                    target.unlink()
                    with (folder / 'retired-local-cache.jsonl').open('a') as stream:
                        stream.write(json.dumps(token, allow_nan=False) + '\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                write(folder / 'audit.json', dict(status='PASS_COMPLETED_DEV_LATE_GEOMETRY_LOCAL_DUPLICATES_RETIRED',
                    retired=tokens, bytes_retired=sum(t['bytes'] for t in tokens),
                    canonical_file4_retained=True, guides_pixels_metrics_and_checkpoints_retained=True,
                    original505_numerical_sources_unchanged=True, full_goal_complete=False))
                close(folder, policy, f'batch-{offset:04d}-result')
                selected.extend(tokens)
                print(json.dumps(dict(retired=len(selected), total=640,
                    GiB_retired=sum(t['bytes'] for t in selected) / 2**30)), flush=True)
    guard()
    result = dict(status='PASS_ACTUAL_DEV640_LATE_GEOMETRY_RAW_LOCAL_DUPLICATES_RETIRED',
        files=len(selected), bytes_retired=sum(t['bytes'] for t in selected),
        actual_fresh_remote_hash_verified=True, canonical_file4_retained=True,
        original505_numerical_sources_unchanged=True, no_new_NN_or_quality_computation=True,
        full_goal_complete=False)
    write(output / 'audit.json', result)
    close(output, policy, 'complete')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    print(json.dumps(execute(parser.parse_args().plan)), flush=True)


if __name__ == '__main__':
    main()
