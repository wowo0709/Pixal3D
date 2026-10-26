"""Offload only local guide duplicates from the second, fully reused TEST producer.

The scored archives and every case/group audit remain at their original paths.
Each guide is unlinked only after its own NAS byte hash, terminal, group
retirement and live-reader checks pass.  The running successor is untouched.
"""

import argparse
import json
import os
from pathlib import Path
import time

from .cascade_view_roles import ClosedInputs, file_sha256 as sha
from .collect_module_seed_confirmation import read_file, verify_terminal
from .operational_multi_campaign_reuse import load_manifest
from .operational_process_access import process_access_waits
from .operational_reused_guide_retirement import (
    close, reader_snapshot, retained_case_files, validate_candidate,
)
from .retire_selected_final_raw import identity, remote_hashes
from .run_selected_final_campaign import load_population
from .selected_final_metric_inputs import read_pixels
from .selected_pipeline_freeze_v2 import exact
from .selected_pipeline_model_contract import require
from .run_selected_final_campaign import write


SCHEMA = 'second-reused-TEST-guide-local-cache-retirement-v1'
STATUS = 'PASS_ACTUAL_SECOND_REUSED_490_GUIDE_DUPLICATES_OFFLOADED'


def execute(path):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only guide retirement required')
    os.sched_setaffinity(0, {92, 93}); os.nice(10)
    path = Path(path); policy = json.loads(path.read_text())
    require(policy.get('schema') == SCHEMA
        and policy['repo'] == str(Path(__file__).resolve().parents[2]),
        'explicit second-producer policy and actual checkout required')
    base = Path(policy['canonical_local_root']); output = Path(policy['output'])
    require(base.resolve() == base and output.is_absolute() and output.resolve() == output
        and output.is_relative_to(base) and output != base and not output.exists()
        and str(Path(__file__).resolve()) in policy['operational_source_SHA256'],
        'fresh separate output and pinned operational source required')
    check = ClosedInputs(base); check.check(path, path.parent/'file4-closure.json')
    context = read_file(policy['campaign_context'], check)
    require(len(context['source_SHA256']) == 505
        and exact(context['source_SHA256'], policy['scientific_source_SHA256'])
        and context['campaign_root'] != policy['source_campaign_root'],
        'frozen 505 numerical sources and separate live successor required')

    def guard():
        for key in ['scientific_source_SHA256', 'operational_source_SHA256']:
            require(all(sha(p) == d for p, d in policy[key].items()),
                'pinned scientific or retirement source changed')

    guard(); parent, controls, models = load_population(context, check)
    manifest = load_manifest(context, check, parent, models, verify_groups=True)
    producers = {r['campaign_root']: r for r in manifest['source_campaigns']}
    old = Path(policy['source_campaign_root'])
    require(str(old) in producers and producers[str(old)]['complete_groups'] == 98
        and old.resolve() == old and old.is_relative_to(base)
        and old != Path(context['campaign_root'])
        and output != old and not output.is_relative_to(old),
        'only the drained second original producer may be selected')
    selected = {sid: record for sid, record in manifest['groups'].items()
        if record['source_campaign_root'] == str(old)}
    rows = read_file(policy['case_index'], check)
    require(len(selected) == 98 and len(rows) == 490
        and {(r['sample_id'], r['alias']) for r in rows}
            == {(sid, alias) for sid in selected for alias in models}
        and all(r['path'] == str(Path(selected[r['sample_id']]['cases'][r['alias']]['audit']['path'])
            .parent/'geometry/native-shape-guides.pt') for r in rows),
        'exact 98 complete groups and 490 original guides required')
    require(type(policy['batch_size']) is int and 0 < policy['batch_size'] <= 16,
        'bounded NAS verification batches required')

    output.mkdir(parents=True, exist_ok=False); retired = []
    def emit(event):
        with (output/'process-access-events.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')

    with process_access_waits(emit=emit):
        for offset in range(0, len(rows), policy['batch_size']):
            batch = rows[offset:offset+policy['batch_size']]; records = []
            for row in batch:
                sid, alias = row['sample_id'], row['alias']
                record = selected[sid]; group = read_file(record['group_audit'], check)
                descriptor = record['cases'][alias]
                case = read_file(descriptor['audit'], check)
                terminal = read_file(descriptor['worker_terminal'], check)
                verify_terminal(terminal)
                for phase in case['terminals']: verify_terminal(phase)
                target = Path(row['path']); retained_case_files(target)
                before = identity(target)
                digest = check.check(target, target.parent/'file4-closure.json')
                require(identity(target) == before, 'guide changed while hashing')
                receipt = json.loads((target.parent/'file4-closure.json').read_text())
                entry = next(e for e in receipt['files']
                    if e['relative_path'] == str(target.relative_to(base)))
                token = validate_candidate(target, row=row, manifest=manifest,
                    group=group, case=case, terminal=terminal,
                    canonical_entry=entry, digest=digest, readers=[])
                records.append((token, before))
            attempt = 0
            while True:
                paths = [token['path'] for token, _ in records]
                readers, queues = reader_snapshot(policy, paths)
                require(not readers, 'unfinished or actual guide readers remain')
                require(all(identity(token['path']) == before for token, before in records),
                    'guide inode changed before NAS verification')
                remote_rows = [dict(relative_path=str(Path(token['path']).relative_to(base)),
                    sha256=token['sha256'], bytes=token['bytes']) for token, _ in records]
                actual = remote_hashes(policy, remote_rows)
                require(set(actual) == {r['relative_path'] for r in remote_rows}
                    and all(actual[r['relative_path']] == {k:r[k] for k in ['sha256', 'bytes']}
                        for r in remote_rows), 'fresh canonical NAS SHA256/size required')
                folder = output/f'batch-{offset:04d}-attempt-{attempt:04d}'
                folder.mkdir(); intent = folder/'before-retirement'; intent.mkdir()
                write(intent/'audit.json', dict(status='PASS_FRESH_SECOND_REUSED_NAS_GUIDES',
                    selected=[token for token, _ in records], actual_remote=actual,
                    reader_queue_SHA256=queues, scored_archives_retained=True,
                    full_goal_complete=False))
                close(intent, policy, f'batch-{offset:04d}-attempt-{attempt:04d}-intent')
                guard(); readers2, queues2 = reader_snapshot(policy, paths)
                require(not readers2, 'new guide reader prevents local offload')
                if queues2 != queues:
                    write(folder/'audit.json', dict(
                        status='DEFERRED_READER_DESCRIPTOR_CHANGE_NO_GUIDE_UNLINKED',
                        old_queues=queues, new_queues=queues2, actual_deletions=0,
                        full_goal_complete=False))
                    close(folder, policy, f'batch-{offset:04d}-attempt-{attempt:04d}-deferred')
                    attempt += 1; time.sleep(1); continue
                require(all(identity(token['path']) == before for token, before in records),
                    'guide inode changed immediately before unlink')
                for token, before in records:
                    target = Path(token['path'])
                    require(identity(target) == before, 'guide inode changed before own unlink')
                    target.unlink()
                    with (folder/'retired-local-cache.jsonl').open('a') as stream:
                        stream.write(json.dumps(token)+'\n'); stream.flush(); os.fsync(stream.fileno())
                write(folder/'audit.json', dict(
                    status='PASS_SECOND_REUSED_GUIDE_LOCAL_DUPLICATES_OFFLOADED',
                    retired=[token for token, _ in records], canonical_NAS_guides_retained=True,
                    scored_pixels_and_case_audits_retained=True, full_goal_complete=False))
                close(folder, policy, f'batch-{offset:04d}-attempt-{attempt:04d}-result')
                retired.extend(token for token, _ in records)
                print(json.dumps(dict(retired=len(retired), total=490,
                    GiB_retired=sum(token['bytes'] for token in retired)/2**30)), flush=True)
                break

    verified_pixels = 0
    for row in rows:
        descriptor = selected[row['sample_id']]['cases'][row['alias']]
        case = read_file(descriptor['audit'], check)
        require(not Path(row['path']).exists(), 'offloaded guide reappeared')
        for cid, expected in zip([8, 9, 10], case['scored_image_observations']):
            pixel = read_pixels(Path(case['case_output'])/'shared'/f'camera{cid:02d}.npz',
                alias=row['alias'], sid=row['sample_id'],
                cluster=case['source_cluster_id'], check=check)
            require(exact(pixel['pixel_observation'], expected),
                'original scored RGB/GT/alpha changed after guide offload')
            verified_pixels += 1
    require(verified_pixels == 1470, 'all 1,470 original scored archives required')
    guard(); result = dict(status=STATUS, files=len(retired),
        bytes_retired=sum(token['bytes'] for token in retired),
        actual_fresh_NAS_hash_verified=True,
        actual_original_scored_archives_exact_with_guides_absent=verified_pixels,
        canonical_NAS_guides_retained=True, scored_pixels_and_native_case_audits_retained=True,
        original505_unchanged=True, live_successor_context_and_workers_not_modified=True,
        CPU_NN_forward_or_training=0, full_goal_complete=False)
    write(output/'audit.json', result); close(output, policy, 'complete')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    print(json.dumps(execute(parser.parse_args().plan)), flush=True)


if __name__ == '__main__': main()
