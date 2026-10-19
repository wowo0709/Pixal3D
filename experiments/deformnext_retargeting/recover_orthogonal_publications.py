"""Recover a rejected concurrent report append after the other writer finished."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from . import utility_followup_wandb as publication
from .semantic_topk_search import write_new


ORIGINAL = publication.FUTURE_FACTORY
RECOVERY = 'recover-orthogonal-utility-publications-v1'
PREDECESSOR = 'queue-orthogonal-utility-confirmation-stage'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recover(args):
    campaign, work = args.campaign_root, args.work_root
    folder = campaign / 'orthogonal-utility-followups-v1'
    output = folder / 'publication-queue-decision-v1.json'
    if output.exists():
        raise FileExistsError('preserve the actual publication branch decision')
    state = read(work / 'queue-state.json')
    original = state[ORIGINAL]
    if (original['status'] != 'FAILED' or original.get('returncode') != 1
            or 'concurrent queue change; preserve and reconcile' not in Path(original['log']).read_text()
            or state[PREDECESSOR]['status'] != 'completed'):
        raise ValueError('specific completed conflicting writer and rejected append required')
    if (read(folder / 'queue-decision-v1.json')['status'] != 'ORTHOGONAL_FOLLOWUPS_QUEUED'
            or read(folder / 'confirmation-stage-v1.json')['status'] != 'CONFIRMATION_DECISION_QUEUED'):
        raise ValueError('actual launched two-candidate branch required')
    source_receipt = folder / 'provenance/development-publication-v1/file4-closure.json'
    source = read(source_receipt)
    if source.get('status') != 'PASS' or source.get('local_equals_actual_file4_SHA256') is not True:
        raise ValueError('actual publication-source closure required')
    expected = str((folder / 'provenance/development-publication-v1/source/experiments/deformnext_retargeting/utility_followup_wandb.py').relative_to(campaign.parent))
    rows = [row for row in source['files'] if row['relative_path'] == expected]
    if len(rows) != 1 or rows[0]['sha256'] != sha(Path(publication.__file__)):
        raise ValueError('do not change the committed publication implementation during recovery')
    jobs = []
    for namespace in ['voxel-query-followup-v1', 'utility-softprior-followup-v1']:
        root = campaign / namespace
        name = publication.profile(root)
        jobs.extend(publication.publication_jobs(root, ['close-fixed-final-' + name, 'close-value-address-' + name],
            work_root=work, previous_work=args.previous_work))
    path = work / 'queue-jobs.json'
    before = path.read_bytes()
    existing = json.loads(before)
    if (ORIGINAL in {job['name'] for job in existing}
            or RECOVERY not in {job['name'] for job in existing}):
        raise ValueError('retire only the failed descriptor; preserve its historical state and use the named recovery')
    names = [job['name'] for job in existing + jobs]
    if len(names) != len(set(names)) or any(dep not in names for job in jobs for dep in job.get('depends_on', [])):
        raise ValueError('preserve existing jobs; no duplicate or absent publication dependencies')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    for job in jobs:
        job['source_commit'] = commit
        job.setdefault('requires', []).append(str(source_receipt))
    with (work / 'queue-before-recovered-orthogonal-publications-v1.json').open('xb') as stream:
        stream.write(before)
    if path.read_bytes() != before:
        raise RuntimeError('a new concurrent writer appeared; preserve and reconcile')
    temporary = path.with_suffix('.recovered-orthogonal-publications.tmp')
    temporary.write_text(json.dumps(existing + jobs, indent=2) + '\n')
    os.replace(temporary, path)
    write_new(output, {'status': 'ALTERNATIVE_REPORTS_QUEUED', 'jobs': len(jobs),
        'recovered_from_rejected_concurrent_append': True, 'recovery_job': RECOVERY,
        'failed_attempt_preserved': original, 'serialized_after_completed_writer': PREDECESSOR,
        'original_failed_backup_preserved': str(work / 'queue-before-orthogonal-publications-v1.json'),
        'source_commit': commit, 'recovery_source_SHA256': sha(Path(__file__)),
        'publication_source_SHA256': sha(Path(publication.__file__)),
        'all_original_training_and_evaluation_commands_unchanged': True,
        'model_adopted': False, 'full_goal_complete': False})
    print(json.dumps({'status': 'RECOVERED_REJECTED_PUBLICATION_APPEND', 'queued_report_jobs': len(jobs),
        'historical_failed_attempt_retained': True, 'training_restarts': 0}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['campaign-root', 'work-root', 'previous-work']:
        parser.add_argument('--' + name, type=Path, required=True)
    recover(parser.parse_args())


if __name__ == '__main__':
    main()
