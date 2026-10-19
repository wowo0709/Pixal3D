import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting import recover_orthogonal_publications as module


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    campaign = tmp_path / 'campaign'; work = tmp_path / 'work'
    folder = campaign / 'orthogonal-utility-followups-v1'
    log = work / 'failed.log'; log.parent.mkdir(parents=True)
    log.write_text('RuntimeError: concurrent queue change; preserve and reconcile\n')
    save(work / 'queue-state.json', {module.ORIGINAL: {'status': 'FAILED', 'returncode': 1, 'log': str(log)},
                                   module.PREDECESSOR: {'status': 'completed'}})
    save(folder / 'queue-decision-v1.json', {'status': 'ORTHOGONAL_FOLLOWUPS_QUEUED'})
    save(folder / 'confirmation-stage-v1.json', {'status': 'CONFIRMATION_DECISION_QUEUED'})
    archived = folder / 'provenance/development-publication-v1/source/experiments/deformnext_retargeting/utility_followup_wandb.py'
    save(folder / 'provenance/development-publication-v1/file4-closure.json', {'status': 'PASS',
        'local_equals_actual_file4_SHA256': True, 'files': [{'relative_path': str(archived.relative_to(campaign.parent)),
        'sha256': module.sha(Path(module.publication.__file__))}]})
    jobs = [{'name': module.RECOVERY}, {'name': module.PREDECESSOR}]
    for namespace in ['voxel-query-followup-v1', 'utility-softprior-followup-v1']:
        root = campaign / namespace
        root.mkdir()
        name = module.publication.profile(root)
        jobs += [{'name': 'close-fixed-final-' + name}, {'name': 'close-value-address-' + name}]
    save(work / 'queue-jobs.json', jobs)
    return SimpleNamespace(campaign_root=campaign, work_root=work, previous_work=tmp_path / 'previous')


def test_recovers_only_missing_eight_report_jobs_and_preserves_failed_history(tmp_path):
    args = fixture(tmp_path)
    state_before = (args.work_root / 'queue-state.json').read_bytes()
    before = module.read(args.work_root / 'queue-jobs.json')
    module.recover(args)
    after = module.read(args.work_root / 'queue-jobs.json')
    assert after[:len(before)] == before and len(after) == len(before) + 8
    assert all(job['resource'] == 'cpu' and '-m' in job['command'] for job in after[len(before):])
    assert (args.work_root / 'queue-state.json').read_bytes() == state_before
    report = module.read(args.campaign_root / 'orthogonal-utility-followups-v1/publication-queue-decision-v1.json')
    assert report['failed_attempt_preserved']['status'] == 'FAILED' and not report['model_adopted']
    with pytest.raises(FileExistsError): module.recover(args)


def test_does_not_rerun_while_the_other_queue_writer_is_active(tmp_path):
    args = fixture(tmp_path)
    state = module.read(args.work_root / 'queue-state.json')
    state[module.PREDECESSOR]['status'] = 'running'
    save(args.work_root / 'queue-state.json', state)
    before = (args.work_root / 'queue-jobs.json').read_bytes()
    with pytest.raises(ValueError, match='completed conflicting writer'): module.recover(args)
    assert (args.work_root / 'queue-jobs.json').read_bytes() == before


def test_existing_publication_jobs_are_not_silently_overwritten(tmp_path):
    args = fixture(tmp_path)
    path = args.work_root / 'queue-jobs.json'
    jobs = module.read(path) + [{'name': 'voxel-query-c1-development-report-render'}]
    save(path, jobs)
    with pytest.raises(ValueError, match='duplicate'): module.recover(args)
    assert module.read(path) == jobs
