"""Intermediate comparisons must bind actual snapshot bytes before reporting trends."""
import hashlib
import json

import pytest

from experiments.deformnext_retargeting.compare_frozen_utility_epochs import closed_snapshot, summarize


def test_snapshot_receipt_binds_all_three_actual_files(tmp_path):
    campaign = tmp_path / 'campaign'
    folder = campaign / 'utility-strength-followup-v1/epoch01'
    folder.mkdir(parents=True)
    files = []
    for name in ['audit.json', 'validation-snapshot.json', 'initial-validation-snapshot.json']:
        path = folder / name
        path.write_text('{}')
        files.append({'relative_path': str(path.relative_to(campaign.parent)),
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': 2})
    path = folder / 'file4-closure.json'
    path.write_text(json.dumps({'status': 'PASS', 'local_equals_actual_file4_SHA256': True, 'files': files}))
    assert closed_snapshot(folder, campaign) == path
    (folder / 'validation-snapshot.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='snapshot bytes'):
        closed_snapshot(folder, campaign)


def test_missing_epoch_or_undeclared_candidate_fails_before_reading_results(tmp_path):
    for epoch, candidate in [(0, 'frozen-utility-c1'), (6, 'utility-scale16-c1'), (1, 'unknown')]:
        with pytest.raises(ValueError):
            summarize(tmp_path, epoch, candidate_name=candidate)
