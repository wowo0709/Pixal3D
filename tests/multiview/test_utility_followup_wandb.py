"""Publication routing/conversion tests; synthetic metrics are never model evidence."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting import utility_followup_wandb as module


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))


def test_namespaces_keep_candidate_and_reference_model_identity(tmp_path):
    expected = {'frozen-utility-followup-v1': ('frozen-utility-c1', 7),
        'utility-strength-followup-v1': ('utility-scale16-c1', 8),
        'voxel-query-followup-v1': ('voxel-query-c1', 9),
        'utility-softprior-followup-v1': ('utility-softprior-c1', 9)}
    for namespace, (name, count) in expected.items():
        root = tmp_path / 'campaign' / namespace
        assert module.profile(root) == name
        sources = module.sources(root)
        assert len(sources) == count and sources[name] == (root, name)
    with pytest.raises(ValueError): module.profile(tmp_path / 'unknown')


def test_jobs_require_both_closed_quality_and_closed_address_results(tmp_path):
    root = tmp_path / 'utility-strength-followup-v1'
    jobs = module.publication_jobs(root, ['close-final', 'close-value'], work_root=tmp_path / 'work',
                                   previous_work=tmp_path / 'previous')
    assert len(jobs) == 4 and jobs[0]['depends_on'] == ['close-final', 'close-value']
    assert jobs[1]['depends_on'] == [jobs[0]['name']] and jobs[2]['depends_on'] == [jobs[1]['name']]
    assert all(job['resource'] == 'cpu' and 'gpu' not in job for job in jobs)
    assert 'task-value-address-v1/file4-closure.json' in jobs[0]['requires'][1]
    with pytest.raises(ValueError): module.publication_jobs(tmp_path / 'frozen-utility-followup-v1', [],
        work_root=tmp_path / 'work', previous_work=tmp_path / 'previous')


def test_actual_receipt_binding_checks_content_size_and_duplicates(tmp_path):
    source = tmp_path / 'input.json'; source.write_text('{}')
    receipt = tmp_path / 'receipt.json'
    value = {'status': 'PASS', 'local_equals_actual_file4_SHA256': True,
        'files': [{'relative_path': 'input.json', 'sha256': module.sha(source), 'bytes': 2}]}
    save(receipt, value); module.closed_file(source, receipt, tmp_path)
    value['files'][0]['bytes'] = 3; save(receipt, value)
    with pytest.raises(ValueError): module.closed_file(source, receipt, tmp_path)
    value['files'][0]['bytes'] = 2; value['files'] *= 2; save(receipt, value)
    with pytest.raises(ValueError): module.closed_file(source, receipt, tmp_path)


def test_payload_uses_percentage_points_and_preserves_false_adoption_scope(tmp_path, monkeypatch):
    root = tmp_path / 'campaign/utility-strength-followup-v1'
    input_path = root / 'synthetic.json'; save(input_path, {'synthetic': True})
    scope = {'full_goal_complete': False, 'model_adopted': False}
    contrast = {'gain': .01, 'paired_cluster_bootstrap_ci95': [.002, .02]}
    data = {'scope': scope, 'files': [input_path], 'order': ['synthetic'], 'curves': [],
        'report': {'models': {'synthetic': {'micro_voxel_iou': .5, 'micro_voxel_f1': 2 / 3,
                                          'counts': {'tp': 10, 'fp': 5, 'fn': 5}}},
                   'contrasts': {'baseline': contrast}},
        'diagnostic': {'normal_minus_wrong_address': contrast}}
    save(root / module.FOLDER / 'presentation/summary.json', {'status': 'PASS_COMPLETED_CAMPAIGN_PRESENTATION',
        'scope': scope, 'input_SHA256': {str(input_path): module.sha(input_path)}})
    monkeypatch.setattr(module, 'collect', lambda _: data)
    actual = module.payload(root)
    assert actual['metrics']['development128/candidate-vs-baseline/gain_pp'] == 1.
    assert actual['metrics']['development128/candidate-vs-baseline/ci95_lower_pp'] == .2
    assert actual['metrics']['development128/synthetic/micro_voxel_precision'] == 2 / 3
    assert actual['scope'] == scope
    assert actual['tables']['development128/paired_contrasts']['data'][0] == ['baseline', 1., .2, 2.]


@pytest.mark.parametrize('launched', [False, True])
def test_future_publishing_only_follows_actual_alternative_launch_branch(tmp_path, launched):
    campaign = tmp_path / 'campaign'; work = tmp_path / 'work'
    save(campaign / 'orthogonal-utility-followups-v1/queue-decision-v1.json', {
        'status': 'ORTHOGONAL_FOLLOWUPS_QUEUED' if launched else 'SKIPPED_FOR_STRENGTH_CONFIRMATION',
        'new_training_runs': 2 if launched else 0})
    existing = []
    for name in ['voxel-query-c1', 'utility-softprior-c1']:
        existing += [{'name': 'close-fixed-final-' + name}, {'name': 'close-value-address-' + name}]
    save(work / 'queue-jobs.json', existing)
    module.queue_future(SimpleNamespace(campaign_root=campaign, work_root=work, previous_work=tmp_path / 'previous'))
    result = json.loads((campaign / 'orthogonal-utility-followups-v1/publication-queue-decision-v1.json').read_text())
    actual = json.loads((work / 'queue-jobs.json').read_text())
    assert len(actual) == len(existing) + (8 if launched else 0)
    assert result['status'] == ('ALTERNATIVE_REPORTS_QUEUED' if launched else 'NO_ALTERNATIVE_REPORTS_QUEUED')
