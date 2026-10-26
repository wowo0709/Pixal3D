"""The 277 completed TEST groups keep their three physical producers."""
from copy import deepcopy
import hashlib

import pytest

from experiments.deformnext_retargeting.operational_campaign_entry import SerializedRetirementCampaign
from experiments.deformnext_retargeting.operational_three_campaign_entry import MultiSourceCampaign
from experiments.deformnext_retargeting.operational_three_campaign_reuse import (
    STATUS, physical_case_root, validate_manifest,
)
from experiments.deformnext_retargeting.operational_three_metric_entry import collector_descriptors


def fixture(tmp_path):
    ids = [f'asset-{i:03d}' for i in range(508)]
    aliases = ['zero-source4', 'zero-edited4', 'V1', 'RETARGET', 'SELECTED']
    models = {alias: {'actual_model': alias} for alias in aliases}
    inputs = {key: dict(path=str(tmp_path / key), SHA256='a' * 64,
        receipt=str(tmp_path / 'receipt')) for key in ['parent_freeze', 'control_freeze']}
    context = dict(canonical_local_root=str(tmp_path), campaign_root=str(tmp_path / 'successor'),
        source_SHA256={str(tmp_path / f'source-{i}.py'): 'a' * 64 for i in range(505)},
        source_inputs=inputs)
    roots = [tmp_path / name for name in ['v586', 'v595', 'v611']]
    groups = {}
    for index, sid in enumerate(ids[:277]):
        root = roots[0 if index < 129 else 1 if index < 227 else 2]
        token = hashlib.sha256(sid.encode()).hexdigest()[:24]
        groups[sid] = dict(source_campaign_root=str(root),
            group_audit=dict(path=str(root / 'groups' / token / 'audit.json')),
            cases={alias: dict(audit=dict(path=str(root / 'cases' / token / alias / 'audit.json')),
                worker_terminal=dict(path=str(root / 'workers' / token / alias / 'terminal.json')))
                for alias in aliases})
    manifest = dict(status=STATUS, reused_sample_ids=ids[:277], groups=groups,
        models=deepcopy(models), original_source_SHA256=deepcopy(context['source_SHA256']),
        source_campaigns=[dict(campaign_root=str(root), complete_groups=count)
            for root, count in zip(roots, [129, 98, 50])],
        actual_same_scored_archive_bytes_verified=True,
        no_case_or_terminal_relocated_or_synthesized=True, **inputs)
    return manifest, context, dict(population=dict(sample_ids=ids)), models


@pytest.mark.parametrize('problem', [None, 'duplicate-root', 'missing-root', 'partial',
    'reordered', 'relocated', 'wrong-count', 'changed-source', 'unverified'])
def test_three_original_producers_are_complete_and_disjoint(tmp_path, problem):
    manifest, context, parent, models = deepcopy(fixture(tmp_path))
    sid = parent['population']['sample_ids'][227]
    if problem == 'duplicate-root':
        manifest['source_campaigns'][2]['campaign_root'] = manifest['source_campaigns'][1]['campaign_root']
    elif problem == 'missing-root':
        manifest['groups'][sid]['source_campaign_root'] = str(tmp_path / 'unknown')
    elif problem == 'partial':
        manifest['groups'][sid]['cases'].pop('SELECTED')
    elif problem == 'reordered':
        manifest['reused_sample_ids'].reverse()
    elif problem == 'relocated':
        manifest['groups'][sid]['cases']['V1']['audit']['path'] = str(tmp_path / 'successor/audit.json')
    elif problem == 'wrong-count':
        manifest['source_campaigns'][2]['complete_groups'] = 49
    elif problem == 'changed-source':
        manifest['original_source_SHA256'] = {}
    elif problem == 'unverified':
        manifest['actual_same_scored_archive_bytes_verified'] = False
    if problem is None:
        assert validate_manifest(manifest, context=context, parent=parent, models=models) == parent['population']['sample_ids'][:277]
    else:
        with pytest.raises(ValueError):
            validate_manifest(manifest, context=context, parent=parent, models=models)


def test_reused_case_paths_and_remaining_union(tmp_path, monkeypatch):
    manifest, context, parent, models = fixture(tmp_path)
    for index, root_name in [(0, 'v586'), (129, 'v595'), (227, 'v611'), (277, 'successor')]:
        sid = parent['population']['sample_ids'][index]
        token = hashlib.sha256(sid.encode()).hexdigest()[:24]
        assert physical_case_root(context, manifest, sid, 'SELECTED') == \
            tmp_path / root_name / 'cases' / token / 'SELECTED'
    campaign = MultiSourceCampaign.__new__(MultiSourceCampaign)
    campaign.context = dict(source_inputs=dict(case_reuse_manifest=dict(path='/closed-manifest.json')))
    campaign.reused = manifest['groups']
    campaign.full_ids = parent['population']['sample_ids']
    campaign.ids = campaign.full_ids[277:]
    campaign.models = models
    fresh = {sid: {alias: dict(actual_terminal=f'{sid}-{alias}') for alias in models}
        for sid in campaign.ids}
    monkeypatch.setattr(SerializedRetirementCampaign, 'run',
        lambda self: dict(actual_cases=fresh, assets=231))
    result = campaign.run()
    assert result['assets'] == 508
    assert result['actual_reused_assets'] == 277
    assert result['actual_reused_native_cases'] == 1385
    assert result['actual_new_native_cases'] == 1155
    assert result['actual_cases'][campaign.full_ids[227]] is manifest['groups'][campaign.full_ids[227]]['cases']


def test_metric_adapter_redirects_only_the_cpu_collector():
    jobs = [dict(name=f'gpu-{i}', resource='exclusive', command=['python', '-u', '-m',
        'experiments.deformnext_retargeting.run_selected_final_metric', '--plan', f'/gpu{i}.json'])
        for i in range(11)]
    jobs.append(dict(name='collector', resource='cpu', command=['python', '-u', '-m',
        'experiments.deformnext_retargeting.collect_selected_final_metrics', '--plan', '/cpu.json']))
    adjusted = collector_descriptors(jobs)
    assert adjusted[:11] == jobs[:11]
    assert adjusted[-1]['command'] == ['python', '-u', '-m',
        'experiments.deformnext_retargeting.operational_three_metric_entry',
        '--phase', 'collect', '--plan', '/cpu.json']
    with pytest.raises(ValueError):
        collector_descriptors(jobs[:11])
