from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from experiments.deformnext_retargeting.operational_campaign_entry import SerializedRetirementCampaign
from experiments.deformnext_retargeting.operational_multi_campaign_entry import MultiSourceCampaign
from experiments.deformnext_retargeting.operational_multi_campaign_reuse import (
    STATUS, physical_case_root, validate_manifest,
)
from experiments.deformnext_retargeting.operational_multi_metric_entry import collector_descriptors


def fixture(tmp_path):
    ids = [f'asset-{i:03d}' for i in range(508)]
    models = {a: {'actual_model': a} for a in
        ['zero-source4', 'zero-edited4', 'V1', 'RETARGET', 'SELECTED']}
    inputs = {k: dict(path=str(tmp_path / k), SHA256='a' * 64,
        receipt=str(tmp_path / 'receipt')) for k in ['parent_freeze', 'control_freeze']}
    context = dict(canonical_local_root=str(tmp_path), campaign_root=str(tmp_path / 'new'),
        source_SHA256={str(tmp_path / f'source-{i}.py'): 'a' * 64 for i in range(505)},
        source_inputs=inputs)
    roots = [tmp_path / 'old1', tmp_path / 'old2']
    groups = {}
    for sid in ids[:227]:
        root = roots[0 if len(groups) < 129 else 1]
        token = hashlib.sha256(sid.encode()).hexdigest()[:24]
        groups[sid] = dict(source_campaign_root=str(root),
            group_audit=dict(path=str(root / 'groups' / token / 'audit.json')),
            cases={a: dict(audit=dict(path=str(root / 'cases' / token / a / 'audit.json')),
                worker_terminal=dict(path=str(root / 'workers' / token / a / 'terminal.json')))
                for a in models})
    manifest = dict(status=STATUS, reused_sample_ids=ids[:227], groups=groups,
        models=deepcopy(models), original_source_SHA256=deepcopy(context['source_SHA256']),
        source_campaigns=[dict(campaign_root=str(roots[0]), complete_groups=129),
            dict(campaign_root=str(roots[1]), complete_groups=98)],
        actual_same_scored_archive_bytes_verified=True,
        no_case_or_terminal_relocated_or_synthesized=True, **inputs)
    return manifest, context, dict(population=dict(sample_ids=ids)), models


@pytest.mark.parametrize('problem', [None, 'same-root', 'third-root', 'partial',
    'different-model', 'reordered', 'unverified', 'relocated', 'wrong-count', 'changed-source'])
def test_exact_two_real_producers_only(tmp_path, problem):
    m, c, p, models = deepcopy(fixture(tmp_path))
    sid = p['population']['sample_ids'][129]
    if problem == 'same-root':
        m['source_campaigns'][1]['campaign_root'] = m['source_campaigns'][0]['campaign_root']
    elif problem == 'third-root':
        m['groups'][sid]['source_campaign_root'] = str(tmp_path / 'unknown')
    elif problem == 'partial':
        m['groups'][sid]['cases'].pop('SELECTED')
    elif problem == 'different-model':
        m['models']['SELECTED'] = {}
    elif problem == 'reordered':
        m['reused_sample_ids'].reverse()
    elif problem == 'unverified':
        m['actual_same_scored_archive_bytes_verified'] = False
    elif problem == 'relocated':
        m['groups'][sid]['cases']['SELECTED']['audit']['path'] = str(tmp_path / 'new/audit.json')
    elif problem == 'wrong-count':
        m['source_campaigns'][1]['complete_groups'] = 97
    elif problem == 'changed-source':
        m['original_source_SHA256'] = {}
    if problem is None:
        assert validate_manifest(m, context=c, parent=p, models=models) == p['population']['sample_ids'][:227]
    else:
        with pytest.raises(ValueError):
            validate_manifest(m, context=c, parent=p, models=models)


def test_three_physical_case_roots(tmp_path):
    m, c, p, _ = fixture(tmp_path)
    for index, label in [(0, 'old1'), (129, 'old2'), (227, 'new')]:
        sid = p['population']['sample_ids'][index]
        token = hashlib.sha256(sid.encode()).hexdigest()[:24]
        assert physical_case_root(c, m, sid, 'SELECTED') == tmp_path / label / 'cases' / token / 'SELECTED'


@pytest.mark.parametrize('problem', [None, 'missing-new', 'overlap', 'partial-new'])
def test_full508_union_requires_actual_complete_remaining(tmp_path, monkeypatch, problem):
    m, _, p, models = fixture(tmp_path)
    obj = MultiSourceCampaign.__new__(MultiSourceCampaign)
    obj.context = dict(source_inputs=dict(case_reuse_manifest=dict(path='/real-closed-manifest.json')))
    obj.reused = m['groups']
    obj.full_ids = p['population']['sample_ids']
    obj.ids = obj.full_ids[227:]
    obj.models = models
    new = {sid: {a: dict(actual_terminal=f'real-{sid}-{a}') for a in models} for sid in obj.ids}
    if problem == 'missing-new':
        new.pop(obj.ids[-1])
    elif problem == 'overlap':
        new[obj.full_ids[0]] = deepcopy(new[obj.ids[0]])
    elif problem == 'partial-new':
        new[obj.ids[0]].pop('SELECTED')
    monkeypatch.setattr(SerializedRetirementCampaign, 'run',
        lambda self: dict(actual_cases=new, assets=281))
    if problem is None:
        result = obj.run()
        assert result['assets'] == 508
        assert result['actual_reused_native_cases'] == 1135
        assert result['actual_new_native_cases'] == 1405
        assert result['actual_cases'][obj.full_ids[129]] is m['groups'][obj.full_ids[129]]['cases']
    else:
        with pytest.raises(ValueError):
            obj.run()


def test_only_cpu_collector_command_changes():
    jobs = [dict(name=f'gpu-{i}', resource='exclusive', command=['python', '-u', '-m',
        'experiments.deformnext_retargeting.run_selected_final_metric', '--plan', f'/gpu{i}.json'])
        for i in range(11)]
    jobs.append(dict(name='collector', resource='cpu', command=['python', '-u', '-m',
        'experiments.deformnext_retargeting.collect_selected_final_metrics', '--plan', '/cpu.json']))
    result = collector_descriptors(jobs)
    assert result[:11] == jobs[:11]
    assert result[-1]['command'] == ['python', '-u', '-m',
        'experiments.deformnext_retargeting.operational_multi_metric_entry',
        '--phase', 'collect', '--plan', '/cpu.json']
    with pytest.raises(ValueError):
        collector_descriptors(jobs[:11])
