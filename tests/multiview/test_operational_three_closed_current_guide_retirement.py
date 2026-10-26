"""The v636 guide procedure must retain the original retirement checks."""

import pytest

from experiments.deformnext_retargeting import operational_closed_current_guide_retirement as original
from experiments.deformnext_retargeting import operational_multi_campaign_reuse as two_source
from experiments.deformnext_retargeting import operational_three_campaign_reuse as three_source
from experiments.deformnext_retargeting import operational_three_closed_current_guide_retirement as adapter


def test_only_manifest_reader_changes_during_existing_retirement(tmp_path, monkeypatch):
    path = tmp_path / 'policy.json'
    path.write_text('{}')
    seen = []

    def procedure(actual_path):
        seen.append((actual_path, original.load_manifest))
        return {'status': 'PASS'}

    monkeypatch.setattr(original, 'execute', procedure)
    assert original.load_manifest is two_source.load_manifest
    assert adapter.execute(path) == {'status': 'PASS'}
    assert seen == [(path, three_source.load_manifest)]
    assert original.load_manifest is two_source.load_manifest


def test_manifest_reader_restored_after_retirement_failure(tmp_path, monkeypatch):
    path = tmp_path / 'policy.json'
    path.write_text('{}')

    def failed_procedure(_):
        assert original.load_manifest is three_source.load_manifest
        raise RuntimeError('preserved retirement failure')

    monkeypatch.setattr(original, 'execute', failed_procedure)
    with pytest.raises(RuntimeError, match='preserved retirement failure'):
        adapter.execute(path)
    assert original.load_manifest is two_source.load_manifest


def test_missing_policy_rejected_before_retirement(tmp_path, monkeypatch):
    monkeypatch.setattr(original, 'execute', lambda _: pytest.fail('retirement started'))
    with pytest.raises(ValueError, match='explicit closed guide retirement policy'):
        adapter.execute(tmp_path / 'absent.json')
