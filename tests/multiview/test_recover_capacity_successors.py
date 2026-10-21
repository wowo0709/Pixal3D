from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.recover_capacity_successors import descendants, rebind_exact, replacement_successors


def fixture():
    jobs = [dict(name='old-capacity', outputs=['/old/failure']),
            dict(name='science', depends_on=['old-capacity', 'healthy-native'], outputs=['/new/science']),
            dict(name='eval', depends_on=['science'], outputs=['/new/eval']),
            dict(name='summary', depends_on=['eval', 'healthy-native'], outputs=['/new/summary']),
            dict(name='healthy-native', resource='exclusive'), dict(name='new-capacity')]
    state = {j['name']: dict(status='HELD') for j in jobs}
    state['old-capacity'] = dict(status='FAILED', pid=1, returncode=1)
    state['healthy-native'] = dict(status='running', pid=2, started=10.)
    state['new-capacity'] = dict(status='pending')
    return jobs, state


def test_append_only_aliases_preserve_failure_and_healthy_work_and_artifact_destinations():
    jobs, state = fixture(); original_jobs, original_state = deepcopy(jobs), deepcopy(state)
    new, mapping = replacement_successors(jobs, state, {'old-capacity': 'new-capacity'},
        suffix='-memory-v2', output_exists=lambda _: False)
    assert jobs == original_jobs and state == original_state
    assert len(new) == 3 and mapping['science'] == 'science-memory-v2'
    assert new[0]['depends_on'] == ['new-capacity', 'healthy-native']
    assert new[1]['depends_on'] == ['science-memory-v2']
    assert new[0]['outputs'] == jobs[1]['outputs']
    assert descendants(jobs, {'old-capacity'}) == {'science', 'eval', 'summary'}


@pytest.mark.parametrize('kind', ['running', 'completed', 'already-started', 'existing-output'])
def test_refuses_to_relaunch_started_work_or_overwrite_results(kind):
    jobs, state = fixture()
    if kind in ['running', 'completed']: state['science']['status'] = kind
    if kind == 'already-started': state['science']['pid'] = 77
    with pytest.raises(ValueError):
        replacement_successors(jobs, state, {'old-capacity': 'new-capacity'}, suffix='-memory-v2',
            output_exists=lambda _: kind == 'existing-output')


def test_exact_rebinding_preserves_prefix_strings_and_immutable_pin_values():
    original = dict(activation_job='science', description='science is a substring',
        command=['--config', '/old/config.json'], source_SHA256={'/old/config.json': 'old-digest'},
        input_SHA256={'/old/config.json': 'old-digest'},
        closed_inputs={'/old/config.json': '/old/receipt.json'}, future={'science': ['science']})
    changed = rebind_exact(original, {'science': 'science-memory-v2', '/old/config.json': '/new/config.json'},
        owned_receipts={'/new/config.json': '/new/receipt.json'})
    assert changed['activation_job'] == 'science-memory-v2'
    assert changed['description'] == original['description']
    assert changed['source_SHA256'] == original['source_SHA256']
    assert changed['input_SHA256'] == original['input_SHA256']
    assert changed['closed_inputs'] == {'/new/config.json': '/new/receipt.json'}
    assert changed['future'] == {'science-memory-v2': ['science-memory-v2']}
