from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.queue_resource_coalescence import backfill_decision, with_resource_permit


def decision(wave=None, status='pending', **kwargs):
    fields = dict(wave_stage_running=False, grouped_parent_live=False, grouped_session_live=False)
    fields.update(kwargs)
    return backfill_decision(wave or {'SS-a': 'pending', 'SS-b': 'running'}, status, **fields)


def test_free_cards_can_backfill_while_the_preceding_scientific_wave_is_in_progress():
    assert decision()['allow_lower_priority_GPU_backfill']
    assert decision({'a': 'completed', 'b': 'running'})['allow_lower_priority_GPU_backfill']
    assert decision({'a': 'completed', 'b': 'HELD'}, wave_stage_running=True)['allow_lower_priority_GPU_backfill']


@pytest.mark.parametrize('status', ['completed', 'FAILED', 'HELD'])
def test_drained_wave_stops_new_backfill_without_relabelling_failed_science(status):
    r = decision({'a': 'completed', 'b': status})
    assert not r['allow_lower_priority_GPU_backfill'] and not r['resource_transition_done']
    assert r['wave_scientific_states']['b'] == status


@pytest.mark.parametrize('status', ['completed', 'FAILED', 'HELD'])
@pytest.mark.parametrize('parent,session', [(True, False), (False, True), (True, True)])
def test_a_terminal_record_never_releases_backfill_over_live_owned_processes(status, parent, session):
    r = decision(status=status, grouped_parent_live=parent, grouped_session_live=session)
    assert not r['allow_lower_priority_GPU_backfill'] and not r['resource_transition_done']


@pytest.mark.parametrize('status', ['completed', 'FAILED', 'HELD'])
def test_terminal_and_drained_group_restores_other_work_even_after_group_failure(status):
    r = decision(status=status)
    assert r['allow_lower_priority_GPU_backfill'] and r['resource_transition_done']
    assert r['grouped_scientific_state'] == status


def test_running_group_keeps_cards_coalesced_and_unknown_observations_are_rejected():
    assert not decision(status='running')['allow_lower_priority_GPU_backfill']
    with pytest.raises(ValueError):
        decision(status='probably-finished')
    with pytest.raises(ValueError):
        decision(grouped_session_live=0)


@pytest.mark.parametrize('has_requires', [True, False])
def test_pending_GPU_job_only_gains_one_resource_requirement(has_requires):
    job = dict(name='case', resource='exclusive', command=['python', 'unchanged.py'],
               depends_on=['scientific-gate'], outputs=['actual-output'], priority=-45)
    if has_requires:
        job['requires'] = ['actual-reference']
    original = deepcopy(job)
    result = with_resource_permit(job, '/owned-live-resource/permit')
    assert job == original and result['command'] == original['command']
    assert result['requires'] == job.get('requires', []) + ['/owned-live-resource/permit']
    with pytest.raises(ValueError):
        with_resource_permit(result, '/owned-live-resource/permit')


@pytest.mark.parametrize('damage', ['CPU', 'attached', 'relative-permit'])
def test_CPU_or_attached_process_admission_is_not_changed(damage):
    job = dict(resource='cpu' if damage == 'CPU' else 'exclusive')
    if damage == 'attached':
        job['attached_pid'] = 123
    with pytest.raises(ValueError):
        with_resource_permit(job, 'relative' if damage == 'relative-permit' else '/owned/permit')
