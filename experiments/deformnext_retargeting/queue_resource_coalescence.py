"""Admission policy for draining backfill before an unchanged grouped GPU job.

This changes resource availability only. Scientific admission, job completion,
NN commands and the queue's own process/state handling remain independent.
"""
from copy import deepcopy
import json
from pathlib import Path


STATUSES = {'pending', 'running', 'completed', 'FAILED', 'HELD'}
TERMINAL = {'completed', 'FAILED', 'HELD'}


def backfill_decision(wave, grouped_status, *, wave_stage_running,
                      grouped_parent_live, grouped_session_live):
    if (not isinstance(wave, dict) or not wave or any(s not in STATUSES for s in wave.values())
            or grouped_status not in STATUSES
            or any(type(v) is not bool for v in
                   (wave_stage_running, grouped_parent_live, grouped_session_live))):
        raise ValueError('explicit actual wave/group states and process observations required')
    drained = all(s in TERMINAL for s in wave.values()) and not wave_stage_running
    if grouped_status in TERMINAL and not grouped_parent_live and not grouped_session_live:
        phase, allow, done = 'group-terminal-and-owned-session-drained', True, True
    elif grouped_status == 'running' or grouped_parent_live or grouped_session_live:
        phase, allow, done = 'group-running-or-owned-session-draining', False, False
    elif drained:
        phase, allow, done = 'coalesce-free-GPUs-before-group-launch', False, False
    else:
        phase, allow, done = 'allow-backfill-during-preceding-wave', True, False
    return dict(phase=phase, allow_lower_priority_GPU_backfill=allow, resource_transition_done=done,
                scope='resource availability only', grouped_scientific_state=grouped_status,
                wave_scientific_states=deepcopy(wave),
                failed_or_held_scientific_jobs_not_relabelled_success=True)


def with_resource_permit(job, permit):
    if (job.get('resource', 'exclusive') == 'cpu' or job.get('attached_pid')
            or not isinstance(permit, str) or not Path(permit).is_absolute()):
        raise ValueError('ordinary pending GPU job and explicit absolute resource permit required')
    value = deepcopy(job)
    previous = value.get('requires', [])
    if not isinstance(previous, list) or permit in previous:
        raise ValueError('preserve original requires and apply resource policy once')
    value['requires'] = [*previous, permit]
    reverse = deepcopy(value)
    if 'requires' in job:
        reverse['requires'] = deepcopy(job['requires'])
    else:
        reverse.pop('requires')
    if json.dumps(reverse, sort_keys=True) != json.dumps(job, sort_keys=True):
        raise ValueError('resource permit changed an existing command or scientific dependency')
    return value
