"""Whole-asset disk credits and actual two-dispatcher GPU ownership.

Credits transfer atomically to the unchanged case reservation. A second byte
per admitted byte covers written case data while the original DiskBudget still
counts a raw case's entire reservation. No timeout expires either ownership.
"""
import json
import os
from pathlib import Path

import psutil

from .cascade_view_roles import file_sha256 as sha
from .native_run_disk_budget import DiskBudget
from .selected_pipeline_model_contract import require


def live(identity):
    try:
        p = psutil.Process(identity['PID'])
        return p.create_time() == identity['birth'] and p.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
    except psutil.NoSuchProcess:
        return False


def process_identity(pid=None):
    p = psutil.Process(os.getpid() if pid is None else pid)
    return dict(PID=p.pid, birth=p.create_time())


def verify_queue_owner(group, *, identity, command):
    """An actual dispatcher, its running descriptor and the actual child agree."""
    require(live(group['dispatcher']) and live(identity), 'both actual dispatcher and keeper/coordinator must be live')
    jobs = json.loads(Path(group['jobs']).read_text())
    require(len({j['name'] for j in jobs}) == len(jobs), 'unique dispatcher job names required')
    rows = [j for j in jobs if j['name'] == group['job_name']]
    require(len(rows) == 1 and rows[0]['command'] == command, 'actual dispatcher descriptor must name this exact closed entry')
    states = json.loads(Path(group['states']).read_text()); state = states.get(group['job_name'], {})
    expected = ','.join(group['assigned_GPUs'])
    require(state.get('status') == 'running' and state.get('pid') == identity['PID']
        and state.get('command') == command and state.get('gpu') == expected
        and state.get('resource') == ('exclusive' if expected else 'cpu'),
        'actual running queue reservation and exact GPU group required')
    job = rows[0]
    require(job.get('resource', 'exclusive') == ('exclusive' if expected else 'cpu')
        and (not expected or [str(g) for g in job.get('gpus', [])] == group['assigned_GPUs']),
        'descriptor must reserve the entire exclusive group')
    p = psutil.Process(identity['PID'])
    require(p.ppid() == group['dispatcher']['PID'] and p.cmdline() == command,
        'reservation must be an actual direct child of the bound dispatcher')
    environment = p.environ()
    require(environment.get('CUDA_VISIBLE_DEVICES') == expected
        and set(group['assigned_GPUs']) <= set(environment.get('MATCHING_V2_ALLOWED_GPUS','').split(',')),
        'actual reservation environment must agree with its scheduler assignment')
    return state


def geometry_owners(models):
    owners = {}
    for alias, model in models.items():
        owners.setdefault(model['geometry_group'], alias)
    return {alias:owners[model['geometry_group']] for alias,model in models.items()}


def ready_aliases(models, completed, running):
    """All models of an admitted asset have credits before its reference starts."""
    owners = geometry_owners(models)
    if 'zero-source4' not in completed:
        return [] if 'zero-source4' in running else ['zero-source4']
    return [alias for alias in models if alias not in completed and alias not in running
        and (owners[alias] == alias or owners[alias] in completed)]


def reserve_asset_group(budget, name, *, cases, output, owner):
    """Atomically admit every future consumer, never just geometry producers."""
    require(cases and len({v['output'] for v in cases.values()}) == len(cases)
        and all(type(v['bytes']) is int and v['bytes'] > 0 for v in cases.values()),
        'unique explicit positive whole-case credits required')
    require(live(owner), 'actual live campaign owner required')
    total = 2*sum(v['bytes'] for v in cases.values())
    with budget._locked() as value:
        require(name not in value['runs'], 'never resume or overwrite an asset credit')
        admission = budget._snapshot(value, total)
        if not admission['admitted']:
            return admission
        value['runs'][name] = dict(checkpoint_budget_bytes=total, native_output=str(output), wrapper_output=str(output),
            owner=owner, released=False, admission=admission, selected_campaign_asset_credit=True,
            cases={key:dict(v,claimed=False) for key,v in cases.items()}, original_case_budget_bytes=total//2)
    return admission


class CampaignCaseBudget(DiskBudget):
    """Transfer one two-byte future credit to one normal one-byte case entry."""
    def __init__(self, *args, credit, **kwargs):
        super().__init__(*args, **kwargs); self.credit = credit

    def reserve(self, name, *, checkpoint_budget_bytes, native_output, wrapper_output, owner):
        credit = self.credit
        require(credit['case_name'] == name and live(credit['coordinator'])
            and credit['coordinator']['PID'] == os.getppid() and owner == process_identity(),
            'actual direct campaign child may claim only its own credit')
        with self._locked() as value:
            require(name not in value['runs'], 'never reuse a claimed case reservation')
            group = value['runs'][credit['group_name']]
            require(group.get('selected_campaign_asset_credit') is True and group.get('released') is False
                and group['owner'] == credit['coordinator'], 'own live campaign asset reservation required')
            case = group['cases'][name]
            require(case['claimed'] is False and case['bytes'] == checkpoint_budget_bytes
                and case['output'] == str(native_output) == str(wrapper_output)
                and group['checkpoint_budget_bytes'] >= 2*checkpoint_budget_bytes,
                'exact unclaimed whole-case disk credit required')
            admission = self._snapshot(value, 0)
            require(admission['admitted'], 'actual disk no longer covers all admitted consumers and workspace')
            group['checkpoint_budget_bytes'] -= 2*checkpoint_budget_bytes
            case.update(claimed=True, owner=owner)
            value['runs'][name] = dict(checkpoint_budget_bytes=checkpoint_budget_bytes,
                native_output=str(native_output),wrapper_output=str(wrapper_output),owner=owner,released=False,
                admission=admission,transferred_from_campaign_asset=credit['group_name'])
        return dict(admission, actual_atomic_campaign_credit_transfer=True)


def token_processes(token, *, field='SELECTED_FINAL_CAMPAIGN_TOKEN'):
    """Inspect actual inherited ownership, including sessions orphaned by a crash.

    AccessDenied is an error, not proof of absence. Keepers themselves do not
    carry this token; every campaign case and all its phase children do.
    """
    require(field in ('SELECTED_FINAL_CAMPAIGN_TOKEN','SELECTED_FINAL_WORKER_TOKEN'), 'explicit ownership token field required')
    rows = []
    for p in psutil.process_iter(['pid','create_time','status']):
        try:
            if p.info['status'] in (psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD):
                continue
            if p.environ().get(field) == token:
                rows.append(dict(PID=p.pid,birth=p.create_time(),session_ID=os.getsid(p.pid)))
        except (psutil.NoSuchProcess,psutil.ZombieProcess,ProcessLookupError):
            continue
    return rows


def source_guard(context):
    require(all(sha(path) == digest for path,digest in context['source_SHA256'].items()),
        'actual source files changed after campaign preparation')
