"""Review explicitly documented historical failures without rewriting the queue."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def checked(root, record):
    path = root / record['relative_path']
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('failure-review evidence must stay inside the campaign')
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('failure-review evidence SHA256 differs')
    return path


def review_historical_failures(root: Path, state):
    """Unknown failures remain unresolved; a PASS file alone is insufficient."""
    failed = {name for name, row in state.items() if row['status'] == 'FAILED'}
    manifest = root / 'provenance/resolved-historical-queue-failures-v1.json'
    resolved = {}
    if manifest.exists():
        value = json.loads(manifest.read_text())
        if value['status'] != 'PASS_RESOLVED_HISTORICAL_FAILURES':
            raise ValueError('historical failure review is incomplete')
        for review in value['reviews']:
            name = review['failed_job']
            if name not in failed or name in resolved:
                raise ValueError('failure review is duplicated or names a nonfailed job')
            log = checked(root, review['failure_log'])
            if log.name != name + '.log' or not log.stat().st_size:
                raise ValueError('failure log identity differs')
            successors = review['successor_jobs']
            if not successors or any(state.get(job, {}).get('status') != 'completed' for job in successors):
                raise ValueError('historical failure has no completed replacement')
            if not review['evidence']:
                raise ValueError('historical failure has no immutable evidence')
            for evidence in review['evidence']:
                actual = json.loads(checked(root, evidence).read_text())
                if not evidence['status'].startswith('PASS') or actual.get('status') != evidence['status']:
                    raise ValueError('replacement evidence is not the reviewed PASS')
            resolved[name] = successors
    return {'resolved_historical_failures': resolved,
            'unresolved_failures': sorted(failed - resolved.keys())}
