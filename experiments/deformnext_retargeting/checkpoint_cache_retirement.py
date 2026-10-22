"""Conservative local-cache retirement decisions; this module never deletes.

Canonical file4 checkpoints remain the scientific record. A runtime worker
must additionally verify actual remote bytes immediately before local unlink.
"""
from pathlib import Path
import re


PATTERN = re.compile(r'^(denoiser|misc|deform_conditioner)_step([0-9]{7})\.pt$')


def pending_checkpoint_reads(jobs, states, candidates):
    """Collect explicit plan/hash reads, including declared checkpoint bundles.

    Receipts are provenance indexes and do not imply consuming every archived
    checkpoint. Input-hash keys and closed-input keys do imply real file reads.
    Unresolved planned bundles conservatively protect their declared step.
    """
    import json
    candidates = {str(Path(p)): Path(p) for p in candidates}
    readers = {p: [] for p in candidates}
    for job in jobs:
        # A newly appended descriptor may not yet have a dispatcher state.
        # Treat it as pending, never as evidence that its inputs are unused.
        if states.get(job['name'], {}).get('status', 'pending') not in {'pending', 'running'}:
            continue
        command = job.get('command', [])
        if '--plan' not in command:
            continue
        path = Path(command[command.index('--plan')+1])
        if not path.is_file():
            raise ValueError('unreadable pending job plan prevents cache retirement: '+job['name'])
        plan = json.loads(path.read_text())

        def scan(value, *, provenance=False):
            if isinstance(value, str):
                if value in readers and not provenance:
                    readers[value].append(job['name'])
            elif isinstance(value, list):
                for item in value: scan(item, provenance=provenance)
            elif isinstance(value, dict):
                for key, item in value.items():
                    if key in {'source_SHA256', 'input_SHA256', 'closed_inputs'} and isinstance(item, dict):
                        for actual_path in item: scan(actual_path)
                    else:
                        scan(item, provenance=provenance or key in {'verified_inputs', 'files'})

        scan(plan)
        for field in ['bundle', 'checkpoint_bundle']:
            bundle = plan.get(field)
            if not isinstance(bundle, str): continue
            bundle_path = Path(bundle)
            if bundle_path.is_file():
                scan(json.loads(bundle_path.read_text()))
            else:
                # This campaign declares the bundle's step in its path.
                match = re.search(r'bundles-step([0-9]+)', bundle)
                if match:
                    step = int(match.group(1))
                    for candidate in candidates.values():
                        token = PATTERN.fullmatch(candidate.name)
                        if token and int(token.group(2)) == step:
                            readers[str(candidate)].append(job['name']+' (unresolved bundle)')
                else:
                    raise ValueError('unknown pending bundle prevents cache retirement: '+job['name'])
    return {path: sorted(set(names)) for path, names in readers.items() if names}


def validate_retirement_candidate(path, *, run_root, final_step, allowed_steps, terminal,
                                  progress, canonical_entry, actual_local_sha, pending_reads):
    """Require own completed run, explicit old step, canonical identity, no reader."""
    path, root = Path(path), Path(run_root)
    match = PATTERN.fullmatch(path.name)
    if (not match or not path.is_absolute() or not root.is_absolute()
            or path.parent != root/'ckpts' or path.is_symlink() or root.resolve() != root
            or (root/'ckpts').is_symlink() or path.resolve().parent != root/'ckpts'
            or int(match.group(2)) not in allowed_steps
            or int(match.group(2)) >= final_step or terminal.get('returncode') != 0
            or progress.get('stage') != 'completed' or progress.get('current') != final_step):
        raise ValueError('only explicit old checkpoints of this completed owned run may retire')
    if str(path) in pending_reads:
        raise ValueError('pending/running reader still requires this checkpoint')
    if (not path.is_file() or canonical_entry.get('sha256') != actual_local_sha
            or canonical_entry.get('bytes') != path.stat().st_size):
        raise ValueError('local checkpoint does not match its closed canonical bytes')
    return dict(path=str(path), step=int(match.group(2)), kind=match.group(1),
        bytes=path.stat().st_size, sha256=actual_local_sha, local_cache_only=True)


def remaining_checkpoint_budget(total, written):
    """Free disk already excludes written checkpoints; reserve only future growth."""
    if (isinstance(total, bool) or not isinstance(total, int) or total < 0
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in written)):
        raise ValueError('explicit nonnegative byte budgets and observed checkpoint sizes required')
    return max(0, total-sum(written))


def disk_admission(*, actual_free, new_budget, outstanding, workspace):
    """A stale/missing PID never implicitly releases an outstanding reservation."""
    values = [actual_free, new_budget, workspace, *outstanding]
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in values):
        raise ValueError('actual nonnegative integer free/budget/workspace bytes required')
    required = new_budget+sum(outstanding)+workspace
    return dict(admitted=actual_free >= required, actual_free_bytes=actual_free,
        new_checkpoint_budget_bytes=new_budget, outstanding_future_bytes=sum(outstanding),
        workspace_bytes=workspace, required_free_bytes=required)
