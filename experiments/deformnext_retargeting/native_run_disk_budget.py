"""Atomic local checkpoint growth reservations for concurrent native runs.

Operational bookkeeping only; it never marks a scientific run complete and
never expires a reservation merely because its PID is missing.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil

from .checkpoint_cache_retirement import disk_admission, remaining_checkpoint_budget


class DiskBudget:
    def __init__(self, path, *, workspace_bytes, free_bytes=None):
        self.path = Path(path)
        self.workspace = workspace_bytes
        self.free_bytes = free_bytes or (lambda: shutil.disk_usage(self.path.parent).free)

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            value = json.loads(self.path.read_text()) if self.path.exists() else dict(version=1, runs={})
            yield value
            temporary = self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps(value, indent=2)+'\n')
            os.replace(temporary, self.path)

    def _snapshot(self, value, new_budget):
        outstanding = []
        for entry in value['runs'].values():
            if entry.get('released'):
                continue
            folder = Path(entry['native_output'])/'ckpts'
            sizes = [p.stat().st_size for p in folder.glob('*.pt') if p.is_file() and not p.is_symlink()]
            outstanding.append(remaining_checkpoint_budget(entry['checkpoint_budget_bytes'], sizes))
        # Observe free disk AFTER current file sizes. Concurrent saves then
        # make this conservative, rather than counting just-written bytes twice.
        return disk_admission(actual_free=self.free_bytes(), new_budget=new_budget,
            outstanding=outstanding, workspace=self.workspace)

    def inspect(self, new_budget):
        with self._locked() as value:
            return self._snapshot(value, new_budget)

    def reserve(self, name, *, checkpoint_budget_bytes, native_output, wrapper_output, owner):
        with self._locked() as value:
            if name in value['runs']:
                raise ValueError('a scientific run reservation cannot be reused or silently resumed')
            admission = self._snapshot(value, checkpoint_budget_bytes)
            if not admission['admitted']:
                raise ValueError('actual free disk cannot cover pending checkpoints and workspace: '+str(admission))
            value['runs'][name] = dict(checkpoint_budget_bytes=checkpoint_budget_bytes,
                native_output=str(Path(native_output)), wrapper_output=str(Path(wrapper_output)),
                owner=owner, released=False, admission=admission)
            return admission

    def release(self, name, *, actual_wrapper_receipt):
        receipt = Path(actual_wrapper_receipt)
        report = json.loads(receipt.read_text())
        with self._locked() as value:
            entry = value['runs'][name]
            if (receipt.parent != Path(entry['wrapper_output']) or report.get('status') != 'PASS'
                    or report.get('local_equals_actual_file4_SHA256') is not True):
                raise ValueError('release requires this run actual canonical wrapper receipt')
            entry.update(released=True, actual_wrapper_receipt=str(receipt))
