"""Fill single-GPU tails while reserving a later multi-GPU job.

This changes resource admission only. The existing queue's ``requires`` check
uses Path.exists(), so an absent or dangling lease waits without changing job
state. Do not use this revocable lease as the queue's JSON ``gate_file``.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping


@dataclass(frozen=True)
class Admission:
    action: str
    reason: str
    unfinished: tuple[str, ...] = ()


def decide_admission(frontier: Mapping[str, Mapping], gang: Mapping, *,
                     gang_output_verified: bool, available_bytes: int,
                     reserve_bytes: int) -> Admission:
    """Allow spare work until the explicitly enumerated training frontier ends.

    A ready gang dependency alone must not drain the GPUs: that would suppress
    useful work throughout the preceding training. Once its frontier finishes,
    stop new admissions and let already running finite jobs finish naturally.
    """
    if not frontier or reserve_bytes < 0 or available_bytes < 0:
        raise ValueError('explicit frontier and nonnegative storage budget required')
    status = gang.get('status')
    if status == 'completed' and gang.get('returncode') in (None, 0):
        return Admission('release' if gang_output_verified else 'hold',
                         'gang_completed_and_verified' if gang_output_verified
                         else 'gang_completion_output_not_verified')
    if status == 'running':
        return Admission('drain', 'gang_running')
    if status != 'pending':
        return Admission('hold', 'gang_unavailable')
    bad = tuple(name for name, row in frontier.items()
                if row.get('status') not in {'pending', 'running', 'completed'}
                or (row.get('status') == 'completed'
                    and row.get('returncode') not in (None, 0)))
    if bad:
        return Admission('hold', 'frontier_unavailable', bad)
    unfinished = tuple(name for name, row in frontier.items()
                       if row['status'] != 'completed')
    if not unfinished:
        return Admission('drain', 'frontier_completed')
    if available_bytes < reserve_bytes:
        return Admission('hold', 'checkpoint_storage_reserve', unfinished)
    return Admission('open', 'frontier_still_running', unfinished)


class ProcessRequirementLease:
    """Linux process-owned existence requirement, invalid after owner exit.

    A symlink to an anonymous temporary descriptor exists only while the owning
    process keeps it open. A crash stops new work without an external
    cleanup process. Normal completion replaces it with a persistent receipt.
    The caller owns the path exclusively; stale or foreign files are rejected.
    """
    def __init__(self, path: Path, payload: Mapping):
        self.path = Path(path)
        if os.path.lexists(self.path):
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = tempfile.TemporaryFile(mode='w+b', dir=self.path.parent)
        self.fd = self.stream.fileno()
        os.set_inheritable(self.fd, False)
        self.target = f'/proc/{os.getpid()}/fd/{self.fd}'
        self.finished = False
        body = dict(payload, status='PASS', owner_PID=os.getpid())
        os.write(self.fd, (json.dumps(body, sort_keys=True)+'\n').encode())
        os.lseek(self.fd, 0, os.SEEK_SET)

    def _check_owned_path(self):
        if os.path.lexists(self.path):
            if not self.path.is_symlink() or os.readlink(self.path) != self.target:
                raise ValueError('lease path was replaced by another owner')

    def set_open(self, enabled: bool):
        if self.finished or self.fd < 0:
            raise ValueError('closed lease cannot admit more work')
        self._check_owned_path()
        if enabled:
            if not os.path.lexists(self.path):
                self.path.symlink_to(self.target)
        else:
            self.path.unlink(missing_ok=True)

    def release_after_gang(self, payload: Mapping):
        if payload.get('status') != 'PASS':
            raise ValueError('verified successful gang completion required')
        if self.finished or self.fd < 0:
            raise ValueError('lease already closed')
        self._check_owned_path()
        with tempfile.NamedTemporaryFile(mode='w', dir=self.path.parent,
                                         prefix=self.path.name+'.', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(dict(payload), stream)
            stream.write('\n')
        os.replace(temporary, self.path)
        self.finished = True
        self.stream.close()
        self.fd = -1

    def close(self):
        if self.fd >= 0:
            try:
                if not self.finished:
                    self._check_owned_path()
                    self.path.unlink(missing_ok=True)
            finally:
                self.stream.close()
                self.fd = -1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
