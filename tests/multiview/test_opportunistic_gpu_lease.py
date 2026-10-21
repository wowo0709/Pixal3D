"""Resource-only admission and actual owner-exit behavior; no CUDA work."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.deformnext_retargeting.opportunistic_gpu_lease import (
    ProcessRequirementLease, decide_admission,
)
from experiments.deformnext_retargeting.matching_v2_queue import choose_gpu


def decision(frontier, gang='pending', *, verified=False, free=100, reserve=20):
    return decide_admission(frontier, {'status': gang, 'returncode': 0},
                            gang_output_verified=verified,
                            available_bytes=free, reserve_bytes=reserve)


def test_ready_gang_does_not_idle_spare_devices_through_training():
    frontier = {'short': {'status': 'completed', 'returncode': 0},
                'long': {'status': 'running'}}
    d = decision(frontier)
    assert d.action == 'open' and d.unfinished == ('long',)
    frontier['long'] = {'status': 'completed', 'returncode': 0}
    assert decision(frontier).action == 'drain'
    assert decision(frontier, 'running').action == 'drain'
    assert decision(frontier, 'completed').action == 'hold'
    assert decision(frontier, 'completed', verified=True).action == 'release'


@pytest.mark.parametrize('status', ['FAILED', 'HELD', None, 'unknown'])
def test_unavailable_frontier_stops_new_admissions(status):
    assert decision({'broken': {'status': status}}).action == 'hold'


@pytest.mark.parametrize('status', ['FAILED', 'HELD', None])
def test_unavailable_gang_stops_new_admissions(status):
    assert decision({'ss': {'status': 'running'}}, status).action == 'hold'


def test_bad_exit_code_and_storage_reserve_are_not_success():
    assert decision({'ss': {'status': 'completed', 'returncode': 1}}).action == 'hold'
    assert decision({'ss': {'status': 'running'}}, free=19).reason == 'checkpoint_storage_reserve'
    assert decision({'ss': {'status': 'running'}}, free=20).action == 'open'
    for frontier, free, reserve in [({}, 100, 20), ({'ss': {}}, -1, 20),
                                     ({'ss': {}}, 100, -1)]:
        with pytest.raises(ValueError):
            decision(frontier, free=free, reserve=reserve)


def test_revoke_uses_exists_only_and_never_marks_a_waiting_job_held(tmp_path):
    path = tmp_path/'requirement'
    with ProcessRequirementLease(path, {'run': 'test'}) as lease:
        lease.set_open(True)
        assert path.exists() and path.is_symlink()
        assert json.loads(path.read_text())['status'] == 'PASS'
        lease.set_open(False)
        assert not path.exists()
        lease.set_open(True)
        assert path.exists()
    assert not path.exists()


def test_success_persists_after_owner_closes_descriptor(tmp_path):
    path = tmp_path/'requirement'
    with ProcessRequirementLease(path, {}) as lease:
        lease.set_open(True)
        with pytest.raises(ValueError):
            lease.release_after_gang({'status': 'FAIL'})
        lease.release_after_gang({'status': 'PASS', 'actual_gang_receipt': 'verified'})
    assert path.is_file() and not path.is_symlink()
    assert json.loads(path.read_text())['actual_gang_receipt'] == 'verified'
    with pytest.raises(FileExistsError):
        ProcessRequirementLease(path, {})


def test_foreign_replacement_is_preserved(tmp_path):
    path = tmp_path/'requirement'
    lease = ProcessRequirementLease(path, {})
    lease.set_open(True)
    path.unlink()
    path.write_text('foreign')
    with pytest.raises(ValueError):
        lease.set_open(False)
    with pytest.raises(ValueError):
        lease.close()
    assert path.read_text() == 'foreign'


def test_unexpected_owner_exit_invalidates_actual_queue_requirement(tmp_path):
    path = tmp_path/'requirement'
    code = '''
import os, sys
from pathlib import Path
from experiments.deformnext_retargeting.opportunistic_gpu_lease import ProcessRequirementLease
lease = ProcessRequirementLease(Path(sys.argv[1]), {'test': 'owner_exit'})
lease.set_open(True)
print('READY', flush=True)
sys.stdin.readline()
os._exit(23)
'''
    child = subprocess.Popen([sys.executable, '-u', '-c', code, str(path)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'READY'
        assert path.exists()
        child.communicate('\n', timeout=10)
        assert child.returncode == 23
        assert path.is_symlink() and not path.exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_cloexec_prevents_subprocess_from_extending_owner_lease(tmp_path):
    path = tmp_path/'requirement'
    with ProcessRequirementLease(path, {}) as lease:
        child = subprocess.run([sys.executable, '-c',
            'import os,sys; print(os.path.exists(sys.argv[1]))',
            f'/proc/self/fd/{lease.fd}'], close_fds=False,
            check=True, capture_output=True, text=True)
        assert child.stdout.strip() == 'False'


def test_existing_queue_fills_tail_then_obtains_all_four_devices(tmp_path):
    path = tmp_path/'requirement'
    allowed = ['0', '1', '2', '3']
    gang = {'resource': 'exclusive', 'gpus': allowed}
    case = {'resource': 'exclusive', 'requires': [str(path)]}
    state = {'long_ss': {'status': 'running', 'gpu': '3', 'resource': 'exclusive'}}
    with ProcessRequirementLease(path, {}) as lease:
        lease.set_open(True)
        assert all(Path(p).exists() for p in case['requires'])
        device = choose_gpu(case, allowed, state)
        assert device == '0' and choose_gpu(gang, allowed, state) is None
        state['finite_case'] = {'status': 'running', 'gpu': device, 'resource': 'exclusive'}
        state['long_ss']['status'] = 'completed'
        lease.set_open(False)
        assert not all(Path(p).exists() for p in case['requires'])
        assert choose_gpu(gang, allowed, state) is None
        # Drain the already admitted case naturally; no process is cancelled.
        state['finite_case']['status'] = 'completed'
        assert choose_gpu(gang, allowed, state) == '0,1,2,3'
        state['gang'] = {'status': 'running', 'gpu': '0,1,2,3', 'resource': 'exclusive'}
        assert choose_gpu(case, allowed, state) is None
        state['gang']['status'] = 'completed'
        lease.release_after_gang({'status': 'PASS'})
        assert all(Path(p).exists() for p in case['requires'])
        assert choose_gpu(case, allowed, state) == '0'
