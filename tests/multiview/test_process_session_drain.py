import subprocess
import sys
import time

import psutil
import pytest

from experiments.deformnext_retargeting.process_session_drain import drain_owned_session,owned_session_processes


def test_exited_parent_does_not_hide_its_live_orphaned_child(tmp_path):
    # Real OS process/session behavior, no CUDA or neural work.
    marker=tmp_path/'child-started'
    code="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(.5)']); open(sys.argv[1],'w').write(str(p.pid)); time.sleep(.05); sys.exit(7)"
    parent=subprocess.Popen([sys.executable,'-c',code,str(marker)],start_new_session=True)
    birth=psutil.Process(parent.pid).create_time()
    assert parent.wait()==7 and marker.is_file()
    child_pid=int(marker.read_text())
    assert child_pid in {r['PID']for r in owned_session_processes(parent.pid,birth)}
    def failed_disk(_):raise OSError('simulated full disk')
    observed=drain_owned_session(parent.pid,birth,progress=failed_disk,interval=.02)
    assert observed and not owned_session_processes(parent.pid,birth)


def test_drain_rejects_unowned_or_invalid_session_parameters():
    with pytest.raises(ValueError):owned_session_processes(0,time.time())
    with pytest.raises(ValueError):drain_owned_session(1,time.time(),interval=60)
