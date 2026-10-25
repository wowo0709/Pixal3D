"""Real protected-reader waits and real CPU scheduling order; no model forward."""
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest

from experiments.deformnext_retargeting.operational_process_access import process_access_waits
from experiments.deformnext_retargeting import operational_campaign_entry as entry
from experiments.deformnext_retargeting import run_selected_final_campaign as frozen


@pytest.mark.parametrize('method', ['open_files', 'environ'])
def test_actual_nondumpable_reader_is_not_treated_as_absent(tmp_path, method):
    raw = tmp_path / 'owned-reader.txt'
    raw.write_text('owned CPU test input')
    source = '''
import ctypes,sys,time
stream=open(sys.argv[1])
assert ctypes.CDLL(None).prctl(4,0,0,0,0)==0
print('READY',flush=True)
time.sleep(.7)
'''
    child = subprocess.Popen([sys.executable, '-c', source, str(raw)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'READY'
        process = psutil.Process(child.pid)
        try:
            getattr(process, method)()
        except psutil.AccessDenied:
            pass
        else:
            pytest.skip('host grants access to nondumpable owned test process')
        events = []
        began = time.monotonic()
        with process_access_waits(emit=events.append):
            with pytest.raises((psutil.NoSuchProcess, psutil.ZombieProcess)):
                getattr(process, method)()
        assert time.monotonic() - began >= .4
        child.wait(timeout=5)
        assert child.returncode == 0
        assert events[0]['event'] == 'waiting-for-process-access'
        assert events[-1]['resolution'] == 'original-process-exited'
        assert raw.read_text() == 'owned CPU test input'
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


def test_reader_adapter_restores_methods_even_after_error():
    originals = {n:getattr(psutil.Process,n) for n in ['open_files','environ']}
    with pytest.raises(RuntimeError):
        with process_access_waits():
            assert psutil.Process(os.getpid()).environ() == originals['environ'](psutil.Process(os.getpid()))
            raise RuntimeError('owned test exit')
    assert all(getattr(psutil.Process,n) is v for n,v in originals.items())


def test_retirement_finishes_before_scheduler_can_change_reader_ledger(tmp_path, monkeypatch):
    ledger = tmp_path / 'jobs.json'
    ledger.write_text('ORIGINAL-READER-PLAN')
    result = tmp_path / 'observed.txt'
    actual = []
    child = []
    def spawn(self, name, context, module, **kwargs):
        actual.append((module, context, kwargs))
        code = 'from pathlib import Path;import sys,time;p=Path(sys.argv[1]);a=p.read_text();time.sleep(.15);assert a==p.read_text();Path(sys.argv[2]).write_text("PASSED")'
        p = subprocess.Popen([sys.executable,'-c',code,str(ledger),str(result)])
        child.append(p)
        self.own[name] = {'child':p}
    monkeypatch.setattr(frozen.Campaign, 'spawn', spawn)
    obj = object.__new__(entry.SerializedRetirementCampaign)
    obj.context = {'operational_source_SHA256':{'cpu-test':'cpu-test'}}
    obj.root = tmp_path
    obj.own = {}
    def collect():
        assert child[0].returncode == 0 and result.read_text() == 'PASSED'
        obj.own.clear()
    obj.collect = collect
    obj.spawn('retire', {}, 'experiments.deformnext_retargeting.retire_selected_final_raw',
              sid='cpu-test', alias='zero-source4', kind='retirement')
    ledger.write_text('NEXT-READER-PLAN')
    assert actual[0][0] == entry.RETIREMENT and child[0].returncode == 0


def test_native_case_arguments_are_preserved_and_entry_patch_is_scoped(monkeypatch):
    seen = []
    monkeypatch.setattr(frozen.Campaign,'spawn',lambda self,*a,**k:seen.append((a,k)))
    obj = object.__new__(entry.SerializedRetirementCampaign)
    context = {'untouched':'native-case'}
    obj.spawn('case',context,'experiments.deformnext_retargeting.run_selected_campaign_case',
              sid='cpu-test',alias='V1',kind='case',gpu='6')
    assert seen == [(('case',context,'experiments.deformnext_retargeting.run_selected_campaign_case'),
                     dict(sid='cpu-test',alias='V1',kind='case',gpu='6',role=None))]
    old = frozen.command, frozen.Campaign
    with entry.operational_runtime():
        assert frozen.command is entry.command and frozen.Campaign is entry.SerializedRetirementCampaign
    assert (frozen.command, frozen.Campaign) == old
