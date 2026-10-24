"""Actual metadata-only keeper processes; no GPU model construction."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from experiments.deformnext_retargeting import run_selected_final_campaign as module
from experiments.deformnext_retargeting.cascade_view_roles import file_sha256 as sha
from experiments.deformnext_retargeting.selected_campaign_resources import process_identity


def wait_for(predicate,child,seconds=20):
    deadline=time.monotonic()+seconds
    while not predicate():
        assert child.poll() is None, 'actual keeper exited before its release handshake'
        assert time.monotonic() < deadline, 'test observation expired; this does not authorize restart'
        time.sleep(.05)


def prepare(tmp_path):
    base=tmp_path/'canonical';base.mkdir();root=base/'campaign';manifest=base/'manifests';manifest.mkdir()
    helper=base/'test_local_closure.py';helper.write_text('''import argparse,json,pathlib,hashlib
p=argparse.ArgumentParser();p.add_argument('--manifest');p.add_argument('--receipt');p.add_argument('--scope');a=p.parse_args()
base=pathlib.Path(__file__).parent;rows=[]
for name in json.load(open(a.manifest)):
 q=base/name;rows.append(dict(relative_path=name,sha256=hashlib.sha256(q.read_bytes()).hexdigest(),bytes=q.stat().st_size))
dest=base/a.receipt;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(dict(status='PASS',local_equals_actual_file4_SHA256=True,files=rows)))
''')
    path=base/'plans/context.json';path.parent.mkdir()
    for name in ['jobs','states']:(base/(name+'.json')).write_text('[]' if name=='jobs' else '{}')
    group=dict(dispatcher=process_identity(),jobs=str(base/'jobs.json'),states=str(base/'states.json'),job_name='keeper',assigned_GPUs=['0'])
    other=dict(group,job_name='other-keeper',assigned_GPUs=['6'])
    context=dict(schema=module.SCHEMA,name='keeper-fixture',repo=str(Path(__file__).resolve().parents[2]),python=sys.executable,
        canonical_local_root=str(base),campaign_root=str(root),manifest_directory=str(manifest),disk_reservations=str(base/'r.json'),
        workspace_bytes=1,assigned_GPUs=['0','6'],max_asset_groups=2,CPU_affinity=sorted(os.sched_getaffinity(0)),
        keeper_groups=dict(main=group,aux=other),coordinator_queue=dict(group,job_name='coordinator',assigned_GPUs=[]),
        source_SHA256={p:sha(p) for p in [*module.required_sources(),str(helper)]},closure_command=[sys.executable,str(helper)],
        reader_queues=[dict(jobs=group['jobs'],states=group['states'])],
        case_defaults=dict(CPU_affinity_per_GPU={'0':[92],'6':[93]},minimum_free_GPU_bytes=1))
    path.write_text(json.dumps(context));(path.parent/'file4-closure.json').write_text(json.dumps(dict(status='PASS',local_equals_actual_file4_SHA256=True,
        files=[dict(relative_path=str(path.relative_to(base)),sha256=sha(path),bytes=path.stat().st_size)])))
    command=module.command(context,path,'main')
    Path(group['jobs']).write_text(json.dumps([dict(name='keeper',command=command,resource='exclusive',gpus=['0'])]))
    return base,root,path,context,command,helper


@pytest.mark.parametrize('stop_kind',['signal','source-change'])
def test_actual_keeper_holds_through_interrupt_or_guard_failure_and_live_other_session(tmp_path,stop_kind):
    base,root,path,context,command,helper=prepare(tmp_path)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',MATCHING_V2_ALLOWED_GPUS='0',PYTHONPATH=context['repo'])
    log=tmp_path/'keeper.log'
    with log.open('wb') as stream:child=subprocess.Popen(command,env=env,cwd=context['repo'],stdout=stream,stderr=subprocess.STDOUT)
    group=context['keeper_groups']['main'];owned=None
    try:
        Path(group['states']).write_text(json.dumps({'keeper':dict(status='running',pid=child.pid,command=command,gpu='0',resource='exclusive')}))
        ready=root/'ownership/main/ready.json';wait_for(ready.exists,child)
        owned=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True,
            env=dict(os.environ,SELECTED_FINAL_CAMPAIGN_TOKEN=sha(path)))
        if stop_kind=='signal':child.send_signal(signal.SIGTERM)
        else:helper.write_text(helper.read_text()+'\n# deliberate fixture source change\n')
        wait_for((root/'ownership/main/stop-request.json').exists,child)
        # A release CLAIM cannot override a real live descendant in a separate
        # session. Its fresh actual process is what keeps this keeper alive.
        identity=process_identity();(root/'coordinator.json').write_text(json.dumps(dict(identity=identity)))
        release=root/'release.json';release.write_text(json.dumps(dict(status='FAILED_FIXTURE_ONLY',plan_SHA256=sha(path),
            coordinator=identity,all_actual_campaign_children_drained=True)))
        receipt=root/'release-closure/file4-closure.json';receipt.parent.mkdir()
        receipt.write_text(json.dumps(dict(status='PASS',local_equals_actual_file4_SHA256=True,
            files=[dict(relative_path=str(release.relative_to(base)),sha256=sha(release),bytes=release.stat().st_size)])))
        time.sleep(2.2);assert child.poll() is None
        owned.terminate();owned.wait();owned=None
        child.wait(timeout=15)
        assert child.returncode==0,log.read_text()
        terminal=json.load(open(root/'ownership/main/terminal.json'))
        assert terminal['actual_owned_GPU_descendants_absent'] is True
        assert (root/'ownership/main/file4-closure.json').exists()
    finally:
        if owned is not None:owned.terminate();owned.wait()
        # A failed fixture never leaves a real CPU keeper behind. SIGKILL here
        # is test cleanup only, after the owned stand-in descendant is reaped.
        if child.poll() is None:child.kill();child.wait()


def test_case_adapter_restores_original_budget_after_failure(tmp_path,monkeypatch):
    from experiments.deformnext_retargeting import run_selected_campaign_case as adapter
    pins={str(Path(adapter.__file__).resolve()):sha(adapter.__file__),
        str(Path(module.__file__).with_name('selected_campaign_resources.py')):sha(Path(module.__file__).with_name('selected_campaign_resources.py'))}
    context=dict(campaign_credit=dict(group_name='explicit-credit'),campaign_token='actual-test-token',source_SHA256=pins)
    path=tmp_path/'context.json';path.write_text(json.dumps(context));original=adapter.native.DiskBudget
    monkeypatch.setenv('SELECTED_FINAL_CAMPAIGN_TOKEN','actual-test-token')
    def fail(actual):
        assert Path(actual)==path and adapter.native.DiskBudget is not original
        raise RuntimeError('intentional fixture failure before any NN')
    monkeypatch.setattr(adapter.native,'execute',fail)
    with pytest.raises(RuntimeError,match='fixture failure'):adapter.execute(path)
    assert adapter.native.DiskBudget is original
