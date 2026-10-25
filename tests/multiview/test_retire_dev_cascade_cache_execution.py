import json
import os
import sys
from pathlib import Path

import pytest

from experiments.deformnext_retargeting import retire_dev_cascade_cache as module
from experiments.deformnext_retargeting.cascade_view_roles import file_sha256 as sha
from test_dev_cascade_cache_retirement import fixture


def emit(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def setup(tmp_path,monkeypatch):
    root,args=fixture(tmp_path);arm=args['case']['SS_arm'];token=root.name
    closure=tmp_path/'close.py';closure.write_text('test-only closure dependency')
    worker=tmp_path/'original_worker.py';worker.write_text('test-only completed producer')
    queue=tmp_path/'jobs.json';states=tmp_path/'states.json';job='case-'+arm+'-'+token
    planpath=tmp_path/'plans/context.json';policyfile=tmp_path/'policy/policy.json'
    caseplan=tmp_path/'caseplans'/(job+'-case.json')
    def seal(folder):
        files=[dict(relative_path=str(p.relative_to(tmp_path)),bytes=p.stat().st_size,sha256=sha(p))
            for p in folder.rglob('*') if p.is_file() and p!=folder/'file4-closure.json']
        emit(folder/'file4-closure.json',dict(status='PASS',local_equals_actual_file4_SHA256=True,files=files))
    monkeypatch.setattr(module,'required_sources',lambda:[])
    monkeypatch.setattr(module.os,'sched_setaffinity',lambda *_:None)
    monkeypatch.setattr(module.os,'nice',lambda *_:None)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(module,'verify_terminal',lambda t:dict(unit_test=True))
    monkeypatch.setattr(module,'close',lambda folder,*_:seal(folder))
    checked=[]
    def readers(context,folder):
        checked.append(folder)
        return [],{str(queue):sha(queue)}
    monkeypatch.setattr(module,'reader_snapshot',readers)
    def remote(context,rows):
        return {r['relative_path']:dict(sha256=sha(tmp_path/r['relative_path']),bytes=(tmp_path/r['relative_path']).stat().st_size) for r in rows}
    monkeypatch.setattr(module,'remote_hashes',remote)
    case=args['case'];case.update(actual_SS_model_binding={'model':arm},actual_PBR_model_binding={'model':'top1'})
    emit(root/'audit.json',case);emit(root/'geometry/generation.json',args['geometry']);emit(root/'PBR/generation.json',args['replay'])
    emit(root/'shared/render-evaluation.json',dict(sample_id=case['sample_id'],same_reference_pixels_for_every_model=True,per_view=[dict(camera_id=c) for c in [8,9,10]]))
    retained=[root/'geometry/prediction.npz',root/'geometry/native-shape-guides.pt',*[root/'shared'/f'camera{c:02d}.npz' for c in [8,9,10]]]
    for p in retained:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'must retain')
    for folder in [root/'geometry',root/'PBR',root/'shared',root]:seal(folder)
    emit(caseplan,dict(name=job,SS_arm=arm,sample_id=case['sample_id'],output=str(root),
        SS_model_binding=case['actual_SS_model_binding'],PBR_model_binding=case['actual_PBR_model_binding']))
    seal(caseplan.parent)
    emit(queue,[dict(name=job,resource='exclusive',command=[sys.executable,'-u',str(worker),'--plan',str(caseplan)])])
    emit(states,{job:dict(status='completed',returncode=0)})
    policy=dict(schema=module.SCHEMA,repo=str(Path(module.__file__).resolve().parents[2]),canonical_local_root=str(tmp_path),
        owned_outputs_root=str(args['owned_root']),output_root=str(tmp_path/'journals'),
        case_job_prefixes={arm:'case-'+arm},case_plan_roots={arm:str(caseplan.parent)},case_workers={arm:str(worker)},
        CPU_affinity=[92,93],reader_lock=str(tmp_path/'reader.lock'),manifest_directory=str(tmp_path/'manifests'),
        reader_queues=[dict(jobs=str(queue),states=str(states))],source_SHA256={str(closure):sha(closure),str(worker):sha(worker)},
        closure_command=[sys.executable,str(closure)],python=sys.executable,ssh_argv=['ssh','unit-test-host'],canonical_remote_root='/canonical-test-root')
    emit(policyfile,policy);seal(policyfile.parent)
    plan=dict(schema=module.SCHEMA,policy=dict(path=str(policyfile),SHA256=sha(policyfile),receipt=str(policyfile.parent/'file4-closure.json')),
        name='retire-'+job+'-cache-CPU',arm=arm,case_token=token,case_job=job,case_root=str(root),case_plan=str(caseplan),
        sample_id=case['sample_id'],output=str(tmp_path/'journals'/arm/token))
    emit(planpath,plan);seal(planpath.parent)
    return dict(root=root,plan=planpath,output=Path(plan['output']),retained=retained,
        states=states,job=job,remote=remote,readers=readers,checked=checked)


def test_execute_removes_exactly_three_verified_duplicates_and_journals(tmp_path,monkeypatch):
    a=setup(tmp_path,monkeypatch);before={str(p):sha(p) for p in a['retained']}
    result=module.execute(a['plan'])
    assert result['bytes_retired']==3*len(b'canonical raw')
    assert not any(p.exists() for p in module.candidates(a['root']))
    assert before=={str(p):sha(p) for p in a['retained']}
    assert (a['output']/'before-retirement/file4-closure.json').exists()
    assert len((a['output']/'retired-local-cache.jsonl').read_text().splitlines())==3
    assert (a['output']/'file4-closure.json').exists()


@pytest.mark.parametrize('problem',['remote-hash','late-reader','inode-change','unfinished-job'])
def test_never_unlinks_before_all_checks_pass(tmp_path,monkeypatch,problem):
    a=setup(tmp_path,monkeypatch)
    if problem=='remote-hash':
        def bad(context,rows):
            actual=a['remote'](context,rows);actual[rows[-1]['relative_path']]['sha256']='0'*64;return actual
        monkeypatch.setattr(module,'remote_hashes',bad)
    elif problem=='late-reader':
        def late(context,folder):
            r,q=a['readers'](context,folder)
            return (['actual new reader'] if len(a['checked'])>2 else r),q
        monkeypatch.setattr(module,'reader_snapshot',late)
    elif problem=='inode-change':
        def changed(context,rows):
            actual=a['remote'](context,rows);p=module.candidates(a['root'])[0];s=p.stat()
            os.utime(p,ns=(s.st_atime_ns,s.st_mtime_ns+1_000_000));return actual
        monkeypatch.setattr(module,'remote_hashes',changed)
    else:emit(a['states'],{a['job']:dict(status='FAILED',returncode=1)})
    with pytest.raises(ValueError):module.execute(a['plan'])
    assert all(p.exists() for p in module.candidates(a['root']))
    assert all(p.exists() for p in a['retained'])
    assert not (a['output']/'retired-local-cache.jsonl').exists()
