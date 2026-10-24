"""Real tiny local/mirror files and CPU process handles; no NN or file4 deletion."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import psutil
import pytest

from experiments.deformnext_retargeting import retire_selected_final_raw as module
from experiments.deformnext_retargeting.cascade_view_roles import file_sha256 as sha
from experiments.deformnext_retargeting.metric_image_identity import scored_image_observation
from test_selected_final_raw_retirement import candidate


REPO=Path(__file__).resolve().parents[2]


def prepare(tmp_path,monkeypatch,damage=None):
    base=tmp_path/'local';base.mkdir();mirror=tmp_path/'canonical-mirror';mirror.mkdir()
    raw,args=candidate(base);root=args['case_root'];parent=args['parent'];controls=args['controls'];cases=args['consumers']
    helper=base/'closure-helper.py';helper.write_text('# test double, never executes a remote command\n')
    pins={p:sha(p) for p in [*module.required_sources(),str(helper)]};parent['evaluation_protocol']['source_SHA256'].update(pins)
    receipt=base/'file4-closure.json'
    def emit(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        return dict(path=str(path),SHA256=sha(path),receipt=str(receipt))
    parent_spec=emit(base/'frozen/parent.json',parent);control_spec=emit(base/'frozen/controls.json',controls)
    workers={};live=[]
    for name,case in cases.items():
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.1)'],start_new_session=True)
        birth=psutil.Process(child.pid).create_time();child.wait()
        terminal=dict(PID=child.pid,birth=birth,session_ID=child.pid,returncode=0,actual_session_drained=True,case_output=case['case_output'])
        case['terminals']=[deepcopy(terminal)]
        case.update(parent_freeze={k:parent_spec[k] for k in ['path','SHA256']},control_freeze={k:control_spec[k] for k in ['path','SHA256']})
        observed=[];shared=Path(case['case_output'])/'shared';shared.mkdir(parents=True)
        for cid in [8,9,10]:
            path=shared/f'camera{cid:02d}.npz';arrays=dict(prediction=np.full((3,512,512),.25,np.float32),
                GT_shared_renderer=np.full((3,512,512),.5,np.float32),GT_alpha=np.ones((512,512),np.float32))
            np.savez_compressed(path,**arrays)
            observed.append(scored_image_observation(arrays,sample_id=case['sample_id'],camera_id=cid,
                source_cluster_id=case['source_cluster_id'],archive=str(path),archive_SHA256=sha(path)))
        case['scored_image_observations']=observed
        workers[name]=terminal
    own=cases['V1'];geometry=root/'geometry';(geometry/'geometry-evaluation.npz').write_bytes(b'synthetic scoring raw')
    emit(geometry/'generation.json',args['generation']);(geometry/'native-shape-guides.pt').write_bytes(b'retained guide')
    pbr=root/'PBR';pbr.mkdir()
    for name in ['prediction.npz','geometry-evaluation.npz']:(pbr/name).write_bytes(b'synthetic later PBR raw')
    gen=deepcopy(args['generation']);gen['status']='PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE';emit(pbr/'generation.json',gen)
    audits={name:emit(Path(c['case_output'])/'audit.json',c) for name,c in cases.items()}
    terminals={name:emit(base/'workers'/name/'terminal.json',t) for name,t in workers.items()}
    queue=emit(base/'ledger/jobs.json',[]);states=emit(base/'ledger/states.json',{})
    manifests=base/'manifests';manifests.mkdir();reservation=tmp_path/'disk-reservations.json'
    emit(reservation,dict(version=1,runs={'owned-case':dict(native_output=str(root),wrapper_output=str(root),released=False,
        owner={k:workers['V1'][k] for k in ['PID','birth']},checkpoint_budget_bytes=1000)}))
    ctx=dict(schema=module.SCHEMA,name='retire-geometry',repo=str(REPO),python=sys.executable,canonical_local_root=str(base),
        owned_outputs_root=str(args['owned_root']),output=str(base/'retirements/geometry'),alias='V1',role='geometry',
        CPU_affinity=sorted(__import__('os').sched_getaffinity(0)),source_SHA256=pins,closure_command=[sys.executable,str(helper)],
        manifest_directory=str(manifests),disk_reservations=str(reservation),disk_reservation_name='owned-case',workspace_bytes=1,
        ssh_argv=['ssh','test-mirror-not-contacted'],canonical_remote_root='/synthetic-test-mirror',
        reader_queues=[dict(jobs=queue['path'],states=states['path'])],
        source_inputs=dict(parent_freeze=parent_spec,control_freeze=control_spec,case_audits=audits,worker_terminals=terminals))
    path=base/'plans/geometry/context.json';emit(path,ctx)
    keep=base/'unrelated-checkpoint.pt';keep.write_bytes(b'never selected')
    def receipt_for(files):
        return dict(status='PASS',local_equals_actual_file4_SHA256=True,
            files=[dict(relative_path=str(p.relative_to(base)),sha256=sha(p),bytes=p.stat().st_size) for p in files])
    all_files=[p for p in base.rglob('*') if p.is_file()]
    payload=receipt_for(all_files);receipt.write_text(json.dumps(payload))
    for folder in [path.parent,geometry,pbr,root,*[Path(c['case_output'])/'shared' for c in cases.values()]]:
        (folder/'file4-closure.json').write_text(json.dumps(payload))
    for p in base.rglob('*'):
        if p.is_file():
            target=mirror/p.relative_to(base);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    remote_calls=[]
    def remote(context,rows):
        remote_calls.append(deepcopy(rows))
        if damage=='remote-failure':raise RuntimeError('canonical mirror unavailable')
        if damage=='new-reader-plan':Path(queue['path']).write_text('[{"name":"newly-added-reader"}]')
        if damage=='changed-inode':
            blob=raw.read_bytes();raw.unlink();raw.write_bytes(blob)
        values={r['relative_path']:dict(sha256=sha(mirror/r['relative_path']),bytes=(mirror/r['relative_path']).stat().st_size) for r in rows}
        if damage=='wrong-remote':next(iter(values.values()))['sha256']='f'*64
        return values
    def close(folder,context,label):
        if damage=='intent-closure' and label=='intent':raise RuntimeError('canonical intent closure failed')
        folder=Path(folder);files=[p for p in folder.rglob('*') if p.is_file() and p!=folder/'file4-closure.json']
        payload=receipt_for(files)
        for p in files:
            target=mirror/p.relative_to(base);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        (folder/'file4-closure.json').write_text(json.dumps(payload))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setattr(module,'remote_hashes',remote);monkeypatch.setattr(module,'close',close)
    # A deliberately live worker must prevent even a remote verification call.
    if damage=='live-worker':
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True);live.append(child)
        own_worker=Path(terminals['V1']['path']);t=dict(workers['V1'],PID=child.pid,birth=psutil.Process(child.pid).create_time(),session_ID=child.pid)
        own_worker.write_text(json.dumps(t));ctx['source_inputs']['worker_terminals']['V1']['SHA256']=sha(own_worker)
        path.write_text(json.dumps(ctx));payload=receipt_for([p for p in base.rglob('*') if p.is_file() and p.name!='file4-closure.json'])
        receipt.write_text(json.dumps(payload));(path.parent/'file4-closure.json').write_text(json.dumps(payload))
    return path,ctx,raw,args,mirror,remote_calls,live,close


def test_real_raw_files_wait_for_all_consumers_and_two_stages_release_only_their_reservation(tmp_path,monkeypatch):
    path,ctx,raw,args,mirror,calls,live,close=prepare(tmp_path,monkeypatch)
    original=raw.read_bytes();result=module.execute(path)
    assert not raw.exists() and not (raw.parent/'geometry-evaluation.npz').exists()
    assert (mirror/raw.relative_to(Path(ctx['canonical_local_root']))).read_bytes()==original
    assert not result['actual_own_disk_reservation_released'] and len(result['remaining_own_raw_files'])==2
    assert (raw.parent/'native-shape-guides.pt').exists()
    assert (Path(ctx['canonical_local_root'])/'unrelated-checkpoint.pt').exists()
    assert all((Path(c['case_output'])/'shared/camera08.npz').exists() for c in args['consumers'].values())
    # A separate closed PBR context reads the immutable original receipts. It
    # does not recreate or overwrite the already retired geometry files.
    second=deepcopy(ctx);second.update(name='retire-own-PBR',role='PBR',output=str(Path(ctx['canonical_local_root'])/'retirements/PBR'))
    for name in ['case_audits','worker_terminals']:second['source_inputs'][name]={'V1':second['source_inputs'][name]['V1']}
    second_path=Path(ctx['canonical_local_root'])/'plans/PBR/context.json';second_path.parent.mkdir();second_path.write_text(json.dumps(second))
    close(second_path.parent,second,'new-plan')
    result=module.execute(second_path)
    assert result['actual_own_disk_reservation_released'] and not result['remaining_own_raw_files']
    record=json.loads(Path(ctx['disk_reservations']).read_text());assert record['runs']['owned-case']['released'] is True
    assert (mirror/args['case_root'].relative_to(Path(ctx['canonical_local_root']))/'PBR/prediction.npz').is_file()
    assert len(calls)==2


@pytest.mark.parametrize('damage',['remote-failure','wrong-remote','intent-closure','new-reader-plan','changed-inode','live-worker'])
def test_missing_canonical_proof_reader_changes_or_live_workers_never_delete_raws(tmp_path,monkeypatch,damage):
    path,ctx,raw,args,mirror,calls,live,close=prepare(tmp_path,monkeypatch,damage)
    try:
        with pytest.raises((ValueError,RuntimeError)):module.execute(path)
        assert raw.is_file() and (raw.parent/'geometry-evaluation.npz').is_file()
        assert json.loads(Path(ctx['disk_reservations']).read_text())['runs']['owned-case']['released'] is False
        if damage=='live-worker':assert not calls
    finally:
        for child in live:child.terminate();child.wait()


def test_actual_open_raw_handle_prevents_retirement_before_remote_IO(tmp_path,monkeypatch):
    path,ctx,raw,args,mirror,calls,live,close=prepare(tmp_path,monkeypatch)
    with raw.open('rb'):
        with pytest.raises(ValueError):module.execute(path)
    assert raw.exists() and not calls


def test_actual_pending_raw_consumer_prevents_retirement_before_remote_IO(tmp_path,monkeypatch):
    path,ctx,raw,args,mirror,calls,live,close=prepare(tmp_path,monkeypatch)
    future=Path(ctx['canonical_local_root'])/'reader-plan.json';future.write_text(json.dumps(dict(generated_shape_root=str(raw.parent))))
    queue=ctx['reader_queues'][0];Path(queue['jobs']).write_text(json.dumps([dict(name='future-replay',command=['python','--plan',str(future)])]))
    Path(queue['states']).write_text(json.dumps({'future-replay':{'status':'pending'}}))
    with pytest.raises(ValueError):module.execute(path)
    assert raw.exists() and not calls


@pytest.mark.parametrize('damage',['role','namespace','missing-source','relative-remote','bad-closure','empty-ledgers','CPU'])
def test_retirement_requires_explicit_owned_namespace_sources_connection_and_reader_ledgers(tmp_path,monkeypatch,damage):
    path,ctx,*_=prepare(tmp_path,monkeypatch)
    if damage=='role':ctx['role']='weights'
    elif damage=='namespace':ctx['output']=ctx['owned_outputs_root']+'/inside-closed-case'
    elif damage=='missing-source':ctx['source_SHA256'].pop(module.required_sources()[0])
    elif damage=='relative-remote':ctx['canonical_remote_root']='relative'
    elif damage=='bad-closure':ctx['closure_command'][0]='/wrong/python'
    elif damage=='empty-ledgers':ctx['reader_queues']=[]
    else:ctx['CPU_affinity']=[]
    with pytest.raises(ValueError):module.validate_context(ctx)
