"""Cross-queue metadata transactions; never construct a final freeze or NN."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.deformnext_retargeting import prepare_selected_final_campaign as module


def setup_queues(tmp_path):
    main=tmp_path/'main';aux=tmp_path/'aux';main.mkdir();aux.mkdir()
    old=dict(name='actual-prior-descriptor',command=['preserved-argv'],depends_on=['old-failed-run'])
    for folder in [main,aux]:
        (folder/'jobs.json').write_text(json.dumps([old]));(folder/'states.json').write_text('{"old-failed-run":{"status":"FAILED"}}')
    group=lambda folder,name,gpus:dict(jobs=str(folder/'jobs.json'),states=str(folder/'states.json'),job_name=name,assigned_GPUs=gpus)
    ctx=dict(python='/explicit/python',campaign_root=str(tmp_path/'runtime'),keeper_groups={
        'main':group(main,'hold-main',['0','1','2','3']),'aux':group(aux,'hold-aux',['6','7'])},
        coordinator_queue=group(main,'coordinator',[]))
    path=tmp_path/'closed-context/context.json';activation=tmp_path/'activation/file4-closure.json'
    return ctx,path,activation,old


def test_both_dispatchers_and_coordinator_wait_on_one_not_yet_created_activation(tmp_path):
    ctx,path,activation,old=setup_queues(tmp_path);jobs=module.descriptors(ctx,path,activation)
    assert len(jobs)==3 and jobs['hold-main']['gpus']==['0','1','2','3'] and jobs['hold-aux']['gpus']==['6','7']
    assert jobs['coordinator']['resource']=='cpu'
    assert all(j['requires']==[str(path.parent/'file4-closure.json'),str(activation)] for j in jobs.values())
    assert not activation.exists()


def test_append_preserves_every_prior_descriptor_and_never_writes_dispatcher_states(tmp_path):
    ctx,path,activation,old=setup_queues(tmp_path);jobs=module.descriptors(ctx,path,activation)
    ledgers=[Path(g['states']) for g in ctx['keeper_groups'].values()];states={str(p):p.read_bytes() for p in ledgers}
    result=module.append_queues(ctx,jobs)
    for group in ctx['keeper_groups'].values():
        rows=json.load(open(group['jobs']));assert rows[0]==old
        assert group['job_name'] in {j['name'] for j in rows}
    assert len(json.load(open(ctx['coordinator_queue']['jobs'])))==3
    assert len(result)==2 and all(Path(p).read_bytes()==value for p,value in states.items())
    # Appending alone is not activation and does not prove an actual keeper or
    # any actual final native case has run.
    assert not activation.exists()


def test_duplicate_names_refuse_both_queue_writes(tmp_path):
    ctx,path,activation,old=setup_queues(tmp_path);jobs=module.descriptors(ctx,path,activation)
    target=Path(ctx['keeper_groups']['aux']['jobs']);target.write_text(json.dumps([old,jobs['hold-aux']]))
    before={g['jobs']:Path(g['jobs']).read_bytes() for g in ctx['keeper_groups'].values()}
    with pytest.raises(ValueError,match='duplicate'):module.append_queues(ctx,jobs)
    assert all(Path(p).read_bytes()==v for p,v in before.items()) and not activation.exists()


@pytest.mark.parametrize('damage',['second-write-fails','concurrent-second-append'])
def test_partial_cross_queue_install_cannot_publish_activation(tmp_path,monkeypatch,damage):
    ctx,path,activation,old=setup_queues(tmp_path);jobs=module.descriptors(ctx,path,activation);actual=module.campaign.atomic
    paths=[g['jobs'] for g in ctx['keeper_groups'].values()]
    def atomic(path,value):
        if str(path)==paths[1] and damage=='second-write-fails':raise OSError('fixture second filesystem write failed')
        actual(path,value)
        if str(path)==paths[0] and damage=='concurrent-second-append':
            other=Path(paths[1]);rows=json.load(open(other));other.write_text(json.dumps(rows+[dict(name='other-actual-appender')]))
    monkeypatch.setattr(module.campaign,'atomic',atomic)
    with pytest.raises((ValueError,OSError)):module.append_queues(ctx,jobs)
    installed=json.load(open(paths[0]));assert 'hold-main' in {j['name'] for j in installed}
    assert all(str(activation) in j['requires'] for j in installed if j['name']!='actual-prior-descriptor')
    assert not activation.exists()


def test_preparer_refuses_fixture_before_any_queue_or_model_execution(tmp_path,monkeypatch):
    path=tmp_path/'fixture.json';path.write_text('{"schema":"test-fixture-only"}')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    with pytest.raises(ValueError):module.prepare(path)
