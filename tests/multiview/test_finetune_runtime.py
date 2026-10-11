import json
from types import SimpleNamespace
import pytest

from experiments.multiview_finetune import runtime


def trainer(tmp_path, step=500, total=4210, validation_every=842):
    return SimpleNamespace(output_dir=str(tmp_path), step=step, max_steps=total,
        _runtime_total_steps=total, _runtime_validation_every=validation_every,
        _runtime_checkpoint_step=None, is_master=True, world_size=1, rank=0,
        batch_size=96, device='cpu', optimizer=SimpleNamespace(param_groups=[{'lr':1e-4}]), _runtime_last_row={'loss': {'loss': .2}})


def test_explicit_gpu_count_matches_visible_assignment(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,3,4')
    assert runtime.assigned_gpu_count(3) == 3
    with pytest.raises(ValueError, match='visible'):
        runtime.assigned_gpu_count(1)


def test_periodic_checkpoint_does_not_force_validation(tmp_path, monkeypatch):
    t=trainer(tmp_path); calls=[]
    monkeypatch.setattr(runtime, '_native_save', lambda self, **kw: calls.append(('save', kw)))
    monkeypatch.setattr(runtime, 'evaluate', lambda self: calls.append(('validation', {})))
    runtime.save(t)
    assert calls == [('save', {'non_blocking': False})]


def test_epoch_validation_and_final_save(tmp_path, monkeypatch):
    t=trainer(tmp_path, step=842); calls=[]
    t.save=lambda **kw: calls.append('save')
    monkeypatch.setattr(runtime, '_native_check_abort', lambda self: None)
    monkeypatch.setattr(runtime, 'evaluate', lambda self: calls.append('eval'))
    runtime.check_abort(t)
    assert calls == ['save','eval']
    t.step=t.max_steps=4210; calls.clear();runtime.check_abort(t)
    assert calls == ['save','eval']
    state=json.loads((tmp_path/'progress.json').read_text())
    assert state['current']==state['total']==4210 and state['stage']=='completed'


def test_stop_after_update_saves_without_validation(tmp_path, monkeypatch):
    t=trainer(tmp_path, step=17); calls=[];t.save=lambda **kw: calls.append('save')
    monkeypatch.setattr(runtime, '_native_check_abort', lambda self: None)
    monkeypatch.setattr(runtime, 'evaluate', lambda self: pytest.fail('stop must not run validation'))
    (tmp_path/'STOP').touch();runtime.check_abort(t)
    assert calls==['save'] and t.max_steps==17
    state=json.loads((tmp_path/'progress.json').read_text())
    assert state['stage']=='stopped' and state['current']==17 and state['total']==4210


def test_validation_restores_ddp_wrapper_on_failure(tmp_path, monkeypatch):
    import torch
    from pixal3d import datasets
    model=torch.nn.Linear(1,1);wrapped=object();mapping={'denoiser':wrapped}
    t=SimpleNamespace(output_dir=str(tmp_path), training_models=mapping,
        models={'denoiser':model}, p_uncond=.1)
    (tmp_path/'roots.json').write_text('{}')
    (tmp_path/'config.json').write_text(json.dumps({'validation':{
        'dataset_name':'FailDataset','roots_file':str(tmp_path/'roots.json'),
        'dataset_args':{},'assets_per_source':1}}))
    def fail(*args,**kwargs):
        assert t.training_models['denoiser'] is model
        raise RuntimeError('fixture data failure')
    monkeypatch.setattr(datasets,'FailDataset',fail,raising=False)
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda:torch.zeros(1,dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda,'set_rng_state',lambda state:None)
    with pytest.raises(RuntimeError,match='fixture data failure'):
        runtime.evaluate(t)
    assert t.training_models is mapping and model.training and t.p_uncond==.1
