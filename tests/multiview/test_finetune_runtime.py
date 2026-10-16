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


@pytest.mark.parametrize('healthy',[True,False])
def test_pilot_evaluates_then_extends_without_resetting_optimizer_or_rng(tmp_path,monkeypatch,healthy):
    import torch
    t=trainer(tmp_path,step=96,total=96,validation_every=96);calls=[]
    recipe=tmp_path/'recipe.json'
    cfg={'dataset':{'name':'DeformNeXtSS64'},'trainer':{'args':{'max_steps':96}},
         'experiment':{'updates_per_epoch':96,'planned_total_steps':480,'auto_continue_after_pilot':True,
                       'recipe_path':str(recipe)}}
    t._runtime_config=cfg;t.wandb_run=None
    t.deform_conditioner=SimpleNamespace(fusion_gate=torch.tensor(.01))
    t.save=lambda **kw:calls.append('save')
    t.sampler=object();optimizer=t.optimizer;sampler=t.sampler
    initial={'step':0,'quick_panel_ids':['a'],'quick_metrics':{'voxel_iou':.45}}
    (tmp_path/'validation.jsonl').write_text(json.dumps(initial)+'\n')
    def evaluate(self):
        calls.append('eval')
        row={'step':96,'quick_panel_ids':['a'],'quick_metrics':{'voxel_iou':.5 if healthy else .2}}
        with (tmp_path/'validation.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
    monkeypatch.setattr(runtime,'evaluate',evaluate)
    monkeypatch.setattr(runtime,'_native_check_abort',lambda self:None)
    rng=torch.get_rng_state().clone()
    runtime.check_abort(t)
    assert calls==['save','eval'] and t.optimizer is optimizer and t.sampler is sampler
    assert torch.equal(torch.get_rng_state(),rng)
    if healthy:
        assert t.max_steps==t._runtime_total_steps==480
        assert json.loads(recipe.read_text())['experiment']['release_status']=='pilot-validated'
        assert json.loads((tmp_path/'progress.json').read_text())['stage']=='running'
    else:
        assert t.max_steps==96 and (tmp_path/'PILOT_HOLD.json').exists()


def test_ddp_pilot_decision_is_consumed_on_every_rank_before_final_barrier(tmp_path, monkeypatch):
    import copy
    import torch
    cfg={'dataset':{'name':'DeformNeXtSS64'},'trainer':{'args':{'max_steps':96}},
         'experiment':{'updates_per_epoch':96,'planned_total_steps':480,
                       'auto_continue_after_pilot':True,'recipe_path':str(tmp_path/'recipe.json')}}
    records=[{'step':step,'quick_panel_ids':['a'],'quick_metrics':{'voxel_iou':iou}}
             for step,iou in [(0,.45),(96,.50)]]
    (tmp_path/'validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    active={'rank':0};decisions={};collectives={rank:[] for rank in range(3)}
    def broadcast(value,src):
        rank=active['rank'];collectives[rank].append('pilot_broadcast')
        if rank==src:decisions['extension']=int(value)
        else:value.fill_(decisions['extension'])
    monkeypatch.setattr(runtime,'_native_check_abort',lambda self:None)
    monkeypatch.setattr(runtime,'evaluate',lambda self:None)
    monkeypatch.setattr(runtime.dist,'all_reduce',lambda *a,**kw:None)
    monkeypatch.setattr(runtime.dist,'barrier',lambda:collectives[active['rank']].append('barrier'))
    monkeypatch.setattr(runtime.dist,'broadcast',broadcast)
    workers=[]
    for rank in range(3):
        active['rank']=rank
        t=trainer(tmp_path,step=96,total=96,validation_every=96)
        t.rank=rank;t.is_master=rank==0;t.world_size=3
        t._runtime_config=copy.deepcopy(cfg);t.wandb_run=None;t.save=lambda **kw:None
        runtime.check_abort(t)
        assert t.max_steps==t._runtime_total_steps==480
        assert t._runtime_config['experiment']['auto_continue_after_pilot'] is False
        workers.append(t)
    # Rank zero used to skip the pilot branch here while ranks 1/2 entered
    # another broadcast, deadlocking against rank zero's completion barrier.
    for rank,t in enumerate(workers):
        active['rank']=rank;t.step=480
        runtime.check_abort(t)
    assert all(events.count('pilot_broadcast')==1 for events in collectives.values())
    assert all(events==collectives[0] for events in collectives.values())
    assert json.loads((tmp_path/'progress.json').read_text())['stage']=='completed'
