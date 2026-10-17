import json
from types import SimpleNamespace

import pytest
import torch

from experiments.deformnext_retargeting import matching_v2_ss as ss


@pytest.fixture
def stopped_run(tmp_path):
    run=tmp_path/'runs_ss64'/'parent';(run/'ckpts').mkdir(parents=True)
    cfg={'default_output_dir':str(run),'num_gpus':2,'master_port':'29931','wandb_id':'same-run',
        'models':{'denoiser':{'name':'Original'}},'dataset':{'args':{'match_root':'same-head-cache'}},
        'validation':{'every_steps':96},'data_dir':'same-data',
        'trainer':{'args':{'max_steps':480,'batch_size_per_gpu':48,'batch_split':1,'lora':{'rank':16}}},
        'experiment':{'arm':'parent','head_sha256':'frozen-head'}}
    (run/'config.json').write_text(json.dumps(cfg))
    (run/'progress.json').write_text(json.dumps({'stage':'stopped','current':103,'last_checkpoint_step':103}))
    (run/'STOP').touch()
    for name in ['denoiser','deform_conditioner']:
        torch.save({'weight':torch.ones(2)},run/'ckpts'/f'{name}_step0000103.pt')
    misc={'step':103,'optimizer':{'state':{0:{'exp_avg':torch.ones(2)}},'param_groups':[{'lr':1e-5}]},
        'data_sampler':{'epoch':1,'idx':384},'lora':{'rank':16},'lr_scheduler':{},'grad_clip':{}}
    torch.save(misc,run/'ckpts'/'misc_step0000103.pt')
    path=tmp_path/'recipe.json';path.write_text(json.dumps(cfg))
    return tmp_path,run,cfg,path


def test_resume_pins_saved_step_and_preserves_full_state(stopped_run):
    root,run,cfg,_=stopped_run
    _,pinned,receipt=ss.resume_spec(cfg,root)
    assert pinned['ckpt']=='103'
    assert pinned['load_dir']==str(run)
    assert receipt['sampler']=={'epoch':1,'idx':384}
    assert receipt['optimizer_learning_rates']==[1e-5]
    assert 'optimizer' in receipt['restores']
    assert (run/'STOP').exists()


@pytest.mark.parametrize('changed', ['dataset','trainer','models','wandb_id'])
def test_recipe_change_cannot_clear_stop_sentinel(stopped_run,changed):
    root,run,cfg,_=stopped_run
    cfg[changed]={'different':True}
    with pytest.raises(ValueError,match='recipe changed'):
        ss.resume_spec(cfg,root)
    assert (run/'STOP').exists()


def test_incomplete_stop_checkpoint_cannot_resume(stopped_run):
    root,run,cfg,_=stopped_run
    (run/'ckpts'/'deform_conditioner_step0000103.pt').unlink()
    with pytest.raises(ValueError,match='checkpoint set is incomplete'):
        ss.resume_spec(cfg,root)
    assert (run/'STOP').exists()


def test_resume_launches_native_stateful_load_without_fresh_proofs(stopped_run,monkeypatch):
    root,run,cfg,path=stopped_run
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2,3');monkeypatch.setenv('MATCHING_V2_ALLOWED_GPUS','0,1,2,3')
    calls=[]
    def launch(command,**kwargs):
        calls.append(command)
        assert 'train.py' in command
        assert command[command.index('--ckpt')+1]=='103'
        assert command[command.index('--load_dir')+1]==str(run)
        assert not (run/'STOP').exists()
        assert (run/'STOP.before-resume-step0000103').exists()
        resolved=json.loads((run/'resume-config.json').read_text())
        assert resolved['trainer']==cfg['trainer']
        assert resolved['ckpt']=='103'
        (run/'progress.json').write_text(json.dumps({'stage':'completed','current':480}))
    monkeypatch.setattr(ss.subprocess,'run',launch)
    ss.resume(SimpleNamespace(root=root,config=path))
    assert len(calls)==1
    assert json.loads((run/'completed.json').read_text())['resumed_from_step']==103


def test_unassigned_resume_gpu_cannot_remove_stop(stopped_run,monkeypatch):
    root,run,_,path=stopped_run
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','6,7');monkeypatch.setenv('MATCHING_V2_ALLOWED_GPUS','0,1,2,3')
    with pytest.raises(ValueError,match='explicitly assigned'):
        ss.resume(SimpleNamespace(root=root,config=path))
    assert (run/'STOP').exists()
