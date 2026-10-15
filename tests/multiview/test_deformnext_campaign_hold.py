import json
import sys
import subprocess
from pathlib import Path
import pytest
from experiments.deformnext_retargeting import run_campaign


def recipe(tmp_path, monkeypatch, arm):
    configs=tmp_path/'configs';configs.mkdir()
    verification=tmp_path/'verification/ss64';verification.mkdir(parents=True)
    (verification/'verification.json').write_text(json.dumps({
        'status':'PASS','native_denoiser_max_abs_error':0,
        'arms':[{'arm':name} for name in run_campaign.ARMS]}))
    output=tmp_path/'runs/ss64'/arm
    (configs/f'ss64-{arm}.json').write_text(json.dumps({
        'default_output_dir':str(output),
        'experiment':{'conditioning_schema':'native-mv-anchor-frame-v2','world_size':1},
        'trainer':{'args':{'deform_conditioner':None if arm=='v1' else {'arm':arm},'max_steps':1}}}))
    output.parent.parent.mkdir(parents=True)
    (output.parent.parent/'SEMANTIC_AUDIT_HOLD.json').write_text('{}')
    monkeypatch.setattr(sys,'argv',['campaign','--stage','ss64','--gpus','0','--config-root',str(configs),
        '--verification-root',str(verification.parent),'--log-root',str(tmp_path/'logs'),
        '--master-port','29999','--arms',arm])
    return output


def test_semantic_audit_hold_blocks_before_gpu_or_training_launch(tmp_path, monkeypatch):
    output=recipe(tmp_path,monkeypatch,'b1')
    def forbidden(*args,**kwargs):
        raise AssertionError('held arm must never launch a subprocess')
    monkeypatch.setattr(run_campaign.subprocess,'run',forbidden)
    with pytest.raises(ValueError,match='semantic implementation audit hold'):
        run_campaign.main()
    assert not output.exists()


def test_semantic_hold_allows_unaffected_native_baseline(tmp_path, monkeypatch):
    output=recipe(tmp_path,monkeypatch,'v1');commands=[]
    def complete(command,**kwargs):
        commands.append(command)
        (output/'progress.json').write_text(json.dumps({'stage':'completed','current':1,'total':1}))
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(run_campaign.subprocess,'run',complete)
    run_campaign.main()
    assert len(commands)==1
    assert (tmp_path/'logs/campaign-native-v2-ss64.done').exists()
