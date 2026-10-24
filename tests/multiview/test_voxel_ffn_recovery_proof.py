"""Constructor delegation only; actual FFN numerical audit requires assigned GPU."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting.voxel_ffn_recovery_proof import audited_constructor


@pytest.mark.parametrize('behavior',['success','missing','twice','audit-failure','constructor-failure','wrong-arm','wrong-chunk'])
def test_recovery_audit_is_read_only_scoped_and_cannot_replace_the_original_proof(tmp_path,behavior):
    model=object();events=[]
    class Parent:
        def __init__(self):
            events.append('original-constructor')
            if behavior=='constructor-failure':raise RuntimeError('original failure')
            self.models={'denoiser':model}
    original=lambda _:Parent;runtime=SimpleNamespace(selected_trainer=original)
    cfg=dict(output_dir=str(tmp_path),trainer={'args':{'ffn_chunk_assets':24 if behavior=='wrong-chunk' else 12}})
    def audit(actual):
        assert actual is model;events.append('read-only-audit')
        if behavior=='audit-failure':raise RuntimeError('GPU numerical audit failed')
        return dict(status='synthetic metadata fixture, not an actual GPU audit')
    def run():
        with audited_constructor(runtime,cfg,audit):
            if behavior=='missing':return
            trainer=runtime.selected_trainer('FULL_QK_ONLY' if behavior=='wrong-arm' else 'FULL_VOXEL_Q')
            trainer()
            if behavior=='twice':trainer()
            events.append('original-proof-continues')
    if behavior=='success':
        run();assert events==['original-constructor','read-only-audit','original-proof-continues']
        assert (tmp_path/'FFN12-recovery-parity.json').is_file()
    else:
        with pytest.raises(RuntimeError if behavior in ['audit-failure','constructor-failure'] else ValueError):run()
    assert runtime.selected_trainer is original
