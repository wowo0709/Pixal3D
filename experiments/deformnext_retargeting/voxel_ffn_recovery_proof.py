"""Audit FFN24→12 on the actual frozen SS weights before original voxel proof.

The scientific global/micro batch remains96 and accumulation remains1. Only
the independent FFN expansion chunks change; no matcher or attention changes.
"""
from contextlib import contextmanager
import json
from pathlib import Path


def audit_ffn(denoiser):
    import torch
    from pixal3d.modules.transformer.blocks import FeedForwardNet
    from pixal3d.modules.checkpointed_ffn_chunks import install_checkpointed_ffn_chunks
    from .unified_design_native_runtime import state_sha
    original = denoiser.blocks[0].mlp
    if not isinstance(original,FeedForwardNet) or any(p.requires_grad for p in original.parameters()):
        raise ValueError('actual frozen native SS FFN required')
    first,_,last = original.mlp
    if (first.in_features,first.out_features,last.out_features) != (1536,8192,1536) or first.weight.device.type != 'cuda':
        raise ValueError('actual assigned GPU and original SS1536/8192 FFN required')
    before = state_sha(original.state_dict().items()); copies = []
    for chunk in [24,12]:
        with torch.device('meta'):copied = FeedForwardNet(1536,8192/1536)
        copied.load_state_dict(original.state_dict(),strict=True,assign=True);copied.requires_grad_(False)
        install_checkpointed_ffn_chunks(copied,chunk_assets=chunk,expected_layers=1);copies.append(copied)
    with torch.random.fork_rng(devices=[first.weight.device]):
        torch.manual_seed(7841);torch.cuda.manual_seed(7841)
        left = torch.randn(25,4096,1536,device=first.weight.device,dtype=torch.bfloat16,requires_grad=True)
        right = left.detach().clone().requires_grad_()
        with torch.autocast('cuda',dtype=torch.bfloat16):expected,actual = copies[0](left),copies[1](right)
        upstream = torch.randn_like(expected)
        (expected.float()*upstream.float()).sum().backward();(actual.float()*upstream.float()).sum().backward()
    errors = {}
    for name,a,b in [('output',actual,expected),('input_gradient',right.grad,left.grad)]:
        a,b = a.detach().float(),b.detach().float()
        relative = float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-30))
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()) or relative > .005:
            raise ValueError('actual FFN24/12 BF16 output or input-gradient differs beyond prespecified relative-L2 .005')
        errors[name] = dict(exact=bool(torch.equal(a,b)),relative_L2=relative,max_abs=float((a-b).abs().max()))
    if state_sha(original.state_dict().items()) != before or any(p.grad is not None for p in original.parameters()):
        raise ValueError('read-only original FFN weights changed during numerical recovery audit')
    return dict(status='PASS_ACTUAL_FFN24_TO12_NATIVE_WEIGHT_BF16_FORWARD_INPUT_GRADIENT',batch=25,tokens=4096,
        channels=1536,hidden=8192,original_chunk=24,recovery_chunk=12,errors=errors,
        actual_original_FFN_SHA256=before,parameter_updates=0,RNG_preserved_by_fork=True,
        scientific_global_batch=96,scientific_microbatch=96,accumulation=1,
        full_training_capacity_not_proven=True,quality_gain_proven=False)


@contextmanager
def audited_constructor(runtime,cfg,audit):
    """Add a read-only GPU audit; delegate every original proof statement."""
    previous = runtime.selected_trainer; calls = []
    def selected(variant):
        if variant != 'FULL_VOXEL_Q' or cfg['trainer']['args']['ffn_chunk_assets'] != 12:
            raise ValueError('only the failed voxel-query FFN12 recovery is authorized by this entry')
        Parent = previous(variant)
        class Audited(Parent):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                if calls:raise ValueError('one actual recovery trainer constructor required')
                report = audit(self.models['denoiser']);calls.append(report)
                with (Path(cfg['output_dir'])/'FFN12-recovery-parity.json').open('x') as stream:
                    json.dump(report,stream,indent=2,allow_nan=False)
        return Audited
    runtime.selected_trainer = selected
    try:
        yield
        if len(calls) != 1:raise ValueError('actual FFN24/12 read-only audit was not executed')
    finally:runtime.selected_trainer = previous


def main():
    import sys
    from . import unified_design_native_runtime as runtime
    if '--phase' not in sys.argv or sys.argv[sys.argv.index('--phase')+1] != 'proof':
        raise ValueError('this entry adds a numerical parity audit only to the original proof phase')
    cfg = json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text())
    with audited_constructor(runtime,cfg,audit_ffn):runtime.main()


if __name__ == '__main__':main()
