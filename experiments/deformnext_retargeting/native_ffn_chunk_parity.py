"""A real frozen SS FFN forward/input-gradient check on its assigned GPU."""
import torch


def audit_native_ffn_chunk_parity(denoiser):
    from pixal3d.modules.transformer.blocks import FeedForwardNet
    from pixal3d.modules.checkpointed_ffn_chunks import install_checkpointed_ffn_chunks

    original = denoiser.blocks[0].mlp
    if not isinstance(original, FeedForwardNet) or any(value.requires_grad for value in original.parameters()):
        raise ValueError('original frozen dense SS FFN required')
    first, _, last = original.mlp
    if (first.in_features,first.out_features,last.out_features)!=(1536,8192,1536):
        raise ValueError('explicit original SS1536/8192 FFN required')
    with torch.device('meta'):
        copied = FeedForwardNet(1536,8192/1536)
    copied.load_state_dict(original.state_dict(),strict=True,assign=True)
    copied.requires_grad_(False)
    install_checkpointed_ffn_chunks(copied,chunk_assets=2,expected_layers=1)
    device = first.weight.device
    if device.type!='cuda':
        raise ValueError('actual assigned GPU required for native BF16 parity')
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(7841);torch.cuda.manual_seed(7841)
        left = torch.randn(4,4096,1536,device=device,dtype=torch.bfloat16,requires_grad=True)
        right = left.detach().clone().requires_grad_()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            expected = original(left)
            actual = copied(right)
        upstream = torch.randn_like(expected)
        (expected.float()*upstream.float()).sum().backward()
        (actual.float()*upstream.float()).sum().backward()
    errors={}
    for name,a,b in [('output',actual,expected),('input_gradient',right.grad,left.grad)]:
        a,b=a.detach().float(),b.detach().float()
        relative=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-30))
        errors[name]=dict(exact=bool(torch.equal(a,b)),max_abs=float((a-b).abs().max()),relative_L2=relative)
        if not torch.isfinite(a).all() or relative>0.005:
            raise ValueError('native BF16 FFN chunk numerical change exceeds relative-L2 limit')
    if any(value.grad is not None for value in original.parameters()):
        raise ValueError('frozen native FFN weights cannot receive gradients')
    return dict(status='PASS_ACTUAL_NATIVE_SS_BF16_FFN_CHUNK_FORWARD_AND_INPUT_GRADIENT',
        batch=4,tokens=4096,channels=1536,hidden=8192,chunk_assets=2,
        original_checkpoint_weight_storage_shared_readonly=True,parameter_updates=0,
        errors=errors,capacity_for_micro96_not_proven=True)
