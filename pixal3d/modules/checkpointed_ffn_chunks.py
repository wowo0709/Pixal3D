"""Per-instance native dense FFN memory control without changing parameters.

Chunking is internal to the FFN, not gradient accumulation. The native outer
DiT checkpoint remains enabled; each FFN chunk also checkpoints its expansion
so its GELU input need not remain live for all batch members simultaneously.
GPU capacity and throughput still require measurement with the real model.
"""
from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint


def install_checkpointed_ffn_chunks(model, *, chunk_assets, expected_layers=30):
    from .transformer.blocks import FeedForwardNet
    if isinstance(chunk_assets, bool) or not isinstance(chunk_assets, int) or chunk_assets < 1:
        raise ValueError('positive integer FFN asset chunk required')
    selected = [(name, module) for name, module in model.named_modules()
                if isinstance(module, FeedForwardNet)]
    if len(selected) != expected_layers:
        raise ValueError('explicit native dense FFN layer count required')
    for name, module in selected:
        if hasattr(module, '_native_ffn_chunk_assets'):
            raise ValueError('FFN chunking is already installed: ' + name)
        if (len(module.mlp) != 3 or not isinstance(module.mlp[0], torch.nn.Linear)
                or not isinstance(module.mlp[1], torch.nn.GELU)
                or not isinstance(module.mlp[2], torch.nn.Linear)):
            raise ValueError('native independent Linear/GELU/Linear FFN required')
    before_keys = tuple(model.state_dict())
    before_parameters = {name: id(parameter) for name, parameter in model.named_parameters()}
    for _, module in selected:
        original = module.forward

        def forward(self, x, _original=original):
            if x.ndim != 3:
                raise ValueError('dense FFN expects [batch, tokens, channels]')
            if not torch.is_grad_enabled() or x.shape[0] <= chunk_assets:
                return _original(x)
            return torch.cat([checkpoint(_original, part, use_reentrant=False)
                              for part in x.split(chunk_assets, dim=0)], dim=0)

        module._native_ffn_chunk_assets = chunk_assets
        module.forward = MethodType(forward, module)
    if (before_keys != tuple(model.state_dict())
            or before_parameters != {name: id(parameter) for name, parameter in model.named_parameters()}):
        raise ValueError('FFN chunking must preserve parameter identities and checkpoint keys')
    return dict(status='INSTALLED_NATIVE_FFN_CHUNKS_GPU_CAPACITY_NOT_YET_MEASURED',
                chunk_assets=chunk_assets, layers=[name for name, _ in selected],
                parameter_ids_and_state_dict_keys_unchanged=True,
                outer_native_checkpoint_preserved=True, inner_checkpoint=True,
                global_batch_or_accumulation_changed=False)
