import importlib
import importlib.util

import torch
from torch import nn


def lora_api():
    assert importlib.util.find_spec('pixal3d.modules.lora') is not None, 'LoRA module is not implemented'
    return importlib.import_module('pixal3d.modules.lora')


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.to_qkv = nn.Linear(4, 12)
        self.self_attn.to_out = nn.Linear(4, 4)
        self.cross_attn = nn.Module()
        self.cross_attn.cross_attn_block = nn.Module()
        cross = self.cross_attn.cross_attn_block
        cross.to_q = nn.Linear(4, 4)
        cross.to_kv = nn.Linear(4, 8)
        cross.to_out = nn.Linear(4, 4)
        self.cross_attn.proj_linear = nn.Linear(4, 4)
        self.mlp = nn.Linear(4, 4)

    def forward(self, x):
        cross = self.cross_attn.cross_attn_block
        h = self.self_attn.to_out(self.self_attn.to_qkv(x)[..., :4])
        h = h + cross.to_out(cross.to_q(x) + cross.to_kv(x)[..., :4])
        return self.mlp(h + self.cross_attn.proj_linear(x))


class TinyFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([AttentionBlock()])
        self.dtype = torch.float32

    def convert_to(self, dtype):
        self.dtype = dtype
        self.blocks.to(dtype)

    def forward(self, x):
        return self.blocks[0](x.to(self.dtype))


def test_zero_adapter_preserves_base_output_and_checkpoint_keys():
    api = lora_api()
    base = nn.Linear(4, 3)
    x = torch.randn(2, 5, 4)
    expected = base(x).detach()
    layer = api.LoRALinear(base, rank=2, alpha=2, dropout=0.0)
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
    assert layer.weight is base.weight and layer.bias is base.bias
    assert set(layer.state_dict()) == {'weight', 'bias', 'lora_A', 'lora_B'}


def test_adapter_changes_output_without_mutating_frozen_base():
    api = lora_api()
    base = nn.Linear(2, 1, bias=False)
    base.weight.data.copy_(torch.tensor([[2., 3.]]))
    layer = api.LoRALinear(base, rank=1, alpha=2, dropout=0.0)
    layer.lora_A.data.copy_(torch.tensor([[1., -1.]]))
    layer.lora_B.data.fill_(3.)
    torch.testing.assert_close(layer(torch.tensor([[4., 1.]])), torch.tensor([[29.]]))
    torch.testing.assert_close(base.weight, torch.tensor([[2., 3.]]))


def test_all_attention_paths_train_but_mlp_and_base_weights_stay_frozen():
    api = lora_api()
    model = TinyFlow()
    base = {name: p.detach().clone() for name, p in model.named_parameters()}
    api.inject_lora(model, rank=2, alpha=2, dropout=0.0)
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert len(trainable) == 12
    assert all(name.endswith(('lora_A', 'lora_B')) for name in trainable)
    optim = torch.optim.AdamW(trainable.values(), lr=.01)
    model(torch.randn(2, 4)).square().mean().backward()
    assert any(p.grad.abs().sum() > 0 for name, p in trainable.items() if name.endswith('lora_B'))
    optim.step()
    assert any(p.detach().abs().sum() > 0 for name, p in trainable.items() if name.endswith('lora_B'))
    for name, p in model.named_parameters():
        if name in base:
            torch.testing.assert_close(p, base[name], rtol=0, atol=0)


def test_bf16_base_uses_fp32_adapter_gradients_and_inference_works_without_autocast():
    api = lora_api()
    layer = api.LoRALinear(nn.Linear(4, 3).bfloat16(), rank=2, alpha=2, dropout=0.0)
    x = torch.randn(2, 4).bfloat16()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        y = layer(x)
    y.float().square().mean().backward()
    assert y.dtype == torch.bfloat16
    assert layer.lora_A.dtype == layer.lora_B.dtype == torch.float32
    assert layer.lora_B.grad.dtype == torch.float32
    assert layer.weight.grad is None
    with torch.inference_mode():
        assert layer(x).dtype == torch.bfloat16


def test_frozen_inputs_still_train_adapter_with_nonreentrant_checkpoint():
    api = lora_api()
    from torch.utils.checkpoint import checkpoint
    layer = api.LoRALinear(nn.Linear(4, 3), rank=2, alpha=2, dropout=0.0)
    checkpoint(layer, torch.ones(2, 4), use_reentrant=False).square().mean().backward()
    assert layer.lora_B.grad.abs().sum() > 0
