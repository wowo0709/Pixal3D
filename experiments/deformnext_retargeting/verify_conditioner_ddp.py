"""Two-update DDP synchronization acceptance for the new gates and adapters.

Run using torch.distributed.run with explicitly assigned visible GPUs.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank); torch.set_num_threads(2)
    dist.init_process_group('nccl')
    results = []
    try:
        for width in (1024, 2048):
            for arm in ('b1', 'b2', 'b3', 'c1', 'c2'):
                torch.manual_seed(37 + rank)
                module = DeformationAwareConditioner(arm=arm, key_dim=1024, value_dim=width, query_dim=1024,
                    **({'deformation_dim': 2048} if arm == 'c2' else {})).cuda()
                if module.q2c is not None: module.q2c.requires_grad_(False)
                ddp = DistributedDataParallel(module, device_ids=[rank], find_unused_parameters=False)
                bridge = object.__new__(ImageConditionedProjMixin)
                bridge.deform_conditioner = module
                bridge.training_models = {'deform_conditioner': ddp}
                opt = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=.001)
                data = {'source_values': torch.randn(1, 3, 24, width, device='cuda'),
                        'source_valid': torch.ones(1, 3, 24, device='cuda', dtype=torch.bool),
                        'semantic_indices': torch.arange(64, device='cuda')[None, None].expand(1, 4, -1),
                        'matching_logits': torch.randn(1, 4, 64, device='cuda'),
                        'target_valid': torch.tensor([[True, True, False, True]], device='cuda'),
                        'patch_grid': torch.arange(8, device='cuda').reshape(1, 2, 2, 2) % 4}
                if arm in ('c1', 'c2'): data['target_query'] = torch.randn(1, 4, 1024, device='cuda')
                if arm == 'c2': data['deformation'] = torch.randn(1, 4, 2048, device='cuda')
                native = torch.randn(1, 8, width, device='cuda'); anchor = torch.randn_like(native)
                for step in range(2):
                    opt.zero_grad(set_to_none=True)
                    output = bridge._inject_deform_transport({'proj': native.clone()}, data, None,
                        anchor_condition={'proj': anchor})['proj']
                    if step == 0: torch.testing.assert_close(output, native, rtol=0, atol=0)
                    output.square().mean().backward()
                    assert module.fusion_gate.grad.abs() > 0
                    if module.retargeter is not None: assert module.retargeter.net[-1].weight.grad.abs().sum() > 0
                    opt.step()
                    for parameter in module.parameters():
                        if not parameter.requires_grad: continue
                        reference = parameter.detach().clone(); dist.broadcast(reference, src=0)
                        torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
                results.append({'arm': arm, 'value_dim': width, 'updates': 2, 'parameters_exact_across_ranks': True})
                del ddp, module, opt, bridge
        if dist.get_rank() == 0:
            report = {'status': 'PASS', 'world_size': dist.get_world_size(), 'results': results}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__': main()
