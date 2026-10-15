"""Compare vendored promoted scorer to an explicit historical authority source."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
from pixal3d.modules.r9_matching import AUTHORITY_SHA256, PromotedR9Matcher


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--authority', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = p.parse_args()
    torch.set_num_threads(4)
    digest = hashlib.sha256(args.authority.read_bytes()).hexdigest()
    if digest != AUTHORITY_SHA256:
        raise ValueError('Historical authority checksum differs')
    spec = importlib.util.spec_from_file_location('r9_historical_authority', args.authority)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    results = []
    for seed in (42017, 42018, 42019):
        torch.manual_seed(seed)
        ref = original.R9Scorer(decoder_arm='D1-Q2C').to(args.device)
        torch.manual_seed(seed)
        port = PromotedR9Matcher().to(args.device)
        assert ref.state_dict().keys() == port.state_dict().keys()
        for key in ref.state_dict():
            torch.testing.assert_close(port.state_dict()[key], ref.state_dict()[key], rtol=0, atol=0)
        for learned in (False, True):
            if learned:
                # Exercise contextual attention and scalar corrections as well
                # as the zero-start shortcut. Apply identical nonzero states.
                with torch.no_grad():
                    for model in (ref, port):
                        model.decoder.blocks[0].attention_gate.fill_(.31)
                        model.decoder.blocks[0].ffn_gate.fill_(-.17)
                        model.fusion_head.output.weight.fill_(.023)
                        model.fusion_head.output.bias.fill_(.011)
            torch.manual_seed(seed + 1)
            query = torch.randn(5, 1024, device=args.device)
            source = torch.randn(100, 1024, device=args.device)
            ids = torch.stack([torch.randperm(100, device=args.device)[:64] for _ in query])
            raw = torch.randn(5, 64, device=args.device)
            ref.zero_grad(set_to_none=True)
            port.zero_grad(set_to_none=True)
            expected = ref({12: query}, {12: source}, ids, raw_scores=raw)['scores']
            actual = port({12: query}, {12: source}, ids, raw_scores=raw)['scores']
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            weight = torch.randn_like(actual)
            (expected * weight).sum().backward()
            (actual * weight).sum().backward()
            for (name, a), (_, b) in zip(ref.named_parameters(), port.named_parameters()):
                assert (a.grad is None) == (b.grad is None), name
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            # The memory-bounded bridge uses the same math, allowing only
            # GEMM batch-shape rounding across different query chunks.
            batched = port.score_candidates(query[None], source[ids][None], raw[None], chunk_size=2)[0]
            torch.testing.assert_close(batched, expected, rtol=1e-5, atol=1e-6)
            results.append({'seed': seed, 'learned_state': learned,
                            'direct_output_max_abs_error': float((actual - expected).abs().max().detach()),
                            'state_and_parameter_gradients_exact': True,
                            'chunked_max_abs_error': float((batched - expected).abs().max().detach())})
    report = {'status': 'PASS', 'authority_sha256': digest, 'device': args.device,
              'parameter_count': sum(p.numel() for p in port.parameters()), 'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
