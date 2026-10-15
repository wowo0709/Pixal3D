"""Compare real cached/online R9 routing under AMP with an active C2 adapter."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--match-root', type=Path, required=True)
    parser.add_argument('--q2c-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from pixal3d import datasets
    from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
    from pixal3d.utils.data_utils import recursive_to_device
    from .precompute_matches import sha256, valid_cache
    torch.set_num_threads(4); torch.cuda.set_device(0)
    results = []
    for stage, width in (('ss64', 1024), ('pbr1024', 2048)):
        cfg = json.loads((args.config_root / f'{stage}-v1.json').read_text())
        kwargs = dict(cfg['validation']['dataset_args'])
        kwargs.update(transport_arm='c2', transport_implementation='r9-native-v5',
                      expected_q2c_sha256=sha256(args.q2c_checkpoint),
                      projection_root=str(Path(cfg['validation']['roots_file']).parent.parent / 'projection_maps_world_v4'))
        cached = getattr(datasets, cfg['dataset']['name'])(str(args.manifest), **kwargs, match_root=str(args.match_root))
        online = getattr(datasets, cfg['dataset']['name'])(str(args.manifest), **kwargs, match_root=None)
        module = DeformationAwareConditioner(arm='c2', key_dim=1024, value_dim=width,
            query_dim=1024, deformation_dim=2048, q2c_checkpoint=str(args.q2c_checkpoint), freeze_q2c=True).cuda().eval()
        with torch.no_grad(): module.retargeter.net[-1].weight.normal_(std=.003)
        for index in range(len(cached)):
            assert valid_cache(args.match_root / f"{cached.rows[index]['sample_id']}.npz", kwargs['expected_q2c_sha256'], 64)
            def call(ds):
                sample = ds.collate_fn([ds[index]])
                pack = recursive_to_device(sample, 'cuda')
                return {key.removeprefix('transport_'): value for key, value in pack.items()
                        if key.startswith('transport_') and key not in {'transport_patch_grid', 'transport_patch_ids'}}
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                a, b = module(**call(cached)), module(**call(online))
                assert torch.equal(a.candidate_indices, b.candidate_indices)
                for key in ('evidence', 'matching_logits', 'candidate_weights', 'view_weights', 'retarget_residual'):
                    torch.testing.assert_close(getattr(a, key), getattr(b, key), rtol=0, atol=1e-5)
            results.append({'stage': stage, 'sample_id': cached.rows[index]['sample_id'],
                            'evidence_max_abs_error': float((a.evidence-b.evidence).abs().max()),
                            'adapter_nonzero': bool(a.retarget_residual.abs().sum() > 0)})
        del module
    report = {'status': 'PASS', 'amp': 'bf16', 'active_adapter': True, 'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
