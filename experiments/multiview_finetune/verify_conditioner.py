"""Compare real K=1 MV conditioning with the existing SV path on held-out data."""
import argparse
import json
import os
from pathlib import Path
import sys

from configure import REPO, STAGES
from paths import single_visible_device
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--roots-dir', type=Path, help='Override validation mappings directory')
    parser.add_argument('--configs-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    single_visible_device()
    import torch
    from pixal3d import datasets
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    from pixal3d.utils.data_utils import recursive_to_device

    torch.set_num_threads(4)
    report = {}
    for stage in STAGES:
        config = json.loads((args.configs_dir / f'{stage}.json').read_text())
        dataset = getattr(datasets, config['dataset']['name'])(
            (args.roots_dir / f'validation_{stage}_roots.json' if args.roots_dir
             else Path(config['validation']['roots_file'])).read_text(), **config['dataset']['args'])
        dataset.min_condition_views = dataset.max_condition_views = 1
        batch = dataset.collate_fn([dataset[0]])
        batch = batch if isinstance(batch, dict) else batch[0]
        batch = recursive_to_device(batch, torch.device('cuda:0'))
        extractor = DinoV3ProjFeatureExtractor(**config['trainer']['args']['image_cond_model']['args']).cuda().eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            legacy = extractor(batch['cond'][:, 0], camera_angle_x=batch['camera_angle_x'][:, 0],
                distance=batch['camera_distance'][:, 0], mesh_scale=batch['mesh_scale'])
            mv = extractor(batch['cond'], camera_angle_x=batch['camera_angle_x'],
                distance=batch['camera_distance'], mesh_scale=batch['mesh_scale'],
                transform_matrix=batch['transform_matrix'])
        errors = []
        for a, b in zip(legacy, mv):
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            torch.testing.assert_close(a.float(), b.float(), rtol=1e-5, atol=1e-5)
            errors.append(float((a.float() - b.float()).abs().max()))
        report[stage] = {'K': 1, 'max_abs_errors': errors, 'passed': True}
        print(stage, json.dumps(report[stage]), flush=True)
        del extractor, batch, legacy, mv, dataset
        torch.cuda.empty_cache()
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
