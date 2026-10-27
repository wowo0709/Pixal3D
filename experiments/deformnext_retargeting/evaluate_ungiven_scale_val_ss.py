"""Paired VAL SS generation with frozen weights and instance scale unprovided.

The legacy target latent remains a scoring label. It is removed from the
inference-condition pack by quick_metrics before the denoiser is called.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('ATTN_BACKEND', 'sdpa')

import numpy as np
import torch


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--arm', choices=('V1', 'RETARGET', 'SELECTED'), required=True)
    parser.add_argument('--mode', choices=('legacy-target', 'scale-ungiven', 'image-estimated'), required=True)
    parser.add_argument('--scale-model', type=Path)
    parser.add_argument('--panel-file', type=Path, required=True)
    parser.add_argument('--seed-map-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generation-batch', type=int, default=8)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.generation_batch <= 8:
        raise ValueError('fresh output and bounded native generation batch required')
    freeze = json.loads(args.freeze.read_text())
    if freeze.get('status') != 'FROZEN_FROM_DEVELOPMENT_ONLY':
        raise ValueError('frozen model set required')
    model = freeze['models'][args.arm]['ss64']
    config_path = Path(model['native_config']['path'])
    checkpoint = Path(model['checkpoint']['path'])
    conditioner = model.get('conditioner_checkpoint')
    for spec in [model['native_config'], model['checkpoint'], *([conditioner] if conditioner else [])]:
        path = Path(spec['path'])
        if not path.is_file() or sha(path) != spec['SHA256']:
            raise ValueError('frozen model input differs: ' + str(path))
    config = json.loads(config_path.read_text())
    if args.split == 'test':
        frozen = freeze['population']
        validation = Path(frozen['manifest']['path'])
        if (len(frozen['sample_ids']) != 508 or sha(validation) != frozen['manifest']['SHA256']
                or json.loads(args.panel_file.read_text())['sample_ids'] != frozen['sample_ids']
                or json.loads(args.seed_map_file.read_text()) != frozen['generation_seeds']):
            raise ValueError('TEST508 population, manifest, and seeds must match development freeze')
    else:
        validation = Path(config['validation']['roots_file'])
    rows = {row['sample_id']: row for row in map(json.loads, validation.read_text().splitlines())}
    panel = json.loads(args.panel_file.read_text())['sample_ids']
    seeds = json.loads(args.seed_map_file.read_text())
    if (len(panel) != len(set(panel)) or not set(panel) <= set(rows)
            or not set(panel) <= set(seeds)):
        raise ValueError('paired VAL panel or seed map incomplete')
    image_scales = {}
    if args.mode == 'image-estimated':
        if args.scale_model is None or not args.scale_model.is_file():
            raise ValueError('frozen image-only scale estimator required')
        import joblib
        from .probe_image_estimated_scale import features
        scale_estimator = joblib.load(args.scale_model)
        if (scale_estimator.get('feature_schema') != 'RGBA128-mask-bbox-area-camera-v1'
                or set(panel) & set(scale_estimator['training_sample_ids'])):
            raise ValueError('image estimator schema or TRAIN/VAL separation differs')
        path_prefix = Path(config['validation']['dataset_args']['path_map']['/file4'])
        vectors = []
        for sid in panel:
            row = rows[sid]
            permitted = {key: row[key] for key in ('target_anchor', 'source_images',
                                                   'transform_matrix', 'camera_angle_x')}
            vectors.append(features(permitted, path_prefix))
        predictions = np.exp(scale_estimator['estimator'].predict(
            np.asarray(vectors, dtype=np.float32)))
        for sid, value in zip(panel, predictions):
            predicted = float(value)
            if not .4 < predicted < 1.6:
                raise ValueError('image-only scale prediction outside contract')
            image_scales[sid] = predicted
    elif args.scale_model is not None:
        raise ValueError('scale model is only allowed for image-estimated mode')
    from pixal3d import datasets, models
    from PIL import Image
    from train import setup_rng
    from . import quick_metrics as quick
    from .evaluate_ss_artifact_subset import capture_ss_artifacts
    from .projection_maps import anchor_patch_indices
    if args.arm == 'V1':
        from .r_mediated_native_recipe import validate_config
        from .r_mediated_native_runtime import selected_trainer, parameter_groups, state_sha
    elif args.arm == 'RETARGET':
        from .r_top1_native_recipe import validate_config
        from .r_top1_native_runtime import selected_trainer, parameter_groups, state_sha
    else:
        from .consistent_evidence_native_recipe import validate_config
        from .consistent_evidence_native_runtime import selected_trainer, parameter_groups, state_sha
    variant = validate_config(config, 'science')
    if variant != model['variant']:
        raise ValueError('frozen arm and validated native variant differ')

    original = datasets.DeformNeXtSS64
    grid = np.indices((16, 16, 16)).reshape(3, -1).T
    events = []

    class UngivenScaleSS(original):
        def __getitem__(self, index):
            item = super().__getitem__(index)
            if args.mode != 'legacy-target':
                row = self.rows[index]
                sid = row['sample_id']
                scale = 1.0 if args.mode == 'scale-ungiven' else image_scales[sid]
                item['mesh_scale'] = torch.tensor(scale, dtype=torch.float32)
                if self.transport_arm != 'v1':
                    canonical_row = dict(row, mesh_scale=scale)
                    patch = anchor_patch_indices(canonical_row, grid, 16).reshape(16, 16, 16)
                    item['transport_patch_grid'] = torch.from_numpy(patch).long()
                events.append(sid)
            return item

    setup_rng(0, config['training_seed'])
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    dataset_args = copy.deepcopy(config['validation']['dataset_args'])
    dataset_args['expected_split'] = args.split
    if args.split == 'test' and dataset_args.get('projection_root') is not None:
        # The frozen training config points to train/validation projections.
        # TEST508 has its own previously archived projection cache alongside
        # the frozen test manifest; keep the same patch-map algorithm/bytes.
        test_projection = validation.parents[2] / 'projection_maps_world_v4'
        if any(not (test_projection / f'{sid}.npz').is_file() for sid in panel):
            raise FileNotFoundError('frozen TEST508 projection cache is incomplete')
        dataset_args['projection_root'] = str(test_projection)
    ds = UngivenScaleSS(validation, **dataset_args)
    validation_cfg = copy.deepcopy(config['validation'])
    validation_cfg['roots_file'] = str(validation)
    validation_cfg['dataset_args'] = dataset_args
    validation_cfg['quick_metrics'].update(panel_size=len(panel), panel_file=str(args.panel_file),
        seed_map_file=str(args.seed_map_file), generation_batch_size=args.generation_batch,
        log_progress=True)
    with torch.device('meta'):
        model_defs = {name: getattr(models, spec['name'])(**spec['args'])
                      for name, spec in config['models'].items()}
    kwargs = copy.deepcopy(config['trainer']['args'])
    kwargs.update(continuation_ckpt={'denoiser': str(checkpoint)},
                  batch_size_per_gpu=1, batch_split=1, num_workers=0,
                  prefetch_data=False, max_steps=1)
    args.output.mkdir(parents=True, exist_ok=False)
    trainer = selected_trainer(variant)(model_defs, ds, **kwargs,
        output_dir=str(args.output / 'loader'), load_dir=None, step=None, wandb_run=None)
    if conditioner:
        trainer.deform_conditioner.load_state_dict(
            torch.load(conditioner['path'], map_location='cpu', weights_only=True), strict=True)
    lora, source = parameter_groups(trainer, variant)
    before = state_sha(list(lora.items()) + list(source.items()))
    if trainer.optimizer.state or trainer.step != 0:
        raise ValueError('validation must not resume optimizer or update weights')
    for loaded in trainer.models.values():
        loaded.eval()
    if trainer.deform_conditioner is not None:
        trainer.deform_conditioner.eval()
    trainer.p_uncond = 0.
    captured = []
    with capture_ss_artifacts(quick, args.output, captured), torch.no_grad(), \
            torch.autocast('cuda', dtype=trainer.mix_precision_dtype):
        metrics, panel_image, actual_ids = quick.evaluate_quick(trainer, ds, validation_cfg)
    if (trainer.optimizer.state or before != state_sha(list(lora.items()) + list(source.items()))
            or actual_ids != panel or len(captured) != len(panel)):
        raise ValueError('evaluation changed weights, population or SS artifact count')
    result = {'status': 'PASS', 'split': args.split, 'arm': args.arm, 'variant': variant,
              'quick_metrics': metrics, 'quick_panel_ids': actual_ids,
              'per_asset': trainer._quick_per_asset,
              'optimizer_updates_during_evaluation': 0,
              'checkpoint': str(checkpoint), 'conditioner_checkpoint':
                  conditioner['path'] if conditioner else None,
              'arm_config': str(config_path), 'native_SS_artifacts':
                  {'artifacts': captured, 'archived_asset_count': len(captured),
                   'GT_SS_is_for_scoring_only': True}}
    (args.output / 'evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
    Image.fromarray(panel_image).save(args.output / 'panel.png')
    if (result['status'] != 'PASS' or result['quick_panel_ids'] != panel
            or len(result['per_asset']) != len(panel)
            or (args.mode != 'legacy-target' and (not set(panel) <= set(events)
                                              or not set(events) <= set(rows)))):
        raise ValueError('paired VAL SS generation/scale override did not cover exactly the panel')
    (args.output / 'scale-contract.json').write_text(json.dumps({
        'status': ('PASS_TEST_SS_SCALE_UNPROVIDED_FROZEN' if args.split == 'test'
                   else 'PASS_VAL_SS_SCALE_SENSITIVITY'),
        'split': args.split, 'arm': args.arm, 'mode': args.mode,
        'samples': len(panel),
        'frozen_checkpoint_sha256': model['checkpoint']['SHA256'],
        'instance_scale_supplied_to_generation': args.mode == 'legacy-target',
        'effective_native_default_scale': 1.0 if args.mode == 'scale-ungiven' else None,
        'image_estimator_checkpoint': str(args.scale_model) if args.scale_model else None,
        'image_estimator_sha256': sha(args.scale_model) if args.scale_model else None,
        'image_estimated_scales': image_scales,
        'target_M1_scale_not_used_for_inference': args.mode != 'legacy-target',
        'source_M0_mesh_used_in_generation': False,
        'legacy_target_latent_used_as_score_label_only': True,
        'canonical_projection_recomputed': args.mode != 'legacy-target' and args.arm != 'V1',
        'matching_cache_unchanged': True,
        'projection_root': dataset_args.get('projection_root'),
        'voxel_IoU_only_not_world_surface_quality': True,
        'panel': panel, 'seeds': {sid: seeds[sid] for sid in panel},
    }, indent=2) + '\n')
    print(json.dumps({'status': ('PASS_TEST_SS_SCALE_UNPROVIDED_FROZEN' if args.split == 'test'
                                  else 'PASS_VAL_SS_SCALE_SENSITIVITY'), 'arm': args.arm,
                      'mode': args.mode, 'samples': len(panel),
                      'voxel_iou': result['quick_metrics']['voxel_iou']}), flush=True)


if __name__ == '__main__':
    main()
