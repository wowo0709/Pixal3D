"""Fixed4 native PBR flow and anchor renders for both V1 and feature datasets."""
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .evaluate_pbr_streaming import aggregate_metrics, tensor_sha
from .evaluate_r_mediated_pbr_initial import audited_sampling
from .probe_r_mediated_pbr_capacity import unwrapped_validation
from .r_mediated_native_runtime import state_sha
from .r_mediated_pbr_science_recipe import validate_initial4


def optimizer_token(trainer):
    return dict(step=trainer.step, states={id(p): int(s['step']) for p, s in trainer.optimizer.state.items()},
        sampler=trainer.data_sampler.state_dict(),
        scheduler=repr(trainer.lr_scheduler.state_dict()) if trainer.lr_scheduler is not None else None)


def evaluate_science_pbr(trainer, cfg, plan):
    from PIL import Image
    from pixal3d.datasets.deformnext import DeformNeXtPBR1024
    from pixal3d.datasets.deformnext_feature import DeformNeXtFeaturePBR1024
    from pixal3d.utils.data_utils import recursive_to_device
    from . import quick_metrics as quick

    if not trainer.is_master:
        raise ValueError('PBR fixed-panel validation runs on rank0 only')
    started = time.monotonic()
    validation = cfg['validation']; options = validation['quick_metrics']
    Dataset = DeformNeXtPBR1024 if plan['mode'] == 'vanilla' else DeformNeXtFeaturePBR1024
    dataset = Dataset(validation['roots_file'], **validation['dataset_args'])
    indices = quick.panel_indices(dataset, options)
    if len(indices) != 4 or not all(dataset.rows[i]['split'] == 'validation' for i in indices):
        raise ValueError('fixed4 actual validation assets required')
    state = optimizer_token(trainer)
    weights = {n: state_sha(m.state_dict().items()) for n, m in trainer.models.items()}
    events, numeric, panels, sources = [], [], [], []
    folder = Path(trainer.output_dir)/'validation-panels'/f'step{trainer.step:07d}'
    folder.mkdir(parents=True, exist_ok=False)
    with unwrapped_validation(trainer, device=trainer.local_rank), audited_sampling(quick, trainer, events):
        for number, (index, seed) in enumerate(zip(indices, quick.panel_seeds(dataset, indices, options))):
            pack = dataset.collate_fn([dataset[index]])
            pack = recursive_to_device(pack, trainer.device)
            pack.pop('_dataset_name', None); pack.pop('_sha256', None)
            torch.manual_seed(seed); torch.cuda.manual_seed(seed)
            result, _ = trainer.training_losses(**pack)
            flow = float(result['loss'])
            del pack, result
            metrics, panel = quick._pbr_metrics(trainer, dataset, [index], dict(options, seed=seed))
            if len(events) != number+1 or not all(math.isfinite(v) for v in [flow, *metrics.values()]):
                raise ValueError('finite native loss and actual PBR4 generation/rendering audits required')
            events[-1].update(metrics=metrics, flow_loss=flow)
            numeric.append(metrics); panels.append(panel)
            sources.append(dict(source=dataset.rows[index].get('category', 'DeformNeXt'),
                assets=1, K1_loss=None, K4_loss=flow))
            Image.fromarray(panel).save(folder/f'case{number:03d}.png')
    if state != optimizer_token(trainer) or weights != {
            n: state_sha(m.state_dict().items()) for n, m in trainer.models.items()}:
        raise ValueError('PBR validation modified models, sampler, optimizer, scheduler or training step')
    row = dict(step=trainer.step, weights='raw', scope='fixed4 anchor-stage rendering;GT target Shape',
        sources=sources, quick_metrics=aggregate_metrics(numeric),
        quick_panel_ids=[r['sample_id'] for r in events], quick_per_asset=events,
        actual_validation_dataset=validation['dataset_name'], fixed_GT_target_shape=True,
        full_cascade=False, source_preservation_metrics=False, optimizer_updates_during_evaluation=0,
        seconds=time.monotonic()-started)
    if trainer.step == 0:
        if trainer.optimizer.state:
            raise ValueError('initial PBR validation requires fresh optimizer0')
        reference = json.loads(Path(plan['initial_reference']).read_text())
        validate_initial4(row, reference)
        trainer._PBR_initial_guard_passed = True
        (folder/'initial-identity.json').write_text(json.dumps(dict(
            status='PASS_NATIVE_PBR_SCIENCE_INITIAL4_BEFORE_FIRST_UPDATE',
            actual_inputs_noise_prediction_and_rendering_equal_original32=True,
            optimizer_updates=0, model_adopted=False), indent=2)+'\n')
    trainer._quick_per_asset = events
    with (Path(trainer.output_dir)/'validation.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')
    panel = np.concatenate(panels, axis=0)
    Image.fromarray(panel).save(folder/'panel.png')
    log_validation(trainer, row, panel)
    print('PBR_NATIVE_VALIDATION', json.dumps(dict(step=trainer.step,
        metrics=row['quick_metrics'], seconds=row['seconds'])), flush=True)
    return row


def log_validation(trainer, row, panel):
    if trainer.wandb_run is None:
        raise ValueError('scientific PBR must log metrics and images to its W&B run')
    import wandb
    payload = {f'validation/{key}': value for key, value in row['quick_metrics'].items()}
    payload.update({'validation/K4_loss': float(np.mean([r['flow_loss'] for r in row['quick_per_asset']])),
        'validation/seconds': row['seconds'],
        'validation/quick_panel': wandb.Image(panel,
            caption='Fixed GT Shape; target RGB anchor. GT | prediction | absolute difference\n'+' | '.join(row['quick_panel_ids']))})
    trainer.wandb_run.log(payload, step=row['step'])
