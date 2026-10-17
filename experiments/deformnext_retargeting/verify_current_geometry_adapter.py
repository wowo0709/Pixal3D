"""Two-update native SS loss proof for adapter, then adapter+predicted query.

This gradient/initial-output check is isolated from production global96 runs.
It does not measure module quality or perform a five-epoch training arm.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .audit_predicted_geometry import check_arrays
from .current_geometry_context import geometry_content_context
from .geometry_content_adapter import GeometryContentRetargetAdapter
from .reproduce_q2c import checked, read, reference, write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'checkpoint', 'manifest', 'query-preflight', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--wandb-id')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    import os
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    if (args.output / 'summary.json').exists():
        raise FileExistsError('immutable native geometry proof exists')
    from pixal3d import datasets, models, trainers
    from pixal3d.utils.data_utils import recursive_to_device
    cfg, preflight = read(args.config), read(args.query_preflight)
    if (preflight['status'] != 'PASS_PREFLIGHT_ONLY' or len(preflight['records']) != 1
            or preflight['protocol']['consumer'] != reference(args.checkpoint)):
        raise ValueError('actual query GPU preflight consumer differs')
    record = preflight['records'][0]
    with np.load(checked(record['artifact']), allow_pickle=False) as data:
        check_arrays(data)
        geometry = torch.from_numpy(data['geometry9'].copy())[None].cuda()
    ds_args = copy.deepcopy(cfg['dataset']['args'])
    ds_args.update(transport_arm='v1', expected_split='train')
    for key in ['match_root', 'projection_root', 'deformation_root', 'expected_q2c_sha256']:
        ds_args.pop(key, None)
    dataset = getattr(datasets, cfg['dataset']['name'])(str(args.manifest), **ds_args)
    index = next(i for i, row in enumerate(dataset.rows) if row['sample_id'] == record['sample_id'])
    if dataset.rows[index]['source_cluster_id'] != record['source_cluster_id']:
        raise ValueError('query and SS proof asset identity differs')
    with torch.device('meta'):
        model_dict = {name: getattr(models, spec['name'])(**spec['args']) for name, spec in cfg['models'].items()}
    kw = copy.deepcopy(cfg['trainer']['args'])
    kw.update(continuation_ckpt={'denoiser': str(args.checkpoint)}, deform_conditioner=None,
              batch_size_per_gpu=1, batch_split=1, num_workers=0, prefetch_data=False, max_steps=1)
    kw.get('model_learning_rates', {}).pop('deform_conditioner', None)
    args.output.mkdir(parents=True, exist_ok=True)
    trainer = getattr(trainers, cfg['trainer']['name'])(model_dict, dataset, **kw,
        output_dir=str(args.output / 'runtime'), load_dir=None, step=None, wandb_run=None)
    trainer.models['denoiser'].eval().requires_grad_(False)
    trainer.p_uncond = 0
    if trainer.image_cond_model is None:
        trainer._init_image_cond_model()
    # GT aligned latent enters only the ordinary flow loss here, never O/D query.
    pack = recursive_to_device(dataset.collate_fn([dataset[index]]), trainer.device)
    target = pack.pop('x_0')
    pack.pop('_dataset_name'); pack.pop('_sha256')
    cases = []
    run_log = None
    if args.wandb_id:
        import wandb
        run_log = wandb.init(project='pixal3d-deformnext-retargeting', entity='youngwoojeon-unist',
            id=args.wandb_id, resume='never', dir=str(args.output), config={
                'scope': 'one-TRAIN-asset actual SS loss/gradient proof, no quality evaluation',
                'optimizer_updates_per_case': 2, 'production_global_batch_changed': False,
                'query_preflight': reference(args.query_preflight), 'sample_id': record['sample_id']})
    for with_query in [False, True]:
        torch.manual_seed(42017)
        adapter = GeometryContentRetargetAdapter(with_query=with_query).cuda().train()
        if not with_query and (adapter.content_encoder is not None or adapter.geometry_hidden != 0):
            raise ValueError('No-query still has an encoder/extra channels')
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=.0001, weight_decay=.01)
        torch.manual_seed(42018); torch.cuda.manual_seed(42018)
        rng = (torch.get_rng_state(), torch.cuda.get_rng_state())
        with torch.no_grad(), torch.autocast('cuda', dtype=trainer.mix_precision_dtype):
            baseline, _ = trainer.training_losses(x_0=target, **pack)
        torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1])
        history, events = [], []
        with geometry_content_context(trainer, adapter, geometry if with_query else None, events):
            for update in range(2):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=trainer.mix_precision_dtype):
                    losses, _ = trainer.training_losses(x_0=target, **pack)
                if update == 0 and (not events[-1]['native_projection_exact'] or
                        not torch.equal(baseline['loss'], losses['loss'])):
                    raise ValueError('initial native SS loss/output parity failed')
                losses['loss'].backward()
                gradients = [p.grad for p in adapter.parameters() if p.grad is not None]
                if (not gradients or not all(torch.isfinite(g).all() for g in gradients)
                        or not any(bool(g.abs().sum() > 0) for g in gradients)):
                    raise ValueError('actual SS loss has no finite/nonzero adapter gradient')
                if any(p.grad is not None for p in trainer.models['denoiser'].parameters()):
                    raise ValueError('frozen proof acquired SS parameter gradients')
                query_gradient = (sum(float(p.grad.abs().sum()) for p in adapter.content_encoder.parameters()
                                  if p.grad is not None) if with_query else None)
                if with_query and update == 1 and not query_gradient > 0:
                    raise ValueError('predicted nine-channel encoder is disconnected after unblocking the zero output layer')
                row = {'update': update+1, 'loss': float(losses['loss'].detach()),
                    'query_encoder_gradient_L1': query_gradient, **events[-1]}
                history.append(row)
                optimizer.step()
                print(json.dumps({'with_query': with_query, **row}), flush=True)
                if run_log:
                    run_log.log({('adapter_plus_query' if with_query else 'adapter_only') + '/' + k: v
                                 for k, v in row.items() if v is not None})
        cases.append({'with_query': with_query, 'raw_geometry_input_channels': 9 if with_query else 0,
            'content_encoder_present': adapter.content_encoder is not None, 'history': history,
            'baseline_loss': float(baseline['loss']), 'initial_SS_loss_exact': True,
            'frozen_SS_gradients_absent': True})
    write_new(args.output / 'summary.json', {'status': 'PASS', 'scope': 'actual current-native SS gradient proof only',
        'sample_id': record['sample_id'], 'source_cluster_id': record['source_cluster_id'], 'cases': cases,
        'checkpoint': reference(args.checkpoint), 'query_preflight': reference(args.query_preflight),
        'optimizer_updates_per_case': 2, 'quality_evaluation_performed': False,
        'production_global96_recipe_changed': False, 'new_five_epoch_training_started': False,
        'production_promotion': False, 'current_SS_gain_claim': False})
    if run_log:
        run_log.summary.update({'status': 'PASS', 'SS_gain_claim': False})
        run_log.finish()


if __name__ == '__main__':
    main()
