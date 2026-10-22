"""Discarded native PBR global24 capacity, including rendering and next update.

Repeat the predeclared real TRAIN31900-token case on every rank. This measures
maximum-token memory, not real-data throughput or scientific model quality.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from .cascade_view_roles import ClosedInputs, file_sha256
from .evaluate_pbr_streaming import aggregate_metrics, tensor_sha
from .evaluate_r_mediated_pbr_initial import audited_sampling
from .probe_member_feature_fullbatch import _frozen_digest
from .probe_r_mediated_pbr_native import validate_config
from .r_mediated_native_runtime import state_sha


def validate_capacity(plan, cfg, original):
    validate_config(cfg, original, plan['mode'])
    if (isinstance(plan['world_size'], bool) or plan['world_size'] not in {2, 4}
            or plan['microbatch'] != 24//plan['world_size']
            or plan['global_batch'] != 24 or plan['accumulation'] != 1
            or plan['max_tokens'] != 31900 or plan['temporary_updates'] != 4
            or plan['validation']['dataset_name'] != cfg['dataset']['name']
            or plan['validation']['dataset_args'] != dict(cfg['dataset']['args'], expected_split='validation')
            or plan['validation']['quick_metrics']['panel_size'] != 4
            or plan['validation']['quick_metrics']['sampling_steps'] != 12
            or plan['validation']['quick_metrics']['render_resolution'] != 256
            or not plan['validation']['quick_metrics'].get('panel_file')
            or not plan['validation']['quick_metrics'].get('seed_map_file')):
        raise ValueError('discarded native global24/micro12 or6, accumulation1, max-token and render4 proof required')


@contextmanager
def unwrapped_validation(trainer, *, device):
    """Rank0 rendering must neither call collective DDP forward nor consume RNG."""
    previous = trainer.training_models
    modes = {name: model.training for name, model in trainer.models.items()}
    dropout, python_rng, numpy_rng = trainer.p_uncond, random.getstate(), np.random.get_state()
    try:
        trainer.training_models = trainer.models
        for model in trainer.models.values():
            model.eval()
        trainer.p_uncond = 0.
        with torch.random.fork_rng(devices=[device]), torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            yield
    finally:
        trainer.training_models = previous
        for name, model in trainer.models.items():
            model.train(modes[name])
        trainer.p_uncond = dropout
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def guard_plan(path, plan):
    check = ClosedInputs(plan['canonical_local_root'])
    check.check(path, path.parent/'file4-closure.json')
    for name, digest in {**plan['source_SHA256'], **plan['input_SHA256']}.items():
        if file_sha256(name) != digest:
            raise ValueError('capacity source/input changed: '+name)
    for name, receipt in plan['closed_inputs'].items():
        check.check(name, receipt)
    for name, status in [('micro1_gate', 'PASS_FOUR_PBR_NATIVE_MICRO1_INTEGRATION_ONLY'),
                         ('initial32_gate', 'PASS_FOUR_PBR_NATIVE_COLD_GENERATED32_IDENTITY')]:
        report = json.loads(Path(plan[name]).read_text())
        if report['status'] != status or report['scientific_training_admitted']:
            raise ValueError('actual own-schema micro1 and cold generated32 checks required')
    return check


def worker(rank, plan_path):
    plan_path = Path(plan_path); plan = json.loads(plan_path.read_text())
    from pixal3d import models
    from pixal3d.datasets.deformnext import DeformNeXtPBR1024
    from pixal3d.datasets.deformnext_feature import DeformNeXtFeaturePBR1024
    from pixal3d.trainers.flow_matching.r_mediated_pbr_source import RMediatedPBRNativeTrainer
    from pixal3d.trainers.flow_matching.sparse_flow_matching import ImageConditionedProjSparseFlowMatchingCFGTrainer
    from pixal3d.utils.dist_utils import setup_dist
    from train import setup_rng
    from . import quick_metrics as quick
    from .frozen_projection_streaming import install_frozen_projection_streaming

    gpu = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[rank]
    os.sched_setaffinity(0, {84+2*int(gpu), 85+2*int(gpu)})
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    setup_dist(rank, rank, plan['world_size'], '127.0.0.1', str(plan['master_port']))
    setup_rng(rank, 101)
    cfg = json.loads(Path(plan['config']).read_text())
    original = json.loads(Path(plan['original_config']).read_text())
    validate_capacity(plan, cfg, original)
    output = Path(plan['output'])
    install_frozen_projection_streaming(max_assets=1, audit_directory=output/'projection-streaming-audit')
    Dataset = DeformNeXtPBR1024 if plan['mode'] == 'vanilla' else DeformNeXtFeaturePBR1024
    dataset = Dataset(cfg['data_dir'], **cfg['dataset']['args'])
    ids = [row['sample_id'] for row in dataset.rows]
    if len(ids) != len(set(ids)) or len(ids) != 9139 or ids.count(plan['sample_id']) != 1:
        raise ValueError('complete common eligible TRAIN9139 and declared max-token case required')
    item = dataset[ids.index(plan['sample_id'])]
    if len(item['coords']) != 31900:
        raise ValueError('actual original TRAIN31900-token capacity case required')
    with torch.device('meta'):
        md = {name: getattr(models, spec['name'])(**spec['args']) for name, spec in cfg['models'].items()}
    kw = copy.deepcopy(cfg['trainer']['args'])
    kw.update(batch_size_per_gpu=plan['microbatch'], batch_split=1)
    Trainer = ImageConditionedProjSparseFlowMatchingCFGTrainer if plan['mode'] == 'vanilla' else RMediatedPBRNativeTrainer
    trainer = Trainer(md, dataset, **kw, output_dir=str(output/'native'), load_dir=None, step=None, wandb_run=None)
    source = {} if plan['mode'] == 'vanilla' else {
        n: p for n, p in trainer.deform_conditioner.named_parameters() if p.requires_grad}
    lora = {n: p for n, p in trainer.models['denoiser'].named_parameters() if p.requires_grad}
    expected = 0 if plan['mode'] == 'vanilla' else (10 if plan['mode'] == 'top1-only' else 19)
    if (len(lora) != 360 or len(source) != expected or trainer.optimizer.state or trainer.step != 0
            or (trainer.world_size, trainer.batch_size, trainer.batch_size_per_gpu, trainer.batch_split)
                != (plan['world_size'], 24, plan['microbatch'], 1) or trainer.p_uncond != .1):
        raise ValueError('fresh native360+source0/10/19, actual global24 and native dropout.1 required')
    before = {name: _frozen_digest(model) for name, model in trainer.models.items()}
    snapshots = {id(p): p.detach().cpu().clone() for p in [*lora.values(), *source.values()]}
    initial_lora = state_sha(lora.items())
    initial_r = state_sha((n, p) for n, p in source.items() if not n.startswith('ray_readout.')) if source else None
    updates, events, numeric, traces, inputs = [], [], [], [], {}
    torch.cuda.reset_peak_memory_stats()

    def update():
        began = time.monotonic()
        data = dataset.collate_fn([item]*plan['microbatch'], split_size=1)
        if len(data) != 1 or len(data[0]['x_0'].coords) != plan['microbatch']*31900:
            raise ValueError('one full maximum-token sparse forward, no accumulation required')
        if not updates:
            inputs.update({key: tensor_sha(data[0][key]) for key in ['cond', 'x_0', 'concat_cond']})
        handle = trainer.models['denoiser'].register_forward_pre_hook(lambda m, a: traces.append(dict(
            x_t_SHA256=tensor_sha(a[0]), t_SHA256=tensor_sha(a[1]), global_SHA256=tensor_sha(a[2]['global']))))
        try:
            result = trainer.run_step(data)
        finally:
            handle.remove()
        torch.cuda.synchronize(rank)
        trainer.step += 1
        if not math.isfinite(result['loss']['loss']):
            raise ValueError('actual native maximum-token fullbatch loss must be finite')
        gradients = {name: None if p.grad is None else float(p.grad.float().abs().sum()) for name, p in source.items()}
        if any(v is None or not math.isfinite(v) for v in gradients.values()):
            raise ValueError('all native source gradients must be present and finite')
        updates.append(dict(step=trainer.step, loss=result['loss'], seconds=time.monotonic()-began,
            source_gradient_L1=gradients, native_dropout=trainer.p_uncond))

    for _ in range(3):
        update()
    torch.distributed.barrier()
    if rank == 0:
        v = plan['validation']; validation = Dataset(v['roots_file'], **v['dataset_args'])
        indices = quick.panel_indices(validation, v['quick_metrics'])
        if len(indices) != 4:
            raise ValueError('actual fixed4 rendering validation required')
        from PIL import Image
        with unwrapped_validation(trainer, device=rank), audited_sampling(quick, trainer, events):
            for number, (index, seed) in enumerate(zip(indices, quick.panel_seeds(validation, indices, v['quick_metrics']))):
                metrics, panel = quick._pbr_metrics(trainer, validation, [index], dict(v['quick_metrics'], seed=seed))
                if len(events) != number+1 or not all(math.isfinite(x) for x in metrics.values()):
                    raise ValueError('actual finite PBR4 rendering and input/noise/prediction audits required')
                numeric.append(metrics); events[-1]['metrics'] = metrics
                Image.fromarray(panel).save(output/f'validation-case{number:03d}.png')
        (output/'validation4.json').write_text(json.dumps(dict(per_asset=events,
            metrics=aggregate_metrics(numeric), fixed_GT_shape=True, source_preservation=False), indent=2)+'\n')
    torch.distributed.barrier()
    update()
    trainer.check_ddp()
    steps = {int(state['step']) for state in trainer.optimizer.state.values()}
    changed_lora = sum(not torch.equal(p.detach().cpu(), snapshots[id(p)]) for p in lora.values())
    changed_source = sum(not torch.equal(p.detach().cpu(), snapshots[id(p)]) for p in source.values())
    if (len(trainer.optimizer.state) != 360+expected or steps != {4} or len(traces) != 4
            or changed_lora != 360 or changed_source != expected
            or any(x <= 0 for x in updates[2]['source_gradient_L1'].values())
            or before != {name: _frozen_digest(model) for name, model in trainer.models.items()}):
        raise ValueError('fullbatch optimizer4, source learnability, changed LoRA and frozen integrity required')
    report = dict(status='PASS_ACTUAL_NATIVE_PBR_GLOBAL24_MAX_TOKEN_CAPACITY',
        mode=plan['mode'], rank=rank, physical_GPU=gpu,
        world_size=plan['world_size'], global_batch=24, microbatch=plan['microbatch'], accumulation=1,
        sample_id=plan['sample_id'], actual_TRAIN_assets=len(dataset),
        actual_TRAIN_order_SHA256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
        real_TRAIN_tokens_per_asset=31900, sparse_tokens_per_rank=31900*plan['microbatch'],
        native_training_dropout=.1, initial_LoRA_SHA256=initial_lora, initial_common_R10_SHA256=initial_r,
        source_multiview_step=16850, fresh_optimizer_states=0,
        effective_native_trainer_args=kw, first_four_native_noise_timestep_global_traces=traces,
        actual_native_inputs_SHA256=inputs,
        actual_optimizer_states=len(trainer.optimizer.state), optimizer_updates=4,
        changed_LoRA_tensors=changed_lora, changed_source_tensors=changed_source,
        actual_rendering_validation4_and_next_fullbatch_update_passed=True,
        post_update_DDP_parameters_exact=True, frozen_NN_matcher_and_anchor_states_exact=True,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30, updates=updates,
        fixed_repeated_TRAIN_batch=True, real_data_throughput_measured=False,
        temporary_weights_discarded=True, scientific_training_updates=0,
        scientific_training_admitted=False, quality_gain_proven=False, full_goal_complete=False)
    (output/f'audit.rank{rank}.json').write_text(json.dumps(report, indent=2)+'\n')
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    allowed = set(os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(','))
    if (len(visible) != plan['world_size'] or len(visible) != len(set(visible))
            or not set(visible) <= {'0', '1', '2', '3'} & allowed):
        raise ValueError('exactly the declared two or four assigned physical GPUs required')
    validate_capacity(plan, json.loads(Path(plan['config']).read_text()),
        json.loads(Path(plan['original_config']).read_text()))
    check = guard_plan(args.plan, plan)
    output = Path(plan['output']); output.mkdir(parents=True, exist_ok=False)
    mp.spawn(worker, args=(str(args.plan),), nprocs=plan['world_size'], join=True)
    guard_plan(args.plan, plan)
    reports = [json.loads((output/f'audit.rank{rank}.json').read_text()) for rank in range(plan['world_size'])]
    if any(r['status'] != 'PASS_ACTUAL_NATIVE_PBR_GLOBAL24_MAX_TOKEN_CAPACITY' for r in reports):
        raise ValueError('actual PASS from every capacity rank required')
    report = dict(status='PASS_ACTUAL_NATIVE_PBR_GLOBAL24_MAX_TOKEN_CAPACITY',
        mode=plan['mode'], world_size=plan['world_size'], global_batch=24,
        microbatch=plan['microbatch'], accumulation=1, actual_rank_reports=reports,
        maximum_allocated_GiB=max(r['peak_allocated_GiB'] for r in reports),
        actual_rendering_validation4_and_next_update_passed=True, optimizer_updates=4,
        prototype_weights_discarded=True, real_data_throughput_measured=False,
        scientific_training_admitted=False, quality_gain_proven=False, full_goal_complete=False,
        verified_inputs=check.inputs)
    (output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], mode=plan['mode'], world_size=plan['world_size'],
        peak_GiB=report['maximum_allocated_GiB'])), flush=True)


if __name__ == '__main__':
    main()
